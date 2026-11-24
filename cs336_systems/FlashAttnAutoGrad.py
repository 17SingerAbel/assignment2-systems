import torch
import math
from einops import einsum, rearrange

def next_power_of_2(n):
    n = math.ceil(n)
    return 1 << (n - 1).bit_length()

class FlashAttnAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 16
        B_k = 16
        # acutally have to reshape Q from batch, num_head, seq, d_head to N_q, d_head
        num_q_tiles = math.ceil(Q.shape[-2] / B_q)
        num_k_tiles = math.ceil(K.shape[-2] / B_k)
        d_head = Q.shape[-1] 

        final_O = torch.zeros_like(Q)
        final_L = Q.new_zeros(Q.shape[:-1])
        print('num_q_tiles, ', num_q_tiles)
        for i in range(num_q_tiles):
            q_start = i * B_q
            q_tile = Q[..., q_start:q_start+B_q, :]
            # initialize O_i(0)
            # padding 
            print("q_tile shape,", q_tile.shape)
            prev_l = Q.new_zeros((*q_tile.shape[:-1], 1))
            prev_max = Q.new_full((*q_tile.shape[:-1], 1), -math.inf)
            prev_O = Q.new_zeros((*q_tile.shape[:-1], d_head))

            q_pos = q_start + torch.arange(q_tile.shape[-2], device=Q.device)

            for j in range(num_k_tiles):
                k_start = j * B_k
                k_tile = K[..., k_start: k_start+B_k, :]
                v_tile = V[..., k_start: k_start+B_k, :]
                # print(q_tile.shape, k_tile.shape)

                scores = einsum(q_tile, k_tile, "... query d_head, ... key d_head -> ... query key") / math.sqrt(d_head)

                if is_causal:
                    k_pos = k_start + torch.arange(k_tile.shape[-2], device=K.device)
                    future = k_pos[None, :] > q_pos[:, None]  # [当前 query 数, 当前 key 数]
                    scores = scores.masked_fill(future, -math.inf)



                curr_max = torch.maximum(prev_max, torch.amax(scores, dim=-1, keepdim=True)) 
                P = torch.exp(scores - curr_max)

                l = torch.exp(prev_max - curr_max) * prev_l + torch.sum(P, dim=-1, keepdim=True)
                O = torch.exp(prev_max - curr_max) * prev_O + P @ v_tile

                prev_max = curr_max
                prev_l = l
                prev_O = O
                # print('max diff shape, ', torch.exp(prev_max - curr_max).shape)
                # print("l shape, ", prev_l.shape)
                # print("o shape, ", prev_O.shape)

            final_O[..., q_start:q_start+B_q, :] = prev_O / l
            final_L[..., q_start:q_start+B_q] = (prev_max + torch.log(prev_l)).squeeze(-1)

        ctx.save_for_backward(Q, K, V, final_O, final_L)
        ctx.is_causal = is_causal

        return final_O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])
        # scores = einsum(Q, K, '... n_query d_model, ... n_key d_model -> ... n_query n_key') * scale
        # P = torch.exp(scores - L.unsqueeze(-1))

        # dV = einsum(P, grad_output, '... n_query n_key, ... n_query d_model -> ... n_key d_model')
        # dP = einsum(grad_output, V, '... n_query d_model, ... n_key d_model -> ... n_query n_key')

        # D = torch.sum(O * grad_output, dim=-1, keepdim=True)
        # dS = P * (dP - D)
        # dQ = dS @ K * scale
        # dK = dS.transpose(-1, -2) @ Q * scale
        # return (dQ, dK, dV, None)[:len(ctx.needs_input_grad)]

        dQ, dK, dV = flash_backward(Q, K, V, O, grad_output, L, scale, ctx.is_causal)
        return (dQ, dK, dV, None)[:len(ctx.needs_input_grad)]

def causal_mask_tile(s_tile, q_start, k_start):
    """Backward recomputes P from Q, K, L, so the forward's mask is gone.
    Re-apply it: future keys (k_pos > q_pos) get -inf, so exp() makes P exactly 0."""
    q_pos = q_start + torch.arange(s_tile.shape[-2], device=s_tile.device)  # global query index per row
    k_pos = k_start + torch.arange(s_tile.shape[-1], device=s_tile.device)  # global key index per column
    future = k_pos[None, :] > q_pos[:, None]  # [n_q, n_k], broadcasts over batch dims
    return s_tile.masked_fill(future, -math.inf)

# No @torch.compile: dynamo would unroll this Python double loop into one graph
# with ~(N/16)^2 iterations, so compile time explodes with N (measured
# 24s/102s/446s at N=128/256/512). Eager keeps tile=16 comparable with Triton.
# TODO(after A2): compile only the per-tile body + larger tiles; Triton backward.
def flash_backward(Q, K, V, O, dO, L, scale, is_causal=False):
    # 在这里写 PyTorch 分块循环
    # 计算 D、重算每块 P、累计 dQ/dK/dV
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    B_q = 16
    B_k = 16

    num_q_tiles = math.ceil(Q.shape[-2] / B_q)
    num_k_tiles = math.ceil(K.shape[-2] / B_k)

    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(Q)

    for j in range(num_k_tiles):
        k_start = j * B_k
        k_tile = K[..., k_start:k_start+B_k, :]
        v_tile = V[..., k_start:k_start+B_k, :]

        dk_tile = torch.zeros_like(k_tile)
        dv_tile = torch.zeros_like(k_tile)

        for i in range(num_q_tiles):
            q_start = i * B_q
            if is_causal and k_start > q_start + B_q -1:
                continue
            q_tile = Q[..., q_start:q_start+B_q, :]
            l_tile = L[..., q_start:q_start+B_q]
            d_tile = D[..., q_start:q_start+B_q, :]
            # o_tile = O[..., q_start:q_start+B_q, :]
            do = dO[..., q_start:q_start+B_q, :]

            s_tile = q_tile @ k_tile.transpose(-1, -2) * scale
            if is_causal:
                s_tile = causal_mask_tile(s_tile, q_start, k_start)
            p_tile = torch.exp(s_tile - l_tile.unsqueeze(-1))

            dv_tile = dv_tile + p_tile.transpose(-1, -2) @ do
            dp = do @ v_tile.transpose(-1, -2)
            ds = p_tile * (dp - d_tile)
            dk_tile = dk_tile + ds.transpose(-1, -2) @ q_tile * scale

            dK[..., k_start:k_start+B_k, :] = dk_tile
            dV[..., k_start:k_start+B_k, :] = dv_tile

    for i in range(num_q_tiles):
        q_start = i * B_q
        q_tile = Q[..., q_start:q_start+B_q, :]
        do = dO[..., q_start:q_start+B_q, :]
        l_tile = L[..., q_start:q_start+B_q]
        d_tile = D[..., q_start:q_start+B_q, :]
        dq_tile = torch.zeros_like(q_tile)

        for j in range(num_k_tiles):
            k_start = j * B_k
            if is_causal and k_start > q_start + B_q - 1:
                break
            k_tile = K[..., k_start:k_start+B_k, :]
            v_tile = V[..., k_start:k_start+B_k, :]
            s_tile = q_tile @ k_tile.transpose(-1, -2) * scale
            if is_causal:
                s_tile = causal_mask_tile(s_tile, q_start, k_start)
            p_tile = torch.exp(s_tile - l_tile.unsqueeze(-1))
            dp = do @ v_tile.transpose(-1, -2)
            ds = p_tile * (dp - d_tile)
            dq_tile = dq_tile + ds @ k_tile * scale

            dQ[..., q_start:q_start+B_q, :] = dq_tile
            
    return dQ, dK, dV   
