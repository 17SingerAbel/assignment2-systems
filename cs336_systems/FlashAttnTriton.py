import torch
import math
from einops import einsum, rearrange
import triton
from cs336_systems.FlashAttnAutoGrad import flash_backward

import triton.language as tl

def next_power_of_2(n):
    n = math.ceil(n)
    return 1 << (n - 1).bit_length()

class FlashAttnTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d_head = Q.shape
        n_keys = K.shape[-2]
        q_tile_size = k_tile_size = 16

        O = torch.empty_like(Q)  # [batch_size, n_queries, d_head]
        L = torch.empty(
            (batch_size, n_queries),
            device=Q.device,
            dtype=torch.float32,
        )

        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

        FlashAttnTriton.flash_fwd_kernel[grid](
            Q, K, V,
            O, L,                         # 对应 O_ptr, L_ptr
            *Q.stride(), *K.stride(), *V.stride(),
            *O.stride(),                  # stride_ob, stride_oq, stride_od
            *L.stride(),                  # stride_lb, stride_lq
            n_queries, n_keys,
            1.0 / math.sqrt(d_head),
            D=d_head,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
        
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr
        ):

        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,  
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        
        K_block_ptr = tl.make_block_ptr(
                K_ptr + batch_index * stride_kb,  
                shape=(N_KEYS, D),
                strides=(stride_kk, stride_kd),
                offsets=(0, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )

        V_block_ptr = tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,  
                shape=(N_KEYS, D),
                strides=(stride_vk, stride_vd),
                offsets=(0, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,  
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES, ),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,)
        )


        q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        prev_m = tl.full((Q_TILE_SIZE, ), float("-inf"), tl.float32)
        l_tile = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
        o_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        
        key_cols = tl.arange(0, K_TILE_SIZE)
        for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            key_indices = i * K_TILE_SIZE + key_cols

            # input_precision="ieee": real fp32 matmul for fp32 inputs. The default
            # ("tf32") rounds inputs to a 10-bit mantissa on tensor cores, which
            # made "fp32" Triton ~1e-3 off and unfairly faster than fp32 PyTorch.
            scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale
            valid = key_indices[None, :] < N_KEYS
            if IS_CAUSAL:
                valid = valid & (key_indices[None, :] <= query_indices[:, None])
            scores = tl.where(valid, scores, float("-inf"))

            tile_max = tl.max(scores, axis=1)
            curr_m = tl.maximum(prev_m, tile_max)

            p = tl.exp(scores - curr_m[:, None])

            alpha = tl.exp(prev_m - curr_m)
            l_tile = alpha * l_tile + tl.sum(p, axis=1)
            o_tile = tl.dot(p.to(v_tile.dtype), v_tile,  alpha[:, None] * o_tile, input_precision="ieee")


            prev_m = curr_m

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # o_tile accumulates in fp32; O may be bf16 -> cast on store
        tl.store(O_block_ptr, (o_tile / l_tile[:, None]).to(O_block_ptr.type.element_ty), boundary_check=(0,))
        tl.store(L_block_ptr, prev_m + tl.log(l_tile), boundary_check=(0,))



    @staticmethod
    def backward(ctx, grad_output):
        # "Partially Triton": forward is the Triton kernel above, backward reuses
        # the eager PyTorch tiled backward. It only needs the saved Q, K, V, O, L,
        # so it doesn't care which forward produced them. flash_backward follows
        # the shared precision policy (fp32 stats/accumulators, input-dtype matmul
        # inputs), so no blanket fp32 upcast is needed here.
        Q, K, V, O, L = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])
        dQ, dK, dV = flash_backward(Q, K, V, O, grad_output, L, scale, ctx.is_causal)
        return dQ, dK, dV, None