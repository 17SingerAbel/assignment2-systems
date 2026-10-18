import torch
import math


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
        q_start = 0
        k_start = 0
        d_head = Q.shape[-1] 

        final_O = torch.zeros(Q.shape)
        final_L = torch.zeros((Q.shape[-2], ))

        for i in range(num_q_tiles):
            q_start = i * B_q
            q_tile = Q[q_start:q_start+B_q, :]
            # initialize O_i(0)
            prev_l = torch.zeros((B_q, ))
            prev_max = - math.inf * torch.ones((B_q,))

            prev_O =  torch.zeros((B_q, d_head))
            for j in range(num_k_tiles):
                k_start = j * B_k
                k_tile = K[k_start: k_start+B_k, :]
                v_tile = V[k_start: k_start+B_k, :]

                scores = q_tile @ k_tile.T / math.sqrt(d_head)
                curr_max = torch.maximum(prev_max, torch.amax(scores, dim=-1)) 
                P = torch.exp(scores - curr_max[:, None])
                l = torch.exp(prev_max - curr_max) * prev_l + torch.sum(P, dim=-1)
                O = torch.diag(torch.exp(prev_max - curr_max)) @ prev_O + P @ v_tile

                prev_max = curr_max
                prev_l = l
                prev_O = O

            final_O[q_start:q_start+B_q, :] = prev_O / l[:, None]
            final_L[q_start:q_start+B_q] = prev_max + torch.log(prev_l)

        return final_O, final_L

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError