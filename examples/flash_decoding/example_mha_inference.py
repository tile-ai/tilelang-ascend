import tilelang
import tilelang.language as T
import torch
import argparse
import math
import torch.nn.functional as F

torch.set_default_device('npu')
torch.manual_seed(42)

tilelang.disable_cache()

num_split = 4

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flashattn(batch, heads, seqlen_q, seqlen_kv, dim, block_M=128, block_N=128):
    dtype = "float16"
    accum_dtype = "float"
    scale = (1.0 / dim) ** 0.5
    
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
    shape_o = [batch, seqlen_q, heads, dim]
    
    num_q_blocks = math.ceil(seqlen_q / block_M)
    num_kv_blocks = math.ceil(seqlen_kv / block_N)
    block_num = num_q_blocks * batch * heads
    
    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_o, dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid // (batch * heads)
            by = cid % (batch * heads)
            
            hid = by % heads
            bid = by // heads
            
            q_l1 = T.alloc_shared([block_M, dim], dtype)
            k_l1 = T.alloc_shared([block_N, dim], dtype)
            v_l1 = T.alloc_shared([block_N, dim], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)
            
            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, dim], accum_dtype)
            
            acc_o = T.alloc_shared([block_M // 2, dim], accum_dtype)
            m_i = T.alloc_shared([block_M // 2, 1], accum_dtype)
            sumexp = T.alloc_shared([block_M // 2, 1], accum_dtype)
            
            acc_s_ub = T.alloc_shared([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([block_M // 2, 1], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M // 2, 1], accum_dtype)
            acc_s_half = T.alloc_shared([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_shared([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_shared([block_M // 2, dim], dtype)
            
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -1e9)
            
            T.copy(Q[bid, bx * block_M : (bx + 1) * block_M, hid, :], q_l1)
            
            for k_iter in T.serial(num_kv_blocks):
                kv_block_start = k_iter * block_N
                
                T.copy(K[bid, kv_block_start : kv_block_start + block_N, hid, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])
                
                T.copy(m_i, m_i_prev)
                T.copy(workspace_1[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_s_ub)
                
                T.tile.mul(acc_s_ub, acc_s_ub, scale)
                
                T.reduce_max(acc_s_ub, m_i, dim=-1, clear=False)
                T.tile.max(m_i, m_i, m_i_prev)
                
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)
                
                for i in range(block_M // 2):
                    T.tile.sub(acc_s_ub[i, :], acc_s_ub[i, :], m_i[i, 0])
                T.tile.exp(acc_s_ub, acc_s_ub)
                
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)
                
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_2[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :])
                
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bid, kv_block_start : kv_block_start + block_N, hid, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])
                
                for i in range(block_M // 2):
                    T.tile.mul(acc_o[i, :], acc_o[i, :], m_i_prev[i, 0])
                T.copy(workspace_3[cid, vid * block_M // 2 : vid * block_M // 2 + block_M // 2, :], acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)
            
            for i in range(block_M // 2):
                T.tile.div(acc_o[i, :], acc_o[i, :], sumexp[i, 0])
            
            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[bid, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, hid, :])
    
    return main


def flashattn_mha_inference_split_kv(batch, heads, seqlen_q, seqlen_kv, dim, num_split=4, block_M=128, block_N=128):
    kv_per_split = seqlen_kv // num_split
    
    kernels = []
    workspaces = []
    
    for sid in range(num_split):
        kernel = flashattn(batch, heads, seqlen_q, kv_per_split, dim, block_M, block_N)
        
        num_q_blocks = math.ceil(seqlen_q / block_M)
        block_num = num_q_blocks * batch * heads
        
        workspace_1 = torch.zeros((block_num, block_M, block_N), dtype=torch.float)
        workspace_2 = torch.zeros((block_num, block_M, block_N), dtype=torch.float16)
        workspace_3 = torch.zeros((block_num, block_M, dim), dtype=torch.float)
        
        kernels.append(kernel)
        workspaces.append((workspace_1, workspace_2, workspace_3))
    
    def wrapper(Q, K, V):
        scale = 1.0 / math.sqrt(dim)
        
        outputs = []
        
        for sid in range(num_split):
            kv_start = sid * kv_per_split
            kv_end = kv_start + kv_per_split
            
            K_split = K[:, kv_start:kv_end, :, :]
            V_split = V[:, kv_start:kv_end, :, :]
            
            Output_split = kernels[sid](Q, K_split, V_split, *workspaces[sid])
            
            outputs.append(Output_split)
        
        scores_all = torch.einsum('bqhd,bkhd->bhqk', Q, K) * scale
        
        weights = []
        for sid in range(num_split):
            kv_start = sid * kv_per_split
            kv_end = kv_start + kv_per_split
            
            scores_split = scores_all[:, :, :, kv_start:kv_end]
            
            max_full = scores_all.max(dim=-1).values
            sum_full = torch.exp(scores_all - max_full.unsqueeze(-1)).sum(dim=-1)
            
            max_s = scores_split.max(dim=-1).values
            sum_s = torch.exp(scores_split - max_s.unsqueeze(-1)).sum(dim=-1)
            
            weight = (torch.exp(max_s - max_full) * sum_s / sum_full)
            weights.append(weight)
        
        final_output = torch.zeros(batch, seqlen_q, heads, dim, dtype=torch.float16)
        for sid in range(num_split):
            scale_s = weights[sid].transpose(1, 2).unsqueeze(-1)
            final_output += outputs[sid] * scale_s
        
        return final_output
    
    return wrapper


def ref_program(Q, K, V):
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / math.sqrt(dim)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seqlen_q", type=int, default=128)
    parser.add_argument("--seqlen_kv", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_split", type=int, default=4)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    args = parser.parse_args()
    
    batch = args.batch
    heads = args.heads
    seqlen_q = args.seqlen_q
    seqlen_kv = args.seqlen_kv
    dim = args.dim
    num_split = args.num_split
    block_M = args.block_M
    block_N = args.block_N
    
    assert seqlen_kv % num_split == 0
    assert seqlen_q % block_M == 0
    
    kernel = flashattn_mha_inference_split_kv(batch, heads, seqlen_q, seqlen_kv, dim, num_split, block_M, block_N)
    
    Q = torch.randn(batch, seqlen_q, heads, dim, dtype=torch.float16)
    K = torch.randn(batch, seqlen_kv, heads, dim, dtype=torch.float16)
    V = torch.randn(batch, seqlen_kv, heads, dim, dtype=torch.float16)
    
    torch.npu.synchronize()
    print("init successful!")
    
    warmup_output = kernel(Q, K, V)
    torch.npu.synchronize()
    print("warmup done!")
    
    output = kernel(Q, K, V)
    
    ref_output = ref_program(Q, K, V)
    
    torch.npu.synchronize()
    
    print("TileLang output shape:", output.shape)
    print("Reference output shape:", ref_output.shape)
    
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("Test Passed!")