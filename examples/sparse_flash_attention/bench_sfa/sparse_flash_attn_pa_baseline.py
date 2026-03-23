import tilelang
from tilelang import DataType, language as T
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def init_test():
    torch.set_default_device('npu')
    torch.manual_seed(42)
    tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[7,8,9,10,11], pass_configs=pass_configs)
def sparse_attention_fwd(
    q_heads,
    dim,
    rope_dim,
    topk,
    kv_heads=1,
    scale=None,
    is_causal=True,
    block_N=64,
    dtype="bfloat16",
    block_size = 128,
    core_num = 24,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert rope_dim == tilelang.math.next_power_of_2(
        rope_dim), f"haven't check padding correctness yet, dim={rope_dim}"
    assert is_causal, 'non-casual is not supported'
    assert topk % block_N == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'

    # NOTE: ascend only support exp interface instead of exp2
    scale = (1.0 / (dim + rope_dim))**0.5 if scale is None else scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    block_table_len = T.symbolic("block_table_len")
    block_num = T.symbolic("block_num")

    g = q_heads // kv_heads
    q_shape = [batch, seq_len, q_heads, dim + rope_dim]
    o_shape = [batch, seq_len, q_heads, dim]
    kv_shape = [block_num, block_size, 1, dim + rope_dim]
    indices_shape = [seq_len, kv_heads, topk]

    indices_dtype = "int32"
    accum_dtype = "float"

    n_block_num = T.ceildiv(topk, block_N)

    block_M = 64
    if g > block_M:
        assert g % block_M == 0, 'g should be a multiple of {block_M}'
        g_block_num = g // block_M
    else:
        g_block_num = 1

    vec_block_M = block_M // 2
    vec_block_N = block_N // 2
    
    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Output: T.Tensor(o_shape, dtype),
            actual_q_len: T.Tensor([batch], indices_dtype),
            actual_kv_len: T.Tensor([batch], indices_dtype),
            block_table: T.Tensor([batch, block_table_len], indices_dtype),
            workspace_1: T.Tensor([core_num, block_N, dim], dtype),
            workspace_2: T.Tensor([core_num, block_N, rope_dim], dtype),
            workspace_3: T.Tensor([core_num, block_M, block_N], accum_dtype),
            workspace_4: T.Tensor([core_num, block_M, block_N], dtype),
            workspace_5: T.Tensor([core_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):

            q_l1 = T.alloc_L1([block_M, dim], dtype)
            q_tail_l1 = T.alloc_L1([block_M, rope_dim], dtype)
            kv_l1 = T.alloc_L1([block_N, dim], dtype)
            kv_tail_l1 = T.alloc_L1([block_N, rope_dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([vec_block_M, dim], accum_dtype)
            sumexp = T.alloc_ub([vec_block_M], accum_dtype)
            m_i = T.alloc_ub([vec_block_M], accum_dtype)
            indices_ub_ = T.alloc_ub([block_N], indices_dtype)
            indices_ub_float = T.alloc_ub([block_N], "float")
            kv_ub = T.alloc_ub([dim], dtype)
            kv_tail_ub = T.alloc_ub([rope_dim], dtype)
            acc_s_ub = T.alloc_ub([vec_block_M, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([vec_block_M], accum_dtype)
            acc_s_ub_ = T.alloc_ub([vec_block_M, block_N], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * vec_block_M * block_N], "uint8")
            sumexp_i_ub = T.alloc_ub([vec_block_M], accum_dtype)
            acc_s_half = T.alloc_ub([vec_block_M, block_N], dtype)
            acc_o_ub = T.alloc_ub([vec_block_M, dim], accum_dtype)
            acc_o_half = T.alloc_ub([vec_block_M, dim], dtype)
            mask_ub = T.alloc_ub([block_N // 8], "uint8")
            mask_ub_2 = T.alloc_ub([block_N // 8], "uint8")

            for core_index in T.serial(T.ceildiv(seq_len * g_block_num * batch * kv_heads, core_num)):
                pid = core_index * core_num + cid
                if pid < seq_len * g_block_num * batch * kv_heads:
                    bx = pid % (seq_len * g_block_num)
                    by = pid // (seq_len * g_block_num) % batch
                    bz = pid // (seq_len * g_block_num) // batch % kv_heads
                  
                    b_i = by
                    g_i = bz

                    s_i = (bx // g_block_num)
                    h_i = (bx % g_block_num)

                    H0 = g_i * g_block_num + (0 if g_block_num == 1 else (bx % g_block_num) * 64)
                    H1 = H0 + block_M
                    act_q_len = actual_q_len[b_i]


                    if s_i < act_q_len:
                        T.copy(Q[b_i, s_i, H0:H1, :dim], q_l1)
                        T.copy(Q[b_i, s_i, H0:H1, dim:], q_tail_l1)

                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2.0**30)             

                        for i_i in T.serial(n_block_num):
                          
                            T.copy(workspace_1[cid, 0:block_N, 0:dim], kv_l1)
                            T.copy(workspace_2[cid, 0:block_N, 0:rope_dim], kv_tail_l1)
                          

                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                          
                            T.copy(acc_s_l0c, workspace_3[cid, 0:block_M, 0:block_N])
                            T.copy(workspace_4[cid, 0:block_M, 0:block_N], acc_s_l1)
                          

                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                          
                            T.copy(acc_o_l0c, workspace_5[cid, 0:block_M, 0:dim])
                      

                            T.copy(Indices[s_i, g_i, i_i * block_N:i_i * block_N + block_N], indices_ub_)
                          
                            T.copy(indices_ub_, indices_ub_float)
                          
                            actual_len = actual_kv_len[b_i]
                          
                            valid_kv_len = T.Min(T.float32(s_i), T.float32(actual_len))
                          
                            T.tile.compare(mask_ub, indices_ub_float, T.float32(actual_len - act_q_len + s_i), "LE")
                            T.tile.compare(mask_ub_2, indices_ub_float, T.float32(-1.0), "NE")
                          
                            T.tile.bitwise_and(mask_ub, mask_ub, mask_ub_2)

                            for block_N_i in range(block_N // 2):
                                index_i = indices_ub_[block_N_i + vid * block_N // 2]
                              
                                if index_i > -1:
                                    block_idx = index_i // block_size
                                    block_i = block_table[b_i, block_idx]
                                    block_inter = index_i % block_size
                                  
                                    T.copy(KV[block_i, block_inter, 0, :dim], kv_ub)
                                    T.copy(KV[block_i, block_inter, 0, dim:], kv_tail_ub)
                                else:
                                    T.tile.fill(kv_ub, 0.0)
                                    T.tile.fill(kv_tail_ub, 0.0)
                              
                                T.copy(kv_ub, workspace_1[cid, block_N_i + vid * block_N // 2, :])
                                T.copy(kv_tail_ub, workspace_2[cid, block_N_i + vid * block_N // 2, :])
                              


                            T.tile.fill(acc_s_ub_, 0.0)
                          

                            for i in T.serial(vec_block_M):
                                T.tile.select(acc_s_ub[i, :], mask_ub, acc_s_ub_[i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")
                              

                            T.copy(m_i, m_i_prev)
                          

                            T.copy(
                                workspace_3[cid, vid * vec_block_M:vid * vec_block_M + vec_block_M, :],
                                acc_s_ub_)
                          

                            T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                          

                            T.tile.mul(acc_s_ub, acc_s_ub, scale)
                          

                            T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)

                            T.tile.max(m_i, m_i, m_i_prev)
                          


                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                          

                            T.tile.exp(m_i_prev, m_i_prev)
                          

                            for h_i in range(vec_block_M):
                              
                                T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i]) 
                              

                            T.tile.exp(acc_s_ub, acc_s_ub)
                          

                            T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
                          

                            T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                          

                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                          

                            for h_i in range(vec_block_M):
                              
                                T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                              

                            T.copy(acc_s_ub, acc_s_half)
                          

                            T.copy(
                                acc_s_half, workspace_4[cid,
                                                        vid * vec_block_M:vid * vec_block_M + vec_block_M, :])
                          

                            T.copy(
                                workspace_5[cid, vid * vec_block_M:vid * vec_block_M + vec_block_M, :],
                                acc_o_ub)
                          

                            T.tile.add(acc_o, acc_o, acc_o_ub)
                          

                          

                        for h_i in range(vec_block_M):
                          
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                          

                        T.copy(acc_o, acc_o_half)
                      
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * vec_block_M:H1 + vid * vec_block_M, :])

    return main


def sparse_attn_tilelang(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope=None, key_rope=None,
    layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None):

    query = query.unsqueeze(0)
    query_rope = query_rope.unsqueeze(0)
    block_num, block_size, num_head_kv, dim = key.size()

    query = torch.cat((query, query_rope), dim=-1)
    key_value = torch.cat((key, key_rope), dim=-1)

    kernel = sparse_attention_fwd(
        q_heads=128,
        dim=512,
        rope_dim=64,
        topk=2048,
        scale=scale_value,
        core_num=24,
        block_size=block_size
    )
    output = kernel(query, key_value, sparse_indices, actual_seq_lengths_query, actual_seq_lengths_kv, block_table)
    output = output.squeeze(0)
    return output