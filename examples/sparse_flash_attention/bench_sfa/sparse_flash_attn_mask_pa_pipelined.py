import tilelang
from tilelang import DataType, language as T
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


def init_test():
    torch.set_default_device('npu')
    torch.manual_seed(0)

    tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[7,8,9,10,11], pass_configs=pass_configs)
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    dtype="bfloat16",
    block_num = 516,
    block_size = 128,
    core_num = 24,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal, 'non-casual is not supported'
    assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'

    # NOTE: ascend only support exp interface instead of exp2
    sm_scale = (1.0 / (dim + tail_dim))**0.5 if sm_scale is None else sm_scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    block_table_len = T.symbolic("block_table_len")

    seq_len_kv = T.symbolic("seq_len_kv")
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [seq_len, kv_group, topk]
    # lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    accum_dtype = "float"

    H = head_kv

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, 'here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)'

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    block_H = 16
    if head_kv > block_H:
        assert head_kv % block_H == 0, 'head_kv should be a multiple of {block_H}'
        REPLICATE_H = head_kv // block_H
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else block_H

    v_block = H_per_block // 2

    kv_shape = [block_num, block_size, 1, D + D_tail]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            actual_q_len: T.Tensor([batch], indices_dtype),
            actual_kv_len: T.Tensor([batch], indices_dtype),
            block_table: T.Tensor([batch, block_table_len], indices_dtype),
            workspace_1: T.Tensor([core_num, BI, D], dtype),  # T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([core_num, BI, D_tail], dtype),  # T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([core_num, H_per_block, BI], accum_dtype),  # T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([core_num, H_per_block, BI], dtype),  # T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([core_num, H_per_block, D], accum_dtype),  # T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # Alloc Memory
            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            ## 2. Vector
            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block, 1], accum_dtype)
            m_i = T.alloc_ub([v_block, 1], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            indices_ub_float = T.alloc_ub([BI], "float")

            kv_ub_gather = T.alloc_ub([BI // 2, D], dtype)
            kv_tail_ub_gather = T.alloc_ub([BI // 2, D_tail], dtype)

            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block, 1], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            tmp_ub = T.alloc_ub([1 * DataType(accum_dtype).bits // 8 * v_block * BI], "uint8")
            sumexp_i_ub = T.alloc_ub([v_block, 1], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)
            # mask_ub = T.alloc_ub([BI // 8], "uint8")
            mask_ub = T.alloc_ub([32], "uint8") # T.Pipelined need to align


            # Broadcast target buffers
            m_i_broadcast = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev_broadcast = T.alloc_ub([v_block, D], accum_dtype)
            sumexp_broadcast = T.alloc_ub([v_block, D], accum_dtype)

            # fixed core
            for core_index in T.serial(T.ceildiv(seq_len * REPLICATE_H * batch * kv_group, core_num)):
                pid = core_index * core_num + cid
                if pid < seq_len * REPLICATE_H * batch * kv_group:
                    bx = pid % (seq_len * REPLICATE_H)
                    by = pid // (seq_len * REPLICATE_H) % batch
                    bz = pid // (seq_len * REPLICATE_H) // batch % kv_group
                
                    b_i = by
                    g_i = bz

                    s_i = (bx // REPLICATE_H)
                    h_i = (bx % REPLICATE_H)

                    H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * block_H)
                    H1 = H0 + H_per_block
                    act_q_len = actual_q_len[b_i]
                    actual_len = actual_kv_len[b_i]


                    if s_i < act_q_len:
                        T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                        T.copy(Q[b_i, s_i, H0:H1, D:], q_tail_l1)

                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2.0**30)           

                        for i_i in T.serial(NI):
                        # for i_i in T.Pipelined(NI, num_stages=2):

                        
                            T.copy(workspace_1[cid, 0:BI, 0:D], kv_l1)
                            T.copy(workspace_2[cid, 0:BI, 0:D_tail], kv_tail_l1)
                        

                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                        
                            T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])
                            T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1)
                        

                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                        
                            T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])
                    

                            T.copy(Indices[s_i, g_i, i_i * BI:i_i * BI + BI], indices_ub_)
                        
                            T.copy(indices_ub_, indices_ub_float)
                        
                        
                            T.tile.compare(mask_ub, indices_ub_float, T.float32(actual_len - act_q_len + s_i), "LE")
                        

                            for bi_i in range(BI // 2):
                                index_i = indices_ub_[bi_i + vid * BI // 2]
                            

                                block_idx = index_i // block_size
                                block_i = block_table[b_i, block_idx]
                                block_inter = index_i % block_size
                              
                                T.copy(KV[block_i, block_inter, 0, :D], kv_ub_gather[bi_i,:])
                                T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub_gather[bi_i,:])

                            
                            T.copy(kv_ub_gather, workspace_1[cid, vid * BI // 2: (vid + 1) * BI // 2, :])
                            T.copy(kv_tail_ub_gather, workspace_2[cid, vid * BI // 2: (vid + 1) * BI // 2, :])

                            T.tile.fill(acc_s_ub_, 0.0)
                        

                            for i in T.serial(v_block):
                                T.tile.select(acc_s_ub[i, :], mask_ub, acc_s_ub_[i, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")
                            

                            T.copy(m_i, m_i_prev)
                        

                            T.copy(
                                workspace_3[cid, vid * v_block:vid * v_block + v_block, :],
                                acc_s_ub_)
                        

                            T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                        

                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                        

                            T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)

                            T.tile.max(m_i, m_i, m_i_prev)
                        

                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                        

                            T.tile.exp(m_i_prev, m_i_prev)
                        

                            T.tile.broadcast(m_i_broadcast, m_i, tmp_ub)
                            T.tile.sub(acc_s_ub, acc_s_ub, m_i_broadcast)

                            T.tile.exp(acc_s_ub, acc_s_ub)
                        

                            T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
                        

                            T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                        

                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                                                    
                            T.copy(acc_s_ub, acc_s_half)


                            T.copy(
                                acc_s_half, workspace_4[cid,
                                                        vid * v_block:vid * v_block + v_block, :])
 

                            T.copy(
                                workspace_5[cid, vid * v_block:vid * v_block + v_block, :],
                                acc_o_ub)


                            T.tile.broadcast(m_i_prev_broadcast, m_i_prev, tmp_ub)
                            T.tile.mul(acc_o, acc_o, m_i_prev_broadcast)  

                            T.tile.add(acc_o, acc_o, acc_o_ub)

                        
                        T.tile.broadcast(sumexp_broadcast, sumexp, tmp_ub)
                        T.tile.div(acc_o, acc_o, sumexp_broadcast)

                        T.copy(acc_o, acc_o_half)
                    
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block:H1 + vid * v_block, :])

    return main




core_num = 24

block_num = 20
block_size = 128




def sparse_attn_tilelang(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope=None, key_rope=None,
    layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None):

    query = query.unsqueeze(0)
    query_rope = query_rope.unsqueeze(0)
    block_num, block_size, num_head_kv, dim = key.size()
    # print("query_rope.shape=",query_rope.shape)
    # print("key_rope.shape=",key_rope.shape)
    print("*" * 50)
    query = torch.cat((query, query_rope), dim=-1)
    key_value = torch.cat((key, key_rope), dim=-1)
    print("q.shape=",query.shape)
    print("kv.shape=",key_value.shape)
    print("indices=",sparse_indices.shape)
    print("actual_q_len=",actual_seq_lengths_query)
    print("actual_kv_len=",actual_seq_lengths_kv)
    print("block_table=",block_table.shape)
    kernel = sparse_attention_fwd(
        heads=128,
        dim=512,
        tail_dim=64,
        topk=2048,
        sm_scale=scale_value,
        core_num=24,
        block_num=block_num,
        block_size=block_size
    )
    print(kernel.get_kernel_source())
    output = kernel(query, key_value, sparse_indices, actual_seq_lengths_query, actual_seq_lengths_kv, block_table)
    output = output.squeeze(0)
    return output


