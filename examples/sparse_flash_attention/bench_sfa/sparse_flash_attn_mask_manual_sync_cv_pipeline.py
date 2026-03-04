import tilelang
from tilelang import DataType, language as T
import torch
import os
import sys

from tilelang.intrinsics import make_zn_layout

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


def init_test():
    # torch.set_default_device('npu')
    torch.manual_seed(42)
    tilelang.disable_cache()


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[7, 8, 9, 10, 11], pass_configs=pass_configs)
def sparse_attention_fwd(
        q_heads,
        dim,
        rope_dim,
        topk,
        kv_heads=1,
        scale=None,
        is_causal=True,
        n_base_size=64,
        m_base_size=16,
        dtype="bfloat16",
        block_num=516,
        block_size=128,
        core_num=24,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert rope_dim == tilelang.math.next_power_of_2(
        rope_dim), f"haven't check padding correctness yet, dim={rope_dim}"
    assert is_causal, 'non-casual is not supported'
    assert topk % n_base_size == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'

    # NOTE: ascend only support exp interface instead of exp2
    sm_scale = (1.0 / (dim + rope_dim)) ** 0.5 if scale is None else scale

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    block_table_len = T.symbolic("block_table_len")

    seq_len_kv = T.symbolic("seq_len_kv")
    g = q_heads // kv_heads  # GQA G
    q_shape = [batch, seq_len, q_heads, dim + rope_dim]

    o_shape = [batch, seq_len, q_heads, dim]
    indices_shape = [seq_len, kv_heads, topk]
    kv_shape = [block_num, block_size, 1, dim + rope_dim]

    indices_dtype = "int32"
    accum_dtype = "float"

    n_block_num = T.ceildiv(topk, n_base_size)

    if g > m_base_size:
        assert g % m_base_size == 0, 'head_kv should be a multiple of {block_H}'
        g_block_num = g // m_base_size
    else:
        g_block_num = 1

    kernel_count = batch * seq_len * g_block_num * kv_heads
    m_base_size_v = m_base_size // 2
    n_base_size_v = n_base_size // 2
    vec0_copy_out_size = 16

    n_num = 2
    block_k = T.ceildiv(n_base_size, n_num)

    @T.prim_func
    def main_pipelined(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            actual_q_len: T.Tensor([batch], indices_dtype),
            actual_kv_len: T.Tensor([batch], indices_dtype),
            block_table: T.Tensor([batch, block_table_len], indices_dtype),
            workspace_1: T.Tensor([core_num, n_base_size, dim], dtype),
            workspace_2: T.Tensor([core_num, n_base_size, rope_dim], dtype),
            workspace_3: T.Tensor([core_num, m_base_size, n_base_size], accum_dtype),
            workspace_4: T.Tensor([core_num, m_base_size, n_base_size], dtype),
            workspace_5: T.Tensor([core_num, m_base_size, dim], accum_dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            # Alloc Memory
            q_l1 = T.alloc_L1([m_base_size, dim], dtype)
            q_tail_l1 = T.alloc_L1([m_base_size, rope_dim], dtype)
            kv_l1 = T.alloc_L1([n_base_size, dim], dtype)
            kv_tail_l1 = T.alloc_L1([n_base_size, rope_dim], dtype)
            acc_s_l1 = T.alloc_L1([m_base_size, n_base_size], dtype)

            acc_o_l0c = T.alloc_L0C([m_base_size, dim], accum_dtype)

            acc_s_l0c = T.alloc_L0C([m_base_size, n_base_size], accum_dtype)

            ## 2. Vector
            acc_o = T.alloc_ub([m_base_size_v, dim], accum_dtype)
            sumexp = T.alloc_ub([m_base_size_v, 1], accum_dtype)
            m_i = T.alloc_ub([m_base_size_v, 1], accum_dtype)

            indices_ub_ = T.alloc_ub([n_base_size], indices_dtype)
            indices_ub_float = T.alloc_ub([n_base_size], "float")

            kv_ub_gather = T.alloc_ub([2, vec0_copy_out_size, dim], dtype)
            kv_rope_ub_gather = T.alloc_ub([2, vec0_copy_out_size, rope_dim], dtype)

            acc_s_ub = T.alloc_ub([m_base_size_v, n_base_size], accum_dtype)
            m_i_prev = T.alloc_ub([m_base_size_v, 1], accum_dtype)
            acc_s_ub_ = T.alloc_ub([m_base_size_v, n_base_size], accum_dtype)
            tmp_ub = T.alloc_ub([1 * DataType(accum_dtype).bits // 8 * m_base_size_v * n_base_size], "uint8")
            sumexp_i_ub = T.alloc_ub([m_base_size_v, 1], accum_dtype)
            acc_s_half = T.alloc_ub([m_base_size_v, n_base_size], dtype)
            acc_o_ub = T.alloc_ub([m_base_size_v, dim], accum_dtype)
            acc_o_half = T.alloc_ub([m_base_size_v, dim], dtype)
            mask_ub = T.alloc_ub([32], "uint8")  # T.Pipelined need to align

            # Broadcast target buffers
            m_i_broadcast = T.alloc_ub([m_base_size_v, n_base_size], accum_dtype)
            m_i_prev_broadcast = T.alloc_ub([m_base_size_v, dim], accum_dtype)
            sumexp_broadcast = T.alloc_ub([m_base_size_v, dim], accum_dtype)

            single_core_load = T.ceildiv(kernel_count, core_num)
            used_core_num = T.ceildiv(kernel_count, single_core_load)
            tail_block_size = kernel_count - (used_core_num - 1) * single_core_load
            start_idx = cid * single_core_load
            end_idx = T.if_then_else(cid == used_core_num - 1, start_idx + tail_block_size,
                                     start_idx + single_core_load)

            if cid < used_core_num:
                for block_idx in T.serial(start_idx, end_idx):

                    # bz = block_idx % kv_heads  # h轴方向
                    # bx = block_idx // kv_heads % (g_block_num * seq_len)  # s1g方向
                    # by = block_idx // kv_heads // (g_block_num * seq_len) % batch  # batch方向

                    bx = block_idx % (g_block_num * seq_len)  # s1g方向
                    by = block_idx // (g_block_num * seq_len) % batch  # b方向
                    bz = block_idx // (g_block_num * seq_len) // batch % kv_heads  # n2方向

                    b_i = by  # batch
                    g_i = bz  # n2
                    s1g_i = bx

                    s_i = (s1g_i // g_block_num)  # s1

                    H0 = g_i * g + (0 if g_block_num == 1 else (s1g_i % g_block_num) * m_base_size)
                    H1 = H0 + m_base_size
                    act_q_len = actual_q_len[b_i]
                    actual_len = actual_kv_len[b_i]

                    if s_i < act_q_len:
                        # 初始化操作
                        T.copy(Q[b_i, s_i, H0:H1, :dim], q_l1)
                        T.copy(Q[b_i, s_i, H0:H1, dim:], q_tail_l1)

                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2.0 ** 30)

                        # s2轴切分
                        # for i_i in T.serial(n_block_num):
                        for i_i in T.Pipelined(n_block_num, num_stages=2):
                            # ******************** V0(处理topk) ********************
                            T.copy(Indices[s_i, g_i, i_i * n_base_size:i_i * n_base_size + n_base_size], indices_ub_)
                            T.set_flag("mte2", "v", 5)
                            T.wait_flag("mte2", "v", 5)
                            T.copy(indices_ub_, indices_ub_float)
                            T.pipe_barrier("v")
                            T.tile.compare(mask_ub, indices_ub_float, T.float32(actual_len - act_q_len + s_i), "LE")

                            # 后续开始判断
                            for bi_i in range(n_base_size_v):
                                inner_block_id = T.floordiv(bi_i, vec0_copy_out_size)  # 当前任务
                                idx = bi_i % vec0_copy_out_size
                                task_id = inner_block_id % 2

                                if bi_i > (2 * vec0_copy_out_size - 1) and bi_i % vec0_copy_out_size == 0:
                                    T.wait_flag("mte3", "mte2", task_id)

                                # 从topk中取出一个index索引
                                index_i = indices_ub_[bi_i + vid * n_base_size_v]
                                T.pipe_barrier("v")
                                block_idx = index_i // block_size
                                block_i = block_table[b_i, block_idx]
                                block_inter = index_i % block_size

                                T.copy(KV[block_i, block_inter, 0, :dim], kv_ub_gather[task_id, idx, :])
                                T.copy(KV[block_i, block_inter, 0, dim:], kv_rope_ub_gather[task_id, idx, :])

                                if (bi_i + 1) % vec0_copy_out_size == 0:
                                    T.set_flag("mte2", "mte3", task_id)
                                    T.wait_flag("mte2", "mte3", task_id)
                                    T.copy(kv_ub_gather[task_id, :, :],
                                           workspace_1[cid,
                                           inner_block_id * vec0_copy_out_size + vid * n_base_size_v
                                           : (inner_block_id + 1) * vec0_copy_out_size + vid * n_base_size_v, :])
                                    T.copy(kv_rope_ub_gather[task_id, :, :],
                                           workspace_2[cid,
                                           inner_block_id * vec0_copy_out_size + vid * n_base_size_v:
                                           (inner_block_id + 1) * vec0_copy_out_size + vid * n_base_size_v, :])

                                    if bi_i < n_base_size_v - 2 * vec0_copy_out_size:
                                        T.set_flag("mte3", "mte2", task_id)

                            # ******************** BMM1(Q*K) ********************
                            T.copy(workspace_1[cid, :, 0:dim], kv_l1)
                            T.copy(workspace_2[cid, :, 0:rope_dim], kv_tail_l1)

                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)

                            T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                            T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)

                            T.set_flag("m", "fix", 2)
                            T.wait_flag("m", "fix", 2)

                            T.copy(acc_s_l0c, workspace_3[cid, :, :])

                            # ******************** V1 ********************
                            T.tile.fill(acc_s_ub_, 0.0)
                            for i in T.serial(m_base_size_v):
                                T.tile.select(acc_s_ub[i, :], mask_ub, acc_s_ub_[i, :], -T.infinity(accum_dtype),
                                              "VSEL_TENSOR_SCALAR_MODE")
                            T.copy(m_i, m_i_prev)

                            # T.barrier_all()  # 需优化

                            T.copy(
                                workspace_3[cid, vid * m_base_size_v:vid * m_base_size_v + m_base_size_v, :],
                                acc_s_ub_)

                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                            T.pipe_barrier("v")

                            T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                            T.pipe_barrier("v")

                            T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
                            T.pipe_barrier("v")

                            T.tile.max(m_i, m_i, m_i_prev)
                            T.pipe_barrier("v")

                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                            T.pipe_barrier("v")

                            T.tile.exp(m_i_prev, m_i_prev)
                            T.pipe_barrier("v")

                            T.tile.broadcast(m_i_broadcast, m_i, tmp_ub)
                            T.pipe_barrier("v")

                            T.tile.sub(acc_s_ub, acc_s_ub, m_i_broadcast)
                            T.pipe_barrier("v")

                            T.tile.exp(acc_s_ub, acc_s_ub)
                            T.pipe_barrier("v")

                            T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
                            T.pipe_barrier("v")

                            T.tile.mul(sumexp, sumexp, m_i_prev)  # check
                            T.pipe_barrier("v")

                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                            T.pipe_barrier("v")

                            T.copy(acc_s_ub, acc_s_half)
                            T.pipe_barrier("v")

                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)

                            # T.dump_tensor(acc_s_half, 111, 513)

                            T.copy(
                                acc_s_half, workspace_4[cid,
                                            vid * m_base_size_v:vid * m_base_size_v + m_base_size_v, :])

                            # ******************** BMM2(S*V) ********************
                            T.copy(workspace_4[cid, :, :], acc_s_l1)
                            T.set_flag("mte2", "mte1", 3)
                            T.wait_flag("mte2", "mte1", 3)
                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                            T.set_flag("m", "fix", 4)
                            T.wait_flag("m", "fix", 4)
                            T.copy(acc_o_l0c, workspace_5[cid, :, :])

                            # ******************** VEC2 ********************
                            T.copy(
                                workspace_5[cid, vid * m_base_size_v:vid * m_base_size_v + m_base_size_v, :],
                                acc_o_ub)

                            T.tile.broadcast(m_i_prev_broadcast, m_i_prev, tmp_ub)
                            T.pipe_barrier("v")

                            T.tile.mul(acc_o, acc_o, m_i_prev_broadcast)
                            T.pipe_barrier("v")

                            T.set_flag("mte2", "v", 2)
                            T.wait_flag("mte2", "v", 2)

                            T.tile.add(acc_o, acc_o, acc_o_ub)

                        T.tile.broadcast(sumexp_broadcast, sumexp, tmp_ub)
                        T.pipe_barrier("v")

                        T.tile.div(acc_o, acc_o, sumexp_broadcast)
                        T.pipe_barrier("v")

                        T.copy(acc_o, acc_o_half)

                        T.set_flag("v", "mte3", 9)
                        T.wait_flag("v", "mte3", 9)
                        T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * m_base_size_v:H1 + vid * m_base_size_v, :])

    return main_pipelined


core_num = 24

block_num = 20
block_size = 128


def sparse_attn_tilelang(
        query, key, value, sparse_indices, scale_value, sparse_block_size,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope=None, key_rope=None,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None, attention_mode=None):
    query = query.unsqueeze(0)
    query_rope = query_rope.unsqueeze(0)
    print(query.shape)
    block_num, block_size, num_head_kv, dim = key.size()
    # print("query_rope.shape=",query_rope.shape)
    # print("key_rope.shape=",key_rope.shape)
    print("*" * 50)
    query = torch.cat((query, query_rope), dim=-1)
    key_value = torch.cat((key, key_rope), dim=-1)
    print("q.shape=", query.shape)
    print("kv.shape=", key_value.shape)
    print("indices=", sparse_indices.shape)
    print("actual_q_len=", actual_seq_lengths_query)
    print("actual_kv_len=", actual_seq_lengths_kv)
    print("block_table=", block_table.shape)
    kernel = sparse_attention_fwd(
        q_heads=128,
        dim=512,
        rope_dim=64,
        topk=2048,
        scale=scale_value,
        core_num=24,
        block_num=block_num,
        block_size=block_size
    )
    print(kernel.get_kernel_source())
    output = kernel(query, key_value, sparse_indices, actual_seq_lengths_query, actual_seq_lengths_kv, block_table)
    output = output.squeeze(0)
    return output
