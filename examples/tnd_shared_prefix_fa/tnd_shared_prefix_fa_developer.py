"""TND Shared-Prefix FlashAttention (Developer mode, direct TND packed tensors).

Workarounds for Developer mode compiler issues:
1. CombineCV sync mismatch: GEMM + L0C→UB transfer outside if/else.
   Only T.copy(GM→L1) and T.tile.fill(L0C mask) are branched.
2. AIV cid mapping bug (threads=2 + MIX_AIC_1_2): ALL block_metadata reads
   and masking on Cube side only (AIC cid is correct). Vector side has
   NO block_metadata read.
3. V-core vid offset bug for 3D output: Output = [q_head, total_q, head_dim]
   so row stride = head_dim. Host does permute(1,0,2).
"""

import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

NEG_INF = -(2.0**30)


@tilelang.jit(out_idx=[6], pass_configs=pass_configs)
def tnd_shared_prefix_fa_developer(
    q_head,
    kv_head,
    head_dim,
    shared_prefix_len,
    max_private_kv_len,
    total_q,
    total_private_kv,
    total_q_blocks,
    block_M=128,
    block_N=64,
    sm_scale=None,
    dtype_str="float16",
    causal_mask=False,
    threads=2,
):
    sm_scale = (1.0 / head_dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = dtype_str
    accum_dtype = "float32"
    group_size = q_head // kv_head

    max_shared_iters = (shared_prefix_len + block_N - 1) // block_N
    max_private_iters = (max_private_kv_len + block_N - 1) // block_N
    total_kv_iters = max_shared_iters + max_private_iters
    block_num = total_q_blocks * q_head

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, q_head, head_dim], dtype),  # type: ignore
        K_shared: T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),  # type: ignore
        V_shared: T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),  # type: ignore
        K_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        V_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        block_metadata: T.Tensor([total_q_blocks, 4], "int32"),  # type: ignore
        Output: T.Tensor([q_head, total_q, head_dim], dtype),  # type: ignore
    ):
        with T.Kernel(block_num, threads=threads, is_npu=True) as (cid):
            tile_id = cid // q_head
            h_q = cid % q_head
            h_kv = h_q // group_size

            q_packed_start = block_metadata[tile_id, 0]
            q_valid = block_metadata[tile_id, 1]
            private_kv_start = block_metadata[tile_id, 2]
            private_kv_len = block_metadata[tile_id, 3]

            q_l1 = T.alloc_shared([block_M, head_dim], dtype)
            k_l1 = T.alloc_shared([block_N, head_dim], dtype)
            v_l1 = T.alloc_shared([block_N, head_dim], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, head_dim], accum_dtype)

            acc_o = T.alloc_shared([block_M, head_dim], accum_dtype)
            sumexp = T.alloc_shared([block_M], accum_dtype)
            m_i = T.alloc_shared([block_M], accum_dtype)

            acc_s_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([block_M], accum_dtype)
            acc_s_ub_ = T.alloc_shared([block_M, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)
            acc_s_half = T.alloc_shared([block_M, block_N], dtype)
            acc_o_ub = T.alloc_shared([block_M, head_dim], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, head_dim], dtype)

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, NEG_INF)

            T.copy(Q[q_packed_start : q_packed_start + block_M, h_q, :], q_l1)

            for k in T.serial(total_kv_iters):
                kv_start_shared = k * block_N
                priv_k = k - max_shared_iters
                kv_start_priv = priv_k * block_N
                private_offset = private_kv_start + kv_start_priv

                if k < max_shared_iters:
                    if kv_start_shared < shared_prefix_len:
                        T.copy(
                            K_shared[
                                kv_start_shared : kv_start_shared + block_N,
                                h_kv,
                                :,
                            ],
                            k_l1,
                        )
                else:
                    if kv_start_priv < private_kv_len:
                        T.copy(
                            K_private[
                                private_offset : private_offset + block_N,
                                h_kv,
                                :,
                            ],
                            k_l1,
                        )

                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)

                T.copy(acc_s_l0c, acc_s_ub_)

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)

                if k < max_shared_iters:
                    if kv_start_shared >= shared_prefix_len:
                        T.tile.fill(acc_s_ub, NEG_INF)
                    else:
                        kv_valid = shared_prefix_len - kv_start_shared
                        if kv_valid < block_N:
                            for row in T.serial(block_M):
                                for col in T.serial(block_N):
                                    if col >= kv_valid:
                                        acc_s_ub[row, col] = NEG_INF
                        if causal_mask:  # noqa: SIM102
                            if q_packed_start < shared_prefix_len:  # noqa: SIM102
                                if kv_start_shared + block_N > q_packed_start:
                                    for row in T.serial(block_M):
                                        for col in T.serial(block_N):
                                            q_pos = q_packed_start + row
                                            kv_pos = kv_start_shared + col
                                            if kv_pos > q_pos and col < kv_valid:
                                                acc_s_ub[row, col] = NEG_INF
                else:
                    if kv_start_priv >= private_kv_len:
                        T.tile.fill(acc_s_ub, NEG_INF)
                    else:
                        kv_valid = private_kv_len - kv_start_priv
                        if kv_valid < block_N:
                            for row in T.serial(block_M):
                                for col in T.serial(block_N):
                                    if col >= kv_valid:
                                        acc_s_ub[row, col] = NEG_INF
                        if causal_mask:  # noqa: SIM102
                            if shared_prefix_len + kv_start_priv + block_N > q_packed_start:
                                for row in T.serial(block_M):
                                    for col in T.serial(block_N):
                                        q_pos = q_packed_start + row
                                        kv_pos = shared_prefix_len + kv_start_priv + col
                                        if kv_pos > q_pos and col < kv_valid:
                                            acc_s_ub[row, col] = NEG_INF

                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in range(block_M):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_l1)

                if k < max_shared_iters:
                    if kv_start_shared < shared_prefix_len:
                        T.copy(
                            V_shared[
                                kv_start_shared : kv_start_shared + block_N,
                                h_kv,
                                :,
                            ],
                            v_l1,
                        )
                else:
                    if kv_start_priv < private_kv_len:
                        T.copy(
                            V_private[
                                private_offset : private_offset + block_N,
                                h_kv,
                                :,
                            ],
                            v_l1,
                        )

                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, acc_o_ub)

                for h_i in range(block_M):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(block_M):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            if threads == 1:
                T.copy(
                    acc_o_half,
                    Output[
                        h_q,
                        q_packed_start : q_packed_start + q_valid,
                        :,
                    ],
                )
            else:
                T.copy(
                    acc_o_half,
                    Output[
                        h_q,
                        q_packed_start : q_packed_start + block_M,
                        :,
                    ],
                )

    return main
