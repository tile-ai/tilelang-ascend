# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""Unchanged TA kernel bodies for eager IR migration checks.

Source: TileLang-Ascend 0deb8b93da11b7f0becbcded732d9633b7ca4e34.
Only the outer JIT decorators and example harnesses are removed.
"""

import tilelang.ascend.a2a3.language as T


# Source: examples/gemm/example_gemm.py
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# Source: examples/pipeline/matmul_add_pipeline.py
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    vec_proc = 4

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
        D: T.Tensor((M, N), dtype),  # type: ignore
        workspace_1: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            d_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            e_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.Pipelined(loop_k, num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)

                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, workspace_1[bx * block_M, by * block_N])

            for i in T.Pipelined(vec_proc, num_stages=2):
                T.copy(workspace_1[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], c_ub)
                T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc], d_ub)

                for j, k in T.Parallel(block_M // VEC_NUM, block_N // vec_proc):
                    e_ub[j, k] = c_ub[j, k] + d_ub[j, k]

                T.copy(e_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N + i * block_N // vec_proc])

    return main


# Source: examples/lightning_indexer/example_lightning_indexer.py
def indexer(B, N2, G, S1, S2, D, TOP_K, VECTOR_BASEN, VECTOR_BASEG, BLOCK_M, BLOCK_N, BLOCK_K, input_dtype="float16", calc_dtype="float"):
    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int"),
    ):
        total_process_num = N2 * S1
        each_core_process_num = total_process_num // 2
        with T.Kernel(B * N2, is_npu=True) as (cid, vid):
            n2_id = cid % N2

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)

                C_L0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                T.barrier_all()
                for n2 in T.serial(N2):
                    for g in T.serial(G):
                        for m in T.serial(S1 // BLOCK_M):
                            for n in T.serial(S2 // BLOCK_N):
                                T.barrier_all()
                                T.copy(Query[cid, m * BLOCK_M : (m + 1) * BLOCK_M, n2, g * D : (g + 1) * D], Q_L1)
                                T.barrier_all()
                                T.copy(KEY[cid, n * BLOCK_N : (n + 1) * BLOCK_N, n2, 0:D], K_L1)
                                T.barrier_all()
                                T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                                T.barrier_all()
                                T.copy(
                                    C_L0,
                                    QK_RES[
                                        cid, n2, m * BLOCK_M : (m + 1) * BLOCK_M, g, n * BLOCK_N : (n + 1) * BLOCK_N
                                    ],  # [B, N2, S1, G, S2]
                                    enable_relu=True,
                                )
                                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                # Accumulate all S2 scores, then topk once
                score_accum_ub = T.alloc_ub(S2, calc_dtype)
                topk_dst_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                topk_index_ub = T.alloc_ub(TOP_K, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int")

                s1_start_idx = vid * each_core_process_num
                s1_end_idx = s1_start_idx + each_core_process_num

                T.wait_cross_flag(0)
                for s1_id in T.serial(s1_start_idx, s1_end_idx):
                    T.barrier_all()
                    for s2_id in T.serial(S2 // VECTOR_BASEN):
                        T.barrier_all()
                        T.tile.fill(reduce_tmp_ub, 0)
                        T.tile.fill(reduce_g_ub, 0)
                        T.barrier_all()

                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.barrier_all()
                            T.copy(
                                QK_RES[
                                    cid,
                                    n2_id,
                                    s1_id,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                    s2_id * VECTOR_BASEN : (s2_id + 1) * VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.barrier_all()
                            T.copy(WEIGHTS[cid, s1_id, n2_id, g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG], weight_ub)
                            T.barrier_all()
                            for i in range(VECTOR_BASEG):
                                T.barrier_all()
                                T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                T.barrier_all()
                            T.barrier_all()
                            T.tile.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.barrier_all()
                        T.barrier_all()
                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                        T.barrier_all()
                        # Accumulate scores into flat buffer
                        for i in range(VECTOR_BASEN):
                            score_accum_ub[s2_id * VECTOR_BASEN + i] = reduce_g_ub[i]
                        T.barrier_all()
                    # After all s2 iterations, extract top K
                    T.barrier_all()
                    T.tile.topk(topk_dst_ub, score_accum_ub, TOP_K, S2)
                    T.barrier_all()
                    T.tile.gather_mask(topk_index_ub, topk_dst_ub, "P1010")
                    T.barrier_all()
                    T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                    T.barrier_all()
                    T.copy(output_ub, OUT[cid, n2_id, s1_id, 0:TOP_K])
                    T.barrier_all()

    return main


# Source: examples/lightning_indexer/example_lightning_indexer_dynamic_shape.py
def indexer_dynamic(
    N2, G, D, TOP_K, VECTOR_BASEN, VECTOR_BASEG, BLOCK_M, BLOCK_N, BLOCK_K, MAX_S2=4096, input_dtype="float16", calc_dtype="float"
):

    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int"),
    ):

        total_process_num = N2 * S1
        each_core_process_num = total_process_num // 2
        with T.Kernel(B * N2, is_npu=True) as (cid, vid):
            n2_id = cid % N2

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)

                C_L0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                T.barrier_all()
                for n2 in T.serial(N2):
                    for g in T.serial(G):
                        for m in T.serial(S1 // BLOCK_M):
                            for n in T.serial(S2 // BLOCK_N):
                                T.barrier_all()
                                T.copy(Query[cid, m * BLOCK_M : (m + 1) * BLOCK_M, n2, g * D : (g + 1) * D], Q_L1)
                                T.barrier_all()
                                T.copy(KEY[cid, n * BLOCK_N : (n + 1) * BLOCK_N, n2, 0:D], K_L1)
                                T.barrier_all()
                                T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                                T.barrier_all()
                                T.copy(
                                    C_L0,
                                    QK_RES[
                                        cid, n2, m * BLOCK_M : (m + 1) * BLOCK_M, g, n * BLOCK_N : (n + 1) * BLOCK_N
                                    ],  # [B, N2, S1, G, S2]
                                    enable_relu=True,
                                )
                                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                # Accumulate all S2 scores, then topk once
                # Use MAX_S2 for buffer sizes since S2 is dynamic
                score_accum_ub = T.alloc_ub(MAX_S2, calc_dtype)
                topk_dst_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                topk_index_ub = T.alloc_ub(TOP_K, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int")

                s1_start_idx = vid * each_core_process_num
                s1_end_idx = s1_start_idx + each_core_process_num

                T.wait_cross_flag(0)
                for s1_id in T.serial(s1_start_idx, s1_end_idx):
                    T.barrier_all()
                    for s2_id in T.serial(S2 // VECTOR_BASEN):
                        T.barrier_all()
                        T.tile.fill(reduce_tmp_ub, 0)
                        T.tile.fill(reduce_g_ub, 0)
                        T.barrier_all()

                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.barrier_all()
                            T.copy(
                                QK_RES[
                                    cid,
                                    n2_id,
                                    s1_id,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                    s2_id * VECTOR_BASEN : (s2_id + 1) * VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.barrier_all()
                            T.copy(WEIGHTS[cid, s1_id, n2_id, g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG], weight_ub)
                            T.barrier_all()
                            for i in range(VECTOR_BASEG):
                                T.barrier_all()
                                T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                T.barrier_all()
                            T.barrier_all()
                            T.tile.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.barrier_all()
                        T.barrier_all()
                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                        T.barrier_all()
                        # Accumulate scores into flat buffer
                        for i in range(VECTOR_BASEN):
                            score_accum_ub[s2_id * VECTOR_BASEN + i] = reduce_g_ub[i]
                        T.barrier_all()
                    # After all s2 iterations, extract top K
                    T.barrier_all()
                    T.tile.topk(topk_dst_ub, score_accum_ub, TOP_K, S2)
                    T.barrier_all()
                    T.tile.gather_mask(topk_index_ub, topk_dst_ub, "P1010")
                    T.barrier_all()
                    T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                    T.barrier_all()
                    T.copy(output_ub, OUT[cid, n2_id, s1_id, 0:TOP_K])
                    T.barrier_all()

    return main
