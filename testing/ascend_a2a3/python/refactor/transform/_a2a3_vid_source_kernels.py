"""Unchanged kernel factories from fixed TA 0deb8b93 vid-reduction tests.

Source: testing/python/language/cvseparate/
    test_tilelang_ascend_language_vid_reduction.py.
Only the dialect import and formatter output differ; runtime tests stay separate.
"""

import tilelang.ascend.a2a3.language as T


def gm_ub_gm_identity(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


def parallel_elementwise_1d(M, block_M, dtype="float16"):
    m_num = M // block_M

    @T.prim_func
    def main(A: T.Tensor((M,), dtype), C: T.Tensor((M,), dtype)):
        with T.Kernel(m_num, threads=2, is_npu=True) as (cid):
            bx = cid % m_num
            a_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M,), dtype)
            T.copy(A[bx * block_M], a_ub)
            for i in T.Parallel(block_M):
                c_ub[i] = a_ub[i] * 2.0 + 1.0
            T.copy(c_ub, C[bx * block_M])

    return main


def gm_gather_skip_set_dual(B, SKV, G, D_full, D_main, D_tail, BI, dtype="float16"):
    block_num = B * 4

    @T.prim_func
    def main(
        KV: T.Tensor((B, SKV, G, D_full), dtype),
        IDX: T.Tensor((B, 4, G, BI), "int32"),
        ws1: T.Tensor((block_num, BI, D_main), dtype),
        ws2: T.Tensor((block_num, BI, D_tail), dtype),
    ):
        with T.Kernel(block_num, threads=2, is_npu=True) as (cid):
            b_i = cid // 4
            s_i = cid % 4
            g_i = 0

            idx_ub = T.alloc_shared((BI,), "int32")
            kv_ub = T.alloc_shared((D_main,), dtype)
            kv_tail_ub = T.alloc_shared((D_tail,), dtype)

            T.copy(IDX[b_i, s_i, g_i, 0:BI], idx_ub)

            for bi_i in range(BI):
                T.copy(KV[b_i, idx_ub[bi_i], g_i, :D_main], kv_ub)
                T.copy(KV[b_i, idx_ub[bi_i], g_i, D_main:], kv_tail_ub)
                T.copy(kv_ub, ws1[cid, bi_i, :])
                T.copy(kv_tail_ub, ws2[cid, bi_i, :])

    return main


def tile_ops_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            b_ub = T.alloc_shared((block_M, block_N), dtype)
            temp_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)

            # binary_op: add/sub/max/min/mul/div — chain: temp = 1
            T.tile.add(temp_ub, a_ub, a_ub)  # temp = 2a
            T.tile.sub(temp_ub, temp_ub, a_ub)  # temp = a
            T.tile.max(temp_ub, temp_ub, a_ub)  # temp = a
            T.tile.min(temp_ub, temp_ub, a_ub)  # temp = a
            T.tile.mul(temp_ub, temp_ub, a_ub)  # temp = a^2
            T.tile.add(temp_ub, temp_ub, 1.0)  # temp = a^2 + 1
            T.tile.div(temp_ub, temp_ub, temp_ub)  # temp = 1

            # unary_op: exp/sqrt/relu — temp = sqrt(e)
            T.tile.exp(temp_ub, temp_ub)  # temp = e
            T.tile.sqrt(temp_ub, temp_ub)  # temp = sqrt(e)
            T.tile.relu(temp_ub, temp_ub)  # temp = sqrt(e)

            # scalar_op: leaky_relu/axpy — temp = a + sqrt(e)
            T.tile.leaky_relu(temp_ub, temp_ub, 0.1)  # temp = sqrt(e)
            T.tile.axpy(temp_ub, a_ub, 1.0)  # temp = a + sqrt(e)

            # sin/cos: b = sin(temp) + cos(a) = sin(a + sqrt(e)) + cos(a)
            T.tile.sin(b_ub, temp_ub)  # b = sin(a + sqrt(e))
            T.tile.cos(temp_ub, a_ub)  # temp = cos(a)
            T.tile.add(b_ub, b_ub, temp_ub)  # b = sin(a + sqrt(e)) + cos(a)

            # compare (Pass processes its size; result not in output chain)
            T.tile.compare(temp_ub, a_ub, 0.0, "GT")

            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def tile_cast_ops_kernel(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), "float16"), B: T.Tensor((M, N), "float16")):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), "float16")
            w_ub = T.alloc_shared((block_M, block_N), "float16")
            a_f32 = T.alloc_shared((block_M, block_N), "float32")
            w_f32 = T.alloc_shared((block_M, block_N), "float32")
            out_f32 = T.alloc_shared((block_M, block_N), "float32")
            b_ub = T.alloc_shared((block_M, block_N), "float16")

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(A[bx * block_M, by * block_N], w_ub)

            # cast: last arg is the element count
            T.tile.cast(a_f32, a_ub, "CAST_NONE", block_M * block_N)
            T.tile.cast(w_f32, w_ub, "CAST_NONE", block_M * block_N)

            # mul_add_dst: a_f32 = w_f32 * w_f32 + a_f32
            T.tile.mul_add_dst(a_f32, w_f32, w_f32)

            # silu: out = a / (1 + exp(-a))
            T.tile.silu(out_f32, a_f32)

            # cast back to fp16
            T.tile.cast(b_ub, out_f32, "CAST_RINT", block_M * block_N)

            T.copy(b_ub, B[bx * block_M, by * block_N])

    return main


def tile_parallel_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            scale_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.fill(scale_ub, 2.0)
            for h_i, j in T.Parallel(block_M, block_N):
                c_ub[h_i, j] = a_ub[h_i, j] * scale_ub[h_i]
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def tile_range_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            scale_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.tile.fill(scale_ub, 3.0)
            T.copy(a_ub, c_ub)
            for h_i in range(block_M):
                T.tile.mul(c_ub[h_i, :], c_ub[h_i, :], scale_ub[h_i])
            T.copy(c_ub, C[bx * block_M, by * block_N])

    return main


def buffer_region_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        ws: T.Tensor((m_num * n_num, block_M, block_N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            out_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, ws[cid, 0:block_M, :])
            T.copy(ws[cid, 0:block_M, :], out_ub)
            T.copy(out_ub, C[bx * block_M, by * block_N])

    return main


def gm_ub_gm_identity_t1(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, threads=1, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(a_ub, C[bx * block_M, by * block_N])

    return main


def parallel_elementwise_1d_t1(M, block_M, dtype="float16"):
    m_num = M // block_M

    @T.prim_func
    def main(A: T.Tensor((M,), dtype), C: T.Tensor((M,), dtype)):
        with T.Kernel(m_num, threads=1, is_npu=True) as (cid):
            bx = cid % m_num
            a_ub = T.alloc_shared((block_M,), dtype)
            c_ub = T.alloc_shared((block_M,), dtype)
            T.copy(A[bx * block_M], a_ub)
            for i in T.Parallel(block_M):
                c_ub[i] = a_ub[i] * 2.0 + 1.0
            T.copy(c_ub, C[bx * block_M])

    return main
