"""T.Parallel 2D offset variants.

The existing ``test_tilelang_ascend_language_parallel.py`` covers a single 2D
offset case (column offset on the read side, no output offset).  This file adds
the missing variants:

* output-side (store) offset:        ``c[i, j + h] = f(a[i, j])``
* asymmetric per-operand offsets:    ``c[i, j] = a[i, j] * b[i, j + h]``
* row-dimension (outer) offset:      ``c[i, j] = a[i + h, j]``

NOTE: these kernels target Ascend NPU and must be run on NPU hardware.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests."""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(0)
    yield


# ---------------------------------------------------------------------------
# Output-side (store) offset: c[i, j + h] = a[i, j] + 1.0
# Writes the result into the *second* half of the output tile; the first half
# is filled with zeros.  Tests that a loop-variable offset on the BufferStore
# index is lowered correctly.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def offset_store_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM
    half = block_N // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                T.tile.fill(c_ub, 0.0)
                for i, j in T.Parallel(rows, half):
                    c_ub[i, j + half] = a_ub[i, j] + 1.0
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_offset_store(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    half = block_N // 2
    func = offset_store_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)

    ref = torch.zeros_like(a)
    n_num = N // block_N
    for by in range(n_num):
        cs = by * block_N
        ref[:, cs + half : cs + block_N] = a[:, cs : cs + half] + 1.0
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Asymmetric per-operand offsets: c[i, j] = a[i, j] * b[i, j + h]
# The two operands are read at different column offsets within the same tile.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def offset_dual_operand_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM
    half = block_N // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            b_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * rows, by * block_N], b_ub)
                T.tile.fill(c_ub, 0.0)
                for i, j in T.Parallel(rows, half):
                    c_ub[i, j] = a_ub[i, j] * b_ub[i, j + half]
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_offset_dual_operand(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    half = block_N // 2
    func = offset_dual_operand_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a, b)

    ref = torch.zeros_like(a)
    n_num = N // block_N
    for by in range(n_num):
        cs = by * block_N
        ref[:, cs : cs + half] = a[:, cs : cs + half] * b[:, cs + half : cs + block_N]
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Row-dimension (outer) offset: c[i, j] = a[i + h, j]
# Offsets the OUTER loop variable.  The row offset operates within each VEC_NUM
# chunk (the block_M dimension is split across vector cores by `vid`).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def offset_row_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM
    half = rows // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                T.tile.fill(c_ub, 0.0)
                for i, j in T.Parallel(rows - half, block_N):
                    c_ub[i, j] = a_ub[i + half, j]
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_offset_row(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    rows = block_M // VEC_NUM
    half = rows // 2
    func = offset_row_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)

    # Row offset operates within each `rows`-sized chunk (one vector core's slice).
    ref = torch.zeros_like(a)
    a_chunks = a.reshape(-1, rows, N)
    ref_chunks = ref.reshape(-1, rows, N)
    ref_chunks[:, : rows - half, :] = a_chunks[:, half:, :]
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
