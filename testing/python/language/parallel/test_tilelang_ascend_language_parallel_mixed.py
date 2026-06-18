"""T.Parallel mixed with other features.

Covers scenario "与其他特性混开": composing a T.Parallel element-wise loop with
another primitive in the same kernel.  Each combination reuses a pattern already
proven elsewhere in the test suite:

* T.Parallel + T.tile.fill          (fill then accumulate)
* T.Parallel + cast-on-copy         (compute in fp32, store as fp16)
* T.Parallel + T.reduce_sum         (square then row-reduce)
* T.Parallel under a serial loop    (serial-outer / parallel-inner tiling)

The "T.Parallel + gemm" composition is covered by the passing
test_parallel_copy_gm_l1_l0c_gm in the copy test file; doing vector arithmetic
in the Cube scope (e.g. an L0C->GM scale epilogue) instead triggers the
"undefined Variable v_thread" codegen path, so it is not exercised here.

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
# T.Parallel + T.tile.fill: fill c with a constant, then accumulate a into it.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_fill_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                T.tile.fill(c_ub, 3.0)
                T.barrier_all()
                for i, j in T.Parallel(rows, block_N):
                    c_ub[i, j] = c_ub[i, j] + a_ub[i, j]
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_mixed_fill(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    func = parallel_fill_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, a + 3.0, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# T.Parallel + cast-on-copy: compute in fp32, then T.copy to an fp16 buffer
# (an implicit cast) before writing out.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_cast_kernel(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), "float"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), "float")
            f_ub = T.alloc_ub((rows, block_N), "float")
            h_ub = T.alloc_ub((rows, block_N), "float16")
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                for i, j in T.Parallel(rows, block_N):
                    f_ub[i, j] = a_ub[i, j] * 2.0
                T.barrier_all()
                T.copy(f_ub, h_ub)  # fp32 -> fp16 cast on copy
                T.barrier_all()
                T.copy(h_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_mixed_cast(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    func = parallel_cast_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, (a * 2.0).to(torch.float16), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# T.Parallel + T.reduce_sum: square each element with T.Parallel, then reduce
# along the row to produce per-row sums of squares.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_reduce_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            sq_ub = T.alloc_ub((rows, block_N), dtype)
            r_ub = T.alloc_ub((rows,), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * rows, by * block_N], a_ub)
                for i, j in T.Parallel(rows, block_N):
                    sq_ub[i, j] = a_ub[i, j] * a_ub[i, j]
                T.barrier_all()
                T.reduce_sum(sq_ub, r_ub, -1, [rows, block_N])
                T.barrier_all()
                T.copy(r_ub, B[bx * block_M + vid * rows])

    return main


def test_parallel_mixed_reduce(setup_random_seed):
    # N == block_N so each row's full sum lives within one column block.
    M, N, block_M, block_N = 1024, 64, 64, 64
    func = parallel_reduce_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, (a * a).sum(dim=1), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# T.Parallel nested inside a serial (range) outer loop: the most common tiling
# control-flow mix -- a serial outer loop drives a vectorized parallel inner
# pass.  Mirrors the proven row-split pattern (test_row_split_mul).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_serial_nest_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

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
                for i in range(rows):
                    for j in T.Parallel(block_N):
                        c_ub[i, j] = a_ub[i, j] * b_ub[i, j] + a_ub[i, j]
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_mixed_serial_nest(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    func = parallel_serial_nest_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a, b)
    torch.testing.assert_close(out, a * b + a, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
