"""T.Parallel mixed with other features.

Covers scenario "与其他特性混开": composing a T.Parallel element-wise loop with
another primitive in the same kernel.  Each combination reuses a pattern already
proven elsewhere in the test suite:

* T.Parallel + T.tile.fill          (fill then accumulate)
* T.Parallel + cast-on-copy         (compute in fp32, store as fp16)
* T.Parallel + T.reduce_sum         (square then row-reduce)
* T.Parallel + gemm epilogue        (matmul then a parallel scale epilogue)
* T.Parallel + T.tile.select        (the vectorized replacement for if/else)
* T.Parallel with a data-dependent if/else (documented scalar-fallback path:
  it still compiles and produces correct results, just not vectorized)

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
# T.Parallel as a gemm epilogue: matmul into L0C, then a parallel scale epilogue
# while writing the accumulator back to GM.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_gemm_epilogue_kernel(M, N, K, block_M, block_N, K_L1, scale, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
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
                # Parallel scale epilogue while spilling the accumulator to GM.
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, by * block_N + j] = C_L0[i, j] * scale

    return main


def test_parallel_mixed_gemm_epilogue(setup_random_seed):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, K_L1 = 128, 256, 64
    scale = 2.0
    func = parallel_gemm_epilogue_kernel(M, N, K, block_M, block_N, K_L1, scale)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    torch.npu.synchronize()
    out = func(a, b)
    torch.testing.assert_close(out, (a @ b) * scale, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# T.Parallel + T.tile.select: select is the documented vectorized replacement
# for data-dependent if/else.  We first scale A with T.Parallel, then select
# between (A*2) and B according to a bit-packed mask.
# ---------------------------------------------------------------------------
def _bit_pack_mask_cpu(mask_bool: torch.Tensor) -> torch.Tensor:
    """Pack a bool mask into uint8 (8 bits per byte) on CPU."""
    M, N = mask_bool.shape
    mask_reshaped = mask_bool.view(M, N // 8, 8)
    mask_packed = torch.zeros((M, N // 8), dtype=torch.uint8)
    for i in range(8):
        mask_packed |= mask_reshaped[..., i].to(torch.uint8) << i
    return mask_packed


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_select_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM
    block_mask_width = block_N // 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        Mask: T.Tensor((M, N // 8), "uint8"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            b_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            mask_ub = T.alloc_ub((rows, block_mask_width), "uint8")
            offset_m = bx * block_M + vid * rows
            offset_n = by * block_N
            with T.Scope("V"):
                T.copy(A[offset_m : offset_m + rows, offset_n : offset_n + block_N], a_ub)
                T.copy(B[offset_m : offset_m + rows, offset_n : offset_n + block_N], b_ub)
                T.copy(Mask[offset_m : offset_m + rows, offset_n // 8 : offset_n // 8 + block_mask_width], mask_ub)
                # Parallel pre-step: scale A by 2.
                for i, j in T.Parallel(rows, block_N):
                    a_ub[i, j] = a_ub[i, j] * 2.0
                T.barrier_all()
                T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
                T.barrier_all()
                T.copy(c_ub, C[offset_m : offset_m + rows, offset_n : offset_n + block_N])

    return main


def test_parallel_mixed_select(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 256
    func = parallel_select_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).half().npu()
    b = torch.randn(M, N).half().npu()
    mask_bool = torch.randint(0, 2, (M, N)).bool()
    mask_packed = _bit_pack_mask_cpu(mask_bool).npu()
    torch.npu.synchronize()
    out = func(a, b, mask_packed)
    ref = torch.where(mask_bool.npu(), a * 2.0, b)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# T.Parallel with a data-dependent if/else.  Per the API spec this is NOT
# vectorized (it falls back to a scalar loop + scalar if), but it must still
# compile and produce correct results.  Equivalent to relu.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_if_else_kernel(M, N, block_M, block_N, dtype="float"):
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
                for i, j in T.Parallel(rows, block_N):
                    if a_ub[i, j] > 0:
                        c_ub[i, j] = a_ub[i, j]
                    else:
                        c_ub[i, j] = 0.0
                T.copy(c_ub, C[bx * block_M + vid * rows, by * block_N])

    return main


def test_parallel_mixed_if_else_fallback(setup_random_seed):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    func = parallel_if_else_kernel(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, torch.relu(a), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
