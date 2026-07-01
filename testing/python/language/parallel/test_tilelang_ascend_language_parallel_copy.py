"""T.Parallel as a copy primitive across the memory hierarchy.

The existing ``test_tilelang_ascend_language_parallel_auto_copy.py`` exercises the
UB->GM auto-copy path and a matmul, but the individual copy directions are
tangled inside the matmul body.  This file isolates each direction the user
cares about into clear, minimal kernels:

* GM -> UB   (parallel load,  vector scope)
* UB -> GM   (parallel store, vector scope)
* GM -> L1   (parallel load,  cube scope)
* L0C -> GM  (parallel store of the matmul accumulator, cube scope)

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
# GM -> UB -> GM identity, both directions expressed with T.Parallel.
# The first loop is a parallel GM->UB load, the second a parallel UB->GM store.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gm_ub_gm_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                # GM -> UB
                for i, j in T.Parallel(rows, block_N):
                    a_ub[i, j] = A[bx * block_M + vid * rows + i, by * block_N + j]
                # UB -> GM
                for i, j in T.Parallel(rows, block_N):
                    C[bx * block_M + vid * rows + i, by * block_N + j] = a_ub[i, j]

    return main


@pytest.mark.parametrize("dtype", ["float", "float16", "int32"])
def test_parallel_copy_gm_ub_gm(setup_random_seed, dtype):
    M, N, block_M, block_N = 1024, 1024, 128, 128
    func = gm_ub_gm_kernel(M, N, block_M, block_N, dtype)
    if dtype == "int32":
        a = torch.randint(-100, 100, (M, N), dtype=torch.int32).npu()
    elif dtype == "float16":
        a = torch.randn(M, N).npu().to(torch.float16)
    else:
        a = torch.randn(M, N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 1D GM -> UB -> GM identity (parallel load + parallel store).
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gm_ub_gm_1d_kernel(N, block_N, dtype="float"):
    n_num = N // block_N
    chunk = block_N // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            by = cid % n_num
            a_ub = T.alloc_ub((chunk,), dtype)
            with T.Scope("V"):
                for j in T.Parallel(chunk):
                    a_ub[j] = A[vid * chunk + by * block_N + j]
                for j in T.Parallel(chunk):
                    C[vid * chunk + by * block_N + j] = a_ub[j]

    return main


def test_parallel_copy_gm_ub_gm_1d(setup_random_seed):
    N, block_N = 1024, 128
    func = gm_ub_gm_1d_kernel(N, block_N)
    a = torch.randn(N).npu()
    torch.npu.synchronize()
    out = func(a)
    torch.testing.assert_close(out, a, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# GM -> L1 (parallel load, cube scope) + gemm + L0C -> GM (parallel store).
# A and B are loaded into L1 with T.Parallel, the matmul accumulates into L0C,
# and the accumulator is written back to GM with T.Parallel.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gm_l1_l0c_gm_kernel(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
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
                    # GM -> L1 (parallel load), each operand with its own shape.
                    for i, j in T.Parallel(block_M, K_L1):
                        A_L1[i, j] = A[bx * block_M + i, k * K_L1 + j]
                    for i, j in T.Parallel(K_L1, block_N):
                        B_L1[i, j] = B[k * K_L1 + i, by * block_N + j]
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                # L0C -> GM (parallel store of the accumulator).
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, by * block_N + j] = C_L0[i, j]

    return main


def test_parallel_copy_gm_l1_l0c_gm(setup_random_seed):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, K_L1 = 128, 256, 64
    func = gm_l1_l0c_gm_kernel(M, N, K, block_M, block_N, K_L1)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    torch.npu.synchronize()
    out = func(a, b)
    torch.testing.assert_close(out, a @ b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
