import pytest
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


# Test Case: Direct GM write in T.Parallel with automatic UB to GM copy
# for i, j in T.Parallel:
#     C[cid // 8 * 128 + vid * 64 + i, cid % 8 * 128 + j] = a_ub[i, j] + b_ub[i, j]
# The pass should:
# 1. Create a temp UB buffer sized [64, 128] (the computation block)
# 2. Write the result to the temp UB
# 3. Copy from temp UB to GM using T.ascend_copy
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_auto_copy_kernel(M, N, block_M=128, block_N=128, dtype="float"):
    """Vector addition kernel demonstrating automatic UB to GM copy using T.Parallel.

    This kernel shows how the system automatically handles copying data from
    UB (Unified Buffer) to GM (Global Memory) when writing directly to GM tensors
    within T.Parallel loops, without requiring explicit T.copy() calls.

    Key features:
    - Explicit T.copy() for input: GM -> UB (required)
    - Computation and direct GM write using T.Parallel
    - Automatic copy for output: UB -> GM (handled by system)
    """
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # Allocate UB buffers for input data
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                # Step 1: Copy inputs from GM to UB (explicit copy required)
                # Data must be explicitly loaded from global memory to UB for computation
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # Step 2: Compute and write directly to GM using T.Parallel
                # The parallel loop will be vectorized by ascend_lower_parallel_to_vector pass
                # When writing to GM tensor C directly, the system automatically handles
                # copying the computed result from UB to GM without explicit T.copy()
                for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] + b_ub[i, j]

    return main


def ref_parallel_auto_copy(A, B):
    """Reference implementation for parallel auto copy"""
    return A + B


@pytest.mark.parametrize(
    "M,N,block_M,block_N",
    [
        (1024, 1024, 128, 128),
        (512, 512, 64, 64),
        (2048, 1024, 128, 128),
    ],
)
def test_parallel_auto_copy(setup_random_seed, M, N, block_M, block_N):
    """Test automatic UB to GM copy in T.Parallel loops"""
    func = parallel_auto_copy_kernel(M, N, block_M, block_N)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = ref_parallel_auto_copy(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# Test Case: Different data types
@pytest.mark.parametrize("dtype", ["float", "float16", "int32", "int16"])
def test_parallel_auto_copy_different_dtypes(setup_random_seed, dtype):
    """Test automatic UB to GM copy with different data types"""
    M, N = 1024, 1024
    block_M, block_N = 128, 128

    func = parallel_auto_copy_kernel(M, N, block_M, block_N, dtype)

    if dtype == "float":
        a = torch.randn(M, N).npu().to(torch.float32)
        b = torch.randn(M, N).npu().to(torch.float32)
    elif dtype == "float16":
        a = torch.randn(M, N).npu().to(torch.float16)
        b = torch.randn(M, N).npu().to(torch.float16)
    elif dtype == "int32":
        a = torch.randint(-100, 100, (M, N), dtype=torch.int32).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch.int32).npu()
    elif dtype == "int16":
        a = torch.randint(-100, 100, (M, N), dtype=torch.int16).npu()
        b = torch.randint(-100, 100, (M, N), dtype=torch.int16).npu()

    torch.npu.synchronize()
    c = func(a, b)

    if dtype in ["float", "float16"]:
        ref_c = a + b
    else:
        ref_c = a + b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# Test Case: Complex expression in T.Parallel
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_auto_copy_complex_kernel(M, N, block_M=128, block_N=128, dtype="float"):
    """Test complex expression: C = A * B + A - B"""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] * b_ub[i, j] + a_ub[i, j] - b_ub[i, j]

    return main


def ref_parallel_auto_copy_complex(A, B):
    """Reference implementation for complex expression"""
    return A * B + A - B


def test_parallel_auto_copy_complex(setup_random_seed):
    """Test complex expression with automatic UB to GM copy"""
    M, N = 1024, 1024
    block_M, block_N = 128, 128

    func = parallel_auto_copy_complex_kernel(M, N, block_M, block_N)

    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()
    c = func(a, b)

    ref_c = ref_parallel_auto_copy_complex(a, b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# Test Case: Unary operation in T.Parallel
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_auto_copy_unary_kernel(M, N, block_M=128, block_N=128, dtype="float"):
    """Test unary operation: C = exp(A)"""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = T.exp(a_ub[i, j])

    return main


def test_parallel_auto_copy_unary(setup_random_seed):
    """Test unary operation with automatic UB to GM copy"""
    M, N = 1024, 1024
    block_M, block_N = 128, 128

    func = parallel_auto_copy_unary_kernel(M, N, block_M, block_N)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    c = func(a)

    ref_c = torch.exp(a)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


# Test Case: Scalar operation in T.Parallel
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def parallel_auto_copy_scalar_kernel(M, N, block_M=128, block_N=128, dtype="float"):
    """Test scalar operation: C = A + 1.0"""
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = a_ub[i, j] + 1.0

    return main


def test_parallel_auto_copy_scalar(setup_random_seed):
    """Test scalar operation with automatic UB to GM copy"""
    M, N = 1024, 1024
    block_M, block_N = 128, 128

    func = parallel_auto_copy_scalar_kernel(M, N, block_M, block_N)

    a = torch.randn(M, N).npu()

    torch.npu.synchronize()
    c = func(a)

    ref_c = a + 1.0

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
