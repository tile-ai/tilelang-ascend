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
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


# Test Case 1: Multiple buffer assignments in parallel
# for i, j in T.Parallel:
#     k[i,j] = a[i,j] + b[i,j]
#     k2[i,j] = c[i,j] + b[i,j]
@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def complex_dual_assignment(M, N, block_M=128, block_N=128, dtype="float"):
    """Two parallel assignments to different buffers"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        K1: T.Tensor((M, N), dtype),
        K2: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] + b[i, j]
                    k2[i, j] = c[i, j] + b[i, j]
                T.copy(k1, K1[bx * block_M + vid * (block_M // vec_num), by * block_N])
                T.copy(k2, K2[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_dual_assignment(A, B, C):
    """Reference implementation for dual assignment"""
    k1 = A + B
    k2 = C + B
    return k1, k2


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_dual_assignment(setup_random_seed, dtype):
    """Test two parallel buffer assignments"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_dual_assignment(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.randn((M, N), device="npu", dtype=torch.float32)
    K1 = torch.empty((M, N), device="npu", dtype=torch.float32)
    K2 = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out_k1, out_k2 = func(A, B, C, K1, K2)
    ref_k1, ref_k2 = ref_complex_dual_assignment(A, B, C)
    
    torch.testing.assert_close(out_k1, ref_k1, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_k2, ref_k2, rtol=1e-2, atol=1e-2)


# Test Case 2: Chained operations using intermediate results
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] + b[i,j]
#     k2[i,j] = k1[i,j] * c[i,j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_chained_operations(M, N, block_M=128, block_N=128, dtype="float"):
    """Chained operations: result of first op used in second op"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        K: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] + b[i, j]
                    k2[i, j] = k1[i, j] * c[i, j]
                T.copy(k2, K[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_chained_operations(A, B, C):
    """Reference implementation for chained operations"""
    k1 = A + B
    k2 = k1 * C
    return k2


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_chained_operations(setup_random_seed, dtype):
    """Test chained operations using intermediate buffer"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_chained_operations(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.randn((M, N), device="npu", dtype=torch.float32)
    K = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B, C, K)
    ref = ref_complex_chained_operations(A, B, C)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 3: Triple nested assignments
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] + b[i,j]
#     k2[i,j] = c[i,j] * d[i,j]
#     k3[i,j] = k1[i,j] + k2[i,j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_triple_assignments(M, N, block_M=128, block_N=128, dtype="float"):
    """Three assignments with chaining"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        K: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            d = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k3 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)
                T.copy(D[bx * block_M + vid * (block_M // vec_num), by * block_N], d)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] + b[i, j]
                    k2[i, j] = c[i, j] * d[i, j]
                    k3[i, j] = k1[i, j] + k2[i, j]
                T.copy(k3, K[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_triple_assignments(A, B, C, D):
    """Reference implementation for triple assignments"""
    k1 = A + B
    k2 = C * D
    k3 = k1 + k2
    return k3


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_triple_assignments(setup_random_seed, dtype):
    """Test three assignments with dependency"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_triple_assignments(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.randn((M, N), device="npu", dtype=torch.float32)
    D = torch.randn((M, N), device="npu", dtype=torch.float32)
    K = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B, C, D, K)
    ref = ref_complex_triple_assignments(A, B, C, D)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 4: Mixed operations with scalar and vector
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] + 1.0
#     k2[i,j] = k1[i,j] * 2.0
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_scalar_operations(M, N, block_M=128, block_N=128, dtype="float"):
    """Operations with scalar constants"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        K: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] + 1.0
                    k2[i, j] = k1[i, j] * 2.0
                T.copy(k2, K[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_scalar_operations(A):
    """Reference implementation for scalar operations"""
    k1 = A + 1.0
    k2 = k1 * 2.0
    return k2


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_scalar_operations(setup_random_seed, dtype):
    """Test chained operations with scalar constants"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_scalar_operations(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    K = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, K)
    ref = ref_complex_scalar_operations(A)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 5: Operations with row/column vectors
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] + b_row[i]
#     k2[i,j] = k1[i,j] + c_col[j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_broadcast_operations(M, N, block_M=128, block_N=128, dtype="float"):
    """Operations with broadcast from row and column vectors"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B_row: T.Tensor((M,), dtype),
        C_col: T.Tensor((N,), dtype),
        K: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b_row = T.alloc_ub((block_M // vec_num,), dtype)
            c_col = T.alloc_ub((block_N,), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B_row[bx * block_M + vid * (block_M // vec_num)], b_row)
                T.copy(C_col[by * block_N], c_col)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] + b_row[i]
                    k2[i, j] = k1[i, j] + c_col[j]
                T.copy(k2, K[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_broadcast_operations(A, B_row, C_col):
    """Reference implementation for broadcast operations"""
    k1 = A + B_row.unsqueeze(1)
    k2 = k1 + C_col.unsqueeze(0)
    return k2


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_broadcast_operations(setup_random_seed, dtype):
    """Test operations with broadcast from row/column vectors"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_broadcast_operations(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B_row = torch.randn((M,), device="npu", dtype=torch.float32)
    C_col = torch.randn((N,), device="npu", dtype=torch.float32)
    K = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B_row, C_col, K)
    ref = ref_complex_broadcast_operations(A, B_row, C_col)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 6: Complex expression with multiple intermediate buffers
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] * b[i,j]
#     k2[i,j] = c[i,j] / d[i,j]
#     k3[i,j] = k1[i,j] - k2[i,j]
#     k4[i,j] = k3[i,j] + e[i,j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_deep_expression(M, N, block_M=128, block_N=128, dtype="float"):
    """Complex expression with 4 intermediate buffers"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        E: T.Tensor((M, N), dtype),
        K: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            d = T.alloc_ub((block_M // vec_num, block_N), dtype)
            e = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k3 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k4 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)
                T.copy(D[bx * block_M + vid * (block_M // vec_num), by * block_N], d)
                T.copy(E[bx * block_M + vid * (block_M // vec_num), by * block_N], e)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] * b[i, j]
                    k2[i, j] = c[i, j] / d[i, j]
                    k3[i, j] = k1[i, j] - k2[i, j]
                    k4[i, j] = k3[i, j] + e[i, j]
                T.copy(k4, K[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_deep_expression(A, B, C, D, E):
    """Reference implementation for complex expression"""
    k1 = A * B
    k2 = C / D
    k3 = k1 - k2
    k4 = k3 + E
    return k4


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_deep_expression(setup_random_seed, dtype):
    """Test complex expression with 4 intermediate buffers"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_deep_expression(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.randn((M, N), device="npu", dtype=torch.float32)
    D = torch.randn((M, N), device="npu", dtype=torch.float32) + 0.1  # Avoid division by zero
    E = torch.randn((M, N), device="npu", dtype=torch.float32)
    K = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B, C, D, E, K)
    ref = ref_complex_deep_expression(A, B, C, D, E)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 7: Independent parallel paths with different operations
# for i, j in T.Parallel:
#     k1[i,j] = a[i,j] * b[i,j] + c[i,j]
#     k2[i,j] = d[i,j] / e[i,j] - f[i,j]
@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def complex_independent_paths(M, N, block_M=128, block_N=128, dtype="float"):
    """Two independent computational paths in parallel"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        E: T.Tensor((M, N), dtype),
        F: T.Tensor((M, N), dtype),
        K1: T.Tensor((M, N), dtype),
        K2: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            d = T.alloc_ub((block_M // vec_num, block_N), dtype)
            e = T.alloc_ub((block_M // vec_num, block_N), dtype)
            f = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k1 = T.alloc_ub((block_M // vec_num, block_N), dtype)
            k2 = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)
                T.copy(D[bx * block_M + vid * (block_M // vec_num), by * block_N], d)
                T.copy(E[bx * block_M + vid * (block_M // vec_num), by * block_N], e)
                T.copy(F[bx * block_M + vid * (block_M // vec_num), by * block_N], f)
                for i, j in T.Parallel(block_M // vec_num, block_N):
                    k1[i, j] = a[i, j] * b[i, j] + c[i, j]
                    k2[i, j] = d[i, j] / e[i, j] - f[i, j]
                T.copy(k1, K1[bx * block_M + vid * (block_M // vec_num), by * block_N])
                T.copy(k2, K2[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_independent_paths(A, B, C, D, E, F):
    """Reference implementation for independent paths"""
    k1 = A * B + C
    k2 = D / E - F
    return k1, k2


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_independent_paths(setup_random_seed, dtype):
    """Test two independent computational paths"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_independent_paths(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.randn((M, N), device="npu", dtype=torch.float32)
    D = torch.randn((M, N), device="npu", dtype=torch.float32)
    E = torch.randn((M, N), device="npu", dtype=torch.float32) + 0.1
    F = torch.randn((M, N), device="npu", dtype=torch.float32)
    K1 = torch.empty((M, N), device="npu", dtype=torch.float32)
    K2 = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out_k1, out_k2 = func(A, B, C, D, E, F, K1, K2)
    ref_k1, ref_k2 = ref_complex_independent_paths(A, B, C, D, E, F)
    
    torch.testing.assert_close(out_k1, ref_k1, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_k2, ref_k2, rtol=1e-2, atol=1e-2)


# Test Case 8: Nested computation need to use extra temp buffer
# for i, j in T.Parallel:
#     c[i, j] = a[i, j] * a[i, j] + a[i, j] - b[i, j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_nested_temp_buffer(M, N, block_M=128, block_N=128, dtype="float"):
    """Nested computation requiring extra temp buffer"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            temp = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)

                for i, j in T.Parallel(block_M // vec_num, block_N):
                    c[i, j] = a[i, j] * a[i, j] + a[i, j] - b[i, j]

                T.copy(c, C[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main

def ref_complex_nested_temp_buffer(A, B):
    """Reference implementation for nested computation with temp buffer"""
    C = A * A + A - B
    return C

@pytest.mark.parametrize("dtype", ["float"])
def test_complex_nested_temp_buffer(setup_random_seed, dtype):
    """Test nested computation requiring extra temp buffer"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_nested_temp_buffer(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32)
    C = torch.empty((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B)
    ref = ref_complex_nested_temp_buffer(A, B)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# Test Case 9: Nested Computation need to use multiple temp buffers
# for i, j in T.Parallel:
#     c[i, j] = a[i, j] * b[i, j] + a[i, j] / b[i, j] + c[i, j]
#     d[i, j] = c[i, j] - a[i, j] + b[i, j]
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def complex_nested_multiple_temp_buffers(M, N, block_M=128, block_N=128, dtype="float"):
    """Nested computation requiring multiple temp buffers"""
    m_num = M // block_M
    n_num = N // block_N
    vec_num = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub((block_M // vec_num, block_N), dtype)
            b = T.alloc_ub((block_M // vec_num, block_N), dtype)
            c = T.alloc_ub((block_M // vec_num, block_N), dtype)
            d = T.alloc_ub((block_M // vec_num, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * (block_M // vec_num), by * block_N], a)
                T.copy(B[bx * block_M + vid * (block_M // vec_num), by * block_N], b)
                T.copy(C[bx * block_M + vid * (block_M // vec_num), by * block_N], c)

                for i, j in T.Parallel(block_M // vec_num, block_N):
                    c[i, j] = a[i, j] * b[i, j] + a[i, j] / b[i, j] + c[i, j]
                    d[i, j] = c[i, j] - a[i, j] + b[i, j]

                T.copy(d, D[bx * block_M + vid * (block_M // vec_num), by * block_N])

    return main


def ref_complex_nested_multiple_temp_buffers(A, B, C):
    """Reference implementation for nested computation with multiple temp buffers"""
    c = A * B + A / B + C
    d = c - A + B
    return d


@pytest.mark.parametrize("dtype", ["float"])
def test_complex_nested_multiple_temp_buffers(setup_random_seed, dtype):
    """Test nested computation requiring multiple temp buffers"""
    torch.manual_seed(0)
    M, N = 1024, 1024
    block_M, block_N = 64, 64
    
    func = complex_nested_multiple_temp_buffers(M, N, block_M, block_N, dtype)
    A = torch.randn((M, N), device="npu", dtype=torch.float32)
    B = torch.randn((M, N), device="npu", dtype=torch.float32) + 0.1  # Avoid division by zero
    C = torch.randn((M, N), device="npu", dtype=torch.float32)

    torch.npu.synchronize()
    out = func(A, B, C)
    ref = ref_complex_nested_multiple_temp_buffers(A, B, C)
    
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
