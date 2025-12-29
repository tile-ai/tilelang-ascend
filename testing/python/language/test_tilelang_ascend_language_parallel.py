
import pytest
import tilelang
import tilelang.language as T
import torch
import random

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@pytest.fixture(scope="session")
def clear_cache():
    """Clear tilelang cache before tests"""
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(0)
    yield


class KernelTestHelper:
    """Helper class for common kernel testing operations"""
  
    @staticmethod
    def run_binary_kernel_test(kernel_func, input_generator, reference_func, M=1024, N=1024, block_M=128, block_N=128):
        """Common test pattern for binary kernel execution and verification"""
        func = kernel_func(M, N, block_M, block_N)
        inputs = input_generator(M, N)
        torch.npu.synchronize()
        output = func(*inputs)
        ref_output = reference_func(*inputs)
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
  
    @staticmethod
    def run_unary_kernel_test(kernel_func, input_generator, reference_func, M=1024, N=1024, block_M=128, block_N=128, equal_nan=False):
        """Common test pattern for unary kernel execution and verification"""
        func = kernel_func(M, N, block_M, block_N)
        input_tensor = input_generator(M, N)
        torch.npu.synchronize()
        output = func(input_tensor)
        ref_output = reference_func(input_tensor)
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2, equal_nan=equal_nan)
  
    @staticmethod
    def run_1d_binary_kernel_test(kernel_func, input_generator, reference_func, N=1024, block_N=128):
        """Common test pattern for 1D binary kernel execution and verification"""
        func = kernel_func(N, block_N)
        inputs = input_generator(N)
        torch.npu.synchronize()
        output = func(*inputs)
        ref_output = reference_func(*inputs)
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)


class TestTileLangKernels:
    """Test suite for TileLang kernels"""
  
    # Binary operation kernels (2D)
    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def binary_op_kernel_float(M, N, block_M, block_N, op_func, dtype="float"):
        """Generic binary operation kernel for float types"""
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
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                    
                    for (local_x, local_y) in T.Parallel(block_M // VEC_NUM, block_N):
                        c_ub[local_x, local_y] = op_func(a_ub[local_x, local_y], b_ub[local_x, local_y])
                    

                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main
  
    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def binary_op_kernel_int(M, N, block_M, block_N, op_func, dtype="int16"):
        """Generic binary operation kernel for int types"""
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
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                    
                    for (local_x, local_y) in T.Parallel(block_M // VEC_NUM, block_N):
                        c_ub[local_x, local_y] = op_func(a_ub[local_x, local_y], b_ub[local_x, local_y])
                    

                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main
  
    # Unary operation kernels (2D)
    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def unary_op_kernel_float(M, N, block_M, block_N, op_func, dtype="float"):
        """Generic unary operation kernel for float types"""
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                    
                    for (local_x, local_y) in T.Parallel(block_M // VEC_NUM, block_N):
                        b_ub[local_x, local_y] = op_func(a_ub[local_x, local_y])
                    

                    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main
  
    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def unary_op_kernel_int(M, N, block_M, block_N, op_func, dtype="int16"):
        """Generic unary operation kernel for int types"""
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

                    
                    for (local_x, local_y) in T.Parallel(block_M // VEC_NUM, block_N):
                        b_ub[local_x, local_y] = op_func(a_ub[local_x, local_y])
                    

                    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main
  
    # 1D binary operation kernel
    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def binary_op_kernel_1d(N, block_N, op_func, dtype="float"):
        """Generic binary operation kernel for 1D tensors"""
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
                A: T.Tensor((N,), dtype),
                B: T.Tensor((N,), dtype),
                C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(n_num, is_npu=True) as (cid, vid):
                by = cid % n_num

                a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
                b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
                c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
                with T.Scope("V"):
                    T.copy(A[vid * block_N // VEC_NUM + by * block_N], a_ub)
                    T.copy(B[vid * block_N // VEC_NUM + by * block_N], b_ub)

                    
                    for local_y in T.Parallel(block_N // VEC_NUM):
                        c_ub[local_y] = op_func(a_ub[local_y], b_ub[local_y])
                    

                    T.copy(c_ub, C[vid * block_N // VEC_NUM + by * block_N])

        return main

    # Basic binary operation tests
    def test_add_operation(self, clear_cache, setup_random_seed):
        """Test addition operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a + b
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a + b
        )
  
    def test_sub_operation(self, clear_cache, setup_random_seed):
        """Test subtraction operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a - b
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a - b
        )
  
    def test_mul_operation(self, clear_cache, setup_random_seed):
        """Test multiplication operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a * b
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a * b
        )
  
    def test_div_operation(self, clear_cache, setup_random_seed):
        """Test division operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a / b
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a / b
        )
  
    def test_and_operation(self, clear_cache, setup_random_seed):
        """Test AND operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_int(
            M, N, block_M, block_N, lambda a, b: a & b
        )
      
        def input_gen(M, N):
            a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
            b = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a & b
        )
  
    def test_or_operation(self, clear_cache, setup_random_seed):
        """Test OR operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_int(
            M, N, block_M, block_N, lambda a, b: a | b
        )
      
        def input_gen(M, N):
            a = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
            b = torch.randint(0, 10, (M, N), dtype=torch.int16).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a | b
        )

    def test_min_operation(self, clear_cache, setup_random_seed):
        """Test element-wise minimum operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: T.min(a, b)
        )

        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b

        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: torch.min(a, b)
        )

    def test_max_operation(self, clear_cache, setup_random_seed):
        """Test element-wise maximum operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: T.max(a, b)
        )

        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b

        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: torch.max(a, b)
        )

    # Unary operation tests
    def test_abs_operation(self, clear_cache, setup_random_seed):
        """Test absolute value operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.abs(a)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.randn(M, N).npu(),
            reference_func=lambda a: torch.abs(a)
        )
  
    def test_exp_operation(self, clear_cache, setup_random_seed):
        """Test exponential operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.exp(a)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.randn(M, N).npu(),
            reference_func=lambda a: torch.exp(a)
        )
  
    def test_log_operation(self, clear_cache, setup_random_seed):
        """Test logarithm operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.log(a)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.abs(torch.randn(M, N).npu()),
            reference_func=lambda a: torch.log(a)
        )
  
    def test_sqrt_operation(self, clear_cache, setup_random_seed):
        """Test square root operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.sqrt(a)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.rand(M, N).npu(),
            reference_func=lambda a: torch.sqrt(a)
        )
  
    def test_rsqrt_operation(self, clear_cache, setup_random_seed):
        """Test reciprocal square root operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.rsqrt(a)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.randn(M, N).npu(),
            reference_func=lambda a: torch.rsqrt(a),
            equal_nan=True
        )
  
    def test_relu_operation(self, clear_cache, setup_random_seed):
        """Test ReLU operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_float(
            M, N, block_M, block_N, lambda a: T.max(a, 0)
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.randn(M, N).npu(),
            reference_func=lambda a: torch.relu(a)
        )
  
    def test_not_operation(self, clear_cache, setup_random_seed):
        """Test bitwise NOT operation kernel"""
        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_int(
            M, N, block_M, block_N, lambda a: ~a
        )
      
        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=lambda M, N: torch.randint(0, 10, (M, N), dtype=torch.int16).npu(),
            reference_func=lambda a: ~a
        )
  
    def test_shiftleft_operation(self, clear_cache, setup_random_seed):
        """Test bitwise left shift kernel"""
        scalar_value = random.randint(1, 16)

        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_int(
            M, N, block_M, block_N,
            op_func=lambda a: a << scalar_value,
            dtype="int32"
        )

        def input_generator(M, N):
            return torch.randint(1, 101, (M, N), dtype=torch.int32).npu()

        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_generator,
            reference_func=lambda a: a * (2 ** scalar_value),
        )

    def test_shiftright_operation(self, clear_cache, setup_random_seed):
        """Test bitwise right shift kernel"""
        scalar_value = random.randint(1, 32)

        kernel_func = lambda M, N, block_M, block_N: self.unary_op_kernel_int(
            M, N, block_M, block_N,
            op_func=lambda a: a >> scalar_value,
            dtype="int32"
        )

        def input_generator(M, N):
            return torch.randint(1, 101, (M, N), dtype=torch.int32).npu()

        KernelTestHelper.run_unary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_generator,
            reference_func=lambda a: a // (2 ** scalar_value),
        )

    # Compound operation tests
    def test_fused_mul_add_operation(self, clear_cache, setup_random_seed):
        """Test fused multiply-add operation: a * b + a"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a * b + a
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a * b + a
        )
  
    def test_fused_add_mul_operation(self, clear_cache, setup_random_seed):
        """Test fused add-multiply operation: a * (b + a)"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a * (b + a)
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a * (b + a)
        )
  
    # 1D operation tests
    def test_1d_add_operation(self, clear_cache, setup_random_seed):
        """Test 1D addition operation kernel"""
        kernel_func = lambda N, block_N: self.binary_op_kernel_1d(
            N, block_N, lambda a, b: a + b
        )
      
        def input_gen(N):
            a = torch.randn(N).npu()
            b = torch.randn(N).npu()
            return a, b
      
        KernelTestHelper.run_1d_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a + b
        )
  
    # Scalar operation tests
    def test_add_scalar_operation(self, clear_cache, setup_random_seed):
        """Test vector + scalar operation"""
        kernel_func = lambda M, N, block_M, block_N: self.binary_op_kernel_float(
            M, N, block_M, block_N, lambda a, b: a + 1
        )
      
        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M, N).npu()  # Not used but needed for compatibility
            return a, b
      
        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a + 1
        )

    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def row_split_mul_kernel(M, N, block_M, block_N, dtype="float"):
        "Test row split kernel"
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype),
                 B: T.Tensor((M, N), dtype),
                 C: T.Tensor((M, N), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                    
                    for i in range(block_M // VEC_NUM):
                        for j in T.Parallel(block_N):
                            c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
                    

                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    def test_row_split_mul(self, clear_cache, setup_random_seed):

        def kernel_func(M, N, block_M, block_N):
            return self.row_split_mul_kernel(M, N, block_M, block_N)

        def input_gen(M, N):
            a = torch.rand(M, N).npu()
            b = torch.rand(M, N).npu()
            return a, b

        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=lambda a, b: a * b
        )

    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def column_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N, dtype="float"):
        "Test buffer scaler mul kernel"
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype),
                 B: T.Tensor((M,), dtype),
                 C: T.Tensor((M, N), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM,), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[bx * block_M + vid * block_M // VEC_NUM], b_ub)
                    
                    for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
                        c_ub[i, j] = a_ub[i, j] * b_ub[i]

                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    def test_column_parallel_buffer_scalar_mul(self, clear_cache, setup_random_seed):

        def kernel_func(M, N, block_M, block_N):
            return self.column_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N)

        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(M).npu()
            return a, b

        def reference_func(a, b):
            return a * b.unsqueeze(1)

        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=reference_func
        )

    @staticmethod
    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def row_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N, dtype="float"):
        "Test buffer scaler mul kernel"
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype),
                 B: T.Tensor((N,), dtype),
                 C: T.Tensor((M, N), dtype)):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_ub((block_N,), dtype)
                c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.copy(B[by * block_N], b_ub)

                    for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
                        c_ub[i, j] = a_ub[i, j] * b_ub[j]


                    T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    def test_row_parallel_buffer_scalar_mul(self, clear_cache, setup_random_seed):

        def kernel_func(M, N, block_M, block_N):
            return self.row_parallel_buffer_scalar_mul_kernel(M, N, block_M, block_N)

        def input_gen(M, N):
            a = torch.randn(M, N).npu()
            b = torch.randn(N).npu()
            return a, b

        def reference_func(a, b):
            return a * b.unsqueeze(0)

        KernelTestHelper.run_binary_kernel_test(
            kernel_func=kernel_func,
            input_generator=input_gen,
            reference_func=reference_func
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
