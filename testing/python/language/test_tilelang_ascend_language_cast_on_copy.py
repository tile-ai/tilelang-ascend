import pytest
import tilelang
import tilelang.language as T
import torch
import random

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
    def run_unary_kernel_test(kernel_func, input_generator, reference_func, M = 1024, N = 1024, block_M = 128, block_N = 256):
        """Common test pattern for unary kernel execution and verification"""
        func = kernel_func(M, N, block_M, block_N)
        input_tensor = input_generator(M, N)
        torch.npu.synchronize()
        output = func(input_tensor)
        ref_output = reference_func(input_tensor)
        torch.testing.assert_close(output, ref_output, rtol = 1e-2, atol = 1e-2)

class TestTileLangKernels:
    """Test suite for TileLang kernels"""

    @staticmethod
    @tilelang.jit(out_idx=[-1])
    def cast_on_copy_kernel(M, N, block_M, block_N, a_dtype, b_dtype):
        m_num = M // block_M
        n_num = N // block_N
        VEC_NUM = 2

        @T.prim_func
        def main(
                A: T.Tensor((M, N), a_dtype),
                B: T.Tensor((M, N), b_dtype),
        ):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num

                a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), a_dtype)
                b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), b_dtype)

                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                    T.barrier_all()
                    
                    T.copy(a_ub, b_ub)
                    T.barrier_all()

                    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

        return main

    def test_float_to_half(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float32", "float16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float32)
            return a
        def ref_func(a):
            b = a.to(torch.float16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_float_to_int32(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float32", "int32"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float32)
            return a
        def ref_func(a):
            b = torch.round(a).to(torch.int32)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )
    
    def test_float_to_int16(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float32", "int16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float32)
            return a
        def ref_func(a):
            b = torch.round(a).to(torch.int16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_half_to_float(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float16", "float32"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float16)
            return a
        def ref_func(a):
            b = a.to(torch.float)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_half_to_int32(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float16", "int32"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float16)
            return a
        def ref_func(a):
            b = torch.round(a).to(torch.int32)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_half_to_int16(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float16", "int16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float16)
            return a
        def ref_func(a):
            b = torch.round(a).to(torch.int16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_half_to_int8(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "float16", "int8"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.float16)
            return a
        def ref_func(a):
            b = torch.round(a).to(torch.int8)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_int32_to_float(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "int32", "float32"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.int32)
            return a
        def ref_func(a):
            b = a.to(torch.float32)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_int32_to_half(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "int32", "float16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.int32)
            return a
        def ref_func(a):
            b = a.to(torch.float16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_int16_to_float(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "int16", "float32"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.int16)
            return a
        def ref_func(a):
            b = a.to(torch.float32)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_int16_to_half(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "int16", "float16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.int16)
            return a
        def ref_func(a):
            b = a.to(torch.float16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )

    def test_int8_to_half(self, clear_cache, setup_random_seed):
        kernel_func = lambda M, N, block_M, block_N: self.cast_on_copy_kernel(
            M, N, block_M, block_N, "int8", "float16"
        )
        def input_gen(M, N):
            a = (torch.randn(M, N) * 10.0).npu().to(torch.int8)
            return a
        def ref_func(a):
            b = a.to(torch.float16)
            return b
        KernelTestHelper.run_unary_kernel_test(
            kernel_func = kernel_func,
            input_generator = input_gen,
            reference_func = ref_func,
        )
    
    # int8 and float32 cannot be directly casted to each other

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "8"])
