# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import torch

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 256
N = 256

def vec_select(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            Cond_VEC = T.alloc_ub((block_M, block_N), "bool")
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], A_VEC)
            T.copy(B[bx * block_M, by * block_N], B_VEC)

            T.npuir_cmp(A_VEC, B_VEC, Cond_VEC, "ge")
            T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)

            T.copy(C_VEC, C[bx * block_M, by * block_N])

    return main

def vec_select_partial(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((N, ), dtype),
            B: T.Tensor((N, ), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            Cond_VEC = T.alloc_ub((block_M, block_N), "bool")
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            
            T.copy(A[by * block_N : (by + 1) * block_N], A_VEC[0, :])
            T.copy(B[by * block_N : (by + 1) * block_N], B_VEC[0, :])
            T.vbrc(A_VEC[0, :], A_VEC)
            T.vbrc(B_VEC[0, :], B_VEC)

            T.npuir_cmp(A_VEC, B_VEC, Cond_VEC, "ge")
            T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)

            T.copy(C_VEC, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main

def test_vec_select():
    os.environ['TILELANG_ASCEND_MODE'] = "Developer"

    # Test float16
    func_full = vec_select(M, N, 32, 64, dtype="float16")
    compiled_kernel_full = tilelang.compile(func_full, target="npuir")

    A_full = torch.randn((M, N), dtype=torch.float16).npu()
    B_full = torch.randn((M, N), dtype=torch.float16).npu()
    C_full = torch.empty((M, N), dtype=torch.float16).npu()

    compiled_kernel_full(A_full, B_full, C_full)
    ref_full = torch.where(A_full >= B_full, A_full, B_full)
    print('float16 kernel_full results:')
    print(C_full[:8, :8])
    print(ref_full[:8, :8])
    torch.testing.assert_close(C_full, ref_full, rtol=1e-3, atol=1e-3)

    # Test float32
    func_full_f32 = vec_select(M, N, 32, 64, dtype="float32")
    compiled_kernel_full_f32 = tilelang.compile(func_full_f32, target="npuir")

    A_full_f32 = torch.randn((M, N), dtype=torch.float32).npu()
    B_full_f32 = torch.randn((M, N), dtype=torch.float32).npu()
    C_full_f32 = torch.empty((M, N), dtype=torch.float32).npu()

    compiled_kernel_full_f32(A_full_f32, B_full_f32, C_full_f32)
    ref_full_f32 = torch.where(A_full_f32 >= B_full_f32, A_full_f32, B_full_f32)
    print('float32 kernel_full results:')
    torch.testing.assert_close(C_full_f32, ref_full_f32, rtol=1e-3, atol=1e-3)

    # Test int32
    func_full_i32 = vec_select(M, N, 32, 64, dtype="int32")
    compiled_kernel_full_i32 = tilelang.compile(func_full_i32, target="npuir")

    A_full_i32 = torch.randint(-100, 100, (M, N), dtype=torch.int32).npu()
    B_full_i32 = torch.randint(-100, 100, (M, N), dtype=torch.int32).npu()
    C_full_i32 = torch.empty((M, N), dtype=torch.int32).npu()

    compiled_kernel_full_i32(A_full_i32, B_full_i32, C_full_i32)
    ref_full_i32 = torch.where(A_full_i32 >= B_full_i32, A_full_i32, B_full_i32)
    print('int32 kernel_full results:')
    torch.testing.assert_close(C_full_i32, ref_full_i32)

    # Test int8
    func_full_i8 = vec_select(M, N, 32, 64, dtype="int8")
    compiled_kernel_full_i8 = tilelang.compile(func_full_i8, target="npuir")

    A_full_i8 = torch.randint(-10, 10, (M, N), dtype=torch.int8).npu()
    B_full_i8 = torch.randint(-10, 10, (M, N), dtype=torch.int8).npu()
    C_full_i8 = torch.empty((M, N), dtype=torch.int8).npu()

    compiled_kernel_full_i8(A_full_i8, B_full_i8, C_full_i8)
    ref_full_i8 = torch.where(A_full_i8 >= B_full_i8, A_full_i8, B_full_i8)
    print('int8 kernel_full results:')
    torch.testing.assert_close(C_full_i8, ref_full_i8)

    # test vec_select_partial
    func_partial = vec_select_partial(M, N, 32, 64)

    compiled_kernel_partial = tilelang.compile(func_partial, target="npuir")

    A_partial = torch.randn((N,), dtype=torch.float16).npu()
    B_partial = torch.randn((N,), dtype=torch.float16).npu()
    C_partial = torch.empty((M, N), dtype=torch.float16).npu()

    compiled_kernel_partial(A_partial, B_partial, C_partial)

    ref_partial = torch.where(A_partial >= B_partial,A_partial,B_partial)[None, :].expand(M, -1)

    print('kernel_partial results:')
    print(C_partial[:8, :8])
    print(ref_partial[:8, :8])
    torch.testing.assert_close(C_partial, ref_partial, rtol=1e-3, atol=1e-3)

    print("\033[92mSelect end2end check passed!\033[0m")

if __name__ == "__main__":
    test_vec_select()