# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4, help="")
parser.add_argument("--N", type=int, default=4, help="")
parser.add_argument("--n", type=int, default=32, help="")


def vec_add(M, N, n):
    dtype = "float32"

    @T.prim_func
    def add(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((3,), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((3,), dtype)
            C_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            T.copy(B, B_ub)

            T.npuir_add(A_ub, B_ub[0], C_ub)

            T.copy(C_ub, C[i, :])

    return add

def vec_mul(M, N, n):
    dtype = "float32"

    @T.prim_func
    def mul(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((3,), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((3,), dtype)
            C_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            T.copy(B, B_ub)

            T.npuir_mul(A_ub, B_ub[0], C_ub)

            T.copy(C_ub, C[i, :])

    return mul

def vec_sub(M, N, n):
    dtype = "float32"

    @T.prim_func
    def sub(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((3,), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((3,), dtype)
            C_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            T.copy(B, B_ub)

            T.npuir_sub(A_ub, B_ub[0], C_ub)

            T.copy(C_ub, C[i, :])

    return sub

def vec_div(M, N, n):
    dtype = "float32"

    @T.prim_func
    def div(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((3,), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n, is_npu=True) as (cid, _):
            i = cid
            A_ub = T.alloc_shared((1, N), dtype)
            B_ub = T.alloc_shared((3,), dtype)
            C_ub = T.alloc_shared((1, N), dtype)

            T.copy(A[i, :], A_ub)
            T.copy(B, B_ub)

            T.npuir_div(A_ub, B_ub[0], C_ub)

            T.copy(C_ub, C[i, :])

    return div


def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def run_test_add(main_args):
    func = vec_add(
        main_args.M,
        main_args.N,
        main_args.n,

    )
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [main_args.M, main_args.N]
    shape2 = [3]
    shape3 = [main_args.M, main_args.N]

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float32"

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a + b[0]
    compiled_kernel(a, b, c)
    print("Actual Result:")
    print(c)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

def run_test_mul(main_args):
    func = vec_mul(
        main_args.M,
        main_args.N,
        main_args.n,

    )
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [main_args.M, main_args.N]
    shape2 = [3]
    shape3 = [main_args.M, main_args.N]

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float32"

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a * b[0]
    compiled_kernel(a, b, c)
    print("Actual Result:")
    print(c)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


def run_test_sub(main_args):
    func = vec_sub(
        main_args.M,
        main_args.N,
        main_args.n,

    )
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [main_args.M, main_args.N]
    shape2 = [3]
    shape3 = [main_args.M, main_args.N]

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float32"

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a - b[0]
    compiled_kernel(a, b, c)
    print("Actual Result:")
    print(c)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

def run_test_div(main_args):
    func = vec_div(
        main_args.M,
        main_args.N,
        main_args.n,

    )
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [main_args.M, main_args.N]
    shape2 = [3]
    shape3 = [main_args.M, main_args.N]

    torch.manual_seed(88888888)  # set the random seed for torch
    dtype = "float32"

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a / b[0]
    compiled_kernel(a, b, c)
    print("Actual Result:")
    print(c)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    args = parser.parse_args()
    run_test_add(args)
    run_test_mul(args)
    run_test_sub(args)
    run_test_div(args)