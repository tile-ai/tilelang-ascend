# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import sys
import os
import argparse
import torch

import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

M = 4
N = 1024
num = 4

@tilelang.jit(target="npuir")
def vec_for_add(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForAdd(
            Input: T.Tensor((M, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            Add_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, Add_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_add(src, Add_result, Add_result)

            T.copy(Add_result, Output)

    return vecForAdd

def test_for_add():
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForAdd = vec_for_add()
    vecForAdd(input, output)
    ref_output = torch.sum(input, dim=0, keepdim=True)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_sub(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForSub(
            Input: T.Tensor((M, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            Sub_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, Sub_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_sub(Sub_result, src, Sub_result)

            T.copy(Sub_result, Output)

    return vecForSub


def test_for_sub():
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForSub = vec_for_sub()
    vecForSub(input, output)
    ref_output = -torch.sum(input, dim=0, keepdim=True)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_mul(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForMul(
            Input: T.Tensor((M, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            Mul_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 1
            T.npuir_brc(value_zero, Mul_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_mul(Mul_result, src, Mul_result)

            T.copy(Mul_result, Output)

    return vecForMul


def test_for_mul():
    input = torch.randn([M, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForSub = vec_for_mul()
    vecForSub(input, output)
    ref_output = torch.prod(input, dim=0, keepdim=True)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_div(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForDiv(
            Input: T.Tensor((M, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            Div_result = T.alloc_shared([1, N], dtype=dtype)
            src = T.alloc_shared([1, N], dtype=dtype)

            value_zero = 1
            T.npuir_brc(value_zero, Div_result)
            for i in T.Pipelined(M):
                T.copy(Input[i, 0], src)
                T.npuir_div(Div_result, src, Div_result)

            T.copy(Div_result, Output)

    return vecForDiv


def test_for_div():
    input = (torch.rand([M, N], dtype=torch.float16) * 1.0 + 1.0).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForSub = vec_for_div()
    vecForSub(input, output)
    ref_output = 1 / torch.prod(input, dim=0, keepdim=True)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_exp(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForExp(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_exp(src, src)
            T.copy(src, Output)

    return vecForExp


def test_for_exp():
    input = torch.randn([1, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForExp = vec_for_exp()
    vecForExp(input, output)
    for _ in range(4):
        input.exp_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_ln(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForLn(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(2):
                T.npuir_ln(src, src)
            T.copy(src, Output)

    return vecForLn


def test_for_ln():
    r = torch.randn([1, N], dtype=torch.float16) + 3
    r = torch.clamp(r, min=1.0)
    input = torch.exp(r).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForLn = vec_for_ln()
    vecForLn(input, output)
    for _ in range(2):
        input.log_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_sqrt(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForSqrt(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_sqrt(src, src)
            T.copy(src, Output)

    return vecForSqrt


def test_for_sqrt():
    input = torch.exp(torch.randn([1, N], dtype=torch.float16)).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForSqrt = vec_for_sqrt()
    vecForSqrt(input, output)
    for _ in range(4):
        input.sqrt_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_rsqrt(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForRsqrt(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(4):
                T.npuir_rsqrt(src, src)
            T.copy(src, Output)

    return vecForRsqrt


def test_for_rsqrt():
    input = torch.exp(torch.randn([1, N], dtype=torch.float16)).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForRsqrt = vec_for_rsqrt()
    vecForRsqrt(input, output)
    for _ in range(4):
        input.rsqrt_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_rec(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForRec(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_rec(src, src)
            T.copy(src, Output)

    return vecForRec


def test_for_rec():
    input = torch.randn([1, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForRec = vec_for_rec()
    vecForRec(input, output)
    for _ in range(5):
        input.reciprocal_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


def npuir_not_float16(x):
    x_int16 = x.view(torch.int16)
    x_int16_not = torch.bitwise_not(x_int16)
    x_not = x_int16_not.view(torch.float16)
    return x_not


@tilelang.jit(target="npuir")
def vec_for_not(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForNot(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_not(src, src)
            T.copy(src, Output)

    return vecForNot

def test_for_not():
    input = torch.randn([1, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForNot = vec_for_not()
    vecForNot(input, output)
    for _ in range(5):
        input = npuir_not_float16(input)

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_reduce(dtype = "float32"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForReduce(
            Input: T.Tensor((num, M, N), dtype),
            Output: T.Tensor((M, 1), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            reduce_result = T.alloc_shared([M, 1], dtype=dtype)
            src = T.alloc_shared([M, N], dtype=dtype)

            value_zero = 0
            T.npuir_brc(value_zero, reduce_result)
            for i in T.Pipelined(num):
                T.copy(Input[i, :, :], src)
                T.npuir_reduce(src, reduce_result, 1, "sum", clear = False)

            T.copy(reduce_result, Output)

    return vecForReduce


def test_for_reduce():
    input = torch.randn([num, M, N], dtype=torch.float32).npu()
    output = torch.randn([M, 1], dtype=torch.float32).npu()
    vecForReduce = vec_for_reduce()
    vecForReduce(input, output)

    ref_output = torch.sum(input, dim=2, keepdim=True).sum(dim=0)

    print("output")
    print(output)

    print("ref_output")
    print(ref_output)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_abs(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForAbs(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_abs(src, src)
            T.copy(src, Output)

    return vecForAbs


def test_for_abs():
    input = torch.randn([1, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForAbs = vec_for_abs()
    vecForAbs(input, output)
    for _ in range(5):
        input.abs_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


@tilelang.jit(target="npuir")
def vec_for_relu(dtype = "float16"):
    BLOCK_SIZE = 1
    @T.prim_func
    def vecForRelu(
            Input: T.Tensor((1, N), dtype),
            Output: T.Tensor((1, N), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src = T.alloc_shared([1, N], dtype=dtype)
            T.copy(Input, src)
            for i in T.Pipelined(5):
                T.npuir_relu(src, src)
            T.copy(src, Output)

    return vecForRelu


def test_for_relu():
    input = torch.randn([1, N], dtype=torch.float16).npu()
    output = torch.randn([1, N], dtype=torch.float16).npu()
    vecForRelu = vec_for_relu()
    vecForRelu(input, output)
    for _ in range(5):
        input.relu_()

    print("output")
    print(output)

    print("ref_output")
    print(input)

    torch.testing.assert_close(output, input, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    print("Running in developer mode")
    print(">>>>>> Test add in T.pipelined <<<<<<")
    test_for_add()
    print()
    print(">>>>>> Test sub in T.pipelined <<<<<<")
    test_for_sub()
    print()
    print(">>>>>> Test mul in T.pipelined <<<<<<")
    test_for_mul()
    print()
    print(">>>>>> Test div in T.pipelined <<<<<<")
    test_for_div()
    print()
    print(">>>>>> Test exp in T.pipelined <<<<<<")
    test_for_exp()
    print()
    print(">>>>>> Test ln in T.pipelined <<<<<<")
    test_for_ln()
    print()
    print(">>>>>> Test sqrt in T.pipelined <<<<<<")
    test_for_sqrt()
    print()
    print(">>>>>> Test rsqrt in T.pipelined <<<<<<")
    test_for_rsqrt()
    print()
    print(">>>>>> Test rec in T.pipelined <<<<<<")
    test_for_rec()
    print()
    print(">>>>>> Test not in T.pipelined <<<<<<")
    test_for_not()
    print()
    print(">>>>>> Test reduce in T.pipelined <<<<<<")
    test_for_reduce()
    print()
    print(">>>>>> Test abs in T.pipelined <<<<<<")
    test_for_abs()
    print()
    print(">>>>>> Test relu in T.pipelined <<<<<<")
    test_for_relu()
    print()