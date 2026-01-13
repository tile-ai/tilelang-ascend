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


if __name__ == "__main__":
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
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
    