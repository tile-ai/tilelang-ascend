# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import tilelang
import tilelang.language as T

import pytest
import testcommon as tc

M, N = 4, 4
n = 32
DATATYPE_CASES = ["float16", "float32"]
pytestmark = [pytest.mark.mode("Developer")]


def vec_add(M, N, n, dtype):

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

def vec_mul(M, N, n, dtype):

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

def vec_sub(M, N, n, dtype):

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

def vec_div(M, N, n, dtype):

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

@pytest.mark.op("vadd_extract")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_add_extract(dtype):
    func = vec_add(M, N, n, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [M, N]
    shape2 = [3]
    shape3 = [M, N]

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a + b[0]
    compiled_kernel(a, b, c)
    tc.assert_close(c.cpu(), ref_output.cpu(), rtol=1e-2, atol=1e-2)

@pytest.mark.op("vmul_extract")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_mul_extract(dtype):
    func = vec_mul(M, N, n, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [M, N]
    shape2 = [3]
    shape3 = [M, N]

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a * b[0]
    compiled_kernel(a, b, c)
    tc.assert_close(c.cpu(), ref_output.cpu(), rtol=1e-2, atol=1e-2)

@pytest.mark.op("vsub_extract")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_sub_extract(dtype):
    func = vec_sub(M, N, n, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [M, N]
    shape2 = [3]
    shape3 = [M, N]

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a - b[0]
    compiled_kernel(a, b, c)
    tc.assert_close(c.cpu(), ref_output.cpu(), rtol=1e-2, atol=1e-2)

@pytest.mark.op("vdiv_extract")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_div_extract(dtype):
    func = vec_div(M, N, n, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')

    shape = [M, N]
    shape2 = [3]
    shape3 = [M, N]

    a = generate_tensor(shape, dtype).npu()
    b = generate_tensor(shape2, dtype).npu()
    c = generate_tensor(shape3, dtype).npu()

    ref_output = a / b[0]
    compiled_kernel(a, b, c)
    tc.assert_close(c.cpu(), ref_output.cpu(), rtol=1e-2, atol=1e-2)