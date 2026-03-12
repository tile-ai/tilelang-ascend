# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import argparse
import torch

import tilelang
import tilelang.language as T

import pytest
import testcommon as tc

M, N = 4, 4
a, b = 1, 16
DATATYPE_CASES = ["float16", "float32"]
pytestmark = [pytest.mark.mode("Developer")]


def vec_insert(M, N, a, b, dtype):

    @T.prim_func
    def insert(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((a, b), dtype),
    ):
        with T.Kernel(M, is_npu=True) as (cid, _):
            A_ub = T.alloc_shared((4, 4), dtype)
            B_ub = T.alloc_shared((a, b), dtype)
            T.copy(B, B_ub)

            for i in T.serial(4):
                for j in T.serial(4):
                    A_ub[i, j] = B_ub[0, i * 4 + j]

            T.copy(A_ub, A)

    return insert


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


@pytest.mark.op("insert_dev")
@pytest.mark.parametrize("dtype", DATATYPE_CASES)
def test_insert(dtype):
    func = vec_insert(M, N, a, b, dtype)
    compiled_kernel = tilelang.compile(func, target='npuir')
    shape = [M, N]
    shape2 = [a, b]
    output = generate_tensor(shape, dtype).npu()
    intput = generate_tensor(shape2, dtype).npu()

    ref_output = intput.reshape(M, N)
    compiled_kernel(output, intput)
    tc.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
