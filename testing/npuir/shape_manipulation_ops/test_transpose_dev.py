# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""Transpose (npuir_transpose) in Ascend Developer mode vs torch.transpose."""
import os

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("transpose"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
TRANSPOSE_SHAPES = [(32, 32)]


def vec_transpose_dev(block_M, block_N, dtype="float16"):
    block_size = 1

    @T.prim_func
    def transpose_kernel_dev(
        A: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((block_N, block_M), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as _:
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            C_VEC = T.alloc_shared((block_N, block_M), dtype)
            T.copy(A, A_VEC)
            T.npuir_transpose(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_N, :block_M],
                (1, 0),
            )
            T.copy(C_VEC, C)

    return transpose_kernel_dev


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", TRANSPOSE_SHAPES)
def test_transpose_dev(dtype, M, N):
    A = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((N, M), dtype, kind="zeros")
    ref_C = torch.transpose(A.cpu(), 0, 1)

    func = vec_transpose_dev(M, N, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
