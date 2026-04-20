# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Gather (npuir_gather) in Ascend Developer mode vs torch.gather."""
import os

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gather"),
    pytest.mark.mode("Developer"),
]

GATHER_CASES = [(32, 32)]
DTYPES = ["float16"]
IndicesType = "int64"


def vec_gather_dev(block_M, block_N, dtype="float16"):
    block_size = 1

    @T.prim_func
    def gather_kernel_dev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), IndicesType),
        C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as _:
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            index_VEC = T.alloc_shared((block_M, block_N), IndicesType)
            C_VEC = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A, A_VEC)
            T.copy(B, index_VEC)
            T.npuir_gather(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_M, :block_N],
                index_VEC,
            )
            T.copy(C_VEC, C)

    return gather_kernel_dev


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", GATHER_CASES)
def test_gather_dev(dtype, M, N):
    compile_kernel = vec_gather_dev(M, N, dtype=dtype)
    kernel = tilelang.compile(compile_kernel, target="npuir")

    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((M, N), IndicesType, kind="randint", low=0, high=N - 1)
    c = gen_tensor((M, N), dtype, kind="zeros")

    ref_c = torch.gather(a.cpu(), dim=1, index=b.cpu())
    kernel(a, b, c)

    assert_close(c.cpu(), ref_c, dtype=dtype, rtol=1e-2, atol=1e-2)

