# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Developer-mode npuir debug print on shared memory does not affect numeric results."""
import os

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("print"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
PRINT_SHAPES = [(32, 32)]


def vec_add_with_print_dev(M, N, dtype="float16"):
    block_size = 1

    @T.prim_func
    def add_print_kernel_dev(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as _:
            A_VEC = T.alloc_shared((M, N), dtype)
            B_VEC = T.alloc_shared((M, N), dtype)
            C_VEC = T.alloc_shared((M, N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.print(C_VEC[:4, :4], msg="tile_dbg")
            T.copy(C_VEC, C)

    return add_print_kernel_dev


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", PRINT_SHAPES)
def test_print_dev_numeric(dtype, M, N):
    a = gen_tensor((M, N), dtype, kind="randn")
    b = gen_tensor((M, N), dtype, kind="randn")
    c = gen_tensor((M, N), dtype, kind="zeros")
    expected = a.cpu() + b.cpu()

    func = vec_add_with_print_dev(M, N, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(a, b, c)

    assert_close(c.cpu(), expected, dtype=dtype, rtol=1e-2, atol=1e-2)
