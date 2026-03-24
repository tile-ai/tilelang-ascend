# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

import testcommon as tc

pytestmark = [
    pytest.mark.op("workspace_copy"),
    pytest.mark.mode("Expert"),
]

DTYPE = "float16"
TEST_SHAPES = [
    (16, 16),
    (32, 64),
    (64, 32),
]


def workspace_copy_kernel(M, N, dtype=DTYPE):
    """GM -> UB -> workspace -> UB -> GM roundtrip via T.alloc_workspace."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            ub_in = T.alloc_shared((M, N), dtype, multi_buffer=2)
            ws = T.alloc_workspace((M, N), dtype, multi_buffer=2)
            ub_out = T.alloc_fragment((M, N), dtype, multi_buffer=2)

            T.copy(A, ub_in)       # GM -> UB
            T.copy(ub_in, ws)      # UB -> workspace (memref.copy)
            T.copy(ws, ub_out)     # workspace -> UB (memref.copy)
            T.copy(ub_out, B)      # UB -> GM

    return kernel


@pytest.mark.parametrize("M, N", TEST_SHAPES)
def test_workspace_copy_exp(M, N):
    func = workspace_copy_kernel(M, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    src = tc.gen_tensor((M, N), DTYPE, kind="randn")
    dst = tc.gen_tensor((M, N), DTYPE, kind="zeros")
    ref = src.clone()

    compiled_kernel(src, dst)
    tc.assert_close(dst.cpu(), ref.cpu(), dtype=DTYPE)
