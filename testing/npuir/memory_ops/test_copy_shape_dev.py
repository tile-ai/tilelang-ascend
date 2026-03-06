# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import ascend_mode, assert_close, gen_tensor


dtype = "float16"


def copy_shape_1d_2d(M, N, block_M, block_N):
    @T.prim_func
    def copyShapeDev1D2D(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                T.copy(A[bx, by:by + block_N], A_BUF)
                T.copy(A_BUF, B[bx, by:by + block_N])

    return copyShapeDev1D2D


def copy_shape_2d_3d(M, N, block_M, block_N):
    @T.prim_func
    def copyShapeDev2D3D(
        A: T.Tensor((1, M, N), dtype),
        B: T.Tensor((1, M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((1, block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                T.copy(A[0, bx, by:by + block_N], A_BUF)
                T.copy(A_BUF, B[0, bx, by:by + block_N])

    return copyShapeDev2D3D


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_shape_1d_2d_dev():
    M = 256
    N = 1024
    v1 = gen_tensor((M, N), dtype, kind="randn")
    v2 = gen_tensor((M, N), dtype, kind="zeros")
    v_ref = v1.clone()

    with ascend_mode("Developer"):
        func = copy_shape_1d_2d(M, N, block_M=32, block_N=32)
        compiled_kernel = tilelang.compile(func, target="npuir")
        compiled_kernel(v1, v2, M, N)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_shape_2d_3d_dev():
    M = 256
    N = 1024

    with ascend_mode("Developer"):
        func = copy_shape_2d_3d(M, N, 32, 32)
        compiled_kernel = tilelang.compile(func, target="npuir")

        v1 = gen_tensor((1, M, N), dtype, kind="randn")
        v2 = gen_tensor((1, M, N), dtype, kind="randn")
        v_ref = v1.clone()
        compiled_kernel(v1, v2)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
