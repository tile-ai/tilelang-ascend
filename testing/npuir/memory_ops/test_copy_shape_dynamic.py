# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


dtype = "float16"


def copy_shape_1d_2d(M, N, block_M, block_N):
    @T.prim_func
    def copyShapeDynamic1D2D(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_ub((block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0)
                T.copy(A[bx, by:by + tile_size_N], A_BUF[0:tile_size_N])
                T.copy(A_BUF[0:tile_size_N], B[bx, by:by + tile_size_N])

    return copyShapeDynamic1D2D


def copy_shape_2d_3d(M, N, block_M, block_N):
    @T.prim_func
    def copyShapeDynamic2D3D(
        A: T.Tensor((1, M, N), dtype),
        B: T.Tensor((1, M, N), dtype),
        shape_M: T.int32,
        shape_N: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_ub((1, block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                t0 = shape_N - by
                tile_size_N = T.min(block_N, t0)
                T.copy(A[0, bx, by:by + tile_size_N], A_BUF[0, 0:tile_size_N])
                T.copy(A_BUF[0, 0:tile_size_N], B[0, bx, by:by + tile_size_N])

    return copyShapeDynamic2D3D


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
def test_copy_shape_1d_2d_dynamic():
    M = 8
    N = 8
    v1 = gen_tensor((M, N), dtype, kind="randn")
    v2 = gen_tensor((M, N), dtype, kind="zeros")
    v_ref = v1.clone()

    func = copy_shape_1d_2d(M, N, block_M=3, block_N=3)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(v1, v2, M, N)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
def test_copy_shape_2d_3d_dynamic():
    M = 8
    N = 8

    func = copy_shape_2d_3d(M, N, 3, 3)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((1, M, N), dtype, kind="randn")
    v2 = gen_tensor((1, M, N), dtype, kind="randn")
    v_ref = v1.clone()
    compiled_kernel(v1, v2, M, N)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
