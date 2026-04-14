# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


pytestmark = [
    pytest.mark.op("copy_general"),
    pytest.mark.mode("Developer"),
]

DTYPE = "float16"


@pytest.fixture(autouse=True)
def _force_expert_backend(monkeypatch):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", "expert")


def _compile(func):
    return tilelang.compile(func, target="npuir")


# src.shape = [64, 128], src.range = [:, :], src.slice = [64, 128]
# dst.shape = [64, 128], dst.range = [:, :], dst.slice = [64, 128]
# Tests exact-shape contiguous copy.
@T.prim_func
def same_shape_2d_kernel(A: T.Tensor((64, 128), DTYPE), B: T.Tensor((64, 128), DTYPE)):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((64, 128), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [64, 128], src.range = [:, :], src.slice = [64, 128]
# dst.shape = [1, 64, 128], dst.range = [:, :, :], dst.slice = [1, 64, 128]
# Tests whole-buffer copy into a destination with an extra leading singleton dim.
@T.prim_func
def add_leading_one_dim_kernel(
    A: T.Tensor((64, 128), DTYPE), B: T.Tensor((1, 64, 128), DTYPE)
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((1, 64, 128), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [1, 64, 1, 128], src.range = [:, :, :, :], src.slice = [1, 64, 1, 128]
# dst.shape = [64, 128], dst.range = [:, :], dst.slice = [64, 128]
# Tests whole-buffer copy that removes multiple singleton dims on the source side.
@T.prim_func
def remove_multiple_one_dims_kernel(
    A: T.Tensor((1, 64, 1, 128), DTYPE), B: T.Tensor((64, 128), DTYPE)
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((64, 128), DTYPE)
        T.copy(A, UB)
        T.copy(UB, B)


# src.shape = [4, 16, 16, 32]
# src.range = [i, idx1:idx1+8, j, idx2:idx2+8], src.slice = [8, 8]
# dst.shape = [4, 16, 16, 32]
# dst.range = [i, idx1:idx1+8, j, idx2:idx2+8], dst.slice = [8, 8]
# Tests dynamic range.min with equal logical slice shapes.
@T.prim_func
def dynamic_min_offsets_kernel(
    A: T.Tensor((4, 16, 16, 32), DTYPE),
    B: T.Tensor((4, 16, 16, 32), DTYPE),
    i: T.int32,
    idx1: T.int32,
    j: T.int32,
    idx2: T.int32,
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((8, 8), DTYPE)
        T.copy(A[i, idx1 : idx1 + 8, j, idx2 : idx2 + 8], UB)
        T.copy(UB, B[i, idx1 : idx1 + 8, j, idx2 : idx2 + 8])


# src.shape = [4, 32], src.range = [row, start:start+tail], src.slice = [tail]
# dst.shape = [4, 32], dst.range = [row, start:start+tail], dst.slice = [tail]
# Tests dynamic range.extent via tail = T.min(32 - start, 16).
@T.prim_func
def dynamic_extent_tail_kernel(
    A: T.Tensor((4, 32), DTYPE),
    B: T.Tensor((4, 32), DTYPE),
    row: T.int32,
    start: T.int32,
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((16,), DTYPE)
        tail = T.min(32 - start, 16)
        T.copy(A[row, start : start + tail], UB[0:tail])
        T.copy(UB[0:tail], B[row, start : start + tail])


dynamic_gm_active_static_s = T.symbolic("dynamic_gm_active_static_s")
dynamic_backbone_tail = T.symbolic("dynamic_backbone_tail")
dynamic_gm_active_dynamic_s = T.symbolic("dynamic_gm_active_dynamic_s")
dynamic_2d_tile_m = T.symbolic("dynamic_2d_tile_m")
dynamic_2d_tile_n = T.symbolic("dynamic_2d_tile_n")


# src.shape = [16, S, 16, 32]
# src.range = [i, j, idx1:idx1+8, idx2:idx2+16], src.slice = [8, 16]
# dst.shape = [8, 16], dst.range = [:, :], dst.slice = [8, 16]
# Tests the supported sparse-attention-style case where the dynamic GM dim
# is indexed away before aligned-layout projection.
@T.prim_func
def dynamic_gm_active_stride_static_kernel(
    A: T.Tensor((16, dynamic_gm_active_static_s, 16, 32), DTYPE),
    B: T.Tensor((8, 16), DTYPE),
    i: T.int32,
    j: T.int32,
    idx1: T.int32,
    idx2: T.int32,
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((8, 16), DTYPE)
        T.copy(A[i, j, idx1 : idx1 + 8, idx2 : idx2 + 16], UB)
        T.copy(UB, B)


# src.shape = [1, 8, tail], src.range = [:, :, :], src.slice = [1, 8, tail]
# dst.shape = [1, 8, 16], dst.range = [:, :, 0:tail], dst.slice = [1, 8, tail]
# Tests same-rank dynamic GM copy where the projected row stride stays dynamic
# while the logical slice shapes remain equal on both sides.
@T.prim_func
def dynamic_backbone_tail_kernel(
    A: T.Tensor((1, 8, dynamic_backbone_tail), DTYPE),
    B: T.Tensor((1, 8, 16), DTYPE),
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((1, 8, 16), DTYPE)
        T.copy(A, UB[:, :, 0:dynamic_backbone_tail])
        T.copy(UB[:, :, 0:dynamic_backbone_tail], B[:, :, 0:dynamic_backbone_tail])


# src.shape = [4, S, 16, 32]
# src.range = [i:i+1, idx1:idx1+8, j:j+1, idx2:idx2+tail], src.slice = [1, 8, 1, tail]
# dst.shape = [1, 8, 1, 16], dst.range = [:, :, :, 0:tail], dst.slice = [1, 8, 1, tail]
# Tests direct support for dynamic projected strides on contiguous GM layouts.
@T.prim_func
def dynamic_gm_active_stride_dynamic_kernel(
    A: T.Tensor((4, dynamic_gm_active_dynamic_s, 16, 32), DTYPE),
    B: T.Tensor((1, 8, 1, 16), DTYPE),
    i: T.int32,
    idx1: T.int32,
    j: T.int32,
    idx2: T.int32,
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((1, 8, 1, 16), DTYPE)
        tail = T.min(32 - idx2, 16)
        T.copy(
            A[i : i + 1, idx1 : idx1 + 8, j : j + 1, idx2 : idx2 + tail],
            UB[:, :, :, 0:tail],
        )
        T.copy(UB[:, :, :, 0:tail], B[:, :, :, 0:tail])


# src.shape = [M, N]
# src.range = [bx:bx+remain_m, by:by+remain_n], src.slice = [remain_m, remain_n]
# dst.shape = [32, 32] / [M, N]
# dst.range = [:, :] borrowed from the source slice on the UB side, then
# [bx:bx+remain_m, by:by+remain_n] on the GM side
# Tests the vec-add-style dynamic 2D tail-tile copy with plain UB buffers.
@T.prim_func
def dynamic_gm_2d_tile_kernel(
    A: T.Tensor((dynamic_2d_tile_m, dynamic_2d_tile_n), DTYPE),
    B: T.Tensor((dynamic_2d_tile_m, dynamic_2d_tile_n), DTYPE),
    bx: T.int32,
    by: T.int32,
    remain_m: T.int32,
    remain_n: T.int32,
):
    with T.Kernel(1, is_npu=True):
        UB = T.alloc_ub((32, 32), DTYPE)
        T.copy(A[bx : bx + remain_m, by : by + remain_n], UB)
        T.copy(UB, B[bx : bx + remain_m, by : by + remain_n])


def test_copy_general_same_shape_2d():
    kernel = _compile(same_shape_2d_kernel)
    a = gen_tensor((64, 128), DTYPE, kind="randn")
    b = gen_tensor((64, 128), DTYPE, kind="zeros")
    kernel(a, b)
    assert_close(b.cpu(), a.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_add_leading_one_dim():
    kernel = _compile(add_leading_one_dim_kernel)
    a = gen_tensor((64, 128), DTYPE, kind="randn")
    b = gen_tensor((1, 64, 128), DTYPE, kind="zeros")
    kernel(a, b)

    expected = torch.zeros_like(b)
    expected[0, :, :] = a
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_remove_multiple_one_dims():
    kernel = _compile(remove_multiple_one_dims_kernel)
    a = gen_tensor((1, 64, 1, 128), DTYPE, kind="randn")
    b = gen_tensor((64, 128), DTYPE, kind="zeros")
    kernel(a, b)

    expected = a[0, :, 0, :].clone()
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_min_offsets():
    args = (2, 3, 5, 7)
    kernel = _compile(dynamic_min_offsets_kernel)
    a = gen_tensor((4, 16, 16, 32), DTYPE, kind="randn")
    b = gen_tensor((4, 16, 16, 32), DTYPE, kind="zeros")
    kernel(a, b, *args)

    expected = torch.zeros_like(b)
    i, idx1, j, idx2 = args
    expected[i, idx1 : idx1 + 8, j, idx2 : idx2 + 8] = a[
        i, idx1 : idx1 + 8, j, idx2 : idx2 + 8
    ]
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_extent_tail():
    row, start = (2, 23)
    kernel = _compile(dynamic_extent_tail_kernel)
    a = gen_tensor((4, 32), DTYPE, kind="randn")
    b = gen_tensor((4, 32), DTYPE, kind="zeros")
    kernel(a, b, row, start)

    tail = min(32 - start, 16)
    expected = torch.zeros_like(b)
    expected[row, start : start + tail] = a[row, start : start + tail]
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_gm_active_stride_static():
    runtime_s = 9
    args = (3, 4, 2, 8)
    kernel = _compile(dynamic_gm_active_stride_static_kernel)
    a = gen_tensor((16, runtime_s, 16, 32), DTYPE, kind="randn")
    b = gen_tensor((8, 16), DTYPE, kind="zeros")
    kernel(a, b, *args)

    i, j, idx1, idx2 = args
    expected = a[i, j, idx1 : idx1 + 8, idx2 : idx2 + 16].clone()
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_backbone_tail():
    tail = 13
    kernel = _compile(dynamic_backbone_tail_kernel)
    a = gen_tensor((1, 8, tail), DTYPE, kind="randn")
    b = gen_tensor((1, 8, 16), DTYPE, kind="zeros")
    kernel(a, b)

    expected = torch.zeros_like(b)
    expected[:, :, 0:tail] = a
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_gm_active_stride_dynamic():
    runtime_s = 9
    args = (0, 1, 2, 7)
    kernel = _compile(dynamic_gm_active_stride_dynamic_kernel)
    a = gen_tensor((4, runtime_s, 16, 32), DTYPE, kind="randn")
    b = gen_tensor((1, 8, 1, 16), DTYPE, kind="zeros")
    kernel(a, b, *args)

    i, idx1, j, idx2 = args
    tail = min(32 - idx2, 16)
    expected = torch.zeros_like(b)
    expected[:, :, :, 0:tail] = a[
        i : i + 1, idx1 : idx1 + 8, j : j + 1, idx2 : idx2 + tail
    ]
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)


def test_copy_general_dynamic_gm_2d_tile():
    runtime_m, runtime_n = 77, 88
    bx, by = 64, 64
    remain_m = runtime_m - bx
    remain_n = runtime_n - by
    kernel = _compile(dynamic_gm_2d_tile_kernel)
    a = gen_tensor((runtime_m, runtime_n), DTYPE, kind="randn")
    b = gen_tensor((runtime_m, runtime_n), DTYPE, kind="zeros")
    kernel(a, b, bx, by, remain_m, remain_n)

    expected = torch.zeros_like(b)
    expected[bx : bx + remain_m, by : by + remain_n] = a[
        bx : bx + remain_m, by : by + remain_n
    ]
    assert_close(b.cpu(), expected.cpu(), dtype=DTYPE, rtol=1e-2, atol=1e-2)
