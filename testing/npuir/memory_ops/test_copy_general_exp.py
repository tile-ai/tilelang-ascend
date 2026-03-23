# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""
Expert-mode copy coverage focused on rank-mismatch slice copies.

The goal is to cover the copy semantics that are already exercised in
developer-mode sliced tests, but with a compact expert-mode matrix:
  1. Low-rank tensor <-> high-rank buffer slice
  2. High-rank tensor slice <-> low-rank buffer
  3. 3D tensor slice <-> 2D buffer
  4. 4D tensor slice <-> 2D buffer
"""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Expert"),
]

DTYPES = ["float16"]
VECTOR_2D_BUF_CASES = [
    (3, 32, 1),
]
ROW_1D_BUF_CASES = [
    (16, 32, 5),
]
SLICE_3D_2D_BUF_CASES = [
    (4, 8, 16, 2),
]
SLICE_4D_2D_BUF_CASES = [
    (2, 3, 8, 16, 1, 2),
]
STRIDED_2D_CASES = [
    (96, 64, 3, 8),
]


def vector_copy_via_matrix_buffer(buf_rows, width, dtype):
    @T.prim_func
    def vectorCopyViaMatrixBuffer(
        A: T.Tensor((width,), dtype),
        Zero: T.Tensor((buf_rows, width), dtype),
        B: T.Tensor((width,), dtype),
        Debug: T.Tensor((buf_rows, width), dtype),
        row_idx: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            A_BUF = T.alloc_ub((buf_rows, width), dtype)

            T.copy(Zero, A_BUF)
            T.copy(A, A_BUF[row_idx, :])
            T.copy(A_BUF, Debug)
            T.copy(A_BUF[row_idx, :], B)

    return vectorCopyViaMatrixBuffer


def row_copy_via_vector_buffer(M, N, dtype):
    @T.prim_func
    def rowCopyViaVectorBuffer(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        Debug: T.Tensor((N,), dtype),
        row_idx: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            A_BUF = T.alloc_ub((N,), dtype)

            T.copy(A[row_idx, :], A_BUF)
            T.copy(A_BUF, Debug)
            T.copy(A_BUF, B[row_idx, :])

    return rowCopyViaVectorBuffer


def plane_copy_via_matrix_buffer(B, M, N, dtype):
    @T.prim_func
    def planeCopyViaMatrixBuffer(
        A: T.Tensor((B, M, N), dtype),
        Out: T.Tensor((B, M, N), dtype),
        Debug: T.Tensor((M, N), dtype),
        idx_b: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            A_BUF = T.alloc_ub((M, N), dtype)

            T.copy(A[idx_b, :, :], A_BUF)
            T.copy(A_BUF, Debug)
            T.copy(A_BUF, Out[idx_b, :, :])

    return planeCopyViaMatrixBuffer


def slice_copy_via_matrix_buffer(B, H, M, N, dtype):
    @T.prim_func
    def sliceCopyViaMatrixBuffer(
        A: T.Tensor((B, H, M, N), dtype),
        Out: T.Tensor((B, H, M, N), dtype),
        Debug: T.Tensor((M, N), dtype),
        idx_b: T.int32,
        idx_h: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            A_BUF = T.alloc_ub((M, N), dtype)

            T.copy(A[idx_b, idx_h, :, :], A_BUF)
            T.copy(A_BUF, Debug)
            T.copy(A_BUF, Out[idx_b, idx_h, :, :])

    return sliceCopyViaMatrixBuffer


def strided_copy_2d(total_h, width, stride, block_h, dtype):
    assert total_h % stride == 0
    out_h = total_h // stride
    assert out_h % block_h == 0

    @T.prim_func
    def stridedCopy2D(
        A: T.Tensor((total_h, width), dtype),
        B: T.Tensor((out_h, width), dtype),
    ):
        with T.Kernel(1, is_npu=True):
            A_BUF = T.alloc_ub((block_h, width), dtype)

            for block_idx in T.serial(out_h // block_h):
                out_start = block_idx * block_h
                in_start = out_start * stride

                for i in T.serial(block_h):
                    row_idx = in_start + i * stride
                    T.copy(A[row_idx : row_idx + 1, 0:width], A_BUF[i : i + 1, 0:width])

                T.copy(A_BUF, B[out_start : out_start + block_h, 0:width])

    return stridedCopy2D


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("buf_rows, width, row_idx", VECTOR_2D_BUF_CASES)
def test_copy_1d_with_2d_buffer_exp(dtype, buf_rows, width, row_idx):
    func = vector_copy_via_matrix_buffer(buf_rows, width, dtype)
    compiled = tilelang.compile(func, target="npuir")

    inp = gen_tensor((width,), dtype, kind="randn")
    zero = gen_tensor((buf_rows, width), dtype, kind="zeros")
    out = gen_tensor((width,), dtype, kind="zeros")
    debug = gen_tensor((buf_rows, width), dtype, kind="zeros")

    compiled(inp, zero, out, debug, row_idx)

    expected_debug = torch.zeros_like(debug)
    expected_debug[row_idx] = inp

    assert_close(debug.cpu(), expected_debug.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
    assert_close(out.cpu(), inp.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, row_idx", ROW_1D_BUF_CASES)
def test_copy_2d_slice_with_1d_buffer_exp(dtype, M, N, row_idx):
    func = row_copy_via_vector_buffer(M, N, dtype)
    compiled = tilelang.compile(func, target="npuir")

    inp = gen_tensor((M, N), dtype, kind="randn")
    out = gen_tensor((M, N), dtype, kind="zeros")
    debug = gen_tensor((N,), dtype, kind="zeros")

    compiled(inp, out, debug, row_idx)

    expected_row = inp[row_idx]
    expected_out = torch.zeros_like(out)
    expected_out[row_idx] = expected_row

    assert_close(debug.cpu(), expected_row.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B, M, N, idx_b", SLICE_3D_2D_BUF_CASES)
def test_copy_3d_slice_with_2d_buffer_exp(dtype, B, M, N, idx_b):
    func = plane_copy_via_matrix_buffer(B, M, N, dtype)
    compiled = tilelang.compile(func, target="npuir")

    inp = gen_tensor((B, M, N), dtype, kind="randn")
    out = gen_tensor((B, M, N), dtype, kind="zeros")
    debug = gen_tensor((M, N), dtype, kind="zeros")

    compiled(inp, out, debug, idx_b)

    expected_slice = inp[idx_b]
    expected_out = torch.zeros_like(out)
    expected_out[idx_b] = expected_slice

    assert_close(debug.cpu(), expected_slice.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B, H, M, N, idx_b, idx_h", SLICE_4D_2D_BUF_CASES)
def test_copy_4d_slice_with_2d_buffer_exp(dtype, B, H, M, N, idx_b, idx_h):
    func = slice_copy_via_matrix_buffer(B, H, M, N, dtype)
    compiled = tilelang.compile(func, target="npuir")

    inp = gen_tensor((B, H, M, N), dtype, kind="randn")
    out = gen_tensor((B, H, M, N), dtype, kind="zeros")
    debug = gen_tensor((M, N), dtype, kind="zeros")

    compiled(inp, out, debug, idx_b, idx_h)

    expected_slice = inp[idx_b, idx_h]
    expected_out = torch.zeros_like(out)
    expected_out[idx_b, idx_h] = expected_slice

    assert_close(debug.cpu(), expected_slice.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H, W, stride, block_h", STRIDED_2D_CASES)
def test_copy_strided_2d_exp(dtype, H, W, stride, block_h):
    func = strided_copy_2d(H, W, stride, block_h, dtype)
    compiled = tilelang.compile(func, target="npuir")

    inp = gen_tensor((H, W), dtype, kind="randn")
    out = gen_tensor((H // stride, W), dtype, kind="zeros")

    compiled(inp, out)

    expected_out = inp[::stride, :].contiguous()
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
