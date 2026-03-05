# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import ascend_mode, assert_close, gen_tensor


@tilelang.jit(target="npuir")
def slice_copy_2d_kernel(block_M, block_N, idx, idx2, dtype="float16"):
    @T.prim_func
    def slice_copy_2d_kernel(
        In_ones: T.Tensor((block_M, block_N), dtype),
        In_zeros: T.Tensor((block_M, block_N), dtype),
        Out1: T.Tensor((block_M, block_N), dtype),
        Out2: T.Tensor((block_M, block_N), dtype),
        Out3: T.Tensor(block_N, dtype),
        Out4: T.Tensor(block_N, dtype),
        Out5: T.Tensor((block_M, block_N), dtype),
        Out6: T.Tensor((block_M, block_N), dtype),
        Out7: T.Tensor((block_M, block_N), dtype),
        Out8: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((block_M, block_N), dtype)
            B_frag = T.alloc_fragment((block_M, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), dtype)
            D_frag = T.alloc_fragment((block_M, block_N), dtype)

            A_slice = T.alloc_fragment(block_N, dtype)
            B_slice = T.alloc_fragment(block_N, dtype)

            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag)
            T.copy(In_zeros, D_frag)

            T.copy(A_frag[idx, :], A_slice)
            T.copy(A_slice, B_frag[idx, :])

            T.copy(In_ones[idx, :], B_slice)
            T.copy(B_slice, Out5[idx, :])

            T.copy(In_ones[idx, :], C_frag[idx2, :])
            T.copy(C_frag[idx2, :], Out6[idx, :])

            T.copy(A_frag[idx, :], D_frag[idx2, :])

            T.copy(C_frag, Out7)
            T.copy(D_frag, Out8)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)

    return slice_copy_2d_kernel


@tilelang.jit(target="npuir")
def slice_copy_3d_kernel(B, M, N, idx, idx2, dtype="float16"):
    @T.prim_func
    def slice_copy_3d_kernel(
        In_ones: T.Tensor((B, M, N), dtype),
        In_zeros: T.Tensor((B, M, N), dtype),
        Out1: T.Tensor((B, M, N), dtype),
        Out2: T.Tensor((B, M, N), dtype),
        Out3: T.Tensor((M, N), dtype),
        Out4: T.Tensor((M, N), dtype),
        Out5: T.Tensor((B, M, N), dtype),
        Out6: T.Tensor((B, M, N), dtype),
        Out7: T.Tensor((B, M, N), dtype),
        Out8: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((B, M, N), dtype)
            B_frag = T.alloc_fragment((B, M, N), dtype)
            C_frag = T.alloc_fragment((B, M, N), dtype)
            D_frag = T.alloc_fragment((B, M, N), dtype)

            A_slice = T.alloc_fragment((M, N), dtype)
            B_slice = T.alloc_fragment((M, N), dtype)

            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag)
            T.copy(In_zeros, D_frag)

            T.copy(A_frag[idx, :, :], A_slice)
            T.copy(A_slice, B_frag[idx, :, :])
            T.copy(In_ones[idx, :, :], B_slice)
            T.copy(B_slice, Out5[idx, :, :])
            T.copy(In_ones[idx, :, :], C_frag[idx2, :, :])
            T.copy(C_frag[idx2, :, :], Out6[idx, :, :])

            T.copy(A_frag[idx, :, :], D_frag[idx2, :, :])

            T.copy(C_frag, Out7)
            T.copy(D_frag, Out8)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)

    return slice_copy_3d_kernel


@tilelang.jit(target="npuir")
def slice_copy_4d_kernel(B, H, M, N, idx_b, idx_h, idx_b2, idx_h2, dtype="float16"):
    @T.prim_func
    def slice_copy_4d_kernel(
        In_ones: T.Tensor((B, H, M, N), dtype),
        In_zeros: T.Tensor((B, H, M, N), dtype),
        Out1: T.Tensor((B, H, M, N), dtype),
        Out2: T.Tensor((B, H, M, N), dtype),
        Out3: T.Tensor((M, N), dtype),
        Out4: T.Tensor((M, N), dtype),
        Out5: T.Tensor((B, H, M, N), dtype),
        Out6: T.Tensor((B, H, M, N), dtype),
        Out7: T.Tensor((B, H, M, N), dtype),
        Out8: T.Tensor((B, H, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((B, H, M, N), dtype)
            B_frag = T.alloc_fragment((B, H, M, N), dtype)
            C_frag = T.alloc_fragment((B, H, M, N), dtype)
            D_frag = T.alloc_fragment((B, H, M, N), dtype)

            A_slice = T.alloc_fragment((M, N), dtype)
            B_slice = T.alloc_fragment((M, N), dtype)

            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag)
            T.copy(In_zeros, D_frag)

            T.copy(A_frag[idx_b, idx_h, :, :], A_slice)
            T.copy(A_slice, B_frag[idx_b, idx_h, :, :])
            T.copy(In_ones[idx_b, idx_h, :, :], B_slice)
            T.copy(B_slice, Out5[idx_b, idx_h, :, :])
            T.copy(In_ones[idx_b, idx_h, :, :], C_frag[idx_b2, idx_h2, :, :])
            T.copy(C_frag[idx_b2, idx_h2, :, :], Out6[idx_b, idx_h, :, :])

            T.copy(A_frag[idx_b, idx_h, :, :], D_frag[idx_b2, idx_h2, :, :])

            T.copy(C_frag, Out7)
            T.copy(D_frag, Out8)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)

    return slice_copy_4d_kernel


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_sliced_2d_dev():
    block_M, block_N = 32, 128
    idx, idx2 = 8, 16

    with ascend_mode("Developer"):
        kernel = slice_copy_2d_kernel(block_M, block_N, idx, idx2)

        input_ones = gen_tensor((block_M, block_N), "float16", kind="ones")
        input_zeros = gen_tensor((block_M, block_N), "float16", kind="zeros")

        outs = [gen_tensor((block_M, block_N), "float16", kind="zeros") for _ in range(6)]
        out3 = gen_tensor((block_N,), "float16", kind="zeros")
        out4 = gen_tensor((block_N,), "float16", kind="zeros")

        kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])

    expected_full = gen_tensor((block_M, block_N), "float16", kind="ones")
    expected_row = gen_tensor((block_N,), "float16", kind="ones")

    assert_close(outs[0].cpu(), expected_full.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        assert_close(out[idx].cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
        out[idx] = 0
        assert_close(out.cpu(), torch.zeros_like(out).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(out3.cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(out4.cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[3][idx].cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[3][idx] = 0
    assert_close(outs[3].cpu(), torch.zeros_like(outs[3]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[4][idx2].cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[4][idx2] = 0
    assert_close(outs[4].cpu(), torch.zeros_like(outs[4]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[5][idx2].cpu(), expected_row.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[5][idx2] = 0
    assert_close(outs[5].cpu(), torch.zeros_like(outs[5]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_sliced_3d_dev():
    B, M, N = 4, 32, 128
    idx, idx2 = 1, 3

    with ascend_mode("Developer"):
        kernel = slice_copy_3d_kernel(B, M, N, idx, idx2)

        input_ones = gen_tensor((B, M, N), "float16", kind="ones")
        input_zeros = gen_tensor((B, M, N), "float16", kind="zeros")

        outs = [gen_tensor((B, M, N), "float16", kind="zeros") for _ in range(6)]
        out3 = gen_tensor((M, N), "float16", kind="zeros")
        out4 = gen_tensor((M, N), "float16", kind="zeros")

        kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])

    expected_full = gen_tensor((B, M, N), "float16", kind="ones")
    expected_slice = gen_tensor((M, N), "float16", kind="ones")

    assert_close(outs[0].cpu(), expected_full.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        assert_close(out[idx].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
        out[idx] = 0
        assert_close(out.cpu(), torch.zeros_like(out).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(out3.cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(out4.cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[3][idx].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[3][idx] = 0
    assert_close(outs[3].cpu(), torch.zeros_like(outs[3]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[4][idx2].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[4][idx2] = 0
    assert_close(outs[4].cpu(), torch.zeros_like(outs[4]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[5][idx2].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[5][idx2] = 0
    assert_close(outs[5].cpu(), torch.zeros_like(outs[5]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)


@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_sliced_4d_dev():
    B, H, M, N = 2, 4, 32, 64
    idx_b, idx_h = 0, 1
    idx_b2, idx_h2 = 1, 3

    with ascend_mode("Developer"):
        kernel = slice_copy_4d_kernel(B, H, M, N, idx_b, idx_h, idx_b2, idx_h2)

        input_ones = gen_tensor((B, H, M, N), "float16", kind="ones")
        input_zeros = gen_tensor((B, H, M, N), "float16", kind="zeros")

        outs = [gen_tensor((B, H, M, N), "float16", kind="zeros") for _ in range(6)]
        out3 = gen_tensor((M, N), "float16", kind="zeros")
        out4 = gen_tensor((M, N), "float16", kind="zeros")

        kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])

    expected_full = gen_tensor((B, H, M, N), "float16", kind="ones")
    expected_slice = gen_tensor((M, N), "float16", kind="ones")

    assert_close(outs[0].cpu(), expected_full.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        assert_close(out[idx_b, idx_h].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
        out[idx_b, idx_h] = 0
        assert_close(out.cpu(), torch.zeros_like(out).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(out3.cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    assert_close(out4.cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[3][idx_b, idx_h].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[3][idx_b, idx_h] = 0
    assert_close(outs[3].cpu(), torch.zeros_like(outs[3]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[4][idx_b2, idx_h2].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[4][idx_b2, idx_h2] = 0
    assert_close(outs[4].cpu(), torch.zeros_like(outs[4]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)

    assert_close(outs[5][idx_b2, idx_h2].cpu(), expected_slice.cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
    outs[5][idx_b2, idx_h2] = 0
    assert_close(outs[5].cpu(), torch.zeros_like(outs[5]).cpu(), dtype="float16", rtol=1e-5, atol=1e-5)
