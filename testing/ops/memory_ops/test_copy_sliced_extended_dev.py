# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]
SLICED_EXTENDED_4D_CASES = [
    (2, 64, 8, 128, 32, 64, 1, 3, 16, 32),
]
SLICED_EXTENDED_5D_CASES = [
    ((2, 2, 64, 4, 128), 16, 32, 1, 0, 2, 32, 64),
]


@tilelang.jit(target="npuir")
def strided_copy_4d_kernel(
    B,
    S,
    H,
    D,
    S_blk,
    D_blk,
    dtype="float16",
):
    @T.prim_func
    def strided_copy_4d_kernel(
        In: T.Tensor((B, S, H, D), dtype),
        Out: T.Tensor((B, S, H, D), dtype),
        Debug_Frag: T.Tensor((S_blk, D_blk), dtype),
        idx_b: T.int32,
        idx_h: T.int32,
        off_s: T.int32,
        off_d: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            frag = T.alloc_fragment((S_blk, D_blk), dtype)

            T.copy(In[idx_b, off_s:off_s + S_blk, idx_h, off_d:off_d + D_blk], frag)
            T.copy(frag, Debug_Frag)
            T.copy(frag, Out[idx_b, off_s:off_s + S_blk, idx_h, off_d:off_d + D_blk])

    return strided_copy_4d_kernel


@tilelang.jit(target="npuir")
def strided_copy_5d_kernel(
    D0,
    D1,
    D2,
    D3,
    D4,
    Blk_2,
    Blk_4,
    dtype="float16",
):
    @T.prim_func
    def strided_copy_5d_kernel(
        In: T.Tensor((D0, D1, D2, D3, D4), dtype),
        Out: T.Tensor((D0, D1, D2, D3, D4), dtype),
        Debug_Frag: T.Tensor((Blk_2, Blk_4), dtype),
        idx_01: T.int32,
        idx_3: T.int32,
        off_2: T.int32,
        off_4: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            frag = T.alloc_fragment((Blk_2, Blk_4), dtype)

            r_idx_0 = idx_01 // D1
            r_idx_1 = idx_01 % D1

            T.copy(
                In[r_idx_0, r_idx_1, off_2:off_2 + Blk_2, idx_3, off_4:off_4 + Blk_4],
                frag,
            )
            T.copy(frag, Debug_Frag)
            T.copy(
                frag,
                Out[r_idx_0, r_idx_1, off_2:off_2 + Blk_2, idx_3, off_4:off_4 + Blk_4],
            )

    return strided_copy_5d_kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "B, S, H, D, S_blk, D_blk, idx_b, idx_h, off_s, off_d",
    SLICED_EXTENDED_4D_CASES,
)
def test_copy_sliced_extended_4d_dev(dtype, B, S, H, D, S_blk, D_blk, idx_b, idx_h, off_s, off_d):
    kernel = strided_copy_4d_kernel(B, S, H, D, S_blk, D_blk, dtype=dtype)

    inp = gen_tensor((B, S, H, D), dtype, kind="randn")
    out = gen_tensor((B, S, H, D), dtype, kind="zeros")
    debug = gen_tensor((S_blk, D_blk), dtype, kind="zeros")

    kernel(inp, out, debug, idx_b, idx_h, off_s, off_d)

    expected_slice = inp[idx_b, off_s:off_s + S_blk, idx_h, off_d:off_d + D_blk]
    assert_close(debug.cpu(), expected_slice.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)

    expected_out = torch.zeros_like(out)
    expected_out[idx_b, off_s:off_s + S_blk, idx_h, off_d:off_d + D_blk] = expected_slice
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dims, blk_2, blk_4, idx_0, idx_1, idx_3, off_2, off_4",
    SLICED_EXTENDED_5D_CASES,
)
def test_copy_sliced_extended_5d_dev(dtype, dims, blk_2, blk_4, idx_0, idx_1, idx_3, off_2, off_4):
    D1 = dims[1]
    idx_01_merged = idx_0 * D1 + idx_1

    kernel = strided_copy_5d_kernel(*dims, blk_2, blk_4, dtype=dtype)

    inp = gen_tensor(dims, dtype, kind="randn")
    out = gen_tensor(dims, dtype, kind="zeros")
    debug = gen_tensor((blk_2, blk_4), dtype, kind="zeros")

    kernel(inp, out, debug, idx_01_merged, idx_3, off_2, off_4)

    expected_slice = inp[idx_0, idx_1, off_2:off_2 + blk_2, idx_3, off_4:off_4 + blk_4]
    assert_close(debug.cpu(), expected_slice.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)

    expected_out = torch.zeros_like(out)
    expected_out[idx_0, idx_1, off_2:off_2 + blk_2, idx_3, off_4:off_4 + blk_4] = expected_slice
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-5, atol=1e-5)
