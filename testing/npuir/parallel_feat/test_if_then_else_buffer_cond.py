# Copyright (c) Huawei Technologies Co., Ltd. 2025.
#
# This unit test verifies the support of T.Parallel for the following scenarios:
# 1. Vectorization of mask generation
# 2. Correct vectorization of 2D T.Parallel with if_then_else
# 3. Correct handling of broadcast-style access: mask[b_i] used in (h_i, b_i)
#

import pytest
import torch

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_if_then_else"),
    pytest.mark.mode("Developer"),
]

CASES = [
    (4, 8, 4, 8),
]


def kernel_mask_fill_simple(BS, H, block_BS, block_H, accum_dtype="float32"):
    bs_num = T.ceildiv(BS, block_BS)
    h_num = T.ceildiv(H, block_H)

    @T.prim_func
    def mask_fill_simple(
        A: T.Tensor((BS,), accum_dtype),
        B: T.Tensor((BS,), accum_dtype),
        acc_p: T.Tensor((H, BS), accum_dtype),
        bs_shape: T.int32,
        h_shape: T.int32,
    ):
        with T.Kernel(bs_num * h_num, is_npu=True) as (cid, _):
            bi_i = cid % bs_num
            hi_i = cid // bs_num

            A_VEC = T.alloc_ub((block_BS,), accum_dtype)
            B_VEC = T.alloc_ub((block_BS,), accum_dtype)
            acc_p_BUF = T.alloc_ub((block_H, block_BS), accum_dtype)
            mask = T.alloc_ub((block_BS,), "bool")

            bs_start = bi_i * block_BS
            h_start = hi_i * block_H

            bs_remaining = bs_shape - bs_start
            h_remaining = h_shape - h_start
            bs_tail = T.min(block_BS, bs_remaining)
            h_tail = T.min(block_H, h_remaining)

            T.copy(A[bs_start : bs_start + bs_tail], A_VEC[:bs_tail])
            T.copy(B[bs_start : bs_start + bs_tail], B_VEC[:bs_tail])
            T.copy(
                acc_p[h_start : h_start + h_tail, bs_start : bs_start + bs_tail],
                acc_p_BUF[:h_tail, :bs_tail],
            )

            for b_i in T.Parallel(block_BS):
                mask[b_i] = A_VEC[b_i] > B_VEC[b_i]

            for h_i, b_i in T.Parallel(block_H, block_BS):
                acc_p_BUF[h_i, b_i] = T.if_then_else(
                    mask[b_i],
                    T.float32(0),
                    -T.infinity(accum_dtype),
                )

            T.copy(
                acc_p_BUF[:h_tail, :bs_tail],
                acc_p[h_start : h_start + h_tail, bs_start : bs_start + bs_tail],
            )

    return mask_fill_simple


@pytest.mark.parametrize("BS, H, block_BS, block_H", CASES)
def test_parallel_mask_fill_simple(BS, H, block_BS, block_H):
    func = kernel_mask_fill_simple(BS, H, block_BS, block_H)
    kernel = tilelang.compile(func, target="npuir")

    dtype = torch.float32

    A = torch.rand(size=(BS,), dtype=dtype, device="npu")
    B = torch.rand(size=(BS,), dtype=dtype, device="npu")
    acc_p = torch.zeros(size=(H, BS), dtype=dtype, device="npu")

    ref_A = A.unsqueeze(0).expand(H, BS)
    ref_B = B.unsqueeze(0).expand(H, BS)
    ref_output = torch.where(
        ref_A > ref_B,
        torch.zeros_like(acc_p),
        torch.full_like(acc_p, float("-inf")),
    )

    kernel(A, B, acc_p, BS, H)

    torch.testing.assert_close(acc_p, ref_output, rtol=1e-3, atol=1e-2)
