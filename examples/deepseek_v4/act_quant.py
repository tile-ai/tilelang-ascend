# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import logging
from typing import Tuple

import tilelang
import tilelang.language as T


torch.set_default_device("npu")
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)

tilelang.disable_cache()

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def act_quant_kernel_int8_optimized(N: int, block_M: int = 32, block_N: int = 32, round_scale: bool = False):
    M = T.symbolic("M")
    VEC_NUM = 2
    CAST_MODE = "CAST_NONE"

    int8_min = -128
    int8_max = 127
    int8_abs_max = 127.0

    m_num = M // block_M
    n_num = N // block_N
    block_M_2 = block_M // VEC_NUM

    @T.prim_func
    def main(
        X: T.Tensor([M, N], "bfloat16"),
        Y: T.Tensor([M, N], "int8"),
        S: T.Tensor([M], "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            x_ub = T.alloc_ub([block_M_2, block_N], "bfloat16")
            x_ub_half = T.alloc_ub([block_M_2, block_N], "float16")
            x_ub_fp_abs = T.alloc_ub([block_M_2, block_N], "float")
            y_ub = T.alloc_ub([block_M_2, block_N], "int8")

            max_ub = T.alloc_ub([block_M_2], "float")
            scale_ub = T.alloc_ub([block_M_2], "float")
            x_ub_fp = T.alloc_ub([block_M_2, block_N], "float")
            x_ub_fp_1 = T.alloc_ub([block_M_2, block_N], "float")
            scale_global = T.alloc_ub([block_M_2, 1], "float")

            with T.Scope("V"):
                T.copy(X[bm * block_M + vid * block_M_2, bn * block_N], x_ub)
                T.tile.fill(max_ub, 0.0)

                T.tile.cast(x_ub_fp, x_ub, mode=CAST_MODE, count=block_M_2 * block_N)
                T.tile.abs(x_ub_fp_abs, x_ub_fp)

                T.reduce_max(x_ub_fp_abs, max_ub, dim=-1)

                for i in T.Parallel(block_M_2):
                    scale_ub[i] = max_ub[i] / int8_abs_max

                for i, j in T.Parallel(block_M_2, block_N):
                    x_ub_fp[i, j] = x_ub_fp[i, j] / scale_ub[i]

                T.tile.clamp(x_ub_fp, x_ub_fp, -127.0, 127.0, block_M_2 * block_N)

                T.tile.round(x_ub_fp, x_ub_fp, block_M_2 * block_N)

                T.tile.cast(x_ub_half, x_ub_fp, mode=CAST_MODE, count=block_M_2 * block_N)

                T.tile.cast(y_ub, x_ub_half, mode=CAST_MODE, count=block_M_2 * block_N)
                T.copy(
                    y_ub,
                    Y[bm * block_M + vid * block_M_2 : bm * block_M + vid * block_M_2 + block_M_2, bn * block_N : bn * block_N + block_N],
                )

                T.copy(scale_ub, S[bm * block_M + vid * block_M_2 : bm * block_M + vid * block_M_2 + block_M_2])

    return main


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


# golden
def act_quant_torch(x: torch.Tensor, round_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    original_shape = x.shape

    if x.dim() == 3:
        batch, seq, N = x.shape
        M = batch * seq
        x_2d = x.view(M, N)
    else:
        x_2d = x
        M, N = x_2d.shape

    x_fp32 = x_2d.float()

    abs_max = torch.max(torch.abs(x_fp32), dim=1, keepdim=True)[0]  # [M, 1]

    abs_max = torch.clamp(abs_max, min=1e-4)

    if round_scale:
        scales = 2 ** torch.ceil(torch.log2(abs_max / 127.0))
    else:
        scales = abs_max / 127.0

    scaled = x_fp32 / scales
    clipped = torch.clamp(scaled, -127, 127)
    x_int8 = torch.round(clipped).to(torch.float16).to(torch.int8)

    if len(original_shape) == 3:
        x_int8 = x_int8.view(original_shape)
        scales = scales.view(batch, seq, 1)

    return x_int8, scales


def validate_act_quant_kernel(x_bf16, M, N):
    x = x_bf16
    x_int8_torch, scales_torch = act_quant_torch(x, round_scale=False)

    return x_int8_torch, scales_torch


def test(custom_args=None):
    M, N = 128, 1024

    x_bf16 = torch.randn(M, N, dtype=torch.bfloat16)

    kernel = act_quant_kernel_int8_optimized(N, block_M=16, block_N=N, round_scale=False)
    logging.info("init successful!")

    Y, S = kernel(x_bf16.npu())

    x_int8_torch, scales_torch = validate_act_quant_kernel(x_bf16, M, N)
    torch.npu.synchronize()

    torch.testing.assert_close(Y, x_int8_torch, rtol=1e-2, atol=1)
    torch.testing.assert_close(S.reshape(M), scales_torch.reshape(M), rtol=1e-2, atol=1e-2)
    logging.info("Kernel Output Match!")


if __name__ == "__main__":
    test()
