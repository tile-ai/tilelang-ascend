# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""DequantSwigluQuant TileLang kernel — fused dequant -> SwiGLU -> smooth-quant -> INT8.

Fuses per-column dequantization (int32 W8A8 path), the SwiGLU activation
(silu of one half times the other, selectable via activate_left), an
optional per-column quant_scale multiply, and dynamic per-token INT8
quantization into a single kernel.  The host wrapper validates the input
contract, selects block_H/rows from a UB budget formula, and dispatches
to one of four kernel variants (int32/fp16 x quant_scale on/off).

Algorithm:
    d   = x (fp16/bf16)  |  x * weight_scale[1,2H] * activation_scale[M] (int32)
    s   = silu(d_left) * d_right  (or silu(d_right) * d_left)
    s  *= quant_scale[1,H]        (optional)
    y   = round_int8(s / scale),  scale = max(|s|, row) / 127 per token

Reference: golden_dequant_swiglu_quant in test_dequant_swiglu_quant.py

Interface: dequant_swiglu_quant(x, weight_scale, activation_scale,
          quant_scale, activate_left) -> (y int8 [M,H], scale fp32 [M])
"""

from typing import Optional, Tuple

import torch

import tilelang
import tilelang.language as T


ACC = "float32"
VEC_NUM = 2
UB_BUDGET = 192 * 1024
UB_SAFETY_MARGIN = 8 * 1024
PASS2_REUSE_FACTOR = 0.5
DEFAULT_ROWS = 2

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_kernel_cache = {}


def _make_main(M, H_orig, block_M, block_H, in_dtype, has_ws, has_qs, activate_left, rows):
    TwoH_orig = 2 * H_orig
    n_full = H_orig // block_H
    partial = H_orig % block_H
    has_partial = partial > 0
    n_total = n_full + (1 if has_partial else 0)
    m_num = (M + block_M - 1) // block_M
    rows_per_vid = block_M // VEC_NUM
    ROWS = min(rows, rows_per_vid)
    serial_iters = rows_per_vid // ROWS
    tile_elems = ROWS * block_H

    @T.prim_func
    def main(
        x: T.Tensor((M, TwoH_orig), in_dtype),
        weight_scale: T.Tensor((1, TwoH_orig), ACC),
        activation_scale: T.Tensor((M,), ACC),
        quant_scale: T.Tensor((1, H_orig), ACC),
        swiglu_ws: T.Tensor((M, H_orig), ACC),
        y: T.Tensor((M, H_orig), "int8"),
        scale: T.Tensor((M,), ACC),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):  # noqa: SIM117
            with T.Scope("V"):
                a_raw = T.alloc_ub((2, ROWS, block_H), in_dtype)
                b_raw = T.alloc_ub((2, ROWS, block_H), in_dtype)
                a_ub = T.alloc_ub((ROWS, block_H), ACC)
                b_ub = T.alloc_ub((ROWS, block_H), ACC)
                wsa_ub = T.alloc_ub((2, 1, block_H), ACC)
                wsb_ub = T.alloc_ub((2, 1, block_H), ACC)
                qs_ub = T.alloc_ub((2, 1, block_H), ACC)
                wsa_tile = T.alloc_ub((ROWS, block_H), ACC)
                wsb_tile = T.alloc_ub((ROWS, block_H), ACC)
                qs_tile = T.alloc_ub((ROWS, block_H), ACC)
                as_ub = T.alloc_ub((ROWS), ACC)
                as_tile = T.alloc_ub((ROWS, block_H), ACC)
                silu_ub = T.alloc_ub((ROWS, block_H), ACC)
                swiglu_ub = T.alloc_ub((2, ROWS, block_H), ACC)
                abs_ub = T.alloc_ub((ROWS, block_H), ACC)
                running_max = T.alloc_ub((ROWS), ACC)
                scale_ub = T.alloc_ub((ROWS), ACC)
                scale_tile = T.alloc_ub((ROWS, block_H), ACC)
                q_ub = T.alloc_ub((ROWS, block_H), ACC)
                q_fp16_ub = T.alloc_ub((ROWS, block_H), "float16")
                y_ub = T.alloc_ub((2, ROWS, block_H), "int8")

                row_base = cid * block_M + vid * rows_per_vid

                for r in T.serial(serial_iters):
                    row_start = row_base + r * ROWS
                    T.tile.fill(running_max, 0.0)
                    if has_ws:
                        T.copy(activation_scale[row_start : row_start + ROWS], as_ub)
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.tile.broadcast(as_tile, as_ub, axis=1)

                    if n_full == 1 and not has_partial:
                        T.copy(x[row_start : row_start + ROWS, 0:block_H], a_raw[0, :, :])
                        T.copy(x[row_start : row_start + ROWS, H_orig : H_orig + block_H], b_raw[0, :, :])
                        if has_ws:
                            T.copy(weight_scale[0, 0:block_H], wsa_ub[0, 0, :])
                            T.copy(weight_scale[0, H_orig : H_orig + block_H], wsb_ub[0, 0, :])
                        if has_qs:
                            T.copy(quant_scale[0, 0:block_H], qs_ub[0, 0, :])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.cast(a_ub, a_raw[0, :, :], "CAST_NONE", tile_elems)
                        T.tile.cast(b_ub, b_raw[0, :, :], "CAST_NONE", tile_elems)
                        if has_ws:
                            T.tile.broadcast(wsa_tile, wsa_ub[0, :, :])
                            T.tile.mul(a_ub, a_ub, wsa_tile)
                            T.tile.broadcast(wsb_tile, wsb_ub[0, :, :])
                            T.tile.mul(b_ub, b_ub, wsb_tile)
                            T.tile.mul(a_ub, a_ub, as_tile)
                            T.tile.mul(b_ub, b_ub, as_tile)
                        if activate_left:
                            T.tile.silu(silu_ub, a_ub)
                            T.tile.mul(swiglu_ub[0, :, :], silu_ub, b_ub)
                        else:
                            T.tile.silu(silu_ub, b_ub)
                            T.tile.mul(swiglu_ub[0, :, :], silu_ub, a_ub)
                        if has_qs:
                            T.tile.broadcast(qs_tile, qs_ub[0, :, :])
                            T.tile.mul(swiglu_ub[0, :, :], swiglu_ub[0, :, :], qs_tile)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(swiglu_ub[0, :, :], swiglu_ws[row_start : row_start + ROWS, 0:block_H])
                        T.set_flag("mte3", "mte2", 7)
                        T.tile.abs(abs_ub, swiglu_ub[0, :, :])
                        T.reduce_max(abs_ub, running_max, dim=-1, clear=False)

                        T.tile.fill(as_ub, 127.0)
                        T.tile.div(scale_ub, running_max, as_ub)
                        T.tile.max(scale_ub, scale_ub, 1e-12)
                        T.set_flag("v", "mte3", 6)
                        T.wait_flag("v", "mte3", 6)
                        T.copy(scale_ub, scale[row_start : row_start + ROWS])
                        T.tile.broadcast(scale_tile, scale_ub, axis=1)

                        T.wait_flag("mte3", "mte2", 7)
                        T.copy(swiglu_ws[row_start : row_start + ROWS, 0:block_H], swiglu_ub[0, :, :])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.div(q_ub, swiglu_ub[0, :, :], scale_tile)
                        T.tile.clamp(q_ub, q_ub, -128.0, 127.0, tile_elems)
                        T.tile.cast(q_fp16_ub, q_ub, "CAST_NONE", tile_elems)
                        T.tile.cast(y_ub[0, :, :], q_fp16_ub, "CAST_RINT", tile_elems)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(y_ub[0, :, :], y[row_start : row_start + ROWS, 0:block_H])
                    else:
                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)

                        T.wait_flag("mte3", "mte2", 0)
                        T.copy(x[row_start : row_start + ROWS, 0:block_H], a_raw[0, :, :])
                        T.copy(x[row_start : row_start + ROWS, H_orig : H_orig + block_H], b_raw[0, :, :])
                        if has_ws:
                            T.copy(weight_scale[0, 0:block_H], wsa_ub[0, 0, :])
                            T.copy(weight_scale[0, H_orig : H_orig + block_H], wsb_ub[0, 0, :])
                        if has_qs:
                            T.copy(quant_scale[0, 0:block_H], qs_ub[0, 0, :])
                        T.set_flag("mte2", "v", 0)

                        for j in T.serial(0, n_full - 1):
                            cur = j % 2
                            nxt = (j + 1) % 2
                            ca = j * block_H
                            cb = H_orig + j * block_H  # noqa: F841
                            ca_n = (j + 1) * block_H
                            cb_n = H_orig + (j + 1) * block_H

                            T.wait_flag("mte3", "mte2", nxt)
                            T.copy(x[row_start : row_start + ROWS, ca_n : ca_n + block_H], a_raw[nxt, :, :])
                            T.copy(x[row_start : row_start + ROWS, cb_n : cb_n + block_H], b_raw[nxt, :, :])
                            if has_ws:
                                T.copy(weight_scale[0, ca_n : ca_n + block_H], wsa_ub[nxt, 0, :])
                                T.copy(weight_scale[0, cb_n : cb_n + block_H], wsb_ub[nxt, 0, :])
                            if has_qs:
                                T.copy(quant_scale[0, ca_n : ca_n + block_H], qs_ub[nxt, 0, :])
                            T.set_flag("mte2", "v", nxt)

                            T.wait_flag("mte2", "v", cur)
                            T.tile.cast(a_ub, a_raw[cur, :, :], "CAST_NONE", tile_elems)
                            T.tile.cast(b_ub, b_raw[cur, :, :], "CAST_NONE", tile_elems)
                            if has_ws:
                                T.tile.broadcast(wsa_tile, wsa_ub[cur, :, :])
                                T.tile.mul(a_ub, a_ub, wsa_tile)
                                T.tile.broadcast(wsb_tile, wsb_ub[cur, :, :])
                                T.tile.mul(b_ub, b_ub, wsb_tile)
                                T.tile.mul(a_ub, a_ub, as_tile)
                                T.tile.mul(b_ub, b_ub, as_tile)
                            if activate_left:
                                T.tile.silu(silu_ub, a_ub)
                                T.tile.mul(swiglu_ub[cur, :, :], silu_ub, b_ub)
                            else:
                                T.tile.silu(silu_ub, b_ub)
                                T.tile.mul(swiglu_ub[cur, :, :], silu_ub, a_ub)
                            if has_qs:
                                T.tile.broadcast(qs_tile, qs_ub[cur, :, :])
                                T.tile.mul(swiglu_ub[cur, :, :], swiglu_ub[cur, :, :], qs_tile)
                            T.set_flag("v", "mte3", cur)
                            T.wait_flag("v", "mte3", cur)
                            T.copy(swiglu_ub[cur, :, :], swiglu_ws[row_start : row_start + ROWS, ca : ca + block_H])
                            T.tile.abs(abs_ub, swiglu_ub[cur, :, :])
                            T.reduce_max(abs_ub, running_max, dim=-1, clear=False)
                            T.set_flag("mte3", "mte2", cur)

                        last = (n_full - 1) % 2
                        ca_l = (n_full - 1) * block_H
                        cb_l = H_orig + (n_full - 1) * block_H  # noqa: F841
                        T.wait_flag("mte2", "v", last)
                        T.tile.cast(a_ub, a_raw[last, :, :], "CAST_NONE", tile_elems)
                        T.tile.cast(b_ub, b_raw[last, :, :], "CAST_NONE", tile_elems)
                        if has_ws:
                            T.tile.broadcast(wsa_tile, wsa_ub[last, :, :])
                            T.tile.mul(a_ub, a_ub, wsa_tile)
                            T.tile.broadcast(wsb_tile, wsb_ub[last, :, :])
                            T.tile.mul(b_ub, b_ub, wsb_tile)
                            T.tile.mul(a_ub, a_ub, as_tile)
                            T.tile.mul(b_ub, b_ub, as_tile)
                        if activate_left:
                            T.tile.silu(silu_ub, a_ub)
                            T.tile.mul(swiglu_ub[last, :, :], silu_ub, b_ub)
                        else:
                            T.tile.silu(silu_ub, b_ub)
                            T.tile.mul(swiglu_ub[last, :, :], silu_ub, a_ub)
                        if has_qs:
                            T.tile.broadcast(qs_tile, qs_ub[last, :, :])
                            T.tile.mul(swiglu_ub[last, :, :], swiglu_ub[last, :, :], qs_tile)
                        T.set_flag("v", "mte3", last)
                        T.wait_flag("v", "mte3", last)
                        T.copy(swiglu_ub[last, :, :], swiglu_ws[row_start : row_start + ROWS, ca_l : ca_l + block_H])
                        T.tile.abs(abs_ub, swiglu_ub[last, :, :])
                        T.reduce_max(abs_ub, running_max, dim=-1, clear=False)
                        T.set_flag("mte3", "mte2", last)

                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)

                        if has_partial:
                            pa_off = n_full * block_H
                            pb_off = H_orig + n_full * block_H
                            T.tile.fill(a_raw[0, :, :], 0.0)
                            T.tile.fill(b_raw[0, :, :], 0.0)
                            T.copy(x[row_start : row_start + ROWS, pa_off:H_orig], a_raw[0, :, :], pad_value=0)
                            T.copy(x[row_start : row_start + ROWS, pb_off:TwoH_orig], b_raw[0, :, :], pad_value=0)
                            if has_ws:
                                T.copy(weight_scale[0, pa_off:H_orig], wsa_ub[0, 0, :], pad_value=0)
                                T.copy(weight_scale[0, pb_off:TwoH_orig], wsb_ub[0, 0, :], pad_value=0)
                            if has_qs:
                                T.copy(quant_scale[0, pa_off:H_orig], qs_ub[0, 0, :], pad_value=0)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.cast(a_ub, a_raw[0, :, :], "CAST_NONE", tile_elems)
                            T.tile.cast(b_ub, b_raw[0, :, :], "CAST_NONE", tile_elems)
                            if has_ws:
                                T.tile.broadcast(wsa_tile, wsa_ub[0, :, :])
                                T.tile.mul(a_ub, a_ub, wsa_tile)
                                T.tile.broadcast(wsb_tile, wsb_ub[0, :, :])
                                T.tile.mul(b_ub, b_ub, wsb_tile)
                                T.tile.mul(a_ub, a_ub, as_tile)
                                T.tile.mul(b_ub, b_ub, as_tile)
                            if activate_left:
                                T.tile.silu(silu_ub, a_ub)
                                T.tile.mul(swiglu_ub[0, :, :], silu_ub, b_ub)
                            else:
                                T.tile.silu(silu_ub, b_ub)
                                T.tile.mul(swiglu_ub[0, :, :], silu_ub, a_ub)
                            if has_qs:
                                T.tile.broadcast(qs_tile, qs_ub[0, :, :])
                                T.tile.mul(swiglu_ub[0, :, :], swiglu_ub[0, :, :], qs_tile)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(swiglu_ub[0, :, :partial], swiglu_ws[row_start : row_start + ROWS, pa_off:H_orig])
                            T.tile.abs(abs_ub, swiglu_ub[0, :, :])
                            T.reduce_max(abs_ub, running_max, dim=-1, clear=False)

                        T.tile.fill(as_ub, 127.0)
                        T.tile.div(scale_ub, running_max, as_ub)
                        T.tile.max(scale_ub, scale_ub, 1e-12)
                        T.set_flag("v", "mte3", 6)
                        T.wait_flag("v", "mte3", 6)
                        T.copy(scale_ub, scale[row_start : row_start + ROWS])
                        T.tile.broadcast(scale_tile, scale_ub, axis=1)

                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)

                        T.wait_flag("mte3", "mte2", 0)
                        T.copy(swiglu_ws[row_start : row_start + ROWS, 0:block_H], swiglu_ub[0, :, :])
                        T.set_flag("mte2", "v", 0)

                        for j in T.serial(0, n_total - 1):
                            cur = j % 2
                            nxt = (j + 1) % 2
                            ca_c = j * block_H
                            ca_n = (j + 1) * block_H

                            T.wait_flag("mte3", "mte2", nxt)
                            if has_partial and (j == n_total - 2):
                                T.copy(swiglu_ws[row_start : row_start + ROWS, ca_n:H_orig], swiglu_ub[nxt, :, :partial])
                            else:
                                T.copy(swiglu_ws[row_start : row_start + ROWS, ca_n : ca_n + block_H], swiglu_ub[nxt, :, :])
                            T.set_flag("mte2", "v", nxt)

                            T.wait_flag("mte2", "v", cur)
                            T.tile.div(q_ub, swiglu_ub[cur, :, :], scale_tile)
                            T.tile.clamp(q_ub, q_ub, -128.0, 127.0, tile_elems)
                            T.tile.cast(q_fp16_ub, q_ub, "CAST_NONE", tile_elems)
                            T.tile.cast(y_ub[cur, :, :], q_fp16_ub, "CAST_RINT", tile_elems)
                            T.set_flag("v", "mte3", cur)
                            T.wait_flag("v", "mte3", cur)
                            T.copy(y_ub[cur, :, :], y[row_start : row_start + ROWS, ca_c : ca_c + block_H])
                            T.set_flag("mte3", "mte2", cur)

                        last = (n_total - 1) % 2
                        ca_l = (n_total - 1) * block_H
                        T.wait_flag("mte2", "v", last)
                        T.tile.div(q_ub, swiglu_ub[last, :, :], scale_tile)
                        T.tile.clamp(q_ub, q_ub, -128.0, 127.0, tile_elems)
                        T.tile.cast(q_fp16_ub, q_ub, "CAST_NONE", tile_elems)
                        T.tile.cast(y_ub[last, :, :], q_fp16_ub, "CAST_RINT", tile_elems)
                        T.set_flag("v", "mte3", last)
                        T.wait_flag("v", "mte3", last)
                        if has_partial:
                            T.copy(y_ub[last, :, :partial], y[row_start : row_start + ROWS, ca_l:H_orig])
                        else:
                            T.copy(y_ub[last, :, :], y[row_start : row_start + ROWS, ca_l : ca_l + block_H])
                        T.set_flag("mte3", "mte2", last)

                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[5, 6], pass_configs=PASS_CONFIGS)
def _kernel_int32_qs(M, H_orig, block_M, block_H, activate_left=False, rows=DEFAULT_ROWS):
    return _make_main(M, H_orig, block_M, block_H, "int32", True, True, activate_left, rows)


@tilelang.jit(out_idx=[5, 6], pass_configs=PASS_CONFIGS)
def _kernel_int32_noqs(M, H_orig, block_M, block_H, activate_left=False, rows=DEFAULT_ROWS):
    return _make_main(M, H_orig, block_M, block_H, "int32", True, False, activate_left, rows)


@tilelang.jit(out_idx=[5, 6], pass_configs=PASS_CONFIGS)
def _kernel_fp16_qs(M, H_orig, block_M, block_H, in_dtype, activate_left=False, rows=DEFAULT_ROWS):
    return _make_main(M, H_orig, block_M, block_H, in_dtype, False, True, activate_left, rows)


@tilelang.jit(out_idx=[5, 6], pass_configs=PASS_CONFIGS)
def _kernel_fp16_noqs(M, H_orig, block_M, block_H, in_dtype, activate_left=False, rows=DEFAULT_ROWS):
    return _make_main(M, H_orig, block_M, block_H, in_dtype, False, False, activate_left, rows)


# =============================================================================
# UB Budget Formula (guide section 2.11)
# =============================================================================
def _find_max_tile(total_dim, rows, n_cal_p1, n_input, n_1d_cal, n_cal_p2, n_fp16_p2, n_int8_2d_p2, dtype_str):
    cal_bytes = 4
    if dtype_str in ("float16", "bfloat16"):
        input_bytes = 2
    elif dtype_str in ("float32", "int32"):
        input_bytes = 4
    else:
        input_bytes = 2

    p1_cost = rows * (n_cal_p1 * cal_bytes + n_input * input_bytes) + n_1d_cal * cal_bytes
    p2_cost = rows * (n_cal_p2 * cal_bytes + n_fp16_p2 * 2) + 2 * rows * n_int8_2d_p2
    p2_non_reusable = int(p2_cost * PASS2_REUSE_FACTOR)
    per_unit = p1_cost + p2_non_reusable
    effective_budget = UB_BUDGET - UB_SAFETY_MARGIN

    if per_unit == 0:
        return max(16, ((total_dim + 15) // 16) * 16)

    align = 256 if total_dim >= 256 else 16
    max_tile = (effective_budget // per_unit // align) * align
    max_tile = min(max_tile, ((total_dim + align - 1) // align) * align)
    max_tile = min(max_tile, total_dim)

    for t in range(max_tile, 0, -align):
        if total_dim % t == 0:
            return t

    best = max_tile
    best_n_num = (total_dim + max_tile - 1) // max_tile
    best_pad = best_n_num * max_tile - total_dim
    for t in range(max_tile, 0, -align):
        n_num = (total_dim + t - 1) // t
        pad = n_num * t - total_dim
        if n_num < best_n_num or (n_num == best_n_num and pad < best_pad):
            best_n_num = n_num
            best_pad = pad
            best = t
    return max(16, best)


def _count_buffers(has_ws, has_qs):
    n_input = 4
    n_cal_p1 = 2 + 4
    n_1d_cal = 0
    if has_ws:
        n_cal_p1 += 3
        n_1d_cal += 4
    if has_qs:
        n_cal_p1 += 1
        n_1d_cal += 2
    n_cal_p2 = 2
    n_fp16_p2 = 1
    n_int8_2d_p2 = 1
    return n_input, n_cal_p1, n_1d_cal, n_cal_p2, n_fp16_p2, n_int8_2d_p2


def _pick_block_h_and_rows(H, in_dtype, has_ws, has_qs, block_M):
    rows_per_vid = block_M // VEC_NUM
    n_input, n_cal_p1, n_1d, n_cal_p2, n_fp16_p2, n_int8_p2 = _count_buffers(has_ws, has_qs)

    best = None
    best_score = None

    for rows in [4, 2, 1]:
        if rows > rows_per_vid or rows_per_vid % rows != 0:
            continue

        bh = _find_max_tile(H, rows, n_cal_p1, n_input, n_1d, n_cal_p2, n_fp16_p2, n_int8_p2, in_dtype)
        n_full = H // bh
        partial = H % bh
        n_total = n_full + (1 if partial > 0 else 0)
        serial_iters = rows_per_vid // rows
        total_work = n_total * serial_iters

        if n_total == 1 and serial_iters > 1:
            continue

        score = (total_work, serial_iters, -bh)
        if best_score is None or score < best_score:
            best_score = score
            best = (rows, bh)

    return best


def _pick_block_m(M: int) -> int:
    for bm in (8, 4, 2):
        if M % bm == 0:
            return bm
    return 2


# =============================================================================
# Host Wrapper
# =============================================================================
def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    activate_left: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host entry: select kernel variant by dtype and quant_scale, compile and run."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"dequant_swiglu_quant: x must be a torch tensor, got {type(x).__name__}")
    if x.dim() != 2 or x.shape[1] % 2 != 0:
        raise ValueError(f"dequant_swiglu_quant: x must be 2D [M, 2H] with an even second dimension, got shape {tuple(x.shape)}")

    M_orig, TwoH = x.shape
    H_orig = TwoH // 2

    if x.dtype == torch.int32:
        in_dtype = "int32"
        has_ws = True
        if weight_scale is None or activation_scale is None:
            raise ValueError("dequant_swiglu_quant: int32 input requires weight_scale [1, 2H] and activation_scale [M]")
    elif x.dtype == torch.float16:
        in_dtype = "float16"
        has_ws = False
        if weight_scale is not None or activation_scale is not None:
            raise ValueError("dequant_swiglu_quant: weight_scale/activation_scale apply to int32 input only")
    elif x.dtype == torch.bfloat16:
        in_dtype = "bfloat16"
        has_ws = False
        if weight_scale is not None or activation_scale is not None:
            raise ValueError("dequant_swiglu_quant: weight_scale/activation_scale apply to int32 input only")
    else:
        raise ValueError(f"Unsupported x dtype: {x.dtype}")

    has_qs = quant_scale is not None

    M = M_orig
    block_M = _pick_block_m(M)
    rows, block_H = _pick_block_h_and_rows(H_orig, in_dtype, has_ws, has_qs, block_M)

    dummy_ws = torch.empty(1, TwoH, dtype=torch.float32, device=x.device)
    dummy_as = torch.empty(M, dtype=torch.float32, device=x.device)
    dummy_qs = torch.empty(1, H_orig, dtype=torch.float32, device=x.device)
    ws_in = weight_scale if has_ws else dummy_ws
    as_in = activation_scale if has_ws else dummy_as
    qs_in = quant_scale if has_qs else dummy_qs
    swiglu_ws = torch.empty(M, H_orig, dtype=torch.float32, device=x.device)

    rows_per_vid = block_M // VEC_NUM
    rows = min(rows, rows_per_vid)

    key = (M, H_orig, block_M, block_H, in_dtype, has_ws, has_qs, activate_left, rows)
    if key not in _kernel_cache:
        if in_dtype == "int32":
            if has_qs:
                _kernel_cache[key] = _kernel_int32_qs(M, H_orig, block_M, block_H, activate_left, rows)
            else:
                _kernel_cache[key] = _kernel_int32_noqs(M, H_orig, block_M, block_H, activate_left, rows)
        else:
            if has_qs:
                _kernel_cache[key] = _kernel_fp16_qs(M, H_orig, block_M, block_H, in_dtype, activate_left, rows)
            else:
                _kernel_cache[key] = _kernel_fp16_noqs(M, H_orig, block_M, block_H, in_dtype, activate_left, rows)

    kernel = _kernel_cache[key]
    y, scale = kernel(x, ws_in, as_in, qs_in, swiglu_ws)

    y = y[:M_orig, :H_orig]
    scale = scale[:M_orig]

    return y, scale


# ----------------------------------------------------------------------------
# CI smoke test (examples/bench_test.sh runs this file directly and requires
# "Kernel Output Match!" or "TEST PASSED!" in the output, same as the
# lstm/gru/depthwise_conv_2d/nms/group_norm/rms_norm/softmax smoke tests).
# The full layered suite over the 20 cann-bench cases lives in
# test_dequant_swiglu_quant.py.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    def _ref(x, ws, as_, qs, al):
        if x.dtype == torch.int32:
            d = x.float() * ws.float() * as_.float().unsqueeze(-1)
        else:
            d = x.float()
        half = d.shape[-1] // 2
        A, B = d[..., :half], d[..., half:]
        s = F.silu(A) * B if al else F.silu(B) * A
        if qs is not None:
            s = s * qs.float()
        mx = s.abs().amax(dim=-1)
        sc = (mx / 127.0).clamp_min(1e-12)
        y = torch.clamp((s / sc.unsqueeze(-1)).round(), -128, 127).to(torch.int8)
        return y, sc.float()

    # one case per dispatch variant: fp16 / int32+ws / int32+ws+qs
    for M, H, dt, has_ws, has_qs, al in (
        (512, 1024, torch.float16, False, False, True),
        (512, 1024, torch.int32, True, False, True),
        (512, 1024, torch.int32, True, True, False),
    ):
        TwoH = 2 * H
        if dt == torch.int32:
            x = torch.randint(-128, 128, (M, TwoH), dtype=torch.int32)
            w = torch.rand(1, TwoH) * 0.2 - 0.1
            a = torch.rand(M) * 1.0 - 0.5
        else:
            x = (torch.rand(M, TwoH) * 2.0 - 1.0).to(dt)
            w = None
            a = None
        q = torch.rand(1, H) * 2.0 - 1.0 if has_qs else None
        y, scale = dequant_swiglu_quant(
            x.npu(), w.npu() if w is not None else None, a.npu() if a is not None else None, q.npu() if q is not None else None, al
        )
        torch.npu.synchronize()
        ry, rs = _ref(x, w, a, q, al)
        diff = (y.cpu().int() - ry.int()).abs()
        y_ratio = (diff > 1).float().mean().item()
        scale_ok = torch.allclose(scale.cpu().float(), rs.float(), atol=1e-5, rtol=1e-3)
        assert y_ratio < 1e-3 and scale_ok, f"smoke failed: M={M} H={H} {dt} y_ratio={y_ratio} scale_ok={scale_ok}"
        print(
            f"[SMOKE] M={M} H={H} {str(dt).split('.')[-1]} ws={has_ws} qs={has_qs} al={al} y_maxdiff={diff.max().item()} scale_ok={scale_ok}"
        )
    print("Kernel Output Match!")
