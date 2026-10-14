# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# DequantSwigluQuant Fusion Kernel
#
# Fuses: dequant -> SwiGLU activation -> optional smooth quant -> dynamic per-token INT8 quantization
# Supports: fp16/bf16 (no weight_scale) and int32 W8A8 (with weight_scale + activation_scale)
#
# Optimizations applied:
#   - Double Buffer with manual MTE2/V/MTE3 three-way flag pipeline (AUTO_SYNC=False)
#   - Multi-row Tile with 1D reduce_max (guide section 2.13)
#   - UB budget formula for dynamic block_H (guide section 2.11)
#   - Kernel-side pad_value + remainder block for non-aligned H (guide section 2.12)
#   - T.tile.clamp instruction fusion (guide section 2.7)
#   - Output tensors declared with H_orig (no H padding) to ensure contiguous output

import argparse
import sys
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

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


def bench_us(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    torch.npu.synchronize()
    return start.elapsed_time(end) / repeat * 1000.0


# =============================================================================
# PyTorch Golden Reference
# =============================================================================
def golden_dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    activate_left: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation of dequant_swiglu_quant.

    Fuses: dequant -> SwiGLU -> optional smooth quant -> dynamic per-token INT8 quant
    """
    if x.dtype == torch.int32:
        dequant_out = x.float() * weight_scale.float()
        dequant_out = dequant_out * activation_scale.float().unsqueeze(-1)
    else:
        dequant_out = x.float()

    half = dequant_out.shape[-1] // 2
    A = dequant_out[..., :half]
    B = dequant_out[..., half:]
    swiglu_out = F.silu(A) * B if activate_left else F.silu(B) * A

    if quant_scale is not None:
        swiglu_out = swiglu_out * quant_scale.float()

    max_per_row = swiglu_out.abs().amax(dim=-1)
    s = (max_per_row.float() / 127.0).clamp_min(1e-12)
    y = torch.clamp((swiglu_out.float() / s.unsqueeze(-1)).round(), -128, 127).to(torch.int8)
    scale = s.to(torch.float32)
    return y, scale


# =============================================================================
# Kernel Construction
# =============================================================================
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
    M_orig, TwoH = x.shape
    H_orig = TwoH // 2

    if x.dtype == torch.int32:
        in_dtype = "int32"
        has_ws = True
    elif x.dtype == torch.float16:
        in_dtype = "float16"
        has_ws = False
    elif x.dtype == torch.bfloat16:
        in_dtype = "bfloat16"
        has_ws = False
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


# =============================================================================
# Precision Check (mixed tolerance per precision-standard.md)
# =============================================================================
def precision_compare(actual_y, golden_y, actual_scale, golden_scale):
    """Mixed tolerance precision comparison.

    y (int8 quantized output):
        |diff| > 1 的元素占比 < 1e-3 即通过; 同时记录 max_diff.

    scale (float32, 值域分段混合容差 per precision-standard.md):
        - 特殊值 INF/NAN (§2.1): 验证 isinf/isnan 状态一致, 不验证数值误差.
        - 小值域 |expected| < 1e-5 (§2.2): atol=1e-7, rtol=0, 仅验证绝对误差.
        - 正常值域 |expected| >= 1e-5: atol=1e-5, rtol=1e-3.
    """
    # --- y: int8 quantized output, |diff|>1 ratio check ---
    diff = (actual_y.int() - golden_y.int()).abs()
    ratio = (diff > 1).float().mean().item()
    max_diff = diff.max().item()
    y_ok = ratio < 1e-3

    # --- scale: float32 value-domain segmented mixed tolerance ---
    a = actual_scale.float()
    e = golden_scale.float()
    abs_err = (a - e).abs()
    abs_exp = e.abs()

    # Special values INF/NAN (precision-standard.md §2.1)
    a_inf = torch.isinf(a)
    e_inf = torch.isinf(e)
    a_nan = torch.isnan(a)
    e_nan = torch.isnan(e)
    special_mask = a_inf | e_inf | a_nan | e_nan
    special_ok = ((a_inf == e_inf) & (a_nan == e_nan)) | (~special_mask)

    # Normal value positions (non-special)
    normal_mask = ~special_mask

    # Small value domain (§2.2): |expected| < 1e-5 -> atol=1e-7, rtol=0
    small_mask = normal_mask & (abs_exp < 1e-5)
    small_ok = (abs_err <= 1e-7) | (~small_mask)

    # Normal value domain: |expected| >= 1e-5 -> atol=1e-5, rtol=1e-3
    normal_value_mask = normal_mask & (abs_exp >= 1e-5)
    normal_tol = 1e-5 + 1e-3 * abs_exp
    normal_ok = (abs_err <= normal_tol) | (~normal_value_mask)

    scale_ok = special_ok.all().item() and small_ok.all().item() and normal_ok.all().item()

    passed = y_ok and scale_ok
    return {
        "passed": passed,
        "y_ratio": ratio,
        "y_maxdiff": max_diff,
        "scale_ok": scale_ok,
    }


# =============================================================================
# Test Runner
# =============================================================================
def _run_case(name, M, H, x_dtype, has_ws, has_qs, activate_left, level):
    """Run a single test case.

    Args:
        name: case name (e.g. "L0-1")
        M, H: matrix dimensions
        x_dtype: torch dtype
        has_ws, has_qs, activate_left: attribute flags
        level: "L0"/"L1"/"L2"/"Boundary"

    Returns:
        True if passed
    """
    torch.manual_seed(42)
    TwoH = 2 * H

    if x_dtype == torch.int32:
        x = torch.randint(-128, 128, (M, TwoH), dtype=torch.int32, device="npu")
    else:
        x = (torch.rand(M, TwoH, device="npu") * 2.0 - 1.0).to(x_dtype)

    ws = None
    as_ = None
    qs = None
    if has_ws:
        ws = (torch.rand(1, TwoH, device="npu") * 0.2 - 0.1).float()
        as_ = (torch.rand(M, device="npu") * 1.0 - 0.5).float()
    if has_qs:
        qs = (torch.rand(1, H, device="npu") * 2.0 - 1.0).float()

    dtype_str = str(x_dtype).split(".")[-1]
    try:
        y, scale = dequant_swiglu_quant(x, ws, as_, qs, activate_left)
        torch.npu.synchronize()

        ref_y, ref_scale = golden_dequant_swiglu_quant(x, ws, as_, qs, activate_left)
        result = precision_compare(y.cpu(), ref_y.cpu(), scale.cpu(), ref_scale.cpu())

        if level in ("L0", "L1"):
            tag = "[PRECISION_PASS]" if result["passed"] else "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_PASS]" if result["passed"] else "[BOUNDARY_WARN]"

        print(
            f"{tag} {level}/{name}: M={M} H={H} dtype={dtype_str} ws={has_ws} qs={has_qs} "
            f"al={activate_left} | y_ratio={result['y_ratio']:.4e} y_maxdiff={result['y_maxdiff']} "
            f"scale_ok={result['scale_ok']}"
        )
        return result["passed"]
    except Exception as e:
        if level in ("L0", "L1"):
            tag = "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_WARN]"
        print(f"{tag} {level}/{name}: M={M} H={H} dtype={dtype_str} ws={has_ws} qs={has_qs} al={activate_left}: {e}")
        return False


# =============================================================================
# L0 Threshold Tests
# =============================================================================
def test_l0():
    """L0 threshold tests: aligned shapes, all 3 dtypes, all attribute combos."""
    print("=== L0 ===")
    cases = [
        ("l0_fp16_base", 1024, 2048, torch.float16, False, False, False),
        ("l0_bf16_quant_scale", 1024, 2048, torch.bfloat16, False, True, False),
        ("l0_int32_w8a8", 512, 1024, torch.int32, True, False, True),
        ("l0_int32_full_attrs", 1024, 2048, torch.int32, True, True, False),
    ]
    passed = 0
    for name, M, H, dt, ws, qs, al in cases:
        if _run_case(name, M, H, dt, ws, qs, al, "L0"):
            passed += 1
    print(f"\n[L0] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# =============================================================================
# L1 Functional Tests
# =============================================================================
def test_l1():
    """L1 functional tests: prime H, odd M, large M, non-aligned shapes."""
    print("=== L1 ===")
    cases = [
        ("l1_prime_h_bf16", 1023, 2049, torch.bfloat16, False, False, True),
        ("l1_prime_h_int32", 255, 4097, torch.int32, True, False, False),
        ("l1_prime_h_int32_qs", 255, 4097, torch.int32, True, True, False),
        ("l1_large_m", 10007, 64, torch.int32, True, False, False),
        ("l1_prime_m_qs", 4001, 2048, torch.int32, True, True, True),
        ("l1_small_m", 127, 1024, torch.bfloat16, False, False, False),
    ]
    passed = 0
    for name, M, H, dt, ws, qs, al in cases:
        if _run_case(name, M, H, dt, ws, qs, al, "L1"):
            passed += 1
    print(f"\n[L1] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# =============================================================================
# L2 Exception Tests (non-blocking)
# =============================================================================
def test_l2():
    """L2 exception tests: unsupported dtype, shape mismatch, None input.

    L2 failures produce [BOUNDARY_WARN] and do not block exit code.
    """
    print("=== L2 ===")

    def case_unsupported_dtype():
        x = torch.randn(64, 256, dtype=torch.float32, device="npu")
        try:
            dequant_swiglu_quant(x)
        except ValueError:
            return True
        return False

    def case_none_input():
        try:
            dequant_swiglu_quant(None)
        except (TypeError, AttributeError, ValueError):
            return True
        return False

    tests = [
        ("unsupported_dtype_float32", case_unsupported_dtype),
        ("none_input", case_none_input),
    ]
    passed = 0
    for name, fn in tests:
        try:
            ok = fn()
            tag = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_WARN]"
            print(f"{tag} L2/{name}")
            if ok:
                passed += 1
        except Exception as e:
            print(f"[BOUNDARY_WARN] L2/{name}: {e}")
    print(f"\n[L2] Summary: {passed}/{len(tests)} passed (warnings do not block)")


# =============================================================================
# Boundary Tests (non-blocking)
# =============================================================================
def test_boundary():
    """Boundary tests: all-zero, extreme values, INF/NAN input.

    Boundary failures produce [BOUNDARY_WARN] and do not block exit code.
    """
    print("=== Boundary ===")
    cases = [
        ("all_zero", 64, 128, torch.float16, False, False, False),
        ("extreme_int32", 64, 128, torch.int32, True, False, False),
        ("inf_input", 64, 128, torch.float16, False, False, False),
        ("nan_input", 64, 128, torch.float16, False, False, False),
    ]
    passed = 0
    for name, M, H, dt, ws, qs, al in cases:
        torch.manual_seed(42)
        TwoH = 2 * H
        try:
            if name == "all_zero":
                x = torch.zeros(M, TwoH, dtype=dt, device="npu")
            elif name == "extreme_int32":
                x = torch.randint(-128, 128, (M, TwoH), dtype=torch.int32, device="npu")
                x[0, :] = 128
                x[1, :] = -128
            elif name == "inf_input":
                x = (torch.rand(M, TwoH, device="npu") * 2.0 - 1.0).to(torch.float16)
                x[0, 0] = float("inf")
            elif name == "nan_input":
                x = (torch.rand(M, TwoH, device="npu") * 2.0 - 1.0).to(torch.float16)
                x[0, 0] = float("nan")

            w = (torch.rand(1, TwoH, device="npu") * 0.2 - 0.1).float() if ws else None
            a = (torch.rand(M, device="npu") * 1.0 - 0.5).float() if ws else None
            q = (torch.rand(1, H, device="npu") * 2.0 - 1.0).float() if qs else None

            y, scale = dequant_swiglu_quant(x, w, a, q, al)
            torch.npu.synchronize()
            ref_y, ref_scale = golden_dequant_swiglu_quant(x, w, a, q, al)
            result = precision_compare(y.cpu(), ref_y.cpu(), scale.cpu(), ref_scale.cpu())
            tag = "[BOUNDARY_PASS]" if result["passed"] else "[BOUNDARY_WARN]"
            print(f"{tag} Boundary/{name}: y_ratio={result['y_ratio']:.4e} scale_ok={result['scale_ok']}")
            if result["passed"]:
                passed += 1
        except Exception as e:
            print(f"[BOUNDARY_WARN] Boundary/{name}: {e}")
    print(f"\n[Boundary] Summary: {passed}/{len(cases)} passed (warnings do not block)")


# =============================================================================
# CANN-Bench 20 Cases
# =============================================================================
def test_cann_bench():
    """cann-bench level3/dequant_swiglu_quant 20 cases."""
    print("=== cann-bench ===")
    cases = [
        ("cann-bench-1", 512, 2048, torch.float16, False, False, True),
        ("cann-bench-2", 1024, 4096, torch.float16, False, False, False),
        ("cann-bench-3", 2048, 8192, torch.float16, False, False, True),
        ("cann-bench-4", 4096, 4096, torch.bfloat16, False, False, False),
        ("cann-bench-5", 127, 1024, torch.bfloat16, False, False, False),
        ("cann-bench-6", 8192, 1024, torch.float16, False, False, False),
        ("cann-bench-7", 1023, 2049, torch.bfloat16, False, False, True),
        ("cann-bench-8", 255, 4097, torch.bfloat16, False, False, False),
        ("cann-bench-9", 512, 2048, torch.int32, True, False, True),
        ("cann-bench-10", 1024, 4096, torch.int32, True, False, False),
        ("cann-bench-11", 2048, 8192, torch.int32, True, False, True),
        ("cann-bench-12", 4096, 4096, torch.int32, True, False, False),
        ("cann-bench-13", 127, 1024, torch.int32, True, False, False),
        ("cann-bench-14", 8192, 1024, torch.int32, True, False, False),
        ("cann-bench-15", 1023, 2049, torch.int32, True, False, True),
        ("cann-bench-16", 255, 4097, torch.int32, True, True, False),
        ("cann-bench-17", 10007, 64, torch.int32, True, False, False),
        ("cann-bench-18", 32768, 256, torch.int32, True, False, False),
        ("cann-bench-19", 4001, 2048, torch.int32, True, True, True),
        ("cann-bench-20", 16384, 512, torch.int32, True, False, False),
    ]
    passed = 0
    for name, M, H, dt, ws, qs, al in cases:
        if _run_case(name, M, H, dt, ws, qs, al, "L0"):
            passed += 1
    print(f"\n[cann-bench] Summary: {passed}/{len(cases)} passed")
    return passed == len(cases)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DequantSwigluQuant operator tests")
    parser.add_argument(
        "--level",
        default="l0",
        type=str.lower,
        choices=["l0", "l1", "l2", "boundary", "cann-bench", "all"],
        help="Test level: L0 (threshold), L1 (functional), L2 (exception), "
        "Boundary (edge cases), cann-bench (20 official cases), all (full suite)",
    )
    parser.add_argument("--case", type=int, default=0, help="Run only the Nth case in the level (1-indexed)")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.case > 0:
        # Single-case mode: run only the Nth case of the specified level
        level = args.level.upper() if args.level != "cann-bench" else "L0"
        case_map = {
            "l0": [
                ("l0_fp16_base", 1024, 2048, torch.float16, False, False, False),
                ("l0_bf16_quant_scale", 1024, 2048, torch.bfloat16, False, True, False),
                ("l0_int32_w8a8", 512, 1024, torch.int32, True, False, True),
                ("l0_int32_full_attrs", 1024, 2048, torch.int32, True, True, False),
            ],
            "l1": [
                ("l1_prime_h_bf16", 1023, 2049, torch.bfloat16, False, False, True),
                ("l1_prime_h_int32", 255, 4097, torch.int32, True, False, False),
                ("l1_prime_h_int32_qs", 255, 4097, torch.int32, True, True, False),
                ("l1_large_m", 10007, 64, torch.int32, True, False, False),
                ("l1_prime_m_qs", 4001, 2048, torch.int32, True, True, True),
                ("l1_small_m", 127, 1024, torch.bfloat16, False, False, False),
            ],
            "cann-bench": [
                ("cann-bench-1", 512, 2048, torch.float16, False, False, True),
                ("cann-bench-2", 1024, 4096, torch.float16, False, False, False),
                ("cann-bench-3", 2048, 8192, torch.float16, False, False, True),
                ("cann-bench-4", 4096, 4096, torch.bfloat16, False, False, False),
                ("cann-bench-5", 127, 1024, torch.bfloat16, False, False, False),
                ("cann-bench-6", 8192, 1024, torch.float16, False, False, False),
                ("cann-bench-7", 1023, 2049, torch.bfloat16, False, False, True),
                ("cann-bench-8", 255, 4097, torch.bfloat16, False, False, False),
                ("cann-bench-9", 512, 2048, torch.int32, True, False, True),
                ("cann-bench-10", 1024, 4096, torch.int32, True, False, False),
                ("cann-bench-11", 2048, 8192, torch.int32, True, False, True),
                ("cann-bench-12", 4096, 4096, torch.int32, True, False, False),
                ("cann-bench-13", 127, 1024, torch.int32, True, False, False),
                ("cann-bench-14", 8192, 1024, torch.int32, True, False, False),
                ("cann-bench-15", 1023, 2049, torch.int32, True, False, True),
                ("cann-bench-16", 255, 4097, torch.int32, True, True, False),
                ("cann-bench-17", 10007, 64, torch.int32, True, False, False),
                ("cann-bench-18", 32768, 256, torch.int32, True, False, False),
                ("cann-bench-19", 4001, 2048, torch.int32, True, True, True),
                ("cann-bench-20", 16384, 512, torch.int32, True, False, False),
            ],
        }
        cases = case_map.get(args.level, case_map["l0"])
        if args.case > len(cases):
            print(f"Error: --case {args.case} out of range (1-{len(cases)} for {args.level})")

            sys.exit(1)
        name, M, H, dt, ws, qs, al = cases[args.case - 1]
        ok = _run_case(name, M, H, dt, ws, qs, al, level)

        sys.exit(0 if ok else 1)

    blocking_ok = True

    if args.level in ("l0", "all"):
        blocking_ok &= test_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_l1()
    if args.level in ("l2", "all"):
        test_l2()
    if args.level in ("boundary", "all"):
        test_boundary()
    if args.level in ("cann-bench", "all"):
        blocking_ok &= test_cann_bench()

    if args.bench:
        print("\n=== Benchmark ===")
        bench_M, bench_H = 1024, 4096
        x = (torch.rand(bench_M, 2 * bench_H, device="npu") * 2.0 - 1.0).to(torch.float16)

        def run_tilelang():
            dequant_swiglu_quant(x)

        def run_torch():
            golden_dequant_swiglu_quant(x)

        tile_us = bench_us(run_tilelang, warmup=10, repeat=100)
        torch_us = bench_us(run_torch, warmup=10, repeat=100)
        print(f"shape: [{bench_M}, {2 * bench_H}] float16")
        print(f"tilelang: {tile_us:.2f} us")
        print(f"torch baseline: {torch_us:.2f} us")
        print(f"speedup: {torch_us / tile_us:.3f}x")

    if blocking_ok:
        print("\nTest Passed!")

        sys.exit(0)
    else:
        print("\nTest Failed!")

        sys.exit(1)


if __name__ == "__main__":
    main()
