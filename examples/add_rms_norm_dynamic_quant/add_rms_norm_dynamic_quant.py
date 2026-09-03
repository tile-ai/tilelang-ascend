# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""Add + RMSNorm + Dynamic Quantization fused kernel for Ascend NPU.

Operator semantics:
    xOut      = x1 + x2
    y_norm    = xOut / sqrt(mean(xOut^2) + eps) * gamma
    scaleOut  = amax(|y_norm|, dim=-1) / 127            (per-token, fp32)
    y         = clamp(round(y_norm / scaleOut), -127, 127).int8

Architecture: **2-pass** (pure Vector / AIV), with abs_max accumulated in pass 1.

Plan A optimization:
    - GM tensors use ORIGINAL shapes (no host-side F.pad / .contiguous())
    - Kernel handles boundary tiles internally via zero-fill + partial T.copy
      with dynamic slicing (T.min/T.if_then_else for valid_rows, valid_n)
    - Zero-overhead wrapper: no extra kernel launches for pad/unpad
"""

import logging

import torch
import tilelang
from tilelang import language as T

logger = logging.getLogger(__name__)

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

VEC_NUM = 2
INT8_ABS_MAX = 127.0
ABS_MIN = 1e-12


def _get_tiling(M, N, dtype):
    """Compute block_M, block_N and padded M, N."""
    UB_BUDGET = 192 * 1024
    cal_bytes = 4
    input_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    MIN_BLOCK_M = 16

    best = None
    best_score = float("inf")

    for bn in (1024, 512, 256, 128, 64, 32, 16):
        if bn >= N * 2:
            continue
        N_padded = ((N + bn - 1) // bn) * bn

        for bm in (32, 16, 64, 128):
            if bm < MIN_BLOCK_M or bm % VEC_NUM != 0:
                continue
            M_padded = ((M + bm - 1) // bm) * bm
            ROWS = bm // VEC_NUM
            tile_bytes = (
                2 * ROWS * bn * input_bytes
                + 6 * ROWS * bn * cal_bytes
                + 2 * ROWS * bn * (2 if dtype in ("float16", "bfloat16") else 1)
                + 1 * ROWS * bn * 4
                + 10 * ROWS * cal_bytes
            )
            if tile_bytes > UB_BUDGET * 0.95:
                continue

            score = -(ROWS * bn)

            import math as _math

            bn_log2 = _math.floor(_math.log2(max(bn, 1)))
            bn_vector_penalty = (9 - bn_log2) * 2

            padding_ratio_N = (N_padded - N) / max(N, 1)
            if padding_ratio_N > 0.5:
                score += 100
            else:
                padding_term = (M_padded + N_padded) * 0.001
                score = score + bn_vector_penalty + padding_term

            if best is None or score < best_score:
                best_score = score
                best = (bm, bn, M_padded, N_padded)

    if best is None:
        bm = MIN_BLOCK_M
        for bn in (256, 128, 64, 32, 16):
            ROWS = bm // VEC_NUM
            tile_bytes = 2 * ROWS * bn * input_bytes + 6 * ROWS * bn * cal_bytes + 10 * ROWS * cal_bytes
            if tile_bytes <= UB_BUDGET * 0.95:
                best = (bm, bn, ((M + bm - 1) // bm) * bm, ((N + bn - 1) // bn) * bn)
                break
        if best is None:
            best = (bm, 16, ((M + bm - 1) // bm) * bm, ((N + 16 - 1) // 16) * 16)
    return best


@tilelang.jit(out_idx=[3, 4, 5], pass_configs=pass_configs)
def _kernel_impl(M_orig, N_orig, M_padded, N_padded, block_M, block_N, eps=1e-6, dtype="float16"):
    """Low-level fused Add + RMSNorm + DynamicQuant kernel.

    Plan A with JIT compile-time dual-path dispatch:
    - is_aligned=true  (shape exactly divisible by block): FAST path, fixed-length T.copy, no overhead
    - is_aligned=false (boundary tiles exist): BOUNDARY path, zero-fill + partial T.copy

    Each shape compiles only one IR branch; no runtime cost for the dispatch.
    """
    m_num = M_padded // block_M
    n_num = N_padded // block_N
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N

    # JIT compile-time flag: computed in Python, captured by closure, evaluated at codegen
    is_aligned = (M_orig == M_padded) and (N_orig == N_padded)

    acc_dtype = "float32"
    need_cast = dtype not in ("float", acc_dtype)
    CAST_MODE = "CAST_NONE"

    @T.prim_func
    def tilelang_add_rms_norm_dynamic_quant(
        x1: T.Tensor((M_orig, N_orig), dtype),
        x2: T.Tensor((M_orig, N_orig), dtype),
        gamma: T.Tensor((N_orig,), dtype),
        y: T.Tensor((M_orig, N_orig), "int8"),
        xOut: T.Tensor((M_orig, N_orig), dtype),
        scaleOut: T.Tensor((M_orig,), "float32"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # --- UB Buffer Allocations (shared across both paths) ---
            in0_ub = T.alloc_ub([ROWS, block_N], dtype)
            in1_ub = T.alloc_ub([ROWS, block_N], dtype)
            xOut_fp32 = T.alloc_ub([ROWS, block_N], acc_dtype)
            tmp_fp32 = T.alloc_ub([ROWS, block_N], acc_dtype)
            xOut_out_cast = T.alloc_ub([ROWS, block_N], dtype)

            sq_acc = T.alloc_ub([ROWS, block_N], acc_dtype)
            abs_max_xg = T.alloc_ub([ROWS, block_N], acc_dtype)

            sq_row = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            sqrt_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            abs_max_xg_row = T.alloc_ub([ROWS, 1], acc_dtype)
            scale_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            abs_min_sc = T.alloc_ub([ROWS, 1], acc_dtype)
            scalar_ub = T.alloc_ub([ROWS, 1], acc_dtype)

            combined_tile = T.alloc_ub([ROWS, block_N], acc_dtype)

            gamma_1d = T.alloc_ub([block_N], dtype)
            gamma_fp1d = T.alloc_ub([block_N], acc_dtype)
            gamma_bc = T.alloc_ub([ROWS, block_N], acc_dtype)

            y_q_fp32 = T.alloc_ub([ROWS, block_N], acc_dtype)
            y_clamped = T.alloc_ub([ROWS, block_N], acc_dtype)
            y_rounded = T.alloc_ub([ROWS, block_N], acc_dtype)
            y_fp16 = T.alloc_ub([ROWS, block_N], "float16")
            y_i8 = T.alloc_ub([ROWS, block_N], "int8")

            scale_out_1d = T.alloc_ub([ROWS], acc_dtype)

            with T.Scope("V"):
                row_start = cid * block_M + vid * ROWS

                # =============================================================
                # PASS 1: sq_sum + amax(|xOut*gamma|) + write xOut
                # =============================================================
                T.tile.fill(sq_acc, 0.0)
                T.tile.fill(abs_max_xg, 0.0)

                if is_aligned:
                    # --- FAST PATH: fixed-length T.copy, no zero-fill overhead ---
                    for by in T.serial(n_num):
                        col_off = by * block_N
                        T.copy(x1[row_start : row_start + ROWS, col_off : col_off + block_N], in0_ub)
                        T.copy(x2[row_start : row_start + ROWS, col_off : col_off + block_N], in1_ub)
                        if need_cast:
                            T.tile.cast(xOut_fp32, in0_ub, mode=CAST_MODE, count=tile_elements)
                            T.tile.cast(tmp_fp32, in1_ub, mode=CAST_MODE, count=tile_elements)
                        else:
                            T.copy(in0_ub, xOut_fp32)
                            T.copy(in1_ub, tmp_fp32)
                        T.tile.add(xOut_fp32, xOut_fp32, tmp_fp32)
                        T.tile.mul_add_dst(sq_acc, xOut_fp32, xOut_fp32)
                        if need_cast:
                            T.tile.cast(xOut_out_cast, xOut_fp32, mode="CAST_RINT", count=tile_elements)
                            T.copy(
                                xOut_out_cast,
                                xOut[row_start : row_start + ROWS, col_off : col_off + block_N],
                            )
                        else:
                            T.copy(
                                xOut_fp32,
                                xOut[row_start : row_start + ROWS, col_off : col_off + block_N],
                            )
                        T.copy(gamma[col_off : col_off + block_N], gamma_1d)
                        if need_cast:
                            T.tile.cast(gamma_fp1d, gamma_1d, mode=CAST_MODE, count=block_N)
                        else:
                            T.copy(gamma_1d, gamma_fp1d)
                        T.tile.broadcast(gamma_bc, gamma_fp1d)
                        T.tile.mul(tmp_fp32, xOut_fp32, gamma_bc)
                        T.tile.abs(xOut_fp32, tmp_fp32)
                        T.tile.max(abs_max_xg, abs_max_xg, xOut_fp32)
                else:
                    # --- BOUNDARY PATH PASS 1: zero-fill + partial T.copy ---
                    # Compute M-direction valid_rows for this scope
                    remaining_rows = M_orig - cid * block_M
                    vid_offset = vid * ROWS
                    vid_remaining = T.if_then_else(
                        remaining_rows > vid_offset,
                        remaining_rows - vid_offset,
                        0,
                    )
                    valid_rows = T.min(ROWS, vid_remaining)

                    for by in T.serial(n_num):
                        col_off = by * block_N
                        # N-direction boundary: compute valid columns
                        valid_n = T.min(block_N, N_orig - col_off)

                        # Zero-fill + partial copy for boundary handling
                        T.tile.fill(in0_ub, 0.0)
                        T.tile.fill(in1_ub, 0.0)
                        T.copy(
                            x1[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            in0_ub[0:valid_rows, 0:valid_n],
                        )
                        T.copy(
                            x2[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            in1_ub[0:valid_rows, 0:valid_n],
                        )
                        if need_cast:
                            T.tile.cast(xOut_fp32, in0_ub, mode=CAST_MODE, count=tile_elements)
                            T.tile.cast(tmp_fp32, in1_ub, mode=CAST_MODE, count=tile_elements)
                        else:
                            T.copy(in0_ub, xOut_fp32)
                            T.copy(in1_ub, tmp_fp32)
                        T.tile.add(xOut_fp32, xOut_fp32, tmp_fp32)
                        T.tile.mul_add_dst(sq_acc, xOut_fp32, xOut_fp32)
                        if need_cast:
                            T.tile.cast(xOut_out_cast, xOut_fp32, mode="CAST_RINT", count=tile_elements)
                            T.copy(
                                xOut_out_cast[0:valid_rows, 0:valid_n],
                                xOut[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            )
                        else:
                            T.copy(
                                xOut_fp32[0:valid_rows, 0:valid_n],
                                xOut[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            )
                        # Load gamma with N-boundary handling
                        T.tile.fill(gamma_1d, 0.0)
                        T.copy(gamma[col_off : col_off + valid_n], gamma_1d[0:valid_n])
                        if need_cast:
                            T.tile.cast(gamma_fp1d, gamma_1d, mode=CAST_MODE, count=block_N)
                        else:
                            T.copy(gamma_1d, gamma_fp1d)
                        T.tile.broadcast(gamma_bc, gamma_fp1d)
                        T.tile.mul(tmp_fp32, xOut_fp32, gamma_bc)
                        T.tile.abs(xOut_fp32, tmp_fp32)
                        T.tile.max(abs_max_xg, abs_max_xg, xOut_fp32)

                # =============================================================
                # INTER-PASS: reduce, compute inv_rms, compute scale (common)
                # =============================================================
                T.reduce_sum(sq_acc, sq_row, dim=-1)
                inv_n = T.cast(1.0 / N_orig, acc_dtype)
                eps_val = T.cast(eps, acc_dtype)
                T.tile.mul(sq_row, sq_row, inv_n)
                T.tile.add(sq_row, sq_row, eps_val)
                T.tile.sqrt(sqrt_ub, sq_row)
                T.tile.fill(scalar_ub, 1.0)
                T.tile.div(inv_rms_ub, scalar_ub, sqrt_ub)

                T.reduce_max(abs_max_xg, abs_max_xg_row, dim=-1)
                T.tile.mul(abs_max_xg_row, abs_max_xg_row, inv_rms_ub)
                T.tile.fill(abs_min_sc, ABS_MIN)
                T.tile.max(abs_max_xg_row, abs_max_xg_row, abs_min_sc)
                T.tile.fill(scalar_ub, INT8_ABS_MAX)
                T.tile.div(scale_ub, abs_max_xg_row, scalar_ub)

                T.tile.div(inv_rms_ub, inv_rms_ub, scale_ub)
                T.tile.broadcast(combined_tile, inv_rms_ub)

                # =============================================================
                # PASS 2: normalize + quantize, write y, scaleOut
                # =============================================================
                if is_aligned:
                    # --- FAST PATH ---
                    for by in T.serial(n_num):
                        col_off = by * block_N
                        T.copy(x1[row_start : row_start + ROWS, col_off : col_off + block_N], in0_ub)
                        T.copy(x2[row_start : row_start + ROWS, col_off : col_off + block_N], in1_ub)
                        if need_cast:
                            T.tile.cast(xOut_fp32, in0_ub, mode=CAST_MODE, count=tile_elements)
                            T.tile.cast(tmp_fp32, in1_ub, mode=CAST_MODE, count=tile_elements)
                        else:
                            T.copy(in0_ub, xOut_fp32)
                            T.copy(in1_ub, tmp_fp32)
                        T.tile.add(xOut_fp32, xOut_fp32, tmp_fp32)
                        T.copy(gamma[col_off : col_off + block_N], gamma_1d)
                        if need_cast:
                            T.tile.cast(gamma_fp1d, gamma_1d, mode=CAST_MODE, count=block_N)
                        else:
                            T.copy(gamma_1d, gamma_fp1d)
                        T.tile.broadcast(gamma_bc, gamma_fp1d)
                        T.tile.mul(y_q_fp32, xOut_fp32, gamma_bc)
                        T.tile.mul(y_q_fp32, y_q_fp32, combined_tile)
                        T.tile.clamp(y_clamped, y_q_fp32, -INT8_ABS_MAX, INT8_ABS_MAX, tile_elements)
                        T.tile.round(y_rounded, y_clamped, tile_elements)
                        T.tile.cast(y_fp16, y_rounded, mode="CAST_RINT", count=tile_elements)
                        T.tile.cast(y_i8, y_fp16, mode=CAST_MODE, count=tile_elements)
                        T.copy(y_i8, y[row_start : row_start + ROWS, col_off : col_off + block_N])
                else:
                    # --- BOUNDARY PATH PASS 2 ---
                    # Recompute valid_rows for this scope (recomputed per dispatch block)
                    remaining_rows = M_orig - cid * block_M
                    vid_offset = vid * ROWS
                    vid_remaining = T.if_then_else(
                        remaining_rows > vid_offset,
                        remaining_rows - vid_offset,
                        0,
                    )
                    valid_rows = T.min(ROWS, vid_remaining)

                    for by in T.serial(n_num):
                        col_off = by * block_N
                        valid_n = T.min(block_N, N_orig - col_off)

                        T.tile.fill(in0_ub, 0.0)
                        T.tile.fill(in1_ub, 0.0)
                        T.copy(
                            x1[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            in0_ub[0:valid_rows, 0:valid_n],
                        )
                        T.copy(
                            x2[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                            in1_ub[0:valid_rows, 0:valid_n],
                        )
                        if need_cast:
                            T.tile.cast(xOut_fp32, in0_ub, mode=CAST_MODE, count=tile_elements)
                            T.tile.cast(tmp_fp32, in1_ub, mode=CAST_MODE, count=tile_elements)
                        else:
                            T.copy(in0_ub, xOut_fp32)
                            T.copy(in1_ub, tmp_fp32)
                        T.tile.add(xOut_fp32, xOut_fp32, tmp_fp32)
                        T.tile.fill(gamma_1d, 0.0)
                        T.copy(gamma[col_off : col_off + valid_n], gamma_1d[0:valid_n])
                        if need_cast:
                            T.tile.cast(gamma_fp1d, gamma_1d, mode=CAST_MODE, count=block_N)
                        else:
                            T.copy(gamma_1d, gamma_fp1d)
                        T.tile.broadcast(gamma_bc, gamma_fp1d)
                        T.tile.mul(y_q_fp32, xOut_fp32, gamma_bc)
                        T.tile.mul(y_q_fp32, y_q_fp32, combined_tile)
                        T.tile.clamp(y_clamped, y_q_fp32, -INT8_ABS_MAX, INT8_ABS_MAX, tile_elements)
                        T.tile.round(y_rounded, y_clamped, tile_elements)
                        T.tile.cast(y_fp16, y_rounded, mode="CAST_RINT", count=tile_elements)
                        T.tile.cast(y_i8, y_fp16, mode=CAST_MODE, count=tile_elements)
                        T.copy(
                            y_i8[0:valid_rows, 0:valid_n],
                            y[row_start : row_start + valid_rows, col_off : col_off + valid_n],
                        )

                # --- Write scaleOut ---
                T.copy(scale_ub[:, 0], scale_out_1d)
                if is_aligned:
                    # FAST PATH: fixed-length write
                    T.copy(scale_out_1d, scaleOut[row_start : row_start + ROWS])
                else:
                    # BOUNDARY PATH: recompute valid_rows for this scope
                    remaining_rows = M_orig - cid * block_M
                    vid_offset = vid * ROWS
                    vid_remaining = T.if_then_else(
                        remaining_rows > vid_offset,
                        remaining_rows - vid_offset,
                        0,
                    )
                    valid_rows = T.min(ROWS, vid_remaining)
                    T.copy(
                        scale_out_1d[0:valid_rows],
                        scaleOut[row_start : row_start + valid_rows],
                    )

    return tilelang_add_rms_norm_dynamic_quant


# =============================================================================
# Public wrapper
# =============================================================================

_str_dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Fused Add + RMSNorm + DynamicQuant.

    Plan A: Zero wrapper overhead.
    - No F.pad for inputs (kernel handles boundaries internally)
    - No narrow+contiguous for outputs (kernel writes original-shape tensors directly)
    - Only reshape for >2D inputs (view-only, zero-copy)
    """
    assert x1.shape == x2.shape and x1.shape[-1] == gamma.shape[-1]
    assert x1.dtype == x2.dtype == gamma.dtype

    M_flat = int(x1.shape[:-1].numel()) if x1.dim() > 2 else x1.shape[0]
    N = x1.shape[-1]
    dtype_str = _str_dtype_map[x1.dtype]

    x1_2d = x1.reshape(M_flat, N)
    x2_2d = x2.reshape(M_flat, N)

    block_M, block_N, M_padded, N_padded = _get_tiling(M_flat, N, dtype_str)

    # Pass original shapes to kernel; it handles boundary tiles internally
    impl = _kernel_impl(M_flat, N, M_padded, N_padded, block_M, block_N, epsilon, dtype_str)
    y, xOut, scale = impl(x1_2d, x2_2d, gamma)

    if x1.dim() > 2:
        y = y.reshape(*x1.shape)
        xOut = xOut.reshape(*x1.shape)
        scale = scale.reshape(*x1.shape[:-1])

    return y, xOut, scale


# =============================================================================
# Golden reference + tests
# =============================================================================


def golden_add_rms_norm_dynamic_quant(x1, x2, gamma, epsilon=1e-6):
    out_dtype = x1.dtype
    x1_f = x1.to(torch.float32)
    x2_f = x2.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    xOut = x1_f + x2_f
    variance = xOut.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(variance + epsilon)
    y_norm = xOut / rms * gamma_f
    abs_max = y_norm.abs().amax(dim=-1, keepdim=True)
    scale_out = (abs_max.clamp(min=1e-12) / 127.0).to(torch.float32)
    y = torch.clamp((y_norm / scale_out).round(), -128, 127).to(torch.int8)
    scale = scale_out.squeeze(-1)
    return y, xOut.to(out_dtype), scale


if __name__ == "__main__":
    torch.manual_seed(0)
    test_configs = [
        (256, 256, torch.float16),
        (512, 512, torch.float16),
        (1024, 2048, torch.bfloat16),
        (4096, 4096, torch.bfloat16),
        (8192, 2048, torch.bfloat16),
        # Padding test cases
        (1021, 1023, torch.bfloat16),  # Case 16
        (4093, 4093, torch.bfloat16),  # Case 12
        (4093, 2053, torch.float16),  # Case 13
    ]
    for M, N, dt in test_configs:
        print(f"Testing M={M}, N={N}, dtype={dt}")
        x1 = torch.randn(M, N, device="npu", dtype=dt)
        x2 = torch.randn(M, N, device="npu", dtype=dt)
        g = torch.randn(N, device="npu", dtype=dt)
        y, xOut, scale = add_rms_norm_dynamic_quant(x1, x2, g)
        y_ref, xOut_ref, scale_ref = golden_add_rms_norm_dynamic_quant(x1, x2, g)
        torch.testing.assert_close(y.cpu(), y_ref.cpu(), atol=2, rtol=0.02)
        torch.testing.assert_close(xOut.cpu(), xOut_ref.cpu(), rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(scale.cpu(), scale_ref.cpu(), rtol=2e-3, atol=2e-3)
        print("  PASS")
    print("\nKERNEL OUTPUT MATCH")
    print("TEST PASSED!")
