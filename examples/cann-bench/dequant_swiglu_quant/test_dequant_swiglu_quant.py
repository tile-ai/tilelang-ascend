# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for dequant_swiglu_quant.

L0/L1 are built from the 20 cann-bench level3/dequant_swiglu_quant cases
(transcribed from cann-bench/tasks/level3/dequant_swiglu_quant/cases.yaml
per the proto.yaml contract x [M, 2H] -> y [M, H], quant_scale [1, H]),
redistributed by case nature (dispatchable via --level; all levels also
run under pytest where a failure raises AssertionError):

  L0        gate: one case per dispatch variant — fp16 / bf16+qs /
            int32+ws / int32+ws+qs (dtype x quant_scale x activate_left)
  L1        functional: the remaining 16 official cases (qs sweeps,
            unaligned M/H, large-M-small-H, [32768,256]/[16384,512])
  L2        negative: invalid inputs must be rejected by the wrapper
  Boundary  special values: all-zero / int32 extremes (+128/-128) /
            inf / NaN inputs (cann-bench has no special-value cases
            for this operator)

Judgement: mixed tolerance per precision-standard.md — y (int8) allows
|diff| > 1 on < 0.1% of elements; scale (fp32) uses value-domain
segmented tolerances with INF/NAN state matching.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dequant_swiglu_quant import dequant_swiglu_quant  # noqa: E402


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
            tag = "[BOUNDARY_PASS]" if result["passed"] else "[BOUNDARY_FAIL]"

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
            tag = "[BOUNDARY_FAIL]"
        print(f"{tag} {level}/{name}: M={M} H={H} dtype={dtype_str} ws={has_ws} qs={has_qs} al={activate_left}: {e}")
        return False


# =============================================================================
# L0 Threshold Tests
# =============================================================================

# ========== cann-bench case tables (20 cases, redistributed by nature) ==========
# (cann-bench case id, M, H, dtype, has_ws, has_qs, activate_left)
L0_CASES = [
    (1, 512, 1024, torch.float16, False, False, True),  # fp16 baseline
    (4, 4096, 2048, torch.bfloat16, False, True, False),  # bf16 + qs
    (9, 512, 1024, torch.int32, True, False, True),  # int32 + ws/as
    (11, 2048, 4096, torch.int32, True, True, True),  # int32 + ws/as + qs (full attrs)
]

L1_CASES = [
    (2, 1024, 2048, torch.float16, False, False, False),
    (3, 2048, 4096, torch.float16, False, True, True),  # fp16 + qs
    (5, 127, 512, torch.bfloat16, False, False, False),  # unaligned M
    (6, 8192, 512, torch.float16, False, True, False),
    (7, 1023, 2049, torch.bfloat16, False, False, True),  # unaligned M/H
    (8, 255, 4097, torch.bfloat16, False, False, False),
    (10, 1024, 2048, torch.int32, True, False, False),
    (12, 4096, 2048, torch.int32, True, True, False),
    (13, 127, 512, torch.int32, True, False, False),
    (14, 8192, 512, torch.int32, True, True, False),
    (15, 1023, 2049, torch.int32, True, False, True),
    (16, 255, 4097, torch.int32, True, True, False),
    (17, 10007, 32, torch.int32, True, False, False),  # large M, tiny H
    (18, 32768, 128, torch.int32, True, False, False),  # very large M
    (19, 4001, 1024, torch.int32, True, True, True),
    (20, 16384, 256, torch.int32, True, False, False),
]


# ========== L0 (gate) ==========
def test_dequant_swiglu_quant_l0():
    failures = [cid for cid, M, H, dt, ws, qs, al in L0_CASES if not _run_case(f"cb{cid}", M, H, dt, ws, qs, al, "L0")]
    assert not failures, f"L0 gate failed for cann-bench cases: {failures}"


# ========== L1 (functional) ==========
def test_dequant_swiglu_quant_l1():
    failures = [cid for cid, M, H, dt, ws, qs, al in L1_CASES if not _run_case(f"cb{cid}", M, H, dt, ws, qs, al, "L1")]
    assert not failures, f"L1 functional failed for cann-bench cases: {failures}"


# ========== L2 (negative: invalid inputs must be rejected) ==========
def test_dequant_swiglu_quant_l2():
    def _expect_reject(desc, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True
        print(f"[L2_FAIL] {desc}: invalid input silently accepted")
        return False

    cases = [
        ("unsupported_dtype_float32", lambda: dequant_swiglu_quant(torch.randn(64, 256, dtype=torch.float32, device="npu"))),
        ("none_input", lambda: dequant_swiglu_quant(None)),
        ("int32_missing_weight_scale", lambda: dequant_swiglu_quant(torch.randint(-128, 128, (64, 256), dtype=torch.int32, device="npu"))),
        ("odd_second_dim", lambda: dequant_swiglu_quant((torch.rand(64, 129, device="npu") * 2 - 1).to(torch.float16))),
        ("non_2d_input", lambda: dequant_swiglu_quant((torch.rand(2, 64, 256, device="npu") * 2 - 1).to(torch.float16))),
    ]
    failures = [desc for desc, fn in cases if not _expect_reject(desc, fn)]
    assert not failures, f"L2 negative tests failed for: {failures}"


# ========== Boundary (special values) ==========
def test_dequant_swiglu_quant_boundary():
    failures = []

    def _boundary(name, fn):
        try:
            result = fn()
            if result is None or result:
                print(f"[BOUNDARY_PASS] Boundary/{name}")
            else:
                print(f"[BOUNDARY_FAIL] Boundary/{name}")
                failures.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_FAIL] Boundary/{name}: {e}")
            failures.append(name)

    def case_all_zero():
        M, H = 64, 128
        x = torch.zeros(M, 2 * H, dtype=torch.float16, device="npu")
        y, scale = dequant_swiglu_quant(x)
        torch.npu.synchronize()
        ry, rs = golden_dequant_swiglu_quant(x)
        return precision_compare(y.cpu(), ry.cpu(), scale.cpu(), rs.cpu())["passed"]

    def case_extreme_int32():
        M, H = 64, 128
        x = torch.randint(-128, 128, (M, 2 * H), dtype=torch.int32, device="npu")
        x[0, :] = 128
        x[1, :] = -128
        ws = (torch.rand(1, 2 * H, device="npu") * 0.2 - 0.1).float()
        as_ = (torch.rand(M, device="npu") * 1.0 - 0.5).float()
        y, scale = dequant_swiglu_quant(x, ws, as_)
        torch.npu.synchronize()
        ry, rs = golden_dequant_swiglu_quant(x, ws, as_)
        return precision_compare(y.cpu(), ry.cpu(), scale.cpu(), rs.cpu())["passed"]

    def case_inf_input():
        M, H = 64, 128
        x = (torch.rand(M, 2 * H, device="npu") * 2.0 - 1.0).to(torch.float16)
        x[0, 0] = float("inf")
        y, scale = dequant_swiglu_quant(x)
        torch.npu.synchronize()
        ry, rs = golden_dequant_swiglu_quant(x)
        return precision_compare(y.cpu(), ry.cpu(), scale.cpu(), rs.cpu())["passed"]

    def case_nan_input():
        M, H = 64, 128
        x = (torch.rand(M, 2 * H, device="npu") * 2.0 - 1.0).to(torch.float16)
        x[0, 0] = float("nan")
        y, scale = dequant_swiglu_quant(x)
        torch.npu.synchronize()
        ry, rs = golden_dequant_swiglu_quant(x)
        return precision_compare(y.cpu(), ry.cpu(), scale.cpu(), rs.cpu())["passed"]

    _boundary("all_zero", case_all_zero)
    _boundary("extreme_int32", case_extreme_int32)
    _boundary("inf_input", case_inf_input)
    _boundary("nan_input", case_nan_input)

    assert not failures, f"boundary tests failed for: {failures}"


def main():
    parser = argparse.ArgumentParser(description="DequantSwigluQuant layered test suite (cann-bench cases)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.level in ("l0", "all"):
        test_dequant_swiglu_quant_l0()
    if args.level in ("l1", "all"):
        test_dequant_swiglu_quant_l1()
    if args.level in ("l2", "all"):
        test_dequant_swiglu_quant_l2()
    if args.level in ("boundary", "all"):
        test_dequant_swiglu_quant_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
