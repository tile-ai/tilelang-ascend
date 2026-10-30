# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for softmax.

Levels are built from the 20 cann-bench level2/softmax cases (verified
against cann-bench/tasks/level2/softmax/cases.yaml, including the dim
attribute), redistributed by case nature (dispatchable via --level; all
levels also run under pytest where a failure raises AssertionError):

  L0        gate: dtype baselines + one case per kernel dispatch path
            (2D last-dim / 2D dim-0 / interior-dim 3D)
  L1        functional: remaining regular cases (dim sweeps, unaligned /
            prime shapes, 3D/4D/5D, value-range sweeps)
  L2        negative: invalid inputs must be rejected by the wrapper
  Boundary  special values: inf / NaN / all-zero inputs and the fp16
            dynamic-range bound (±65504) from cann-bench, plus small-shape
            edges (1x1 / 1xN / Mx1 / 2x3)

Judgement: the cann-bench relative_error checker (two-stage MERE/MARE with
small-value / cancellation domain classification and the ErrorCount
NPU/CPU <= 2 fallback), computed against an FP64 golden and a
same-precision native reference.
"""

import argparse
import os
import sys
import zlib

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from softmax import softmax_impl  # noqa: E402

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

# ========== Precision thresholds (cann-bench standard) ==========

_THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float": 2**-13,
    "float32": 2**-13,
}

# Small value domain thresholds (cann-bench standard)
_SMALL_VALUE_THRESHOLDS = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float": 2**-14,
    "float32": 2**-14,
}

_SMALL_VALUE_ERROR_THRESHOLDS = {
    "float16": 2**-16,
    "bfloat16": 2**-16,
    "float": 2**-30,
    "float32": 2**-30,
}

# Cancellation domain thresholds (cann-bench standard)
_CANCEL_BOUNDARY = {
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "float": 2**-8,
    "float32": 2**-8,
}

_CANCEL_ZERO = {
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "float": 2**-8,
    "float32": 2**-8,
}


# ========== Golden ==========
def golden_softmax(x, dim=-1):
    """PyTorch golden reference in FP64 precision (matches cann-bench fp64_cpu strategy).

    Computes softmax in float64 on CPU for maximum accuracy.
    """
    return torch.nn.functional.softmax(x.double(), dim=dim)


def native_softmax(x, dim=-1):
    """Same-precision reference (matches cann-bench native_output).

    Computes softmax in the original dtype on CPU, used for small-value
    domain fallback comparison.
    """
    return torch.nn.functional.softmax(x, dim=dim)


# ========== Precision ==========


def check_precision(actual, golden, native, dtype_str):
    """Check precision matching cann-bench relative_error checker.

    Implements the full cann-bench logic:
    1. First stage: overall MERE < threshold AND MARE < 10*threshold
    2. Second stage (if first fails): classify mismatches into
       small-value / cancellation / normal domains
    3. Fallback: if only small-value/cancellation mismatches, use
       ErrorCount ratio (NPU/CPU <= 2)

    Args:
        actual: kernel output (NPU), same dtype as input
        golden: FP64 golden reference (CPU)
        native: same-precision reference (CPU, original dtype)
        dtype_str: dtype string for threshold lookup

    Returns:
        (passed, mere, mare) — mere/mare are display values
    """
    threshold = _THRESHOLDS[dtype_str]
    mare_threshold = 10 * threshold
    sv_threshold = _SMALL_VALUE_THRESHOLDS[dtype_str]
    sv_error = _SMALL_VALUE_ERROR_THRESHOLDS[dtype_str]
    cancel_boundary = _CANCEL_BOUNDARY[dtype_str]
    cancel_zero = _CANCEL_ZERO[dtype_str]

    # Move to CPU and upcast for comparison
    if actual.device.type == "npu":
        actual = actual.cpu()
    target_dtype = actual.dtype
    golden_trunc = golden.to(target_dtype).double()
    output_fp64 = actual.double()

    # NaN position check
    nan_out = torch.isnan(output_fp64)
    nan_gold = torch.isnan(golden_trunc)
    if not torch.all(nan_out == nan_gold):
        return False, float("inf"), float("inf")

    # Inf saturation handling: replace one-sided inf with max finite value
    inf_out = torch.isinf(output_fp64)
    inf_gold = torch.isinf(golden_trunc)
    inf_mismatch = inf_out != inf_gold
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)
    if torch.any(inf_mismatch):
        max_finite = float(torch.finfo(target_dtype).max)
        if torch.any(inf_out & ~inf_gold):
            mask = inf_out & ~inf_gold
            output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
        if torch.any(inf_gold & ~inf_out):
            mask = inf_gold & ~inf_out
            golden_trunc[mask] = torch.sign(golden_trunc[mask]) * max_finite
    both_inf = inf_out & inf_gold
    if torch.any(both_inf):
        if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_trunc[both_inf])):
            return False, float("inf"), float("inf")
        inf_match_mask[both_inf] = True

    # Compute relative error
    diff = torch.abs(output_fp64 - golden_trunc)
    golden_abs = torch.abs(golden_trunc)
    denominator = golden_abs + 1e-7
    relative_error = diff / denominator

    # Exclude NaN, Inf, and matched-Inf positions
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_re = relative_error[valid_mask]

    if len(valid_re) == 0:
        return True, 0.0, 0.0

    overall_mere = float(valid_re.mean())
    overall_mare = float(valid_re.max())

    # First stage: overall MERE/MARE check
    if overall_mere < threshold and overall_mare < mare_threshold:
        return True, overall_mere, overall_mare

    # Second stage: classify mismatches
    mismatch_mask = (relative_error > mare_threshold) & valid_mask

    # Small value domain: |golden| < sv_threshold
    sv_mask = (golden_abs < sv_threshold) & valid_mask
    # Cancellation: output ≈ 0 AND golden in [sv_threshold, cancel_boundary)
    output_abs = torch.abs(output_fp64)
    cancel_mask = (output_abs < cancel_zero) & (golden_abs >= sv_threshold) & (golden_abs < cancel_boundary) & valid_mask
    # Normal: everything else
    normal_mismatch = mismatch_mask & ~sv_mask & ~cancel_mask

    if torch.any(normal_mismatch):
        # Normal domain has mismatches → FAIL
        return False, overall_mere, overall_mare

    # Only small-value/cancellation mismatches → fallback
    # NPU small-value errors: |golden| < sv_threshold AND |diff| > sv_error
    sv_npu_err = sv_mask & (diff > sv_error)
    sv_npu_count = int(sv_npu_err.sum())

    # CPU (native) small-value errors
    native_fp64 = native.to(target_dtype).double()
    cpu_diff = torch.abs(native_fp64 - golden_trunc)
    sv_cpu_err = sv_mask & (cpu_diff > sv_error)
    sv_cpu_count = int(sv_cpu_err.sum())

    if sv_cpu_count == 0:
        sv_passed = sv_npu_count == 0
    else:
        sv_passed = sv_npu_count / max(sv_cpu_count, 1) <= 2

    # Cancellation errors
    cancel_npu_err = cancel_mask & (relative_error > mare_threshold)
    cancel_npu_count = int(cancel_npu_err.sum())
    cpu_re = cpu_diff / (golden_abs + 1e-7)
    cancel_cpu_err = cancel_mask & (cpu_re > mare_threshold)
    cancel_cpu_count = int(cancel_cpu_err.sum())

    if cancel_cpu_count == 0:
        cancel_passed = cancel_npu_count == 0
    else:
        cancel_passed = cancel_npu_count / max(cancel_cpu_count, 1) <= 2

    passed = sv_passed and cancel_passed

    # Display MERE/MARE excluding small-value/cancellation
    normal_mask = ~sv_mask & ~cancel_mask & valid_mask
    normal_re = relative_error[normal_mask]
    if len(normal_re) > 0:
        display_mere = float(normal_re.mean())
        display_mare = float(normal_re.max())
    else:
        display_mere = 0.0
        display_mare = 0.0

    return passed, display_mere, display_mare


# ========== Input Generation ==========
def _gen_normal(shape, dtype_torch, lo, hi):
    """Generate uniform random input in [lo, hi]."""
    x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)
    return x.to(dtype_torch)


def _gen_zeros(shape, dtype_torch):
    """Generate all-zeros input."""
    return torch.zeros(shape, dtype=dtype_torch)


def _gen_with_inf(shape, dtype_torch):
    """Generate input with inf/-inf special values.

    Ensures some slices contain +inf (-> NaN output) and some are all -inf
    (-> NaN output), matching torch.nn.functional.softmax behavior.
    """
    x = torch.randn(shape, dtype=torch.float32) * 10
    x = x.to(dtype_torch)
    if x.dim() == 2:
        # Every 2000th row: all -inf (-> NaN output)
        x[::2000, :] = float("-inf")
        # Every 1000th row (offset): +inf in col 0 (-> NaN output)
        x[1000::2000, 0] = float("inf")
    else:
        xf = x.view(-1)
        xf[::10000] = float("inf")
        xf[1::10000] = float("-inf")
    return x


def _gen_with_nan(shape, dtype_torch):
    """Generate input with NaN values (-> NaN output propagation)."""
    x = torch.randn(shape, dtype=torch.float32)
    x = x.to(dtype_torch)
    xf = x.view(-1)
    xf[::10000] = float("nan")
    return x


# ========== Test Runner ==========
def _run_case(name, shape, dtype_torch, dim, value_range, level):
    """Run a single test case.

    Args:
        name: case name (e.g. "L0-1")
        shape: input shape
        dtype_torch: torch dtype
        dim: softmax dimension
        value_range: (lo, hi) tuple or "zero"/"inf"/"nan"
        level: "L0"/"L1"/"L2"/"Boundary"

    Returns:
        True if passed
    """
    # crc32-keyed seed: deterministic per case, independent of call order
    torch.manual_seed(zlib.crc32(name.encode()) % (2**31))
    dtype_str = _DTYPE_MAP[dtype_torch]
    shape_list = list(shape)
    try:
        # Generate input
        if value_range == "zero":
            x_cpu = _gen_zeros(shape, dtype_torch)
        elif value_range == "inf":
            x_cpu = _gen_with_inf(shape, dtype_torch)
        elif value_range == "nan":
            x_cpu = _gen_with_nan(shape, dtype_torch)
        else:
            lo, hi = value_range
            x_cpu = _gen_normal(shape, dtype_torch, lo, hi)

        # Run kernel on NPU
        x_npu = x_cpu.npu()
        y = softmax_impl(x_npu, dim=dim).cpu()

        # Compute golden (FP64) and native (same precision) on CPU
        ref = golden_softmax(x_cpu, dim=dim)
        native = native_softmax(x_cpu, dim=dim)

        # Check precision (cann-bench relative_error checker logic)
        ok, mere, mare = check_precision(y, ref, native, dtype_str)

        if level in ("L0", "L1"):
            tag = "[PRECISION_PASS]" if ok else "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_FAIL]"

        print(f"{tag} {level} {name} shape={shape_list} dtype={dtype_str} dim={dim} MERE={mere:.2e} MARE={mare:.2e}")
        return ok
    except Exception as e:
        if level in ("L0", "L1"):
            tag = "[PRECISION_FAIL]"
        else:
            tag = "[BOUNDARY_FAIL]"
        print(f"{tag} {level} {name} shape={shape_list} dtype={dtype_str} dim={dim}: {e}")
        return False


# ========== cann-bench case tables (20 cases, redistributed by nature) ==========
# (cann-bench case id, shape, dtype, dim, value_range)
L0_CASES = [
    (1, [1024, 1024], torch.float16, -1, (-1, 1)),  # 2D last-dim baseline fp16
    (2, [2048, 2048], torch.float32, -1, (-2, 2)),  # 2D last-dim baseline fp32
    (3, [4096, 4096], torch.bfloat16, -1, (-3, 3)),  # 2D last-dim baseline bf16
    (4, [8192, 8192], torch.float16, 0, (-10, 10)),  # dim=0 kernel path
    (9, [127, 257, 1023], torch.bfloat16, -2, (-0.5, 0.5)),  # interior dim -> 3D kernel
]

L1_CASES = [
    (5, [8192, 8192], torch.float32, 1, (-100, 100)),
    (6, [31, 67, 127, 257], torch.bfloat16, 2, (-5, 5)),
    (7, [1023, 2047], torch.float16, -1, (-0.1, 0.1)),  # unaligned
    (8, [2049, 4097], torch.float32, -1, (-1, 1)),  # prime shapes
    (10, [1009, 1021], torch.float16, -1, (-1, 2)),
    (11, [367, 373, 379], torch.float32, 1, (-50, 100)),  # 3D
    (12, [11, 13, 17, 4001], torch.bfloat16, -1, (-3, 6)),  # 4D
    (16, [512, 2049], torch.float16, 0, (-0.5, 0.5)),  # dim=0 sweep
    (17, [255, 8193], torch.float32, 1, (-1000, 1000)),  # wide value range
    (18, [2, 511, 2049], torch.bfloat16, -1, (-0.2, 0.2)),  # 3D unaligned
    (20, [2, 3, 17, 1024, 101], torch.float32, 3, (-20, 40)),  # 5D interior dim
]

BOUNDARY_CB_CASES = [
    (13, [1000003, 2], torch.float16, -1, "inf"),  # ±inf injection, huge M
    (14, [11, 13, 17, 67, 67], torch.float32, -1, "nan"),  # NaN injection, 5D
    (15, [3, 7, 11, 13, 1013], torch.bfloat16, -1, "zero"),  # all-zero input, 5D
    (19, [4, 255, 2049], torch.float16, 1, (-65504, 65504)),  # fp16 range bound
]

# small-shape edges, not covered by any cann-bench case
BOUNDARY_SMALL_CASES = [
    ("small_1x1", [1, 1], torch.float16, -1, (-1, 1)),
    ("small_1xN", [1, 1024], torch.float32, -1, (-1, 1)),
    ("small_Mx1", [1024, 1], torch.float16, -1, (-1, 1)),
    ("small_2x3", [2, 3], torch.bfloat16, -1, (-1, 1)),
]


# ========== L0 (gate) ==========
def test_softmax_l0():
    failures = [cid for cid, shape, dt, dim, vr in L0_CASES if not _run_case(f"cb{cid}", shape, dt, dim, vr, "L0")]
    assert not failures, f"L0 gate failed for cann-bench cases: {failures}"


# ========== L1 (functional) ==========
def test_softmax_l1():
    failures = [cid for cid, shape, dt, dim, vr in L1_CASES if not _run_case(f"cb{cid}", shape, dt, dim, vr, "L1")]
    assert not failures, f"L1 functional failed for cann-bench cases: {failures}"


# ========== L2 (negative: invalid inputs must be rejected) ==========
def test_softmax_l2():
    def _expect_reject(desc, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True
        print(f"[L2_FAIL] {desc}: invalid input silently accepted")
        return False

    cases = [
        ("unsupported_dtype_int32", lambda: softmax_impl(torch.ones(64, 128, dtype=torch.int32).npu(), dim=-1)),
        ("dim_out_of_range", lambda: softmax_impl(torch.randn(64, 128, dtype=torch.float16).npu(), dim=5)),
        ("dim_negative_out_of_range", lambda: softmax_impl(torch.randn(64, 128, dtype=torch.float16).npu(), dim=-3)),
        ("none_input", lambda: softmax_impl(None, dim=-1)),
    ]
    failures = [desc for desc, fn in cases if not _expect_reject(desc, fn)]
    assert not failures, f"L2 negative tests failed for: {failures}"


# ========== Boundary (special values + small shapes) ==========
def test_softmax_boundary():
    failures = [cid for cid, shape, dt, dim, vr in BOUNDARY_CB_CASES if not _run_case(f"cb{cid}", shape, dt, dim, vr, "Boundary")]
    failures += [name for name, shape, dt, dim, vr in BOUNDARY_SMALL_CASES if not _run_case(name, shape, dt, dim, vr, "Boundary")]
    assert not failures, f"boundary tests failed for: {failures}"


def main():
    parser = argparse.ArgumentParser(description="Softmax layered test suite (cann-bench cases)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    if args.level in ("l0", "all"):
        test_softmax_l0()
    if args.level in ("l1", "all"):
        test_softmax_l1()
    if args.level in ("l2", "all"):
        test_softmax_l2()
    if args.level in ("boundary", "all"):
        test_softmax_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
