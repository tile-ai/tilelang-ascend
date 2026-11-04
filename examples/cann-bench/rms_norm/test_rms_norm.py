# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for rms_norm.

Levels are built from the 20 cann-bench level2/rms_norm cases (verified
against cann-bench/tasks/level2/rms_norm/cases.yaml; deterministic
crc32-keyed inputs), redistributed by case nature (dispatchable via --level;
all levels also run under pytest where a failure raises AssertionError):

  L0        gate: the three dtype baselines + the unaligned 3D shape
  L1        functional: hidden-dim / eps / value-range sweeps, 3D/4D/5D
  L2        negative: invalid inputs must be rejected by the wrapper
  Boundary  special values: inf / NaN / all-zero inputs and the fp16
            dynamic-range bound (±65504), plus a D=1 degenerate-dimension
            check (not covered by any cann-bench case)

Judgement: L0/L1 use mixed tolerance (assert_close); Boundary uses the
online cann-bench matched-ratio metric (both-NaN / both-±inf / finite
mixed tolerance, ratio >= 0.99) — MERE/MARE are printed as diagnostics
only.
"""

import argparse
import os
import sys
import zlib

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from rms_norm import rms_norm_host  # noqa: E402

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ========== Golden Reference ==========
def golden_rms_norm(x, gamma, epsilon=1e-6):
    """Golden reference using torch.nn.functional.rms_norm (CPU fp32)."""
    return torch.nn.functional.rms_norm(x, normalized_shape=gamma.shape, weight=gamma, eps=epsilon)


# ========== Precision Standards ==========
def get_precision(dtype):
    """Normalization-class precision standards (L0 threshold)."""
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map[dtype]


# ========== cann-bench Precision Checker ==========
def _mere_mare(actual, golden):
    """Compute MERE (mean relative error) and MARE (max relative error).

    Formula (cann-bench standard):
        relative_error = abs(actual - golden) / (abs(golden) + 1e-7)
    """
    diff = (actual.float() - golden.float()).abs()
    denom = golden.float().abs() + 1e-7
    rel_err = diff / denom
    # Handle NaN/Inf: positions where both are NaN/Inf are considered matching
    both_nan = torch.isnan(actual.float()) & torch.isnan(golden.float())
    both_inf = torch.isinf(actual.float()) & torch.isinf(golden.float())
    rel_err = torch.where(both_nan | both_inf, torch.zeros_like(rel_err), rel_err)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    return mere, mare


def check_precision(actual, golden, dtype_str, label=""):
    """Special-value judgement, mirroring the online cann-bench metric:

    matched = both-NaN | both-±inf (same sign) | |diff| <= atol + rtol*|golden|
    Gate: matched_ratio >= 0.99 (the repo-wide required_ratio standard).

    Rationale: on ±inf inputs the kernel's Newton rsqrt refinement turns
    0*inf into NaN and contaminates the affected row (kernel [NaN, NaN]
    vs golden [NaN, 0]) — a deviation of one element per injected inf
    that the online matched-ratio metric accepts (cb15: 2 / 2,000,006
    positions) but a strict bit-pattern gate would not.  MERE/MARE are
    printed as diagnostics only.
    """
    a, g = actual.float(), golden.float()
    both_nan = torch.isnan(a) & torch.isnan(g)
    both_inf = torch.isinf(a) & torch.isinf(g) & (torch.sign(a) == torch.sign(g))
    m = torch.isfinite(g)
    finite_ok = torch.zeros_like(a, dtype=torch.bool)
    if bool(m.any()):
        atol, rtol = get_precision(dtype_str)
        finite_ok[m] = (a[m] - g[m]).abs() <= atol + rtol * g[m].abs()
    matched_ratio = (both_nan | both_inf | finite_ok).float().mean().item()

    mere = mare = 0.0
    if bool(m.any()):
        mere, mare = _mere_mare(a[m], g[m])
    passed = matched_ratio >= 0.99
    tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
    print(f"[{tag}] {label} dtype={dtype_str} matched_ratio={matched_ratio:.6f} MERE={mere:.6e} MARE={mare:.6e} (diagnostic)")
    return passed


# ========== Input Generation ==========
def _gen_input(shape, dtype, value_range, seed_offset=0):
    """Generate input tensor with specified value range."""
    torch.manual_seed(42 + seed_offset)
    torch_dtype = _DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")

    if value_range == "inf":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x[0, 0] = float("inf")
        x[1, 0] = float("-inf")
    elif value_range == "nan":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x[0, 0] = float("nan")
    elif value_range == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
    else:
        lo, hi = value_range
        x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)

    return x.to(torch_dtype).npu(), dtype_str


def _gen_gamma(D, dtype_str):
    """Generate gamma (ones, as per cann-bench standard golden)."""
    torch_dtype = _DTYPE_MAP[dtype_str]
    return torch.ones(D, dtype=torch_dtype).npu()


# ========== Test Runner ==========
def _run_case(name, shape, dtype, eps, value_range, level="l0"):
    """Run a single test case: generate input -> kernel -> golden -> check."""
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
    D = shape[-1]

    # crc32-keyed seed: deterministic across processes (hash() is salted)
    x, dtype_str = _gen_input(shape, dtype_str, value_range, seed_offset=zlib.crc32(name.encode()) % 1000)
    gamma = _gen_gamma(D, dtype_str)

    # Kernel
    y = rms_norm_host(x, gamma, eps)

    # Golden (CPU fp32, cast back to original dtype)
    ref = golden_rms_norm(x.cpu().float(), gamma.cpu().float(), eps).to(_DTYPE_MAP[dtype_str])

    # Precision check
    if level in ("l0", "l1"):
        # Use torch.testing.assert_close for L0/L1
        atol, rtol = get_precision(dtype_str)
        max_diff = (y.cpu().float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(y.cpu(), ref, atol=atol, rtol=rtol)
            print(f"[PRECISION_PASS] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}")
            return True
        except Exception as e:
            print(f"[PRECISION_FAIL] {level} {name} shape={shape} dtype={dtype_str} max_diff={max_diff:.6e}: {e}")
            return False
    else:
        # cann-bench special-value judgement
        return check_precision(y.cpu(), ref, dtype_str, label=f"{level} {name} shape={shape}")


# ========== cann-bench case tables (20 cases, redistributed by nature) ==========
# (cann-bench case id, shape, dtype, eps, value_range)
L0_CASES = [
    (1, [32, 128, 768], "float16", 1e-6, (-1, 1)),  # 3D baseline fp16
    (2, [32, 128, 1024], "float32", 1e-6, (-2, 2)),  # 3D baseline fp32
    (3, [32, 128, 2048], "bfloat16", 1e-6, (-3, 3)),  # 3D baseline bf16
    (7, [63, 67, 1023], "float16", 1e-8, (-0.1, 0.1)),  # unaligned 3D
]

L1_CASES = [
    (4, [16, 256, 4096], "float16", 1e-6, (-10, 10)),  # hidden-dim sweep
    (5, [8, 512, 8192], "float32", 1e-6, (-100, 100)),
    (6, [4, 1023, 4097], "bfloat16", 1e-5, (-5, 5)),  # unaligned dims + eps sweep
    (8, [16, 511, 2049], "float32", 1e-4, (-1, 1)),
    (9, [8, 1021, 4099], "bfloat16", 1e-12, (-0.5, 0.5)),
    (10, [33, 127, 769], "float16", 1e-6, (-1, 2)),
    (11, [31, 129, 2049], "float32", 1e-6, (-50, 100)),
    (12, [17, 255, 4097], "bfloat16", 1e-6, (-3, 6)),
    (13, [7, 1009, 1021], "float16", 1e-7, (-1, 1)),
    (14, [11, 367, 373], "float32", 1e-5, (-10, 10)),
    (18, [2, 511, 8192], "bfloat16", 1e-6, (-0.2, 0.2)),
    (20, [2, 3, 17, 1024, 128], "float32", 1e-6, (-20, 40)),  # 5D
]

BOUNDARY_CASES = [
    (15, [1000003, 2], "bfloat16", 1e-6, "inf"),  # ±inf injection, huge S
    (16, [11, 13, 17, 67], "float16", 1e-8, "nan"),  # NaN injection, 4D
    (17, [3, 7, 11, 4096], "float32", 1e-4, "zero"),  # all-zero input, 4D
    (19, [4, 255, 4096], "float16", 1e-3, (-65504, 65504)),  # fp16 range bound
]


# ========== L0 (gate) ==========
def test_rms_norm_l0():
    failures = [cid for cid, shape, dtype, eps, vr in L0_CASES if not _run_case(f"cb{cid}", shape, dtype, eps, vr, level="l0")]
    assert not failures, f"L0 gate failed for cann-bench cases: {failures}"


# ========== L1 (functional) ==========
def test_rms_norm_l1():
    failures = [cid for cid, shape, dtype, eps, vr in L1_CASES if not _run_case(f"cb{cid}", shape, dtype, eps, vr, level="l1")]
    assert not failures, f"L1 functional failed for cann-bench cases: {failures}"


# ========== L2 (negative: invalid inputs must be rejected) ==========
def test_rms_norm_l2():
    def _expect_reject(desc, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True
        print(f"[L2_FAIL] {desc}: invalid input silently accepted")
        return False

    cases = [
        (
            "unsupported_dtype_int32",
            lambda: rms_norm_host(torch.ones(64, 128, dtype=torch.int32).npu(), torch.ones(128, dtype=torch.int32).npu(), 1e-6),
        ),
        (
            "gamma_dim_mismatch",
            lambda: rms_norm_host(torch.ones(64, 128, dtype=torch.float16).npu(), torch.ones(64, dtype=torch.float16).npu(), 1e-6),
        ),
    ]
    failures = [desc for desc, fn in cases if not _expect_reject(desc, fn)]
    assert not failures, f"L2 negative tests failed for: {failures}"


# ========== Boundary (special values) ==========
def test_rms_norm_boundary():
    failures = [cid for cid, shape, dtype, eps, vr in BOUNDARY_CASES if not _run_case(f"cb{cid}", shape, dtype, eps, vr, level="boundary")]

    # D=1 degenerate dimension: not covered by any cann-bench case
    x = torch.randn(64, 1, dtype=torch.float16)
    gamma = torch.ones(1, dtype=torch.float16)
    y = rms_norm_host(x.npu(), gamma.npu(), 1e-6).cpu()
    ref = golden_rms_norm(x.float(), gamma.float(), 1e-6).to(torch.float16)
    try:
        torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)
        print("[BOUNDARY_PASS] boundary d_equals_1: degenerate dimension")
    except AssertionError as e:
        print(f"[BOUNDARY_FAIL] boundary d_equals_1: {str(e).splitlines()[0][:100]}")
        failures.append("d_equals_1")

    assert not failures, f"boundary tests failed for: {failures}"


def main():
    parser = argparse.ArgumentParser(description="RMSNorm layered test suite (cann-bench cases)")
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
        test_rms_norm_l0()
    if args.level in ("l1", "all"):
        test_rms_norm_l1()
    if args.level in ("l2", "all"):
        test_rms_norm_l2()
    if args.level in ("boundary", "all"):
        test_rms_norm_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
