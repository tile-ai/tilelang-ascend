# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms of the CANN Open Software License Agreement Version 2.0 (the
# "License"). You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED.
# ----------------------------------------------------------------------------

"""Layered test suite for group_norm.

Levels are built from the 20 cann-bench group_norm cases (deterministic
crc32-keyed inputs), redistributed by case nature (dispatchable via --level;
all levels also run under pytest where a failure raises AssertionError):

  L0        gate: the three baseline dtype/shape cases
  L1        functional: irregular shapes, 2D/3D/4D/5D, num_groups / eps sweeps
  L2        negative: invalid inputs must be rejected by the wrapper
  Boundary  special values: inf / NaN / all-zero inputs and the fp16
            dynamic-range bound (±65504)

Judgement: L0/L1 use mixed tolerance (assert_close); Boundary additionally
requires exact NaN/Inf pattern equality vs the golden, then mixed tolerance
over the finite domain (mirroring the online cann-bench matched-ratio
metric) — MERE/MARE are printed as diagnostics only.
"""

import argparse
import gc
import os
import sys
import zlib

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import tilelang  # noqa: E402
from group_norm import group_norm  # noqa: E402

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ========== Golden Reference ==========
def golden_group_norm(x, gamma, beta, num_groups, eps=1e-5):
    """PyTorch reference: torch.nn.functional.group_norm."""
    if x.ndim == 2:
        return torch.nn.functional.group_norm(x.unsqueeze(-1), num_groups, gamma, beta, eps).squeeze(-1)
    return torch.nn.functional.group_norm(x, num_groups, gamma, beta, eps)


# ========== Precision Standards ==========
def get_precision(dtype):
    precision_map = {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
    return precision_map[dtype]


# ========== cann-bench Precision Checker ==========
def _mere_mare(actual, golden):
    diff = (actual.float() - golden.float()).abs()
    denom = golden.float().abs() + 1e-7
    rel_err = diff / denom
    both_nan = torch.isnan(actual.float()) & torch.isnan(golden.float())
    both_inf = torch.isinf(actual.float()) & torch.isinf(golden.float())
    rel_err = torch.where(both_nan | both_inf, torch.zeros_like(rel_err), rel_err)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    return mere, mare


def check_precision(actual, golden, dtype_str, label=""):
    """Special-value judgement, mirroring the online cann-bench metric:

    1. NaN/Inf patterns must match the golden exactly (bit-level semantics)
    2. finite domain: mixed tolerance (atol + rtol*|golden|), same as L0/L1

    MERE/MARE are reported as diagnostics only — strict MARE is meaningless
    near zero (fp16-subnormal outputs give |diff|/|golden| ~ O(1) with
    |diff| < 1 ulp), which is why it is not a gate here.
    """
    a, g = actual.float(), golden.float()
    nan_ok = torch.equal(torch.isnan(a), torch.isnan(g))
    inf_ok = torch.equal(torch.isinf(a), torch.isinf(g))
    if not (nan_ok and inf_ok):
        print(f"[PRECISION_FAIL] {label} dtype={dtype_str} NaN/Inf pattern mismatch (nan_ok={nan_ok} inf_ok={inf_ok})")
        return False
    m = torch.isfinite(g)
    mere = mare = 0.0
    if bool(m.any()):
        atol, rtol = get_precision(dtype_str)
        try:
            torch.testing.assert_close(a[m], g[m], atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {label} dtype={dtype_str} mixed tolerance: {str(e).splitlines()[0][:100]}")
            return False
        mere, mare = _mere_mare(a[m], g[m])
    print(f"[PRECISION_PASS] {label} dtype={dtype_str} MERE={mere:.6e} MARE={mare:.6e} (diagnostic)")
    return True


# ========== Input Generation ==========
def _gen_input(shape, dtype, value_range, seed_offset=0):
    torch.manual_seed(42 + seed_offset)
    torch_dtype = _DTYPE_MAP[dtype] if isinstance(dtype, str) else dtype
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")

    if value_range == "inf":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x.view(-1)[0] = float("inf")
        if x.numel() > 1:
            x.view(-1)[1] = float("-inf")
    elif value_range == "nan":
        x = torch.randn(shape, dtype=torch.float32).uniform_(-1, 1)
        x.view(-1)[0] = float("nan")
    elif value_range == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
    else:
        lo, hi = value_range
        x = torch.empty(shape, dtype=torch.float32).uniform_(lo, hi)

    return x.to(torch_dtype).npu(), dtype_str


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()


# ========== Test Runner ==========
def _run_case(name, shape, dtype, num_groups, eps, value_range, level="l0"):
    _cleanup_npu()
    dtype_str = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
    C = shape[1]

    # crc32-keyed seed: deterministic across processes (hash() is salted)
    x, dtype_str = _gen_input(shape, dtype_str, value_range, seed_offset=zlib.crc32(name.encode()) % 1000)
    gamma = torch.randn(C, dtype=_DTYPE_MAP[dtype_str]).npu()
    beta = torch.randn(C, dtype=_DTYPE_MAP[dtype_str]).npu()

    y = group_norm(x, gamma, beta, num_groups, eps)
    ref = golden_group_norm(x.cpu().float(), gamma.cpu().float(), beta.cpu().float(), num_groups, eps).to(_DTYPE_MAP[dtype_str])

    if level in ("l0", "l1"):
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
        return check_precision(y.cpu(), ref, dtype_str, label=f"{level} {name} shape={shape}")


# ========== cann-bench case tables (20 cases, redistributed by nature) ==========
# (cann-bench case id, shape, dtype, num_groups, eps, value_range)
L0_CASES = [
    (1, [8, 32, 64, 64], "float16", 8, 1e-5, (-1, 1)),  # 4D baseline fp16
    (2, [4, 64, 128, 128], "float32", 16, 1e-5, (-2, 2)),  # 4D baseline fp32
    (3, [2, 128, 256, 256], "bfloat16", 32, 1e-5, (-3, 3)),  # 4D baseline bf16
]

L1_CASES = [
    (4, [16, 257, 32, 31], "float16", 1, 1e-5, (-10, 10)),  # irregular C/S
    (5, [8, 512, 17, 15], "float32", 2, 1e-5, (-100, 100)),
    (6, [64, 64, 128], "bfloat16", 4, 1e-5, (-5, 5)),  # 3D
    (7, [2, 256, 128, 128], "float16", 16, 1e-5, (-0.1, 0.1)),
    (8, [16, 127, 31, 33], "float32", 1, 1e-6, (-1, 1)),
    (9, [3, 64, 64, 64], "bfloat16", 8, 1e-3, (-0.5, 0.5)),
    (10, [7, 32, 63, 65], "float16", 4, 1e-4, (-1, 2)),
    (11, [3, 64, 127, 129], "float32", 8, 1e-4, (-50, 100)),
    (12, [5, 48, 33, 65], "bfloat16", 6, 1e-4, (-3, 6)),
    (13, [1023, 257], "float16", 1, 1e-6, (-1, 1)),  # 2D developer path
    (14, [2, 60, 5, 7, 480], "float32", 4, 1e-5, (-10, 10)),  # 5D
    (18, [2, 256, 127, 129], "bfloat16", 16, 1e-5, (-0.2, 0.2)),
    (20, [1, 513, 63, 63], "float32", 3, 1e-6, (-20, 40)),
]

BOUNDARY_CASES = [
    (15, [4, 31, 251, 251], "bfloat16", 1, 1e-8, "inf"),  # ±inf injection
    (16, [2, 64, 67, 71], "float16", 8, 1e-7, "nan"),  # NaN injection
    (17, [8, 127, 33, 31], "float32", 1, 1e-4, "zero"),  # all-zero input
    (19, [4, 128, 255, 257], "float16", 32, 1e-3, (-65504, 65504)),  # fp16 range bound
]


# ========== L0 (gate) ==========
def test_group_norm_l0():
    failures = [cid for cid, shape, dtype, g, eps, vr in L0_CASES if not _run_case(f"cb{cid}", shape, dtype, g, eps, vr, level="l0")]
    assert not failures, f"L0 gate failed for cann-bench cases: {failures}"


# ========== L1 (functional) ==========
def test_group_norm_l1():
    failures = [cid for cid, shape, dtype, g, eps, vr in L1_CASES if not _run_case(f"cb{cid}", shape, dtype, g, eps, vr, level="l1")]
    assert not failures, f"L1 functional failed for cann-bench cases: {failures}"


# ========== L2 (negative: invalid inputs must be rejected) ==========
def test_group_norm_l2():
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
            lambda: group_norm(
                torch.ones(2, 8, 4, 4, dtype=torch.int32).npu(),
                torch.ones(8, dtype=torch.int32).npu(),
                torch.ones(8, dtype=torch.int32).npu(),
                2,
                1e-5,
            ),
        ),
        (
            "c_not_divisible_by_num_groups",
            lambda: group_norm(
                torch.randn(2, 7, 4, 4, dtype=torch.float16).npu(),
                torch.randn(7, dtype=torch.float16).npu(),
                torch.randn(7, dtype=torch.float16).npu(),
                3,
                1e-5,
            ),
        ),
        ("none_input", lambda: group_norm(None, None, None, 1, 1e-5)),
    ]
    failures = [desc for desc, fn in cases if not _expect_reject(desc, fn)]
    assert not failures, f"L2 negative tests failed for: {failures}"


# ========== Boundary (special values) ==========
def test_group_norm_boundary():
    failures = [
        cid for cid, shape, dtype, g, eps, vr in BOUNDARY_CASES if not _run_case(f"cb{cid}", shape, dtype, g, eps, vr, level="boundary")
    ]
    assert not failures, f"boundary tests failed for cann-bench cases: {failures}"


def main():
    parser = argparse.ArgumentParser(description="GroupNorm layered test suite (cann-bench cases)")
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
        test_group_norm_l0()
    if args.level in ("l1", "all"):
        test_group_norm_l1()
    if args.level in ("l2", "all"):
        test_group_norm_l2()
    if args.level in ("boundary", "all"):
        test_group_norm_boundary()

    print("Test Passed!")


if __name__ == "__main__":
    main()
