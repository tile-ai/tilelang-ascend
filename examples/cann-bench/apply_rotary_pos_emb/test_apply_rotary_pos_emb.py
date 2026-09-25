"""ApplyRotaryPosEmb tests mirroring cann-bench cases.csv.

Levels:
  - l0: small smoke cases
  - l1: 20 functional cases copied from cann-bench/tasks/level2/apply_rotary_pos_emb/cases.csv
  - l2: argument rejection checks
  - boundary: special values from cases.csv
"""

import argparse
import math
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_rotary_pos_emb import apply_rotary_pos_emb  # noqa: E402


COVERAGE_CATEGORY = "FusedComposite"
COVERAGE_MANIFEST = {}
COVERAGE_NA = {}


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


PRECISION_THRESHOLDS = {
    "float16": (1.0e-2, 1.0e-1),
    "float32": (5.0e-3, 5.0e-2),
    "bfloat16": (1.0e-2, 1.0e-1),
}


def _sf(v):
    if isinstance(v, str):
        return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[v]
    return v


def make_input(shape, dtype_str, value_range, seed):
    torch_dtype = DTYPE_MAP[dtype_str]
    lo, hi = _sf(value_range[0]), _sf(value_range[1])
    generator = torch.Generator().manual_seed(seed)

    if isinstance(lo, float) and isinstance(hi, float) and math.isnan(lo) and math.isnan(hi):
        return torch.full(shape, float("nan"), dtype=torch_dtype).npu()

    if isinstance(lo, float) and (math.isinf(lo) or math.isinf(hi)):
        base = (torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0 - 1.0).to(torch_dtype)
        flat = base.view(-1)
        if flat.numel() > 0:
            flat[0] = float("inf")
        if flat.numel() > 1:
            flat[1] = float("-inf")
        return base.npu()

    if lo == hi:
        return torch.full(shape, float(lo), dtype=torch_dtype).npu()

    return (torch.rand(shape, dtype=torch.float32, generator=generator) * (hi - lo) + lo).to(torch_dtype).npu()


def rotate_half(x, mode):
    if mode == "interleaved":
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def golden_apply_rotary_pos_emb(query, key, cos, sin, layout=0, rotaryMode="half"):
    input_dtype = query.dtype
    compute_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    query = query.to(compute_dtype)
    key = key.to(compute_dtype)
    cos = cos.to(compute_dtype)
    sin = sin.to(compute_dtype)

    def apply_one(x):
        cos_work = cos
        sin_work = sin
        if cos_work.dim() == 2:
            cos_work = cos_work.unsqueeze(0).unsqueeze(2)
            sin_work = sin_work.unsqueeze(0).unsqueeze(2)
        elif cos_work.dim() == 3:
            cos_work = cos_work.unsqueeze(2)
            sin_work = sin_work.unsqueeze(2)

        if layout == 1:
            cos_work = cos_work.transpose(1, 2)
            sin_work = sin_work.transpose(1, 2)

        if rotaryMode == "interleaved":
            cos_work = cos_work.unsqueeze(-1).expand(*cos_work.shape, 2).reshape(*cos_work.shape[:-1], -1)
            sin_work = sin_work.unsqueeze(-1).expand(*sin_work.shape, 2).reshape(*sin_work.shape[:-1], -1)
        else:
            cos_work = cos_work.repeat(1, 1, 1, 2)
            sin_work = sin_work.repeat(1, 1, 1, 2)

        return x * cos_work + rotate_half(x, rotaryMode) * sin_work

    q_out = apply_one(query)
    k_out = apply_one(key)
    if input_dtype in (torch.float16, torch.bfloat16):
        q_out = q_out.to(input_dtype)
        k_out = k_out.to(input_dtype)
    return q_out, k_out


def check_precision(actual, golden, dtype_str):
    threshold, mare_threshold = PRECISION_THRESHOLDS[dtype_str]
    actual = actual.detach().cpu()
    golden = golden.detach().cpu()

    if not torch.equal(torch.isnan(actual), torch.isnan(golden)):
        return False, float("inf"), float("inf"), "NaN position mismatch"
    if not torch.equal(torch.isinf(actual), torch.isinf(golden)):
        return False, float("inf"), float("inf"), "Inf position mismatch"

    mask = torch.isfinite(actual) & torch.isfinite(golden)
    if mask.sum().item() == 0:
        return True, 0.0, 0.0, ""

    a = actual[mask].float()
    g = golden[mask].float()
    abs_err = (a - g).abs()
    rel_err = abs_err / (g.abs() + 1.0e-7)
    mere = rel_err.mean().item()
    mare = rel_err.max().item()
    max_abs = abs_err.max().item()
    passed = mere <= threshold and mare <= mare_threshold
    return passed, mere, mare, f"max_abs={max_abs:.6e}"


def run_case(case_id, input_shapes, dtype_str, layout, rotary_mode, value_ranges, note=""):
    query = make_input(tuple(input_shapes[0]), dtype_str, value_ranges[0], 1000 + case_id * 4)
    key = make_input(tuple(input_shapes[1]), dtype_str, value_ranges[1], 1001 + case_id * 4)
    cos = make_input(tuple(input_shapes[2]), dtype_str, value_ranges[2], 1002 + case_id * 4)
    sin = make_input(tuple(input_shapes[3]), dtype_str, value_ranges[3], 1003 + case_id * 4)

    q_out, k_out = apply_rotary_pos_emb(query, key, cos, sin, layout=layout, rotaryMode=rotary_mode)
    q_golden, k_golden = golden_apply_rotary_pos_emb(query.cpu(), key.cpu(), cos.cpu(), sin.cpu(), layout, rotary_mode)

    q_ok, q_mere, q_mare, q_extra = check_precision(q_out, q_golden, dtype_str)
    k_ok, k_mere, k_mare, k_extra = check_precision(k_out, k_golden, dtype_str)
    passed = q_ok and k_ok
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] case_{case_id:02d} dtype={dtype_str} layout={layout} mode={rotary_mode} "
        f"q(MERE={q_mere:.6e}, MARE={q_mare:.6e}, {q_extra}) "
        f"k(MERE={k_mere:.6e}, MARE={k_mare:.6e}, {k_extra}) {note}"
    )
    return passed


L1_CASES = [
    (
        1,
        [[16, 512, 16, 128], [16, 512, 16, 128], [512, 64], [512, 64]],
        "float16",
        0,
        "half",
        [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
        "M-float16-16M-aligned-layout0-half",
    ),
    (
        2,
        [[7, 1021, 31, 64], [7, 1021, 31, 64], [1021, 32], [1021, 32]],
        "float32",
        0,
        "half",
        [[-2, 2], [-2, 2], [-1, 1], [-1, 1]],
        "L-float32-14M-prime-layout0-half",
    ),
    (
        3,
        [[31, 251, 7, 128], [31, 251, 7, 128], [251, 64], [251, 64]],
        "bfloat16",
        0,
        "half",
        [[-3, 3], [-3, 3], [-1, 1], [-1, 1]],
        "M-bfloat16-7M-prime-layout0-half",
    ),
    (
        4,
        [[15, 16, 511, 128], [15, 16, 511, 128], [511, 64], [511, 64]],
        "float16",
        1,
        "half",
        [[-10, 10], [-10, 10], [-1, 1], [-1, 1]],
        "M-float16-15M-prime-layout1-half",
    ),
    (
        5,
        [[8, 2048, 16, 128], [8, 2048, 16, 128], [2048, 64], [2048, 64]],
        "float32",
        0,
        "interleaved",
        [[-100, 100], [-100, 100], [-1, 1], [-1, 1]],
        "L-float32-33M-aligned-layout0-interleaved",
    ),
    (
        6,
        [[17, 15, 1021, 64], [17, 15, 1021, 64], [1021, 32], [1021, 32]],
        "bfloat16",
        1,
        "interleaved",
        [[-1000, 1000], [-1000, 1000], [-1, 1], [-1, 1]],
        "M-bfloat16-16M-prime-layout1-interleaved",
    ),
    (
        7,
        [[13, 511, 13, 128], [13, 511, 13, 128], [511, 64], [511, 64]],
        "float16",
        0,
        "half",
        [[-0.1, 0.1], [-0.1, 0.1], [-1, 1], [-1, 1]],
        "M-float16-10M-small-range",
    ),
    (
        8,
        [[7, 1009, 7, 128], [7, 1009, 7, 128], [1009, 64], [1009, 64]],
        "float32",
        0,
        "half",
        [[-1, 2], [-1, 2], [-1, 1], [-1, 1]],
        "M-float32-7M-asymmetric",
    ),
    (
        9,
        [[257, 17, 17, 128], [257, 17, 17, 128], [17, 64], [17, 64]],
        "bfloat16",
        1,
        "half",
        [[-5, 10], [-5, 10], [-1, 1], [-1, 1]],
        "M-bfloat16-9M-layout1-half",
    ),
    (
        10,
        [[11, 503, 11, 128], [11, 503, 11, 128], [503, 64], [503, 64]],
        "float16",
        0,
        "interleaved",
        [[-50, 100], [-50, 100], [-1, 1], [-1, 1]],
        "M-float16-8M-interleaved",
    ),
    (
        11,
        [[19, 1023, 19, 64], [19, 1023, 19, 64], [1023, 32], [1023, 32]],
        "float32",
        0,
        "half",
        [[-65504, 65504], [-65504, 65504], [-1, 1], [-1, 1]],
        "L-float32-24M-boundary",
    ),
    (
        12,
        [[4001, 3, 3, 64], [4001, 3, 3, 64], [3, 32], [3, 32]],
        "bfloat16",
        1,
        "interleaved",
        [[-88, 88], [-88, 88], [-1, 1], [-1, 1]],
        "S-bfloat16-2M-layout1-interleaved",
    ),
    (
        13,
        [[1, 500001, 1, 64], [1, 500001, 1, 64], [500001, 32], [500001, 32]],
        "float16",
        0,
        "half",
        [["-inf", "inf"], ["-inf", "inf"], [-1, 1], [-1, 1]],
        "M-float16-32M-inf",
    ),
    (
        14,
        [[3, 13, 17, 128], [3, 13, 17, 128], [13, 64], [13, 64]],
        "float32",
        0,
        "half",
        [["nan", "nan"], ["nan", "nan"], ["nan", "nan"], ["nan", "nan"]],
        "S-float32-85K-nan",
    ),
    (
        15,
        [[509, 4, 4, 128], [509, 4, 4, 128], [4, 64], [4, 64]],
        "bfloat16",
        1,
        "half",
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        "S-bfloat16-360K-zero",
    ),
    (
        16,
        [[16, 61, 16, 128], [16, 61, 16, 128], [61, 64], [61, 64]],
        "float16",
        0,
        "half",
        [[-0.5, 0.5], [-0.5, 0.5], [-1, 1], [-1, 1]],
        "M-float16-2M-small-range",
    ),
    (
        17,
        [[1023, 31, 31, 64], [1023, 31, 31, 64], [31, 32], [31, 32]],
        "float32",
        1,
        "interleaved",
        [[-1, 3], [-1, 3], [-1, 1], [-1, 1]],
        "L-float32-60M-layout1-interleaved",
    ),
    (
        18,
        [[8, 255, 8, 128], [8, 255, 8, 128], [255, 64], [255, 64]],
        "bfloat16",
        0,
        "half",
        [[-1000, 1000], [-1000, 1000], [-1, 1], [-1, 1]],
        "L-bfloat16-2M-large-range",
    ),
    (
        19,
        [[127, 4, 4, 128], [127, 4, 4, 128], [4, 64], [4, 64]],
        "float16",
        1,
        "half",
        [[-0.2, 0.2], [-0.2, 0.2], [-1, 1], [-1, 1]],
        "S-float16-260K-layout1-half",
    ),
    (
        20,
        [[7, 2047, 7, 128], [7, 2047, 7, 128], [2047, 64], [2047, 64]],
        "float32",
        0,
        "interleaved",
        [[-3, 6], [-3, 6], [-1, 1], [-1, 1]],
        "M-float32-12M-interleaved",
    ),
]


def test_apply_rotary_pos_emb_l0():
    smoke_cases = [
        (
            101,
            [[2, 8, 2, 64], [2, 8, 2, 64], [8, 32], [8, 32]],
            "float16",
            0,
            "half",
            [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
            "l0-layout0-half",
        ),
        (
            102,
            [[2, 2, 8, 64], [2, 2, 8, 64], [8, 32], [8, 32]],
            "float32",
            1,
            "interleaved",
            [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
            "l0-layout1-interleaved",
        ),
        (
            103,
            [[1, 16, 1, 128], [1, 16, 1, 128], [16, 64], [16, 64]],
            "bfloat16",
            0,
            "half",
            [[-1, 1], [-1, 1], [-1, 1], [-1, 1]],
            "l0-bfloat16-half",
        ),
    ]
    ok = True
    for case in smoke_cases:
        try:
            ok &= run_case(*case)
        except Exception as e:
            print(f"[PRECISION_FAIL] case_{case[0]:02d}: {e}")
            ok = False
    return ok


def test_apply_rotary_pos_emb_l1():
    ok = True
    for case in L1_CASES:
        try:
            ok &= run_case(*case)
        except Exception as e:
            print(f"[PRECISION_FAIL] case_{case[0]:02d}: {e}")
            ok = False
    return ok


def test_apply_rotary_pos_emb_l2():
    def _expect_reject(desc, fn):
        try:
            fn()
            print(f"[BOUNDARY_WARN] {desc}: silently accepted")
        except (AssertionError, ValueError, RuntimeError, KeyError):
            print(f"[BOUNDARY_PASS] {desc}: correctly rejected")

    q = torch.randn(1, 4, 1, 64, dtype=torch.float32).npu()
    k = torch.randn(1, 4, 1, 64, dtype=torch.float32).npu()
    c = torch.randn(4, 32, dtype=torch.float32).npu()
    s = torch.randn(4, 32, dtype=torch.float32).npu()
    _expect_reject("invalid_layout", lambda: apply_rotary_pos_emb(q, k, c, s, layout=2, rotaryMode="half"))
    _expect_reject("invalid_mode", lambda: apply_rotary_pos_emb(q, k, c, s, layout=0, rotaryMode="bad"))
    _expect_reject("odd_head_dim", lambda: apply_rotary_pos_emb(q[..., :63], k[..., :63], c, s, layout=0, rotaryMode="half"))


def test_apply_rotary_pos_emb_boundary():
    ok = True
    for case in [L1_CASES[i - 1] for i in (13, 14, 15)]:
        try:
            ok &= run_case(*case)
        except Exception as e:
            print(f"[BOUNDARY_FAIL] case_{case[0]:02d}: {e}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_apply_rotary_pos_emb_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_apply_rotary_pos_emb_l1()
    if args.level in ("l2", "all"):
        test_apply_rotary_pos_emb_l2()
    if args.level in ("boundary", "all"):
        blocking_ok &= test_apply_rotary_pos_emb_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
