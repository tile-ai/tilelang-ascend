import argparse
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_mla_fwd import (  # noqa: E402
    sparse_mla_fwd_interface,
    ref_sparse_mla_fwd_interface,
    make_indices,
)

# L0 configs: (name, B, S, SKV, H, HKV, DQK, DV, topk)
L0_CONFIGS = [
    ("l0_min", 1, 128, 128, 16, 1, 576, 512, 64),
    ("l0_default", 1, 4096, 4096, 128, 1, 576, 512, 2048),
    ("l0_long_kv", 1, 4096, 8192, 128, 1, 576, 512, 2048),
    ("l0_multi_batch", 2, 1024, 1024, 64, 1, 576, 512, 512),
    ("l0_gqa", 1, 2048, 2048, 128, 2, 576, 512, 1024),
    ("l0_large_topk", 1, 512, 4096, 128, 1, 576, 512, 4096),
]

# L1 configs: functional (irregular shapes, q<k, multi-batch, GQA variants) [blocking]
L1_CONFIGS = [
    ("l1_small_topk", 1, 256, 256, 32, 1, 576, 512, 64),
    ("l1_multi_batch_gqa", 2, 512, 512, 64, 2, 576, 512, 256),
    ("l1_q_neq_kv", 1, 128, 256, 16, 1, 576, 512, 128),
    ("l1_large_seq", 1, 2048, 2048, 64, 1, 576, 512, 512),
    ("l1_min_heads", 1, 128, 128, 16, 1, 576, 512, 128),
]

# L2 configs: abnormal inputs [non-blocking]
L2_CONFIGS = [
    ("l2_single_iter", 1, 128, 128, 16, 1, 576, 512, 64, 1.0),
    ("l2_large_skv", 1, 128, 4096, 128, 1, 576, 512, 2048, 1.0),
    ("l2_min_batch", 1, 64, 64, 16, 1, 576, 512, 64, 1.0),
]

# Boundary configs: special values [non-blocking]
BOUNDARY_CONFIGS = [
    ("zero_input", 1, 128, 128, 16, 1, 576, 512, 64, 0.0),
    ("large_input", 1, 128, 128, 16, 1, 576, 512, 64, 10.0),
]

ATOL, RTOL = 1e-2, 1e-2


# ===========================================================================
# Test helpers
# ===========================================================================


def _prepare_and_run(B, S, SKV, H, HKV, DQK, DV, topk, device="npu", atol=ATOL, rtol=RTOL, input_scale=1.0):
    """Prepare inputs, run kernel + golden, return (max_diff, passed, tl_out, ref_out)."""
    torch.manual_seed(0)
    dtype = torch.bfloat16

    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device) * input_scale
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device) * input_scale
    indices = make_indices(B, S, SKV, HKV, topk, device=device)

    torch.npu.synchronize()

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices)
    torch.npu.synchronize()

    ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
    torch.npu.synchronize()

    if torch.isnan(tl_out).any():
        return float("nan"), False, tl_out, ref_out

    max_diff = (tl_out.float() - ref_out.float()).abs().max().item()

    try:
        torch.testing.assert_close(tl_out.cpu(), ref_out.cpu(), rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False

    return max_diff, passed, tl_out, ref_out


# ===========================================================================
# L0 gate tests
# ===========================================================================


def test_l0():
    """L0 gate tests: 6 regular shapes. All must PASS."""
    device = "npu"
    ok = True
    for name, b, s, skv, h, hkv, dqk, dv, tk in L0_CONFIGS:
        try:
            max_diff, passed, _, _ = _prepare_and_run(b, s, skv, h, hkv, dqk, dv, tk, device=device)
            if passed:
                print(f"[PRECISION_PASS] l0 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} max_diff={max_diff:.6e}")
            else:
                print(f"[PRECISION_FAIL] l0 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} max_diff={max_diff:.6e}")
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l0 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} error={e}")
            ok = False
    return ok


# ===========================================================================
# L1 functional tests [blocking]
# ===========================================================================


def test_l1():
    """L1 functional tests: irregular shapes, q<k, multi-batch, GQA variants."""
    device = "npu"
    ok = True
    for name, b, s, skv, h, hkv, dqk, dv, tk in L1_CONFIGS:
        try:
            max_diff, passed, _, _ = _prepare_and_run(b, s, skv, h, hkv, dqk, dv, tk, device=device)
            if passed:
                print(f"[PRECISION_PASS] l1 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} max_diff={max_diff:.6e}")
            else:
                print(f"[PRECISION_FAIL] l1 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} max_diff={max_diff:.6e}")
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l1 {name} B={b} S={s} SKV={skv} H={h} HKV={hkv} topk={tk} error={e}")
            ok = False
    return ok


# ===========================================================================
# L2 / Boundary [non-blocking]
# ===========================================================================


def _run_boundary_case(name, B, S, SKV, H, HKV, DQK, DV, topk, input_scale=1.0):
    """Run one L2/Boundary case. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    device = "npu"
    try:
        max_diff, passed, tl_out, ref_out = _prepare_and_run(B, S, SKV, H, HKV, DQK, DV, topk, device=device, input_scale=input_scale)
        if torch.isnan(tl_out).any():
            print(f"[BOUNDARY_WARN] {name} NaN in output")
            return
        try:
            torch.testing.assert_close(tl_out.cpu(), ref_out.cpu(), rtol=RTOL, atol=ATOL)
            print(f"[BOUNDARY_PASS] {name} max_diff={max_diff:.6e}")
        except AssertionError:
            print(f"[BOUNDARY_WARN] {name} max_diff={max_diff:.6e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {name}: {e}")


def test_l2():
    """L2 abnormal input tests. Non-blocking."""
    for name, b, s, skv, h, hkv, dqk, dv, tk, scale in L2_CONFIGS:
        _run_boundary_case(name, b, s, skv, h, hkv, dqk, dv, tk, scale)


def test_boundary():
    """Boundary / special value tests. Non-blocking."""
    for name, b, s, skv, h, hkv, dqk, dv, tk, scale in BOUNDARY_CONFIGS:
        _run_boundary_case(name, b, s, skv, h, hkv, dqk, dv, tk, scale)


# ===========================================================================
# Main entry
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Sparse MLA Forward precision test (Developer mode)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "all"],
        help="Test level to run (l0=precision gate only, all=L0+L1+L2+Boundary).",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    blocking_ok = True

    print("=" * 70)
    print(f"Stage 2: Precision test (Developer mode) — level={args.level}")
    print("=" * 70)

    blocking_ok &= test_l0()

    if args.level == "all":
        print()
        blocking_ok &= test_l1()
        test_l2()
        test_boundary()

    print()
    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    else:
        print("Test FAILED (L0/L1 blocking)")
        sys.exit(1)


if __name__ == "__main__":
    main()
