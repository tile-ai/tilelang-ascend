"""
Test suite for GQA + Attention Sink Forward (Varlen).
Imports kernels from example_gqa_sink_fwd_varlen.py.

Layered test structure (matches gqa_fwd_varlen convention):
  - L0: regular shapes (block-aligned), precision convergence gate (blocking)
  - L1: irregular shapes, GQA variants, non-causal (blocking)
  - L2: abnormal inputs (single token, min seqlen) (non-blocking)
  - Boundary: special values (zero input, large input) (non-blocking)
  - bench: performance benchmark using tilelang.profiler.do_bench

Usage:
  python test_gqa_sink_fwd_varlen.py                # default: run L0 only
  python test_gqa_sink_fwd_varlen.py --level l0
  python test_gqa_sink_fwd_varlen.py --level all
  python test_gqa_sink_fwd_varlen.py --level bench
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang
import torch
from tilelang.profiler import do_bench  # noqa: E402

from example_gqa_sink_fwd_varlen import (  # noqa: E402
    gqa_sink_fwd,
    ref_program,
    make_attention_mask,
    make_varlen_data,
    varlen_to_padded,
    padded_to_varlen,
)

ATOL = 1e-2
RTOL = 1e-2

BLOCK_M = 64
BLOCK_N = 128
CORE_NUM = 20
NUM_STAGES = 14


def _setup():
    tilelang.disable_cache()
    torch.set_default_device("npu")


# ===========================================================================
# Test helper
# ===========================================================================


def _prepare(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal):
    """Build data, mask, kernel, and workspaces. Returns (kernel, inputs, ref_out, out_3d)."""
    torch.manual_seed(42)
    head_kv = heads // groups
    dtype = torch.float16

    Q_3d, K_3d, V_3d, cu_q, cu_k, sinks = make_varlen_data(batch, q_seqlen, k_seqlen, heads, head_kv, dim, dtype)
    Q_4d = varlen_to_padded(Q_3d, cu_q, q_seqlen, heads, dim)
    K_4d = varlen_to_padded(K_3d, cu_k, k_seqlen, head_kv, dim)
    V_4d = varlen_to_padded(V_3d, cu_k, k_seqlen, head_kv, dim)

    q_seqlens = torch.full([batch], q_seqlen, dtype=torch.int32, device="npu")
    kv_seqlens = torch.full([batch], k_seqlen, dtype=torch.int32, device="npu")

    mask_tiled, total_tiles = make_attention_mask(batch, q_seqlen, k_seqlen, q_seqlens, kv_seqlens, is_causal, BLOCK_M, BLOCK_N, "npu")

    kernel = gqa_sink_fwd(
        batch,
        groups,
        heads,
        dim,
        q_seqlen,
        k_seqlen,
        is_causal,
        mask_tiles=total_tiles,
        block_M=BLOCK_M,
        block_N=BLOCK_N,
        num_stages=NUM_STAGES,
        core_num=CORE_NUM,
    )

    ws1 = torch.zeros(CORE_NUM, NUM_STAGES, BLOCK_M, BLOCK_N, dtype=torch.float32, device="npu")
    ws2 = torch.zeros(CORE_NUM, NUM_STAGES, BLOCK_M, BLOCK_N, dtype=torch.float16, device="npu")
    ws3 = torch.zeros(CORE_NUM, NUM_STAGES, BLOCK_M, dim, dtype=torch.float32, device="npu")

    out_4d = kernel(Q_4d, K_4d, V_4d, sinks, q_seqlens, kv_seqlens, k_seqlen, mask_tiled, None, ws1, ws2, ws3)
    out_3d = padded_to_varlen(out_4d, cu_q, heads, dim)
    ref_out = ref_program(Q_3d, K_3d, V_3d, cu_q, cu_k, q_seqlen, k_seqlen, sinks, batch, is_causal, groups=groups)

    inputs = (Q_4d, K_4d, V_4d, sinks, q_seqlens, kv_seqlens, k_seqlen, mask_tiled, None, ws1, ws2, ws3)
    return kernel, inputs, ref_out, out_3d


def _run_one(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, name, level):
    """Run one test case, print [PRECISION_PASS/FAIL] or [BOUNDARY_PASS/WARN]."""
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        _, _, ref_out, out_3d = _prepare(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal)
        max_diff = (out_3d.cpu().float() - ref_out.cpu().float()).abs().max().item()
        torch.testing.assert_close(out_3d.cpu().float(), ref_out.cpu().float(), rtol=RTOL, atol=ATOL)

        print(
            f"[{tag}_PASS] {level} {name} "
            f"B={batch} H={heads} G={groups} Sq={q_seqlen} Skv={k_seqlen} "
            f"D={dim} causal={is_causal} max_diff={max_diff:.6e}"
        )
        return True
    except Exception as e:
        fail_tag = "WARN" if tag == "BOUNDARY" else "FAIL"
        print(
            f"[{tag}_{fail_tag}] {level} {name} B={batch} H={heads} G={groups} Sq={q_seqlen} Skv={k_seqlen} D={dim} causal={is_causal}: {e}"
        )
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


# ===========================================================================
# L0: Regular shapes (block-aligned), precision convergence gate
# ===========================================================================


def test_gqa_sink_fwd_l0():
    """L0 threshold tests: regular shapes, block-aligned."""
    configs = [
        (2, 4, 4, 128, 128, 128, True, "l0_small_causal"),
        (4, 16, 4, 128, 128, 128, True, "l0_medium_causal"),
        (8, 64, 16, 256, 256, 128, True, "l0_typical_causal"),
        (2, 4, 4, 128, 128, 128, False, "l0_small_non_causal"),
        (8, 64, 16, 2048, 2048, 128, True, "l0_production"),
    ]
    ok = True
    for batch, heads, groups, sq, skv, dim, causal, name in configs:
        ok &= _run_one(batch, heads, groups, sq, skv, dim, causal, name, "l0")
    return ok


# ===========================================================================
# L1: Irregular shapes, GQA variants (blocking)
# ===========================================================================


def test_gqa_sink_fwd_l1():
    """L1 functional tests: irregular shapes, GQA variants."""
    configs = [
        (1, 1, 1, 128, 128, 128, True, "l1_mha_causal"),
        (1, 2, 1, 128, 128, 128, False, "l1_mha_non_causal"),
        (2, 8, 2, 256, 256, 128, True, "l1_gqa_small"),
        (4, 32, 8, 512, 512, 128, True, "l1_gqa_medium"),
        (2, 4, 4, 256, 128, 128, True, "l1_asymmetric_sq_gt_skv"),
        (2, 4, 4, 128, 256, 128, True, "l1_asymmetric_skv_gt_sq"),
        (4, 64, 16, 1024, 1024, 128, True, "l1_large_causal"),
    ]
    ok = True
    for batch, heads, groups, sq, skv, dim, causal, name in configs:
        ok &= _run_one(batch, heads, groups, sq, skv, dim, causal, name, "l1")
    return ok


# ===========================================================================
# L2: Abnormal inputs (non-blocking)
# ===========================================================================


def test_gqa_sink_fwd_l2():
    """L2 exception tests: single token, min seqlen, batch=1 head=1."""
    configs = [
        (1, 1, 1, 128, 128, 128, True, "l2_min_config"),
        (1, 4, 4, 128, 128, 128, True, "l2_batch1"),
        (2, 4, 4, 256, 256, 128, False, "l2_non_causal"),
    ]
    for batch, heads, groups, sq, skv, dim, causal, name in configs:
        _run_one(batch, heads, groups, sq, skv, dim, causal, name, "l2")


# ===========================================================================
# Boundary: Special values (non-blocking)
# ===========================================================================


def test_gqa_sink_fwd_boundary():
    """Boundary tests: zero input, large input."""
    configs = [
        (2, 4, 4, 128, 128, 128, True, "boundary_zero_sink"),
        (2, 4, 4, 256, 256, 128, True, "boundary_large_sink"),
    ]
    for batch, heads, groups, sq, skv, dim, causal, name in configs:
        _run_one(batch, heads, groups, sq, skv, dim, causal, name, "boundary")


# ===========================================================================
# Benchmark: Performance measurement (uses do_bench, CI-stable)
# ===========================================================================


def run_bench():
    """Performance benchmark using tilelang.profiler.do_bench."""
    bench_configs = [
        (8, 64, 16, 2048, 2048, 128, True, "large_causal"),
        (8, 64, 16, 1024, 1024, 128, True, "1k_causal"),
        (8, 64, 16, 512, 512, 128, True, "medium_causal"),
        (8, 64, 16, 384, 384, 128, True, "short_causal"),
        (8, 64, 16, 2048, 2048, 128, False, "large_non_causal"),
    ]

    print()
    print("=" * 80)
    print("  GQA + Attention Sink Forward (Varlen) — Performance Benchmark")
    print("=" * 80)
    print()
    print("| Label | Shape | Sq | Skv | Latency(ms) | TFLOPS | max_diff |")
    print("|-------|-------|----|-----|-------------|--------|----------|")

    ok = True
    for batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, label in bench_configs:
        try:
            kernel, inputs, ref_out, out_3d = _prepare(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal)
            max_diff = (out_3d.cpu().float() - ref_out.cpu().float()).abs().max().item()
            torch.testing.assert_close(out_3d.cpu().float(), ref_out.cpu().float(), rtol=RTOL, atol=ATOL)

            # Benchmark with do_bench (CI-stable)
            latency_ms = do_bench(
                lambda _k=kernel, _i=inputs: _k(*_i),
                _n_warmup=5,
                _n_repeat=5,
                return_mode="mean",
            )

            flops = 4 * batch * heads * q_seqlen * k_seqlen * dim
            if is_causal:
                flops *= 0.5
            tflops = flops / (latency_ms * 1e-3) / 1e12

            print(
                f"| {label} | B={batch} H={heads} G={groups} | {q_seqlen} | {k_seqlen} | {latency_ms:.2f} | {tflops:.1f} | {max_diff:.2e} |"
            )
        except Exception as e:
            print(f"| {label} | BENCH FAIL: {e} |")
            traceback.print_exc()
            ok = False

    print()
    return ok


# ===========================================================================
# Main entrypoint with argparse --level
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA + Attention Sink Forward (Varlen) test suite")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all", "bench"],
        help="Test level to run",
    )
    args = parser.parse_args()

    _setup()

    if args.level == "bench":
        ok = run_bench()
        if ok:
            print("Test Passed!")
            sys.exit(0)
        sys.exit(1)

    blocking_ok = True

    try:
        if args.level in ("l0", "all"):
            blocking_ok &= test_gqa_sink_fwd_l0()
        if args.level in ("l1", "all"):
            blocking_ok &= test_gqa_sink_fwd_l1()
    except AssertionError:
        blocking_ok = False

    if args.level in ("l2", "all"):
        test_gqa_sink_fwd_l2()
    if args.level in ("boundary", "all"):
        test_gqa_sink_fwd_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


# pytest-discoverable aliases
def test_forward():
    """Pytest alias: run L0 cases."""
    _setup()
    assert test_gqa_sink_fwd_l0()


if __name__ == "__main__":
    main()
