import argparse
import os
import sys
from typing import Optional

import torch

import tilelang
from tilelang.profiler import do_bench  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mha_sink_fwd_bhsd import (  # noqa: E402
    flashattn,
    build_causal_mask,
)


# ===========================================================================
# Golden function (from GPU source, device-agnostic — runs on NPU)
# ===========================================================================


# Modified from https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py
def ref_program(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sliding_window: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """PyTorch reference for attention sink forward.

    Pure PyTorch implementation (no flash_attn dependency), runs directly on NPU.
    Receives the ORIGINAL ``[heads]`` sinks (not the host pre-broadcast version).
    """
    query = query.transpose(1, 2).contiguous().unsqueeze(3)  # align with original interface
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape
    start_q = num_keys - num_queries

    sm_scale: float = 1.0 / head_dim**0.5

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups, head_dim).to(dtype)
    return output.transpose(1, 2).contiguous()


# ===========================================================================
# Test helpers
# ===========================================================================


def _prepare_and_run(
    batch,
    heads,
    seq_q,
    seq_kv,
    dim,
    window_size,
    block_M,
    block_N,
    device,
    dtype,
    atol,
    rtol,
    input_scale=1.0,
):
    """Prepare inputs + mask + sinks, run Expert kernel + golden, return (max_diff, passed).

    Q/K/V are padded to block-aligned dimensions for the kernel; the golden
    receives original (unpadded) tensors. Output is sliced to seq_q for
    comparison (padded Q rows produce 0, excluded).
    """
    torch.manual_seed(0)

    # Original (unpadded) tensors for golden
    q_orig = torch.randn(batch, heads, seq_q, dim, dtype=dtype, device=device) * input_scale
    k_orig = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device) * input_scale
    v_orig = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device) * input_scale
    sinks = torch.randn(heads, dtype=dtype, device=device) * input_scale

    # Pad to block-aligned for kernel
    padded_q = ((seq_q + block_M - 1) // block_M) * block_M
    padded_kv = ((seq_kv + block_N - 1) // block_N) * block_N
    q = torch.zeros(batch, heads, padded_q, dim, dtype=dtype, device=device)
    q[:, :, :seq_q, :] = q_orig
    k = torch.zeros(batch, heads, padded_kv, dim, dtype=dtype, device=device)
    k[:, :, :seq_kv, :] = k_orig
    v = torch.zeros(batch, heads, padded_kv, dim, dtype=dtype, device=device)
    v[:, :, :seq_kv, :] = v_orig

    # Pre-broadcast sinks [heads] -> [heads, padded_q] for the kernel
    sinks_broad = sinks.unsqueeze(1).expand(-1, padded_q).contiguous()

    # Build causal + optional sliding window mask (padded, shared across batch/head)
    mask = build_causal_mask(seq_q, seq_kv, window_size, device, block_M, block_N)

    # Compile kernel (padded dimensions)
    # Pass real (unpadded) dims + has_window for mask skip optimization (iter2)
    kernel = flashattn(
        batch,
        heads,
        padded_q,
        padded_kv,
        dim,
        block_M=block_M,
        block_N=block_N,
        has_window=(window_size is not None),
        real_seq_q=seq_q,
        real_seq_kv=seq_kv,
    )

    # Run kernel
    out = kernel(q, k, v, sinks_broad, mask)
    torch.npu.synchronize()

    # Golden (device-agnostic, runs on NPU; takes original [heads] sinks, unpadded tensors)
    ref_out = ref_program(q_orig, k_orig, v_orig, sinks, sliding_window=window_size, dtype=dtype)
    torch.npu.synchronize()

    # Slice kernel output to original seq_q (padded rows produce 0, excluded)
    out_valid = out[:, :, :seq_q, :]

    if torch.isnan(out_valid).any():
        return float("nan"), False

    max_diff = (out_valid.float() - ref_out.float()).abs().max().item()

    try:
        torch.testing.assert_close(out_valid.cpu(), ref_out.cpu(), rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False

    return max_diff, passed


# ===========================================================================
# L0 gate tests (DESIGN.md §11.4)
# ===========================================================================


def test_mha_sink_fwd_bhsd_l0():
    """L0 gate tests: regular shapes (block-aligned), for precision convergence.

    Cases from DESIGN.md §11.4 「L0 门槛测试计划」.
    All shapes are divisible by block_M=block_N=128.
    """
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, seq_q, seq_kv, dim, window_size)
    configs = [
        ("l0_min_causal", 1, 1, 128, 128, 128, None),
        ("l0_small_causal", 1, 4, 256, 256, 128, None),
        ("l0_multi_batch", 2, 8, 512, 512, 128, None),
        ("l0_default", 8, 32, 4096, 4096, 128, None),
        ("l0_window", 1, 4, 256, 256, 128, 128),
    ]

    ok = True
    for name, b, h, sq, skv, d, window in configs:
        try:
            max_diff, passed = _prepare_and_run(
                b,
                h,
                sq,
                skv,
                d,
                window,
                block_M,
                block_N,
                device,
                dtype,
                atol,
                rtol,
            )
            if passed:
                print(f"[PRECISION_PASS] l0 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} max_diff={max_diff:.6e}")
            else:
                print(f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} max_diff={max_diff:.6e}")
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} error={e}")
            ok = False
    return ok


# ===========================================================================
# L1 / L2 / Boundary (expanded by tilelang-op-test-design scenario B)
# L1 = functional (irregular shapes, tail blocks), blocking.
# L2 = abnormal inputs, non-blocking. Boundary = special values, non-blocking.
# ===========================================================================


def test_mha_sink_fwd_bhsd_l1():
    """L1 functional tests: irregular shapes (tail blocks), q!=k, multi-batch, window.

    Adapted from the Developer version's L1 cases for block_M=block_N=128.
    Irregular shapes test host padding + mask=0 tail-block handling.
    Returns True iff all cases pass (blocking).
    """
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, seq_q, seq_kv, dim, window_size)
    configs = [
        ("l1_irregular_c", 1, 4, 100, 100, 128, None),  # tail block (100 -> pad 128)
        ("l1_q_short_k_c", 2, 4, 128, 256, 128, None),  # q<k, offset=128
        ("l1_multi_irregular", 3, 8, 200, 200, 128, None),  # multi-batch + tail (200 -> pad 256)
        ("l1_window_irregular", 1, 4, 200, 200, 128, 128),  # window + tail
        ("l1_large_c", 4, 16, 512, 512, 128, None),  # larger scale
    ]

    ok = True
    for name, b, h, sq, skv, d, window in configs:
        try:
            max_diff, passed = _prepare_and_run(b, h, sq, skv, d, window, block_M, block_N, device, dtype, atol, rtol)
            if passed:
                print(f"[PRECISION_PASS] l1 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} max_diff={max_diff:.6e}")
            else:
                print(f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} max_diff={max_diff:.6e}")
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} sq={sq} skv={skv} dim={d} window={window} error={e}")
            ok = False
    return ok


def _run_boundary_case(name, batch, heads, seq_q, seq_kv, dim, window_size, input_scale, block_M, block_N):
    """Run one L2/Boundary case. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    try:
        max_diff, passed = _prepare_and_run(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            window_size,
            block_M,
            block_N,
            device,
            dtype,
            atol,
            rtol,
            input_scale=input_scale,
        )
        if passed:
            print(f"[BOUNDARY_PASS] {name} max_diff={max_diff:.6e}")
        else:
            print(f"[BOUNDARY_WARN] {name} max_diff={max_diff:.6e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {name}: {e}")


def test_mha_sink_fwd_bhsd_l2():
    """L2 abnormal input tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    cases = [
        ("l2_single_block", 1, 1, 128, 128, 128, None, 1.0),  # minimum 1 KV block
        ("l2_min_multi", 1, 1, 256, 256, 128, None, 1.0),  # minimal multi-block
        ("l2_large_offset", 1, 4, 128, 4096, 128, None, 1.0),  # very large offset (32 KV blocks)
    ]
    for name, b, h, sq, skv, d, window, scale in cases:
        _run_boundary_case(name, b, h, sq, skv, d, window, scale, block_M, block_N)


def test_mha_sink_fwd_bhsd_boundary():
    """Boundary / special value tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    cases = [
        ("zero_input", 1, 4, 128, 128, 128, None, 0.0),  # all-zero Q/K/V/sinks
        ("large_input", 1, 4, 128, 128, 128, None, 10.0),  # large values (stability)
    ]
    for name, b, h, sq, skv, d, window, scale in cases:
        _run_boundary_case(name, b, h, sq, skv, d, window, scale, block_M, block_N)


# ===========================================================================
# Performance / benchmark helpers (integrated from perf_mha_sink_fwd_bhsd.py)
# Activated via `--level perf`. The standalone perf_mha_sink_fwd_bhsd.py is
# kept for backward compatibility; both paths share the same kernel + golden.
# ===========================================================================


def build_inputs_for_bench(batch, heads, seq_q, seq_kv, dim, window_size, device, dtype, block_M, block_N):
    """Build padded 4D inputs + sinks + mask tensor (mirrors _prepare_and_run)."""
    torch.manual_seed(0)

    # Original (unpadded) tensors for golden
    q_orig = torch.randn(batch, heads, seq_q, dim, dtype=dtype, device=device)
    k_orig = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)
    v_orig = torch.randn(batch, heads, seq_kv, dim, dtype=dtype, device=device)
    sinks = torch.randn(heads, dtype=dtype, device=device)

    # Pad to block-aligned for kernel
    padded_q = ((seq_q + block_M - 1) // block_M) * block_M
    padded_kv = ((seq_kv + block_N - 1) // block_N) * block_N
    q = torch.zeros(batch, heads, padded_q, dim, dtype=dtype, device=device)
    q[:, :, :seq_q, :] = q_orig
    k = torch.zeros(batch, heads, padded_kv, dim, dtype=dtype, device=device)
    k[:, :, :seq_kv, :] = k_orig
    v = torch.zeros(batch, heads, padded_kv, dim, dtype=dtype, device=device)
    v[:, :, :seq_kv, :] = v_orig

    # Pre-broadcast sinks [heads] -> [heads, padded_q] (avoid T.Parallel broadcast bug)
    sinks_broad = sinks.unsqueeze(1).expand(-1, padded_q).contiguous()

    # Build causal + optional sliding window mask (padded, shared across batch/head)
    mask = build_causal_mask(seq_q, seq_kv, window_size, device, block_M, block_N)

    return q, k, v, sinks_broad, mask, sinks


def bench_tilelang(kernel, q, k, v, sinks_broad, mask, device):
    """Benchmark the TileLang kernel via do_bench (returns ms, median)."""

    def f():
        kernel(q, k, v, sinks_broad, mask)

    # warmup + repeat aligned with perf_gqa_fwd_varlen.py convention
    latency = do_bench(f, _n_warmup=10, _n_repeat=20, return_mode="median")
    return latency


def bench_golden(q, k, v, sinks, heads, seq_q, seq_kv, dim, window_size, device):
    """Benchmark the PyTorch golden (ref_program)."""
    # Slice back to original (unpadded) for golden
    q_orig = q[:, :, :seq_q, :]
    k_orig = k[:, :, :seq_kv, :]
    v_orig = v[:, :, :seq_kv, :]

    def f():
        ref_program(q_orig, k_orig, v_orig, sinks, sliding_window=window_size)
        torch.npu.synchronize()

    latency = do_bench(f, _n_warmup=5, _n_repeat=10, return_mode="median")
    return latency


def compute_flops(batch, heads, seq_q, seq_kv, dim, window_size):
    """Flash attention FLOPs: 2 matmuls (QK^T and PV).

    Causal attention halves the work (lower-triangular).
    Sliding window further reduces work, but we approximate with causal factor.
    """
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total = 2 * flops_per_matmul
    # Causal halves the work (lower-triangular)
    total *= 0.5
    if window_size is not None:
        # Sliding window further reduces; approximate with window/seq_kv ratio
        effective = min(window_size, seq_kv) / seq_kv
        total *= effective
    return total


def run_one(name, batch, heads, seq_q, seq_kv, dim, window_size, block_M, block_N, with_golden, device, dtype):
    """Run a single benchmark config and print results. Returns True on success."""
    print(
        f"\n[{name}] batch={batch} heads={heads} "
        f"seq_q={seq_q} seq_kv={seq_kv} dim={dim} "
        f"window={window_size} block_M={block_M} block_N={block_N}"
    )

    # Build inputs
    q, k, v, sinks_broad, mask, sinks = build_inputs_for_bench(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        window_size,
        device,
        dtype,
        block_M,
        block_N,
    )

    # Compile kernel (compilation cost is NOT counted in bench)
    padded_q = ((seq_q + block_M - 1) // block_M) * block_M
    padded_kv = ((seq_kv + block_N - 1) // block_N) * block_N
    print("  compiling kernel ...")
    kernel = flashattn(
        batch,
        heads,
        padded_q,
        padded_kv,
        dim,
        block_M=block_M,
        block_N=block_N,
        has_window=(window_size is not None),
        real_seq_q=seq_q,
        real_seq_kv=seq_kv,
    )

    # Quick correctness check before bench (so we don't bench a broken kernel)
    out = kernel(q, k, v, sinks_broad, mask)
    torch.npu.synchronize()
    if torch.isnan(out).any():
        print("  [ERROR] kernel output contains NaN, skipping bench")
        return False

    ref_out = ref_program(
        q[:, :, :seq_q, :],
        k[:, :, :seq_kv, :],
        v[:, :, :seq_kv, :],
        sinks,
        sliding_window=window_size,
    )
    torch.npu.synchronize()
    # Compare on visible Q rows only (padded rows are 0 in both)
    out_v = out[:, :, :seq_q, :].cpu()
    ref_v = ref_out[:, :, :seq_q, :].cpu()
    max_diff = (out_v.float() - ref_v.float()).abs().max().item()
    print(f"  correctness: max_diff={max_diff:.6e} (atol=1e-2)")
    if max_diff >= 1e-2:
        print(f"  [ERROR] correctness check failed: max_diff={max_diff:.6e} >= atol=1e-2")
        return False

    # Bench TileLang kernel
    print("  benching TileLang kernel ...")
    tl_ms = bench_tilelang(kernel, q, k, v, sinks_broad, mask, device)
    flops = compute_flops(batch, heads, seq_q, seq_kv, dim, window_size)
    tl_tflops = flops / (tl_ms * 1e-3) * 1e-12
    print(f"  TileLang:  {tl_ms:.4f} ms   {tl_tflops:.2f} TFlops")

    if with_golden:
        print("  benching PyTorch golden ...")
        gold_ms = bench_golden(
            q,
            k,
            v,
            sinks,
            heads,
            seq_q,
            seq_kv,
            dim,
            window_size,
            device,
        )
        gold_tflops = flops / (gold_ms * 1e-3) * 1e-12
        speedup = gold_ms / tl_ms if tl_ms > 0 else float("inf")
        print(f"  Golden:    {gold_ms:.4f} ms   {gold_tflops:.2f} TFlops")
        print(f"  Speedup:   {speedup:.2f}x  (TileLang vs PyTorch golden)")

    return True


def run_perf_mode(args):
    """Performance benchmark mode (integrated from perf_mha_sink_fwd_bhsd.py).

    Supports preset suites (default / sweep / small / window-sweep) or
    single-config run via --batch / --heads / --seq-q / --seq-kv / --dim /
    --window / --block-M / --block-N / --with-golden.
    """
    device = "npu"
    dtype = torch.float16

    results = []

    if args.preset == "default":
        results.append(
            run_one(
                "default",
                args.batch,
                args.heads,
                args.seq_q,
                args.seq_kv,
                args.dim,
                args.window,
                args.block_M,
                args.block_N,
                args.with_golden,
                device,
                dtype,
            )
        )
    elif args.preset == "small":
        # Quick smoke run (faster compile + bench).
        results.append(
            run_one(
                "small",
                1,
                4,
                256,
                256,
                128,
                None,
                args.block_M,
                args.block_N,
                args.with_golden,
                device,
                dtype,
            )
        )
    elif args.preset == "sweep":
        # Vary seqlen (the main perf axis for attention).
        print("=" * 70)
        print("Preset: sweep seqlen (batch=8, heads=32, dim=128, full attention)")
        print("=" * 70)
        for sq in [512, 1024, 2048, 4096]:
            results.append(
                run_one(
                    f"sq{sq}",
                    8,
                    32,
                    sq,
                    sq,
                    128,
                    None,
                    args.block_M,
                    args.block_N,
                    args.with_golden,
                    device,
                    dtype,
                )
            )
    elif args.preset == "window-sweep":
        # Vary window size (sliding window attention).
        print("=" * 70)
        print("Preset: window-sweep (batch=8, heads=32, sq=skv=4096, dim=128)")
        print("=" * 70)
        for w in [512, 1024, 2048, 4096]:
            results.append(
                run_one(
                    f"window{w}",
                    8,
                    32,
                    4096,
                    4096,
                    128,
                    w,
                    args.block_M,
                    args.block_N,
                    args.with_golden,
                    device,
                    dtype,
                )
            )

    print("\nDone.")
    # CI compatibility: bench_test.sh marks a script PASSED only if stdout
    # contains "Test Passed!" / "Kernel Output Match!". Print it when all
    # run_one correctness checks passed; exit 1 otherwise.
    if all(results):
        print("Test Passed!")
    else:
        sys.exit(1)


# ===========================================================================
# Main entry
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Attention Sink MHA Forward (Ascend NPU, Expert)")
    parser.add_argument(
        "--level",
        default="default",
        choices=["default", "l0", "l1", "l2", "boundary", "all", "perf"],
        help="Test level to run. 'default' (no args) = L0 precision + default-shape perf with golden. 'perf' = perf only.",
    )
    # Perf benchmark args (only effective when --level in {default, perf}). These mirror
    # perf_mha_sink_fwd_bhsd.py so the same CLI surface is available in-test.
    parser.add_argument("--batch", type=int, default=8, help="[perf] batch size")
    parser.add_argument("--heads", type=int, default=32, help="[perf] attention heads")
    parser.add_argument("--seq-q", type=int, default=4096, help="[perf] query sequence length")
    parser.add_argument("--seq-kv", type=int, default=4096, help="[perf] key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="[perf] head dim")
    parser.add_argument("--window", type=int, default=None, help="[perf] sliding window size (default: None = full attention)")
    parser.add_argument("--block-M", type=int, default=128, help="[perf] Q block size (Expert kernel fixed at 128)")
    parser.add_argument("--block-N", type=int, default=128, help="[perf] K/V block size (Expert kernel fixed at 128)")
    parser.add_argument(
        "--with-golden",
        action="store_true",
        default=True,
        help="[perf] also benchmark PyTorch golden for speedup comparison (default: ON in 'default' mode)",
    )
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "sweep", "small", "window-sweep"],
        help="[perf] preset benchmark suite (overrides individual args)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    if args.level == "perf":
        # Performance benchmark mode only (integrated from perf_mha_sink_fwd_bhsd.py).
        run_perf_mode(args)
        return

    if args.level == "default":
        # Default mode: run L0 precision gate + one perf bench (default shape, with golden).
        # This is what users get when they run the script with no arguments.
        print("=" * 70)
        print("Stage 1: L0 precision gate")
        print("=" * 70)
        blocking_ok = test_mha_sink_fwd_bhsd_l0()
        print()
        print("=" * 70)
        print("Stage 2: performance benchmark (default shape, with golden)")
        print("=" * 70)
        run_perf_mode(args)
        if not blocking_ok:
            sys.exit(1)
        return

    blocking_ok = True  # Only L0/L1 count toward blocking

    if args.level in ("l0", "all"):
        blocking_ok &= test_mha_sink_fwd_bhsd_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_mha_sink_fwd_bhsd_l1()
    if args.level in ("l2", "all"):
        test_mha_sink_fwd_bhsd_l2()
    if args.level in ("boundary", "all"):
        test_mha_sink_fwd_bhsd_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
