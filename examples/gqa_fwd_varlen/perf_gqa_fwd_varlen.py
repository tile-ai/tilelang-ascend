import argparse
import sys

import torch

# Import kernel + helpers from the example module (same directory).
# Make sure the example dir is on sys.path.
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gqa_fwd_varlen import (  # noqa: E402
    flashattn,
    generate_random_padding_mask,
    mask_to_cu_seqlens,
    build_attention_mask,
)
from test_gqa_fwd_varlen import ref_gqa_varlen_fwd_padded  # noqa: E402

from tilelang.profiler import do_bench  # noqa: E402


def _ceildiv(a, b):
    return (a + b - 1) // b


def build_inputs(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode, device, dtype, block_M, block_N):
    """Build padded 4D inputs + mask tensor (mirrors _prepare_and_run in example)."""
    torch.manual_seed(0)
    head_kv = heads // groups

    # Pad seqlens to block_M/block_N multiples to avoid GM OOB reads
    padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
    padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N

    q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
    q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device)
    k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)
    v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)

    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
    cu_seqlens_q = mask_to_cu_seqlens(q_mask)
    cu_seqlens_k = mask_to_cu_seqlens(k_mask)
    attn_mask = build_attention_mask(
        cu_seqlens_q,
        cu_seqlens_k,
        padded_sq,
        padded_skv,
        is_causal,
        device,
    )
    return q, k, v, attn_mask, cu_seqlens_q, cu_seqlens_k, padded_sq, padded_skv


def bench_tilelang(kernel, q, k, v, attn_mask, device):
    """Benchmark the TileLang kernel via do_bench (returns ms, median)."""

    def f():
        kernel(q, k, v, attn_mask)

    # warmup + repeat aligned with flash_attn_bshd_developer.py convention
    latency = do_bench(f, _n_warmup=5, _n_repeat=5, return_mode="mean")
    return latency


def bench_golden(q, k, v, cu_seqlens_q, cu_seqlens_k, heads, groups, dim, is_causal, device):
    """Benchmark the PyTorch golden (per-batch loop + einsum)."""

    def f():
        ref_gqa_varlen_fwd_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()

    latency = do_bench(f, _n_warmup=5, _n_repeat=10, return_mode="median")
    return latency


def compute_flops(batch, heads, q_seqlen, k_seqlen, dim, is_causal):
    """Flash attention FLOPs: 2 matmuls (QK^T and PV)."""
    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    total = 2 * flops_per_matmul
    if is_causal:
        total *= 0.5
    return total


def run_one(
    name,
    batch,
    heads,
    groups,
    q_seqlen,
    k_seqlen,
    dim,
    is_causal,
    padding_mode,
    block_M,
    block_N,
    num_stages,
    cross_interval,
    with_golden,
    device,
    dtype,
):
    """Run a single benchmark config and print results."""
    head_kv = heads // groups
    print(
        f"\n[{name}] batch={batch} heads={heads} groups={groups} head_kv={head_kv} "
        f"q_seqlen={q_seqlen} k_seqlen={k_seqlen} dim={dim} "
        f"causal={is_causal} pad={padding_mode} block_M={block_M} block_N={block_N}"
    )

    # Build inputs
    q, k, v, attn_mask, cu_seqlens_q, cu_seqlens_k, padded_sq, padded_skv = build_inputs(
        batch,
        heads,
        groups,
        q_seqlen,
        k_seqlen,
        dim,
        is_causal,
        padding_mode,
        device,
        dtype,
        block_M,
        block_N,
    )

    # Compile kernel (compilation cost is NOT counted in bench)
    # Skip mask only when non-causal + full padding + no block padding
    has_block_padding = (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)
    apply_mask = is_causal or padding_mode != "full" or has_block_padding

    print("  compiling kernel ...")
    kernel = flashattn(
        batch,
        groups,
        heads,
        dim,
        padded_sq,
        padded_skv,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        cross_interval=cross_interval,
        apply_mask=apply_mask,
    )

    # Quick correctness check before bench (so we don't bench a broken kernel)
    out = kernel(q, k, v, attn_mask)
    torch.npu.synchronize()
    if torch.isnan(out).any():
        print("  [ERROR] kernel output contains NaN, skipping bench")
        return False
    ref_out = ref_gqa_varlen_fwd_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()
    # Compare on visible Q rows only (padding rows are 0/NaN in both).
    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    # Extend q_mask to padded_sq
    if padded_sq > q_seqlen:
        qm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
        qm_padded[:, :q_seqlen] = q_mask
        q_mask = qm_padded
    out_perm = out.permute(0, 2, 1, 3).contiguous()
    ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()
    valid = q_mask
    out_v = out_perm[valid].cpu()
    ref_v = ref_perm[valid].cpu()
    if torch.isnan(out_v).any():
        # drop NaN rows (causal invisible Q rows)
        non_nan = ~torch.isnan(out_v).any(dim=-1)
        out_v = out_v[non_nan]
        ref_v = ref_v[non_nan]
    max_diff = (out_v.float() - ref_v.float()).abs().max().item()
    print(f"  correctness: max_diff={max_diff:.6e} (atol=1e-2)")
    if max_diff >= 1e-2:
        print(f"  [ERROR] correctness check failed: max_diff={max_diff:.6e} >= atol=1e-2")
        return False

    # Bench TileLang kernel
    print("  benching TileLang kernel ...")
    tl_ms = bench_tilelang(kernel, q, k, v, attn_mask, device)
    flops = compute_flops(batch, heads, q_seqlen, k_seqlen, dim, is_causal)
    tl_tflops = flops / (tl_ms * 1e-3) * 1e-12
    print(f"  TileLang:  {tl_ms:.4f} ms   {tl_tflops:.2f} TFlops")

    if with_golden:
        print("  benching PyTorch golden ...")
        gold_ms = bench_golden(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
            device,
        )
        gold_tflops = flops / (gold_ms * 1e-3) * 1e-12
        speedup = gold_ms / tl_ms if tl_ms > 0 else float("inf")
        print(f"  Golden:    {gold_ms:.4f} ms   {gold_tflops:.2f} TFlops")
        print(f"  Speedup:   {speedup:.2f}x  (TileLang vs PyTorch golden)")

    return True


def main():
    parser = argparse.ArgumentParser(description="GQA varlen Flash Attention (Ascend) performance benchmark")
    # Default shape matches the GPU source's main() defaults.
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="query heads")
    parser.add_argument("--groups", type=int, default=16, help="GQA groups")
    parser.add_argument("--q-seqlen", type=int, default=2048, help="Q sequence length")
    parser.add_argument("--k-seqlen", type=int, default=2048, help="K/V sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--causal", action="store_true", help="causal attention")
    parser.add_argument(
        "--padding", default="full", choices=["full", "random", "third"], help="padding mode (full = no padding / max length)"
    )
    parser.add_argument("--block-M", type=int, default=128, help="Q block size")
    parser.add_argument("--block-N", type=int, default=128, help="K/V block size")
    parser.add_argument("--num-stages", type=int, default=8, help="pipeline depth")
    parser.add_argument(
        "--cross-interval",
        type=int,
        default=1,
        help="cross-core sync interval (iter4 sweep: 2 is faster but has sync bug with apply_mask=False, keep 1 for safety)",
    )
    parser.add_argument("--with-golden", action="store_true", help="also benchmark PyTorch golden for speedup comparison")
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "sweep", "small", "causal-sweep"],
        help="preset benchmark suite (overrides individual args)",
    )
    args = parser.parse_args()

    results = []
    device = "npu"
    dtype = torch.float16

    if args.preset == "default":
        results.append(
            run_one(
                "default",
                args.batch,
                args.heads,
                args.groups,
                args.q_seqlen,
                args.k_seqlen,
                args.dim,
                args.causal,
                args.padding,
                args.block_M,
                args.block_N,
                args.num_stages,
                args.cross_interval,
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
                2,
                128,
                128,
                128,
                False,
                "full",
                args.block_M,
                args.block_N,
                args.num_stages,
                args.cross_interval,
                args.with_golden,
                device,
                dtype,
            )
        )
    elif args.preset == "sweep":
        # Vary seqlen (the main perf axis for attention).
        print("=" * 70)
        print("Preset: sweep seqlen (batch=8, heads=64, groups=16, dim=128, non-causal)")
        print("=" * 70)
        for sq in [512, 1024, 2048, 4096]:
            results.append(
                run_one(
                    f"sq{sq}",
                    8,
                    64,
                    16,
                    sq,
                    sq,
                    128,
                    False,
                    "full",
                    args.block_M,
                    args.block_N,
                    args.num_stages,
                    args.cross_interval,
                    args.with_golden,
                    device,
                    dtype,
                )
            )
    elif args.preset == "causal-sweep":
        print("=" * 70)
        print("Preset: causal-sweep (causal=True, vary seqlen)")
        print("=" * 70)
        for sq in [512, 1024, 2048, 4096]:
            results.append(
                run_one(
                    f"sq{sq}_causal",
                    8,
                    64,
                    16,
                    sq,
                    sq,
                    128,
                    True,
                    "full",
                    args.block_M,
                    args.block_N,
                    args.num_stages,
                    args.cross_interval,
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


if __name__ == "__main__":
    main()
