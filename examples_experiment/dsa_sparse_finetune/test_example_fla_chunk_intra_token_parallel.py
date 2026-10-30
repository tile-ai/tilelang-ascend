"""fla_chunk_intra_token_parallel example test: representative L0 + L1 cases
with a PyTorch golden.

q/k/beta bf16 + gk fp32 (chunk-local cumsum of logsigmoid gates) ->
(Aqk [B,S,H,CS] bf16, Akk [B,S,H,SCS] bf16).

Precision: mixed-tolerance dual gate inlined per dtype
(|actual - golden| <= atol + rtol * |golden| element-wise, plus
matched_ratio >= 0.99 and max_abs_error <= limit overall).
"""

import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401  (registers .npu() on torch)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_fla_chunk_intra_token_parallel import fla_chunk_intra_token_parallel  # noqa: E402


# ============================================================================
# Input construction (upstream prepare_input semantics, all on CPU)
# ============================================================================
def make_inputs(B, S, H, DK, chunk, seed):
    """q/k/beta randn bf16; gk = logsigmoid(randn fp32) followed by a
    chunk-local cumsum (per-chunk reset, aligning the upstream cumsum)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, S, H, DK, generator=gen).to(torch.bfloat16)
    beta = torch.randn(B, S, H, generator=gen).to(torch.bfloat16)
    gk = torch.nn.functional.logsigmoid(torch.randn(B, S, H, DK, generator=gen, dtype=torch.float32))
    gk = gk.view(B, S // chunk, chunk, H, DK).cumsum(dim=2).view(B, S, H, DK)
    cpu = (q, k, gk, beta)
    npu = tuple(t.npu() for t in cpu)
    return npu, cpu


# ============================================================================
# Golden reference (PyTorch, fp32 domain, torch.exp2 exact semantics)
# ============================================================================
def fla_chunk_intra_token_parallel_ref(q, k, gk, beta, chunk_size=64, sub_chunk_size=16):
    """Vectorized golden. q/k [B,S,H,DK] bf16, gk [B,S,H,DK] fp32,
    beta [B,S,H] bf16 -> (Aqk [B,S,H,CS] bf16, Akk [B,S,H,SCS] bf16).
    Cleared-row semantics: uncovered Aqk columns 0, Akk diagonal 0."""
    B, S, H, DK = q.shape
    CS, SCS = chunk_size, sub_chunk_size
    assert S % CS == 0 and CS % SCS == 0
    qf, kf, bf, gf = q.float(), k.float(), beta.float(), gk.float()
    Aqk = torch.zeros(B, S, H, CS, dtype=torch.bfloat16)
    Akk = torch.zeros(B, S, H, SCS, dtype=torch.bfloat16)
    for c in range(S // CS):
        for s in range(CS // SCS):
            i_ts = c * CS + s * SCS
            sl = slice(i_ts, i_ts + SCS)
            g_sub, k_sub = gf[:, sl], kf[:, sl]  # [B,SCS,H,DK]
            e = torch.exp2(g_sub.unsqueeze(2) - g_sub.unsqueeze(1))  # [B,SCS(i),SCS(j),H,DK]
            kgj = k_sub.unsqueeze(1) * e  # k_j * 2^(g_i - g_j)
            aqk = torch.einsum("bihd,bijhd->bijh", qf[:, sl], kgj)
            akk = torch.einsum("bihd,bijhd->bijh", bf[:, sl].unsqueeze(-1) * k_sub, kgj)
            # masks broadcast over (i, j) = dims 1, 2 of the [B,SCS(i),SCS(j),H] tiles
            keep = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool))[None, :, :, None]  # j <= i
            keep_strict = torch.tril(torch.ones(SCS, SCS, dtype=torch.bool), -1)[None, :, :, None]  # j < i
            aqk = aqk.masked_fill(~keep, 0.0)
            akk = akk.masked_fill(~keep_strict, 0.0)
            Aqk[:, sl, :, s * SCS : s * SCS + SCS] = aqk.permute(0, 1, 3, 2).to(torch.bfloat16)
            Akk[:, sl, :, :] = akk.permute(0, 1, 3, 2).to(torch.bfloat16)
    return Aqk, Akk


def check_output(name, out, ref, atol, rtol, max_abs_limit, required_ratio):
    """Inlined mixed-tolerance dual gate (element-wise |a-g| <= atol+rtol*|g|,
    overall matched_ratio >= required and max_abs <= limit)."""
    y_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    m = torch.isfinite(ref_cpu)
    abs_err = (y_cpu[m] - ref_cpu[m]).abs()
    ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"{name} precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
    print(f"Test pass! {name}: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")


def make_deep_gate_inputs(DK, seed):
    """All-ones q/k/beta and gk[t] = -1e8 - 8*t: deep absolute gates with
    small raw-domain differences. 2^-8 must be computed from the RAW
    difference (g_1 - g_0 = -8 exactly at 1-ulp gate spacing); scaling by
    ln2 BEFORE the subtraction would round the scaled values together and
    yield 2^0 = 1. Aqk[1, 0] must read 32 * 2^-8 = 0.125."""
    q = torch.ones(1, 64, 8, DK, dtype=torch.bfloat16)
    k = torch.ones(1, 64, 8, DK, dtype=torch.bfloat16)
    beta = torch.ones(1, 64, 8, dtype=torch.bfloat16)
    t = torch.arange(64, dtype=torch.float32)
    gk = (-1e8 - 8 * t).view(1, 64, 1, 1).expand(1, 64, 8, DK).contiguous()
    cpu = (q, k, gk, beta)
    npu = tuple(x.npu() for x in cpu)
    return npu, cpu


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    # Representative configs: L0 benchmark structure + L1 regressions
    test_configs = [
        ("l0_bench_repr", (1, 1024, 64, 128), 1001, "L0"),  # H=64, full BH=8 head groups
        ("l1_aligned", (2, 128, 8, 128), 2101, "L1"),  # multi-batch regular shape
        # deep absolute gates: raw-domain anchor subtraction (review fix)
        ("l1_deep_gate", (1, 64, 8, 32), 2201, "L1"),
        # DK=256: BH adapts 8 -> 2 under the 192KB UB coexistence budget
        ("l1_dk256_bh2", (1, 64, 8, 256), 2301, "L1"),
    ]

    for name, shape, seed, level in test_configs:
        B, S, H, DK = shape
        print(f"Testing fla_chunk_intra_token_parallel {level}/{name} with shape={shape}")
        if name == "l1_deep_gate":
            npu_in, cpu_in = make_deep_gate_inputs(DK, seed)
        else:
            npu_in, cpu_in = make_inputs(B, S, H, DK, 64, seed)
        aqk, akk = fla_chunk_intra_token_parallel(*npu_in)
        torch.npu.synchronize()
        print("Init successful!")
        g_aqk, g_akk = fla_chunk_intra_token_parallel_ref(*cpu_in)
        if name == "l1_deep_gate":
            # the exact sentinel from the review: Aqk[1, 0] = 32 * 2^-8
            v = aqk[0, 1, 0, 0].item()
            assert abs(v - 0.125) < 1e-3, f"deep-gate sentinel: Aqk[1,0]={v}, expected 0.125"
            print(f"Test pass! deep-gate sentinel: Aqk[1,0] = {v} (expected 0.125)")
        # bf16 mixed tolerance
        atol, rtol, max_abs_limit, required_ratio = 2**-10, 2**-6, 1e0, 0.99
        check_output("Aqk", aqk, g_aqk, atol, rtol, max_abs_limit, required_ratio)
        check_output("Akk", akk, g_akk, atol, rtol, max_abs_limit, required_ratio)

    # Domain gates: out-of-budget / non-pow2 configs must be rejected cleanly
    q = torch.randn(1, 64, 8, 512).to(torch.bfloat16).npu()
    k = torch.randn(1, 64, 8, 512).to(torch.bfloat16).npu()
    beta = torch.randn(1, 64, 8).to(torch.bfloat16).npu()
    gk = torch.randn(1, 64, 8, 512, dtype=torch.float32).npu()
    try:
        fla_chunk_intra_token_parallel(q, k, gk, beta)
        raise AssertionError("DK=512 must be rejected (192KB UB budget)")
    except ValueError as exc:
        print(f"Test pass! DK=512 rejected: {str(exc)[:80]}...")
    try:
        fla_chunk_intra_token_parallel(
            q[:, :, :, :32].contiguous(), k[:, :, :, :32].contiguous(), gk[:, :, :, :32].contiguous(), beta, chunk_size=48, sub_chunk_size=3
        )
        raise AssertionError("CS=48/SCS=3 must be rejected (pow2 tile alignment)")
    except ValueError as exc:
        print(f"Test pass! CS=48/SCS=3 rejected: {str(exc)[:80]}...")

    print("Kernel Output Match!")
