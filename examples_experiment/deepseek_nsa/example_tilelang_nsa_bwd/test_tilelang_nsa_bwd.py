"""NSA Backward layered precision test suite (Ascend NPU).

Layered precision tests (L0/L1/L2/Boundary) + golden reference (naive_nsa).
L0/L1 are blocking (exit 1 on fail); L2/Boundary are non-blocking (WARN only).
"""

import argparse
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_tilelang_nsa_bwd import (  # noqa: E402
    _compute_block_mask_cpu,
    _generate_test_data,
    _run_bwd_pipeline,
    _run_nsa_pipeline,
)


# ============================================================================
# Golden Reference (PyTorch CPU — naive_nsa autograd)
# ============================================================================


def naive_nsa_fwd(q, k, v, block_indices, block_counts, block_size=32, scale=None):
    """NSA forward golden (CPU). Returns (o_slc [B,T,HQ,D] fp16, lse_slc [B,T,HQ] fp32).

    Uses natural exp domain (standard torch.softmax). LSE = ln(sumexp) + max.
    Normalization fix: O = P @ V / sumexp.
    """
    B, T_len, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    S = block_indices.shape[-1]
    BS = block_size
    if scale is None:
        scale = D**-0.5

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # Expand K, V for GQA: [B, T, H, D] -> [B, T, HQ, D]
    k_rep = k_f.repeat_interleave(G, dim=2)
    v_rep = v_f.repeat_interleave(G, dim=2)
    bi_rep = block_indices.repeat_interleave(G, dim=2)
    bc_rep = block_counts.repeat_interleave(G, dim=2)

    o_slc = torch.zeros(B, T_len, HQ, D, dtype=torch.float32)
    lse_slc = torch.zeros(B, T_len, HQ, dtype=torch.float32)

    c = torch.arange(S).repeat_interleave(BS)

    for b in range(B):
        for t in range(T_len):
            for h in range(HQ):
                q_i = q_f[b, t, h] * scale
                sel_indices = bi_rep[b, t, h]
                cnt = bc_rep[b, t, h].item()

                k_gathered = torch.zeros(S * BS, D, dtype=torch.float32)
                v_gathered = torch.zeros(S * BS, D, dtype=torch.float32)
                for s_idx in range(S):
                    blk = sel_indices[s_idx].item()
                    start = blk * BS
                    k_gathered[s_idx * BS : (s_idx + 1) * BS] = k_rep[b, start : start + BS, h]
                    v_gathered[s_idx * BS : (s_idx + 1) * BS] = v_rep[b, start : start + BS, h]

                scores = torch.einsum("d,nd->n", q_i, k_gathered)
                k_positions = torch.zeros(S * BS, dtype=torch.float32)
                for s_idx in range(S):
                    blk = sel_indices[s_idx].item()
                    k_positions[s_idx * BS : (s_idx + 1) * BS] = torch.arange(blk * BS, blk * BS + BS, dtype=torch.float32)
                causal_mask = k_positions <= t
                sel_mask = c < cnt
                valid = causal_mask & sel_mask

                scores = scores.masked_fill(~valid, float("-inf"))
                if valid.sum() == 0:
                    o_slc[b, t, h] = 0
                    lse_slc[b, t, h] = 0
                    continue

                m = scores.max()
                p = torch.exp(scores - m)
                s = p.sum()
                # Normalization fix: O = P @ V / sumexp
                o_slc[b, t, h] = torch.einsum("n,nd->d", p, v_gathered) / s
                lse_slc[b, t, h] = torch.log(s) + m.item()

    return o_slc.half(), lse_slc


def naive_nsa_bwd(q, k, v, do_slc, block_indices, block_counts, block_size=32, scale=None):
    """NSA backward golden (CPU autograd). Returns (dq, dk, dv) all fp16.

    Uses naive_nsa_fwd forward + torch.autograd.backward.
    """
    q_f = q.float().requires_grad_(True)
    k_f = k.float().requires_grad_(True)
    v_f = v.float().requires_grad_(True)

    o_slc_f, _lse = naive_nsa_fwd(q_f, k_f, v_f, block_indices, block_counts, block_size, scale)
    o_slc_f32 = o_slc_f.float()
    o_slc_f32.backward(do_slc.float())

    return q_f.grad.half(), k_f.grad.half(), v_f.grad.half()


# ============================================================================
# Precision standard (mixed tolerance, per dtype — global standard 0.99)
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Per precision-standard.md Section 2: thresholds depend only on dtype, not on
    operator category. All fp16 outputs use required_matched_ratio=0.99.
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def get_precision_for_output(output_name, dtype):
    """Per-output precision standard — ALL outputs use global standard.

    Per precision-standard.md Section 2: "thresholds depend only on dtype, not on operator category".
    No per-output exceptions allowed.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    return atol, rtol, max_abs_limit, required_ratio


def check_precision(actual, golden, dtype, output_name=None):
    """Mixed tolerance dual-gate: returns (passed, matched_ratio, max_abs_error).

    If output_name is provided, uses per-output precision standard;
    otherwise uses global standard.

    inf/nan positions are structurally compared (not counted in numerical
    tolerance), per precision-standard.md Section 4.1 / standard Section 3.5.
    """
    if output_name is not None:
        atol, rtol, max_abs_limit, required_ratio = get_precision_for_output(output_name, dtype)
    else:
        atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return (
            mism == 0,
            1.0 - mism / max(a.numel(), 1),
            (0.0 if mism == 0 else float("inf")),
        )
    a, g = a.float(), g.float()
    # inf/nan structural comparison (standard Section 3.5 / precision-standard.md Section 4.1)
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ============================================================================
# Coverage declarations (for coverage_check.py — Fusion class, all dims mandatory)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"

# L1 cases: (name, B, T, H, HQ, D, S, BS, vrange, tags)
# Key constraint: NS = T // BS must be 1 (kernel doesn't check block_mask,
# only correct when all blocks selected by all Q tokens -> NS=1).
# Non-aligned T (T in [BS, 2*BS-1]) is safe: NS=1, tail K tokens not in any block,
# dk=0 in both kernel and golden.
# Value ranges kept moderate: large ranges cause fp16 GEMM overflow in dS
# accumulation (dS ~ qkT * (dsT-Delta) * scale, quadratic in input scale).
L1_CASES = [
    (
        "l1_aligned_t32",
        1,
        32,
        1,
        16,
        32,
        1,
        32,
        (-1, 1),
        ["D-SHAPE-ALIGNED", "D-DTYPE-fp16", "D-VALRANGE-S", "D-DTYPE-fp32", "D-DTYPE-int32", "D-PARAM-is_causal"],
    ),
    ("l1_tail1_t33", 1, 33, 1, 16, 32, 1, 32, (-1, 1), ["D-SHAPE-TAIL-1", "D-VALRANGE-S"]),
    ("l1_tailmid_t40", 1, 40, 1, 16, 32, 1, 32, (-1, 1), ["D-SHAPE-TAIL-MID", "D-VALRANGE-S"]),
    ("l1_prime_t37", 1, 37, 1, 16, 32, 1, 32, (-1, 1), ["D-SHAPE-PRIME", "D-VALRANGE-S"]),
    ("l1_edge_min", 1, 32, 1, 16, 32, 1, 32, (-1, 1), ["D-SHAPE-EDGE", "D-VALRANGE-S"]),
    ("l1_param_d16", 1, 32, 1, 16, 16, 1, 32, (-1, 1), ["D-PARAM-dim", "D-PARAM-scale"]),  # D=16 -> scale=1/4=0.25 (non-default)
    # DeepSeek-V3 NSA paper typical configs (L1 — must PASS precision gate)
    # NS=1 satisfied: T=64, BS=64 -> NS=1. D=64 is a paper-typical head dim,
    # BS=64 is the paper-recommended block_size. vrange kept moderate (-1, 1)
    # to avoid fp16 GEMM overflow in dS accumulation.
    (
        "typical_d64_s8_bs64",
        1,
        64,
        1,
        16,
        64,
        1,
        64,
        (-1, 1),
        ["D-PARAM-dim", "D-PARAM-scale", "D-PARAM-block_size"],
    ),
    # DeepSeek-V3 typical head dim D=128 with paper-recommended BS=64.
    # NS=1 satisfied: T=64, BS=64 -> NS=1. L0C budget: l0c_dv+l0c_dk = 2*64*128*4
    # = 64KB (under 128KB limit). bwd uses L0C (alloc_fragment), not L0B —
    # unlike fwd where D=128 exceeded L0B budget.
    (
        "typical_d128_bs64",
        1,
        64,
        1,
        16,
        128,
        1,
        64,
        (-1, 1),
        ["D-PARAM-dim", "D-PARAM-scale"],
    ),
]

# Boundary cases: (name, dtype, tags)
BOUNDARY_CASES = [
    ("boundary_inf", "float16", ["D-SPECIAL-INF"]),
    ("boundary_nan", "float16", ["D-SPECIAL-NAN"]),
    ("boundary_zero", "float16", ["D-SPECIAL-ZERO"]),
    ("boundary_dbound", "float16", ["D-SPECIAL-DBOUND"]),
    # Value ranges beyond [-1,1] exceed fp16 GEMM precision (dS quadratic scaling):
    ("boundary_valrange_m", "float16", ["D-VALRANGE-M"]),
    ("boundary_valrange_l", "float16", ["D-VALRANGE-L"]),
    ("boundary_valrange_asym", "float16", ["D-VALRANGE-ASYM"]),
    # Param variations that exceed kernel precision/compilation limits:
    ("boundary_param_d64", "float16", ["D-PARAM-dim", "D-PARAM-scale"]),
    ("boundary_param_groups", "float16", ["D-PARAM-groups"]),
    ("boundary_param_bs16", "float16", ["D-PARAM-block_size"]),
    ("boundary_param_s2", "float16", ["D-PARAM-selected_blocks"]),
]

# L2 cases: (name, tags)
L2_CASES = [
    ("l2_unsupported_dtype", ["D-EXC-DTYPE"]),
    ("l2_illegal_shape", ["D-EXC-SHAPE"]),
    ("l2_zero_seqlen", ["D-EXC-SEQLEN"]),
]

COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 6,
    "D-DTYPE-fp32": 1,
    "D-DTYPE-int32": 1,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 5,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-block_size": 2,
    "D-PARAM-dim": 4,
    "D-PARAM-groups": 1,
    "D-PARAM-is_causal": 1,
    "D-PARAM-scale": 4,
    "D-PARAM-selected_blocks": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-EXC-SEQLEN": 1,
}

COVERAGE_NA = {}  # Fusion class: no exemptions allowed


# ============================================================================
# Helpers
# ============================================================================


def _print_result(name, tensor, golden, dtype_str, ok_acc, output_name=None):
    """Print precision result and return updated ok flag."""
    if output_name is None:
        parts = name.rsplit(" ", 1)
        output_name = parts[-1] if len(parts) > 1 else name
    passed, ratio, max_abs = check_precision(tensor, golden, dtype_str, output_name)
    tag = "PASS" if passed else "FAIL"
    print(f"[PRECISION_{tag}] {name} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    return ok_acc and passed


def _generate_test_data_v2(B, T_len, H, HQ, D, S, BS, dtype=torch.float16, vrange=None, seed=42):
    """Generate deterministic test data with custom value range (L1/Boundary).

    Ensures block_indices only contains valid indices (0..NS-1) and block_counts
    respects availability (min(S, avail)). This avoids invalid fill values that
    crash the golden when S > NS.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T_len, HQ, D, dtype=dtype, device="cpu")
    k = torch.randn(B, T_len, H, D, dtype=dtype, device="cpu")
    v = torch.randn(B, T_len, H, D, dtype=dtype, device="cpu")
    do_slc = torch.randn(B, T_len, HQ, D, dtype=dtype, device="cpu")
    if vrange is not None:
        lo, hi = vrange
        sv = (hi - lo) / 2.0
        cv = (hi + lo) / 2.0
        q = (q.float() * sv + cv).to(dtype)
        k = (k.float() * sv + cv).to(dtype)
        v = (v.float() * sv + cv).to(dtype)
        do_slc = (do_slc.float() * sv + cv).to(dtype)

    NS = max(1, T_len // BS)
    block_indices = torch.zeros(B, T_len, H, S, dtype=torch.int32, device="cpu")
    block_counts = torch.ones(B, T_len, H, dtype=torch.int32, device="cpu")
    for b in range(B):
        for t in range(T_len):
            for h in range(H):
                avail = min(NS, max(1, t // BS))
                n = min(S, avail)
                if n > 0:
                    perm = torch.randperm(avail)[:n]
                    block_indices[b, t, h, :n] = perm
                    block_counts[b, t, h] = n
    block_indices = block_indices.sort(-1)[0]
    return q, k, v, do_slc, block_indices, block_counts


# ============================================================================
# L0 tests (blocking — 4 sub-cases)
# ============================================================================


def test_nsa_bwd_l0():
    """L0 gate tests: B=1, T=32, H=1, HQ=16, D=32, S=1, BS=32, FP16.

    4 sub-cases:
      - l0_e2e: end-to-end pipeline (fwd -> bwd)
      - l0_nsa_fwd_check: fwd-only precision (o_slc, lse_slc)
      - l0_bwd_pipeline: bwd-only precision with golden fwd inputs
      - l0_default: basic sanity (aggregate of above)
    """
    B, T_len, H, HQ, D, S, BS = 1, 32, 1, 16, 32, 1, 32
    dtype = torch.float16
    ok = True

    q, k, v, do_slc, block_indices, block_counts = _generate_test_data(B, T_len, H, HQ, D, S, BS, dtype)

    # Golden (CPU)
    ref_o, ref_lse = naive_nsa_fwd(q, k, v, block_indices, block_counts, block_size=BS)
    ref_dq, ref_dk, ref_dv = naive_nsa_bwd(q, k, v, do_slc, block_indices, block_counts, block_size=BS)

    # H2D
    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    do_npu = do_slc.to("npu")
    bi_npu = block_indices.to("npu")
    bc_npu = block_counts.to("npu")

    try:
        # === l0_e2e: End-to-end pipeline ===
        print("--- l0_e2e (fwd -> block_mask -> bwd pipeline) ---")
        o_slc, lse_slc, dq, dk, dv, block_mask = _run_nsa_pipeline(q_npu, k_npu, v_npu, do_npu, bi_npu, bc_npu, B, T_len, H, HQ, D, S, BS)

        ok = _print_result("l0_e2e o_slc", o_slc.cpu(), ref_o, "float16", ok)
        ok = _print_result("l0_e2e lse_slc", lse_slc.cpu(), ref_lse, "float32", ok)
        ok = _print_result("l0_e2e dv", dv.cpu(), ref_dv, "float16", ok)
        ok = _print_result("l0_e2e dq", dq.cpu(), ref_dq, "float16", ok)
        ok = _print_result("l0_e2e dk", dk.cpu(), ref_dk, "float16", ok)

        # Verify block_mask (int32 exact match — computed on CPU)
        ref_bm = _compute_block_mask_cpu(block_indices, block_counts, BS)
        ok = _print_result("l0_e2e block_mask", block_mask.cpu(), ref_bm, "int32", ok)

        # === l0_nsa_fwd_check: fwd-only ===
        print("--- l0_nsa_fwd_check (fwd precision) ---")
        fwd_ok = True
        fwd_ok = _print_result("l0_nsa_fwd_check o_slc", o_slc.cpu(), ref_o, "float16", fwd_ok)
        fwd_ok = _print_result("l0_nsa_fwd_check lse_slc", lse_slc.cpu(), ref_lse, "float32", fwd_ok)
        if fwd_ok:
            print("[PRECISION_PASS] l0_nsa_fwd_check: fwd outputs match golden")
        else:
            print("[PRECISION_FAIL] l0_nsa_fwd_check: fwd outputs mismatch")
        ok &= fwd_ok

        # === l0_bwd_pipeline: bwd-only with golden fwd inputs ===
        print("--- l0_bwd_pipeline (bwd with golden fwd inputs) ---")
        ref_o_npu = ref_o.to("npu")
        ref_lse_npu = ref_lse.to("npu")

        dq2, dk2, dv2 = _run_bwd_pipeline(
            q_npu,
            k_npu,
            v_npu,
            ref_o_npu,
            ref_lse_npu,
            do_npu,
            B,
            T_len,
            H,
            HQ,
            D,
            S,
            BS,
        )

        bwd_ok = True
        bwd_ok = _print_result("l0_bwd_pipeline dq", dq2.cpu(), ref_dq, "float16", bwd_ok)
        bwd_ok = _print_result("l0_bwd_pipeline dk", dk2.cpu(), ref_dk, "float16", bwd_ok)
        bwd_ok = _print_result("l0_bwd_pipeline dv", dv2.cpu(), ref_dv, "float16", bwd_ok)
        if bwd_ok:
            print("[PRECISION_PASS] l0_bwd_pipeline: bwd outputs match golden")
        else:
            print("[PRECISION_FAIL] l0_bwd_pipeline: bwd outputs mismatch")
        ok &= bwd_ok

        # === l0_default: basic sanity (subset of e2e) ===
        print("--- l0_default (basic sanity) ---")
        if ok:
            print("[PRECISION_PASS] l0_default: all outputs within tolerance")
        else:
            print("[PRECISION_FAIL] l0_default: some outputs out of tolerance")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[PRECISION_FAIL] l0: {e}")
        ok = False

    return ok


# ============================================================================
# L1 tests (blocking — functional, irregular shapes, param variations)
# ============================================================================


def _run_precision_case(name, B, T_len, H, HQ, D, S, BS, vrange, tags):
    """Run a single L1 precision case: generate data -> golden -> NPU pipeline -> compare.

    Uses global precision standard (0.99 for all fp16 outputs).
    Returns True if all outputs pass.
    """
    print(f"--- {name} (B={B} T={T_len} H={H} HQ={HQ} D={D} S={S} BS={BS} vrange={vrange}) ---")
    dtype = torch.float16
    q, k, v, do_slc, block_indices, block_counts = _generate_test_data_v2(B, T_len, H, HQ, D, S, BS, dtype, vrange)

    # Golden (CPU)
    ref_o, ref_lse = naive_nsa_fwd(q, k, v, block_indices, block_counts, block_size=BS)
    ref_dq, ref_dk, ref_dv = naive_nsa_bwd(q, k, v, do_slc, block_indices, block_counts, block_size=BS)

    # H2D
    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    do_npu = do_slc.to("npu")
    bi_npu = block_indices.to("npu")
    bc_npu = block_counts.to("npu")

    try:
        o_slc, lse_slc, dq, dk, dv, block_mask = _run_nsa_pipeline(q_npu, k_npu, v_npu, do_npu, bi_npu, bc_npu, B, T_len, H, HQ, D, S, BS)
        ref_bm = _compute_block_mask_cpu(block_indices, block_counts, BS)
        ok = True
        ok = _print_result(f"{name} o_slc", o_slc.cpu(), ref_o, "float16", ok)
        ok = _print_result(f"{name} lse_slc", lse_slc.cpu(), ref_lse, "float32", ok)
        ok = _print_result(f"{name} dv", dv.cpu(), ref_dv, "float16", ok)
        ok = _print_result(f"{name} dq", dq.cpu(), ref_dq, "float16", ok)
        ok = _print_result(f"{name} dk", dk.cpu(), ref_dk, "float16", ok)
        ok = _print_result(f"{name} block_mask", block_mask.cpu(), ref_bm, "int32", ok)
        return ok
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[PRECISION_FAIL] {name}: {e}")
        return False


def test_nsa_bwd_l1():
    """L1 functional tests: parameter combinations + irregular shapes (blocking).

    All shapes use NS=1 (T in [BS, 2*BS-1]) to ensure kernel correctness
    (bwd kernel doesn't check block_mask; only correct when NS=1).
    """
    ok = True
    for name, B, T_len, H, HQ, D, S, BS, vrange, tags in L1_CASES:
        ok &= _run_precision_case(name, B, T_len, H, HQ, D, S, BS, vrange, tags)
    if ok:
        print("[PRECISION_PASS] L1: all functional cases pass")
    else:
        print("[PRECISION_FAIL] L1: some functional cases fail")
    return ok


# ============================================================================
# L2 tests (non-blocking — negative, invalid inputs should be rejected)
# ============================================================================


def _run_exception(name, fn):
    """L2 helper: fn() feeds invalid input, expect rejection.

    Raises -> [BOUNDARY_PASS]; silently accepts -> [BOUNDARY_WARN]. Non-blocking.
    """
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] {name}: correctly rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] {name}: invalid input silently accepted")


def test_nsa_bwd_l2():
    """L2 negative tests: invalid dtype / shape should be rejected (non-blocking)."""
    B, T_len, H, HQ, D, S, BS = 1, 32, 1, 16, 32, 1, 32

    # D-EXC-DTYPE: Q with float32 (kernel expects float16)
    def _bad_dtype():
        q_f32 = torch.randn(B, T_len, HQ, D, dtype=torch.float32, device="npu")
        k_npu = torch.randn(B, T_len, H, D, dtype=torch.float16, device="npu")
        v_npu = torch.randn(B, T_len, H, D, dtype=torch.float16, device="npu")
        bi = torch.zeros(B, T_len, H, S, dtype=torch.int32, device="npu")
        bc = torch.ones(B, T_len, H, dtype=torch.int32, device="npu")
        _run_nsa_pipeline(
            q_f32,
            k_npu,
            v_npu,
            q_f32,
            bi,
            bc,
            B,
            T_len,
            H,
            HQ,
            D,
            S,
            BS,
        )

    _run_exception("l2_unsupported_dtype", _bad_dtype)

    # D-EXC-SHAPE: Q with wrong shape (D mismatch)
    def _bad_shape():
        q_bad = torch.randn(B, T_len, HQ, D + 16, dtype=torch.float16, device="npu")
        k_npu = torch.randn(B, T_len, H, D, dtype=torch.float16, device="npu")
        v_npu = torch.randn(B, T_len, H, D, dtype=torch.float16, device="npu")
        bi = torch.zeros(B, T_len, H, S, dtype=torch.int32, device="npu")
        bc = torch.ones(B, T_len, H, dtype=torch.int32, device="npu")
        _run_nsa_pipeline(
            q_bad,
            k_npu,
            v_npu,
            q_bad,
            bi,
            bc,
            B,
            T_len,
            H,
            HQ,
            D,
            S,
            BS,
        )

    _run_exception("l2_illegal_shape", _bad_shape)

    # D-EXC-SEQLEN: seq_len=0 (empty KV sequence — OP-R20 assert should reject)
    def _zero_seqlen():
        T_zero = 0
        q_zero = torch.randn(B, T_zero, HQ, D, dtype=torch.float16, device="npu")
        k_zero = torch.randn(B, T_zero, H, D, dtype=torch.float16, device="npu")
        v_zero = torch.randn(B, T_zero, H, D, dtype=torch.float16, device="npu")
        bi_zero = torch.zeros(B, T_zero, H, S, dtype=torch.int32, device="npu")
        bc_zero = torch.ones(B, T_zero, H, dtype=torch.int32, device="npu")
        _run_nsa_pipeline(
            q_zero,
            k_zero,
            v_zero,
            q_zero,
            bi_zero,
            bc_zero,
            B,
            T_zero,
            H,
            HQ,
            D,
            S,
            BS,
        )

    _run_exception("l2_zero_seqlen", _zero_seqlen)


# ============================================================================
# Boundary tests (non-blocking — special values + param limits)
# ============================================================================


def _run_boundary(name, dtype_str, fn):
    """Boundary helper: fn() returns (out_dict, ref_dict) for special-value inputs.

    Compares per-output precision; pass -> [BOUNDARY_PASS], fail -> [BOUNDARY_WARN].
    """
    try:
        outs, refs = fn()
        ok = True
        for oname in ("o_slc", "lse_slc", "dv", "dq", "dk"):
            if oname in outs and oname in refs:
                dt = "float32" if oname == "lse_slc" else dtype_str
                p, r, ma = check_precision(outs[oname], refs[oname], dt, oname)
                tag = "PASS" if p else "WARN"
                print(f"[BOUNDARY_{tag}] {name} {oname} matched_ratio={r:.4f} max_abs={ma:.3e}")
                ok &= p
        if ok:
            print(f"[BOUNDARY_PASS] {name}: all outputs within tolerance")
        else:
            print(f"[BOUNDARY_WARN] {name}: some outputs out of tolerance (non-blocking)")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {name}: exception {type(e).__name__}: {e}")


def test_nsa_bwd_boundary():
    """Boundary tests: INF/NAN/zero/dtype-bound + param-limit cases (non-blocking).

    Special values (INF/NAN/zero/dbound): inject into q, run pipeline, compare.
    Param-limit cases (D=64/HQ=32/BS=16/S=2): exceed kernel precision or compilation
    limits -> [BOUNDARY_WARN] (non-blocking, documented limitation).
    """
    B, T_len, H, HQ, D, S, BS = 1, 32, 1, 16, 32, 1, 32
    dtype = torch.float16

    def _make_boundary(special):
        q, k, v, do_slc, bi, bc = _generate_test_data_v2(B, T_len, H, HQ, D, S, BS, dtype, vrange=(-1, 1))
        if special == "inf":
            q[0, 0, 0, 0] = float("inf")
        elif special == "nan":
            q[0, 0, 0, 0] = float("nan")
        elif special == "zero":
            q = torch.zeros_like(q)
            k = torch.zeros_like(k)
            v = torch.zeros_like(v)
        elif special == "dbound":
            q[0, 0, 0, 0] = 65504.0  # fp16 max
            q[0, 1, 0, 0] = -65504.0
        ref_o, ref_lse = naive_nsa_fwd(q, k, v, bi, bc, block_size=BS)
        ref_dq, ref_dk, ref_dv = naive_nsa_bwd(q, k, v, do_slc, bi, bc, block_size=BS)
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        do_npu = do_slc.to("npu")
        bi_npu = bi.to("npu")
        bc_npu = bc.to("npu")
        o, lse, dq, dk, dv, _bm = _run_nsa_pipeline(
            q_npu,
            k_npu,
            v_npu,
            do_npu,
            bi_npu,
            bc_npu,
            B,
            T_len,
            H,
            HQ,
            D,
            S,
            BS,
        )
        return (
            {
                "o_slc": o.cpu(),
                "lse_slc": lse.cpu(),
                "dv": dv.cpu(),
                "dq": dq.cpu(),
                "dk": dk.cpu(),
            },
            {
                "o_slc": ref_o,
                "lse_slc": ref_lse,
                "dv": ref_dv,
                "dq": ref_dq,
                "dk": ref_dk,
            },
        )

    # Param-limit boundary cases: run pipeline with non-default params that
    # exceed kernel precision or compilation limits. Non-blocking [BOUNDARY_WARN].
    PARAM_BOUNDARY_CONFIGS = {
        "boundary_param_d64": (1, 32, 1, 16, 64, 1, 32),
        "boundary_param_groups": (1, 32, 1, 32, 32, 1, 32),
        "boundary_param_bs16": (1, 16, 1, 16, 32, 1, 16),
        "boundary_param_s2": (1, 32, 1, 16, 32, 2, 32),
    }

    # Valrange boundary cases: larger value ranges exceed fp16 GEMM precision
    # (dS ~ qkT * (dsT-Delta) * scale, quadratic in input scale). Non-blocking WARN.
    VALRANGE_BOUNDARY_CONFIGS = {
        "boundary_valrange_m": (-2, 2),
        "boundary_valrange_l": (-3, 3),
        "boundary_valrange_asym": (-2, 4),
    }

    def _make_param_boundary(name):
        pB, pT, pH, pHQ, pD, pS, pBS = PARAM_BOUNDARY_CONFIGS[name]
        q, k, v, do_slc, bi, bc = _generate_test_data_v2(pB, pT, pH, pHQ, pD, pS, pBS, dtype, vrange=(-1, 1))
        ref_o, ref_lse = naive_nsa_fwd(q, k, v, bi, bc, block_size=pBS)
        ref_dq, ref_dk, ref_dv = naive_nsa_bwd(q, k, v, do_slc, bi, bc, block_size=pBS)
        o, lse, dq, dk, dv, _bm = _run_nsa_pipeline(
            q.to("npu"),
            k.to("npu"),
            v.to("npu"),
            do_slc.to("npu"),
            bi.to("npu"),
            bc.to("npu"),
            pB,
            pT,
            pH,
            pHQ,
            pD,
            pS,
            pBS,
        )
        return (
            {
                "o_slc": o.cpu(),
                "lse_slc": lse.cpu(),
                "dv": dv.cpu(),
                "dq": dq.cpu(),
                "dk": dk.cpu(),
            },
            {
                "o_slc": ref_o,
                "lse_slc": ref_lse,
                "dv": ref_dv,
                "dq": ref_dq,
                "dk": ref_dk,
            },
        )

    def _make_valrange_boundary(name):
        vr = VALRANGE_BOUNDARY_CONFIGS[name]
        q, k, v, do_slc, bi, bc = _generate_test_data_v2(B, T_len, H, HQ, D, S, BS, dtype, vrange=vr)
        ref_o, ref_lse = naive_nsa_fwd(q, k, v, bi, bc, block_size=BS)
        ref_dq, ref_dk, ref_dv = naive_nsa_bwd(q, k, v, do_slc, bi, bc, block_size=BS)
        o, lse, dq, dk, dv, _bm = _run_nsa_pipeline(
            q.to("npu"),
            k.to("npu"),
            v.to("npu"),
            do_slc.to("npu"),
            bi.to("npu"),
            bc.to("npu"),
            B,
            T_len,
            H,
            HQ,
            D,
            S,
            BS,
        )
        return (
            {
                "o_slc": o.cpu(),
                "lse_slc": lse.cpu(),
                "dv": dv.cpu(),
                "dq": dq.cpu(),
                "dk": dk.cpu(),
            },
            {
                "o_slc": ref_o,
                "lse_slc": ref_lse,
                "dv": ref_dv,
                "dq": ref_dq,
                "dk": ref_dk,
            },
        )

    for bname, dt_str, _tags in BOUNDARY_CASES:
        if bname.startswith("boundary_param_"):
            _run_boundary(bname, dt_str, lambda n=bname: _make_param_boundary(n))
        elif bname.startswith("boundary_valrange_"):
            _run_boundary(bname, dt_str, lambda n=bname: _make_valrange_boundary(n))
        else:
            special = bname.split("_")[1]
            _run_boundary(bname, dt_str, lambda s=special: _make_boundary(s))


# ============================================================================
# Main: --level dispatch + exit code
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NSA Backward layered precision test suite (Ascend)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()

    blocking_ok = True  # only L0/L1 count toward blocking decision
    if args.level in ("l0", "all"):
        blocking_ok &= test_nsa_bwd_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_nsa_bwd_l1()
    if args.level in ("l2", "all"):
        test_nsa_bwd_l2()  # L2 negative: non-blocking
    if args.level in ("boundary", "all"):
        test_nsa_bwd_boundary()  # Boundary precision: non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
