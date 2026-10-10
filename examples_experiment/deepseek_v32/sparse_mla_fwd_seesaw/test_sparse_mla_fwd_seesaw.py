"""Layered test suite for sparse_mla_fwd_seesaw.

L0: threshold tests (small configs for fast convergence)
L1: functional coverage (different shapes, topk, causal patterns)
L2: negative tests (abnormal inputs should be rejected)
Boundary: extreme values, minimum shape, all-masked query

Usage:
    python test_sparse_mla_fwd_seesaw.py --level l0       # L0 only (fast convergence)
    python test_sparse_mla_fwd_seesaw.py --level l1       # L1 functional
    python test_sparse_mla_fwd_seesaw.py --level l2       # L2 negative
    python test_sparse_mla_fwd_seesaw.py --level boundary # Boundary
    python test_sparse_mla_fwd_seesaw.py --level all      # All levels
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_sparse_mla_fwd_seesaw import sparse_mla_fwd_interface  # noqa: E402

import tilelang  # noqa: E402

# ---------------------------------------------------------------------------
# Golden reference (CPU, fp32) -- lives in the test file.
# ---------------------------------------------------------------------------


def ref_sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=1, sm_scale=None):
    """CPU golden reference (fp32, vectorized per query position).

    Uses natural exp/log (not exp2/log2) to match the NPU kernel.
    sm_scale = 1/sqrt(576) (no log2(e) factor).
    """
    q = q.float()
    kv = kv.float()
    b, sq, h, dqk = q.shape
    _, sk, hk, _ = kv.shape
    _, _, _, topk = indices.shape
    dim = 512

    if sm_scale is None:
        sm_scale = (1.0 / dqk) ** 0.5

    out = torch.zeros(b, sq, h, dim, dtype=torch.float32)
    lse = torch.zeros(b, sq, h, dtype=torch.float32)

    for bi in range(b):
        for si in range(sq):
            q_i = q_start_index_s + si
            max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            idx = indices[bi, si, 0, :].long()
            valid = (idx <= max_kv_i) & (idx < sk)
            idx_clamped = idx.clamp(0, sk - 1)
            kv_gathered = kv[bi, idx_clamped, 0, :]  # (topk, dqk)

            scores = q[bi, si, :, :] @ kv_gathered.T  # (h, topk)
            scores = scores * sm_scale
            scores[:, ~valid] = float("-inf")

            max_scores = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - max_scores)
            sumexp = exp_scores.sum(dim=-1, keepdim=True)
            p = exp_scores / sumexp

            v_gathered = kv_gathered[:, :dim]
            out[bi, si, :, :] = p @ v_gathered
            lse[bi, si, :] = torch.log(sumexp.squeeze(-1)) + max_scores.squeeze(-1)

    return out.to(torch.bfloat16), lse


# ---------------------------------------------------------------------------
# Coverage manifest (for coverage_check.py)
# ---------------------------------------------------------------------------

COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    "L1_CASES": [
        {
            "name": "l1_batch2",
            "tags": [
                "D-EXC-SHAPE",
                "D-SHAPE-ALIGNED",
                "D-VALRANGE-S",
                "D-DTYPE-bf16",
                "D-DTYPE-int32",
                "D-DTYPE-fp32",
                "D-PARAM-dim",
                "D-PARAM-tail_dim",
                "D-EXC-DTYPE",
            ],
        },
        {"name": "l1_seq8_skv128", "tags": ["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-q_start_index_s"]},
        {"name": "l1_topk128", "tags": ["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-M", "D-PARAM-topk"]},
        {"name": "l1_topk256", "tags": ["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-PARAM-topk"]},
        {"name": "l1_q_start_mid", "tags": ["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-q_start_index_s"]},
        {"name": "l1_prime_seq", "tags": ["D-SHAPE-PRIME"]},
        {"name": "l1_seq1", "tags": ["D-SHAPE-TAIL-1", "D-SHAPE-EDGE"]},
        {"name": "l1_large_values", "tags": ["D-VALRANGE-ASYM"]},
        {"name": "l1_tail_mid_seq", "tags": ["D-SHAPE-TAIL-MID"]},
        {"name": "l1_custom_sm_scale", "tags": ["D-PARAM-sm_scale"]},
        {"name": "l1_kv_stride2", "tags": ["D-PARAM-kv_stride"]},
        {"name": "l1_heads64", "tags": ["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]},
    ],
    "BOUNDARY_CASES": [
        {"name": "bnd_min_shape", "tags": ["D-SHAPE-EDGE"]},
        {"name": "bnd_zero_input", "tags": ["D-SPECIAL-ZERO"]},
        {"name": "bnd_inf_input", "tags": ["D-SPECIAL-INF"]},
        {"name": "bnd_nan_input", "tags": ["D-SPECIAL-NAN"]},
        {"name": "bnd_bf16_max", "tags": ["D-SPECIAL-DBOUND"]},
        {"name": "bnd_all_masked", "tags": ["D-SHAPE-ALIGNED"]},
    ],
}

# Dimensions that are N/A for this operator (with reasons)
COVERAGE_NA = {
    "D-DTYPE-float16": "NPU kernel is bf16-only. fp16 not supported.",
    "D-DTYPE-hifloat32": "NPU kernel does not support hifloat32.",
    "D-DTYPE-float8_e4m3": "NPU kernel does not support float8.",
    "D-DTYPE-float8_e5m2": "NPU kernel does not support float8.",
}

# ---------------------------------------------------------------------------
# Precision check (mixed tolerance dual-threshold)
# ---------------------------------------------------------------------------


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Check precision: returns (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
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
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ---------------------------------------------------------------------------
# Helper: generate random indices
# ---------------------------------------------------------------------------


def gen_indices(batch, seq_len, kv_group, topk, seq_len_kv, q_start_index_s, kv_stride=1, seed=42):
    """Generate sparse indices. Each (b, s) gets topk positions, with causal-valid
    positions first and padding filling the tail.

    Entry contract: every index must be a legal KV position in [0, seq_len_kv)
    (the kernel interface rejects out-of-range values; clamping them would
    silently mis-count the last KV row into the softmax when q_start is late).
    Padding must therefore be an in-range value strictly
    greater than max_kv_i so that the causal mask (idx <= max_kv_i) hides it
    in both kernel and golden. We use min(max_kv_i + 1, seq_len_kv - 1): for
    max_kv_i >= 0 this is the smallest always-masked in-range value; when
    max_kv_i < 0 (query before the first KV position; the only possible
    negative value is -1) the formula naturally yields 0, which is still
    masked because 0 > max_kv_i.
    """
    assert topk <= seq_len_kv, (
        f"topk ({topk}) must be <= seq_len_kv ({seq_len_kv}): with topk > "
        "seq_len_kv, some query would need more padding slots than there are "
        "in-range always-masked values"
    )
    torch.manual_seed(seed)
    indices = torch.zeros(batch, seq_len, kv_group, topk, dtype=torch.int32)
    for bi in range(batch):
        for si in range(seq_len):
            q_i = q_start_index_s + si
            max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            # Clamp max_kv_i to [0, seq_len_kv - 1] for valid position range.
            valid_upper = min(max_kv_i, seq_len_kv - 1)
            # In-range padding: strictly greater than max_kv_i, so the causal
            # mask hides it in both kernel and golden (naturally 0 when
            # max_kv_i < 0, which is still masked).
            pad_value = min(max_kv_i + 1, seq_len_kv - 1)
            if valid_upper < 0:
                # No valid positions (query before first KV); fill all with padding.
                indices[bi, si, 0, :topk] = pad_value
            else:
                n_valid = min(topk, valid_upper + 1)
                perm = torch.randperm(valid_upper + 1)[:n_valid]
                indices[bi, si, 0, :n_valid] = perm
                # Fill remaining with padding (will be masked by causal mask).
                if n_valid < topk:
                    indices[bi, si, 0, n_valid:] = pad_value
    return indices


def gen_indices_all_valid(batch, seq_len, kv_group, topk, seq_len_kv, seed=42):
    """Generate indices where all positions are valid (no causal mask)."""
    torch.manual_seed(seed)
    indices = torch.zeros(batch, seq_len, kv_group, topk, dtype=torch.int32)
    for bi in range(batch):
        for si in range(seq_len):
            perm = torch.randperm(seq_len_kv)[:topk]
            indices[bi, si, 0, :topk] = perm
    return indices


def run_and_check(config_name, B, S, SKV, H, HKV, DQK, DV, TOPK, q_start_index_s, kv_stride=1, scale=0.1, seed=42, tags=None, level="L1"):
    """Run kernel and compare against golden. Returns (out_pass, lse_pass, stats)."""
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, DQK, dtype=torch.bfloat16) * scale
    kv = torch.randn(B, SKV, HKV, DQK, dtype=torch.bfloat16) * scale
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    if q_start_index_s >= SKV - 1:
        indices = gen_indices_all_valid(B, S, HKV, TOPK, SKV, seed=seed)
    else:
        indices = gen_indices(B, S, HKV, TOPK, SKV, q_start_index_s, kv_stride, seed=seed)

    q_npu = q.npu()
    kv_npu = kv.npu()
    indices_npu = indices.npu()

    tl_out, tl_lse = sparse_mla_fwd_interface(q_npu, kv_npu, indices_npu, q_start_index_s, kv_stride)
    tl_out_cpu = tl_out.cpu()
    tl_lse_cpu = tl_lse.cpu()

    ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride)

    out_passed, out_ratio, out_max_abs = check_precision(tl_out_cpu, ref_out, "bfloat16")
    lse_passed, lse_ratio, lse_max_abs = check_precision(tl_lse_cpu, ref_lse, "float32")

    tag_str = f" [{', '.join(tags)}]" if tags else ""
    print(
        f"  [{level}] {config_name}: Output ratio={out_ratio:.6f} max={out_max_abs:.6e} "
        f"{'PASS' if out_passed else 'FAIL'}, "
        f"Lse ratio={lse_ratio:.6f} max={lse_max_abs:.6e} "
        f"{'PASS' if lse_passed else 'FAIL'}{tag_str}"
    )

    if out_passed and lse_passed:
        print(f"    [PRECISION_PASS] {config_name}")
    else:
        print(f"    [PRECISION_FAIL] {config_name}")

    return out_passed and lse_passed, {
        "out_ratio": out_ratio,
        "out_max_abs": out_max_abs,
        "lse_ratio": lse_ratio,
        "lse_max_abs": lse_max_abs,
    }


# ---------------------------------------------------------------------------
# L0: Threshold tests
# ---------------------------------------------------------------------------


def run_l0_suite():
    """L0 threshold tests: fast convergence on small configs."""
    print("=" * 60)
    print("L0: Threshold Tests")
    print("=" * 60)
    tilelang.disable_cache()

    results = []

    # L0-1: smoke_all_valid -- all valid indices, no causal mask
    print("\n[L0-1] l0_smoke_all_valid: B=1,S=4,SKV=64,H=128,topk=64,q_start=63")
    r, _ = run_and_check(
        "l0_smoke_all_valid",
        1,
        4,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-SHAPE-ALIGNED", "D-VALRANGE-S"],
        level="L0",
    )
    results.append(("l0_smoke_all_valid", r))

    # L0-2: small_causal -- causal mask active but non-degenerate (q_start=32,
    # so every query sees >= 33 valid KV positions -- avoids the n_valid=1
    # degenerate case which hits fp32 atol floor on tiny Lse values).
    # Degenerate (n_valid<5) cases are covered in Boundary tests.
    print("\n[L0-2] l0_small_causal: B=1,S=4,SKV=64,H=128,topk=64,q_start=32")
    r, _ = run_and_check(
        "l0_small_causal",
        1,
        4,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=32,
        tags=["D-SHAPE-ALIGNED", "D-VALRANGE-S"],
        level="L0",
    )
    results.append(("l0_small_causal", r))

    # L0-3: golden_config -- the production target shape
    # B=2,S=4096,SKV=8192,H=128,topk=2048,bf16,q_start=4096
    # This is the MANDATORY L0 case matching the user's golden config.
    print("\n[L0-3] l0_golden_config: B=2,S=4096,SKV=8192,H=128,topk=2048,q_start=4096")
    r, _ = run_and_check(
        "l0_golden_config",
        2,
        4096,
        8192,
        128,
        1,
        576,
        512,
        2048,
        q_start_index_s=4096,
        tags=[
            "D-SHAPE-ALIGNED",
            "D-VALRANGE-L",
            "D-DTYPE-bf16",
            "D-DTYPE-int32",
            "D-DTYPE-fp32",
            "D-PARAM-dim",
            "D-PARAM-tail_dim",
            "D-PARAM-topk",
            "D-PARAM-q_start_index_s",
            "D-PARAM-kv_stride",
            "D-PARAM-sm_scale",
        ],
        level="L0",
    )
    results.append(("l0_golden_config", r))

    # L0-4: medium config -- different topk, partial causal
    print("\n[L0-4] l0_medium_topk128: B=1,S=2,SKV=128,H=128,topk=128,q_start=64")
    r, _ = run_and_check(
        "l0_medium_topk128",
        1,
        2,
        128,
        128,
        1,
        576,
        512,
        128,
        q_start_index_s=64,
        tags=["D-SHAPE-ALIGNED", "D-VALRANGE-M"],
        level="L0",
    )
    results.append(("l0_medium_topk128", r))

    all_pass = all(r for _, r in results)
    if all_pass:
        print(f"\n[PRECISION_PASS] All L0 tests passed ({len(results)} cases)")
    else:
        failed = [n for n, r in results if not r]
        print(f"\n[PRECISION_FAIL] L0 failures: {failed}")
    return all_pass


# ---------------------------------------------------------------------------
# L1: Functional coverage
# ---------------------------------------------------------------------------


def run_l1_suite():
    """L1 functional tests: different shapes, topk, causal patterns."""
    print("=" * 60)
    print("L1: Functional Tests")
    print("=" * 60)

    results = []

    # L1-1: batch=2
    print("\n[L1-1] l1_batch2: B=2,S=4,SKV=64,H=128,topk=64,q_start=63")
    r, _ = run_and_check(
        "l1_batch2",
        2,
        4,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    )
    results.append(("l1_batch2", r))

    # L1-2: larger seq
    print("\n[L1-2] l1_seq8_skv128: B=1,S=8,SKV=128,H=128,topk=64,q_start=60")
    r, _ = run_and_check(
        "l1_seq8_skv128",
        1,
        8,
        128,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=60,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    )
    results.append(("l1_seq8_skv128", r))

    # L1-3: topk=128
    print("\n[L1-3] l1_topk128: B=1,S=4,SKV=256,H=128,topk=128,q_start=128")
    r, _ = run_and_check(
        "l1_topk128",
        1,
        4,
        256,
        128,
        1,
        576,
        512,
        128,
        q_start_index_s=128,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    )
    results.append(("l1_topk128", r))

    # L1-4: topk=256 (large)
    print("\n[L1-4] l1_topk256: B=1,S=4,SKV=512,H=128,topk=256,q_start=256")
    r, _ = run_and_check(
        "l1_topk256",
        1,
        4,
        512,
        128,
        1,
        576,
        512,
        256,
        q_start_index_s=256,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    )
    results.append(("l1_topk256", r))

    # L1-5: partial causal
    print("\n[L1-5] l1_q_start_mid: B=1,S=4,SKV=64,H=128,topk=64,q_start=32")
    r, _ = run_and_check(
        "l1_q_start_mid",
        1,
        4,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=32,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    )
    results.append(("l1_q_start_mid", r))

    # L1-6: prime seq_len
    print("\n[L1-6] l1_prime_seq: B=1,S=7,SKV=64,H=128,topk=64,q_start=63")
    r, _ = run_and_check(
        "l1_prime_seq",
        1,
        7,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-SHAPE-PRIME"],
    )
    results.append(("l1_prime_seq", r))

    # L1-7: seq_len=1 (tail-1)
    print("\n[L1-7] l1_seq1: B=1,S=1,SKV=64,H=128,topk=64,q_start=63")
    r, _ = run_and_check(
        "l1_seq1",
        1,
        1,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-SHAPE-TAIL-1", "D-SHAPE-EDGE"],
    )
    results.append(("l1_seq1", r))

    # L1-8: large values (asymmetric range)
    print("\n[L1-8] l1_large_values: B=1,S=4,SKV=64,H=128,topk=64,q_start=63,scale=1.0")
    r, _ = run_and_check(
        "l1_large_values",
        1,
        4,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        scale=1.0,
        tags=["D-VALRANGE-ASYM"],
    )
    results.append(("l1_large_values", r))

    # L1-9: tail-mid seq (seq_len not aligned to block, creates mid-range tail)
    print("\n[L1-9] l1_tail_mid_seq: B=1,S=10,SKV=100,H=128,topk=64,q_start=99")
    r, _ = run_and_check(
        "l1_tail_mid_seq",
        1,
        10,
        100,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=99,
        tags=["D-SHAPE-TAIL-MID"],
    )
    results.append(("l1_tail_mid_seq", r))

    # L1-10: custom sm_scale
    print("\n[L1-10] l1_custom_sm_scale: B=1,S=4,SKV=64,H=128,topk=64,q_start=63,sm_scale=0.05")
    try:
        torch.manual_seed(42)
        B, S, SKV, H, DQK, TOPK = 1, 4, 64, 128, 576, 64
        q = torch.randn(B, S, H, DQK, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, DQK, dtype=torch.bfloat16) / 10
        q.clamp_(-10, 10)
        kv.clamp_(-10, 10)
        indices = gen_indices_all_valid(B, S, 1, TOPK, SKV)
        custom_scale = 0.05
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1, sm_scale=custom_scale)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 63, 1, sm_scale=custom_scale)
        out_p, out_r, out_m = check_precision(tl_out.cpu(), ref_out, "bfloat16")
        lse_p, lse_r, lse_m = check_precision(tl_lse.cpu(), ref_lse, "float32")
        r = out_p and lse_p
        print(
            f"  [L1] l1_custom_sm_scale: Output ratio={out_r:.6f} {'PASS' if out_p else 'FAIL'}, "
            f"Lse ratio={lse_r:.6f} {'PASS' if lse_p else 'FAIL'} [D-PARAM-sm_scale]"
        )
        if r:
            print("    [PRECISION_PASS] l1_custom_sm_scale")
        else:
            print("    [PRECISION_FAIL] l1_custom_sm_scale")
    except Exception as e:
        r = False
        print(f"  [L1] l1_custom_sm_scale: exception {e} [D-PARAM-sm_scale]")
    results.append(("l1_custom_sm_scale", r))

    # L1-11: kv_stride=2
    print("\n[L1-11] l1_kv_stride2: B=1,S=4,SKV=128,H=128,topk=64,q_start=127,kv_stride=2")
    try:
        torch.manual_seed(42)
        B, S, SKV, H, DQK, TOPK = 1, 4, 128, 128, 576, 64
        q = torch.randn(B, S, H, DQK, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, DQK, dtype=torch.bfloat16) / 10
        q.clamp_(-10, 10)
        kv.clamp_(-10, 10)
        indices = gen_indices(B, S, 1, TOPK, SKV, 127, kv_stride=2, seed=42)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 127, kv_stride=2)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 127, kv_stride=2)
        out_p, out_r, out_m = check_precision(tl_out.cpu(), ref_out, "bfloat16")
        lse_p, lse_r, lse_m = check_precision(tl_lse.cpu(), ref_lse, "float32")
        r = out_p and lse_p
        print(
            f"  [L1] l1_kv_stride2: Output ratio={out_r:.6f} {'PASS' if out_p else 'FAIL'}, "
            f"Lse ratio={lse_r:.6f} {'PASS' if lse_p else 'FAIL'} [D-PARAM-kv_stride]"
        )
        if r:
            print("    [PRECISION_PASS] l1_kv_stride2")
        else:
            print("    [PRECISION_FAIL] l1_kv_stride2")
    except Exception as e:
        r = False
        print(f"  [L1] l1_kv_stride2: exception {e} [D-PARAM-kv_stride]")
    results.append(("l1_kv_stride2", r))

    # L1-12: heads=64 (non-golden head count; exercises the hid_tiles=2 /
    # dim_split=1 tiling path and guards the heads % 64 == 0 support boundary)
    print("\n[L1-12] l1_heads64: B=1,S=4,SKV=64,H=64,topk=64,q_start=63")
    r, _ = run_and_check(
        "l1_heads64",
        1,
        4,
        64,
        64,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-EXC-SHAPE", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    )
    results.append(("l1_heads64", r))

    all_pass = all(r for _, r in results)
    if all_pass:
        print(f"\n[PRECISION_PASS] All L1 tests passed ({len(results)} cases)")
    else:
        failed = [n for n, r in results if not r]
        print(f"\n[PRECISION_FAIL] L1 failures: {failed}")
    return all_pass


# ---------------------------------------------------------------------------
# L2: Negative tests (abnormal inputs should be rejected)
# ---------------------------------------------------------------------------


def run_l2_suite():
    """L2 negative tests: verify abnormal inputs are properly rejected."""
    print("=" * 60)
    print("L2: Negative Tests")
    print("=" * 60)

    warnings = []

    # L2-1: topk not divisible by block_I (should assert fail)
    print("\n[L2-1] l2_topk_not_divisible: topk=65 (not divisible by 64)")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 128, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 65), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_topk_not_divisible: expected assert but succeeded")
        warnings.append("l2_topk_not_divisible: no assert raised")
    except AssertionError as e:
        expected = "must be divisible by block_I"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_topk_not_divisible: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_topk_not_divisible: wrong rejection reason: {e}")
            warnings.append(f"l2_topk_not_divisible: wrong rejection reason: {e}")

    # L2-2: dqk != 576 (should assert fail)
    print("\n[L2-2] l2_wrong_dqk: dqk=512 (not 576)")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.randn(B, S, H, 512, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 512, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_wrong_dqk: expected assert but succeeded")
        warnings.append("l2_wrong_dqk: no assert raised")
    except AssertionError as e:
        expected = "dqk must be 576"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_wrong_dqk: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_wrong_dqk: wrong rejection reason: {e}")
            warnings.append(f"l2_wrong_dqk: wrong rejection reason: {e}")

    # L2-3: hk != 1 (should assert fail)
    print("\n[L2-3] l2_hk_not_1: hk=2")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 2, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_hk_not_1: expected assert but succeeded")
        warnings.append("l2_hk_not_1: no assert raised")
    except AssertionError as e:
        expected = "hk must be 1"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_hk_not_1: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_hk_not_1: wrong rejection reason: {e}")
            warnings.append(f"l2_hk_not_1: wrong rejection reason: {e}")

    # L2-4: heads=96 (multiple of 16 and 32 but not 64: the head-block grid
    # would truncate the trailing block)
    print("\n[L2-4] l2_heads_not_supported_96: heads=96")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 128, 96
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_heads_not_supported_96: expected assert but succeeded")
        warnings.append("l2_heads_not_supported_96: no assert raised")
    except AssertionError as e:
        expected = "heads must be a multiple of 64"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_heads_not_supported_96: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_heads_not_supported_96: wrong rejection reason: {e}")
            warnings.append(f"l2_heads_not_supported_96: wrong rejection reason: {e}")

    # L2-5: heads=32 (head-block grid yields num_h_block_pairs=0, a
    # divide-by-zero in the tile map)
    print("\n[L2-5] l2_heads_not_supported_32: heads=32")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 128, 32
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_heads_not_supported_32: expected assert but succeeded")
        warnings.append("l2_heads_not_supported_32: no assert raised")
    except AssertionError as e:
        expected = "heads must be a multiple of 64"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_heads_not_supported_32: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_heads_not_supported_32: wrong rejection reason: {e}")
            warnings.append(f"l2_heads_not_supported_32: wrong rejection reason: {e}")

    # L2-6: heads=16 (not divisible by block_H=32)
    print("\n[L2-6] l2_heads_not_supported_16: heads=16")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 128, 16
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_heads_not_supported_16: expected assert but succeeded")
        warnings.append("l2_heads_not_supported_16: no assert raised")
    except AssertionError as e:
        expected = "heads must be a multiple of 64"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_heads_not_supported_16: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_heads_not_supported_16: wrong rejection reason: {e}")
            warnings.append(f"l2_heads_not_supported_16: wrong rejection reason: {e}")

    # L2-7: kv_stride=0 (would divide by zero in max_kv_i derivation)
    print("\n[L2-7] l2_kv_stride_zero: kv_stride=0")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 4, 128, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 0)
        print("  [BOUNDARY_WARN] l2_kv_stride_zero: expected assert but succeeded")
        warnings.append("l2_kv_stride_zero: no assert raised")
    except AssertionError as e:
        expected = "kv_stride must be a positive integer"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_kv_stride_zero: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_kv_stride_zero: wrong rejection reason: {e}")
            warnings.append(f"l2_kv_stride_zero: wrong rejection reason: {e}")

    # L2-8: indices out of range (>= seq_len_kv); when q_start is late a
    # clamped value would be wrongly counted into the softmax
    print("\n[L2-8] l2_indices_oob: indices contain seq_len_kv (out of range)")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 1, 64, 64
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        indices[0, 0, 0, 0] = SKV
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_indices_oob: expected assert but succeeded")
        warnings.append("l2_indices_oob: no assert raised")
    except AssertionError as e:
        expected = "indices out of range"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_indices_oob: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_indices_oob: wrong rejection reason: {e}")
            warnings.append(f"l2_indices_oob: wrong rejection reason: {e}")

    # L2-9: negative indices (clamping would map them to row 0 and count it
    # into the softmax)
    print("\n[L2-9] l2_indices_negative: indices contain -1")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 1, 64, 64
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.randint(0, SKV, (B, S, 1, 64), dtype=torch.int32)
        indices[0, 0, 0, 0] = -1
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_indices_negative: expected assert but succeeded")
        warnings.append("l2_indices_negative: no assert raised")
    except AssertionError as e:
        expected = "indices contain negative values"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_indices_negative: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_indices_negative: wrong rejection reason: {e}")
            warnings.append(f"l2_indices_negative: wrong rejection reason: {e}")

    # L2-10: seq_len=0 (the tile map divides by seq_len)
    print("\n[L2-10] l2_seq_len_zero: S=0")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 0, 64, 64
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.zeros(B, S, 1, 64, dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_seq_len_zero: expected assert but succeeded")
        warnings.append("l2_seq_len_zero: no assert raised")
    except AssertionError as e:
        expected = "seq_len must be >= 1"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_seq_len_zero: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_seq_len_zero: wrong rejection reason: {e}")
            warnings.append(f"l2_seq_len_zero: wrong rejection reason: {e}")

    # L2-11: topk=0 ("0 % 128 == 0" derives BLOCK_I=128 and bypasses every
    # tiling assert)
    print("\n[L2-11] l2_topk_zero: topk=0")
    try:
        torch.manual_seed(42)
        B, S, SKV, H = 1, 1, 64, 64
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = torch.zeros(B, S, 1, 0, dtype=torch.int32)
        sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        print("  [BOUNDARY_WARN] l2_topk_zero: expected assert but succeeded")
        warnings.append("l2_topk_zero: no assert raised")
    except AssertionError as e:
        expected = "topk must be >= 1"
        if expected in str(e):
            print(f"  [BOUNDARY_PASS] l2_topk_zero: correctly rejected ({type(e).__name__}, reason matched: '{expected}')")
        else:
            print(f"  [BOUNDARY_WARN] l2_topk_zero: wrong rejection reason: {e}")
            warnings.append(f"l2_topk_zero: wrong rejection reason: {e}")

    if warnings:
        print(f"\n[BOUNDARY_WARN] L2 warnings: {len(warnings)}")
    else:
        print("\n[BOUNDARY_PASS] All L2 negative tests passed")
    return len(warnings) == 0


# ---------------------------------------------------------------------------
# Boundary tests (extreme values, minimum shape)
# ---------------------------------------------------------------------------


def run_boundary_suite():
    """Boundary tests: extreme values, minimum shape, all-masked query."""
    print("=" * 60)
    print("Boundary: Extreme Value Tests")
    print("=" * 60)

    warnings = []
    tilelang.disable_cache()

    # BND-1: minimum shape
    print("\n[BND-1] bnd_min_shape: B=1,S=1,SKV=64,H=128,topk=64,q_start=63")
    r, _ = run_and_check(
        "bnd_min_shape",
        1,
        1,
        64,
        128,
        1,
        576,
        512,
        64,
        q_start_index_s=63,
        tags=["D-SHAPE-EDGE"],
        level="BND",
    )
    if not r:
        warnings.append("bnd_min_shape: precision fail")

    # BND-2: zero input
    print("\n[BND-2] bnd_zero_input: all-zero Q and KV")
    try:
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.zeros(B, S, H, 576, dtype=torch.bfloat16)
        kv = torch.zeros(B, SKV, 1, 576, dtype=torch.bfloat16)
        indices = gen_indices_all_valid(B, S, 1, 64, SKV)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 63, 1)
        out_p, out_r, out_m = check_precision(tl_out.cpu(), ref_out, "bfloat16")
        lse_p, lse_r, lse_m = check_precision(tl_lse.cpu(), ref_lse, "float32")
        if out_p and lse_p:
            print(f"  [BOUNDARY_PASS] bnd_zero_input: ratio={out_r:.6f}")
        else:
            print(f"  [BOUNDARY_WARN] bnd_zero_input: out={out_p} lse={lse_p}")
            warnings.append("bnd_zero_input: precision fail")
    except Exception as e:
        print(f"  [BOUNDARY_WARN] bnd_zero_input: exception {e}")
        warnings.append(f"bnd_zero_input: {e}")

    # BND-3: inf in input -- inf/nan positions do structural comparison
    # do structural comparison (positions match), not numeric tolerance.
    print("\n[BND-3] bnd_inf_input: Q contains inf")
    try:
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        q[0, 0, 0, 0] = float("inf")
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = gen_indices_all_valid(B, S, 1, 64, SKV)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 63, 1)
        # Structural check: inf/nan positions must match between actual and golden.
        tl_inf = torch.isinf(tl_out.cpu().float())
        ref_inf = torch.isinf(ref_out.cpu().float())
        tl_nan = torch.isnan(tl_out.cpu().float())
        ref_nan = torch.isnan(ref_out.cpu().float())
        struct_match = torch.equal(tl_inf, ref_inf) and torch.equal(tl_nan, ref_nan)
        if struct_match:
            # On finite positions, apply numeric tolerance per standard.
            finite_mask = torch.isfinite(ref_out.cpu().float())
            if finite_mask.any():
                diff = (tl_out.cpu().float()[finite_mask] - ref_out.cpu().float()[finite_mask]).abs()
                tol = 2**-10 + 2**-6 * ref_out.cpu().float()[finite_mask].abs()
                finite_ratio = (diff <= tol).float().mean().item()
                finite_max = diff.max().item()
                if finite_ratio >= 0.99 and finite_max <= 1e0:
                    print(f"  [BOUNDARY_PASS] bnd_inf_input: struct match, finite ratio={finite_ratio:.6f}")
                else:
                    print(f"  [BOUNDARY_WARN] bnd_inf_input: struct match but finite ratio={finite_ratio:.6f} max={finite_max:.6e}")
                    warnings.append("bnd_inf_input: finite precision fail")
            else:
                print("  [BOUNDARY_PASS] bnd_inf_input: all-inf/nan, struct match")
        else:
            print("  [BOUNDARY_WARN] bnd_inf_input: inf/nan struct mismatch (expected inf propagation)")
            warnings.append("bnd_inf_input: inf/nan struct mismatch")
    except Exception as e:
        print(f"  [BOUNDARY_WARN] bnd_inf_input: exception {e}")
        warnings.append(f"bnd_inf_input: {e}")

    # BND-4: nan in input -- same structural comparison as BND-3.
    print("\n[BND-4] bnd_nan_input: Q contains nan")
    try:
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        q[0, 0, 0, 0] = float("nan")
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        indices = gen_indices_all_valid(B, S, 1, 64, SKV)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 63, 1)
        tl_inf = torch.isinf(tl_out.cpu().float())
        ref_inf = torch.isinf(ref_out.cpu().float())
        tl_nan = torch.isnan(tl_out.cpu().float())
        ref_nan = torch.isnan(ref_out.cpu().float())
        struct_match = torch.equal(tl_inf, ref_inf) and torch.equal(tl_nan, ref_nan)
        if struct_match:
            finite_mask = torch.isfinite(ref_out.cpu().float())
            if finite_mask.any():
                diff = (tl_out.cpu().float()[finite_mask] - ref_out.cpu().float()[finite_mask]).abs()
                tol = 2**-10 + 2**-6 * ref_out.cpu().float()[finite_mask].abs()
                finite_ratio = (diff <= tol).float().mean().item()
                finite_max = diff.max().item()
                if finite_ratio >= 0.99 and finite_max <= 1e0:
                    print(f"  [BOUNDARY_PASS] bnd_nan_input: struct match, finite ratio={finite_ratio:.6f}")
                else:
                    print(f"  [BOUNDARY_WARN] bnd_nan_input: struct match but finite ratio={finite_ratio:.6f} max={finite_max:.6e}")
                    warnings.append("bnd_nan_input: finite precision fail")
            else:
                print("  [BOUNDARY_PASS] bnd_nan_input: all-inf/nan, struct match")
        else:
            print("  [BOUNDARY_WARN] bnd_nan_input: inf/nan struct mismatch (expected nan propagation)")
            warnings.append("bnd_nan_input: inf/nan struct mismatch")
    except Exception as e:
        print(f"  [BOUNDARY_WARN] bnd_nan_input: exception {e}")
        warnings.append(f"bnd_nan_input: {e}")

    # BND-5: bf16 max values (dtype boundary)
    print("\n[BND-5] bnd_bf16_max: Q and KV at bf16 max range")
    try:
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.full((B, S, H, 576), 100.0, dtype=torch.bfloat16)
        kv = torch.full((B, SKV, 1, 576), 100.0, dtype=torch.bfloat16)
        indices = gen_indices_all_valid(B, S, 1, 64, SKV)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 63, 1)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 63, 1)
        out_p, out_r, out_m = check_precision(tl_out.cpu(), ref_out, "bfloat16")
        if out_p:
            print(f"  [BOUNDARY_PASS] bnd_bf16_max: ratio={out_r:.6f}")
        else:
            print(f"  [BOUNDARY_WARN] bnd_bf16_max: ratio={out_r:.6f} max={out_m:.6e}")
            warnings.append("bnd_bf16_max: precision fail")
    except Exception as e:
        print(f"  [BOUNDARY_WARN] bnd_bf16_max: exception {e}")
        warnings.append(f"bnd_bf16_max: {e}")

    # BND-6: all-masked query (q_start=0, all indices > max_kv_i=0)
    print("\n[BND-6] bnd_all_masked: B=1,S=4,SKV=64,H=128,topk=64,q_start=0")
    try:
        B, S, SKV, H = 1, 4, 64, 128
        q = torch.randn(B, S, H, 576, dtype=torch.bfloat16) / 10
        kv = torch.randn(B, SKV, 1, 576, dtype=torch.bfloat16) / 10
        # All indices point to positions > 0 (max_kv_i = 0 for q_start=0)
        indices = torch.randint(1, SKV, (B, S, 1, 64), dtype=torch.int32)
        tl_out, tl_lse = sparse_mla_fwd_interface(q.npu(), kv.npu(), indices.npu(), 0, 1)
        ref_out, ref_lse = ref_sparse_mla_fwd_interface(q, kv, indices, 0, 1)
        out_p, out_r, out_m = check_precision(tl_out.cpu(), ref_out, "bfloat16")
        if out_p:
            print(f"  [BOUNDARY_PASS] bnd_all_masked: ratio={out_r:.6f}")
        else:
            print(f"  [BOUNDARY_WARN] bnd_all_masked: ratio={out_r:.6f} (degenerate case)")
            warnings.append("bnd_all_masked: precision fail (degenerate)")
    except Exception as e:
        print(f"  [BOUNDARY_WARN] bnd_all_masked: exception {e}")
        warnings.append(f"bnd_all_masked: {e}")

    if warnings:
        print(f"\n[BOUNDARY_WARN] Boundary warnings: {len(warnings)}")
    else:
        print("\n[BOUNDARY_PASS] All boundary tests passed")
    return len(warnings) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="sparse_mla_fwd_seesaw layered tests")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    # Blocking gate: L0/L1 (precision) + L2 (negative-rejection).
    # Boundary is non-blocking (warnings only recorded).
    blocking_ok = True

    if args.level in ("l0", "all"):
        blocking_ok &= run_l0_suite()
    if args.level in ("l1", "all"):
        blocking_ok &= run_l1_suite()
    if args.level in ("l2", "all"):
        blocking_ok &= run_l2_suite()
    if args.level in ("boundary", "all"):
        run_boundary_suite()  # non-blocking: prints [BOUNDARY_PASS]/[BOUNDARY_WARN]

    if blocking_ok:
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All blocking tests passed!")
        print("Test Passed!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("[PRECISION_FAIL] Some blocking tests failed (see output above)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
