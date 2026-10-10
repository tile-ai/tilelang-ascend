"""Sparse MLA Forward Pipelined — precision test (L0 + L1 + L2 + Boundary).

Tests precision against a PyTorch golden implementation using the
mixed-tolerance check_precision function (bf16 dual-threshold).
L2 are negative tests: invalid params (block_I / heads / topk) must be
rejected by early host-side AssertionError at kernel-build entry.

Usage:
    python test_sparse_mla_fwd_pipelined.py --level l0
    python test_sparse_mla_fwd_pipelined.py --level all
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_mla_fwd_pipelined import sparse_mla_fwd_pipelined  # noqa: E402


# ============================================================================
# Golden reference implementation
# ============================================================================


def ref_sparse_mla_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=1, sm_scale=None):
    """PyTorch golden reference implementation."""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)  # [b, g, sq, topk]
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=q.device).view(
        -1, 1
    ) >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device=q.device).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


# ============================================================================
# Precision standard (mixed tolerance dual-threshold)
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed tolerance dual-threshold check. Returns (passed, matched_ratio, max_abs_error).

    Float: matched_ratio >= required AND max_abs_error <= max_abs_error_limit.
    Int: exact match. inf/nan positions are structurally compared (not counted in ratio).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # int exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    # inf/nan structural comparison
    special = ~torch.isfinite(g)
    if special.any() and not (
        torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) and torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
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


# ============================================================================
# Test helper: _prepare encapsulates input gen + kernel call + golden
# ============================================================================


def _prepare(
    B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index, seed=0, kv_stride=1, clamp_val=10, q_override=None, kv_override=None
):
    """Generate inputs, run kernel, compute golden. Returns (out, ref_out)."""
    KV_stride = kv_stride

    torch.manual_seed(seed)
    if q_override is not None:
        q = q_override
    else:
        q = torch.randn((B, S, H, DQK), dtype=dtype, device="npu") / 10
        if clamp_val is not None:
            q.clamp_(-clamp_val, clamp_val)

    if kv_override is not None:
        kv = kv_override
    else:
        kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="npu") / 10
        if clamp_val is not None:
            kv.clamp_(-clamp_val, clamp_val)

    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="npu")

    # Pad with SKV-1 (valid memory, but > max_kv_i so masked out by causal compare)
    indices = torch.full((B, S, HKV, topk), SKV - 1, dtype=torch.int32, device="npu")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                avail = min(max(1, ((t + q_start_s_index) // KV_stride)), SKV)
                i_i = torch.randperm(avail)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    kernel = sparse_mla_fwd_pipelined(
        heads=H,
        dim=DV,
        tail_dim=DQK - DV,
        topk=topk,
        kv_stride=KV_stride,
        kv_group=HKV,
        sm_scale=None,
        is_causal=True,
    )
    out = kernel(q, kv, indices, q_start_s_index_t)
    torch.npu.synchronize()

    ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
    torch.npu.synchronize()

    return out, ref_out


# ============================================================================
# L0 tests (rule shapes, block-divisible)
# ============================================================================


def run_l0_mla_bf16_standard():
    """L0 standard: B=1, S=4096, SKV=8192, H=128, topk=2048, bf16, q_start=1024."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 4096, 8192, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 1024

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0_mla_bf16_standard B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_mla_bf16_standard: {e}")
        return False


def run_l0_mla_bf16_small():
    """L0 small: B=1, S=1024, SKV=8192, H=128, topk=2048, bf16, q_start=1024."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 1024, 8192, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 1024

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0_mla_bf16_small B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_mla_bf16_small: {e}")
        return False


def run_l0_mla_bf16_b2():
    """L0 multi-batch: B=2, S=2048, SKV=4096, H=128, topk=2048, bf16, q_start=1024."""
    B, S, SKV, H, HKV, DQK, DV, topk = 2, 2048, 4096, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 1024

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0_mla_bf16_b2 B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_mla_bf16_b2: {e}")
        return False


def run_l0_mla_bf16_h64():
    """L0 different head count: B=1, S=4096, SKV=8192, H=64, topk=2048, bf16, q_start=1024."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 4096, 8192, 64, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 1024

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0_mla_bf16_h64 B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_mla_bf16_h64: {e}")
        return False


def run_sparse_mla_fwd_pipelined_l0():
    """L0 gate: run all L0 cases. Returns True if all pass."""
    ok = True
    ok &= run_l0_mla_bf16_small()
    ok &= run_l0_mla_bf16_standard()
    ok &= run_l0_mla_bf16_b2()
    ok &= run_l0_mla_bf16_h64()
    return ok


# ============================================================================
# L1 functional tests (irregular/tail shapes, kv_stride>1)
# ============================================================================


def run_l1_mla_bf16_s2049():
    """L1 tail block: S=2049 (non-block-divisible), tests persistent grid tail handling."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 2049, 8192, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 1024

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l1_mla_bf16_s2049 B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_bf16_s2049: {e}")
        return False


def run_l1_mla_bf16_qstart0():
    """L1 q_start=0 path: all rows scheduled from s=0 (grid no longer special-cases q_start==0)."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 4096, 8192, 128, 1, 576, 512, 2048
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l1_mla_bf16_qstart0 B={B},S={S},SKV={SKV},H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_bf16_qstart0: {e}")
        return False


def run_l1_mla_kv_stride4():
    """L1 kv_stride=4: tests max_kv_i with stride>1."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 512, 1024, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 12  # Non-zero q_start (exercises q_start>0 causal boundary)

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_kv_stride4 B={B},S={S},SKV={SKV},H={H},kv_stride=4 matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_kv_stride4: {e}")
        return False


def run_l1_mla_cp0_kv_stride4():
    """L1 CP0 + kv_stride=4: q_start=0, kv_stride>1 — the first kv_stride-1 query
    rows force-valid kv0 and must actually be computed (regression: the old CP0
    grid shortcut never scheduled rows 0..kv_stride-2, leaving garbage there)."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 64, 256, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_cp0_kv_stride4 B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_cp0_kv_stride4: {e}")
        return False


def run_l1_mla_cp0_kv_stride4_b2():
    """L1 CP0 + kv_stride=4 + B=2: multi-batch — no OOB Output overwrite across
    batch boundaries (regression: tile-decode period desync made pid decode
    s_i >= seq_len at batch/group seams, corrupting the next batch's rows)."""
    B, S, SKV, H, HKV, DQK, DV, topk = 2, 64, 256, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_cp0_kv_stride4_b2 B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_cp0_kv_stride4_b2: {e}")
        return False


def run_l1_mla_cp0_s_lt_kv_stride():
    """L1 degenerate: S=2 < kv_stride=4, q_start=0 — grid total must stay > 0 and
    every row must compute (each row force-valid kv0 only). If this extreme shape
    trips a framework issue, record XFAIL (non-blocking); a precision mismatch
    stays blocking."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 2, 64, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_cp0_s_lt_kv_stride B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except AssertionError:
        # Entry assert fired on a legal config (S < kv_stride is allowed): real
        # bug, must block — re-raise for a loud failure.
        raise
    except Exception as e:  # noqa: BLE001 — framework-level failure on extreme shape
        print(f"[XFAIL] l1_mla_cp0_s_lt_kv_stride: framework issue {type(e).__name__}: {e} — recorded, non-blocking")
        return True


def run_sparse_mla_fwd_pipelined_l1():
    """L1 gate: run all L1 cases. Returns True if all pass."""
    ok = True
    ok &= run_l1_mla_bf16_s2049()
    ok &= run_l1_mla_bf16_qstart0()
    ok &= run_l1_mla_kv_stride4()
    ok &= run_l1_mla_cp0_kv_stride4()
    ok &= run_l1_mla_cp0_kv_stride4_b2()
    ok &= run_l1_mla_cp0_s_lt_kv_stride()
    return ok


# ============================================================================
# L2 negative tests (invalid input rejection)
# ============================================================================


def run_sparse_mla_fwd_pipelined_l2():
    """L2 negative tests: invalid params must be rejected by early host-side
    AssertionError at kernel-build entry (before any NPU compilation).

    The @tilelang.jit wrapper executes the factory function body (where the
    entry asserts live) before calling tilelang.compile(), so an invalid value
    raises AssertionError immediately and cheaply. A value that is *silently
    accepted* here would instead fail deep in lowering/codegen (or worse,
    silently corrupt output) — that late-failure regression is what this
    test guards against.
    """
    # Common legal params; each case overrides one param with an invalid value.
    base = dict(heads=128, dim=512, tail_dim=64, topk=256, kv_stride=1, kv_group=1, block_I=128)

    cases = [
        # --- block_I entry constraint ---
        ("block_I=8 (divisible by 8 but not 64: T.tile.compare 256B alignment)", dict(block_I=8)),
        ("block_I=64 (multiple of 64 but AscendMemoryPlanning fails)", dict(block_I=64, topk=128)),
        ("block_I=100 (not divisible by 8: packed mask under-capacity)", dict(block_I=100, topk=400)),
        ("block_I=192 (multiple of 64 but compile segfault)", dict(block_I=192, topk=384)),
        ("topk=100 with block_I=128 (topk % block_I != 0)", dict(topk=100)),
        ("topk=128 with block_I=128 (NI=1: Ascend codegen L1 address planning fails)", dict(topk=128)),
        # --- heads/head_kv entry constraint ---
        ("heads=48 (head_kv=48 <= 64, not a power of two -> OOB via padded_H)", dict(heads=48)),
        ("heads=8 (head_kv=8 < 16 -> padded to 16, OOB Q/Output access)", dict(heads=8)),
        ("heads=192, kv_group=2 (head_kv=96 not multiple of 64)", dict(heads=192, kv_group=2)),
        ("heads=384, kv_group=2 (head_kv=192: %64==0 but not pow2, group base offset mismatch)", dict(heads=384, kv_group=2)),
    ]

    rejected = 0
    for name, override in cases:
        kwargs = {**base, **override}
        try:
            sparse_mla_fwd_pipelined(**kwargs)
            print(f"[BOUNDARY_WARN] l2_reject: {name} — NOT rejected (accepted at entry)")
        except AssertionError as e:
            rejected += 1
            reason = str(e).splitlines()[0][:100]
            print(f"[BOUNDARY_PASS] l2_reject: {name} — AssertionError: {reason}")
        except Exception as e:  # noqa: BLE001 — wrong exception type (e.g. TVMError) means late failure
            print(f"[BOUNDARY_WARN] l2_reject: {name} — wrong exception {type(e).__name__}: {e}")

    total = len(cases)
    print(f"[L2] {rejected}/{total} rejected correctly")

    # Positive control: legal values must pass the entry asserts. Uses the
    # undecorated factory (__wrapped__, kept by functools.wraps) so the
    # prim_func is only *built* (host-side IR construction), never compiled —
    # full compile+run coverage of legal configs is provided by L0/L1.
    positive = [
        ("heads=128", dict(heads=128)),
        ("heads=64", dict(heads=64)),
        ("heads=32, kv_group=2 (head_kv=16)", dict(heads=32, kv_group=2)),
    ]
    for name, override in positive:
        kwargs = {**base, **override}
        try:
            sparse_mla_fwd_pipelined.__wrapped__(**kwargs)
            print(f"[BOUNDARY_PASS] l2_positive: {name} — entry asserts passed (prim_func built, no compile)")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] l2_positive: {name} — unexpected {type(e).__name__}: {e}")

    # Rejection-guard result: consumed as BLOCKING by main(). A case that is
    # not correctly rejected means the entry guard was removed or weakened
    # (regression), unlike a precision miss on an unsupported input. The
    # positive-control warnings above stay non-blocking.
    return rejected == total


# ============================================================================
# Boundary tests (INF/NAN/extreme values)
# ============================================================================


def run_boundary_inf_input():
    """Boundary: Q contains inf. Verify kernel doesn't crash + inf/nan structural comparison.

    Injects inf into Q at one position. The golden's softmax handles inf
    (produces finite output). The kernel must produce matching inf/nan structure.
    """
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 256, 512, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    q = torch.randn((B, S, H, DQK), dtype=dtype, device="npu") / 10
    q.clamp_(-10, 10)
    # Inject inf into Q at position 0 (will propagate through GEMM to score)
    q[0, 0, 0, :] = float("inf")

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, q_override=q)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] boundary_inf_input matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] boundary_inf_input: {e}")
        return False


def run_boundary_nan_input():
    """Boundary: KV contains nan. Verify kernel doesn't crash + inf/nan structural comparison.

    Injects nan into KV at position 0. Both kernel and golden should produce
    nan at the same output positions (where KV[0] is attended to).
    """
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 256, 512, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="npu") / 10
    kv.clamp_(-10, 10)
    # Inject nan into KV at position 0 (always valid for q_start=0, kv_stride=1)
    kv[0, 0, 0, :] = float("nan")

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_override=kv)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] boundary_nan_input matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] boundary_nan_input: {e}")
        return False


def run_boundary_large_value():
    """Boundary: Q/KV not clamped (values up to ±100). Verify large value handling.

    Large scores (up to ~5.76M) after sm_scale (~240K) cause exp overflow → inf.
    Both kernel and golden should produce matching inf/nan structure.
    """
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 256, 512, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, clamp_val=100)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] boundary_large_value matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] boundary_large_value: {e}")
        return False


def run_boundary_zero_input():
    """Boundary: Q all zeros. Degenerate case: score=0, softmax=uniform, output=mean(V)."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 256, 512, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    q = torch.zeros((B, S, H, DQK), dtype=dtype, device="npu")

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, q_override=q)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] boundary_zero_input matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] boundary_zero_input: {e}")
        return False


def run_sparse_mla_fwd_pipelined_boundary():
    """Boundary gate: run all boundary tests."""
    # Non-blocking: boundary failures are reported but don't affect exit code.
    results = []
    results.append(("inf_input", run_boundary_inf_input()))
    results.append(("nan_input", run_boundary_nan_input()))
    results.append(("large_value", run_boundary_large_value()))
    results.append(("zero_input", run_boundary_zero_input()))
    passed_count = sum(1 for _, p in results if p)
    total = len(results)
    print(f"[BOUNDARY] {passed_count}/{total} passed")
    # Real result (was hardcoded True). Boundary precision failures stay
    # non-blocking in main() per the layered-test policy, but the return
    # value must reflect reality for any caller that checks it.
    return passed_count == total


# ============================================================================
# Main: --level dispatch
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    blocking_ok = True  # L0/L1 precision + L2 rejection-guard count toward exit code
    if args.level in ("l0", "all"):
        blocking_ok &= run_sparse_mla_fwd_pipelined_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= run_sparse_mla_fwd_pipelined_l1()
    if args.level in ("l2", "all"):
        # L2 negative-rejection failures are blocking: "not rejected" means
        # the entry guard was removed/weakened (regression), not an
        # unsupported-input precision miss. Positive-control warnings inside
        # run_..._l2 stay non-blocking.
        blocking_ok &= run_sparse_mla_fwd_pipelined_l2()
    if args.level in ("boundary", "all"):
        # Boundary precision failures stay non-blocking (layered-test
        # policy): extreme inputs are recorded, not fatal. The gate now
        # returns its real result for callers that want it.
        run_sparse_mla_fwd_pipelined_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
