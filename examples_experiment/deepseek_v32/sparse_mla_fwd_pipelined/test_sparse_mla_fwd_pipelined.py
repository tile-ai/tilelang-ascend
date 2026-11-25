"""Sparse MLA Forward Pipelined — layered test suite.

Levels:
    l0        precision gate on regular, block-divisible shapes
    l1        functional coverage: tail shapes, kv_stride > 1, q_start = 0,
              multi-batch and degenerate shapes
    l2        negative tests: invalid params must be rejected by a host-side
              AssertionError at kernel-build entry, plus the KV0 input contract
    boundary  special values (inf / nan / unclamped / all-zero inputs);
              reported but non-blocking

Precision is compared against a PyTorch golden implementation using the
mixed-tolerance dual gate in check_precision.

Usage:
    python test_sparse_mla_fwd_pipelined.py --level l0
    python test_sparse_mla_fwd_pipelined.py --level all
"""

import argparse
import os
import sys

import numpy as np
import tilelang
import torch

# Import kernel from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_mla_fwd_pipelined import check_indices_kv0_contract, sparse_mla_fwd_pipelined  # noqa: E402


# ============================================================================
# XFAIL whitelist
# ============================================================================

# Known framework issues that may be recorded as a non-blocking XFAIL, keyed by
# test name. Each entry is a tuple of substrings and the exception's
# "TypeName: message" must contain one of them; anything else is a real failure
# and is re-raised.
#
# An entry is only admissible when the failure has actually been reproduced, is
# traced to the framework rather than to this kernel, and has a signature
# specific enough not to match an unrelated error. No case in this suite meets
# that bar, so the table is empty and every exception fails the gate.
KNOWN_FRAMEWORK_XFAILS = {}


def xfail_or_raise(test_name, exc):
    """Return True only if `exc` matches a whitelisted known framework issue.

    Any other exception is reported and re-raised, so the gate fails loudly
    instead of passing silently.
    """
    msg = f"{type(exc).__name__}: {exc}"
    for signature in KNOWN_FRAMEWORK_XFAILS.get(test_name, ()):
        if signature in msg:
            print(f"[XFAIL] {test_name}: whitelisted framework issue (matched {signature!r}) — {msg} — non-blocking")
            return True
    print(f"[PRECISION_FAIL] {test_name}: unexpected {msg} — not whitelisted for XFAIL, treating as real failure")
    raise exc


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
# Precision standard
# ============================================================================

# The kernel computes in bfloat16 with float32 accumulation, so a single
# mixed-tolerance threshold set applies to every case.
BF16_ATOL = 2**-10
BF16_RTOL = 2**-6
BF16_MAX_ABS_ERR = 1e0
REQUIRED_MATCHED_RATIO = 0.99


def check_precision(actual, golden, dtype="bfloat16"):
    """Compare kernel output against the golden with a mixed-tolerance dual gate.

    An element counts as matched when |actual - golden| <= atol + rtol * |golden|.
    The case passes when matched_ratio >= REQUIRED_MATCHED_RATIO and the maximum
    absolute error <= BF16_MAX_ABS_ERR.

    Positions where the golden is not finite are compared structurally (nan
    against nan, inf against inf) and excluded from the ratio.

    Returns (passed, matched_ratio, max_abs_error).
    """
    assert dtype == "bfloat16", f"the kernel produces bfloat16 only, got {dtype!r}"
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()

    special = ~torch.isfinite(g)
    if special.any() and not (
        torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) and torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")

    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0

    abs_err = (a[m] - g[m]).abs()
    matched_ratio = (abs_err <= (BF16_ATOL + BF16_RTOL * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= REQUIRED_MATCHED_RATIO) and (max_abs_error <= BF16_MAX_ABS_ERR)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Test helper: _prepare encapsulates input gen + kernel call + golden
# ============================================================================


def _prepare(
    B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start_s_index, seed=0, kv_stride=1, clamp_val=10, q_override=None, kv_override=None
):
    """Generate inputs, run the kernel and compute the golden. Returns (out, ref_out)."""
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

    # Unused slots are padded with SKV-1: a valid address that the causal
    # comparison always masks out.
    indices = torch.full((B, S, HKV, topk), SKV - 1, dtype=torch.int32, device="npu")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                avail = min(max(1, ((t + q_start_s_index) // kv_stride)), SKV)
                i_i = torch.randperm(avail)[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    # The max(1, ...) clamp above guarantees the leading rows contain index 0,
    # which the kernel requires for rows with an empty causal window. Assert it
    # explicitly instead of relying on the generator's invariant.
    check_indices_kv0_contract(indices, kv_stride, q_start_s_index)

    kernel = sparse_mla_fwd_pipelined(
        heads=H,
        dim=DV,
        tail_dim=DQK - DV,
        topk=topk,
        kv_stride=kv_stride,
        kv_group=HKV,
        sm_scale=None,
        is_causal=True,
    )
    out = kernel(q, kv, indices, q_start_s_index_t)
    torch.npu.synchronize()

    ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index, kv_stride)
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
    """L1 q_start=0: the first query block starts at absolute position 0."""
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


def run_l1_mla_qstart0_kv_stride4():
    """L1 q_start=0 with kv_stride=4: the leading kv_stride-1 query rows have an
    empty causal window and attend KV0 only. They must still be scheduled and
    produce correct output."""
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 64, 256, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_qstart0_kv_stride4 B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_qstart0_kv_stride4: {e}")
        return False


def run_l1_mla_qstart0_kv_stride4_b2():
    """L1 q_start=0, kv_stride=4, B=2: multi-batch tile decoding must not write
    Output out of bounds across batch or group boundaries."""
    B, S, SKV, H, HKV, DQK, DV, topk = 2, 64, 256, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_qstart0_kv_stride4_b2 B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l1_mla_qstart0_kv_stride4_b2: {e}")
        return False


def run_l1_mla_qstart0_s_lt_kv_stride():
    """L1 degenerate: S=2 < kv_stride=4, q_start=0 — grid total must stay > 0 and
    every row must compute (each row attends kv0 only, per the KV0 input
    contract).

    Strict XFAIL: only exceptions whose signature is whitelisted in
    KNOWN_FRAMEWORK_XFAILS are recorded as non-blocking. Everything else
    (compile error, runtime error, OOM, ...) is re-raised so the L1 gate fails
    loudly. A precision mismatch is always blocking.
    """
    B, S, SKV, H, HKV, DQK, DV, topk = 1, 2, 64, 128, 1, 576, 512, 256
    dtype = torch.bfloat16
    q_start = 0

    try:
        out, ref = _prepare(B, S, SKV, H, HKV, DQK, DV, topk, dtype, q_start, kv_stride=4)
        passed, ratio, max_abs = check_precision(out, ref, "bfloat16")
        tag = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{tag}] l1_mla_qstart0_s_lt_kv_stride B={B},S={S},SKV={SKV},H={H},kv_stride=4,q_start=0 "
            f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except AssertionError:
        # Entry assert fired on a legal config (S < kv_stride is allowed): real
        # bug, must block — re-raise for a loud failure.
        raise
    except Exception as e:  # noqa: BLE001 — routed through the strict whitelist
        return xfail_or_raise("l1_mla_qstart0_s_lt_kv_stride", e)


def run_sparse_mla_fwd_pipelined_l1():
    """L1 gate: run all L1 cases. Returns True if all pass."""
    ok = True
    ok &= run_l1_mla_bf16_s2049()
    ok &= run_l1_mla_bf16_qstart0()
    ok &= run_l1_mla_kv_stride4()
    ok &= run_l1_mla_qstart0_kv_stride4()
    ok &= run_l1_mla_qstart0_kv_stride4_b2()
    ok &= run_l1_mla_qstart0_s_lt_kv_stride()
    return ok


# ============================================================================
# L2 negative tests (invalid input rejection)
# ============================================================================


def run_sparse_mla_fwd_pipelined_l2():
    """L2 negative tests: invalid params must be rejected by a host-side
    AssertionError at kernel-build entry, before any NPU compilation.

    The @tilelang.jit wrapper runs the factory body (where the entry asserts
    live) before tilelang.compile(), so an invalid value fails immediately and
    cheaply. A value that was silently accepted here would instead fail deep in
    lowering/codegen, or silently corrupt output.
    """
    # Common legal params; each case overrides one param with an invalid value.
    base = dict(heads=128, dim=512, tail_dim=64, topk=256, kv_stride=1, kv_group=1, block_I=128)

    cases = [
        # --- block_I / topk entry constraints ---
        ("block_I=8 (divisible by 8 but not 64: T.tile.compare needs 256B alignment)", dict(block_I=8)),
        ("block_I=64 (multiple of 64 but memory planning fails)", dict(block_I=64, topk=128)),
        ("block_I=100 (not divisible by 8: packed mask under-capacity)", dict(block_I=100, topk=400)),
        ("block_I=192 (multiple of 64 but compilation crashes)", dict(block_I=192, topk=384)),
        ("topk=100 with block_I=128 (topk % block_I != 0)", dict(topk=100)),
        ("topk=128 with block_I=128 (NI=1: no L1 address plan for the pipelined KV buffer)", dict(topk=128)),
        # --- heads / head_kv domain ---
        ("heads=48 (head_kv=48 <= 64, not a power of two -> OOB via padded_H)", dict(heads=48)),
        ("heads=8 (head_kv=8 < 16 -> padded to 16, OOB Q/Output access)", dict(heads=8)),
        ("heads=192, kv_group=2 (head_kv=96 not multiple of 64)", dict(heads=192, kv_group=2)),
        ("heads=384, kv_group=2 (head_kv=192: %64==0 but not pow2, group base offset mismatch)", dict(heads=384, kv_group=2)),
        # --- heads / kv_group divisibility: a truncated head_kv would leave the
        #     trailing heads unwritten, i.e. uninitialized Output ---
        ("heads=65, kv_group=2 (heads % kv_group != 0 -> head_kv=32 truncates, head 64 uninitialized)", dict(heads=65, kv_group=2)),
        ("heads=100, kv_group=8 (heads % kv_group != 0)", dict(heads=100, kv_group=8)),
        ("kv_group=0 (must be rejected as a contract error, not ZeroDivisionError)", dict(kv_group=0)),
        ("kv_group=-1 (must be >= 1)", dict(kv_group=-1)),
        ("heads=0 (must be >= 1)", dict(heads=0)),
        # --- kv_stride entry constraint (it is the device-side divisor of
        #     raw = (q_i + 1) // kv_stride) ---
        ("kv_stride=0 (would fail late in TVM IR construction with Divide by zero)", dict(kv_stride=0)),
        ("kv_stride=-1 (negative stride gives a meaningless causal boundary)", dict(kv_stride=-1)),
        # --- entry type normalization (numpy scalars are valid integers) ---
        ("heads=128.0 float (not an integer: would silently truncate)", dict(heads=128.0)),
        ("topk='256' str (not an integer)", dict(topk="256")),
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
        ("heads=128, kv_group=4 (head_kv=32, divisible -> must be accepted)", dict(heads=128, kv_group=4)),
        ("heads=256, kv_group=4 (head_kv=64, divisible -> must be accepted)", dict(heads=256, kv_group=4)),
        # numpy scalars are valid integers and must be normalized at entry rather
        # than failing deep in next_power_of_2 (int.bit_length) or T.Pipelined
        # (TVM cannot handle a numpy dtype).
        (
            "numpy int64 scalars (normalized at entry -> must be accepted)",
            dict(
                heads=np.int64(128),
                dim=np.int64(512),
                tail_dim=np.int64(64),
                topk=np.int64(256),
                kv_stride=np.int64(1),
                kv_group=np.int64(1),
                block_I=np.int64(128),
            ),
        ),
    ]
    for name, override in positive:
        kwargs = {**base, **override}
        try:
            sparse_mla_fwd_pipelined.__wrapped__(**kwargs)
            print(f"[BOUNDARY_PASS] l2_positive: {name} — entry asserts passed (prim_func built, no compile)")
        except Exception as e:  # noqa: BLE001
            print(f"[BOUNDARY_WARN] l2_positive: {name} — unexpected {type(e).__name__}: {e}")

    # Blocking: a case that is not rejected means an entry guard is missing or
    # weakened, which is a different class of problem from a precision miss on an
    # unsupported input. The positive-control warnings above stay non-blocking.
    return rejected == total


def run_l2_indices_kv0_contract():
    """L2 test for the early-row KV0 input contract.

    Rows with an empty causal window can only select index 0, so Indices must
    contain 0 there or the kernel produces NaN while the golden attends KV0.
    check_indices_kv0_contract makes that requirement explicit; this test
    verifies the check actually fires.

    Blocking: a verifier that stops detecting violations would reopen the
    silent-NaN path.
    """
    B, S, HKV, topk = 1, 8, 1, 256
    kv_stride = 4
    ok = True

    # --- negative: leading rows (t < kv_stride - 1) missing index 0 ---
    bad = torch.full((B, S, HKV, topk), 5, dtype=torch.int32, device="npu")
    bad[:, kv_stride - 1 :, :, 0] = 0  # only the non-early rows contain 0
    try:
        check_indices_kv0_contract(bad, kv_stride, q_start_index_s=0)
        print("[BOUNDARY_WARN] l2_kv0_contract: violation NOT detected (verifier is broken)")
        ok = False
    except ValueError as e:
        print(f"[BOUNDARY_PASS] l2_kv0_contract: violation rejected — {str(e).splitlines()[0][:110]}")

    # --- negative: non-raising mode must report the offending rows ---
    violations = check_indices_kv0_contract(bad, kv_stride, q_start_index_s=0, raise_on_violation=False)
    expected = [(0, t, 0) for t in range(kv_stride - 1)]
    if violations == expected:
        print(f"[BOUNDARY_PASS] l2_kv0_contract: violations reported {violations}")
    else:
        print(f"[BOUNDARY_WARN] l2_kv0_contract: expected {expected}, got {violations}")
        ok = False

    # --- positive: contract satisfied (index 0 present in every early row) ---
    good = bad.clone()
    good[:, : kv_stride - 1, :, 0] = 0
    try:
        check_indices_kv0_contract(good, kv_stride, q_start_index_s=0)
        print("[BOUNDARY_PASS] l2_kv0_contract: valid indices accepted")
    except ValueError as e:
        print(f"[BOUNDARY_WARN] l2_kv0_contract: valid indices wrongly rejected — {e}")
        ok = False

    # --- positive: kv_stride=1 has no early rows, so anything is accepted ---
    any_idx = torch.full((B, S, HKV, topk), 7, dtype=torch.int32, device="npu")
    try:
        check_indices_kv0_contract(any_idx, 1, q_start_index_s=0)
        print("[BOUNDARY_PASS] l2_kv0_contract: kv_stride=1 correctly has no early rows")
    except ValueError as e:
        print(f"[BOUNDARY_WARN] l2_kv0_contract: kv_stride=1 wrongly rejected — {e}")
        ok = False

    print(f"[L2] kv0 contract verifier: {'OK' if ok else 'BROKEN'}")
    return ok


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
    # Boundary failures are non-blocking in main(), but the return value still
    # reflects the real result for callers that check it.
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

    # L0/L1 precision and the L2 guards decide the exit code; Boundary is
    # reported but non-blocking.
    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= run_sparse_mla_fwd_pipelined_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= run_sparse_mla_fwd_pipelined_l1()
    if args.level in ("l2", "all"):
        # An invalid param that is not rejected means an entry guard is missing,
        # which is blocking. Positive-control warnings stay non-blocking.
        blocking_ok &= run_sparse_mla_fwd_pipelined_l2()
        blocking_ok &= run_l2_indices_kv0_contract()
    if args.level in ("boundary", "all"):
        # Extreme inputs are recorded, not fatal.
        run_sparse_mla_fwd_pipelined_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
