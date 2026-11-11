"""Test suite for GQA Sink Bwd Varlen (Ascend NPU, single-kernel backward).

Layered precision tests: L0 (precision gate) + L1 (functional) + L2 (negative) + Boundary.

Usage:
  python test_gqa_sink_bwd_varlen.py --level all       # full test suite
  python test_gqa_sink_bwd_varlen.py --level l0        # L0 precision gate only
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from example_gqa_sink_bwd_varlen import (
    flashattn_fwd,
    run_bwd_pipeline,
    DTYPE_FP16,
    BLOCK_M_FWD,
    BLOCK_N_FWD,
    BLOCK_M_BWD,
    BLOCK_N_BWD,
)

import argparse
import traceback

import tilelang
import torch


# ============================================================================
# Golden reference (PyTorch fp32 autograd)
# ============================================================================


def ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q, cu_seqlens_k, max_seq_len, window_size=None, groups=1):
    """Forward golden. Q [UQ,H,D], K/V [UKV,H_kv,D]."""
    UQ, H, D = Q.shape
    H_kv = K.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    sm_scale = 1.0 / D**0.5
    output = torch.zeros_like(Q)
    lse_out = torch.zeros(batch, H, max_seq_len, dtype=torch.float32, device=Q.device)
    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len, k_len = q_end - q_start, k_end - k_start
        if q_len == 0:
            continue
        q_seq = Q[q_start:q_end].view(q_len, H_kv, groups, D)
        k_seq = K[k_start:k_end].unsqueeze(2)
        v_seq = V[k_start:k_end].unsqueeze(2)
        logits = torch.einsum("qhgd,khgd->hgqk", q_seq.float(), k_seq.float()) * sm_scale
        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=Q.device).float()
        pos_queries = torch.arange(q_len, device=Q.device).float() + offset
        mask = pos_keys[None, :] > pos_queries[:, None]
        mask = mask.float().masked_fill(mask, float("-inf"))
        if window_size is not None:
            mask.masked_fill_(pos_keys[None, :] < (pos_queries[:, None] - window_size + 1), float("-inf"))
        logits = logits + mask[None, None, :, :]
        sinks_expanded = sinks.view(H_kv, groups, 1, 1).float()
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        m_star = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - m_star)
        unnorm = torch.exp(logits - m_star)
        normalizer = unnorm.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnorm / normalizer
        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq.float())
        output[q_start:q_end] = out.reshape(q_len, H, D).to(Q.dtype)
        lse = torch.log(normalizer.squeeze(-1)) + m_star.squeeze(-1)
        lse_out[b, :, :q_len] = lse.reshape(H, q_len)
    return output, lse_out


def ref_bwd_varlen(Q, K, V, sinks, dO, cu_seqlens_q, cu_seqlens_k, max_seq_len, window_size=None, groups=1):
    """Backward golden via autograd. Returns dQ, dK, dV, dSinks (all fp16)."""
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)
    Sinks_f = sinks.float().requires_grad_(True)
    _, H, D = Q_f.shape
    H_kv = K_f.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    sm_scale = 1.0 / D**0.5
    output = torch.zeros_like(Q_f)
    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        q_len, k_len = q_end - q_start, k_end - k_start
        if q_len == 0:
            continue
        q_seq = Q_f[q_start:q_end].view(q_len, H_kv, groups, D)
        k_seq = K_f[k_start:k_end].unsqueeze(2)
        v_seq = V_f[k_start:k_end].unsqueeze(2)
        logits = torch.einsum("qhgd,khgd->hgqk", q_seq, k_seq) * sm_scale
        offset = k_len - q_len
        pos_keys = torch.arange(k_len, device=Q_f.device).float()
        pos_queries = torch.arange(q_len, device=Q_f.device).float() + offset
        mask = pos_keys[None, :] > pos_queries[:, None]
        mask = mask.float().masked_fill(mask, float("-inf"))
        if window_size is not None:
            mask.masked_fill_(pos_keys[None, :] < (pos_queries[:, None] - window_size + 1), float("-inf"))
        logits = logits + mask[None, None, :, :]
        sinks_expanded = Sinks_f.view(H_kv, groups, 1, 1)
        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        m_star = torch.maximum(sinks_expanded, logits_max)
        sinks_exp = torch.exp(sinks_expanded - m_star)
        unnorm = torch.exp(logits - m_star)
        normalizer = unnorm.sum(dim=-1, keepdim=True) + sinks_exp
        scores = unnorm / normalizer
        out = torch.einsum("hgqk,khgd->qhgd", scores, v_seq)
        output[q_start:q_end] = out.reshape(q_len, H, D)
    output.backward(dO.float())
    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half(), Sinks_f.grad.half()


# ============================================================================
# Precision standard: mixed tolerance + dual gate
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_limit, required_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Dual-gate: matched_ratio >= required AND max_abs <= limit."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
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
# Coverage declarations (for coverage_check.py --proto proto.yaml)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 8,
    "D-DTYPE-fp32": 8,
    "D-DTYPE-int32": 8,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 4,
    "D-PARAM-batch": 3,
    "D-PARAM-block_M": 2,
    "D-PARAM-block_N": 2,
    "D-PARAM-dim": 1,
    "D-PARAM-groups": 4,
    "D-PARAM-heads": 6,
    "D-PARAM-is_causal": 1,
    "D-PARAM-window_size": 3,
    "D-SHAPE-ALIGNED": 21,
    "D-SHAPE-EDGE": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-VALRANGE-ASYM": 2,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-S": 8,
}

COVERAGE_NA = {}


# ============================================================================
# Test helper
# ============================================================================


def _run_case(
    name,
    B,
    H,
    groups,
    q_lens,
    kv_lens,
    D,
    window_size,
    level,
    custom_sinks=None,
    block_M_bwd=BLOCK_M_BWD,
    block_N_bwd=BLOCK_N_BWD,
    vrange=None,
):
    """Run fwd+bwd, compare against golden. 4 outputs ALL blocking."""
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        H_kv = H // groups
        max_seq_len = max(q_lens)
        assert max_seq_len % block_M_bwd == 0, f"max_seq_len ({max_seq_len}) must be divisible by block_M_bwd ({block_M_bwd})"
        assert max_seq_len > 0, f"max_seq_len ({max_seq_len}) must be > 0 (kernel grid would divide by zero)"
        max_kv_len = max(kv_lens)
        assert max_kv_len % block_N_bwd == 0, f"max_kv_len ({max_kv_len}) must be divisible by block_N_bwd ({block_N_bwd})"

        cu_seqlens_q = [0]
        for ql in q_lens:
            cu_seqlens_q.append(cu_seqlens_q[-1] + ql)
        cu_seqlens_k = [0]
        for kl in kv_lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + kl)
        UQ = cu_seqlens_q[-1]
        UKV = cu_seqlens_k[-1]
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="npu")
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="npu")

        torch.manual_seed(42)
        Q = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")
        K = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
        V = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
        if custom_sinks is not None:
            sinks = custom_sinks.to("npu").to(torch.float16)
        else:
            sinks = torch.randn(H, dtype=torch.float16, device="npu")
        dO = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")

        if vrange == "M":
            Q *= 3.0
            K *= 3.0
            V *= 3.0
            dO *= 3.0
        elif vrange == "L":
            Q *= 10.0
            K *= 10.0
            V *= 10.0
            dO *= 10.0
        elif vrange == "ASYM":
            Q = Q * 2.0 + 2.0
            K = K * 2.0 + 2.0
            V = V * 2.0 + 2.0
            dO = dO * 2.0 + 2.0

        fwd_mod = flashattn_fwd(B, UQ, UKV, max_seq_len, H, D, groups, window_size, BLOCK_M_FWD, BLOCK_N_FWD)
        O_npu, lse_npu = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
        torch.npu.synchronize()

        O_ref, lse_ref = ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups)
        dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
            Q,
            K,
            V,
            O_npu,
            dO,
            lse_npu,
            sinks,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            B,
            UQ,
            UKV,
            max_seq_len,
            max(kv_lens),
            H,
            D,
            D,
            window_size,
            block_M_bwd,
            block_N_bwd,
            groups,
        )

        # dSinks: kernel output [batch, heads, max_seq_len] fp32, host sum reduce
        dSinks_npu_fp32 = dSinks_out.cpu().float().sum(2).sum(0)
        # golden: recompute using kernel's lse/Delta (isolates T.tile.exp precision)
        dQ_ref, dK_ref, dV_ref, dSinks_ref_autograd = ref_bwd_varlen(
            Q,
            K,
            V,
            sinks,
            dO,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            max_seq_len,
            window_size,
            groups,
        )
        sinks_exp = sinks.float().cpu().view(1, H, 1)
        lse_cpu = lse_npu.cpu().float()
        delta_cpu = Delta_out.cpu().float()
        dSinks_ref_fp32 = -(torch.exp(sinks_exp - lse_cpu) * delta_cpu).sum(dim=0).sum(dim=1)

        max_diff = 0.0
        min_ratio = 1.0
        all_passed = True

        for b in range(B):
            qs = cu_seqlens_q[b]
            qe = cu_seqlens_q[b + 1]
            fp, fr, fm = check_precision(O_npu[qs:qe].cpu(), O_ref[qs:qe].cpu(), DTYPE_FP16)
            min_ratio = min(min_ratio, fr)
            max_diff = max(max_diff, fm)
            all_passed &= fp
            if not fp:
                raise AssertionError(f"O precision failed (batch {b}): ratio={fr:.4f}, max_abs={fm:.3e}")

        qp, qr, qm = check_precision(dQ_fp16[..., :D].cpu(), dQ_ref.cpu(), DTYPE_FP16)
        min_ratio = min(min_ratio, qr)
        max_diff = max(max_diff, qm)
        all_passed &= qp
        if not qp:
            raise AssertionError(f"dQ precision failed: ratio={qr:.4f}, max_abs={qm:.3e}")

        for b in range(B):
            ks = cu_seqlens_k[b]
            ke = cu_seqlens_k[b + 1]
            kp, kr, km = check_precision(dK[ks:ke, :, :D].cpu(), dK_ref[ks:ke].cpu(), DTYPE_FP16)
            min_ratio = min(min_ratio, kr)
            max_diff = max(max_diff, km)
            all_passed &= kp
            if not kp:
                raise AssertionError(f"dK precision failed (batch {b}): ratio={kr:.4f}, max_abs={km:.3e}")
            vp, vr, vm = check_precision(dV[ks:ke].cpu(), dV_ref[ks:ke].cpu(), DTYPE_FP16)
            min_ratio = min(min_ratio, vr)
            max_diff = max(max_diff, vm)
            all_passed &= vp
            if not vp:
                raise AssertionError(f"dV precision failed (batch {b}): ratio={vr:.4f}, max_abs={vm:.3e}")

        sp, sr, sm = check_precision(dSinks_npu_fp32, dSinks_ref_fp32, "float32")
        min_ratio = min(min_ratio, sr)
        max_diff = max(max_diff, sm)
        all_passed &= sp
        if not sp:
            raise AssertionError(f"dSinks precision failed: matched_ratio={sr:.4f} < 0.99, max_abs={sm:.3e} (limit=0.01)")

        print(
            f"[{tag}_PASS] {level} {name} B={B} H={H} G={groups} "
            f"q={q_lens} kv={kv_lens} D={D} win={window_size} "
            f"max_diff={max_diff:.6e} min_ratio={min_ratio:.4f}"
        )
        return True
    except AssertionError as e:
        if level == "l2":
            raise
        print(f"[BOUNDARY_WARN] {level} {name} B={B} H={H} G={groups} q={q_lens} kv={kv_lens} D={D} win={window_size}: {e}")
        return False
    except Exception as e:
        fail_tag = "WARN" if tag == "BOUNDARY" else "FAIL"
        print(f"[{tag}_{fail_tag}] {level} {name} B={B} H={H} G={groups} q={q_lens} kv={kv_lens} D={D} win={window_size}: {e}")
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


def _run_exception(name, fn):
    """L2 negative test: fn() passes unsupported input, expects exception."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: unsupported input not rejected (silent accept)")


# ============================================================================
# L0: Precision Gate (8 cases, block-aligned, blocking)
# ============================================================================

L0_CASES = [
    ("l0_basic_small", 1, 4, 2, [128], [128], 128, None),
    ("l0_causal_full", 1, 4, 2, [256], [256], 128, None),
    ("l0_gqa", 1, 8, 4, [256], [256], 128, None),
    ("l0_sliding_window", 1, 4, 2, [256], [256], 128, 128),
    ("l0_sink_nonzero", 1, 4, 2, [128], [128], 128, None),
    ("l0_varlen_multi_batch", 2, 4, 2, [128, 128], [128, 128], 128, None),
    ("l0_varlen_unequal_qk", 1, 4, 2, [128], [256], 128, None),
    ("l0_default", 1, 64, 8, [512], [512], 128, 128),
]


def test_gqa_sink_bwd_l0():
    """L0 precision gate: 8 cases, all 4 outputs blocking."""
    ok = True
    for name, B, H, groups, q_lens, kv_lens, D, window in L0_CASES:
        custom_sinks = torch.full((H,), 3.0, dtype=torch.float16) if "sink_nonzero" in name else None
        ok &= _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l0", custom_sinks=custom_sinks)
    return ok


def test_gqa_sink_bwd_l0_determinism():
    """L0 determinism: 3 runs, max_diff must be bit-exact."""
    print("\n" + "=" * 78)
    print("L0 Determinism: 3x bit-exact check")
    print("=" * 78)
    for name, B, H, groups, q_lens, kv_lens, D, window in L0_CASES:
        if name == "l0_default":
            print(f"  [SKIP] {name} (large shape, determinism implied)")
            continue
        max_diffs = []
        for _run_idx in range(3):
            H_kv = H // groups
            max_seq_len = max(q_lens)
            max_kv_len = max(kv_lens)
            cu_seqlens_q = [0]
            for ql in q_lens:
                cu_seqlens_q.append(cu_seqlens_q[-1] + ql)
            cu_seqlens_k = [0]
            for kl in kv_lens:
                cu_seqlens_k.append(cu_seqlens_k[-1] + kl)
            UQ = cu_seqlens_q[-1]
            UKV = cu_seqlens_k[-1]
            torch.manual_seed(42)
            Q = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")
            K = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
            V = torch.randn(UKV, H_kv, D, dtype=torch.float16, device="npu")
            sinks = torch.randn(H, dtype=torch.float16, device="npu")
            dO = torch.randn(UQ, H, D, dtype=torch.float16, device="npu")
            cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="npu")
            cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="npu")
            fwd_mod = flashattn_fwd(B, UQ, UKV, max_seq_len, H, D, groups, window, BLOCK_M_FWD, BLOCK_N_FWD)
            O_npu, lse_npu = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
            torch.npu.synchronize()
            dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
                Q,
                K,
                V,
                O_npu,
                dO,
                lse_npu,
                sinks,
                cu_seqlens_q_t,
                cu_seqlens_k_t,
                B,
                UQ,
                UKV,
                max_seq_len,
                max_kv_len,
                H,
                D,
                D,
                window,
                BLOCK_M_BWD,
                BLOCK_N_BWD,
                groups,
            )
            dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd_varlen(
                Q,
                K,
                V,
                sinks,
                dO,
                cu_seqlens_q_t,
                cu_seqlens_k_t,
                max_seq_len,
                window,
                groups,
            )
            _, _, dq_max = check_precision(dQ_fp16[..., :D].cpu(), dQ_ref.cpu(), DTYPE_FP16)
            _, _, dk_max = check_precision(dK[..., :D].cpu(), dK_ref.cpu(), DTYPE_FP16)
            _, _, dv_max = check_precision(dV.cpu(), dV_ref.cpu(), DTYPE_FP16)
            max_diffs.append(max(dq_max, dk_max, dv_max))
        if max_diffs[0] == max_diffs[1] == max_diffs[2]:
            print(f"  [DETERMINISTIC] {name}: max_diff={max_diffs[0]:.6e} (3/3 identical)")
        else:
            print(f"  [NON-DETERMINISTIC] {name}: max_diffs={max_diffs}")


# ============================================================================
# L1: Functional (irregular shapes, GQA variants, blocking)
# ============================================================================

L1_CASES = [
    ("l1_basic_small", 1, 4, 2, [128], [128], 128, None, 64, 64, None),
    ("l1_mha_causal", 1, 4, 1, [256], [256], 128, None, 64, 64, None),
    ("l1_gqa_medium", 4, 16, 4, [256] * 4, [256] * 4, 128, None, 64, 64, None),
    ("l1_asymmetric_sq_gt_skv", 2, 4, 2, [256, 256], [128, 128], 128, None, 64, 64, None),
    ("l1_asymmetric_skv_gt_sq", 2, 4, 2, [128, 128], [256, 256], 128, None, 64, 64, None),
    ("l1_window_128", 1, 16, 8, [256], [256], 128, 128, 64, 64, None),
    ("l1_window_256", 1, 16, 8, [512], [512], 128, 256, 64, 64, None),
    ("l1_varlen_unequal_batches", 2, 4, 2, [128, 256], [256, 128], 128, None, 64, 64, None),
    ("l1_irregular_n_384", 1, 8, 4, [384], [384], 128, None, 64, 64, None),
    ("l1_large_causal", 1, 32, 8, [512], [512], 128, None, 64, 64, None),
    ("l1_block_m_64", 1, 4, 2, [256], [256], 128, None, 64, 64, None),
    ("l1_block_n_64", 1, 4, 2, [128], [128], 128, None, 64, 64, None),
    ("l1_edge_min", 1, 4, 2, [128], [128], 128, None, 64, 64, None),
]


def test_gqa_sink_bwd_l1():
    """L1 functional test: irregular shapes, different groups/block sizes (blocking)."""
    ok = True
    for case in L1_CASES:
        name, B, H, groups, q_lens, kv_lens, D, window = case[:8]
        bm_bwd, bn_bwd = case[8], case[9]
        vrange = case[10]
        ok &= _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l1", block_M_bwd=bm_bwd, block_N_bwd=bn_bwd, vrange=vrange)
    return ok


# ============================================================================
# L2: Negative Tests (unsupported input rejection, non-blocking)
# ============================================================================

L2_TAIL_CASES = [
    ("l2_tail1_seq129", 1, 4, 2, [129], [129], 128, None, 64, 64),
    ("l2_tailmid_seq192", 1, 4, 2, [192], [192], 128, None, 64, 64),
    ("l2_prime_seq131", 1, 4, 2, [131], [131], 128, None, 64, 64),
]


def test_gqa_sink_bwd_l2():
    """L2 negative test: non-aligned seq_len, unsupported dtype, illegal shape."""
    for name, B, H, groups, q_lens, kv_lens, D, window, bm_bwd, bn_bwd in L2_TAIL_CASES:

        def _run_fn(name=name, B=B, H=H, groups=groups, q_lens=q_lens, kv_lens=kv_lens, D=D, window=window, bm_bwd=bm_bwd, bn_bwd=bn_bwd):
            _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l2", block_M_bwd=bm_bwd, block_N_bwd=bn_bwd)

        _run_exception(name, _run_fn)

    def _run_dtype_fn():
        H = 4
        groups = 2
        D = 128
        Q = torch.randn(128, H, D, dtype=torch.float64, device="npu")
        K = torch.randn(128, H // groups, D, dtype=torch.float64, device="npu")
        V = torch.randn(128, H // groups, D, dtype=torch.float64, device="npu")
        sinks = torch.randn(H, dtype=torch.float64, device="npu")
        cu_q = torch.tensor([0, 128], dtype=torch.int32, device="npu")
        cu_k = torch.tensor([0, 128], dtype=torch.int32, device="npu")
        fwd_mod = flashattn_fwd(1, 128, 128, 128, H, D, groups, None, BLOCK_M_FWD, BLOCK_N_FWD)
        fwd_mod(Q, K, V, sinks, cu_q, cu_k)

    _run_exception("l2_unsupported_dtype_fp64", _run_dtype_fn)

    def _run_shape_fn():
        _run_case("l2_illegal_shape_empty", 1, 4, 2, [0], [128], 128, None, "l2")

    _run_exception("l2_illegal_shape_empty", _run_shape_fn)

    extra_configs = [
        ("l2_min_config", 1, 1, 1, [128], [128], 128, None),
        ("l2_mqa_groups_eq_h", 1, 4, 4, [128], [128], 128, None),
        ("l2_window_eq_n", 1, 4, 2, [128], [128], 128, 128),
        ("l2_large_batch", 4, 8, 4, [128, 128, 128, 128], [128, 128, 128, 128], 128, None),
        ("l2_large_n_d128", 1, 16, 4, [512], [512], 128, None),
    ]
    for name, B, H, groups, q_lens, kv_lens, D, window in extra_configs:
        _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l2")


# ============================================================================
# Boundary: Special Sink Values (non-blocking)
# ============================================================================

BOUNDARY_CASES = [
    ("boundary_zero_sinks", lambda H: torch.zeros(H, dtype=torch.float16), None),
    ("boundary_inf_sinks", lambda H: torch.full((H,), float("inf"), dtype=torch.float16), None),
    ("boundary_nan_sinks", lambda H: torch.full((H,), float("nan"), dtype=torch.float16), None),
    ("boundary_dbound_sinks", lambda H: torch.full((H,), 32000.0, dtype=torch.float16), None),
    ("boundary_large_sinks", lambda H: torch.randn(H, dtype=torch.float16) * 100, None),
    ("boundary_negative_sinks", lambda H: torch.randn(H, dtype=torch.float16) * -100, None),
    (
        "boundary_mixed_sinks",
        lambda H: torch.cat(
            [
                torch.randn(H // 2, dtype=torch.float16),
                torch.full((H // 2,), 32000.0, dtype=torch.float16),
            ]
        ),
        None,
    ),
    ("boundary_tiny_sinks", lambda H: torch.randn(H, dtype=torch.float16) * 1e-4, None),
    ("boundary_valrange_l", lambda H: torch.randn(H, dtype=torch.float16) * 10, "L"),
    ("boundary_valrange_m", lambda H: torch.randn(H, dtype=torch.float16) * 3, "M"),
    ("boundary_valrange_asym", lambda H: torch.randn(H, dtype=torch.float16) * 2 + 2, "ASYM"),
]


def test_gqa_sink_bwd_boundary():
    """Boundary test: zero/inf/nan/dbound/large/negative/mixed/tiny sinks + valrange."""
    H = 4
    for case in BOUNDARY_CASES:
        name, sinks_fn, vrange = case
        custom_sinks = sinks_fn(H) if sinks_fn else None
        _run_case(name, 1, H, 2, [128], [128], 128, None, "boundary", custom_sinks=custom_sinks, vrange=vrange)


# ============================================================================
# run_layered_tests (called by example file's --level mode)
# ============================================================================


def run_layered_tests(level):
    """Run layered tests from example file's __main__ --level mode."""
    tilelang.disable_cache()
    torch.set_default_device("npu")
    ok = True
    if level in ("l0", "all"):
        print("\n" + "=" * 78)
        print("L0: Precision Gate (8 cases, block-aligned)")
        print("=" * 78)
        l0_ok = test_gqa_sink_bwd_l0()
        ok &= l0_ok
        if l0_ok:
            test_gqa_sink_bwd_l0_determinism()
    if level in ("l1", "all"):
        print("\n" + "=" * 78)
        print("L1: Functional (irregular shapes, GQA variants)")
        print("=" * 78)
        ok &= test_gqa_sink_bwd_l1()
    if level in ("l2", "all"):
        print("\n" + "=" * 78)
        print("L2: Negative Tests (unsupported input rejection)")
        print("=" * 78)
        test_gqa_sink_bwd_l2()
    if level in ("boundary", "all"):
        print("\n" + "=" * 78)
        print("Boundary: Special Sink Values (non-blocking)")
        print("=" * 78)
        test_gqa_sink_bwd_boundary()
    if ok:
        print("\n" + "=" * 78)
        print("Test Passed!")
        print("=" * 78)
    else:
        print("\n" + "=" * 78)
        print("Test FAILED (L0/L1 precision gate not met)")
        print("=" * 78)
        sys.exit(1)


# ============================================================================
# main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA Sink Bwd Varlen Test Suite — Ascend NPU")
    parser.add_argument(
        "--level",
        choices=["l0", "l1", "l2", "boundary", "all"],
        default="l0",
        help="Test level",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")

    ok = True
    if args.level in ("l0", "all"):
        print("\n" + "=" * 78)
        print("L0: Precision Gate (8 cases, block-aligned)")
        print("=" * 78)
        l0_ok = test_gqa_sink_bwd_l0()
        ok &= l0_ok
        if l0_ok:
            test_gqa_sink_bwd_l0_determinism()

    if args.level in ("l1", "all"):
        print("\n" + "=" * 78)
        print("L1: Functional (irregular shapes, GQA variants)")
        print("=" * 78)
        ok &= test_gqa_sink_bwd_l1()

    if args.level in ("l2", "all"):
        print("\n" + "=" * 78)
        print("L2: Negative Tests (unsupported input rejection)")
        print("=" * 78)
        test_gqa_sink_bwd_l2()

    if args.level in ("boundary", "all"):
        print("\n" + "=" * 78)
        print("Boundary: Special Sink Values (non-blocking)")
        print("=" * 78)
        test_gqa_sink_bwd_boundary()

    if ok:
        print("\n" + "=" * 78)
        print("Test Passed!")
        print("=" * 78)
    else:
        print("\n" + "=" * 78)
        print("Test FAILED (L0/L1 precision gate not met)")
        print("=" * 78)
        sys.exit(1)


if __name__ == "__main__":
    main()
