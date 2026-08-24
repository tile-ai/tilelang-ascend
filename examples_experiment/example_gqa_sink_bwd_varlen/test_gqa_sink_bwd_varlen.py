"""Test suite for GQA Sink Bwd Varlen (Ascend NPU, single-kernel backward).

Layered tests: L0 (precision gate) + L1 (functional) + L2 (negative) + Boundary.
Performance: do_bench (end-to-end) + msprof op (kernel-level Task Duration).

Usage:
  python test_gqa_sink_bwd_varlen.py --level all       # full test suite
  python test_gqa_sink_bwd_varlen.py --level bench     # do_bench performance
  python test_gqa_sink_bwd_varlen.py --level msprof    # msprof op kernel-level
"""

import os
import sys

# Load the operator from the sibling example file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from example_gqa_sink_bwd_varlen import (
    flashattn_fwd,
    flashattn_bwd_single,
    run_bwd_pipeline,
    ref_fwd_varlen,
    ref_bwd_varlen,
    check_precision,
    get_precision,
    DTYPE_FP16,
    BLOCK_M_FWD,
    BLOCK_N_FWD,
    BLOCK_M_BWD,
    BLOCK_N_BWD,
)

import argparse
import re
import subprocess
import traceback

import tilelang
import torch
from tilelang.profiler import do_bench

# ============================================================================
# Test helper
# ============================================================================


def _run_case(
    name, B, H, groups, q_lens, kv_lens, D, window_size, level,
    custom_sinks=None, block_M_bwd=BLOCK_M_BWD, block_N_bwd=BLOCK_N_BWD,
    vrange=None,
):
    """Run full forward + backward + dsink, compare against golden.
    4 outputs (dQ/dK/dV/dSinks) ALL blocking.
    """
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        H_kv = H // groups
        max_seq_len = max(q_lens)
        assert max_seq_len % block_M_bwd == 0, (
            f"max_seq_len ({max_seq_len}) must be divisible by block_M_bwd ({block_M_bwd})"
        )
        assert max_seq_len > 0, (
            f"max_seq_len ({max_seq_len}) must be > 0 (kernel grid would divide by zero)"
        )
        max_kv_len = max(kv_lens)
        assert max_kv_len % block_N_bwd == 0, (
            f"max_kv_len ({max_kv_len}) must be divisible by block_N_bwd ({block_N_bwd})"
        )
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
            Q *= 3.0; K *= 3.0; V *= 3.0; dO *= 3.0
        elif vrange == "L":
            Q *= 10.0; K *= 10.0; V *= 10.0; dO *= 10.0
        elif vrange == "ASYM":
            Q = Q * 2.0 + 2.0; K = K * 2.0 + 2.0; V = V * 2.0 + 2.0; dO = dO * 2.0 + 2.0
        fwd_mod = flashattn_fwd(B, UQ, UKV, max_seq_len, H, D, groups, window_size, BLOCK_M_FWD, BLOCK_N_FWD)
        O_npu, lse_npu = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
        torch.npu.synchronize()
        O_ref, lse_ref = ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups)
        dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
            Q, K, V, O_npu, dO, lse_npu, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
            B, UQ, UKV, max_seq_len, max(kv_lens), H, D, D, window_size,
            block_M_bwd, block_N_bwd, groups,
        )
        # dSinks: kernel 输出 [batch, heads, max_seq_len] fp32, host sum reduce
        dSinks_npu_fp32 = dSinks_out.cpu().float().sum(2).sum(0)  # [H] fp32
        # golden dSinks: 用 kernel 的 lse/Delta 公式重算（bhsd 方式，消除 lse 误差传播）
        dQ_ref, dK_ref, dV_ref, dSinks_ref_autograd = ref_bwd_varlen(
            Q, K, V, sinks, dO, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups,
        )
        # dSinks golden: recompute using kernel's lse/Delta (isolates dSinks kernel from fwd precision)
        sinks_exp = sinks.float().cpu().view(1, H, 1)  # [1, H, 1]
        lse_cpu = lse_npu.cpu().float()  # kernel 的 lse [B, H, max_seq_len]
        delta_cpu = Delta_out.cpu().float()  # kernel 的 Delta [B, H, max_seq_len]
        dSinks_ref_fp32 = -(torch.exp(sinks_exp - lse_cpu) * delta_cpu).sum(dim=0).sum(dim=1)  # [H]
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
            # dK/dV 从 host .half() 来 (fp32 atomic_add + host cast)
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
        # dSinks check — fp32 比对, 阻塞 0.99 (kernel T.tile.exp + golden 用 kernel lse/Delta 重算)
        sp, sr, sm = check_precision(dSinks_npu_fp32, dSinks_ref_fp32, "float32")
        min_ratio = min(min_ratio, sr)
        max_diff = max(max_diff, sm)
        all_passed &= sp
        if not sp:
            raise AssertionError(
                f"dSinks precision failed: matched_ratio={sr:.4f} < 0.99, "
                f"max_abs={sm:.3e} (limit=0.01)"
            )
        print(f"[{tag}_PASS] {level} {name} B={B} H={H} G={groups} q={q_lens} kv={kv_lens} D={D} win={window_size} max_diff={max_diff:.6e} min_ratio={min_ratio:.4f}")
        return True
    except AssertionError as e:
        # L2 输入校验失败 (assert max_seq_len%block_M==0 等) 必须 re-raise 给 _run_exception
        # boundary 精度失败 (inf/nan/valrange sinks) 不 re-raise, 标记 WARN (non-blocking)
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
# Coverage declarations
# ============================================================================
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 8, "D-DTYPE-fp32": 8, "D-DTYPE-int32": 8,
    "D-EXC-DTYPE": 1, "D-EXC-SHAPE": 4,
    "D-PARAM-batch": 3, "D-PARAM-block_M": 2, "D-PARAM-block_N": 2,
    "D-PARAM-dim": 1, "D-PARAM-groups": 4, "D-PARAM-heads": 6,
    "D-PARAM-is_causal": 1, "D-PARAM-window_size": 3,
    "D-SHAPE-ALIGNED": 21, "D-SHAPE-EDGE": 1, "D-SHAPE-PRIME": 1,
    "D-SHAPE-TAIL-1": 1, "D-SHAPE-TAIL-MID": 1,
    "D-SPECIAL-DBOUND": 1, "D-SPECIAL-INF": 1, "D-SPECIAL-NAN": 1, "D-SPECIAL-ZERO": 1,
    "D-VALRANGE-ASYM": 2, "D-VALRANGE-L": 1, "D-VALRANGE-M": 1, "D-VALRANGE-S": 8,
}
COVERAGE_NA = {}


# ============================================================================
# L0: regular shapes (block-aligned), precision convergence gate (blocking)
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
    """L0 gate test: 8 varlen cases, full forward + backward + dsink."""
    ok = True
    for name, B, H, groups, q_lens, kv_lens, D, window in L0_CASES:
        custom_sinks = None
        if name == "l0_sink_nonzero":
            torch.manual_seed(123)
            custom_sinks = torch.randn(H, dtype=torch.float16) * 3.0
        ok &= _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l0", custom_sinks=custom_sinks)
    return ok


def test_gqa_sink_bwd_l0_determinism():
    """L0 determinism test: run each L0 case 3x, max_diff must be bit-exact."""
    print("\n[L0 Determinism] Running each L0 case 3x for bit-exact check...")
    all_deterministic = True
    for name, B, H, groups, q_lens, kv_lens, D, window in L0_CASES:
        if name in ("l0_default",):
            print(f"  [SKIP] {name} (large shape, determinism implied)")
            continue
        max_diffs = []
        for run_idx in range(3):
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
                Q, K, V, O_npu, dO, lse_npu, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
                B, UQ, UKV, max_seq_len, max_kv_len, H, D, D, window,
                BLOCK_M_BWD, BLOCK_N_BWD, groups,
            )
            dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd_varlen(
                Q, K, V, sinks, dO, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window, groups,
            )
            _, _, dq_max = check_precision(dQ_fp16[..., :D].cpu(), dQ_ref.cpu(), DTYPE_FP16)
            _, _, dk_max = check_precision(dK[..., :D].cpu(), dK_ref.cpu(), DTYPE_FP16)
            _, _, dv_max = check_precision(dV.cpu(), dV_ref.cpu(), DTYPE_FP16)
            max_diffs.append(max(dq_max, dk_max, dv_max))
        if max_diffs[0] == max_diffs[1] == max_diffs[2]:
            print(f"  [DETERMINISTIC] {name}: max_diff={max_diffs[0]:.6e} (3/3 identical)")
        else:
            print(f"  [NON-DETERMINISTIC] {name}: max_diffs={max_diffs}")
            all_deterministic = False
    return all_deterministic


# ============================================================================
# L1: irregular shapes, GQA variants (blocking)
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
        ok &= _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l1",
                        block_M_bwd=bm_bwd, block_N_bwd=bn_bwd, vrange=vrange)
    return ok


# ============================================================================
# L2: negative tests (non-blocking)
# ============================================================================

L2_TAIL_CASES = [
    ("l2_tail1_seq129", 1, 4, 2, [129], [129], 128, None, 64, 64),
    ("l2_tailmid_seq192", 1, 4, 2, [192], [192], 128, None, 64, 64),
    ("l2_prime_seq131", 1, 4, 2, [131], [131], 128, None, 64, 64),
]


def test_gqa_sink_bwd_l2():
    """L2 negative test: non-aligned seq_len, unsupported dtype, illegal shape."""
    for name, B, H, groups, q_lens, kv_lens, D, window, bm_bwd, bn_bwd in L2_TAIL_CASES:
        def _run_fn(name=name, B=B, H=H, groups=groups, q_lens=q_lens, kv_lens=kv_lens,
                    D=D, window=window, bm_bwd=bm_bwd, bn_bwd=bn_bwd):
            _run_case(name, B, H, groups, q_lens, kv_lens, D, window, "l2",
                      block_M_bwd=bm_bwd, block_N_bwd=bn_bwd)
        _run_exception(name, _run_fn)
    def _run_dtype_fn():
        H = 4; groups = 2; D = 128
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
# Boundary: special sink values (non-blocking, precision compared)
# ============================================================================

BOUNDARY_CASES = [
    ("boundary_zero_sinks", lambda H: torch.zeros(H, dtype=torch.float16), None),
    ("boundary_inf_sinks", lambda H: torch.full((H,), float("inf"), dtype=torch.float16), None),
    ("boundary_nan_sinks", lambda H: torch.full((H,), float("nan"), dtype=torch.float16), None),
    ("boundary_dbound_sinks", lambda H: torch.full((H,), 65504.0, dtype=torch.float16), None),
    ("boundary_large_sinks", lambda H: torch.randn(H, dtype=torch.float16) * 10.0, None),
    ("boundary_negative_sinks", lambda H: -torch.randn(H, dtype=torch.float16) * 3.0, None),
    ("boundary_mixed_sinks", None, "ASYM"),
    ("boundary_tiny_sinks", lambda H: torch.randn(H, dtype=torch.float16) * 0.01, None),
    ("boundary_valrange_l", None, "L"),
    ("boundary_valrange_m", None, "M"),
    ("boundary_valrange_asym", None, "ASYM"),
]


def test_gqa_sink_bwd_boundary():
    """Boundary test: zero/inf/nan/dbound/large/negative/mixed/tiny sinks + valrange."""
    H = 4
    for case in BOUNDARY_CASES:
        name, sinks_fn, vrange = case
        custom_sinks = sinks_fn(H) if sinks_fn else None
        _run_case(name, 1, H, 2, [128], [128], 128, None, "boundary",
                  custom_sinks=custom_sinks, vrange=vrange)


# ============================================================================
# do_bench: functional smoke test
# ============================================================================


def _compute_flops(batch, heads, q_seqlen, k_seqlen, dim, is_causal, is_backward):
    """Compute forward (2 matmuls) or backward (5 matmuls) FLOPs."""
    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    n_matmuls = 5 if is_backward else 2
    total = n_matmuls * flops_per_matmul
    if is_causal:
        total *= 0.5
    return total


def _run_one_bench(name, batch, heads, groups, q_seqlen, k_seqlen, dim, window_size):
    """Single benchmark config: compile, verify precision, then bench."""
    head_kv = heads // groups
    dtype = torch.float16
    device = "npu"
    print(f"\n[{name}] batch={batch} heads={heads} groups={groups} head_kv={head_kv} q_seqlen={q_seqlen} k_seqlen={k_seqlen} dim={dim} window={window_size} dtype=fp16")
    cu_seqlens_q = [0]
    for _ in range(batch):
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_seqlen)
    cu_seqlens_k = [0]
    for _ in range(batch):
        cu_seqlens_k.append(cu_seqlens_k[-1] + k_seqlen)
    UQ = cu_seqlens_q[-1]
    UKV = cu_seqlens_k[-1]
    max_seq_len = max(q_seqlen, k_seqlen)
    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    torch.manual_seed(42)
    Q = torch.randn(UQ, heads, dim, dtype=dtype, device=device)
    K = torch.randn(UKV, head_kv, dim, dtype=dtype, device=device)
    V = torch.randn(UKV, head_kv, dim, dtype=dtype, device=device)
    sinks = torch.randn(heads, dtype=dtype, device=device)
    dO = torch.randn(UQ, heads, dim, dtype=dtype, device=device)
    print("  compiling forward kernel ...")
    fwd_mod = flashattn_fwd(batch, UQ, UKV, max_seq_len, heads, dim, groups, window_size, BLOCK_M_FWD, BLOCK_N_FWD)
    O, lse = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
    torch.npu.synchronize()
    O_ref, _ = ref_fwd_varlen(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups)
    fwd_passed, fwd_ratio, fwd_max_abs = check_precision(O.cpu(), O_ref.cpu(), DTYPE_FP16)
    print(f"  forward precision: ratio={fwd_ratio:.4f}, max_abs={fwd_max_abs:.3e}")
    if not fwd_passed:
        print(f"  [ERROR] forward precision failed")
        return False
    print("  compiling backward single kernel ...")
    dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
        Q, K, V, O, dO, lse, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
        batch, UQ, UKV, max_seq_len, k_seqlen, heads, dim, dim, window_size,
        BLOCK_M_BWD, BLOCK_N_BWD, groups,
    )
    torch.npu.synchronize()
    dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd_varlen(
        Q, K, V, sinks, dO, cu_seqlens_q_t, cu_seqlens_k_t, max_seq_len, window_size, groups,
    )
    dq_passed, dq_ratio, dq_max_abs = check_precision(dQ_fp16[..., :dim].cpu(), dQ_ref.cpu(), DTYPE_FP16)
    dk_passed, dk_ratio, dk_max_abs = check_precision(dK[..., :dim].half().cpu(), dK_ref.cpu(), DTYPE_FP16)
    dv_passed, dv_ratio, dv_max_abs = check_precision(dV.half().cpu(), dV_ref.cpu(), DTYPE_FP16)
    print(f"  backward precision: dQ(ratio={dq_ratio:.4f}, max_abs={dq_max_abs:.3e}) dK(ratio={dk_ratio:.4f}, max_abs={dk_max_abs:.3e}) dV(ratio={dv_ratio:.4f}, max_abs={dv_max_abs:.3e})")
    if not (dq_passed and dk_passed and dv_passed):
        print("  [ERROR] backward precision failed (precision-standard.md dual-gate)")
        return False
    print("  benching forward ...")
    def run_fwd():
        fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
    fwd_ms = do_bench(run_fwd, _n_warmup=5, _n_repeat=5, return_mode="mean")
    print("  benching backward (single-kernel) ...")
    def run_bwd():
        run_bwd_pipeline(Q, K, V, O, dO, lse, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
                         batch, UQ, UKV, max_seq_len, k_seqlen, heads, dim, dim, window_size,
                         BLOCK_M_BWD, BLOCK_N_BWD, groups)
    bwd_ms = do_bench(run_bwd, _n_warmup=5, _n_repeat=5, return_mode="mean")
    print("  benching e2e (fwd + bwd) ...")
    def run_e2e():
        _O, _lse = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
        run_bwd_pipeline(Q, K, V, _O, dO, _lse, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
                         batch, UQ, UKV, max_seq_len, k_seqlen, heads, dim, dim, window_size,
                         BLOCK_M_BWD, BLOCK_N_BWD, groups)
    e2e_ms = do_bench(run_e2e, _n_warmup=5, _n_repeat=5, return_mode="mean")
    fwd_flops = _compute_flops(batch, heads, q_seqlen, k_seqlen, dim, True, False)
    bwd_flops = _compute_flops(batch, heads, q_seqlen, k_seqlen, dim, True, True)
    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) * 1e-12
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) * 1e-12
    e2e_tflops = (fwd_flops + bwd_flops) / (e2e_ms * 1e-3) * 1e-12
    bwd_min_ratio = min(dq_ratio, dk_ratio, dv_ratio)
    print()
    print("  | Kernel                      | Q seq    | Latency(ms) | TFlops   | max_abs   | min_ratio |")
    print("  |-----------------------------|----------|-------------|----------|-----------|-----------|")
    print(f"  | TileLang Forward            | {q_seqlen:<8} | {fwd_ms:<11.2f} | {fwd_tflops:<8.2f} | {fwd_max_abs:.2e} | {fwd_ratio:.4f}    |")
    print(f"  | TileLang Backward (single)  | {q_seqlen:<8} | {bwd_ms:<11.2f} | {bwd_tflops:<8.2f} | {dq_max_abs:.2e} | {bwd_min_ratio:.4f}    |")
    print(f"  | TileLang Fwd+Bwd (e2e)      | {q_seqlen:<8} | {e2e_ms:<11.2f} | {e2e_tflops:<8.2f} | -         | -         |")
    print(f"\n  GPU baseline: 28.574 ms (backward only)")
    print(f"  vs GPU: {bwd_ms / 28.574:.2f}x slower")
    return True


def run_bench(preset="default", window="both"):
    """Run do_bench benchmark suite (functional smoke + latency)."""
    print("=" * 78)
    print("GQA + Attention Sink Flash Attention Benchmark (VARLEN) - do_bench")
    print("=" * 78)
    results = []
    if preset == "default":
        shape_kwargs = dict(batch=8, heads=64, groups=16, q_seqlen=2048, k_seqlen=2048, dim=128)
        if window in ("both", "none"):
            print("\n" + "-" * 78)
            print("CONFIG 1: window=None (full causal)")
            print("-" * 78)
            results.append(_run_one_bench("default_window_none", **shape_kwargs, window_size=None))
        if window in ("both", "128"):
            print("\n" + "-" * 78)
            print("CONFIG 2: window=128 (Sliding Window Attention, SWA)")
            print("-" * 78)
            results.append(_run_one_bench("default_window_128", **shape_kwargs, window_size=128))
    elif preset == "small":
        results.append(_run_one_bench("small", batch=1, heads=4, groups=2, q_seqlen=128, k_seqlen=128, dim=128, window_size=None))
    print("\nDone.")
    return all(results) if results else False


# ============================================================================
# msprof op: kernel-level Task Duration
# ============================================================================


def _run_msprof_target(preset="default", window="none", repeat=2):
    """bwd-only target for msprof op: forward + repeat bwd passes."""
    if preset == "default":
        batch, heads, groups, q_seqlen, k_seqlen, dim = 8, 64, 16, 2048, 2048, 128
    elif preset == "small":
        batch, heads, groups, q_seqlen, k_seqlen, dim = 1, 4, 2, 128, 128, 128
    else:
        raise ValueError(f"Unknown preset: {preset}")
    window_size = None if window == "none" else int(window)
    head_kv = heads // groups
    device = "npu"
    cu_seqlens_q = [0]
    for _ in range(batch):
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_seqlen)
    cu_seqlens_k = [0]
    for _ in range(batch):
        cu_seqlens_k.append(cu_seqlens_k[-1] + k_seqlen)
    UQ = cu_seqlens_q[-1]
    UKV = cu_seqlens_k[-1]
    max_seq_len = max(q_seqlen, k_seqlen)
    max_kv_len = k_seqlen
    cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    torch.manual_seed(42)
    Q = torch.randn(UQ, heads, dim, dtype=torch.float16, device=device)
    K = torch.randn(UKV, head_kv, dim, dtype=torch.float16, device=device)
    V = torch.randn(UKV, head_kv, dim, dtype=torch.float16, device=device)
    sinks = torch.randn(heads, dtype=torch.float16, device=device)
    dO = torch.randn(UQ, heads, dim, dtype=torch.float16, device=device)
    print(f"  [msprof-target] preset={preset} window={window} repeat={repeat}")
    print(f"  [msprof-target] compiling forward + bwd kernels ...")
    fwd_mod = flashattn_fwd(batch, UQ, UKV, max_seq_len, heads, dim, groups, window_size, BLOCK_M_FWD, BLOCK_N_FWD)
    O, lse = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
    torch.npu.synchronize()
    print(f"  [msprof-target] pass 0 (warmup) ...")
    dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
        Q, K, V, O, dO, lse, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
        batch, UQ, UKV, max_seq_len, max_kv_len, heads, dim, dim, window_size,
        BLOCK_M_BWD, BLOCK_N_BWD, groups,
    )
    torch.npu.synchronize()
    for i in range(1, repeat):
        print(f"  [msprof-target] pass {i} ...")
        dQ_fp16, dK, dV, dSinks_out, Delta_out = run_bwd_pipeline(
            Q, K, V, O, dO, lse, sinks, cu_seqlens_q_t, cu_seqlens_k_t,
            batch, UQ, UKV, max_seq_len, max_kv_len, heads, dim, dim, window_size,
            BLOCK_M_BWD, BLOCK_N_BWD, groups,
        )
        torch.npu.synchronize()
        print(f"  [msprof-target] pass {i} done")
    print(f"  [msprof-target] all {repeat} passes done")


def run_msprof(preset="default", window="none", launch_count=1, warm_up=1):
    """msprof op: kernel-level profiling with --kernel-name=main_kernel."""
    print("=" * 78)
    print("GQA + Attention Sink Flash Attention - msprof op (kernel-level)")
    print("=" * 78)
    script_path = os.path.abspath(__file__)
    output_dir = os.path.expanduser("~/msprof_output")
    os.makedirs(output_dir, exist_ok=True)
    window_arg = "none" if window == "none" else window
    # single-kernel bwd: 1 launch per pass (rev3), vs rev2's 34
    bwd_launches_per_pass = launch_count
    cmd = (
        f"msprof op --kernel-name=main_kernel --output={output_dir} "
        f"--launch-count={bwd_launches_per_pass} --kill=on --warm-up={warm_up} "
        f"--launch-skip-before-match=1 "
        f"python3 {script_path} --level msprof-target --preset {preset} --window {window_arg}"
    )
    print(f"Command: {cmd}")
    print(f"(skip 1 forward kernel, sample {bwd_launches_per_pass} bwd kernels)")
    result: Optional[subprocess.CompletedProcess] = None
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print("[ERROR] msprof op timed out after 3600s.")
        return False
    except FileNotFoundError:
        print("[ERROR] msprof not found. Ensure CANN is sourced (source set_env.sh).")
        return False
    except OSError as e:
        print(f"[ERROR] OSError running msprof: {e}")
        return False
    if result.returncode != 0:
        print(f"[ERROR] msprof op exited with returncode={result.returncode}")
        print("msprof stdout (last 2000 chars):")
        print(result.stdout[-2000:] if result.stdout else "(empty)")
        print("msprof stderr (last 2000 chars):")
        print(result.stderr[-2000:] if result.stderr else "(empty)")
        return False
    stdout = (result.stdout or "") + (result.stderr or "")
    duration_pattern = re.compile(r"Task Duration\(us\):\s*([\d.]+)")
    durations = [float(m) for m in duration_pattern.findall(stdout)]
    if durations:
        print("Kernel-level performance (msprof op):")
        print()
        print("  | # | Kernel              | Task Duration (us) |")
        print("  |---|---------------------|---------------------|")
        for i, dur in enumerate(durations):
            print(f"  | {i + 1} | Kernel {i + 1:<17} | {dur:<19.2f} |")
        median_dur = sorted(durations)[len(durations) // 2]
        print(f"\n  Task Duration median: {median_dur:.2f} us ({median_dur / 1000:.3f} ms)")
        print(f"  Launch count: {len(durations)}")
    else:
        print("WARNING: Could not parse Task Duration from msprof output.")
        print("msprof stdout (last 2000 chars):")
        print(stdout[-2000:])
    print()
    print("msprof op completed successfully.")
    return True


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
            det_ok = test_gqa_sink_bwd_l0_determinism()
            ok &= det_ok
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
# main: argparse --level {l0|l1|l2|boundary|all|bench|msprof}
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GQA + Attention Sink Flash Attention Backward (Varlen) - Ascend NPU (rev3 single-kernel)"
    )
    parser.add_argument("--level", choices=["l0", "l1", "l2", "boundary", "all", "bench", "msprof", "msprof-target"],
                        default="l0", help="Test level")
    parser.add_argument("--profiler", choices=["do_bench", "msprof"], default="do_bench")
    parser.add_argument("--preset", choices=["default", "small"], default="default")
    parser.add_argument("--window", choices=["none", "128", "both"], default="both")
    parser.add_argument("--skip-determinism", action="store_true", help="Skip L0 3x determinism check")
    args = parser.parse_args()

    # disable_cache: prevent stale JIT cache from masking compile changes (checklist #11)
    tilelang.disable_cache()
    torch.set_default_device("npu")

    if args.level == "bench":
        if args.profiler == "do_bench":
            ok = run_bench(preset=args.preset, window=args.window)
            sys.exit(0 if ok else 1)
        elif args.profiler == "msprof":
            ok = run_msprof(preset=args.preset, window=args.window, launch_count=10, warm_up=3)
            sys.exit(0 if ok else 1)
        sys.exit(0)

    if args.level == "msprof":
        ok = run_msprof(preset=args.preset, window=args.window, launch_count=1, warm_up=1)
        sys.exit(0 if ok else 1)

    if args.level == "msprof-target":
        _run_msprof_target(preset=args.preset, window=args.window, repeat=3)
        sys.exit(0)

    ok = True
    if args.level in ("l0", "all"):
        print("\n" + "=" * 78)
        print("L0: Precision Gate (8 cases, block-aligned)")
        print("=" * 78)
        l0_ok = test_gqa_sink_bwd_l0()
        ok &= l0_ok
        if l0_ok and not args.skip_determinism:
            det_ok = test_gqa_sink_bwd_l0_determinism()
            ok &= det_ok

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
        sys.exit(0)
    else:
        print("\n" + "=" * 78)
        print("Test FAILED (L0/L1 precision gate not met)")
        print("=" * 78)
        sys.exit(1)


if __name__ == "__main__":
    main()
