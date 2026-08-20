"""Test suite for GQA + Attention Sink Flash Attention Backward (Varlen).

Pytest-discoverable test file. Loads the example module via importlib (following
examples/flash_attention/test_flash_attn_bhsd.py convention) and provides:

  - L0: precision gate (8 cases + 3x determinism)
  - L1: functional (irregular shapes, GQA variants)
  - L2: negative tests (unsupported input rejection, non-blocking)
  - Boundary: special sink values (non-blocking)
  - do_bench: latency benchmark (forward + backward + e2e)
  - msprof op: kernel-level Task Duration profiling

Usage:
  pytest test_gqa_sink_bwd_varlen.py -v          # pytest (L0/L1 gate tests)
  python test_gqa_sink_bwd_varlen.py --level all  # full suite standalone
  python test_gqa_sink_bwd_varlen.py --level bench # do_bench
  python test_gqa_sink_bwd_varlen.py --level msprof # msprof op
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType


def _load_example() -> ModuleType:
    """Load example_gqa_sink_bwd_varlen.py as a module (hide pytest args)."""
    source = Path(__file__).with_name("example_gqa_sink_bwd_varlen.py")
    spec = importlib.util.spec_from_file_location("_gqa_sink_bwd_varlen_example", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


# Load example module and bind symbols into this module's namespace.
_M = _load_example()

import tilelang
import torch
from tilelang.profiler import do_bench

# Re-export everything the test/bench/msprof functions need.
flashattn_fwd = _M.flashattn_fwd
run_bwd_pipeline = _M.run_bwd_pipeline
ref_fwd_varlen = _M.ref_fwd_varlen
ref_bwd_varlen = _M.ref_bwd_varlen
get_precision = _M.get_precision
check_precision = _M.check_precision
BLOCK_M_FWD = _M.BLOCK_M_FWD
BLOCK_N_FWD = _M.BLOCK_N_FWD
BLOCK_M_BWD = _M.BLOCK_M_BWD
BLOCK_N_BWD = _M.BLOCK_N_BWD
DTYPE_FP16 = _M.DTYPE_FP16

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
    """Run full forward + backward + dsink, compare against golden.

    Precision gate: precision-standard.md dual-gate (matched_ratio >= 0.99 AND max_abs <= 0.1).
    check_precision returns passed=False -> raise AssertionError -> [PRECISION_FAIL].
    """
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

        dQ_fp16, dK, dV, dSinks_out = run_bwd_pipeline(
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

        dSinks_npu = torch.zeros(H, dtype=torch.float16, device="cpu")
        for b in range(B):
            q_len = q_lens[b]
            dSinks_npu += dSinks_out[b, :, :q_len].cpu().float().sum(1).half()

        dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd_varlen(
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
                raise AssertionError(f"O precision failed (batch {b}): matched_ratio={fr:.4f} < 0.99, max_abs={fm:.3e} (limit=0.1)")

        qp, qr, qm = check_precision(dQ_fp16[..., :D].cpu(), dQ_ref.cpu(), DTYPE_FP16)
        min_ratio = min(min_ratio, qr)
        max_diff = max(max_diff, qm)
        all_passed &= qp
        if not qp:
            raise AssertionError(f"dQ precision failed: matched_ratio={qr:.4f} < 0.99, max_abs={qm:.3e} (limit=0.1)")

        for b in range(B):
            ks = cu_seqlens_k[b]
            ke = cu_seqlens_k[b + 1]
            kp, kr, km = check_precision(dK[ks:ke, :, :D].half().cpu(), dK_ref[ks:ke].cpu(), DTYPE_FP16)
            min_ratio = min(min_ratio, kr)
            max_diff = max(max_diff, km)
            all_passed &= kp
            if not kp:
                raise AssertionError(f"dK precision failed (batch {b}): matched_ratio={kr:.4f} < 0.99, max_abs={km:.3e} (limit=0.1)")
            vp, vr, vm = check_precision(dV[ks:ke].half().cpu(), dV_ref[ks:ke].cpu(), DTYPE_FP16)
            min_ratio = min(min_ratio, vr)
            max_diff = max(max_diff, vm)
            all_passed &= vp
            if not vp:
                raise AssertionError(f"dV precision failed (batch {b}): matched_ratio={vr:.4f} < 0.99, max_abs={vm:.3e} (limit=0.1)")

        sp, sr, sm = check_precision(dSinks_npu.cpu(), dSinks_ref.cpu(), DTYPE_FP16)
        min_ratio = min(min_ratio, sr)
        max_diff = max(max_diff, sm)

        print(
            f"[{tag}_PASS] {level} {name} B={B} H={H} G={groups} "
            f"q={q_lens} kv={kv_lens} D={D} win={window_size} "
            f"max_diff={max_diff:.6e} min_ratio={min_ratio:.4f}"
        )
        return True
    except Exception as e:
        fail_tag = "WARN" if tag == "BOUNDARY" else "FAIL"
        print(f"[{tag}_{fail_tag}] {level} {name} B={B} H={H} G={groups} q={q_lens} kv={kv_lens} D={D} win={window_size}: {e}")
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


def _run_exception(name, fn):
    """L2 negative test: fn() passes unsupported input, expects exception.

    Raises -> [BOUNDARY_PASS] (correctly rejected)
    No exception -> [BOUNDARY_WARN] (should reject but didn't)
    Both non-blocking.
    """
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: unsupported input not rejected (silent accept)")


# ============================================================================
# Coverage declarations (for coverage_check.py --proto proto.yaml)
# ============================================================================
COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    # dtype coverage (from proto.yaml inputs/outputs)
    "D-DTYPE-fp16": 8,  # Q/K/V/O/dO/Sinks all fp16; L0 cases
    "D-DTYPE-fp32": 8,  # lse fp32; ws_s/ws_p/ws_dp/ws_ds fp32
    "D-DTYPE-int32": 8,  # cu_seqlens_q/k int32
    # exception/negative tests
    "D-EXC-DTYPE": 1,  # l2_unsupported_dtype_fp64
    "D-EXC-SHAPE": 4,  # l2_tail1_seq129, l2_tailmid_seq192, l2_prime_seq131, l2_illegal_shape_empty
    # param coverage (from proto.yaml attrs)
    "D-PARAM-batch": 3,  # B=1,2,4
    "D-PARAM-block_M": 2,  # 64, 128
    "D-PARAM-block_N": 2,  # 64, 128
    "D-PARAM-dim": 1,  # D=128
    "D-PARAM-groups": 4,  # G=1,2,4,8
    "D-PARAM-heads": 6,  # H=1,4,8,16,32,64
    "D-PARAM-is_causal": 1,  # always true (varlen causal)
    "D-PARAM-window_size": 3,  # None, 128, 256
    # shape coverage
    "D-SHAPE-ALIGNED": 21,  # all L0+L1 cases (block-aligned)
    "D-SHAPE-EDGE": 1,  # l1_edge_min
    "D-SHAPE-PRIME": 1,  # l2_prime_seq131
    "D-SHAPE-TAIL-1": 1,  # l2_tail1_seq129
    "D-SHAPE-TAIL-MID": 1,  # l2_tailmid_seq192
    # special values
    "D-SPECIAL-DBOUND": 1,  # boundary_dbound_sinks
    "D-SPECIAL-INF": 1,  # boundary_inf_sinks
    "D-SPECIAL-NAN": 1,  # boundary_nan_sinks
    "D-SPECIAL-ZERO": 1,  # boundary_zero_sinks
    # value ranges
    "D-VALRANGE-ASYM": 2,  # boundary_mixed_sinks, boundary_valrange_asym
    "D-VALRANGE-L": 1,  # boundary_valrange_l
    "D-VALRANGE-M": 1,  # boundary_valrange_m
    "D-VALRANGE-S": 8,  # default range (no scaling), all L0 cases
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
            # Generate custom_sinks with a deterministic seed (separate from the
            # test case's manual_seed(42) inside _run_case) to ensure reproducibility
            # across runs and avoid seed-ordering sensitivity.
            torch.manual_seed(123)
            custom_sinks = torch.randn(H, dtype=torch.float16) * 3.0
        ok &= _run_case(
            name,
            B,
            H,
            groups,
            q_lens,
            kv_lens,
            D,
            window,
            "l0",
            custom_sinks=custom_sinks,
        )
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

            # Use SAME seed for all 3 runs (true determinism: same input → same output).
            # Previous version used 42+run_idx (different inputs), which made max_diffs
            # naturally differ — that tested input sensitivity, not kernel determinism.
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

            dQ_fp16, dK, dV, dSinks_out = run_bwd_pipeline(
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
            _, _, dk_max = check_precision(dK[..., :D].half().cpu(), dK_ref.cpu(), DTYPE_FP16)
            _, _, dv_max = check_precision(dV.half().cpu(), dV_ref.cpu(), DTYPE_FP16)
            # Track max across all outputs (dQ, dK, dV) for determinism check
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
    ("l1_basic_small", 1, 4, 2, [128], [128], 128, None, 128, 128, None),
    ("l1_mha_causal", 1, 4, 1, [256], [256], 128, None, 128, 128, None),
    ("l1_gqa_medium", 4, 16, 4, [256] * 4, [256] * 4, 128, None, 128, 128, None),
    ("l1_asymmetric_sq_gt_skv", 2, 4, 2, [256, 256], [128, 128], 128, None, 128, 128, None),
    ("l1_asymmetric_skv_gt_sq", 2, 4, 2, [128, 128], [256, 256], 128, None, 128, 128, None),
    ("l1_window_128", 1, 16, 8, [256], [256], 128, 128, 128, 128, None),
    ("l1_window_256", 1, 16, 8, [512], [512], 128, 256, 128, 128, None),
    ("l1_varlen_unequal_batches", 2, 4, 2, [128, 256], [256, 128], 128, None, 128, 128, None),
    ("l1_irregular_n_384", 1, 8, 4, [384], [384], 128, None, 128, 128, None),
    ("l1_large_causal", 1, 32, 8, [512], [512], 128, None, 128, 128, None),
    ("l1_block_m_64", 1, 4, 2, [256], [256], 128, None, 64, 128, None),
    ("l1_block_n_64", 1, 4, 2, [128], [128], 128, None, 128, 64, None),
    ("l1_edge_min", 1, 4, 2, [128], [128], 128, None, 128, 128, None),
]


def test_gqa_sink_bwd_l1():
    """L1 functional test: irregular shapes, different groups/block sizes (blocking)."""
    ok = True
    for case in L1_CASES:
        name, B, H, groups, q_lens, kv_lens, D, window = case[:8]
        bm_bwd, bn_bwd = case[8], case[9]
        vrange = case[10]
        ok &= _run_case(
            name,
            B,
            H,
            groups,
            q_lens,
            kv_lens,
            D,
            window,
            "l1",
            block_M_bwd=bm_bwd,
            block_N_bwd=bn_bwd,
            vrange=vrange,
        )
    return ok


# ============================================================================
# L2: negative tests (non-blocking) — unsupported input should be rejected
# ============================================================================

L2_TAIL_CASES = [
    ("l2_tail1_seq129", 1, 4, 2, [129], [129], 128, None, 128, 128),
    ("l2_tailmid_seq192", 1, 4, 2, [192], [192], 128, None, 128, 128),
    ("l2_prime_seq131", 1, 4, 2, [131], [131], 128, None, 128, 128),
]


def test_gqa_sink_bwd_l2():
    """L2 negative test: non-aligned seq_len, unsupported dtype, illegal shape."""
    for name, B, H, groups, q_lens, kv_lens, D, window, bm_bwd, bn_bwd in L2_TAIL_CASES:

        def _run_fn(
            name=name,
            B=B,
            H=H,
            groups=groups,
            q_lens=q_lens,
            kv_lens=kv_lens,
            D=D,
            window=window,
            bm_bwd=bm_bwd,
            bn_bwd=bn_bwd,
        ):
            _run_case(
                name,
                B,
                H,
                groups,
                q_lens,
                kv_lens,
                D,
                window,
                "l2",
                block_M_bwd=bm_bwd,
                block_N_bwd=bn_bwd,
            )

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
        _run_case(
            name,
            1,
            H,
            2,
            [128],
            [128],
            128,
            None,
            "boundary",
            custom_sinks=custom_sinks,
            vrange=vrange,
        )


# ============================================================================
# do_bench: functional smoke test (includes host overhead)
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

    print(
        f"\n[{name}] batch={batch} heads={heads} groups={groups} head_kv={head_kv} "
        f"q_seqlen={q_seqlen} k_seqlen={k_seqlen} dim={dim} "
        f"window={window_size} dtype=fp16"
    )

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
        print(f"  [ERROR] forward precision failed: matched_ratio={fwd_ratio:.4f} < 0.99 or max_abs={fwd_max_abs:.3e} > 0.1")
        return False

    print("  compiling backward pipeline (11-kernel split) ...")
    dQ_fp16, dK, dV, dSinks_out = run_bwd_pipeline(
        Q,
        K,
        V,
        O,
        dO,
        lse,
        sinks,
        cu_seqlens_q_t,
        cu_seqlens_k_t,
        batch,
        UQ,
        UKV,
        max_seq_len,
        k_seqlen,
        heads,
        dim,
        dim,
        window_size,
        BLOCK_M_BWD,
        BLOCK_N_BWD,
        groups,
    )
    torch.npu.synchronize()

    dQ_ref, dK_ref, dV_ref, dSinks_ref = ref_bwd_varlen(
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
    dq_passed, dq_ratio, dq_max_abs = check_precision(dQ_fp16[..., :dim].cpu(), dQ_ref.cpu(), DTYPE_FP16)
    dk_passed, dk_ratio, dk_max_abs = check_precision(dK[..., :dim].half().cpu(), dK_ref.cpu(), DTYPE_FP16)
    dv_passed, dv_ratio, dv_max_abs = check_precision(dV.half().cpu(), dV_ref.cpu(), DTYPE_FP16)
    print(
        f"  backward precision: "
        f"dQ(ratio={dq_ratio:.4f}, max_abs={dq_max_abs:.3e}) "
        f"dK(ratio={dk_ratio:.4f}, max_abs={dk_max_abs:.3e}) "
        f"dV(ratio={dv_ratio:.4f}, max_abs={dv_max_abs:.3e})"
    )
    if not (dq_passed and dk_passed and dv_passed):
        print("  [ERROR] backward precision failed (precision-standard.md dual-gate)")
        return False

    print("  benching forward ...")

    def run_fwd():
        fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)

    fwd_ms = do_bench(run_fwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    print("  benching backward (11-kernel split) ...")

    def run_bwd():
        run_bwd_pipeline(
            Q,
            K,
            V,
            O,
            dO,
            lse,
            sinks,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            batch,
            UQ,
            UKV,
            max_seq_len,
            k_seqlen,
            heads,
            dim,
            dim,
            window_size,
            BLOCK_M_BWD,
            BLOCK_N_BWD,
            groups,
        )

    bwd_ms = do_bench(run_bwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    print("  benching e2e (fwd + bwd) ...")

    def run_e2e():
        _O, _lse = fwd_mod(Q, K, V, sinks, cu_seqlens_q_t, cu_seqlens_k_t)
        run_bwd_pipeline(
            Q,
            K,
            V,
            _O,
            dO,
            _lse,
            sinks,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            batch,
            UQ,
            UKV,
            max_seq_len,
            k_seqlen,
            heads,
            dim,
            dim,
            window_size,
            BLOCK_M_BWD,
            BLOCK_N_BWD,
            groups,
        )

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
    print(
        f"  | TileLang Forward            | {q_seqlen:<8} | {fwd_ms:<11.2f} | {fwd_tflops:<8.2f} | {fwd_max_abs:.2e} | {fwd_ratio:.4f}    |"
    )
    print(
        f"  | TileLang Backward (11-kernel) | {q_seqlen:<8} | {bwd_ms:<11.2f} | {bwd_tflops:<8.2f} | {dq_max_abs:.2e} | {bwd_min_ratio:.4f}    |"
    )
    print(f"  | TileLang Fwd+Bwd (e2e)      | {q_seqlen:<8} | {e2e_ms:<11.2f} | {e2e_tflops:<8.2f} | -         | -         |")
    print("\n  GPU baseline: 28.574 ms (backward only)")
    print(f"  vs GPU: {bwd_ms / 28.574:.2f}x slower")

    return True


def run_bench(preset="default", window="both"):
    """Run do_bench benchmark suite (functional smoke + latency)."""
    print("=" * 78)
    print("GQA + Attention Sink Flash Attention Benchmark (VARLEN) — do_bench")
    print("=" * 78)

    results = []

    if preset == "default":
        shape_kwargs = dict(
            batch=8,
            heads=64,
            groups=16,
            q_seqlen=2048,
            k_seqlen=2048,
            dim=128,
        )
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
        results.append(
            _run_one_bench(
                "small",
                batch=1,
                heads=4,
                groups=2,
                q_seqlen=128,
                k_seqlen=128,
                dim=128,
                window_size=None,
            )
        )

    print("\nDone.")
    return all(results) if results else False


# ============================================================================
# msprof op: kernel-level Task Duration (main performance metric)
# ============================================================================


def _run_msprof_target(preset="default", window="none", repeat=3):
    """Lightweight bwd-only target for msprof op (NO forward kernel launch).

    Designed to be invoked as the msprof subprocess target via
    `--level msprof-target`. Does NOT wrap in do_bench (which would multiply
    kernel launches by 10+ and cause msprof timeout / mixed captures).

    CRITICAL: Does NOT call flashattn_fwd (the forward kernel). Uses PyTorch
    ref_fwd_varlen (host-side) to produce O, lse for backward. This ensures
    ALL main_kernel launches in the target are backward kernels, so msprof's
    --launch-skip-before-match can cleanly skip warmup passes and capture
    only backward kernels.

    Strategy (aligned with perf_log bwd_only_prof.py wrapper):
      - Host-side forward (ref_fwd_varlen) produces O, lse — no kernel launch.
      - Run backward pipeline `repeat` times. Each backward pass =
        k0 + [k1,k2,k3c,k3,k3b,k4,k5c,k5,k5b]×16 + k6 = 146 main_kernel
        launches (default preset).
      - msprof --launch-skip-before-match skips first backward pass (warmup),
        then captures kernels from pass 2 onward.

    All captured Task Duration values are backward kernels (zero forward
    contamination, since forward kernel never runs in this target).
    """
    if preset == "default":
        batch, heads, groups = 8, 64, 16
        q_seqlen, k_seqlen, dim = 2048, 2048, 128
    else:  # small
        batch, heads, groups = 1, 4, 2
        q_seqlen, k_seqlen, dim = 128, 128, 128
    window_size = None if window == "none" else int(window)

    head_kv = heads // groups
    max_seq_len = max(q_seqlen, k_seqlen)

    cu_seqlens_q = [0]
    for _ in range(batch):
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_seqlen)
    cu_seqlens_k = [0]
    for _ in range(batch):
        cu_seqlens_k.append(cu_seqlens_k[-1] + k_seqlen)
    UQ = cu_seqlens_q[-1]
    UKV = cu_seqlens_k[-1]

    cu_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="npu")
    cu_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="npu")

    torch.manual_seed(42)
    Q = torch.randn(UQ, heads, dim, dtype=torch.float16, device="npu")
    K = torch.randn(UKV, head_kv, dim, dtype=torch.float16, device="npu")
    V = torch.randn(UKV, head_kv, dim, dtype=torch.float16, device="npu")
    sinks = torch.randn(heads, dtype=torch.float16, device="npu")
    dO = torch.randn(UQ, heads, dim, dtype=torch.float16, device="npu")

    # Host-side forward (PyTorch golden) — produces O, lse WITHOUT launching
    # any main_kernel. This is the key trick: all subsequent main_kernel
    # launches in this target are backward kernels, so msprof capture is
    # purely backward (zero forward contamination).
    O, lse = ref_fwd_varlen(
        Q,
        K,
        V,
        sinks,
        cu_q_t,
        cu_k_t,
        max_seq_len,
        window_size,
        groups,
    )
    torch.npu.synchronize()

    print(
        f"[MSPROF_TARGET] host-side forward done (O, lse ready, no kernel launch). Starting {repeat} backward-only passes...",
        flush=True,
    )

    # Backward-only passes. Zero forward kernel launches here — every
    # main_kernel call is a backward kernel (k0..k6).
    for _i in range(repeat):
        run_bwd_pipeline(
            Q,
            K,
            V,
            O,
            dO,
            lse,
            sinks,
            cu_q_t,
            cu_k_t,
            batch,
            UQ,
            UKV,
            max_seq_len,
            k_seqlen,
            heads,
            dim,
            dim,
            window_size,
            BLOCK_M_BWD,
            BLOCK_N_BWD,
            groups,
        )
        torch.npu.synchronize()

    print(f"[MSPROF_TARGET] done ({repeat} backward-only passes)", flush=True)


def run_msprof(preset="default", window="none", launch_count=10, warm_up=1, timeout_sec=3600, capture_mode="sample"):
    """msprof op: kernel-level profiling via lightweight target.

    Capture modes:
      - sample (default): launch_count=10, captures 10 consecutive bwd kernels
        across K-iterations (matches perf_log record). Use for per-kernel
        breakdown analysis.
      - full-pass: launch_count=146 (default preset) or 11 (small), captures
        ALL kernels in one complete backward pass. Reports TOTAL kernel
        runtime = sum of all Task Durations = the true kernel-only latency
        of one backward pipeline invocation.

    Strategy (aligned with perf_log bwd_only_prof.py wrapper):
      - Subprocess runs `--level msprof-target` (bwd-only, no do_bench wrapping).
      - `--launch-skip-before-match` skips compilation + forward warmup so
        msprof captures ONLY backward kernels.
      - sample mode: skip=82, captures k5b[iter8] + k1..k5b[iter9].
      - full-pass mode: skip=1 (just forward warmup), captures complete
        bwd pass = k0 + [k1..k5b]×16 + k6.

    All tilelang kernels compile to 'main_kernel', merged into one record.
    Checks result.returncode == 0 AND parses Task Duration from output.
    Does NOT claim success if the target script fails.
    """
    print("=" * 78)
    print("GQA + Attention Sink Flash Attention — msprof op (kernel-level)")
    print(f"  capture_mode: {capture_mode}")
    print("=" * 78)

    script_path = os.path.abspath(__file__)
    output_dir = os.path.expanduser("~/msprof_output")
    os.makedirs(output_dir, exist_ok=True)

    # Backward launches per pass:
    #   default: k0(1) + [k1,k2,k3c,k3,k3b,k4,k5c,k5,k5b]×16(144) + k6(1) = 146
    #   small:   k0(1) + [k1..k5b]×1(9) + k6(1) = 11
    if preset == "default":
        bwd_launches_per_pass = 146
    else:
        bwd_launches_per_pass = 11

    window_arg = "none" if window == "none" else str(window)
    # launch-skip-before-match: now that target is bwd-only (no forward kernel),
    # ALL main_kernel launches are backward kernels. skip skips warmup passes.
    #
    # sample mode (default preset): skip=82 (perf_log-verified).
    #   - Pass 0 has 146 launches (k0 + [k1..k5b]×16 + k6).
    #   - skip=82 skips k0 + 8 K-iters of k1..k5b (1 + 8*9 = 73), lands at
    #     k5b[iter8] of pass 0, then captures k5b[iter8] + k1..k5b[iter9] (10
    #     consecutive bwd kernels, matching perf_log record).
    #
    # full-pass mode: skip = bwd_launches_per_pass (skip entire pass 0 as
    #   warmup), warm_up=0, launch_count = bwd_launches_per_pass. Captures
    #   complete pass 1 = k0 + [k1..k5b]×16 + k6. TOTAL = sum = true
    #   kernel-only backward latency.
    if capture_mode == "full-pass":
        skip_before = bwd_launches_per_pass  # skip pass 0 entirely
        warm_up_eff = 0
        launch_count_eff = bwd_launches_per_pass
    else:  # sample
        if preset == "default":
            skip_before = 82
            warm_up_eff = warm_up
        else:
            # small: 11 launches/pass, skip pass 0 (11), capture pass 1's
            # first 10 bwd kernels (skip k6 of pass 1).
            skip_before = bwd_launches_per_pass  # 11
            warm_up_eff = 0
        launch_count_eff = launch_count
    cmd = (
        f"msprof op --kernel-name=main_kernel --output={output_dir} "
        f"--launch-count={launch_count_eff} --kill=on --warm-up={warm_up_eff} "
        f"--launch-skip-before-match={skip_before} "
        f"python3 {script_path} --level msprof-target --preset {preset} "
        f"--window {window_arg}"
    )

    print(f"Command: {cmd}")
    print(
        f"(timeout={timeout_sec}s; first run compiles 12 kernels ~7min, "
        f"subsequent runs reuse JIT cache; full-pass capture is ~15x slower "
        f"than sample due to per-kernel profiling overhead)"
    )
    print()

    result: subprocess.CompletedProcess | None = None
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] msprof op timed out after {timeout_sec}s.")
        print("Hint: ensure tilelang.enable_cache() is active (default in main).")
        print("       First run compiles 12 kernels (~7min); reruns reuse cache (~5min).")
        return False
    except FileNotFoundError:
        print("[ERROR] msprof not found. Ensure CANN is sourced (source set_env.sh).")
        return False
    except OSError as e:
        print(f"[ERROR] OSError running msprof: {e}")
        return False

    # CRITICAL: check returncode == 0 (msprof exit code reflects target script success)
    if result.returncode != 0:
        print(f"[ERROR] msprof op exited with returncode={result.returncode}")
        print("msprof stdout (last 2000 chars):")
        print(result.stdout[-2000:] if result.stdout else "(empty)")
        print("msprof stderr (last 2000 chars):")
        print(result.stderr[-2000:] if result.stderr else "(empty)")
        return False

    stdout = (result.stdout or "") + (result.stderr or "")

    # Parse Task Duration
    duration_pattern = re.compile(r"Task Duration\(us\):\s*([\d.]+)")
    block_dim_pattern = re.compile(r"Block Dim:\s*(\d+)")
    mix_block_dim_pattern = re.compile(r"Mix Block Dim:\s*(\d+)")

    durations = [float(m) for m in duration_pattern.findall(stdout)]
    block_dims = block_dim_pattern.findall(stdout)
    mix_block_dims = mix_block_dim_pattern.findall(stdout)

    if durations:
        # Backward kernel name mapping (default preset, 16 K-iterations):
        # launch order per pass: k0, [k1,k2,k3c,k3,k3b,k4,k5c,k5,k5b]×16, k6
        # Block Dim signature: mix kernels have Block Dim=8192 (Cube+Vector),
        #                       vector kernels have small Block Dim (4/8).
        # We tag captured kernels by their position in the bwd sequence.
        if capture_mode == "full-pass":
            # Build kernel labels for a full backward pass.
            if preset == "default":
                labels = ["k0 (preprocess)"]
                for i in range(16):
                    labels += [
                        f"k1 (Q@K^T) iter{i}",
                        f"k2 (softmax) iter{i}",
                        f"k3c (p_delta) iter{i}",
                        f"k3 (dV+dP) iter{i}",
                        f"k3b (dV corr) iter{i}",
                        f"k4 (dS) iter{i}",
                        f"k5c (ds_delta) iter{i}",
                        f"k5 (dK+dQ) iter{i}",
                        f"k5b (dK corr) iter{i}",
                    ]
                labels.append("k6 (postprocess)")
            else:  # small (1 K-iteration)
                labels = [
                    "k0 (preprocess)",
                    "k1 (Q@K^T)",
                    "k2 (softmax)",
                    "k3c (p_delta)",
                    "k3 (dV+dP)",
                    "k3b (dV corr)",
                    "k4 (dS)",
                    "k5c (ds_delta)",
                    "k5 (dK+dQ)",
                    "k5b (dK corr)",
                    "k6 (postprocess)",
                ]
        else:
            labels = [f"Kernel {i + 1}" for i in range(len(durations))]

        print("Kernel-level performance (msprof op):")
        print()
        print("  | #   | Kernel                      | Task Duration (us) | Block Dim | Mix Block Dim |")
        print("  |-----|-----------------------------|---------------------|-----------|---------------|")
        for i, dur in enumerate(durations):
            bd = block_dims[i] if i < len(block_dims) else "?"
            mbd = mix_block_dims[i] if i < len(mix_block_dims) else "?"
            label = labels[i] if i < len(labels) else f"Kernel {i + 1}"
            print(f"  | {i + 1:<3} | {label:<27} | {dur:<19.2f} | {bd:<9} | {mbd:<13} |")

        total_us = sum(durations)
        median_dur = sorted(durations)[len(durations) // 2]
        max_dur = max(durations)
        min_dur = min(durations)

        print()
        print(f"  Total Task Duration (sum): {total_us:.2f} us ({total_us / 1000:.3f} ms)")
        print(f"  Median:  {median_dur:.2f} us ({median_dur / 1000:.3f} ms)")
        print(f"  Max:     {max_dur:.2f} us ({max_dur / 1000:.3f} ms)")
        print(f"  Min:     {min_dur:.2f} us ({min_dur / 1000:.3f} ms)")
        print(f"  Launch count: {len(durations)}")

        if capture_mode == "full-pass":
            print()
            print(f"  >>> Backward kernel-only total: {total_us / 1000:.3f} ms <<<")
            print(f"      (= sum of all {len(durations)} backward kernel Task Durations)")
            print("      (= one complete backward pipeline invocation, no host overhead)")
            # vs GPU baseline
            gpu_baseline_us = 28574
            print(f"      vs GPU baseline ({gpu_baseline_us} us): {total_us / gpu_baseline_us:.2f}x")

            # Aggregate by kernel name (collapse iter index)
            agg = {}
            for i, dur in enumerate(durations):
                if i < len(labels):
                    # strip " iterN" suffix
                    key = labels[i].rsplit(" iter", 1)[0] if " iter" in labels[i] else labels[i]
                else:
                    key = f"Kernel {i + 1}"
                agg.setdefault(key, []).append(dur)
            print()
            print("  Per-kernel aggregate (collapsed across K-iterations):")
            print("  | Kernel              | Count | Total (us)   | Mean (us)   | % of total |")
            print("  |---------------------|-------|--------------|-------------|------------|")
            for k, v in agg.items():
                kt = sum(v)
                km = kt / len(v)
                pct = kt / total_us * 100
                print(f"  | {k:<19} | {len(v):<5} | {kt:<12.2f} | {km:<11.2f} | {pct:<10.2f} |")
        else:
            print()
            print(f"  (sample mode: {len(durations)} kernels captured; use --capture-mode full-pass for complete backward total)")
    else:
        print("WARNING: Could not parse Task Duration from msprof output.")
        print("msprof stdout (last 2000 chars):")
        print(stdout[-2000:])

    print()
    print("Performance hints from msprof:")
    hint_pattern = re.compile(r"^\s*\d+\)\s*(.+)", re.MULTILINE)
    hints = hint_pattern.findall(stdout)
    if hints:
        for hint in hints:
            hint = hint.strip()
            if hint:
                print(f"  - {hint}")
    else:
        print("  (no hints reported)")

    dir_match = re.search(r"Profiling results saved in (\S+)", stdout)
    if dir_match:
        print(f"\nDetailed profiling data: {dir_match.group(1)}")

    print()
    print("msprof op completed successfully.")
    return True


# ============================================================================
# main: argparse --level {l0|l1|l2|boundary|all|bench|msprof}
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA + Attention Sink Flash Attention Backward (Varlen) — Ascend NPU")
    parser.add_argument(
        "--level",
        choices=["l0", "l1", "l2", "boundary", "all", "bench", "msprof", "msprof-target"],
        default="l0",
        help="Test level: l0 (precision gate), l1 (functional), l2 (negative), "
        "boundary (special values), all (full suite), bench (do_bench), "
        "msprof (msprof op), msprof-target (lightweight target for msprof op)",
    )
    parser.add_argument(
        "--profiler",
        choices=["do_bench", "msprof"],
        default="do_bench",
        help="Profiler for bench level (do_bench=smoke, msprof=kernel-level)",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "small"],
        default="default",
        help="Bench preset: default (B=8,H=64,q=2048), small (B=1,H=4,q=128)",
    )
    parser.add_argument(
        "--window",
        choices=["none", "128", "both"],
        default="both",
        help="Window config for bench (none=full causal, 128=SWA, both=run both)",
    )
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip L0 3x determinism check (faster, for CI smoke)",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["sample", "full-pass"],
        default="sample",
        help="msprof capture mode: sample (10 kernels, per-kernel analysis), "
        "full-pass (all kernels in one bwd pass, reports TOTAL kernel-only "
        "backward latency)",
    )
    args = parser.parse_args()

    # iter6 O6: enable tilelang compile cache (JIT cache in ~/.tilelang/cache/).
    # disable_cache was causing msprof subprocess to recompile 12 kernels (30min+ timeout).
    # enable_cache: first compile ~7min, subsequent runs reuse cache (~15s).
    tilelang.enable_cache()

    torch.set_default_device("npu")

    if args.level == "bench":
        if args.profiler == "do_bench":
            ok = run_bench(preset=args.preset, window=args.window)
            print("\nTest Passed!" if ok else "\nTest FAILED (bench)")
            sys.exit(0 if ok else 1)
        elif args.profiler == "msprof":
            ok = run_msprof(
                preset=args.preset,
                window=args.window,
                launch_count=10,
                warm_up=1,
                capture_mode=args.capture_mode,
            )
            print("\nTest Passed!" if ok else "\nTest FAILED (msprof)")
            sys.exit(0 if ok else 1)
        sys.exit(0)

    if args.level == "msprof-target":
        # Lightweight bwd-only target invoked by run_msprof subprocess.
        # Uses host-side PyTorch forward (no forward kernel launch) so all
        # main_kernel launches are backward kernels.
        # repeat=3: pass 0 = warmup (skipped by launch-skip-before-match),
        #           pass 1+ = captured by msprof.
        _run_msprof_target(
            preset=args.preset,
            window=args.window,
            repeat=3,
        )
        sys.exit(0)

    if args.level == "msprof":
        ok = run_msprof(
            preset=args.preset,
            window=args.window,
            launch_count=10,
            warm_up=1,
            capture_mode=args.capture_mode,
        )
        print("\nTest Passed!" if ok else "\nTest FAILED (msprof)")
        sys.exit(0 if ok else 1)

    # Precision test levels
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

    # Exit code: only L0/L1 failures cause exit 1
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
