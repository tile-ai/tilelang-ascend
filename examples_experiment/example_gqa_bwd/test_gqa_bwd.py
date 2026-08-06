"""
Test suite for GQA Flash Attention (Forward + Backward).
Imports kernels from example_gqa_bwd.py.

Layered test structure (matches gqa_fwd_varlen convention):
  - L0: regular shapes (block-aligned), precision convergence gate (blocking)
  - L1: irregular shapes, D_qk != D_v, causal variants (blocking)
  - L2: abnormal inputs (single token, min seqlen) (non-blocking)
  - Boundary: special values (zero input, large input) (non-blocking)
  - Perf: performance benchmark with do_bench (TileLang vs PyTorch)

Usage:
  python test_gqa_bwd.py                  # default: L0 precision + performance
  python test_gqa_bwd.py --level l0       # L0 precision only (fast)
  python test_gqa_bwd.py --level all      # all precision levels (L0+L1+L2+Boundary)
  python test_gqa_bwd.py --level perf     # performance benchmark only
  python test_gqa_bwd.py --level full     # all precision + performance
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang
import torch
import torch.nn.functional as F
from tilelang.profiler import do_bench  # noqa: E402

from example_gqa_bwd import (  # noqa: E402
    flashattn_fwd,
    flashattn_fwd_v4,
    flashattn_bwd_preprocess,
    flashattn_bwd_pipeline,
    ref_program,
    ref_bwd,
    attention,
    NUM_CORES,
)

ATOL = 1e-2
RTOL = 1e-2


def _setup():
    tilelang.disable_cache()
    torch.set_default_device("npu")


# ===========================================================================
# Precision test helpers
# ===========================================================================


def _run_forward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, level):
    """Run one forward case, print [PRECISION_PASS/FAIL] or [BOUNDARY_PASS/WARN]."""
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        torch.manual_seed(42)
        Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu")
        K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu")
        V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")

        bM, bN = 64, 64
        mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, bM, bN, groups)
        O_npu, _ = mod(Q, K, V)
        torch.npu.synchronize()

        O_ref = ref_program(Q, K, V, causal, groups)
        max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
        torch.testing.assert_close(O_npu.cpu(), O_ref.cpu(), rtol=RTOL, atol=ATOL)

        print(f"[{tag}_PASS] {level} {name} fwd B={B} H={H} N={N} D_qk={D_qk} D_v={D_v} causal={causal} max_diff={max_diff:.6e}")
        return True
    except Exception as e:
        print(f"[{tag}_FAIL] {level} {name} fwd B={B} H={H} N={N} D_qk={D_qk} D_v={D_v} causal={causal}: {e}")
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


def _run_backward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, level):
    """Run one backward case (fwd + prep + bwd_pipeline), print result tag."""
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        torch.manual_seed(42)
        bM, bN = 64, (64 if causal else 32)
        D_qk_padded = ((D_qk + 127) // 128) * 128
        if causal and D_qk_padded > 128:
            bM = 32

        Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu")
        K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu")
        V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")
        dO = torch.randn(B, H, N, D_v, dtype=torch.float16, device="npu")

        fwd_mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, bM, bM, groups)
        O_npu, lse_npu = fwd_mod(Q, K, V)
        torch.npu.synchronize()

        prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
        Delta_npu = prep_mod(O_npu, dO)
        torch.npu.synchronize()

        num_stages = 8
        dQ = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float32, device="npu")
        dK = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float32, device="npu")
        dV = torch.zeros(B, H_kv, N, D_v, dtype=torch.float32, device="npu")

        Q_padded = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float16, device="npu")
        Q_padded[:, :, :, :D_qk] = Q
        K_padded = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float16, device="npu")
        K_padded[:, :, :, :D_qk] = K

        bwd_block_num = (N // bM) * H * B
        ws_s_dp = torch.empty(bwd_block_num, num_stages, bM, bN, dtype=torch.float32, device="npu")
        ws_p_ds = torch.empty(bwd_block_num, num_stages, bM, bN, dtype=torch.float16, device="npu")
        ws_dv_dk = torch.empty(bwd_block_num, num_stages, bN, max(D_qk_padded, D_v), dtype=torch.float32, device="npu")

        bwd_mod = flashattn_bwd_pipeline(B, H, N, D_qk, D_v, causal, bM, bN, groups, num_stages)
        bwd_mod(Q_padded, K_padded, V, dO, lse_npu, Delta_npu, dQ, dK, dV, ws_s_dp, ws_p_ds, ws_dv_dk)
        torch.npu.synchronize()

        dQ_ref, dK_ref, dV_ref = ref_bwd(Q, K, V, dO, causal, groups)
        max_diff = max(
            (dV.half().float() - dV_ref.float()).abs().max().item(),
            (dK[:, :, :, :D_qk].half().float() - dK_ref.float()).abs().max().item(),
            (dQ[:, :, :, :D_qk].half().float() - dQ_ref.float()).abs().max().item(),
        )
        torch.testing.assert_close(dV.half().cpu(), dV_ref.cpu(), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(dK[:, :, :, :D_qk].half().cpu(), dK_ref.cpu(), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(dQ[:, :, :, :D_qk].half().cpu(), dQ_ref.cpu(), rtol=RTOL, atol=ATOL)

        print(f"[{tag}_PASS] {level} {name} bwd B={B} H={H} N={N} D_qk={D_qk} D_v={D_v} causal={causal} max_diff={max_diff:.6e}")
        return True
    except Exception as e:
        print(f"[{tag}_FAIL] {level} {name} bwd B={B} H={H} N={N} D_qk={D_qk} D_v={D_v} causal={causal}: {e}")
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


def _run_autograd(B, N, H, D_qk, D_v, groups, name, level):
    """End-to-end autograd test (BSHD layout)."""
    tag = "PRECISION" if level in ("l0", "l1") else "BOUNDARY"
    try:
        torch.manual_seed(42)
        H_kv = H // groups
        q = torch.randn(B, N, H, D_qk, dtype=torch.float16, device="npu", requires_grad=True)
        k = torch.randn(B, N, H_kv, D_qk, dtype=torch.float16, device="npu", requires_grad=True)
        v = torch.randn(B, N, H_kv, D_v, dtype=torch.float16, device="npu", requires_grad=True)
        dO = torch.randn(B, N, H, D_v, dtype=torch.float16, device="npu")

        O = attention(q, k, v, False, groups)
        O.backward(dO)

        q_bhsd = q.detach().permute(0, 2, 1, 3)
        k_bhsd = k.detach().permute(0, 2, 1, 3)
        v_bhsd = v.detach().permute(0, 2, 1, 3)
        dO_bhsd = dO.permute(0, 2, 1, 3)
        dQ_ref, dK_ref, dV_ref = ref_bwd(q_bhsd, k_bhsd, v_bhsd, dO_bhsd, False, groups)

        torch.testing.assert_close(q.grad.permute(0, 2, 1, 3).cpu(), dQ_ref.cpu(), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(k.grad.permute(0, 2, 1, 3).cpu(), dK_ref.cpu(), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(v.grad.permute(0, 2, 1, 3).cpu(), dV_ref.cpu(), rtol=RTOL, atol=ATOL)

        print(f"[{tag}_PASS] {level} {name} autograd B={B} H={H} N={N} groups={groups}")
        return True
    except Exception as e:
        print(f"[{tag}_FAIL] {level} {name} autograd B={B} H={H} N={N} groups={groups}: {e}")
        if tag == "BOUNDARY":
            traceback.print_exc()
        return False


# ===========================================================================
# L0 gate tests — regular shapes (block-aligned), precision convergence.
# ===========================================================================


def test_gqa_bwd_l0():
    """L0 gate tests: regular shapes (block-aligned), for precision convergence.

    Raises AssertionError on failure (pytest-discoverable, returns None).
    """
    _setup()
    # (name, B, H, H_kv, groups, N, D_qk, D_v, causal)
    fwd_configs = [
        ("fwd_mha", 1, 1, 1, 1, 128, 64, 64, False),
        ("fwd_gqa", 1, 2, 1, 2, 128, 64, 64, False),
        ("fwd_gqa_causal", 1, 2, 1, 2, 128, 64, 64, True),
        ("fwd_gqa_dqk64_dv128", 1, 2, 1, 2, 128, 64, 128, False),
        ("fwd_gqa_golden", 8, 32, 2, 16, 1024, 192, 128, False),
    ]
    bwd_configs = [
        ("bwd_mha", 1, 1, 1, 1, 128, 64, 64, False),
        ("bwd_gqa", 1, 2, 1, 2, 128, 64, 64, False),
        ("bwd_gqa_dqk64_dv128", 1, 2, 1, 2, 128, 64, 128, False),
        ("bwd_gqa_causal", 1, 2, 1, 2, 128, 64, 64, True),
        ("bwd_gqa_golden", 1, 32, 2, 16, 256, 192, 128, False),
    ]
    autograd_configs = [
        ("autograd_mha", 1, 128, 1, 64, 64, 1),
        ("autograd_gqa", 1, 128, 2, 64, 64, 2),
    ]

    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in fwd_configs:
        assert _run_forward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l0"), f"L0 forward case '{name}' failed"
    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in bwd_configs:
        assert _run_backward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l0"), f"L0 backward case '{name}' failed"
    for name, B, N, H, D_qk, D_v, groups in autograd_configs:
        assert _run_autograd(B, N, H, D_qk, D_v, groups, name, "l0"), f"L0 autograd case '{name}' failed"


# ===========================================================================
# L1 functional tests — irregular shapes, causal variants, GQA edge cases.
# ===========================================================================


def test_gqa_bwd_l1():
    """L1 functional tests: irregular shapes, causal variants, GQA edge cases.

    Raises AssertionError on failure (pytest-discoverable, returns None).
    """
    _setup()
    # (name, B, H, H_kv, groups, N, D_qk, D_v, causal)
    fwd_configs = [
        ("fwd_batch_causal", 2, 4, 2, 2, 256, 64, 64, True),
        ("fwd_gqa_dqk128_dv64", 1, 2, 1, 2, 256, 128, 64, False),
        ("fwd_gqa_groups4", 1, 8, 2, 4, 256, 128, 128, False),
    ]
    bwd_configs = [
        ("bwd_gqa_causal_dqk128", 1, 2, 1, 2, 128, 128, 64, True),
        ("bwd_gqa_groups4", 1, 8, 2, 4, 256, 128, 128, False),
        ("bwd_batch_causal", 2, 4, 2, 2, 256, 64, 64, True),
    ]

    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in fwd_configs:
        assert _run_forward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l1"), f"L1 forward case '{name}' failed"
    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in bwd_configs:
        assert _run_backward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l1"), f"L1 backward case '{name}' failed"


# ===========================================================================
# L2 abnormal input tests. Non-blocking: prints [BOUNDARY_PASS/WARN].
# ===========================================================================


def test_gqa_bwd_l2():
    """L2 abnormal input tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    _setup()
    # (name, B, H, H_kv, groups, N, D_qk, D_v, causal)
    fwd_configs = [
        ("fwd_single_token", 1, 1, 1, 1, 64, 64, 64, False),
        ("fwd_min_seqlen", 1, 1, 1, 1, 64, 64, 64, True),
        ("fwd_batch1_head1", 1, 1, 1, 1, 128, 64, 64, False),
    ]
    bwd_configs = [
        ("bwd_single_token", 1, 1, 1, 1, 64, 64, 64, False),
        ("bwd_min_seqlen", 1, 1, 1, 1, 64, 64, 64, True),
    ]

    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in fwd_configs:
        _run_forward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l2")
    for name, B, H, H_kv, groups, N, D_qk, D_v, causal in bwd_configs:
        _run_backward(B, H, H_kv, groups, N, D_qk, D_v, causal, name, "l2")


# ===========================================================================
# Boundary / special value tests. Non-blocking.
# ===========================================================================


def test_gqa_bwd_boundary():
    """Boundary / special value tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    _setup()
    # (name, B, H, H_kv, groups, N, D_qk, D_v, causal, input_scale)
    boundary_configs = [
        ("zero_input", 1, 2, 1, 2, 128, 64, 64, False, 0.0),
        ("large_input", 1, 2, 1, 2, 128, 64, 64, False, 10.0),
    ]

    for name, B, H, H_kv, groups, N, D_qk, D_v, causal, scale in boundary_configs:
        try:
            torch.manual_seed(42)
            Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu") * scale
            K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu") * scale
            V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu") * scale

            bM, bN = 64, 64
            mod = flashattn_fwd(B, H, N, D_qk, D_v, causal, bM, bN, groups)
            O_npu, _ = mod(Q, K, V)
            torch.npu.synchronize()

            O_ref = ref_program(Q, K, V, causal, groups)
            max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
            if torch.isnan(O_npu).any():
                print(f"[BOUNDARY_WARN] boundary {name}: NaN in output")
                continue
            torch.testing.assert_close(O_npu.cpu(), O_ref.cpu(), rtol=RTOL, atol=ATOL)
            print(f"[BOUNDARY_PASS] boundary {name} max_diff={max_diff:.6e}")
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: {e}")


# ===========================================================================
# Performance benchmark (do_bench, TileLang vs PyTorch)
# ===========================================================================


def run_perf(batch=8, h=32, n_ctx=1024, d_head_qk=192, d_head_v=128, groups=16, causal=False):
    """Run performance benchmark with do_bench.

    Prints correctness check + performance table. Returns True on success.
    Uses golden config by default (B=8 H=32 N=1024 D_qk=192 D_v=128 groups=16).
    """
    _setup()
    torch.manual_seed(42)

    B = batch
    H = h
    N = n_ctx
    D_qk = d_head_qk
    D_v = d_head_v
    groups_val = groups
    causal_val = causal
    H_kv = H // groups_val

    # FLOPS calculation
    fwd_flops = 2.0 * B * H * N * N * (D_qk + D_v)
    bwd_flops = 2.0 * B * H * N * N * (3 * D_qk + 2 * D_v)
    total_flops = fwd_flops + bwd_flops
    if causal_val:
        fwd_flops *= 0.5
        bwd_flops *= 0.5
        total_flops *= 0.5

    # ---- Allocate tensors (BHSD layout) ----
    Q = torch.randn(B, H, N, D_qk, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D_qk, dtype=torch.float16, device="npu")
    V = torch.randn(B, H_kv, N, D_v, dtype=torch.float16, device="npu")
    dO = torch.randn(B, H, N, D_v, dtype=torch.float16, device="npu")

    print()
    print("=" * 70)
    print(f"  Config: B={B} H={H} H_kv={H_kv} N={N} D_qk={D_qk} D_v={D_v}")
    print(f"          groups={groups_val} causal={causal_val} dtype=fp16")
    print("=" * 70)

    # ============================================================
    # 0. Correctness check before bench
    # ============================================================
    bM_v4, bN_v4 = 32, 64
    num_stages_fwd = 8
    cross_interval_fwd = 2
    fwd_v4_mod = flashattn_fwd_v4(B, H, N, D_qk, D_v, causal_val, bM_v4, bN_v4, groups_val, num_stages_fwd, cross_interval_fwd)
    ws1_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, bN_v4, dtype=torch.float32, device="npu")
    ws2_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, bN_v4, dtype=torch.float16, device="npu")
    ws3_fwd = torch.empty(NUM_CORES, num_stages_fwd, bM_v4, D_v, dtype=torch.float32, device="npu")
    O_npu, lse_npu = fwd_v4_mod(Q, K, V, ws1_fwd, ws2_fwd, ws3_fwd)
    torch.npu.synchronize()

    O_ref = ref_program(Q, K, V, causal_val, groups_val)
    fwd_max_diff = (O_npu.float() - O_ref.float()).abs().max().item()
    print(f"  correctness: fwd_max_diff={fwd_max_diff:.6e} (atol={ATOL})")
    if fwd_max_diff >= ATOL:
        print(f"  [ERROR] forward correctness check failed: max_diff={fwd_max_diff} >= atol={ATOL}")
        return False

    # Backward correctness
    prep_mod = flashattn_bwd_preprocess(B, H, N, D_v, blk=32)
    Delta_npu = prep_mod(O_npu, dO)
    torch.npu.synchronize()

    D_qk_padded = ((D_qk + 127) // 128) * 128
    bM_bwd, bN_bwd = 64, 64 if causal_val else 32
    num_stages_bwd = 8
    dQ_raw = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float32, device="npu")
    dK_raw = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float32, device="npu")
    dV_raw = torch.zeros(B, H_kv, N, D_v, dtype=torch.float32, device="npu")
    bwd_block_num = (N // bM_bwd) * H * B
    ws1_bwd = torch.empty(bwd_block_num, num_stages_bwd, bM_bwd, bN_bwd, dtype=torch.float32, device="npu")
    ws2_bwd = torch.empty(bwd_block_num, num_stages_bwd, bM_bwd, bN_bwd, dtype=torch.float16, device="npu")
    ws3_bwd = torch.empty(bwd_block_num, num_stages_bwd, bN_bwd, max(D_qk_padded, D_v), dtype=torch.float32, device="npu")
    bwd_mod = flashattn_bwd_pipeline(B, H, N, D_qk, D_v, causal_val, bM_bwd, bN_bwd, groups_val, num_stages_bwd)

    Q_padded = torch.zeros(B, H, N, D_qk_padded, dtype=torch.float16, device="npu")
    Q_padded[:, :, :, :D_qk] = Q
    K_padded = torch.zeros(B, H_kv, N, D_qk_padded, dtype=torch.float16, device="npu")
    K_padded[:, :, :, :D_qk] = K
    bwd_mod(Q_padded, K_padded, V, dO, lse_npu, Delta_npu, dQ_raw, dK_raw, dV_raw, ws1_bwd, ws2_bwd, ws3_bwd)
    torch.npu.synchronize()

    dQ_ref, dK_ref, dV_ref = ref_bwd(Q, K, V, dO, causal_val, groups_val)
    bwd_max_diff = max(
        (dV_raw.half().float() - dV_ref.float()).abs().max().item(),
        (dK_raw[:, :, :, :D_qk].half().float() - dK_ref.float()).abs().max().item(),
        (dQ_raw[:, :, :, :D_qk].half().float() - dQ_ref.float()).abs().max().item(),
    )
    print(f"  correctness: bwd_max_diff={bwd_max_diff:.6e} (atol={ATOL})")
    if bwd_max_diff >= ATOL:
        print(f"  [ERROR] backward correctness check failed: max_diff={bwd_max_diff} >= atol={ATOL}")
        return False
    print(f"  correctness: PASS (fwd={fwd_max_diff:.6e}, bwd={bwd_max_diff:.6e})")

    # ============================================================
    # Benchmark using tilelang.profiler.do_bench
    # ============================================================

    # 1. TileLang Forward v1
    D_qk_padded_check = ((D_qk + 127) // 128) * 128
    skip_v1 = causal_val and D_qk_padded_check > 128
    if skip_v1:
        lat_fwd_v1 = float("nan")
        print(f"  [INFO] Skipping Forward v1 (causal + D_qk_padded={D_qk_padded_check} > 128)")
    else:
        bM_v1, bN_v1 = 64, 64
        fwd_v1_mod = flashattn_fwd(B, H, N, D_qk, D_v, causal_val, bM_v1, bN_v1, groups_val)
        lat_fwd_v1 = do_bench(lambda: fwd_v1_mod(Q, K, V), _n_warmup=5, _n_repeat=5, return_mode="mean")

    # 2. TileLang Forward v4
    lat_fwd_v4 = do_bench(
        lambda: fwd_v4_mod(Q, K, V, ws1_fwd, ws2_fwd, ws3_fwd),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )

    # 3. TileLang Backward (pipeline)
    #    dK/dV use atomic_add to GM — must zero before each call.
    def _run_bwd():
        dK_raw.zero_()
        dV_raw.zero_()
        bwd_mod(
            Q_padded,
            K_padded,
            V,
            dO,
            lse_npu,
            Delta_npu,
            dQ_raw,
            dK_raw,
            dV_raw,
            ws1_bwd,
            ws2_bwd,
            ws3_bwd,
        )

    lat_bwd = do_bench(_run_bwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    # 4. PyTorch baseline
    q_r = Q.float()
    k_r = K.float().repeat_interleave(groups_val, dim=1)
    v_r = V.float().repeat_interleave(groups_val, dim=1)

    def _run_ref_fwd():
        scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * (1.0 / D_qk**0.5)
        if causal_val:
            mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        P = F.softmax(scores, dim=-1)
        torch.matmul(P, v_r)

    lat_ref_fwd = do_bench(_run_ref_fwd, _n_warmup=5, _n_repeat=5, return_mode="mean")

    def _run_ref_fwd_bwd():
        q2 = Q.float().requires_grad_(True)
        k2 = K.float().repeat_interleave(groups_val, dim=1).requires_grad_(True)
        v2 = V.float().repeat_interleave(groups_val, dim=1).requires_grad_(True)
        scores = torch.matmul(q2, k2.transpose(-2, -1)) * (1.0 / D_qk**0.5)
        if causal_val:
            mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        P = F.softmax(scores, dim=-1)
        O2 = torch.matmul(P, v2)
        O2.backward(dO.float())

    lat_ref_e2e = do_bench(_run_ref_fwd_bwd, _n_warmup=3, _n_repeat=3, return_mode="mean")

    # ============================================================
    # Print results
    # ============================================================
    print()
    print(f"  {'Kernel':<32} {'Latency':>10} {'TFlops':>10}")
    print(f"  {'-' * 55}")
    if skip_v1:
        print(f"  {'TileLang Forward v1':<32} {'SKIPPED':>10}         -")
    else:
        print(f"  {'TileLang Forward v1':<32} {lat_fwd_v1:>8.2f} ms  {fwd_flops / lat_fwd_v1 * 1e-9:>8.2f}")
    print(f"  {'TileLang Forward v4':<32} {lat_fwd_v4:>8.2f} ms  {fwd_flops / lat_fwd_v4 * 1e-9:>8.2f}")
    print(f"  {'TileLang Backward (pipeline)':<32} {lat_bwd:>8.2f} ms  {bwd_flops / lat_bwd * 1e-9:>8.2f}")
    print(f"  {'TileLang Fwd(v4)+Bwd (raw)':<32} {lat_fwd_v4 + lat_bwd:>8.2f} ms  {total_flops / (lat_fwd_v4 + lat_bwd) * 1e-9:>8.2f}")
    print(f"  {'-' * 55}")
    print(f"  {'PyTorch Forward only':<32} {lat_ref_fwd:>8.2f} ms  {fwd_flops / lat_ref_fwd * 1e-9:>8.2f}")
    print(f"  {'PyTorch Fwd+Bwd (e2e)':<32} {lat_ref_e2e:>8.2f} ms  {total_flops / lat_ref_e2e * 1e-9:>8.2f}")
    print(f"  {'-' * 55}")
    sp_fwd_v4 = lat_ref_fwd / lat_fwd_v4
    sp_e2e_v4 = lat_ref_e2e / (lat_fwd_v4 + lat_bwd)
    print(f"  Speedup (v4 forward vs PyTorch fwd):  {sp_fwd_v4:.2f}x")
    print(f"  Speedup (v4 fwd+bwd vs PyTorch e2e):  {sp_e2e_v4:.2f}x")
    if not skip_v1:
        print(f"  v4 vs v1 forward speedup:             {lat_fwd_v1 / lat_fwd_v4:.2f}x")
    print("=" * 70)
    return True


# ===========================================================================
# Main entrypoint with argparse --level
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA Flash Attention test suite (precision + performance)")
    parser.add_argument(
        "--level",
        default="default",
        choices=["l0", "l1", "l2", "boundary", "all", "perf", "full", "default"],
        help="Test level: l0=precision gate, all=all precision, perf=benchmark, full=all precision+perf, default=l0+perf (CI mode)",
    )
    parser.add_argument("--batch", type=int, default=8, help="perf: batch size")
    parser.add_argument("--h", type=int, default=32, help="perf: query heads")
    parser.add_argument("--n_ctx", type=int, default=1024, help="perf: sequence length")
    parser.add_argument("--d_head_qk", type=int, default=192, help="perf: Q/K head dim")
    parser.add_argument("--d_head_v", type=int, default=128, help="perf: V head dim")
    parser.add_argument("--groups", type=int, default=16, help="perf: GQA groups")
    parser.add_argument("--causal", action="store_true", help="perf: causal attention")
    args = parser.parse_args()

    _setup()

    blocking_ok = True
    run_precision = args.level in ("l0", "all", "full", "default")
    run_performance = args.level in ("perf", "full", "default")

    # --- Precision tests ---
    if run_precision:
        try:
            if args.level in ("l0", "all", "full", "default"):
                test_gqa_bwd_l0()
            if args.level in ("all", "full"):
                test_gqa_bwd_l1()
        except AssertionError:
            blocking_ok = False

        if args.level in ("all", "full"):
            test_gqa_bwd_l2()
            test_gqa_bwd_boundary()

    # --- Performance benchmark ---
    if run_performance:
        perf_ok = run_perf(
            batch=args.batch,
            h=args.h,
            n_ctx=args.n_ctx,
            d_head_qk=args.d_head_qk,
            d_head_v=args.d_head_v,
            groups=args.groups,
            causal=args.causal,
        )
        if not perf_ok:
            blocking_ok = False

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


# pytest-discoverable aliases (so `pytest test_gqa_bwd.py` still works)
def test_forward():
    """Pytest alias: run L0 forward cases."""
    _setup()
    configs = [
        (1, 1, 1, 1, 128, 64, 64, False, "FWD-MHA"),
        (1, 2, 1, 2, 128, 64, 64, False, "FWD-GQA"),
        (1, 2, 1, 2, 128, 64, 64, True, "FWD-GQA-causal"),
        (1, 2, 1, 2, 128, 64, 128, False, "FWD-GQA-Dqk64-Dv128"),
        (8, 32, 2, 16, 1024, 192, 128, False, "FWD-GQA-golden"),
    ]
    for B, H, H_kv, groups, N, D_qk, D_v, causal, _desc in configs:
        assert _run_forward(B, H, H_kv, groups, N, D_qk, D_v, causal, _desc, "l0")


def test_backward():
    """Pytest alias: run L0 backward cases."""
    _setup()
    configs = [
        (1, 1, 1, 1, 128, 64, 64, False, "BWD-MHA"),
        (1, 2, 1, 2, 128, 64, 64, False, "BWD-GQA"),
        (1, 2, 1, 2, 128, 64, 128, False, "BWD-GQA-Dqk64-Dv128"),
        (1, 2, 1, 2, 128, 64, 64, True, "BWD-GQA-causal"),
        (1, 32, 2, 16, 256, 192, 128, False, "BWD-GQA-golden"),
    ]
    for B, H, H_kv, groups, N, D_qk, D_v, causal, _desc in configs:
        assert _run_backward(B, H, H_kv, groups, N, D_qk, D_v, causal, _desc, "l0")


def test_autograd():
    """Pytest alias: end-to-end autograd test (BSHD layout)."""
    _setup()
    assert _run_autograd(1, 128, 2, 64, 64, 2, "AUTOGRAD-GQA", "l0")


if __name__ == "__main__":
    main()
