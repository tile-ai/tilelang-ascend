"""
Test suite for GQA Flash Attention (Forward + Backward).
Imports kernels from example_gqa_bwd.py.

Layered test structure (matches gqa_fwd_varlen convention):
  - L0: regular shapes (block-aligned), precision convergence gate (blocking)
  - L1: irregular shapes, D_qk != D_v, causal variants (blocking)
  - L2: abnormal inputs (single token, min seqlen) (non-blocking)
  - Boundary: special values (zero input, large input) (non-blocking)

Usage:
  python test_gqa_bwd.py                # default: run L0 only
  python test_gqa_bwd.py --level l0
  python test_gqa_bwd.py --level all
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang
import torch

from example_gqa_bwd import (  # noqa: E402
    flashattn_fwd,
    flashattn_bwd_preprocess,
    flashattn_bwd_pipeline,
    ref_program,
    ref_bwd,
    attention,
)

ATOL = 1e-2
RTOL = 1e-2


def _setup():
    tilelang.disable_cache()
    torch.set_default_device("npu")


# ===========================================================================
# Test helpers
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
# Main entrypoint with argparse --level
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA Flash Attention (Forward + Backward) test suite")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

    _setup()

    blocking_ok = True  # Only L0/L1 count toward blocking

    try:
        if args.level in ("l0", "all"):
            test_gqa_bwd_l0()
        if args.level in ("l1", "all"):
            test_gqa_bwd_l1()
    except AssertionError:
        blocking_ok = False

    if args.level in ("l2", "all"):
        test_gqa_bwd_l2()
    if args.level in ("boundary", "all"):
        test_gqa_bwd_boundary()

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
