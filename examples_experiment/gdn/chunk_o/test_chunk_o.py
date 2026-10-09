"""Test suite for chunk_o (chunk-based linear-attention O forward) on Ascend NPU.

Hierarchical test levels:
  L0: Precision gate (regular shapes, block-divisible) — must PASS
  L1: Functional (irregular/prime shapes, non-default tiling) — must PASS
  L2: Negative (invalid inputs — must be rejected) — blocking
  Boundary: Special values (zero/inf/nan/extreme) — non-blocking

Usage:
  python test_chunk_o.py --level l0      # L0 only (precision convergence)
  python test_chunk_o.py --level all     # Full suite (L0+L1+L2+Boundary)
"""

import argparse
import math
import sys
import zlib

import tilelang
import torch

from example_chunk_o import chunk_o, _prepare_inputs


# ---------------------------------------------------------------------------
# Golden reference implementation
# ---------------------------------------------------------------------------


def golden_chunk_o(Q, K, V, HIDDEN, G, scale, chunk_size=64, use_g=True, output_dtype=torch.bfloat16):
    """Pure PyTorch reference implementation of chunk_o forward.

    Inputs: Q/K/V are (B,H,S,DK/DV), HIDDEN is (B,H,BS,DK,DV), G is (B,S,H).
    Output O is (B,S,H,DV).

    Scale folding: `scale` is accepted for API compatibility but NOT applied
    here — V and HIDDEN are pre-scaled by `scale` on the host.

    Per chunk (BS = chunk_size):
      1. O = Q @ HIDDEN  (HIDDEN already includes scale)
      2. A = Q @ K^T
      3. If use_g: O *= exp(G[i]); A = A*exp(G[i]-G[j]) if (G_diff<=0 and i>=j) else 0
      4. If not use_g: A = 0 if i<j (lower triangle, i>=j kept)
      5. O += A @ V    (V already includes scale)
    """
    B, H, S, DK = Q.shape
    _, _, _, DV = V.shape
    BS = chunk_size
    n_c = S // BS

    # Permute back to BSHD for computation
    Q_bshd = Q.cpu().float().permute(0, 2, 1, 3).contiguous()  # (B,S,H,DK)
    K_bshd = K.cpu().float().permute(0, 2, 1, 3).contiguous()  # (B,S,H,DK)
    V_bshd = V.cpu().float().permute(0, 2, 1, 3).contiguous()  # (B,S,H,DV)
    HIDDEN_bshd = HIDDEN.cpu().float().permute(0, 2, 1, 3, 4).contiguous()  # (B,BS,H,DK,DV)

    Qf = Q_bshd.reshape(B, n_c, BS, H, DK)
    Kf = K_bshd.reshape(B, n_c, BS, H, DK)
    Vf = V_bshd.reshape(B, n_c, BS, H, DV)
    Hf = HIDDEN_bshd  # (B, n_c, H, DK, DV)
    Gf = G.cpu().float().reshape(B, n_c, BS, H)

    Qp = Qf.permute(0, 1, 3, 2, 4)  # (B, n_c, H, BS, DK)
    Kp = Kf.permute(0, 1, 3, 2, 4)  # (B, n_c, H, BS, DK)

    # Step 1: O = Q @ HIDDEN -> (B, n_c, H, BS, DV)
    O = torch.matmul(Qp, Hf)

    # Step 2: A = Q @ K^T -> (B, n_c, H, BS, BS)
    A = torch.matmul(Qp, Kp.transpose(-1, -2))

    # lower_tri: i >= j (causal mask, including diagonal)
    idx = torch.arange(BS, device="cpu")
    lower_tri = idx.unsqueeze(1) >= idx.unsqueeze(0)  # (BS, BS)

    if use_g:
        Gp = Gf.permute(0, 1, 3, 2)  # (B, n_c, H, BS)

        exp_G = torch.exp(Gp)  # (B, n_c, H, BS)
        O = O * exp_G.unsqueeze(-1)

        G_diff = Gp.unsqueeze(-1) - Gp.unsqueeze(-2)  # (B, n_c, H, BS, BS)
        exp_G_diff = torch.exp(G_diff)

        mask = (G_diff <= 0) & lower_tri
        A = A * exp_G_diff
        A = torch.where(mask, A, torch.zeros_like(A))
    else:
        A = torch.where(lower_tri, A, torch.zeros_like(A))

    # Step 5: O += A @ V
    Vp = Vf.permute(0, 1, 3, 2, 4)  # (B, n_c, H, BS, DV)
    O = O + torch.matmul(A, Vp)

    # Output stays BSHD (B,S,H,DV)
    O = O.permute(0, 1, 3, 2, 4).reshape(B, S, H, DV)
    return O.to(output_dtype)


# ---------------------------------------------------------------------------
# Precision check helpers
# ---------------------------------------------------------------------------


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Floating-point: mixed tolerance; integer: exact match (zero error).
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # atol 0.125,  rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # integer: exact match, any mismatch = FAIL
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Precision check: return (passed, matched_ratio, max_abs_error).

    Floating-point dual-gate: matched_ratio >= required AND max_abs_error <= max_abs_error_limit.
    Integer: element-wise exact equality. inf/nan positions are structurally compared
    and excluded from numeric tolerance.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    # INF/NAN structural comparison: actual and golden inf/nan positions must match
    special = ~torch.isfinite(g)  # golden inf/nan positions
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # compare finite positions; actual inf/nan => fail
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()  # actual inf/nan => abs_err=inf => element-wise fail
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run_and_check(B, S, H, DK, DV, chunk_size, use_g, block_DK, block_DV, label, dtype_str="bfloat16", level="L0"):
    """Compile, run, compare against golden, print tri-state marker."""
    scale = DK**-0.5
    torch.manual_seed(42 + zlib.crc32(label.encode()) % 1000)
    Q, K, V, HIDDEN, G, G_T, S_act = _prepare_inputs(B, S, H, DK, DV, chunk_size)

    try:
        kernel = chunk_o(B, S_act, H, DK, DV, dtype_str, dtype_str, "float32", "float32", chunk_size, use_g, block_DK, block_DV)
    except Exception as e:
        if level == "L2":
            if isinstance(e, (ValueError, TypeError, RuntimeError, AssertionError, ZeroDivisionError)):
                print(f"  [{label}] [BOUNDARY_PASS] correctly rejected: {type(e).__name__}")
                return True
            print(f"  [{label}] [BOUNDARY_FAIL] unexpected exception: {type(e).__name__}: {e}")
            return False
        print(f"  [{label}] RUNTIME ERROR: {e}")
        return False

    O_golden = golden_chunk_o(Q, K, V, HIDDEN, G, scale, chunk_size=chunk_size, use_g=use_g)

    try:
        O_actual = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
    except Exception as e:
        if level == "L2":
            if isinstance(e, (ValueError, TypeError, RuntimeError, AssertionError, ZeroDivisionError)):
                print(f"  [{label}] [BOUNDARY_PASS] correctly rejected: {type(e).__name__}")
                return True
            print(f"  [{label}] [BOUNDARY_FAIL] unexpected exception: {type(e).__name__}: {e}")
            return False
        print(f"  [{label}] RUNTIME ERROR: {e}")
        return False

    passed, ratio, max_abs = check_precision(O_actual, O_golden, dtype_str)

    if level in ("L0", "L1"):
        tag = "PRECISION_PASS" if passed else "PRECISION_FAIL"
    else:
        tag = "BOUNDARY_PASS" if passed else "BOUNDARY_WARN"
    print(f"  [{label}] [{tag}] ratio={ratio:.4f} max_abs={max_abs:.3e}")
    return passed


def _run_exception(name, fn, expect_type, expect_msg):
    """L2 single case: fn() feeds illegal input, REQUIRED to be rejected by
    the entry validation with the expected exception type and a message that
    identifies the actual root cause.

    Returns True only for a correct rejection (expected type + actionable
    message). Both failure modes below count toward the exit code:
    - silently accepted illegal input (entry validation missing/too weak);
    - rejection by an unrelated exception (OOM, name error, ...) — must not
      be reported as a correct rejection.
    """
    try:
        fn()
    except Exception as e:
        if isinstance(e, expect_type) and expect_msg in str(e):
            print(f"  [BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
            return True
        print(
            f"  [BOUNDARY_FAIL] l2 {name}: wrong rejection: expected "
            f"{expect_type.__name__} containing {expect_msg!r}, got "
            f"{type(e).__name__}: {str(e)[:120]}"
        )
        return False
    print(f"  [BOUNDARY_FAIL] l2 {name}: illegal input NOT rejected (silently accepted)")
    return False


# ---------------------------------------------------------------------------
# L0: Precision gate (regular shapes)
# ---------------------------------------------------------------------------


def run_l0_tests():
    """L0 precision gate tests — must all PASS."""
    results = []

    print("=== L0: Main gating (B=1, S=32768, H=32, DK=128, DV=128, use_g=True) ===")
    r = _run_and_check(1, 32768, 32, 128, 128, 64, True, 128, 128, "l0_main_gating")
    results.append(("l0_main_gating", r))

    print("\n=== L0: Main no-gating (B=1, S=32768, H=32, DK=128, DV=128, use_g=False) ===")
    r = _run_and_check(1, 32768, 32, 128, 128, 64, False, 128, 128, "l0_main_no_gating")
    results.append(("l0_main_no_gating", r))

    print("\n=== L0: Small gating (B=1, S=64, H=1, DK=64, DV=64, use_g=True) ===")
    r = _run_and_check(1, 64, 1, 64, 64, 64, True, 64, 64, "l0_small_gating")
    results.append(("l0_small_gating", r))

    all_pass = all(r for _, r in results)
    if all_pass:
        print("\n>>> L0: [PRECISION_PASS] All L0 tests passed <<<")
    else:
        failed = [n for n, r in results if not r]
        print(f"\n>>> L0: [PRECISION_FAIL] Failed: {failed} <<<")
    return all_pass


# ---------------------------------------------------------------------------
# L1: Functional tests (irregular/prime shapes, non-default tiling)
# ---------------------------------------------------------------------------


def run_l1_tests():
    """L1 functional tests — must all PASS."""
    results = []

    print("=== L1: Non-divisible S (S=322, chunk=64 → 5 full + 1 tail) ===")
    r = _run_and_check(1, 322, 4, 128, 128, 64, True, 128, 128, "l1_nondiv_s_gating")
    results.append(("l1_nondiv_s_gating", r))

    print("\n=== L1: B>1 (B=2, S=256, H=4, use_g=True) ===")
    r = _run_and_check(2, 256, 4, 128, 128, 64, True, 128, 128, "l1_b2_gating")
    results.append(("l1_b2_gating", r))

    print("\n=== L1: Smaller block_DK (DK=128, block_DK=64, use_g=True) ===")
    r = _run_and_check(1, 512, 4, 128, 128, 64, True, 64, 128, "l1_small_block_dk")
    results.append(("l1_small_block_dk", r))

    print("\n=== L1: Smaller block_DV (DV=128, block_DV=64, use_g=True) ===")
    r = _run_and_check(1, 512, 4, 128, 128, 64, True, 128, 64, "l1_small_block_dv")
    results.append(("l1_small_block_dv", r))

    print("\n=== L1: Non-divisible S no-gating (S=322, chunk=64, use_g=False) ===")
    r = _run_and_check(1, 322, 4, 128, 128, 64, False, 128, 128, "l1_nondiv_s_no_gating")
    results.append(("l1_nondiv_s_no_gating", r))

    print("\n=== L1: Small DK/DV no-gating (DK=64, DV=64, use_g=False) ===")
    r = _run_and_check(1, 256, 4, 64, 64, 64, False, 64, 64, "l1_small_dkdv_no_gating")
    results.append(("l1_small_dkdv_no_gating", r))

    print("\n=== L1: Prime chunks (S=448, 7 chunks, use_g=True) ===")
    r = _run_and_check(1, 448, 4, 128, 128, 64, True, 128, 128, "l1_prime_chunks")
    results.append(("l1_prime_chunks", r))

    print("\n=== L1: Mid tail (S=160, 2 full + tail of 32, use_g=True) ===")
    r = _run_and_check(1, 160, 4, 128, 128, 64, True, 128, 128, "l1_tail_mid")
    results.append(("l1_tail_mid", r))

    print("\n=== L1: chunk_size=32 (block_S=32, use_g=True) ===")
    r = _run_and_check(1, 256, 4, 128, 128, 32, True, 128, 128, "l1_chunk32")
    results.append(("l1_chunk32", r))

    print("\n=== L1: Custom scale (scale=0.1, use_g=True) ===")
    torch.manual_seed(42)
    Q, K, V, HIDDEN, G, G_T, S_act = _prepare_inputs(1, 256, 4, 128, 128, 64, scale=0.1)
    try:
        kernel_sc = chunk_o(1, S_act, 4, 128, 128, "bfloat16", "bfloat16", "float32", "float32", 64, True, 128, 128)
        O_act = kernel_sc(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, 0.1, 64, True)
        p, r_ratio, m_abs = check_precision(O_act, O_gold, "bfloat16")
        tag = "PRECISION_PASS" if p else "PRECISION_FAIL"
        print(f"  [l1_custom_scale] [{tag}] ratio={r_ratio:.4f} max_abs={m_abs:.3e}")
        results.append(("l1_custom_scale", p))
    except Exception as e:
        print(f"  [l1_custom_scale] RUNTIME ERROR: {e}")
        results.append(("l1_custom_scale", False))

    all_pass = all(r for _, r in results)
    if all_pass:
        print("\n>>> L1: [PRECISION_PASS] All L1 tests passed <<<")
    else:
        failed = [n for n, r in results if not r]
        print(f"\n>>> L1: [PRECISION_FAIL] Failed: {failed} <<<")
    return all_pass


# ---------------------------------------------------------------------------
# L2: Negative tests (invalid inputs — must be rejected, blocking)
# ---------------------------------------------------------------------------


def run_l2_tests():
    """L2 negative tests (blocking): illegal inputs MUST be rejected with the
    expected exception type and an actionable message identifying the root
    cause. l2_s_lt_chunk verifies S < chunk_size padding behavior (kernel
    must run correctly; exceptions FAIL)."""
    print("=== L2 Tests ===")

    def _try_chunk_size_0():
        chunk_o(1, 64, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 0, True, 64, 64)

    def _try_chunk_size_33():
        chunk_o(1, 64, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 33, True, 64, 64)

    def _try_chunk_size_40():
        chunk_o(1, 128, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 40, True, 64, 64)

    def _try_chunk_size_8():
        chunk_o(1, 64, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 8, True, 64, 64)

    def _try_block_dk_100():
        chunk_o(1, 128, 1, 100, 64, "bfloat16", "bfloat16", "float32", "float32", 64, True, 100, 64)

    def _try_block_dv_100():
        chunk_o(1, 128, 1, 64, 100, "bfloat16", "bfloat16", "float32", "float32", 64, True, 64, 100)

    def _try_core_num_0():
        chunk_o(1, 128, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 64, True, 64, 64, 0)

    def _try_cpt_0():
        chunk_o(1, 128, 1, 64, 64, "bfloat16", "bfloat16", "float32", "float32", 64, True, 64, 64, 20, 0)

    def _try_accum_fp16():
        chunk_o(1, 128, 1, 64, 64, "bfloat16", "bfloat16", "float16", "float32", 64, True, 64, 64)

    def _try_gate_bf16():
        chunk_o(1, 128, 1, 64, 64, "bfloat16", "bfloat16", "float32", "bfloat16", 64, True, 64, 64)

    results = [
        _run_exception("l2_chunk_size_0", _try_chunk_size_0, ValueError, "positive multiple of 16"),
        _run_and_check(1, 32, 1, 128, 128, 64, True, 128, 128, "l2_s_lt_chunk", level="L1"),
        _run_exception("l2_chunk_size_33", _try_chunk_size_33, ValueError, "must be a multiple of 16"),
        _run_exception("l2_chunk_size_40", _try_chunk_size_40, ValueError, "must be a multiple of 16"),
        _run_exception("l2_chunk_size_8", _try_chunk_size_8, ValueError, "must be a multiple of 16"),
        _run_exception("l2_block_dk_100", _try_block_dk_100, ValueError, "multiples of 16"),
        _run_exception("l2_block_dv_100", _try_block_dv_100, ValueError, "multiples of 16"),
        _run_exception("l2_core_num_0", _try_core_num_0, ValueError, "core_num must be a positive integer"),
        _run_exception("l2_cpt_0", _try_cpt_0, ValueError, "chunks_per_tile must be a positive integer"),
        _run_exception("l2_accum_fp16", _try_accum_fp16, ValueError, "accum_dtype must be 'float32'"),
        _run_exception("l2_gate_bf16", _try_gate_bf16, ValueError, "gate_dtype must be 'float32'"),
    ]
    all_pass = all(results)
    if all_pass:
        print("\n>>> L2: [BOUNDARY_PASS] All L2 tests passed <<<")
    else:
        print("\n>>> L2: [BOUNDARY_FAIL] Some L2 tests failed <<<")
    return all_pass


# ---------------------------------------------------------------------------
# Boundary: Special values (zero/inf/nan/extreme)
# ---------------------------------------------------------------------------


def run_boundary_tests():
    """Boundary special-value tests — non-blocking."""
    results = []
    scale = 128**-0.5
    B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 64
    BS = math.ceil(S / cs)
    S_pad = BS * cs

    kernel = chunk_o(B, S_pad, H, DK, DV, "bfloat16", "bfloat16", "float32", "float32", cs, True, 128, 128)

    def _make_gt(G_cpu):
        """Create G and G_T from a CPU tensor."""
        return G_cpu.npu(), G_cpu.permute(0, 2, 1).contiguous().npu()

    print("=== Boundary: Zero inputs ===")
    Q = torch.zeros(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    K = torch.zeros(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    V = torch.zeros(B, H, S_pad, DV, dtype=torch.bfloat16, device="cpu").npu()
    HIDDEN = torch.zeros(B, H, BS, DK, DV, dtype=torch.bfloat16, device="cpu").npu()
    G_cpu = torch.zeros(B, S_pad, H, dtype=torch.float32, device="cpu")
    G, G_T = _make_gt(G_cpu)
    try:
        O_act = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, scale, cs, True)
        p, r, m = check_precision(O_act, O_gold, "bfloat16")
        tag = "BOUNDARY_PASS" if p else "BOUNDARY_WARN"
        print(f"  [bnd_zero] [{tag}] ratio={r:.4f} max_abs={m:.3e}")
        results.append(("bnd_zero", p))
    except Exception as e:
        print(f"  [bnd_zero] ERROR: {e}")
        results.append(("bnd_zero", False))

    print("\n=== Boundary: Large values ===")
    lv = 1e3
    Q = torch.full((B, H, S_pad, DK), lv, dtype=torch.bfloat16, device="cpu").npu()
    K = torch.full((B, H, S_pad, DK), lv, dtype=torch.bfloat16, device="cpu").npu()
    V = torch.full((B, H, S_pad, DV), lv, dtype=torch.bfloat16, device="cpu").npu()
    HIDDEN = torch.full((B, H, BS, DK, DV), lv, dtype=torch.bfloat16, device="cpu").npu()
    G_cpu = torch.full((B, S_pad, H), 0.5, dtype=torch.float32, device="cpu")
    G, G_T = _make_gt(G_cpu)
    try:
        O_act = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, scale, cs, True)
        p, r, m = check_precision(O_act, O_gold, "bfloat16")
        tag = "BOUNDARY_PASS" if p else "BOUNDARY_WARN"
        print(f"  [bnd_large] [{tag}] ratio={r:.4f} max_abs={m:.3e}")
        results.append(("bnd_large", p))
    except Exception as e:
        print(f"  [bnd_large] ERROR: {e}")
        results.append(("bnd_large", False))

    print("\n=== Boundary: Negative G (exp underflow) ===")
    torch.manual_seed(2)
    Q = torch.randn(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    K = torch.randn(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    V = torch.randn(B, H, S_pad, DV, dtype=torch.bfloat16, device="cpu").npu()
    HIDDEN = torch.randn(B, H, BS, DK, DV, dtype=torch.bfloat16, device="cpu").npu()
    G_cpu = torch.full((B, S_pad, H), -20.0, dtype=torch.float32, device="cpu")
    G, G_T = _make_gt(G_cpu)
    try:
        O_act = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, scale, cs, True)
        p, r, m = check_precision(O_act, O_gold, "bfloat16")
        tag = "BOUNDARY_PASS" if p else "BOUNDARY_WARN"
        print(f"  [bnd_neg_g] [{tag}] ratio={r:.4f} max_abs={m:.3e}")
        results.append(("bnd_neg_g", p))
    except Exception as e:
        print(f"  [bnd_neg_g] ERROR: {e}")
        results.append(("bnd_neg_g", False))

    print("\n=== Boundary: Inf in G ===")
    torch.manual_seed(3)
    Q = torch.randn(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    K = torch.randn(B, H, S_pad, DK, dtype=torch.bfloat16, device="cpu").npu()
    V = torch.randn(B, H, S_pad, DV, dtype=torch.bfloat16, device="cpu").npu()
    HIDDEN = torch.randn(B, H, BS, DK, DV, dtype=torch.bfloat16, device="cpu").npu()
    G_cpu = torch.randn(B, S_pad, H, dtype=torch.float32, device="cpu")
    G_cpu[0, 0, 0] = float("inf")
    G, G_T = _make_gt(G_cpu)
    try:
        O_act = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, scale, cs, True)
        p, r, m = check_precision(O_act, O_gold, "bfloat16")
        tag = "BOUNDARY_PASS" if p else "BOUNDARY_WARN"
        print(f"  [bnd_inf] [{tag}] ratio={r:.4f} max_abs={m:.3e}")
        results.append(("bnd_inf", p))
    except Exception as e:
        print(f"  [bnd_inf] ERROR: {e}")
        results.append(("bnd_inf", False))

    print("\n=== Boundary: NaN in G ===")
    G_cpu = torch.randn(B, S_pad, H, dtype=torch.float32, device="cpu")
    G_cpu[0, 10, 0] = float("nan")
    G, G_T = _make_gt(G_cpu)
    try:
        O_act = kernel(Q, K, V, HIDDEN, G_T)
        torch.npu.synchronize()
        O_gold = golden_chunk_o(Q, K, V, HIDDEN, G, scale, cs, True)
        p, r, m = check_precision(O_act, O_gold, "bfloat16")
        tag = "BOUNDARY_PASS" if p else "BOUNDARY_WARN"
        print(f"  [bnd_nan] [{tag}] ratio={r:.4f} max_abs={m:.3e}")
        results.append(("bnd_nan", p))
    except Exception as e:
        print(f"  [bnd_nan] ERROR: {e}")
        results.append(("bnd_nan", False))

    all_pass = all(r for _, r in results)
    if all_pass:
        print("\n>>> Boundary: [BOUNDARY_PASS] All boundary tests passed <<<")
    else:
        print("\n>>> Boundary: [BOUNDARY_WARN] Some boundary tests flagged <<<")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="chunk_o test suite")
    parser.add_argument(
        "--level",
        choices=["l0", "l1", "l2", "boundary", "all"],
        default="l0",
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")

    l0_pass = l1_pass = l2_pass = True
    bnd_info = ""

    if args.level in ("l0", "all"):
        l0_pass = run_l0_tests()

    if args.level in ("l1", "all") and l0_pass:
        l1_pass = run_l1_tests()

    if args.level in ("l2", "all"):
        l2_pass = run_l2_tests()

    if args.level in ("boundary", "all"):
        bnd_pass = run_boundary_tests()
        bnd_info = "Boundary passed" if bnd_pass else "Boundary warnings"

    if args.level == "all":
        blocking_ok = l0_pass and l1_pass and l2_pass
        if blocking_ok:
            print("\n" + "=" * 60)
            print("Test Passed!")
            print("=" * 60)
            if bnd_info:
                print(f"  {bnd_info}")
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("Test FAILED!")
            print("=" * 60)
            sys.exit(1)
    else:
        level_map = {"l0": l0_pass, "l1": l1_pass, "l2": l2_pass}
        if args.level in level_map:
            if level_map[args.level]:
                print("\nTest Passed!")
                sys.exit(0)
            else:
                print("\nTest FAILED!")
                sys.exit(1)


if __name__ == "__main__":
    main()
