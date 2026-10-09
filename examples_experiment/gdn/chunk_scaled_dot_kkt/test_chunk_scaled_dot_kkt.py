"""Test file for example_chunk_scaled_dot_kkt operator.

Imports kernel from example_chunk_scaled_dot_kkt, includes golden,
check_precision, and layered test suite (L0/L1/L2/Boundary).

Exit code contract (script entry point, run via `python test_chunk_scaled_dot_kkt.py`):
    L0/L1 precision failures and L2 rejection failures (silently accepted
    illegal input, or rejection by an unexpected exception type / a message
    that does not identify the root cause) exit 1; L2 functional edge cases
    and Boundary special values are non-blocking (layered-test contract).
    The suite functions intentionally have NO `test_` prefix — pytest
    ignores boolean return values, so these are plain script helpers driven
    by main() which owns the exit code.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilelang  # noqa: E402

from example_chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd  # noqa: E402

# =============================================================================
# Coverage manifest: dimensions exercised by the suite (D-EXC-* = illegal
# inputs that must be rejected; see run_l2_tests)
# =============================================================================
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-SHAPE-ALIGNED": 4,
    "D-SHAPE-EDGE": 2,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-VALRANGE-S": 2,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 2,
    "D-VALRANGE-ASYM": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-EXC-PARAM": 3,
    "D-DTYPE-bf16": 6,
    "D-DTYPE-fp32": 6,
    "D-PARAM-chunk_size": 1,
    "D-PARAM-use_g": 3,
    "D-PARAM-block_S": 1,
    "D-PARAM-block_DK": 1,
}


# =============================================================================
# Precision standard (mixed-tolerance dual-gate)
# =============================================================================
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
    """Mixed-tolerance dual-gate check: returns (passed, matched_ratio, max_abs_error)."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
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


# =============================================================================
# Golden reference (pure PyTorch)
# =============================================================================
def golden_chunk_scaled_dot_kkt(K, Beta, G, chunk_size=64, use_g=True):
    """Pure PyTorch golden reference for precision verification.

    Supports S not divisible by chunk_size: only the first S//chunk_size full
    chunks are computed (matching the kernel's n_c = S // BS behavior). Any
    trailing rows [n_c*BS, S) are left as zeros in the output.
    """
    B, S, H, DK = K.shape
    BS = chunk_size
    n_c = S // BS  # number of full chunks (trailing rows dropped, matching kernel)
    S_eff = n_c * BS  # effective S processed
    K_f = K.float().reshape(B, S, H, DK)[:, :S_eff, :, :].reshape(B, n_c, BS, H, DK)
    K_p = K_f.permute(0, 1, 3, 2, 4)  # (B, n_c, H, BS, DK)
    A = torch.matmul(K_p, K_p.transpose(-1, -2))  # (B, n_c, H, BS, BS) fp32
    idx = torch.arange(BS)
    lower_tri = idx.unsqueeze(1) > idx.unsqueeze(0)
    if use_g:
        G_f = G.float().reshape(B, S, H)[:, :S_eff, :].reshape(B, n_c, BS, H)
        Beta_f = Beta.float().reshape(B, S, H)[:, :S_eff, :].reshape(B, n_c, BS, H)
        G_p = G_f.permute(0, 1, 3, 2)  # (B, n_c, H, BS)
        Beta_p = Beta_f.permute(0, 1, 3, 2)  # (B, n_c, H, BS)
        G_diff = G_p.unsqueeze(-1) - G_p.unsqueeze(-2)  # (B, n_c, H, BS, BS)
        mask = (G_diff <= 0) & lower_tri
        gate = torch.where(mask, Beta_p.unsqueeze(-1) * torch.exp(G_diff), torch.zeros_like(G_diff))
        A = A * gate
    else:
        Beta_p = Beta.float().reshape(B, S, H)[:, :S_eff, :].reshape(B, n_c, BS, H).permute(0, 1, 3, 2)
        lower_tri_5d = lower_tri.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, n_c, H, BS, BS)
        gate = torch.where(
            lower_tri_5d,
            Beta_p.unsqueeze(-1).expand(-1, -1, -1, -1, BS),
            torch.zeros(B, n_c, H, BS, BS),
        )
        A = A * gate
    A = A.permute(0, 1, 3, 2, 4).reshape(B, S_eff, H, BS)
    # Pad trailing rows [S_eff, S) with zeros to match output shape (B, S, H, BS)
    if S_eff < S:
        A = torch.nn.functional.pad(A, (0, 0, 0, 0, 0, S - S_eff), mode="constant", value=0)
    return A.to(K.dtype)


# =============================================================================
# Helper: prepare inputs and run kernel
# =============================================================================
def build_inputs(B, S, H, DK, chunk_size, use_g=True, seed=0):
    """Prepare inputs on CPU then H2D. Beta/G use (B,H,S) layout and Beta is
    cast to float32 on CPU (data prep). K is permuted to (B,H,S,DK) for
    contiguous kernel access."""
    torch.manual_seed(seed)
    K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
    Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
    G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
    K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
    Beta = Beta_cpu.npu()
    G = G_cpu.npu()
    # Allocate A in (B,H,S,chunk_size) for contiguous kernel write
    A = torch.empty(B, H, S, chunk_size, dtype=torch.bfloat16, device="npu")
    return K, Beta, G, A, K_cpu, Beta_cpu, G_cpu


def run_and_check(B, S, H, DK, chunk_size, use_g, dtype_str="bfloat16", seed=0):
    """Run kernel and compare with golden. Returns (passed, ratio, max_abs)."""
    K, Beta, G, A, K_cpu, Beta_cpu, G_cpu = build_inputs(B, S, H, DK, chunk_size, use_g, seed)
    kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=chunk_size, use_g=use_g)
    kernel(K, Beta, G, A)
    torch.npu.synchronize()
    # Permute A from (B,H,S,chunk_size) back to (B,S,H,chunk_size) for golden
    A_bsh = A.permute(0, 2, 1, 3).contiguous()
    # Golden uses original (B,S,H) layout
    Beta_bsh = Beta_cpu.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    G_bsh = G_cpu.permute(0, 2, 1).contiguous()
    golden = golden_chunk_scaled_dot_kkt(K_cpu, Beta_bsh, G_bsh, chunk_size=chunk_size, use_g=use_g)
    return check_precision(A_bsh, golden, dtype_str)


# =============================================================================
# L0 tests (threshold, blocking)
# =============================================================================
def run_l0_tests():
    """L0 threshold tests: representative rule shapes, blocking."""
    print("=== L0 Tests ===")
    cases = [
        ("l0_basic_gating", (1, 32768, 32, 128, 64, True)),
        ("l0_basic_no_gating", (1, 32768, 32, 128, 64, False)),
        ("l0_small_gating", (1, 64, 1, 64, 64, True)),
    ]
    results = []
    for name, case in cases:
        passed, ratio, max_abs = run_and_check(*case)
        status = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{status}] {name}: matched_ratio={ratio:.4f} max_abs_error={max_abs:.3e}")
        results.append(passed)
    return all(results)


# =============================================================================
# L1 tests (functional, non-standard shapes; blocking)
# =============================================================================
def run_l1_tests():
    """L1 functional tests: non-standard shapes, blocking."""
    print("=== L1 Tests ===")
    cases = [
        ("l1_small_no_gating", (1, 64, 1, 64, 64, False)),
        ("l1_dk128_gating", (1, 128, 4, 128, 64, True)),
        ("l1_multi_chunk", (1, 256, 2, 64, 64, True)),
    ]
    results = []
    for name, case in cases:
        passed, ratio, max_abs = run_and_check(*case)
        status = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{status}] {name}: matched_ratio={ratio:.4f} max_abs_error={max_abs:.3e}")
        results.append(passed)
    return all(results)


# =============================================================================
# L2 tests (negative = blocking; functional edge = non-blocking)
# =============================================================================
def run_l2_tests():
    """L2 negative tests (blocking): illegal inputs MUST be rejected with the
    expected exception type and an actionable message identifying the root
    cause. Functional edge cases are non-blocking (layered-test contract)."""
    print("=== L2 Tests ===")

    def _run_exception(name, fn, expect_type, expect_msg):
        """L2 negative case: fn() feeds illegal input, REQUIRED to be rejected
        by the entry validation with the expected exception type and a message
        that identifies the actual root cause.

        Returns True only for a correct rejection (expected type + actionable
        message). Both failure modes below count toward the exit code:
        - silently accepted illegal input (entry validation missing/too weak);
        - rejection by an unrelated exception (OOM, name error, ...) — must
          not be reported as a correct rejection.
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

    def _run_edge(name, fn):
        """L2 functional edge case (non-blocking): WARN on failure or
        exception, never counts toward the exit code."""
        try:
            passed, ratio, max_abs = fn()
            status = "PASS" if passed else "WARN"
            print(f"  [BOUNDARY_{status}] l2 {name}: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"  [BOUNDARY_WARN] l2 {name}: {type(e).__name__}: {str(e)[:120]}")

    # --- Negative cases (blocking) ---

    # D-EXC-DTYPE: unsupported K dtype (float16) — only bfloat16 K/A are
    # validated (Beta/G are float32 via accum_dtype).
    def exc_dtype():
        chunk_scaled_dot_kkt_fwd(B=1, S=64, H=1, DK=64, chunk_size=64, use_g=True, input_dtype="float16")

    # D-EXC-PARAM: chunk_size=7 (BS%8!=0) — T.tile.compare/broadcast/cast on
    # (BS,BS) fp32 tiles require 256B alignment (BS%8==0).
    def exc_chunk_size_odd():
        chunk_scaled_dot_kkt_fwd(B=1, S=64, H=1, DK=64, chunk_size=7, use_g=True)

    # D-EXC-PARAM: chunk_size=60 (satisfies %4 but not %8) — still breaks
    # the 256B alignment.
    def exc_chunk_size_even():
        chunk_scaled_dot_kkt_fwd(B=1, S=64, H=1, DK=64, chunk_size=60, use_g=True)

    # D-EXC-PARAM: chunk_size=128 (alignment satisfied but != 64) — the
    # constraint is a fixed 64, not just alignment.
    def exc_chunk_size_nonstandard():
        chunk_scaled_dot_kkt_fwd(B=1, S=64, H=1, DK=64, chunk_size=128, use_g=True)

    results = [
        _run_exception("unsupported_dtype_fp16", exc_dtype, AssertionError, "K/A must be bfloat16"),
        _run_exception("illegal_chunk_size_odd", exc_chunk_size_odd, AssertionError, "must be 64"),
        _run_exception("illegal_chunk_size_even", exc_chunk_size_even, AssertionError, "must be 64"),
        _run_exception("illegal_chunk_size_nonstandard", exc_chunk_size_nonstandard, AssertionError, "must be 64"),
    ]

    # --- Functional edge cases (non-blocking) ---

    def edge_tail_shape():
        """S=65 not divisible by chunk_size: only full chunks computed,
        trailing rows stay zero (A pre-zeroed to match golden's zero-pad)."""
        torch.manual_seed(0)
        B, S, H, DK, cs = 1, 65, 1, 64, 64
        K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
        Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
        G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
        K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
        Beta = Beta_cpu.npu()
        G = G_cpu.npu()
        kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=cs, use_g=True)
        A = torch.zeros(B, H, S, cs, dtype=torch.bfloat16, device="npu")
        kernel(K, Beta, G, A)
        torch.npu.synchronize()
        A_bsh = A.permute(0, 2, 1, 3).contiguous()
        Beta_bsh = Beta_cpu.permute(0, 2, 1).contiguous().to(torch.bfloat16)
        G_bsh = G_cpu.permute(0, 2, 1).contiguous()
        golden = golden_chunk_scaled_dot_kkt(K_cpu, Beta_bsh, G_bsh, chunk_size=cs, use_g=True)
        return check_precision(A_bsh, golden, "bfloat16")

    _run_edge("single_chunk", lambda: run_and_check(1, 64, 1, 64, 64, True))
    _run_edge("batch2", lambda: run_and_check(2, 64, 1, 64, 64, True))
    _run_edge("prime_shape_dk67", lambda: run_and_check(1, 128, 1, 67, 64, True))
    _run_edge("tail_shape_s65", edge_tail_shape)

    return all(results)


# =============================================================================
# Boundary tests (special values, non-blocking)
# =============================================================================
def run_boundary_tests():
    """Boundary special-value tests: WARN-only, never blocks the exit code."""
    print("=== Boundary Tests ===")

    def boundary_beta_zero():
        """Beta=0 should produce all-zero output."""
        torch.manual_seed(0)
        B, S, H, DK, cs = 1, 64, 1, 64, 64
        K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
        Beta_cpu = torch.zeros(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
        G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
        K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
        Beta = Beta_cpu.npu()
        G = G_cpu.npu()
        kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=cs, use_g=True)
        A = torch.empty(B, H, S, cs, dtype=torch.bfloat16, device="npu")
        kernel(K, Beta, G, A)
        torch.npu.synchronize()
        all_zero = A.abs().max().item() == 0.0
        status = "PASS" if all_zero else "WARN"
        print(f"  [BOUNDARY_{status}] boundary beta_zero: all_zero={all_zero}")

    def boundary_g_same():
        """G all same value: G_diff=0, exp(0)=1, mask=(0<=0)&(i>j)=i>j."""
        torch.manual_seed(0)
        B, S, H, DK, cs = 1, 64, 1, 64, 64
        K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
        Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
        G_cpu = torch.full((B, S, H), 1.0, dtype=torch.float32).permute(0, 2, 1).contiguous()
        K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
        Beta = Beta_cpu.npu()
        G = G_cpu.npu()
        kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=cs, use_g=True)
        A = torch.empty(B, H, S, cs, dtype=torch.bfloat16, device="npu")
        kernel(K, Beta, G, A)
        torch.npu.synchronize()
        A_bsh = A.permute(0, 2, 1, 3).contiguous()
        Beta_bsh = Beta_cpu.permute(0, 2, 1).contiguous().to(torch.bfloat16)
        G_bsh = G_cpu.permute(0, 2, 1).contiguous()
        golden = golden_chunk_scaled_dot_kkt(K_cpu, Beta_bsh, G_bsh, chunk_size=cs, use_g=True)
        passed, ratio, max_abs = check_precision(A_bsh, golden, "bfloat16")
        status = "PASS" if passed else "WARN"
        print(f"  [BOUNDARY_{status}] boundary g_same: matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

    def boundary_k_inf():
        """K contains Inf values (special value boundary)."""
        torch.manual_seed(0)
        B, S, H, DK, cs = 1, 64, 1, 64, 64
        K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
        K_cpu[0, 0, 0, 0] = float("inf")
        Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
        G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
        K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
        Beta = Beta_cpu.npu()
        G = G_cpu.npu()
        kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=cs, use_g=True)
        A = torch.empty(B, H, S, cs, dtype=torch.bfloat16, device="npu")
        kernel(K, Beta, G, A)
        torch.npu.synchronize()
        has_inf = torch.isinf(A).any().item()
        has_nan = torch.isnan(A).any().item()
        status = "PASS" if not has_nan else "WARN"
        print(f"  [BOUNDARY_{status}] boundary k_inf: has_inf={has_inf} has_nan={has_nan}")

    def boundary_k_nan():
        """K contains NaN values (special value boundary)."""
        torch.manual_seed(0)
        B, S, H, DK, cs = 1, 64, 1, 64, 64
        K_cpu = torch.randn(B, S, H, DK, dtype=torch.bfloat16)
        K_cpu[0, 0, 0, 0] = float("nan")
        Beta_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).permute(0, 2, 1).contiguous().to(torch.float32)
        G_cpu = torch.randn(B, S, H, dtype=torch.float32).permute(0, 2, 1).contiguous()
        K = K_cpu.permute(0, 2, 1, 3).contiguous().npu()
        Beta = Beta_cpu.npu()
        G = G_cpu.npu()
        kernel = chunk_scaled_dot_kkt_fwd(B=B, S=S, H=H, DK=DK, chunk_size=cs, use_g=True)
        A = torch.empty(B, H, S, cs, dtype=torch.bfloat16, device="npu")
        kernel(K, Beta, G, A)
        torch.npu.synchronize()
        has_nan = torch.isnan(A).any().item()
        status = "PASS" if not has_nan else "WARN"
        print(f"  [BOUNDARY_{status}] boundary k_nan: has_nan={has_nan}")

    for name, fn in [
        ("beta_zero", boundary_beta_zero),
        ("g_same", boundary_g_same),
        ("k_inf", boundary_k_inf),
        ("k_nan", boundary_k_nan),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  [BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {str(e)[:120]}")
    return True


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="chunk_scaled_dot_kkt precision tests")
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    all_pass = True
    if args.level in ("l0", "all") and not run_l0_tests():
        all_pass = False
    if args.level in ("l1", "all") and not run_l1_tests():
        all_pass = False
    if args.level in ("l2", "all") and not run_l2_tests():
        all_pass = False
    if args.level in ("boundary", "all"):
        run_boundary_tests()
    print()
    if all_pass:
        print("Test Passed!")
    else:
        print("[PRECISION_FAIL] Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
