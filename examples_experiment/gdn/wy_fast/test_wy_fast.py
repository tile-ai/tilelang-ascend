"""wy_fast layered tests: L0/L1/L2/Boundary + main(--level).

L0 threshold tests (regular shapes, block-aligned).
L1/L2/Boundary extended based on the real kernel interface and constraints.

Usage:
    python examples_experiment/gdn/wy_fast/test_wy_fast.py --level l0   # L0 precision tests
    python examples_experiment/gdn/wy_fast/test_wy_fast.py --level all  # full suite

Precision: check_precision (mixed-tolerance dual-threshold, bf16: atol=2^-10,
rtol=2^-6, max_abs_limit=1e0, required=0.99). inf/nan positions are structurally
compared, not counted in numeric tolerance. For cases where S is not divisible by
BS (tail chunk dropped), only valid_S = (S//BS)*BS rows are compared.
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel + golden + helpers from example_wy_fast.py in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_wy_fast import (  # noqa: E402
    check_precision,
    get_wy_fast_kernel,
    golden_wy_fast,
    prepare_wy_fast_inputs,
    to_bh_major,
)

# ============================================================================
# Coverage annotations
# ============================================================================
COVERAGE_CATEGORY = "GEMM"

# ============================================================================
# L0 threshold test config (regular shapes, all block-aligned)
# ============================================================================
# (name, B, S, H, DK, DV, chunk_size, tags)
L0_CASES = [
    ("l0_golden_bf16", 1, 32768, 32, 128, 128, 64, ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED"]),
    ("l0_smoke_bf16", 1, 256, 4, 128, 128, 64, ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED"]),
]

# ============================================================================
# L1 functional test config (irregular/tail shapes, covers tail dimensions)
# S-axis non-aligned: WY-fast requires full BSxBS chunks, tail dropped, test compares valid_S
# ============================================================================
L1_CASES = [
    # D-SHAPE-ALIGNED, D-DTYPE-bf16 — baseline aligned (fallback path, 2 DK/DV tiles)
    ("l1_aligned_fallback", 1, 256, 4, 256, 256, 64, (-1, 1), ["D-SHAPE-ALIGNED", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-SHAPE-TAIL-1 — S=65, S%64=1, valid_S=64 (smallest tail, most likely to expose off-by-one)
    ("l1_tail_1", 1, 65, 4, 128, 128, 64, (-1, 1), ["D-SHAPE-TAIL-1", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-SHAPE-TAIL-MID — S=96, S%64=32, valid_S=64 (mid remainder)
    ("l1_tail_mid", 1, 96, 4, 128, 128, 64, (-1, 1), ["D-SHAPE-TAIL-MID", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-SHAPE-PRIME — S=67 (prime), valid_S=64 (completely non-aligned)
    ("l1_prime", 1, 67, 4, 128, 128, 64, (-1, 1), ["D-SHAPE-PRIME", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-SHAPE-EDGE — minimal batch/head, single chunk
    ("l1_edge", 1, 64, 1, 128, 128, 64, (-1, 1), ["D-SHAPE-EDGE", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-PARAM-chunk_size — non-default chunk_size=32 (BS=32, aligned S=256=32*8)
    ("l1_chunk_size_32", 1, 256, 4, 128, 128, 32, (-1, 1), ["D-PARAM-chunk_size", "D-SHAPE-ALIGNED", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
    # D-PARAM-chunk_size — cs=96 (BS in (64,128], num_chunks=4 % 8 != 0):
    # fallback merged fast path.
    (
        "l1_chunk_size_96_fallback",
        1,
        384,
        2,
        128,
        128,
        96,
        (-1, 1),
        ["D-PARAM-chunk_size", "D-SHAPE-ALIGNED", "D-DTYPE-bf16", "D-DTYPE-fp32"],
    ),
    # D-PARAM-chunk_size — cs=128 + num_chunks=8 % 8 == 0: exercises the
    # unrolled->fallback guard (BS <= 64) with the largest allowed BS.
    (
        "l1_chunk_size_128_guard",
        1,
        1024,
        2,
        128,
        128,
        128,
        (-1, 1),
        ["D-PARAM-chunk_size", "D-SHAPE-ALIGNED", "D-DTYPE-bf16", "D-DTYPE-fp32"],
    ),
    # DK non-aligned tail — DK=144 (128+16, 16-aligned fractal, fallback path with N-tail)
    ("l1_dk_tail", 1, 256, 4, 144, 128, 64, (-1, 1), ["D-SHAPE-TAIL-1", "D-DTYPE-bf16", "D-DTYPE-fp32"]),
]

COVERAGE_MANIFEST = {
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
}

COVERAGE_NA = {}  # all required dimensions covered, no exemptions


# ============================================================================
# L0/L1: blocking layer (precision), failure prints [PRECISION_FAIL] and counts toward exit code
# ============================================================================
def _gen_inputs(B, S, H, DK, DV, chunk_size, seed=1, vrange=(-1, 1), device="npu"):
    """Generate random inputs in given value range."""
    torch.manual_seed(seed)
    lo, hi = vrange
    K = torch.rand(B, S, H, DK, dtype=torch.bfloat16, device=device) * (hi - lo) + lo
    V = torch.rand(B, S, H, DV, dtype=torch.bfloat16, device=device) * (hi - lo) + lo
    Beta = torch.rand(B, S, H, dtype=torch.bfloat16, device=device) * (hi - lo) + lo
    G = torch.rand(B, S, H, dtype=torch.float32, device=device) * (hi - lo) + lo
    A = torch.rand(B, S, H, chunk_size, dtype=torch.bfloat16, device=device) * (hi - lo) + lo
    return K, V, Beta, G, A


def _run_precision(level, name, B, S, H, DK, DV, chunk_size, seed=1, vrange=(-1, 1)):
    """L0/L1 single case: gen inputs -> prepare -> kernel -> golden -> check_precision(W, U).

    On success prints [PRECISION_PASS], on failure prints [PRECISION_FAIL]; returns
    whether both W and U pass. For S not divisible by BS, only valid_S = (S//BS)*BS
    rows are compared (tail chunk dropped)."""
    try:
        K, V, Beta, G, A = _gen_inputs(B, S, H, DK, DV, chunk_size, seed=seed, vrange=vrange)

        KVG = prepare_wy_fast_inputs(K, V, Beta, G, chunk_size)
        A_bh = to_bh_major(A)  # kernel takes bh-major [B, H, S, BS]

        kernel = get_wy_fast_kernel(B, S, H, DK, DV, chunk_size)
        WU = kernel(KVG, A_bh)
        torch.npu.synchronize()
        # Merged output WU [B,H,S,DV+DK]: [0:DV]=U, [DV:]=W (views)
        W = WU[..., DV:]
        U = WU[..., :DV]

        # Kernel outputs are bh-major [B, H, S, D] — permute back to
        # the [B, S, H, D] reference layout (zero-copy view) before compare.
        W = W.permute(0, 2, 1, 3)
        U = U.permute(0, 2, 1, 3)

        W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, chunk_size)

        # For non-aligned S, compare only the valid (full-chunk) region
        BS = chunk_size
        valid_S = (S // BS) * BS
        if valid_S < S:
            W_cmp, W_ref_cmp = W[:, :valid_S], W_ref[:, :valid_S]
            U_cmp, U_ref_cmp = U[:, :valid_S], U_ref[:, :valid_S]
        else:
            W_cmp, W_ref_cmp = W, W_ref
            U_cmp, U_ref_cmp = U, U_ref

        w_passed, w_ratio, w_max = check_precision(W_cmp, W_ref_cmp, "bfloat16")
        u_passed, u_ratio, u_max = check_precision(U_cmp, U_ref_cmp, "bfloat16")

        shape_str = f"B={B},S={S},H={H},DK={DK},DV={DV},cs={chunk_size}"
        tail_str = f" valid_S={valid_S}" if valid_S < S else ""
        print(
            f"W: [PRECISION_{'PASS' if w_passed else 'FAIL'}] {level} {name} "
            f"shape=({shape_str}){tail_str} matched_ratio={w_ratio:.4f} max_abs={w_max:.3e}"
        )
        print(
            f"U: [PRECISION_{'PASS' if u_passed else 'FAIL'}] {level} {name} "
            f"shape=({shape_str}){tail_str} matched_ratio={u_ratio:.4f} max_abs={u_max:.3e}"
        )
        return w_passed and u_passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} {name}: {type(e).__name__}: {e}")
        return False


def test_wy_fast_l0():
    """L0 threshold tests: regular shapes (block-aligned)."""
    ok = True
    for name, B, S, H, DK, DV, cs, _tags in L0_CASES:
        ok &= _run_precision("l0", name, B, S, H, DK, DV, cs)
    return ok


def test_wy_fast_l1():
    """L1 functional tests: parameter combination coverage, incl. S tail/prime/degenerate shapes + DK tail + chunk_size variation."""
    ok = True
    for name, B, S, H, DK, DV, cs, vrange, _tags in L1_CASES:
        ok &= _run_precision("l1", name, B, S, H, DK, DV, cs, vrange=vrange)
    return ok


# ============================================================================
# L2: exception tests (negative, non-blocking) — illegal inputs should be rejected
# ============================================================================
def _run_exception(name, fn):
    """L2 single case: fn() feeds illegal input, expected to be rejected by the operator.

    Throws exception -> [BOUNDARY_PASS] (correctly rejected); no throw -> [BOUNDARY_WARN]
    (silently accepted). Both are non-blocking."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input not rejected (silently accepted)")


def test_wy_fast_l2():
    """L2 exception tests: unsupported dtype / illegal shape / illegal chunk_size should be rejected."""

    def _try_bad_dtype():
        # K passed as float32 instead of bfloat16 — should cause dtype mismatch
        B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 64
        K = torch.randn(B, S, H, DK, dtype=torch.float32, device="npu")  # wrong dtype
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        # Assignment never completes if the dtype assert rejects earlier
        _W, _U = WU[..., DV:], WU[..., :DV]

    def _try_bad_shape():
        # DK=17 not 16-aligned — violates fractal alignment, should fail at JIT or runtime
        B, S, H, DK, DV, cs = 1, 256, 4, 17, 128, 64
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        # Assignment never completes if the alignment assert rejects earlier
        _W, _U = WU[..., DV:], WU[..., :DV]

    def _try_bad_chunk_size_small():
        # chunk_size=8 < 16 (mma fractal minimum, A is [BS, BS] with
        # M=K=BS) — must be rejected by the entry assertion.
        B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 8
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        _W, _U = WU[..., DV:], WU[..., :DV]

    def _try_bad_chunk_size_large():
        # chunk_size=256 > 128 (merged-path L0B/L0C capacity) — must be
        # rejected by the entry assertion.
        B, S, H, DK, DV, cs = 1, 512, 4, 128, 128, 256
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        _W, _U = WU[..., DV:], WU[..., :DV]

    def _try_bad_beta_dtype():
        # Beta passed as float32 instead of bfloat16 — must be rejected at
        # the entry assertion.
        B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 64
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.float32, device="npu")  # wrong dtype
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        _W, _U = WU[..., DV:], WU[..., :DV]

    def _try_bad_g_dtype():
        # G passed as bfloat16 instead of float32 — must be rejected at the
        # entry assertion (otherwise silently accepted with reduced exp(G)
        # precision).
        B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 64
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.bfloat16, device="npu")  # wrong dtype
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        _W, _U = WU[..., DV:], WU[..., :DV]

    _run_exception("dtype_mismatch", _try_bad_dtype)
    _run_exception("shape_not_aligned", _try_bad_shape)
    _run_exception("chunk_size_too_small", _try_bad_chunk_size_small)
    _run_exception("chunk_size_too_large", _try_bad_chunk_size_large)
    _run_exception("beta_dtype_mismatch", _try_bad_beta_dtype)
    _run_exception("g_dtype_mismatch", _try_bad_g_dtype)


# ============================================================================
# Boundary: edge/special values (precision, non-blocking) — legal extremes must meet precision standard
# ============================================================================
def _run_boundary(name, dtype, fn):
    """Boundary single case: legal special values (INF/NAN/extreme/zero), fn() returns (W, U, W_ref, U_ref, valid_S).

    Compared by precision standard (check_precision): pass -> [BOUNDARY_PASS]; fail or
    exception -> [BOUNDARY_WARN]."""
    try:
        W, U, W_ref, U_ref, valid_S = fn()
        # For non-aligned S, compare only valid region
        if valid_S is not None and valid_S < W.shape[1]:
            W_c, W_r = W[:, :valid_S], W_ref[:, :valid_S]
            U_c, U_r = U[:, :valid_S], U_ref[:, :valid_S]
        else:
            W_c, W_r = W, W_ref
            U_c, U_r = U, U_ref
        wp, wr, wm = check_precision(W_c, W_r, dtype)
        up, ur, um = check_precision(U_c, U_r, dtype)
        tag = "PASS" if (wp and up) else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} W: ratio={wr:.4f} max_abs={wm:.3e} | U: ratio={ur:.4f} max_abs={um:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {type(e).__name__}: {e}")


def test_wy_fast_boundary():
    """Boundary tests: INF/NAN/extreme/zero (legal special values), compared by precision standard."""
    B, S, H, DK, DV, cs = 1, 256, 4, 128, 128, 64
    valid_S = (S // cs) * cs

    def _case_zero():
        """D-SPECIAL-ZERO: Beta=0 -> V_Beta=0, K_Beta_G=0 -> W=0, U=0"""
        torch.manual_seed(42)
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.zeros(B, S, H, dtype=torch.bfloat16, device="npu")  # all zeros
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        W, U = WU[..., DV:], WU[..., :DV]
        W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, cs)
        # Permute kernel outputs [B, H, S, D] -> [B, S, H, D] for comparison
        return W.permute(0, 2, 1, 3), U.permute(0, 2, 1, 3), W_ref, U_ref, valid_S

    def _case_inf():
        """D-SPECIAL-INF: large G -> exp(G)=inf -> K_Beta_G=inf -> W/U contain inf.
        Uses mixed finite + inf input (finite K/V/A, large G at some positions).
        G shape is [B, S, H] (3D), so inject at G[:, s, h] positions."""
        torch.manual_seed(42)
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.ones(B, S, H, dtype=torch.bfloat16, device="npu")  # all ones
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        # Inject large G at a few positions to trigger exp overflow
        G[:, 0, 0] = 100.0  # exp(100) overflows fp32 -> inf
        G[:, 64, 0] = 90.0  # exp(90) overflows fp32 -> inf
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        W, U = WU[..., DV:], WU[..., :DV]
        W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, cs)
        # Permute kernel outputs [B, H, S, D] -> [B, S, H, D] for comparison
        return W.permute(0, 2, 1, 3), U.permute(0, 2, 1, 3), W_ref, U_ref, valid_S

    def _case_nan():
        """D-SPECIAL-NAN: G contains nan -> exp(nan)=nan -> K_Beta_G=nan -> W/U contain nan.
        Uses mixed finite + nan input (finite K/V/A/Beta, nan G at sparse positions).
        G shape is [B, S, H] (3D), so inject at G[:, s, h] positions."""
        torch.manual_seed(42)
        K = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device="npu")
        V = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="npu")
        Beta = torch.ones(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.randn(B, S, H, dtype=torch.float32, device="npu")
        # Inject nan at sparse positions
        G[:, 0, 0] = float("nan")
        G[:, 128, 1] = float("nan")
        A = torch.randn(B, S, H, cs, dtype=torch.bfloat16, device="npu")
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        W, U = WU[..., DV:], WU[..., :DV]
        W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, cs)
        # Permute kernel outputs [B, H, S, D] -> [B, S, H, D] for comparison
        return W.permute(0, 2, 1, 3), U.permute(0, 2, 1, 3), W_ref, U_ref, valid_S

    def _case_dbound():
        """D-SPECIAL-DBOUND: K values at moderate magnitude to test dtype boundary."""
        torch.manual_seed(42)
        # Use values in [-100, 100] to stay within bf16 range after matmul
        K = torch.rand(B, S, H, DK, dtype=torch.bfloat16, device="npu") * 200 - 100
        V = torch.rand(B, S, H, DV, dtype=torch.bfloat16, device="npu") * 200 - 100
        Beta = torch.ones(B, S, H, dtype=torch.bfloat16, device="npu")
        G = torch.zeros(B, S, H, dtype=torch.float32, device="npu")  # exp(0)=1, no overflow
        A = torch.rand(B, S, H, cs, dtype=torch.bfloat16, device="npu") * 2 - 1
        KVG = prepare_wy_fast_inputs(K, V, Beta, G, cs)
        ker = get_wy_fast_kernel(B, S, H, DK, DV, cs)
        WU = ker(KVG, to_bh_major(A))
        torch.npu.synchronize()
        # Merged output: [0:DV]=U, [DV:]=W (zero-copy views)
        W, U = WU[..., DV:], WU[..., :DV]
        W_ref, U_ref = golden_wy_fast(K, V, Beta, G, A, cs)
        # Permute kernel outputs [B, H, S, D] -> [B, S, H, D] for comparison
        return W.permute(0, 2, 1, 3), U.permute(0, 2, 1, 3), W_ref, U_ref, valid_S

    _run_boundary("zero", "bfloat16", _case_zero)
    _run_boundary("inf", "bfloat16", _case_inf)
    _run_boundary("nan", "bfloat16", _case_nan)
    _run_boundary("dbound", "bfloat16", _case_dbound)


# ============================================================================
# Main: --level dispatch + exit code
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # disable compile cache to avoid stale artifacts
    torch.manual_seed(1)

    blocking_ok = True  # only L0/L1 count toward blocking judgment
    if args.level in ("l0", "all"):
        blocking_ok &= test_wy_fast_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_wy_fast_l1()
    if args.level in ("l2", "all"):
        test_wy_fast_l2()  # L2 negative: non-blocking
    if args.level in ("boundary", "all"):
        test_wy_fast_boundary()  # Boundary precision: non-blocking

    if blocking_ok:
        print("Test Passed!")  # L0/L1 all pass; judged by (exit code + this line)
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
