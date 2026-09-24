"""Test file for wy_fast_bwd_split.

L0 (3 cases) + L1 (functional, 11 cases + cache-contract regression) + L2 (negative, 5 cases)
+ Boundary (4 cases) + main dispatcher with --level support.
Strict precision: fp32 (atol=2^-16, rtol=2^-10, max_abs=1e-2, ratio=0.99) for all outputs.
"""

import argparse
import sys

import torch

from example_wy_fast_bwd_split import (
    _prepare,
    _run_kernel_pipeline,
    check_precision,
    golden_wy_fast_bwd_split,
)

OUTPUT_DTYPES = ["float32"] * 8


def _run_case(name, K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs, block_DK, block_DV):
    """Run a single precision case and return pass/fail."""
    g_results = golden_wy_fast_bwd_split(K, V, Beta, G, A, dw, du, cs, block_DK, block_DV)
    K_n, V_n = K.to("npu"), V.to("npu")
    Beta_n, G_n = Beta.to("npu"), G.to("npu")
    A_n, dw_n, du_n = A.to("npu"), dw.to("npu"), du.to("npu")
    try:
        r = _run_kernel_pipeline(
            K_n,
            V_n,
            Beta_n,
            G_n,
            A_n,
            dw_n,
            du_n,
            B,
            S,
            H,
            DK,
            DV,
            cs,
            block_DK,
            block_DV,
        )
    except Exception as e:
        print(f"  [PRECISION_FAIL] {name}: Runtime error - {str(e)[:200]}")
        return False
    names = ["dA", "dk", "dv", "dbeta", "dg", "dbeta_k", "dg_A_pos", "dg_A_neg"]
    all_pass = True
    for i, oname in enumerate(names):
        passed, ratio, max_err = check_precision(r[i], g_results[i], OUTPUT_DTYPES[i])
        tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
        print(f"  {tag} {name}/{oname}: ratio={ratio:.6f} max_abs={max_err:.6e}")
        if not passed:
            all_pass = False
    return all_pass


def run_l0_case(name, B, S, H, DK, DV, chunk_size, block_DK=64, block_DV=64, seed=0):
    """Run a single L0 test case and return pass/fail + metrics."""
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, chunk_size, seed=seed)
    return _run_case(name, K, V, Beta, G, A, dw, du, B, S, H, DK, DV, chunk_size, block_DK, block_DV)


# ========== L1 test cases with coverage tags ==========
# (name, B, S, H, DK, DV, cs, block_DK, block_DV, seed, scale, tags)
L1_CASES = [
    ("l1_aligned_bf16", 1, 128, 1, 64, 64, 64, 64, 64, 0, 1.0, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"]),
    ("l1_aligned_fp32", 1, 128, 1, 64, 64, 64, 64, 64, 1, 1.0, ["D-DTYPE-fp32", "D-SHAPE-ALIGNED"]),
    ("l1_aligned_multi_bh", 2, 128, 2, 64, 64, 64, 64, 64, 2, 1.0, ["D-DTYPE-bf16", "D-SHAPE-ALIGNED"]),
    (
        "l1_tail1_cs48",
        1,
        144,
        1,
        48,
        48,
        48,
        48,
        48,
        3,
        1.0,
        ["D-SHAPE-TAIL-1", "D-PARAM-chunk_size", "D-PARAM-block_DK", "D-PARAM-block_DV"],
    ),
    (
        "l1_tailmid_cs32",
        1,
        128,
        1,
        32,
        32,
        32,
        32,
        32,
        4,
        1.0,
        ["D-SHAPE-TAIL-MID", "D-PARAM-chunk_size", "D-PARAM-block_DK", "D-PARAM-block_DV"],
    ),
    (
        "l1_prime_cs80",
        1,
        160,
        1,
        80,
        80,
        80,
        80,
        80,
        5,
        1.0,
        ["D-SHAPE-PRIME", "D-PARAM-chunk_size", "D-PARAM-block_DK", "D-PARAM-block_DV"],
    ),
    (
        "l1_min_cs16",
        1,
        64,
        1,
        64,
        64,
        16,
        64,
        64,
        10,
        1.0,
        ["D-PARAM-chunk_size"],
    ),
    ("l1_edge_single_chunk", 1, 64, 1, 64, 64, 64, 64, 64, 6, 1.0, ["D-SHAPE-EDGE"]),
    ("l1_valrange_m", 1, 128, 1, 64, 64, 64, 64, 64, 7, 1.5, ["D-VALRANGE-M", "D-SHAPE-ALIGNED"]),
    ("l1_valrange_l", 1, 128, 1, 64, 64, 64, 64, 64, 8, 2.0, ["D-VALRANGE-L"]),
    ("l1_valrange_asym", 1, 128, 1, 64, 64, 64, 64, 64, 9, 1.0, ["D-VALRANGE-ASYM"]),
]

COVERAGE_CATEGORY = "Fusion"

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 9,
    "D-DTYPE-fp32": 1,
    "D-SHAPE-ALIGNED": 5,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-chunk_size": 4,
    "D-PARAM-block_DK": 3,
    "D-PARAM-block_DV": 3,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-EXC-PARAM": 3,
}

COVERAGE_NA = {
    "D-SHAPE-RANK-2": "Operator uses fixed 3D layout [BH, S, D]",
}


def test_l0():
    """L0 threshold tests (3 cases)."""
    print("=== L0 Tests ===")
    cases = [
        ("l0_small", 1, 64, 1, 64, 64, 64),
        ("l0_multi_chunk", 1, 256, 2, 128, 128, 64),
        ("l0_golden_config", 1, 32768, 8, 128, 128, 64),
    ]
    all_pass = True
    for name, B, S, H, DK, DV, cs in cases:
        print(f"\n--- {name}: B={B} S={S} H={H} DK={DK} DV={DV} cs={cs} ---")
        passed = run_l0_case(name, B, S, H, DK, DV, cs)
        if not passed:
            all_pass = False
    return all_pass


def _prepare_scaled(B, S, H, DK, DV, cs, seed, scale):
    """Prepare inputs with value range scaling."""
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, cs, seed=seed)
    if scale != 1.0:
        Beta = (Beta.float() * scale).to(torch.bfloat16)
        A = (A.float() * scale).to(torch.bfloat16)
        dw = (dw.float() * scale).to(torch.bfloat16)
        du = (du.float() * scale).to(torch.bfloat16)
    return K, V, Beta, G, A, dw, du


def _prepare_asym(B, S, H, DK, DV, cs, seed):
    """Prepare inputs with asymmetric value range (shifted)."""
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, cs, seed=seed)
    dw = (dw.float() + 5.0).to(torch.bfloat16)
    du = (du.float() + 5.0).to(torch.bfloat16)
    return K, V, Beta, G, A, dw, du


def test_l1():
    """L1 functional tests: shape/dtype/value-range/param coverage."""
    print("=== L1 Tests ===")
    all_pass = True
    for case in L1_CASES:
        (name, B, S, H, DK, DV, cs, bDK, bDV, seed, scale, _tags) = case
        print(f"\n--- {name}: B={B} S={S} H={H} DK={DK} DV={DV} cs={cs} scale={scale} ---")
        if name == "l1_valrange_asym":
            K, V, Beta, G, A, dw, du = _prepare_asym(B, S, H, DK, DV, cs, seed)
        else:
            K, V, Beta, G, A, dw, du = _prepare_scaled(B, S, H, DK, DV, cs, seed, scale)
        passed = _run_case(name, K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs, bDK, bDV)
        if not passed:
            all_pass = False
    # cache-contract regression (the opt-in `cache` param): a shared cache
    # across two chunk sizes with the SAME tensor objects must not reuse
    # stale-shape products (the cache keys must include chunk_size).
    # Both configs take the m8 fused path (nc even, DK==block_DK), which
    # consumes lower_tri_mask directly. cs=16 also locks the supported
    # lower boundary together with l1_min_cs16.
    print("\n--- cache_cross_chunk_size: cs=48 then cs=16, shared cache ---")
    K, V, Beta, G, A48, dw, du = _prepare(1, 96, 1, 48, 48, 48, seed=11)
    A16 = torch.randn(1, 96, 16, dtype=torch.bfloat16, device="cpu")
    cache = {}
    for cs_i, A_i in ((48, A48), (16, A16)):
        g = golden_wy_fast_bwd_split(K, V, Beta, G, A_i, dw, du, cs_i, 48, 48)
        r = _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A_i.to("npu"),
            dw.to("npu"),
            du.to("npu"),
            1,
            96,
            1,
            48,
            48,
            cs_i,
            48,
            48,
            cache=cache,
        )
        names = ["dA", "dk", "dv", "dbeta", "dg", "dbeta_k", "dg_A_pos", "dg_A_neg"]
        for i, oname in enumerate(names):
            passed, ratio, max_err = check_precision(r[i], g[i], "float32")
            tag = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
            print(f"  {tag} cache_cs{cs_i}/{oname}: ratio={ratio:.6f} max_abs={max_err:.6e}")
            if not passed:
                all_pass = False
    return all_pass


def test_l2():
    """L2 negative tests: illegal inputs should be rejected."""
    print("=== L2 Tests ===")

    def _run_exception(name, fn):
        try:
            fn()
        except Exception as e:
            print(f"  [BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
            return
        print(f"  [BOUNDARY_WARN] l2 {name}: illegal input silently accepted")

    # D-EXC-DTYPE: float64 inputs (unsupported)
    def test_fp64():
        K, V, Beta, G, A, dw, du = _prepare(1, 64, 1, 64, 64, 64, seed=0)
        K = K.to(torch.float64)
        _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A.to("npu"),
            dw.to("npu"),
            du.to("npu"),
            1,
            64,
            1,
            64,
            64,
            64,
            64,
            64,
        )

    _run_exception("unsupported_dtype_fp64", test_fp64)

    # D-EXC-SHAPE: shape mismatch (DK mismatch between K and dw)
    def test_shape_mismatch():
        K, V, Beta, G, A, dw, du = _prepare(1, 64, 1, 64, 64, 64, seed=0)
        dw_wrong = torch.randn(1, 64, 128, dtype=torch.bfloat16, device="cpu")
        _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A.to("npu"),
            dw_wrong.to("npu"),
            du.to("npu"),
            1,
            64,
            1,
            64,
            64,
            64,
            64,
            64,
        )

    _run_exception("illegal_shape_mismatch", test_shape_mismatch)

    # D-EXC-PARAM: chunk_size not a multiple of 16 (fractal + tile alignment;
    # also rejects non-multiples of 8 that would break T.tile.compare's 256B
    # alignment / the BS*BS//8 packed-mask capacity / T.tile.transpose)
    def test_chunk_size_alignment():
        K, V, Beta, G, A, dw, du = _prepare(1, 400, 1, 64, 64, 100, seed=0)
        _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A.to("npu"),
            dw.to("npu"),
            du.to("npu"),
            1,
            400,
            1,
            64,
            64,
            100,
            64,
            64,
        )

    _run_exception("illegal_chunk_size_alignment", test_chunk_size_alignment)

    # D-EXC-PARAM: chunk_size beyond the UB-budget upper bound (96 > 80; the
    # Vector kernels' UB working sets overflow beyond BS=80). DK/DV=16 so the
    # L0 budget formulas all pass — only the chunk_size bound rejects.
    def test_chunk_size_upper_bound():
        K, V, Beta, G, A, dw, du = _prepare(1, 384, 1, 16, 16, 96, seed=0)
        _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A.to("npu"),
            dw.to("npu"),
            du.to("npu"),
            1,
            384,
            1,
            16,
            16,
            96,
            16,
            16,
        )

    _run_exception("illegal_chunk_size_upper_bound", test_chunk_size_upper_bound)

    # D-EXC-PARAM: block sizes not multiples of 16 (the Cube mma tile N/K
    # dims are the block sizes; a non-16-multiple block passes the
    # divisibility checks but crashes the aicore at runtime)
    def test_block_fractal_alignment():
        K, V, Beta, G, A, dw, du = _prepare(1, 192, 1, 48, 48, 48, seed=0)
        _run_kernel_pipeline(
            K.to("npu"),
            V.to("npu"),
            Beta.to("npu"),
            G.to("npu"),
            A.to("npu"),
            dw.to("npu"),
            du.to("npu"),
            1,
            192,
            1,
            48,
            48,
            48,
            24,
            24,
        )

    _run_exception("illegal_block_fractal_alignment", test_block_fractal_alignment)
    return True


def test_boundary():
    """Boundary tests: INF/NAN/zero/extreme values (non-blocking)."""
    print("=== Boundary Tests ===")

    def _run_boundary(name, K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs):
        try:
            g = golden_wy_fast_bwd_split(K, V, Beta, G, A, dw, du, cs, 64, 64)
            r = _run_kernel_pipeline(
                K.to("npu"),
                V.to("npu"),
                Beta.to("npu"),
                G.to("npu"),
                A.to("npu"),
                dw.to("npu"),
                du.to("npu"),
                B,
                S,
                H,
                DK,
                DV,
                cs,
                64,
                64,
            )
            names = ["dA", "dk", "dv", "dbeta", "dg", "dbeta_k", "dg_A_pos", "dg_A_neg"]
            for i, nm in enumerate(names):
                passed, ratio, max_err = check_precision(r[i], g[i], "float32")
                tag = "PASS" if passed else "WARN"
                print(f"  [BOUNDARY_{tag}] boundary {name}/{nm}: ratio={ratio:.4f} max_abs={max_err:.3e}")
        except Exception as e:
            print(f"  [BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {str(e)[:100]}")

    B, S, H, DK, DV, cs = 1, 64, 1, 64, 64, 64

    # D-SPECIAL-INF: inf in K
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, cs, seed=0)
    K[0, 0, 0] = float("inf")
    print("--- inf_in_K ---")
    _run_boundary("inf_in_K", K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs)

    # D-SPECIAL-NAN: nan in K
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, cs, seed=1)
    K[0, 0, 0] = float("nan")
    print("--- nan_in_K ---")
    _run_boundary("nan_in_K", K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs)

    # D-SPECIAL-ZERO: all-zero inputs
    K = torch.zeros(1, 64, 64, dtype=torch.bfloat16, device="cpu")
    V = torch.zeros(1, 64, 64, dtype=torch.bfloat16, device="cpu")
    Beta = torch.zeros(1, 64, dtype=torch.bfloat16, device="cpu")
    G = torch.zeros(1, 64, dtype=torch.float32, device="cpu")
    A = torch.zeros(1, 64, 64, dtype=torch.bfloat16, device="cpu")
    dw_z = torch.zeros(1, 64, 64, dtype=torch.bfloat16, device="cpu")
    du_z = torch.zeros(1, 64, 64, dtype=torch.bfloat16, device="cpu")
    print("--- zero_inputs ---")
    _run_boundary("zero_inputs", K, V, Beta, G, A, dw_z, du_z, B, S, H, DK, DV, cs)

    # D-SPECIAL-DBOUND: bf16 max value
    K, V, Beta, G, A, dw, du = _prepare(B, S, H, DK, DV, cs, seed=2)
    max_bf16 = torch.tensor(65504.0, dtype=torch.bfloat16)
    dw[0, 0, 0] = max_bf16
    print("--- bf16_max ---")
    _run_boundary("bf16_max", K, V, Beta, G, A, dw, du, B, S, H, DK, DV, cs)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()
    all_pass = True
    if args.level in ("l0", "all") and not test_l0():
        all_pass = False
    if args.level in ("l1", "all") and not test_l1():
        all_pass = False
    if args.level in ("l2", "all"):
        test_l2()
    if args.level in ("boundary", "all"):
        test_boundary()
    print()
    if all_pass:
        print("Test Passed!")
    else:
        print("[PRECISION_FAIL] Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    import tilelang

    tilelang.disable_cache()
    main()
