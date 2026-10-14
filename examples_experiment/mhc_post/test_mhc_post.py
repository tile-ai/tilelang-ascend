# ruff: noqa
import argparse
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_mhc_post import (  # noqa: E402
    compute_h_blk,
    mhc_post_kernel,
)


ATOL, RTOL = 1e-2, 5e-3


def golden_mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """PyTorch reference, numerically identical to GPU reference mhc_post_ref.
    post_layer_mix must be UNsqueezed [n, hc, 1]; the kernel takes the
    squeezed [n, hc] form (host squeeze at call time).

    NOTE: GPU ref computes term2 = torch.bmm(comb_res_mix.mT, residual.float()).
    On Ascend, aclnnBatchMatMul rejects M=4 (hc=4) with aicore 507015, so the
    mathematically identical expansion (sum_k A[k,i]*B[k,j], fp32) is used:
        term2[i,j] = (comb.permute(0,2,1) [n,i,k] x residual [n,k,j]).sum(k)
    Pure elementwise ops, same fp32 accumulate order as the kernel.
    """
    term2 = (comb_res_mix.permute(0, 2, 1).unsqueeze(-1) * residual.float().unsqueeze(1)).sum(
        -2
    )  # [n, hc, hc]^T x [n, hc, h] -> [n, hc, h], fp32
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def generate_test_data(n, h, hc=4, device="npu"):
    """Test data generation (same as GPU reference generate_test_data, seed=42).
    post_layer_mix returned as [n, hc, 1] (as generated; kernel input squeezed
    at call time, golden consumes it as-is, matching mhc_post_ref)."""
    torch.random.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc, hc), dtype=torch.float32, device=device)
    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def run_case(n, h, hc=4, device="npu"):
    """Run one case: kernel output vs golden. Returns (out, ref)."""
    h_blk = compute_h_blk(h)
    data = generate_test_data(n, h, hc, device)
    kernel = mhc_post_kernel(n, h, hc, h_blk)
    out = kernel(
        data["comb_res_mix"],
        data["residual"],
        data["post_layer_mix"].squeeze(-1),  # kernel takes [n, hc]
        data["x"],
    )
    ref = golden_mhc_post(**data)  # golden takes [n, hc, 1]
    torch.npu.synchronize()
    return out, ref


# ===========================================================================
# Precision test cases
# ===========================================================================


# L0 cases
L0_CASES = [
    ("l0_smoke", 32, 2560),  # h_blk=512, h_num=5, sub_h_blk=256
    ("l0_typical_h1280", 128, 1280),  # small h regular: h_blk=256, h_num=5
    ("l0_typical_h2560", 256, 2560),  # typical config: h_blk=512, h_num=5
    ("l0_large_h7168", 512, 7168),  # large h regular: h_blk=1024, h_num=7
    ("l0_multi_token", 1024, 2560),  # multi-token regression: block count=1024, wave scheduling
]


REGRESSION = [(4096, 1280), (4096, 2560), (4096, 7168)]

# L1 irregular shapes (7 cases: edge n / small h / min h / prime h / tail h)
L1_CASES = [
    ("l1_edge_n1", 1, 2560),  # single-token edge
    ("l1_small_h512", 4, 512),  # h_num=1
    ("l1_min_h32", 2, 32),  # min h, h_blk=32 -> sub_h_blk=16 (32B row)
    ("l1_irregular_h1344", 8, 1344),  # h_blk=64, h_num=21
    ("l1_irregular_h2176", 3, 2176),  # h_blk=128, h_num=17
    ("l1_irregular_h3328", 16, 3328),  # h_blk=256, h_num=13
    ("l1_irregular_h1536", 6, 1536),  # h_blk=512, h_num=3
]


def _check_one(n, h):
    out, ref = run_case(n, h)
    out_cpu = out.float().cpu()
    ref_cpu = ref.float().cpu()
    max_diff = (out_cpu - ref_cpu).abs().max().item()
    inf_mask = torch.isinf(ref_cpu)
    nan_mask = torch.isnan(ref_cpu)
    normal_mask = ~inf_mask & ~nan_mask
    if inf_mask.any() and not torch.equal(torch.isinf(out_cpu[inf_mask]), torch.isinf(ref_cpu[inf_mask])):
        raise AssertionError("INF structure mismatch")
    if nan_mask.any() and not torch.equal(torch.isnan(out_cpu[nan_mask]), torch.isnan(ref_cpu[nan_mask])):
        raise AssertionError("NAN structure mismatch")
    if normal_mask.any():
        torch.testing.assert_close(out_cpu[normal_mask], ref_cpu[normal_mask], atol=ATOL, rtol=RTOL)
    return max_diff


def test_mhc_post_l0():
    """L0 gate: 5 regular shapes per DESIGN.md §9.2 (block-aligned). Blocking layer."""
    ok = True
    for name, n, h in L0_CASES:
        try:
            max_diff = _check_one(n, h)
            print(f"[PRECISION_PASS] l0 {name} n={n} h={h} max_diff={max_diff:.3e}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l0 {name} n={n} h={h}: {e}")
            ok = False
    assert ok, "L0 gate failed: see [PRECISION_FAIL] lines above"


def test_gpu_reg():
    ok = True
    for n, h in REGRESSION:
        try:
            max_diff = _check_one(n, h)
            print(f"[PRECISION_PASS] gpu_reg n={n} h={h} max_diff={max_diff:.3e}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] gpu_reg n={n} h={h}: {e}")
            ok = False
    assert ok, "gpu_reg failed: see [PRECISION_FAIL] lines above"


def test_mhc_post_l1():
    """L1 functional tests: irregular shapes (h not a multiple of 1024 + edge n). Blocking layer."""
    ok = True
    for name, n, h in L1_CASES:
        try:
            max_diff = _check_one(n, h)
            print(f"[PRECISION_PASS] l1 {name} n={n} h={h} max_diff={max_diff:.3e}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PRECISION_FAIL] l1 {name} n={n} h={h}: {e}")
            ok = False
    assert ok, "L1 failed: see [PRECISION_FAIL] lines above"


# ===========================================================================
# L2 exception tests (non-blocking, only records [BOUNDARY_PASS/WARN])
# ===========================================================================


def _run_exception(name, fn):
    """L2 single case: feed illegal input, expect rejection (non-blocking)."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: not rejected (silently accepted)")


def test_mhc_post_l2():
    """L2 exception tests: wrong dtype / shape should be rejected (non-blocking)."""
    device = "npu"
    n, h, hc = 8, 2560, 4
    h_blk = compute_h_blk(h)
    kernel = mhc_post_kernel(n, h, hc, h_blk)

    def _gen():
        torch.random.manual_seed(42)
        return {
            "x": torch.randn((n, h), dtype=torch.bfloat16, device=device),
            "residual": torch.randn((n, hc, h), dtype=torch.bfloat16, device=device),
            "post_layer_mix": torch.randn((n, hc), dtype=torch.float32, device=device),
            "comb_res_mix": torch.randn((n, hc, hc), dtype=torch.float32, device=device),
        }

    _run_exception(
        "wrong_dtype_residual",
        lambda: kernel(
            _gen()["comb_res_mix"],
            torch.randn((n, hc, h), dtype=torch.float32, device=device),
            _gen()["post_layer_mix"],
            _gen()["x"],
        ),
    )
    _run_exception(
        "wrong_dtype_comb",
        lambda: kernel(
            torch.randn((n, hc, hc), dtype=torch.bfloat16, device=device),
            _gen()["residual"],
            _gen()["post_layer_mix"],
            _gen()["x"],
        ),
    )
    _run_exception(
        "wrong_shape_x",
        lambda: kernel(
            _gen()["comb_res_mix"],
            _gen()["residual"],
            _gen()["post_layer_mix"],
            torch.randn((n, h + 8), dtype=torch.bfloat16, device=device),
        ),
    )
    _run_exception(
        "wrong_shape_residual_hc",
        lambda: kernel(
            _gen()["comb_res_mix"],
            torch.randn((n, 3, h), dtype=torch.bfloat16, device=device),
            _gen()["post_layer_mix"],
            _gen()["x"],
        ),
    )
    _run_exception(
        "wrong_shape_n",
        lambda: kernel(
            _gen()["comb_res_mix"],
            torch.randn((n + 4, hc, h), dtype=torch.bfloat16, device=device),
            _gen()["post_layer_mix"],
            _gen()["x"],
        ),
    )


# ===========================================================================
# Boundary special-value tests (non-blocking, only records [BOUNDARY_PASS/WARN])
# ===========================================================================


def _check_boundary(out, ref, expect_finite=False):
    """Structural + numeric boundary check. Returns (ok, detail)."""
    a = out.float().cpu()
    g = ref.float().cpu()
    special = ~torch.isfinite(g)
    if special.any():
        # inf/nan positions must structurally match golden
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])):
            return False, "nan-structure mismatch"
        if not torch.equal(torch.isinf(a[special]), torch.isinf(g[special])):
            return False, "inf-structure mismatch"
    m = torch.isfinite(g)
    if m.sum().item() > 0:
        abs_diff = (a[m] - g[m]).abs()
        tol = ATOL + RTOL * g[m].abs()
        ratio = (abs_diff <= tol).float().mean().item()
        if ratio < 0.99:
            return False, f"matched_ratio={ratio:.4f}"
    if expect_finite and not torch.isfinite(a).all():
        return False, "unexpected non-finite in output"
    return True, "ok"


def _run_boundary(level, name, fn, expect_finite=False):
    """Boundary single case: [BOUNDARY_PASS] / [BOUNDARY_WARN] (non-blocking)."""
    try:
        out, ref = fn()
        ok, detail = _check_boundary(out, ref, expect_finite=expect_finite)
        marker = "[BOUNDARY_PASS]" if ok else "[BOUNDARY_WARN]"
        print(f"{marker} {level} {name}: {detail}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] {level} {name}: {type(e).__name__}: {e}")


def test_mhc_post_boundary():
    """Boundary tests: zero / inf / nan / bf16 extreme / negative mix (non-blocking)."""
    device = "npu"
    n, h, hc = 8, 2560, 4
    h_blk = compute_h_blk(h)
    kernel = mhc_post_kernel(n, h, hc, h_blk)

    def _make(x=None, residual=None, post_layer_mix=None, comb_res_mix=None):
        torch.random.manual_seed(42)
        d = {
            "x": torch.randn((n, h), dtype=torch.bfloat16, device=device),
            "residual": torch.randn((n, hc, h), dtype=torch.bfloat16, device=device),
            "post_layer_mix": torch.randn((n, hc, 1), dtype=torch.float32, device=device),
            "comb_res_mix": torch.randn((n, hc, hc), dtype=torch.float32, device=device),
        }
        if x is not None:
            d["x"] = x
        if residual is not None:
            d["residual"] = residual
        if post_layer_mix is not None:
            d["post_layer_mix"] = post_layer_mix
        if comb_res_mix is not None:
            d["comb_res_mix"] = comb_res_mix
        out = kernel(d["comb_res_mix"], d["residual"], d["post_layer_mix"].squeeze(-1), d["x"])
        ref = golden_mhc_post(**d)
        torch.npu.synchronize()
        return out, ref

    # all-zero input -> exact zero output
    _run_boundary(
        "boundary",
        "zero",
        lambda: _make(
            x=torch.zeros((n, h), dtype=torch.bfloat16, device=device),
            residual=torch.zeros((n, hc, h), dtype=torch.bfloat16, device=device),
            post_layer_mix=torch.zeros((n, hc, 1), dtype=torch.float32, device=device),
            comb_res_mix=torch.zeros((n, hc, hc), dtype=torch.float32, device=device),
        ),
        expect_finite=True,
    )
    # bf16 max (within bf16 range, finite) -> structural + tolerance match
    _run_boundary(
        "boundary",
        "bf16_max",
        lambda: _make(
            x=torch.full((n, h), 60000.0, dtype=torch.bfloat16, device=device),
            residual=torch.full((n, hc, h), 60000.0, dtype=torch.bfloat16, device=device),
            post_layer_mix=torch.full((n, hc, 1), 1000.0, dtype=torch.float32, device=device),
            comb_res_mix=torch.randn((n, hc, hc), dtype=torch.float32, device=device),
        ),
        expect_finite=True,
    )
    # all NaN input -> NaN propagates structurally
    _run_boundary(
        "boundary",
        "nan",
        lambda: _make(x=torch.full((n, h), float("nan"), dtype=torch.bfloat16, device=device)),
    )
    # all Inf input -> Inf propagates structurally
    _run_boundary(
        "boundary",
        "inf",
        lambda: _make(x=torch.full((n, h), float("inf"), dtype=torch.bfloat16, device=device)),
    )
    # negative post_layer_mix -> sign correctness under tolerance
    _run_boundary(
        "boundary",
        "negative_mix",
        lambda: _make(post_layer_mix=-torch.ones((n, hc, 1), dtype=torch.float32, device=device) * 3.0),
        expect_finite=True,
    )
    # extreme outer product: c up to +-1e3, x up to +-1e3 -> finite fp32 accum
    _run_boundary(
        "boundary",
        "extreme_outer",
        lambda: _make(
            x=torch.randn((n, h), dtype=torch.bfloat16, device=device) * 1000.0,
            post_layer_mix=torch.randn((n, hc, 1), dtype=torch.float32, device=device) * 1000.0,
        ),
        expect_finite=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="mhc_post precision test (Ascend NPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_mhc_post.py                         # L0 + GPU regression (default)\n"
            "  python test_mhc_post.py --level l0              # run L0 gate only\n"
            "  python test_mhc_post.py --level all             # full layered tests (L0/L1/L2/Boundary + GPU regression)\n"
        ),
    )
    parser.add_argument(
        "--level",
        default="default",
        choices=["default", "l0", "l1", "l2", "boundary", "all"],
        help=("Test level. 'default' = L0 + GPU regression (n=4096, h in {1280,2560,7168}). 'all' = L0/L1/L2/Boundary + GPU regression."),
    )
    args = parser.parse_args()

    tilelang.disable_cache()  # avoid stale compile artifacts
    torch.set_default_device("npu")
    torch.manual_seed(0)

    # --level default: L0 gate + GPU source regression
    if args.level == "default":
        print("=" * 70)
        print("Part 1/2: L0 gate (DESIGN §9.2)")
        print("=" * 70)
        test_mhc_post_l0()  # raises AssertionError on failure
        print()
        print("=" * 70)
        print("Part 2/2: GPU source regression test(n=4096, h=1280/2560/7168)")
        print("=" * 70)
        test_gpu_reg()  # raises AssertionError on failure
        print("\nTest Passed!")
        return

    # Other --level values: layered tests
    # Blocking layers (L0/L1/gpu_reg) raise AssertionError on failure.
    if args.level == "l0":
        test_mhc_post_l0()
    if args.level == "all":
        test_mhc_post_l0()
        test_gpu_reg()  # full layered + regression
        test_mhc_post_l1()
        test_mhc_post_l2()  # non-blocking, record only
        test_mhc_post_boundary()  # non-blocking, record only
    if args.level == "l1":
        test_mhc_post_l1()
    if args.level == "l2":
        test_mhc_post_l2()  # non-blocking, record only
    if args.level == "boundary":
        test_mhc_post_boundary()  # non-blocking, record only

    print("\nTest Passed!")


if __name__ == "__main__":
    main()
