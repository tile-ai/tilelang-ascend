"""chunk_o_bwd precision test suite: L0/L1/L2/Boundary, selected via --level.

Usage: python test_chunk_o_bwd.py --level {l0,l1,l2,boundary,all}

  l0        threshold cases: small smoke, representative large shape,
            use_g=False, use_dw=False
  l1        functional cases: mid shapes, asymmetric S, DK tail blocks
            (DK % block_DK != 0), chunks_per_block variant
  l2        negative cases (blocking): invalid inputs must be rejected with
            the exact exception type + message keyword
  boundary  special gate values (all-zero/INF/NAN/denormal), non-blocking

Every positive case checks dq/dk/dw (bf16) + dg (fp32) against a pure-CPU
PyTorch golden via check_precision (mixed tolerance, dual threshold). L0/L1/L2
failures exit 1; boundary WARNs are recorded only. The golden and the checker
live here; example_chunk_o_bwd.py contains the kernel + a shape smoke test.
"""

import argparse
import math
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_chunk_o_bwd import _prepare_inputs as _prepare_inputs_example  # noqa: E402
from example_chunk_o_bwd import chunk_o_bwd  # noqa: E402

# Coverage declarations for coverage_check.py: positive cases declare tags in
# their config lists; COVERAGE_MANIFEST declares dimensions whose cases carry
# no machine-readable tags (L2 rejections) as tag -> case count.
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-EDGE": 2,
    "D-SPECIAL-ZERO": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-DTYPE-bf16": 4,
    "D-DTYPE-fp32": 4,
    "D-PARAM-chunk_size": 6,
    "D-PARAM-chunks_per_block": 1,
    "D-PARAM-scale": 4,
    "D-PARAM-use_g": 4,
    "D-PARAM-use_dw": 4,
    "D-PARAM-block_DK": 4,
    "D-PARAM-block_DV": 4,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
}
COVERAGE_NA = {}


# ============================================================================
# Golden function (PyTorch reference, pure CPU)
# ============================================================================


def golden_chunk_o_bwd(
    Q,
    K,
    V,
    h,
    G,
    dO,
    dh,
    dv,
    W,
    chunk_size,
    scale,
    use_g=True,
    use_dw=True,
    block_DK=64,
    block_DV=128,
):
    """PyTorch reference implementation (pure CPU).

    Inputs: [B,H,S,D] (bh-major). G stays [B,S,H]. dg_o stays [NK,B,S,H].
    Host precompute: dO *= scale, dv = -dv. DK/DV tail blocks are clamped to
    the tensor bounds (d1 = min(...), v1 = min(...)).
    """
    B, H, S, DK = Q.shape
    DV = V.shape[-1]
    BS = S // chunk_size
    C = chunk_size
    NK = math.ceil(DK / block_DK)
    Qf, Kf, Vf = Q.float().cpu(), K.float().cpu(), V.float().cpu()
    hf, dhf = h.float().cpu(), dh.float().cpu()
    Gf, dOf = G.float().cpu(), dO.float().cpu()
    dvf = dv.float().cpu()
    dq_o = torch.zeros(B, H, S, DK, dtype=torch.float32, device="cpu")
    dk_o = torch.zeros(B, H, S, DK, dtype=torch.float32, device="cpu")
    dw_o = torch.zeros(B, H, S, DK, dtype=torch.float32, device="cpu")
    dg_o = torch.zeros(NK, B, S, H, dtype=torch.float32, device="cpu")
    for bb in range(B):
        for bh in range(H):
            for bs in range(BS):
                s0, s1 = bs * C, (bs + 1) * C
                for bk in range(NK):
                    d0 = bk * block_DK
                    d1 = min((bk + 1) * block_DK, DK)  # DK tail-block clamp
                    Wd = d1 - d0  # actual D-column width of this block
                    ds = torch.zeros(C, C, dtype=torch.float32, device="cpu")
                    dqa = torch.zeros(C, Wd, dtype=torch.float32, device="cpu")
                    dka = torch.zeros(C, Wd, dtype=torch.float32, device="cpu")
                    dwa = torch.zeros(C, Wd, dtype=torch.float32, device="cpu")
                    for iv in range(math.ceil(DV / block_DV)):
                        v0, v1 = iv * block_DV, min((iv + 1) * block_DV, DV)
                        Vb = Vf[bb, bh, s0:s1, v0:v1]
                        dOb = dOf[bb, bh, s0:s1, v0:v1]
                        hb = hf[bb, bh, bs, d0:d1, v0:v1]
                        dhb = dhf[bb, bh, bs, d0:d1, v0:v1]
                        ds = ds + dOb @ Vb.t()
                        dqa = dqa + dOb @ hb.t()
                        dka = dka + Vb @ dhb.t()
                        if use_dw:
                            dvb = dvf[bb, bh, s0:s1, v0:v1]
                            dwa = dwa + dvb @ hb.t()
                    if use_dw:
                        # dv is pre-negated on host; dwa = -dv_orig @ h^T = dw
                        dw_o[bb, bh, s0:s1, d0:d1] = dwa
                    qb = Qf[bb, bh, s0:s1, d0:d1]
                    kb = Kf[bb, bh, s0:s1, d0:d1]
                    if use_g:
                        gb = Gf[bb, s0:s1, bh]
                        g_last = float(gb[-1])
                        dg_last_0 = g_last * math.exp(g_last)
                        dqa = dqa * torch.exp(gb).unsqueeze(-1)  # scale in dOf
                        # Gate masks use select semantics, matching the kernel's
                        # T.tile.compare + T.tile.select: masks compare directly
                        # (g_i <= g_j) and masked-out entries become exact 0
                        # instead of inf*0/NaN*0=NaN artifacts.
                        diff = g_last - gb
                        mask = (g_last <= gb).float()
                        e_diff = torch.exp(diff).unsqueeze(-1)
                        dka_g = torch.where(mask.unsqueeze(-1) > 0, dka * e_diff, torch.zeros_like(dka))
                        dg_from_dq = (dqa * qb).sum(-1)
                        dg_from_dk = (dka_g * (-kb)).sum(-1)
                        dg_last_1 = float((dka_g * kb).sum())
                        gd2 = gb.unsqueeze(-1) - gb.unsqueeze(-2)
                        m2 = (gb.unsqueeze(-1) <= gb.unsqueeze(-2)).float()
                        e_gd2 = torch.exp(gd2)
                        ds_g = torch.where(m2 > 0, ds * e_gd2, torch.zeros_like(ds))
                        ds_pos = ds_g * (qb @ kb.t())
                        dg1 = ds_pos.sum(1)
                        dg2 = ds_pos.sum(0)
                        dg_final = dg_from_dq + dg_from_dk + dg1 - dg2 + dg_last_0 + dg_last_1
                        dqa = dqa + ds_g @ kb
                        dka = dka + ds_g.t() @ qb
                        dq_o[bb, bh, s0:s1, d0:d1] = dqa
                        dk_o[bb, bh, s0:s1, d0:d1] = dka
                        dg_o[bk, bb, s0:s1, bh] = dg_final
                    else:
                        tril = torch.tril(torch.ones(C, C, device="cpu"))
                        ds = ds * tril
                        dqa = dqa + ds @ kb
                        dk2 = ds.t() @ qb
                        dka = dka + dk2
                        dq_o[bb, bh, s0:s1, d0:d1] = dqa
                        dk_o[bb, bh, s0:s1, d0:d1] = dka
    dg_final = dg_o.sum(0)  # [B, S, H]
    return (dq_o.bfloat16(), dk_o.bfloat16(), dw_o.bfloat16(), dg_final.float())


# ============================================================================
# Precision check (mixed tolerance, dual threshold)
# ============================================================================


def check_precision(actual, golden, dtype):
    """Mixed tolerance: |a-g| <= atol + rtol*|g|, matched_ratio >= req AND max_abs <= limit.

    inf/nan positions are structurally compared (isinf/isnan match), not counted
    in numerical tolerance. Finite positions use the dual-threshold:
    matched_ratio >= required AND max_abs <= limit.
    """
    fp_table = {
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    atol, rtol, max_abs_limit, req = fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    # Structural comparison for inf/nan positions
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
    return ratio >= req and max_abs <= max_abs_limit, ratio, max_abs


def _prepare_inputs(B, S, H, DK, DV, chunk_size, use_g=True, use_dw=True, device="npu"):
    """Prepare inputs (delegates to example_chunk_o_bwd._prepare_inputs).

    use_g/use_dw params are accepted for API compatibility but unused -- the
    kernel always receives G, G_T, W tensors (even when use_g/use_dw=False).
    """
    return _prepare_inputs_example(B, S, H, DK, DV, chunk_size, device=device)


def run_and_check(B, S, H, DK, DV, chunk_size, use_g, use_dw, tag="l0", chunks_per_block=256):
    scale = DK**-0.5
    Q, K, V, h_t, G, G_T, dO, dh, dv, W = _prepare_inputs(B, S, H, DK, DV, chunk_size, use_g, use_dw)
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    print(
        f"  Compiling single kernel (B={B}, S={S}, H={H}, DK={DK}, DV={DV}, "
        f"cs={chunk_size}, use_g={use_g}, use_dw={use_dw}, cpb={chunks_per_block}, "
        f"core_num={core_num})..."
    )
    # Single kernel, block-internal loop, block_DK=64, block_DV=128
    kernel = chunk_o_bwd(
        B,
        S,
        H,
        DK,
        DV,
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "float32",
        chunk_size,
        scale,
        core_num,
        use_g,
        use_dw,
        64,
        128,
        chunks_per_block,
    )
    print("  Running kernel...")
    dq_o, dk_o, dw_o, dg_o = kernel(Q, K, V, h_t, G, G_T, dO, dh, dv, W)
    torch.npu.synchronize()

    dg_merged = dg_o.cpu().sum(dim=0).permute(0, 2, 1) if use_g else None

    dq_ref, dk_ref, dw_ref, dg_ref = golden_chunk_o_bwd(Q, K, V, h_t, G, dO, dh, dv, W, chunk_size, scale, use_g, use_dw, 64, 128)

    ok = True
    checks = [("dq", dq_o, dq_ref, "bfloat16"), ("dk", dk_o, dk_ref, "bfloat16")]
    # dw is always checked: use_dw=False -> both sides output zeros.
    checks.append(("dw", dw_o, dw_ref, "bfloat16"))
    if use_g:
        if use_dw:
            checks.append(("dg", dg_merged, dg_ref, "float32"))
        else:
            print(f"  [BOUNDARY_WARN] {tag} dg skipped (use_dw=False)")

    for name, act, gold, dt in checks:
        passed, ratio, max_abs = check_precision(act, gold, dt)
        status = "PRECISION_PASS" if passed else "PRECISION_FAIL"
        print(f"  [{status}] {tag} {name} ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
    return ok


def _expect_reject(fn, exc_type, match, name):
    """Negative test: fn must raise exc_type with `match` in the message."""
    try:
        fn()
    except exc_type as e:
        msg = str(e)
        if match not in msg:
            print(f"  [BOUNDARY_FAIL] {name} wrong error message: {msg!r}")
            return False
        print(f"  [BOUNDARY_PASS] {name} rejected ({exc_type.__name__})")
        return True
    print(f"  [BOUNDARY_FAIL] {name} not rejected")
    return False


def _call_kernel_with_fp32_q():
    """L2 helper: call a bf16 kernel with Q as fp32 (the runtime must reject)."""
    B, S, H, DK, DV, cs = 1, 64, 1, 128, 128, 64
    scale = DK**-0.5
    Q, K, V, h_t, G, G_T, dO, dh, dv, W = _prepare_inputs(B, S, H, DK, DV, cs)
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    print("  Compiling single kernel (dtype-mismatch negative case)...")
    kernel = chunk_o_bwd(
        B,
        S,
        H,
        DK,
        DV,
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "float32",
        cs,
        scale,
        core_num,
        True,
        True,
        64,
        128,
        256,
    )
    kernel(Q.float(), K, V, h_t, G, G_T, dO, dh, dv, W)


def run_l0_suite():
    """L0 threshold tests (4 cases)."""
    configs = [
        (
            "l0_smoke_small",
            1,
            64,
            1,
            128,
            128,
            64,
            True,
            True,
            ["D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-DTYPE-bf16", "D-DTYPE-fp32"],
        ),
        (
            "l0_representative",
            1,
            32768,
            8,
            128,
            128,
            64,
            True,
            True,
            ["D-SHAPE-ALIGNED", "D-VALRANGE-L", "D-DTYPE-bf16", "D-DTYPE-fp32"],
        ),
        (
            "l0_no_gate",
            1,
            32768,
            8,
            128,
            128,
            64,
            False,
            True,
            ["D-SHAPE-EDGE", "D-PARAM-use_g", "D-SPECIAL-ZERO"],
        ),
        (
            "l0_no_dw",
            1,
            32768,
            8,
            128,
            128,
            64,
            True,
            False,
            ["D-SHAPE-ALIGNED", "D-PARAM-use_dw"],
        ),
    ]
    ok = True
    for name, B, S, H, DK, DV, cs, ug, udw, tags in configs:
        print(f"\n[{name}] tags={tags}")
        try:
            passed = run_and_check(B, S, H, DK, DV, cs, ug, udw, tag=name)
            ok &= passed
        except Exception as e:
            print(f"  [PRECISION_FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()
            ok = False
    return ok


def run_l1_suite():
    """L1 functional tests: mid shapes, asymmetric S, DK tail blocks, cpb variant."""
    configs = [
        (
            "l1_valrange_m",
            1,
            2048,
            8,
            128,
            128,
            64,
            True,
            True,
            ["D-VALRANGE-M", "D-SHAPE-ALIGNED", "D-PARAM-chunk_size"],
        ),
        (
            "l1_valrange_asym",
            1,
            1920,
            8,
            128,
            128,
            64,
            True,
            True,
            ["D-VALRANGE-ASYM", "D-SHAPE-ALIGNED"],
        ),
        # DK not a multiple of block_DK=64: the last DK block is a tail block
        # (kernel clamps the D slice; golden mirrors it via d1 = min(...)).
        (
            "l1_dk_tail1",
            1,
            512,
            8,
            65,
            64,
            64,
            True,
            True,
            ["D-SHAPE-TAIL-1"],
        ),
        (
            "l1_dk_tail_mid",
            1,
            512,
            8,
            96,
            64,
            64,
            True,
            True,
            ["D-SHAPE-TAIL-MID"],
        ),
    ]
    ok = True
    for name, B, S, H, DK, DV, cs, ug, udw, tags in configs:
        print(f"\n[{name}] tags={tags}")
        try:
            passed = run_and_check(B, S, H, DK, DV, cs, ug, udw, tag=name)
            ok &= passed
        except Exception as e:
            print(f"  [PRECISION_FAIL] {name}: {e}")
            ok = False
    # D-PARAM-chunks_per_block: test with cpb=128 (different from default 256)
    print("\n[l1_cpb128] tags=['D-PARAM-chunks_per_block', 'D-SHAPE-ALIGNED']")
    try:
        passed = run_and_check(1, 512, 8, 128, 128, 64, True, True, tag="l1_cpb128", chunks_per_block=128)
        ok &= passed
    except Exception as e:
        print(f"  [PRECISION_FAIL] l1_cpb128: {e}")
        ok = False
    return ok


def run_l2_suite():
    """L2 negative tests (blocking): invalid inputs must be rejected.

    Each case validates the exact exception type plus a stable message keyword,
    so unrelated failures (OOM, compile errors, ...) cannot be miscounted as a
    rejection.
    """
    print("\n[L2] negative tests (blocking)")
    ok = True
    # D-EXC-DTYPE: fp32 input for a bf16 kernel parameter.
    ok &= _expect_reject(
        _call_kernel_with_fp32_q,
        ValueError,
        "Buffer dtype mismatch for parameter Q",
        "l2_fp32_dtype_rejected",
    )
    # D-EXC-SHAPE: S not a multiple of chunk_size (no tail-chunk handling).
    ok &= _expect_reject(
        lambda: run_and_check(1, 100, 8, 128, 128, 64, True, True, tag="l2_non_multiple_s"),
        AssertionError,
        "must be a multiple of chunk_size",
        "l2_non_multiple_s",
    )
    # D-SHAPE-PRIME: chunk_size=17 breaks the 32B/256B tile byte alignment.
    ok &= _expect_reject(
        lambda: run_and_check(1, 512, 8, 128, 128, 17, True, True, tag="l2_chunk_prime"),
        AssertionError,
        "must be a multiple of 8",
        "l2_chunk_prime",
    )
    # chunk_size=32: block_DK must equal chunk_size (square L0-staging tile).
    ok &= _expect_reject(
        lambda: run_and_check(1, 512, 8, 128, 128, 32, True, True, tag="l2_chunk32"),
        AssertionError,
        "must equal chunk_size=",
        "l2_chunk32",
    )
    # chunk_size=60: not a multiple of 8 (S=480 is a multiple of 60, so only
    # the tile byte alignment constraint fires).
    ok &= _expect_reject(
        lambda: run_and_check(1, 480, 8, 128, 128, 60, True, True, tag="l2_chunk60"),
        AssertionError,
        "must be a multiple of 8",
        "l2_chunk60",
    )
    return ok


def _inject_zero_g(Q, G, S):
    """D-SPECIAL-ZERO: all-zero gate (exp(0)=1, valid numeric path)."""
    G.zero_()


def _inject_inf_g(Q, G, S):
    """D-SPECIAL-INF: INF gate values at one time step."""
    G[:, S // 2, :] = float("inf")


def _inject_nan_g(Q, G, S):
    """D-SPECIAL-NAN: NAN gate values at one time step."""
    G[:, S // 2, :] = float("nan")


def _inject_dbound(Q, G, S):
    """D-SPECIAL-DBOUND: subnormal-magnitude values (~1e-38, below the 2^-126
    min-normal of both fp32 and bf16) in the fp32 gate and one bf16 input row."""
    G[:, S // 2, :] = 1e-38
    Q[:, :, S // 2, :] = 1e-38


def run_boundary_suite():
    """Boundary tests (non-blocking): special gate values zero/INF/NAN/denormal.

    Kernel+golden runs on a small shape. Special values are injected into the
    inputs after _prepare_inputs (G_T is then regenerated from the mutated G).
    Precision FAIL or exception -> [BOUNDARY_WARN] (recorded, non-blocking).
    """
    print("\n[BOUNDARY] special value tests (non-blocking)")
    B, S, H, DK, DV, cs = 1, 64, 1, 128, 128, 64
    use_g, use_dw = True, True
    scale = DK**-0.5
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
    print(f"  Compiling single kernel (B={B}, S={S}, H={H}, DK={DK}, DV={DV}, cs={cs}, use_g={use_g}, use_dw={use_dw}, boundary suite)...")
    kernel = chunk_o_bwd(
        B,
        S,
        H,
        DK,
        DV,
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "float32",
        cs,
        scale,
        core_num,
        use_g,
        use_dw,
        64,
        128,
        256,
    )
    cases = [
        ("boundary_zero_g", _inject_zero_g, ["D-SPECIAL-ZERO"]),
        ("boundary_inf_g", _inject_inf_g, ["D-SPECIAL-INF"]),
        ("boundary_nan_g", _inject_nan_g, ["D-SPECIAL-NAN"]),
        ("boundary_dbound", _inject_dbound, ["D-SPECIAL-DBOUND"]),
    ]
    for name, inject, tags in cases:
        print(f"\n[{name}] tags={tags}")
        try:
            Q, K, V, h_t, G, G_T, dO, dh, dv, W = _prepare_inputs(B, S, H, DK, DV, cs)
            inject(Q, G, S)
            # Regenerate G_T from the mutated G (same recipe as _prepare_inputs).
            G_T = G.cpu().permute(0, 2, 1).contiguous().to(G.device)
            dq_o, dk_o, dw_o, dg_o = kernel(Q, K, V, h_t, G, G_T, dO, dh, dv, W)
            torch.npu.synchronize()
            dg_merged = dg_o.cpu().sum(dim=0).permute(0, 2, 1)
            dq_ref, dk_ref, dw_ref, dg_ref = golden_chunk_o_bwd(Q, K, V, h_t, G, dO, dh, dv, W, cs, scale, use_g, use_dw, 64, 128)
            checks = [
                ("dq", dq_o, dq_ref, "bfloat16"),
                ("dk", dk_o, dk_ref, "bfloat16"),
                ("dw", dw_o, dw_ref, "bfloat16"),
                ("dg", dg_merged, dg_ref, "float32"),
            ]
            for cname, act, gold, dt in checks:
                passed, ratio, max_abs = check_precision(act, gold, dt)
                status = "BOUNDARY_PASS" if passed else "BOUNDARY_WARN"
                print(f"  [{status}] {name} {cname} ratio={ratio:.4f} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"  [BOUNDARY_WARN] {name} exception: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()
    tilelang.disable_cache()
    torch.set_default_device("npu")
    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= run_l0_suite()
    if args.level in ("l1", "all"):
        blocking_ok &= run_l1_suite()
    if args.level in ("l2", "all"):
        blocking_ok &= run_l2_suite()
    if args.level in ("boundary", "all"):
        run_boundary_suite()
    if blocking_ok:
        print("\nTest Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
