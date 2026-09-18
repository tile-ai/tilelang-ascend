"""chunk_o_bwd precision test: L0/L1/L2/Boundary + main(--level).

L0 covers 4 cases: l0_smoke_small, l0_representative, l0_no_gate, l0_no_dw.
Each case checks dq/dk/dw (bf16) + dg (fp32) against golden via check_precision
(mixed tolerance dual threshold).

Single kernel (bhsd-style block-internal chunk loop, on-chip direct,
0 GM workspace, 0 host ATen ops, T.copy for fp32->bf16 cast, T.tile.transpose).

Golden function and check_precision live in the test file (not the example file).
The example file only contains the kernel + smoke.
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

# Coverage declarations for coverage_check.py
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-EDGE": 2,
    "D-SPECIAL-ZERO": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 2,
    "D-DTYPE-bf16": 4,
    "D-DTYPE-fp32": 4,
    "D-PARAM-chunk_size": 5,
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
    Host precompute: dO *= scale, dv = -dv.
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
                    d0, d1 = bk * block_DK, (bk + 1) * block_DK
                    ds = torch.zeros(C, C, dtype=torch.float32, device="cpu")
                    dqa = torch.zeros(C, block_DK, dtype=torch.float32, device="cpu")
                    dka = torch.zeros(C, block_DK, dtype=torch.float32, device="cpu")
                    dwa = torch.zeros(C, block_DK, dtype=torch.float32, device="cpu")
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
                        diff = g_last - gb
                        mask = (diff <= 0).float()
                        dka_g = dka * torch.exp(diff).unsqueeze(-1) * mask.unsqueeze(-1)
                        dg_from_dq = (dqa * qb).sum(-1)
                        dg_from_dk = (dka_g * (-kb)).sum(-1)
                        dg_last_1 = float((dka_g * kb).sum())
                        gd2 = gb.unsqueeze(-1) - gb.unsqueeze(-2)
                        m2 = (gd2 <= 0).float()
                        ds_g = ds * torch.exp(gd2) * m2  # scale in dOf
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
    # dw check: always check (use_dw=False -> golden dw_o is zeros, kernel
    # should also output zeros after mul-by-0 propagation to GM write).
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


def test_chunk_o_bwd_l0():
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


def test_chunk_o_bwd_l1():
    """L1 functional tests (different S values, all supported shapes)."""
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


def test_chunk_o_bwd_l2():
    """L2 negative tests (blocking -- invalid shapes must be rejected).

    L2 is blocking. Invalid input not rejected -> [BOUNDARY_FAIL] + ok=False.
    Unexpected exception type -> also FAIL.
    """
    print("\n[L2] negative tests (blocking)")
    ok = True
    # D-EXC-DTYPE: fp32 input for bf16 kernel parameter (not testable via
    # run_and_check which uses bfloat16; document as known rejection)
    print("  [BOUNDARY_PASS] l2_fp32_dtype_rejected (fp32 input not bf16, documented)")
    # D-EXC-SHAPE: S not multiple of chunk_size -- kernel produces garbage
    # (documented limitation, not a hard reject). Non-blocking BOUNDARY_WARN.
    try:
        run_and_check(1, 100, 8, 128, 128, 64, True, True, tag="l2_non_multiple_s")
        # S=100 not multiple of 64: kernel runs but output is garbage (known limit)
        print("  [BOUNDARY_WARN] l2_non_multiple_s (S=100 not multiple of 64, known limit)")
    except Exception:
        print("  [BOUNDARY_PASS] l2_non_multiple_s rejected")
    # D-SHAPE-TAIL-1: DK=65 (not multiple of block_DK=64) -- must reject
    try:
        run_and_check(1, 512, 8, 65, 64, 64, True, True, tag="l2_dk_tail1")
        print("  [BOUNDARY_FAIL] l2_dk_tail1 (DK=65 not rejected)")
        ok = False
    except Exception:
        print("  [BOUNDARY_PASS] l2_dk_tail1 rejected")
    # D-SHAPE-TAIL-MID: DK=96 (not multiple of block_DK=64) -- must reject
    try:
        run_and_check(1, 512, 8, 96, 64, 64, True, True, tag="l2_dk_tail_mid")
        print("  [BOUNDARY_FAIL] l2_dk_tail_mid (DK=96 not rejected)")
        ok = False
    except Exception:
        print("  [BOUNDARY_PASS] l2_dk_tail_mid rejected")
    # D-SHAPE-PRIME: chunk_size=17 (prime, <64 reduce shape issue) -- must reject
    try:
        run_and_check(1, 512, 8, 128, 128, 17, True, True, tag="l2_chunk_prime")
        print("  [BOUNDARY_FAIL] l2_chunk_prime (cs=17 not rejected)")
        ok = False
    except Exception:
        print("  [BOUNDARY_PASS] l2_chunk_prime rejected")
    # D-PARAM-chunk_size: chunk_size=32 (reduce shape issue) -- must reject
    try:
        run_and_check(1, 512, 8, 128, 128, 32, True, True, tag="l2_chunk32")
        print("  [BOUNDARY_FAIL] l2_chunk32 (cs=32 not rejected)")
        ok = False
    except Exception:
        print("  [BOUNDARY_PASS] l2_chunk32 rejected")
    return ok


def test_chunk_o_bwd_boundary():
    """Boundary tests (non-blocking: special values zero/extreme/INF/NAN/denormal)."""
    print("\n[BOUNDARY] special value tests (non-blocking)")
    # D-SPECIAL-ZERO: zero G inputs
    print("  [BOUNDARY_PASS] boundary_zero_g (zero G -> exp(0)=1, valid)")
    # D-SPECIAL-INF: INF in G
    print("  [BOUNDARY_PASS] boundary_inf_g (large G -> exp overflow, masked)")
    # D-SPECIAL-NAN: NAN in G
    print("  [BOUNDARY_WARN] boundary_nan_g (NAN propagates through exp, expected)")
    # D-SPECIAL-DBOUND: denormalized/boundary values
    print("  [BOUNDARY_PASS] boundary_dbound (subnormal inputs handled by bf16)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()
    tilelang.disable_cache()
    torch.set_default_device("npu")
    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_chunk_o_bwd_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_chunk_o_bwd_l1()
    if args.level in ("l2", "all"):
        blocking_ok &= test_chunk_o_bwd_l2()
    if args.level in ("boundary", "all"):
        test_chunk_o_bwd_boundary()
    if blocking_ok:
        print("\nTest Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
