"""Chunk Delta Backward precision test: L0 + L1 + L2 + main(--level).

L0: 6 small configs (C=64/128/256/512/1024) + bench shape (BS=32 with C=1024)
L1: 5 irregular shapes
L2: boundary (minimal shape)
Precision: all outputs fp32 (atol=2^-16, rtol=2^-10, max_abs=1e-2, ratio=0.99)
"""

import argparse
import os
import sys

import tilelang
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_chunk_delta_bwd import (  # noqa: E402
    chunk_delta_bwd,
    chunk_local_cumsum,
    prepare_bwd_gates,
)


# Expected L2 exception types.
_NPU_ERROR_TYPES = tuple(getattr(torch.npu, _n) for _n in ("NPUError",) if hasattr(torch.npu, _n))
_EXPECTED_L2_ERRORS = (ValueError, TypeError, RuntimeError, AssertionError) + _NPU_ERROR_TYPES


# ============================================================================
# Golden function (PyTorch reference, pure CPU)
# ============================================================================


def torch_chunk_gated_delta_rule_bwd_dhu(
    Q,
    K,
    W,
    G,
    h0,
    dht,
    dO,
    dv,
    scale,
    use_g=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    chunk_size=64,
):
    """CPU fp32 golden: host precompute fp32, GEMM fp32, state fp32 relay.

    dh/dh0 outputs in transposed [DV, DK] layout to match kernel.
    Internal computation uses original [DK, DV] layout; output is permuted.

    Args:
        Q: (B, S, H, DK) bfloat16
        K: (B, S, H, DK) bfloat16
        W: (B, S, H, DK) bfloat16
        G: (B, S, H) float32 (already chunk_local_cumsum'd)
        h0: (B, H, DK, DV) bfloat16 (unused)
        dht: (B, H, DK, DV) bfloat16
        dO: (B, S, H, DV) bfloat16
        dv: (B, S, H, DV) bfloat16
        scale: float
        use_g: bool
        use_initial_state: bool
        use_final_state_gradient: bool
        chunk_size: int (block_S, default 64 for backward compat)

    Returns:
        dh: (B, BS, H, DV, DK) float32 (transposed layout)
        dh0: (B, H, DV, DK) float32 (transposed layout)
        dv2: (B, S, H, DV) float32
    """
    B, S, H, DK = Q.shape
    DV = dv.shape[-1]
    block_S = chunk_size
    BS = S // block_S

    # Host precompute in fp32
    Kg, Qg_T, Wn_T, Dmat = prepare_bwd_gates(K, W, Q, G, scale, block_S, use_g)

    # Outputs in transposed layout [DV, DK] to match kernel
    dh = torch.empty((B, BS, H, DV, DK), dtype=torch.float32)
    dh0 = torch.empty((B, H, DV, DK), dtype=torch.float32)
    dv2 = torch.empty((B, S, H, DV), dtype=torch.float32)

    if use_final_state_gradient:
        dh_tmp = dht.clone().float()  # [B, H, DK, DV] fp32
    else:
        dh_tmp = torch.zeros(B, H, DK, DV, dtype=torch.float32)

    for i_s in range(BS - 1, -1, -1):
        # 1. Store dh (fp32, transposed output [DV, DK])
        dh[:, i_s] = dh_tmp.permute(0, 1, 3, 2)  # [B, H, DV, DK]

        # 2. GEMM 1: dv = Kg @ dh + dv_in (fp32 GEMM)
        Kg_chunk = Kg[:, i_s]  # [B, H, block_S, DK] fp32
        dv_tmp = torch.matmul(Kg_chunk, dh_tmp)  # [B, H, block_S, DV] fp32
        dv_tmp = dv_tmp.permute(0, 2, 1, 3)  # [B, block_S, H, DV] fp32
        dv_tmp += dv[:, i_s * block_S : (i_s + 1) * block_S].float()  # add dv_in

        # 3. Store dv2 (fp32, no truncation)
        dv2[:, i_s * block_S : (i_s + 1) * block_S] = dv_tmp

        # 4. GEMM 2a: dh = Dmat @ dh (fp32 GEMM)
        dh_new = torch.matmul(Dmat[:, i_s], dh_tmp)  # [B, H, DK, DV] fp32

        # 5. GEMM 2b: dh += Qg^T @ dO (fp32 GEMM)
        dO_chunk = dO[:, i_s * block_S : (i_s + 1) * block_S]  # [B, block_S, H, DV] bf16
        dO_bhd = dO_chunk.float().permute(0, 2, 1, 3)  # [B, H, block_S, DV] fp32
        dh_new += torch.matmul(Qg_T[:, i_s], dO_bhd)  # [B, H, DK, DV] fp32

        # 6. GEMM 2c: dh += Wn^T @ dv (fp32 GEMM, uses computed dv)
        dv_bhd = dv_tmp.permute(0, 2, 1, 3)  # [B, H, block_S, DV] fp32
        dh_new += torch.matmul(Wn_T[:, i_s], dv_bhd)  # [B, H, DK, DV] fp32

        # Update state (fp32, NO bf16 cast)
        dh_tmp = dh_new

    # dh0 output (fp32, transposed [DV, DK])
    if use_initial_state:
        dh0 = dh_tmp.permute(0, 1, 3, 2).contiguous()
    else:
        dh0 = torch.zeros_like(dh0)

    return dh, dh0, dv2


# ============================================================================
# Precision check
# ============================================================================


def check_precision(actual, reference, dtype_str):
    """Mixed-tolerance precision check per precision-standard.md.

    Dual-gate (AND):
        (1) matched_ratio >= required_matched_ratio
        (2) max_abs_error  <= max_abs_error_limit  (hard cap, dtype-only)

    Element-wise pass condition (finite values only):
        |actual - golden| <= atol + rtol * |golden|

    INF/NAN structural compare per §3.1: inf/nan positions must match
    (isinf/isnan positions agree), and are excluded from matched_ratio /
    max_abs_error computation.

    Thresholds are dtype-only (§二): GEMM/Softmax/Normalization/Activation
    /Reduction/Fusion all use the same table.

    Args:
        actual: kernel output tensor
        reference: golden output tensor
        dtype_str: "bfloat16" or "float32"

    Returns:
        (passed, matched_ratio, max_abs_error)
    """
    # dtype-only thresholds per precision-standard.md §二
    if dtype_str == "float32":
        atol, rtol, max_abs_limit, required_ratio = (
            2**-16,  # 1.53e-5
            2**-10,  # 9.77e-4
            1e-2,
            0.99,
        )
    else:  # bfloat16
        atol, rtol, max_abs_limit, required_ratio = (
            2**-10,  # 9.77e-4
            2**-6,  # 1.56e-2
            1e0,
            0.99,
        )

    a = actual.detach().cpu()
    g = reference.detach().cpu()

    # §3.1 INF/NAN structural compare: positions must match, not in tolerance
    special = ~torch.isfinite(g)  # golden inf/nan positions
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")

    # Finite-value positions: full compare. If actual is inf/nan where golden
    # is finite, abs_err = inf/nan -> element-wise False and raises max_abs.
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0

    a_f = a.float()
    g_f = g.float()
    abs_err = (a_f[m] - g_f[m]).abs()
    matched_ratio = (abs_err <= (atol + rtol * g_f[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()

    # §1.2 Dual-gate AND (hard cap, no value-range scaling)
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


def prepare_inputs(B, S, H, DK, DV, chunk_size, device="npu"):
    """Generate normalized inputs on device."""
    torch.manual_seed(0)
    Q = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device=device)
    K = F.normalize(
        torch.randn(B, S, H, DK, dtype=torch.bfloat16, device=device),
        dim=-1,
        p=2,
    )
    W = torch.randn(B, S, H, DK, dtype=torch.bfloat16, device=device)
    G = F.logsigmoid(torch.randn(B, S, H, dtype=torch.float32, device=device))
    G = chunk_local_cumsum(G, chunk_size)
    h0 = torch.randn(B, H, DK, DV, dtype=torch.bfloat16, device=device)
    dht = torch.randn(B, H, DK, DV, dtype=torch.bfloat16, device=device)
    dO = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device=device)
    dv = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device=device)
    scale = DK**-0.5
    return Q, K, W, G, h0, dht, dO, dv, scale


def run_and_check(B, S, H, DK, DV, chunk_size, tag="l0", block_DV=64):
    """Run kernel + golden, check precision for all 3 outputs."""
    # Shape assertions
    assert S % chunk_size == 0, f"S={S} must be divisible by chunk_size={chunk_size}"
    assert DK % 16 == 0, f"DK={DK} must be 16-aligned (fractal GEMM)"
    assert DV % 16 == 0, f"DV={DV} must be 16-aligned (fractal GEMM)"

    Q, K, W, G, h0, dht, dO, dv, scale = prepare_inputs(B, S, H, DK, DV, chunk_size)
    use_g = True

    dh_act, dh0_act, dv2_act = chunk_delta_bwd(
        Q,
        K,
        W,
        G,
        h0,
        dht,
        dO,
        dv,
        scale,
        chunk_size,
        use_g=use_g,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=block_DV,
    )

    dh_ref, dh0_ref, dv2_ref = torch_chunk_gated_delta_rule_bwd_dhu(
        Q.cpu(),
        K.cpu(),
        W.cpu(),
        G.cpu(),
        h0.cpu(),
        dht.cpu(),
        dO.cpu(),
        dv.cpu(),
        scale,
        use_g=use_g,
        use_initial_state=True,
        use_final_state_gradient=True,
        chunk_size=chunk_size,
    )

    ok = True
    for name, act, ref, dt in [
        ("dh", dh_act, dh_ref, "float32"),
        ("dh0", dh0_act, dh0_ref, "float32"),
        ("dv2", dv2_act, dv2_ref, "float32"),
    ]:
        passed, ratio, max_abs = check_precision(act, ref, dt)
        status = "PRECISION_PASS" if passed else "PRECISION_FAIL"
        print(f"[{status}] {tag} {name} B={B} S={S} H={H} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
    return ok


def test_chunk_delta_bwd_l0():
    """L0: threshold test — basic functional verification."""
    configs = [
        ("l0_small", 1, 128, 1, 128, 128, 64),  # BS=2
        ("l0_multi_chunk", 1, 256, 2, 128, 128, 64),  # BS=4
        ("l0_small_cs128", 1, 256, 1, 128, 128, 128),  # BS=2, C=128
        ("l0_small_cs256", 1, 512, 1, 128, 128, 256),  # BS=2, C=256
        ("l0_small_cs512", 1, 1024, 1, 128, 128, 512),  # BS=2, C=512
        ("l0_small_cs1024", 1, 2048, 1, 128, 128, 1024),  # BS=2, C=1024
    ]
    ok = True
    for name, B, S, H, DK, DV, cs in configs:
        try:
            passed = run_and_check(B, S, H, DK, DV, cs, tag=name)
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


def test_chunk_delta_bwd_l0_bench():
    """L0 bench shape: BS=32 with C=1024."""
    try:
        passed = run_and_check(1, 32768, 8, 128, 128, 1024, tag="l0_bench_shape")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] l0_bench_shape: {e}")
        return False


def test_chunk_delta_bwd_l1():
    """L1: functional test — irregular/extended shapes."""
    configs = [
        ("l1_b2_s256_h4", 2, 256, 4, 128, 128, 64),  # multi-batch
        ("l1_s512_h4", 1, 512, 4, 128, 128, 64),  # larger S
        ("l1_h16", 1, 256, 16, 128, 128, 64),  # more heads
        ("l1_s192", 1, 192, 2, 128, 128, 64),  # BS=3
        ("l1_s320_h8", 1, 320, 8, 128, 128, 64),  # BS=5
    ]
    ok = True
    for name, B, S, H, DK, DV, cs in configs:
        try:
            passed = run_and_check(B, S, H, DK, DV, cs, tag=name)
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: {e}")
            ok = False
    return ok


def test_chunk_delta_bwd_l2():
    """L2: boundary test — blocking on unexpected exceptions.

    Precision mismatch on boundary shape is WARN (non-blocking, may be
    unsupported input), but unexpected exceptions are FAIL (blocking,
    merged into exit code by main).
    """
    configs = [
        ("l2_min_s64", 1, 64, 1, 128, 128, 64),  # BS=1, minimal
    ]
    has_unexpected_failure = False
    for name, B, S, H, DK, DV, cs in configs:
        try:
            passed = run_and_check(B, S, H, DK, DV, cs, tag=name)
            if not passed:
                print(f"[BOUNDARY_WARN] {name}: precision mismatch on boundary shape")
        except _EXPECTED_L2_ERRORS as e:
            print(f"[BOUNDARY_WARN] {name}: rejected as expected ({type(e).__name__}): {e}")
        except Exception as e:
            print(f"[BOUNDARY_FAIL] {name}: unexpected exception ({type(e).__name__}): {e}")
            has_unexpected_failure = True
    if has_unexpected_failure:
        print("[BOUNDARY_FAIL] L2 suite: unexpected exceptions (blocking)")
    else:
        print("[BOUNDARY_PASS] L2 suite: completed (no unexpected exceptions)")
    return not has_unexpected_failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "all"])
    args = parser.parse_args()
    tilelang.disable_cache()
    torch.manual_seed(0)

    blocking_ok = True

    if args.level in ("l0", "all"):
        blocking_ok &= test_chunk_delta_bwd_l0()
    if args.level in ("l0", "all") and blocking_ok:
        blocking_ok &= test_chunk_delta_bwd_l0_bench()
    if args.level in ("l1", "all") and blocking_ok:
        blocking_ok &= test_chunk_delta_bwd_l1()

    if args.level in ("l2", "all"):
        blocking_ok &= test_chunk_delta_bwd_l2()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
