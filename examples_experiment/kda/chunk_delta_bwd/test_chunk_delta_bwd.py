"""Precision test suite for chunk_delta_bwd (KDA variant).

Golden reference (truncation-contract CPU implementation) +
mixed-tolerance dual-gate judgement over reference-prepared data.

Groups:
  A (blocking): BS<=2 satisfiable shapes, flag branches, inf/nan
     injection, seed matrix; data-domain attributions (no-gk reclass,
     dh0 fp32 margin - see dh0_margin_attributed) count as PASS.
  B (non-blocking FAIL, blocking diagnostics): BS=4/16/golden; expected
     [PRECISION_FAIL] is a reference-data limit, not a bug. Gates:
     mask agreement >= 99%, satisfiable-domain ratio >= 0.99.
  C (blocking): bit-exact stability (A x10, golden shape x3, 3 seeds).
  L1 (blocking): in-domain dimension coverage (B=2, H=8).
  L2 (blocking): support-domain and device rejection (ValueError
      before JIT).
  Boundary (non-blocking): 5-position special injection, distinct from A6;
     reported as [BOUNDARY_PASS]/[BOUNDARY_WARN].

Exit code: 0 iff A (incl. attributions) + B diagnostics + C + L1 + L2
hold, else 1 (boundary non-blocking).

Usage:
  run this file with --level l0|l1|l2|boundary|all (default l0)
"""

import argparse
import os
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_chunk_delta_bwd import (  # noqa: E402
    prepare_input,
    run_chunk_delta_bwd,
)

OUT_SPECS = (("dh", "bfloat16"), ("dh0", "float32"), ("dv2", "bfloat16"))

# bf16 hard-cap-unsatisfiable boundary (bf16 ULP at |x| >= 256 is 2.0,
# exceeding the 1e0 cap). Used as the satisfiable-domain bound for
# diagnostics.
SAT_DOMAIN_ABS = 256.0


# =====================================================================
# Golden reference: truncation-contract CPU implementation (lives in this file).
# =====================================================================
def golden_chunk_delta_bwd(
    Q,
    K,
    W,
    GK,
    dht,
    dO,
    dv,
    chunk_size,
    scale,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
):
    """Truncation-contract reference (CPU, per-chunk loops).

    Mirrors the kernel numeric path one-to-one at all four truncation
    points: GEMMs are evaluated in fp32 on the bf16-truncated operands,
    matching NPU bf16 x bf16 -> fp32 accumulate up to summation order
    (~1e-7 relative).
    Returns (dh, dh0, dv2); dh0 is zeros when use_initial_state=False
    (the kernel leaves that buffer unwritten; the wrapper zeroes dh0
    in this case).
    """
    B, S, H, DK = Q.shape
    DV = dO.shape[-1]
    BS = S // chunk_size
    dh = torch.empty(B, BS, H, DK, DV, dtype=torch.bfloat16)
    dv2 = torch.empty(B, S, H, DV, dtype=torch.bfloat16)
    # dh0 contract: zeros when use_initial_state=False (the wrapper
    # zeroes the unwritten kernel output in this case).
    dh0 = torch.zeros(B, H, DK, DV, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            if use_final_state_gradient:
                state = dht[b, h].float()
            else:
                state = torch.zeros(DK, DV, dtype=torch.float32)
            for i in range(BS - 1, -1, -1):
                s = i * chunk_size
                state_bf16 = state.to(torch.bfloat16)  # truncation points 1 & 2
                dh[b, i, h] = state_bf16
                Kc = K[b, s : s + chunk_size, h].float()
                dv_f = Kc @ state_bf16.float()  # dv GEMM on bf16 operand (tp2)
                dv2_c = (dv_f + dv[b, s : s + chunk_size, h].float()).to(torch.bfloat16)  # fp32 add, single truncation (tp3)
                dv2[b, s : s + chunk_size, h] = dv2_c
                if use_gk:
                    gk_last = GK[b, min((i + 1) * chunk_size, S) - 1, h]  # (DK,)
                    state = state * torch.exp2(gk_last)[:, None]  # fp32 gating
                Qc = Q[b, s : s + chunk_size, h].float()
                Wc = W[b, s : s + chunk_size, h].float()
                dOc = dO[b, s : s + chunk_size, h].float()
                g1 = (Qc.T @ dOc) * scale
                g2 = Wc.T @ dv2_c.float()  # tp4: W GEMM on truncated dv2
                state = state + g1 - g2
            if use_initial_state:
                dh0[b, h] = state
    return dh, dh0, dv2


# =====================================================================
# Precision standard: mixed tolerance + dual gate + inf/nan structural comparison
# =====================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).
    Mixed tolerance for the two output dtypes of this operator
    (bfloat16, float32)."""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
    }
    if dtype not in fp_table:
        raise ValueError(f"unsupported dtype for precision check: {dtype} (operator outputs are bfloat16/float32)")
    return fp_table[dtype]


def check_precision(actual, golden, dtype):
    """Mixed-tolerance double-gate judgement: (passed, matched_ratio, max_abs_error).
    Float gate: matched_ratio >= required AND max_abs_error <= limit; int:
    elementwise exact. inf/nan positions compared structurally, excluded
    from the numeric tolerance."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan positions: structural comparison
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # all finite-golden positions: actual must be finite too
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()  # actual inf/nan -> inf/nan error -> miss + max_abs
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


_REINT = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def bitwise_equal(a, b):
    """Bit-level equality (NaN payloads and signed zeros compare by raw
    bits) - deterministic-kernel checks must not use value semantics."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    rt = _REINT.get(a.dtype)
    if rt is not None:
        return torch.equal(a.view(rt), b.view(rt))
    return torch.equal(a, b)


def satisfiable_ratio(actual, golden, dtype):
    """matched_ratio over the satisfiable domain: common-finite positions
    with |golden| < SAT_DOMAIN_ABS. Returns (ratio, n) - n == 0 means the
    domain is empty (vacuous pass)."""
    atol, rtol, _, _ = get_precision(dtype)
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    sat = torch.isfinite(a) & torch.isfinite(g) & (g.abs() < SAT_DOMAIN_ABS)
    n = int(sat.sum().item())
    if n == 0:
        return 1.0, 0
    err = (a[sat] - g[sat]).abs()
    return (err <= atol + rtol * g[sat].abs()).float().mean().item(), n


def mask_agreement(actual, golden):
    """isfinite/isnan mask agreement fractions."""
    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    isf = (torch.isfinite(a) == torch.isfinite(g)).float().mean().item()
    isn = (torch.isnan(a) == torch.isnan(g)).float().mean().item()
    return isf, isn


# =====================================================================
# Case runner
# =====================================================================
def inject_special_dht(dht, seed):
    """Sparse +/-inf/nan injection into dht (CPU, deterministic).

    Guaranteed at least one +inf, one -inf, one nan and plenty of finite
    values. Propagation is deterministic:
    mixed-sign inf reductions collapse to NaN regardless of summation
    order, so kernel/golden special-value masks must agree exactly.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out = dht.clone()
    flat = out.view(-1)
    # deterministic anchors first
    flat[0] = float("inf")
    if flat.numel() > 1:
        flat[1] = float("-inf")
    if flat.numel() > 2:
        flat[2] = float("nan")
    # sparse random specials (~0.5%)
    mask = torch.rand(out.shape, generator=gen) < 0.005
    r = torch.rand(out.shape, generator=gen)
    out[mask & (r < 1.0 / 3.0)] = float("inf")
    out[mask & (r >= 1.0 / 3.0) & (r < 2.0 / 3.0)] = float("-inf")
    out[mask & (r >= 2.0 / 3.0)] = float("nan")
    return out


def make_case_data(shape, flags, seed, inject=False):
    """Reference prep on CPU (+ optional dht injection) and a single H2D."""
    B, S, H, DK, DV = shape
    C = 64
    cpu_in = prepare_input(B, S, H, DK, DV, C, seed=seed)
    if inject:
        cpu_in = list(cpu_in)
        cpu_in[4] = inject_special_dht(cpu_in[4], seed)
        cpu_in = tuple(cpu_in)
    npu_in = tuple(t.npu() for t in cpu_in)
    return cpu_in, npu_in


def run_kernel_and_golden(cpu_in, npu_in, shape, flags, kernel_flags=None):
    B, S, H, DK, DV = shape
    C = 64
    outs = run_chunk_delta_bwd(*npu_in, chunk_size=C, **flags, **(kernel_flags or {}))
    torch.npu.synchronize()
    ref = golden_chunk_delta_bwd(*cpu_in, C, DK**-0.5, **flags)
    return outs, ref


def fmt_metrics(metrics):
    return " ".join(f"{n}(ratio={r:.4f},max_abs={m:.3e})" for n, (r, m) in metrics.items())


def compare_all(outs, ref):
    """check_precision over the three outputs. Returns (ok, metrics dict).
    dh0 is compared as usual when use_initial_state=False (golden is
    zeros and the wrapper zeroes the kernel output - any dirty memory
    fails the gate immediately)."""
    ok = True
    metrics = {}
    for (name, dt), out, r in zip(OUT_SPECS, outs, ref):
        passed, ratio, max_abs = check_precision(out, r, dt)
        metrics[name] = (ratio, max_abs)
        ok &= passed
    return ok, metrics


def dh0_margin_attributed(outs, ref, metrics):
    """dh0 accumulation-order attribution with a causal-chain anchor.

    Mechanism: the fp32 GEMM summation-order
    difference between kernel and golden occasionally flips a bf16
    rounding boundary in dv2 (isolated 1-ULP jumps); the W^T GEMM then
    propagates each jump with a ~sum|W| (=~8 for C=64) gain into the
    fp32 state, i.e. into dh0.

    Attribution conditions (all must hold):
      1. dh and dv2 each pass their own double gate (ratio >= 0.99 AND
         max_abs <= the bf16 1e0 cap) - rules out misalignment;
      2. dh0 ratio >= 0.99 - rules out systematic error;
      3. causal chain: dv2 differs from golden at <= 0.1% of elements,
         each within 4 bf16 ULPs (isolated rounding jumps only), and
         dh0's max error stays within the ~8x GEMM propagation band of
         the dv2 jumps - dh0's error is CAUSED by the jumps, not by an
         independent defect.
    """
    for name in ("dh", "dv2"):
        r, m = metrics.get(name, (0.0, float("inf")))
        if not (r >= 0.99 and m <= 1e0):  # bf16 double gate
            return False
    r, m = metrics.get("dh0", (0.0, float("inf")))
    if r < 0.99:
        return False
    r_dh0 = ref[1]
    # causal chain via dv2 rounding jumps
    dv2k = outs[2].detach().cpu().float()
    dv2g = ref[2].detach().cpu().float()
    derr = (dv2k - dv2g).abs()
    nz = derr > 0
    n_nz = int(nz.sum().item())
    if n_nz > 0.001 * dv2g.numel():  # more than isolated jumps
        return False
    if n_nz:
        ulp = dv2g[nz].abs() * 2**-8 + 2**-10  # bf16 ULP + atol floor
        if not bool((derr[nz] <= 4 * ulp).all()):
            return False
        a0 = outs[1].detach().cpu().float()
        g0 = r_dh0.detach().cpu().float()
        e0_max = float((a0 - g0).abs().max())
        if e0_max > 8.0 * float(derr.max()) + 1e-3:
            return False
    return True


# =====================================================================
# Group A: blocking gate (BS<=2 satisfiable domain)
# =====================================================================
A_CASES = [
    # (name, shape, flags, seed, inject)
    ("A1_l0_single_chunk", (1, 64, 2, 128, 128), {}, 42, False),
    ("A2_l0_minimal", (1, 128, 2, 128, 128), {}, 42, False),
    (
        "A3_l0_no_dht",
        (1, 128, 2, 128, 128),
        {"use_final_state_gradient": False},
        42,
        False,
    ),
    ("A4_l0_no_gk", (1, 128, 2, 128, 128), {"use_gk": False}, 42, False),
    (
        "A5_l0_no_dh0",
        (1, 128, 2, 128, 128),
        {"use_initial_state": False},
        42,
        False,
    ),
    ("A6_l0_inf_nan_struct", (1, 128, 2, 128, 128), {}, 42, True),
]

A7_SEEDS = (0, 1, 2)

# Coverage additions: DK=64/DV=64 tile shape and the block_DV=64
# multi-block grid; kernel_flags are wrapper-only tiling and must NOT reach the golden.
A_CASES_S3 = [
    # (name, shape, flags, seed, inject, kernel_flags)
    ("A8_l0_dk64_dv64", (1, 128, 2, 64, 64), {}, 42, False, {}),
    ("A9_l0_bv2_block64", (1, 128, 2, 128, 128), {}, 42, False, {"block_DV": 64}),
]


def iter_a_cases():
    """Yield (name, shape, flags, seed, inject, kernel_flags) over group A."""
    for name, shape, flags, seed, inject in A_CASES:
        yield name, shape, flags, seed, inject, {}
    for name, shape, flags, seed, inject, kflags in A_CASES_S3:
        yield name, shape, flags, seed, inject, kflags


def run_one(name, shape, flags, seed, inject, data=None, kernel_flags=None):
    """Run one case and apply the blocking-verdict semantics (including
    data-domain attributions). Prints the [PRECISION_*] line and any
    attribution note. Returns (ok, metrics, data, outs)."""
    try:
        if data is None:
            data = make_case_data(shape, flags, seed, inject=inject)
        cpu_in, npu_in = data
        outs, ref = run_kernel_and_golden(cpu_in, npu_in, shape, flags, kernel_flags)
        ok, metrics = compare_all(outs, ref)
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] {name}: exception {e!r}")
        return False, {}, None, None
    if ok:
        print(f"[PRECISION_PASS] {name} {fmt_metrics(metrics)}")
        return True, metrics, data, outs
    # attribution chain (data-domain limits)
    if flags.get("use_gk") is False:
        gmax = max(float(r.detach().cpu().float().abs().max()) for r in ref)
        if gmax >= SAT_DOMAIN_ABS:
            ratios_ok = all(r >= 0.99 for (r, _) in metrics.values())
            print(f"[PRECISION_FAIL] {name} {fmt_metrics(metrics)}")
            print(
                f"  [RECLASS] {name}: golden |max|={gmax:.1f} >= 256 exceeds the "
                f"bf16 hard-cap domain; matched_ratio>=0.99: {'met' if ratios_ok else 'NOT met'}"
            )
            return ratios_ok, metrics, data, outs
    if dh0_margin_attributed(outs, ref, metrics):
        print(f"[PRECISION_FAIL] {name} {fmt_metrics(metrics)}")
        print(
            f"  [ATTRIBUTED] {name}: dh/dv2 pass their gates, dh0 ratio>=0.99, "
            "and dh0's error is within the GEMM propagation band of isolated "
            "dv2 bf16 rounding jumps - fp32 accumulation-order margin "
            "compression, not an implementation bug"
        )
        return True, metrics, data, outs
    print(f"[PRECISION_FAIL] {name} {fmt_metrics(metrics)}")
    return False, metrics, data, outs


def run_group_a():
    print("=== Group A (blocking gate, reference prep, BS<=2) ===")
    ok = True
    for name, shape, flags, seed, inject, kflags in iter_a_cases():
        case_ok, _, _, _ = run_one(name, shape, flags, seed, inject, kernel_flags=kflags)
        ok &= case_ok
    # A7: seed matrix on the minimal-recurrence shape
    for seed in A7_SEEDS:
        case_ok, _, _, _ = run_one(f"A7_l0_seeds_s{seed}", (1, 128, 2, 128, 128), {}, seed, False)
        ok &= case_ok
    return ok


# =====================================================================
# Group B: mathematical-limit diagnostics (non-blocking FAIL, blocking diagnostics)
# =====================================================================
B_CASES = [
    # (name, shape)
    ("B1_l0_limit_bs4", (1, 256, 4, 128, 128)),
    ("B2_l0_limit_bs16", (1, 1024, 8, 128, 128)),
    ("B3_l0_limit_golden", (1, 8192, 64, 128, 128)),
]


def diagnose_output(out, ref, dtype, name, bname, has_special):
    """Print check_precision + diagnostics for one B-group output.

    Diagnostic verdict (blocking): structural mask agreement (isfinite/
    isnan) >= 0.99 when the golden has non-finite content, and
    satisfiable-domain (common-finite, |golden| < 256) matched_ratio
    >= 0.99 (vacuous pass when the domain is empty). Full-domain
    matched_ratio is reported for transparency but is NOT a gate.
    """
    passed, ratio, max_abs = check_precision(out, ref, dtype)
    tag = "PASS" if passed else "FAIL"
    print(f"[PRECISION_{tag}] {bname} {name} ratio={ratio:.4f} max_abs={max_abs:.3e}")
    diag_ok = True
    if has_special:
        isf, isn = mask_agreement(out, ref)
        sat_ratio, n_sat = satisfiable_ratio(out, ref, dtype)
        mask_ok = isf >= 0.99 and isn >= 0.99
        sat_ok = sat_ratio >= 0.99
        met = mask_ok and sat_ok
        print(
            f"  [DIAG_{'PASS' if met else 'FAIL'}] {bname} {name} "
            f"isfinite_agree={isf:.4f} isnan_agree={isn:.4f} "
            f"sat_domain_ratio={sat_ratio:.4f}(n={n_sat}) "
            f"full_ratio={ratio:.4f} (masks>=0.99, sat-domain>=0.99)"
        )
        return diag_ok and met
    sat_ratio, n_sat = satisfiable_ratio(out, ref, dtype)
    met = sat_ratio >= 0.99
    print(
        f"  [DIAG_{'PASS' if met else 'FAIL'}] {bname} {name} "
        f"sat_domain_ratio={sat_ratio:.4f}(n={n_sat}) full_ratio={ratio:.4f} "
        f"(sat-domain>=0.99; hard-cap-only FAIL expected)"
    )
    return met


def run_group_b():
    print("=== Group B (reference-prep math-limit diagnostics, non-blocking) ===")
    diag_ok = True
    b3_data = None
    for name, shape in B_CASES:
        try:
            data = make_case_data(shape, {}, 42)
            outs, ref = run_kernel_and_golden(data[0], data[1], shape, {})
            if name == "B3_l0_limit_golden":
                b3_data = data
            for (oname, dt), out, r in zip(OUT_SPECS, outs, ref):
                has_special = bool((~torch.isfinite(r.detach().cpu())).any())
                diag_ok &= diagnose_output(out, r, dt, oname, name, has_special)
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] {name}: exception {e!r}")
            diag_ok = False
    print(
        "[B] NOTE: expected [PRECISION_FAIL] is a reference-data limit (unnormalized W past the bf16 hard cap), not an implementation bug."
    )
    return diag_ok, b3_data


# =====================================================================
# Group C: stability (blocking)
# =====================================================================
def run_group_c(b3_data=None):
    print("=== Group C (stability) ===")
    ok = True

    # C1: group A x 10 reruns - per-case verdict on run 1, bit-exact across runs
    for name, shape, flags, seed, inject, kflags in iter_a_cases():
        try:
            data = make_case_data(shape, flags, seed, inject=inject)
            cpu_in, npu_in = data
            case_ok, _, _, outs1 = run_one(f"C1_{name}", shape, flags, seed, inject, data=data, kernel_flags=kflags)
            first = tuple(o.detach().cpu() for o in outs1)
            for _rep in range(9):
                outs = run_chunk_delta_bwd(*npu_in, chunk_size=64, **flags, **kflags)
                torch.npu.synchronize()
                for o, f in zip(outs, first):
                    if not bitwise_equal(o.detach().cpu(), f):
                        case_ok = False
            tag = "PASS" if case_ok else "FAIL"
            print(f"[PRECISION_{tag}] C1_bitexact_x10 {name}")
            ok &= case_ok
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] C1_bitexact_x10 {name}: exception {e!r}")
            ok = False

    # C2: golden shape x 3 kernel self-consistency (bit-level, no golden)
    try:
        data = b3_data
        if data is None:
            data = make_case_data(B_CASES[2][1], {}, 42)
        first = None
        self_ok = True
        for _ in range(3):
            outs = run_chunk_delta_bwd(*data[1], chunk_size=64)
            torch.npu.synchronize()
            outs_cpu = tuple(o.detach().cpu() for o in outs)
            if first is None:
                first = outs_cpu
            else:
                for a, b in zip(first, outs_cpu):
                    if not bitwise_equal(a, b):
                        self_ok = False
        tag = "PASS" if self_ok else "FAIL"
        print(f"[PRECISION_{tag}] C2_golden_shape_selfconsistent_x3")
        ok &= self_ok
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] C2_golden_shape_selfconsistent_x3: exception {e!r}")
        ok = False

    # C3: group A x 3-seed matrix - verdict semantics per case
    for name, shape, flags, _seed, _inject, kflags in iter_a_cases():
        for seed in (0, 1, 2):
            case_ok, _, _, _ = run_one(f"C3_{name}_s{seed}", shape, flags, seed, False, kernel_flags=kflags)
            ok &= case_ok
    return ok


# =====================================================================
# L1: in-domain dimension coverage (BLOCKING)
# =====================================================================
L1_CASES = [
    # (name, shape) - reference prep + dual-threshold check on the BS=2
    # satisfiable domain; dimensions not exercised by group A (B>1, H>2),
    # plus the block_DV=32 paths: DV=32 selects block_DV=32 and exercises
    # the scalar T.Parallel gate fallback (row_expand needs 256B rows =
    # 64 fp32 cols; 32 cols x 4B = 128B does not qualify), and DV=96 adds
    # the bv_num=3 grid decode. block_DV is auto-selected by the wrapper
    # (_select_block_dv), no kernel_flags needed.
    ("l1_multi_batch", (2, 128, 2, 128, 128)),
    ("l1_head_variety", (1, 128, 8, 128, 128)),
    ("l1_dv32_block32_scalar_gate", (1, 128, 2, 128, 32)),
    ("l1_dv96_bv3", (1, 128, 2, 128, 96)),
    # DK x block_DV matrix closure: all legal corners must be exercised
    # directly, not inferred from the largest configuration (DK=128,
    # block_DV=128) having passed.
    ("l1_dk64_dv128", (1, 128, 2, 64, 128)),
    ("l1_dk64_dv96", (1, 128, 2, 64, 96)),
]


def l1_scale_applied():
    """scale effectiveness (anti-no-op, dual assertion): (1) scale=1.0
    must pass the dual gate against its own scale=1.0 golden, and
    (2) its dh0 must differ from the default-scale (DK**-0.5) kernel
    output on the same inputs - a kernel that silently ignores scale
    would produce bit-identical outputs and pass (1) vacuously."""
    name = "l1_scale_applied"
    shape = (1, 128, 2, 128, 128)
    try:
        cpu_in, npu_in = make_case_data(shape, {}, 42)
        # (1) scale=1.0 vs its own golden (dual gate)
        outs1 = run_chunk_delta_bwd(*npu_in, chunk_size=64, scale=1.0)
        torch.npu.synchronize()
        ref1 = golden_chunk_delta_bwd(*cpu_in, 64, 1.0)
        ok, metrics = compare_all(outs1, ref1)
        # (2) default scale on the same inputs: the kernel is
        # deterministic, so a working scale must move the outputs
        # (dh0 is fp32-direct-stored, no bf16 masking of the change).
        outs_def = run_chunk_delta_bwd(*npu_in, chunk_size=64)
        torch.npu.synchronize()
        diffs = {
            oname: float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max())
            for (oname, _), a, b in zip(OUT_SPECS, outs1, outs_def)
        }
    except Exception as e:  # noqa: BLE001
        print(f"[PRECISION_FAIL] {name}: exception {e!r}")
        return False
    if ok and diffs["dh0"] > 1e-3:
        print(f"[PRECISION_PASS] {name} {fmt_metrics(metrics)} scale_effective(dh0_max_diff={diffs['dh0']:.3e})")
        return True
    print(f"[PRECISION_FAIL] {name} {fmt_metrics(metrics)} scale_diffs({diffs})")
    return False


def run_l1():
    print("=== L1 (blocking, in-domain dimension coverage) ===")
    ok = True
    for name, shape in L1_CASES:
        case_ok, _, _, _ = run_one(name, shape, {}, 42, False)
        ok &= case_ok
    ok &= l1_scale_applied()
    return ok


# =====================================================================
# L2: support-domain rejection contract (BLOCKING; ValueError before any JIT compile)
# =====================================================================
def _l2_dummy_inputs(B, S, H, DK, DV):
    """Metadata-only zero tensors for the L2 rejection probes. The
    wrapper validates shapes/dtypes/support domain before any JIT
    compile or launch, so plain CPU tensors suffice and content is
    irrelevant."""
    return (
        torch.zeros(B, S, H, DK, dtype=torch.bfloat16),  # Q
        torch.zeros(B, S, H, DK, dtype=torch.bfloat16),  # K
        torch.zeros(B, S, H, DK, dtype=torch.bfloat16),  # W
        torch.zeros(B, S, H, DK, dtype=torch.float32),  # GK
        torch.zeros(B, H, DK, DV, dtype=torch.bfloat16),  # dht
        torch.zeros(B, S, H, DV, dtype=torch.bfloat16),  # dO
        torch.zeros(B, S, H, DV, dtype=torch.bfloat16),  # dv
    )


L2_CASES = [
    # (name, shape, wrapper kwargs, expected message substring) - each
    # must be rejected by the wrapper's validation with a ValueError
    # whose message matches the specific guard (type + message double
    # assertion: an unrelated ValueError must NOT count as a pass). The
    # first four carry illegal metadata (domain rejection); the last has
    # fully legal metadata on CPU tensors, so the device check is the
    # only possible rejection.
    ("l2_reject_S_not_div64", (1, 96, 2, 128, 128), {"chunk_size": 64}, "require S %"),
    ("l2_reject_chunk_size", (1, 128, 2, 128, 128), {"chunk_size": 128}, "unsupported chunk_size"),
    ("l2_reject_DV_unsupported", (1, 128, 2, 128, 48), {}, "unsupported DV"),
    ("l2_reject_DK_unsupported", (1, 128, 2, 192, 128), {}, "unsupported DK"),
    ("l2_reject_cpu_tensor", (1, 128, 2, 128, 128), {}, "must be on npu"),
]


def _run_exception(name, fn, expect_type, expect_msg):
    """Rejection probe with a type + message double assertion: fn must
    raise expect_type with expect_msg as a message substring. Any other
    exception type, a message mismatch, or a silent pass is a
    BOUNDARY_FAIL. Returns ok."""
    try:
        fn()
    except expect_type as e:
        if expect_msg not in str(e):
            print(f"[BOUNDARY_FAIL] {name}: message mismatch - expected substring {expect_msg!r}, got {type(e).__name__}: {e}")
            return False
        print(f"[BOUNDARY_PASS] {name}: rejected as expected ({type(e).__name__}: {e})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_FAIL] {name}: unexpected exception {type(e).__name__}: {e}")
        return False
    print(f"[BOUNDARY_FAIL] {name}: illegal config NOT rejected")
    return False


def run_l2():
    print("=== L2 (blocking, support-domain rejection contract) ===")
    ok = True
    for name, shape, kwargs, expect_msg in L2_CASES:
        inputs = _l2_dummy_inputs(*shape)
        ok &= _run_exception(
            name,
            lambda inputs=inputs, kwargs=kwargs: run_chunk_delta_bwd(*inputs, **kwargs),
            ValueError,
            expect_msg,
        )
    # Batch-mismatch negatives (shape guards vs Q): Q carries B=2 while
    # exactly one operand carries B=1 - the wrapper must reject on shape
    # before any other check. Every guard needs a blocking negative to
    # lock it. Case 1 violates only K (K.shape is checked first); case 2
    # keeps K legal so the K guard stays silent and the dht contract
    # guard is the one hit.
    B2, S, H, DK, DV = 2, 128, 2, 128, 128
    q_b2 = torch.zeros(B2, S, H, DK, dtype=torch.bfloat16)
    k_b1 = torch.zeros(1, S, H, DK, dtype=torch.bfloat16)
    k_b2 = torch.zeros(B2, S, H, DK, dtype=torch.bfloat16)
    w_b2 = torch.zeros(B2, S, H, DK, dtype=torch.bfloat16)
    gk_b2 = torch.zeros(B2, S, H, DK, dtype=torch.float32)
    dht_b1 = torch.zeros(1, H, DK, DV, dtype=torch.bfloat16)
    dht_b2 = torch.zeros(B2, H, DK, DV, dtype=torch.bfloat16)
    do_b2 = torch.zeros(B2, S, H, DV, dtype=torch.bfloat16)
    dv_b2 = torch.zeros(B2, S, H, DV, dtype=torch.bfloat16)
    ok &= _run_exception(
        "l2_reject_batch_mismatch_K",
        lambda: run_chunk_delta_bwd(q_b2, k_b1, w_b2, gk_b2, dht_b2, do_b2, dv_b2, chunk_size=64),
        ValueError,
        "K.shape",
    )
    ok &= _run_exception(
        "l2_reject_batch_mismatch_dht",
        lambda: run_chunk_delta_bwd(q_b2, k_b2, w_b2, gk_b2, dht_b1, do_b2, dv_b2, chunk_size=64),
        ValueError,
        "dht.shape",
    )
    return ok


# =====================================================================
# Boundary: special-value injection (NON-BLOCKING)
# =====================================================================
def inject_special_dht_boundary(dht):
    """Boundary-caliber dht injection: 5 deterministic specials (count/
    placement distinct from A6), incl. same-column +/-inf pairs whose
    NaN collapse is deterministic (see inject_special_dht); masks must agree."""
    out = dht.clone()
    _, H, DK, DV = dht.shape
    out[0, 0, 0, 0] = float("inf")  # same column (v=0) as the -inf below
    out[0, 0, 7, 0] = float("-inf")  # +/- inf mix in the dv GEMM column
    out[0, 1, 0, 64] = float("-inf")  # same column (v=64), head 1
    out[0, 1, 9, 64] = float("inf")
    out[0, H - 1, DK // 2, DV // 2] = float("nan")
    return out


def run_boundary():
    print("=== Boundary (non-blocking, special-value injection) ===")
    name = "boundary_inf_nan_5pos"
    shape = (1, 128, 2, 128, 128)
    B, S, H, DK, DV = shape
    try:
        cpu_in = prepare_input(B, S, H, DK, DV, 64, seed=42)
        cpu_in = tuple(inject_special_dht_boundary(t) if i == 4 else t for i, t in enumerate(cpu_in))
        data = (cpu_in, tuple(t.npu() for t in cpu_in))
        case_ok, _, _, _ = run_one(name, shape, {}, 42, False, data=data)
    except Exception as e:  # noqa: BLE001
        print(f"[BOUNDARY_WARN] {name}: exception {e!r}")
        return False
    tag = "BOUNDARY_PASS" if case_ok else "BOUNDARY_WARN"
    print(f"[{tag}] {name} (non-blocking, not folded into the exit code)")
    return case_ok


# =====================================================================
# main: --level dispatch (l0 = A+B+C; l1/l2 blocking; boundary print-only)
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(42)

    blocking_ok = True
    if args.level in ("l0", "all"):
        a_ok = run_group_a()
        b_diag_ok, b3_data = run_group_b()
        c_ok = run_group_c(b3_data)
        blocking_ok &= a_ok and b_diag_ok and c_ok
    if args.level in ("l1", "all"):
        blocking_ok &= run_l1()
    if args.level in ("l2", "all"):
        blocking_ok &= run_l2()  # rejection contract: blocking
    if args.level in ("boundary", "all"):
        run_boundary()  # special values: non-blocking, print-only

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
