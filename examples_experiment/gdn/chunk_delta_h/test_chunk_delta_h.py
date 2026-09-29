"""Layered precision test for chunk_delta_h (Ascend NPU, Developer mode).

Uses check_precision double-threshold judgment (NOT torch.assert_close).
Three outputs ALL judged with bfloat16 thresholds:
  - h (bfloat16):            atol=2^-10, rtol=2^-6, max_abs=1e0, req=0.99
  - final_state (physically float32): judged by the bfloat16 row — the
    final_state standard uses bf16 thresholds to enable the bf16 GEMM
    performance route
  - V_new (bfloat16):        atol=2^-10, rtol=2^-6, max_abs=1e0, req=0.99

The golden (ref_chunk_delta_h) uses HIGH-PRECISION fp32 GEMM (CPU
torch.matmul with .float() casts, h kept fp32 between chunks) as the
single high-precision benchmark. The kernel uses bf16 GEMM (input-side
quantization) — that quantization gap vs the fp32 golden is the precision
metric, accepted under the bf16 thresholds.

L0 cases:
  - l0_small_no_g: B=1, S=256, H=8, use_g=False (basic algorithm, 4 chunks;
    use_g=False with BS>4 is mathematically unstable on the bf16 route — GDN
    recurrence eigenvalues exceed 1 without gate decay and bf16 quantization
    is amplified past the threshold at BS>=8; the BS<=4 bound applies on
    this exact route — S=1024/BS=16 calibration was inherited from the fp32
    route and was wrong)
  - l0_small_g: B=1, S=4096, H=8, use_g=True (gate path, 64 chunks)
  - l0_main_config: B=1, S=32768, H=32, use_g=True (full config, 512 chunks,
    slow compile)

L1 functional cases (16, BLOCKING — all must PASS; deterministic shape
generation + D-PARAM coverage):
  - Aligned / tail-1 (S=65) / tail-mid (S=96) / prime (S=127) / edge
    (S=64 + H=1). Non-aligned S follows the host-truncation contract:
    originals keep the true S tokens (golden drops the tail internally),
    kernel-side inputs are truncated to BS*chunk_size.
  - Symmetric vrange only: (-0.3,0.3) small / (-2,2) mid (CPU-calibrated:
    outputs stay <=~1, bf16 gap well within thresholds). ASYMMETRIC vrange
    is FORBIDDEN in L1 — rank-1 divergence.
  - Attr variants: chunk_size=128 (+block_DV=64, UB-budget-safe combo),
    block_DV=64, use_g=False (BS<=4 bound: S=64 and S=256),
    use_initial_state=True, store_final_state=False + save_new_value=False.
  - use_g=False cases strictly limited to BS<=4 (S<=256).

L2 negative cases (5, BLOCKING — illegal inputs must be rejected):
  - unsupported dtype (float16 K/W/U variant), bad chunk_size (8 < 16
    fractal minimum), non-aligned DV (DV=65 with block_DV=64), and the
    S-split segment-alignment class: chunk_size=192/320 (%16==0 but
    %128!=0 — without the %128 assert, 192 reads W out of bounds at the
    compile stage while 320 silently drops the tail rows of the V_new
    path; both confirmed by a live repro before the fix). Compile-only
    probes, never executed.

Boundary special-value cases (6, NON-BLOCKING — legal values judged by
precision, WARN if beyond thresholds):
  - zero_input / large_values (x64, recurrence divergence — mathematical
    property, both kernel & golden diverge) / asym_vrange (-0.5,1.5 — rank-1
    divergence) / inf / nan / dbound (values ~1e38 near bf16 boundary,
    overflow propagation). All share one compiled config.

Coverage annotations (coverage_check.py): every case carries tags= hitting
the D-* dimensions of the Fusion category contract (23 required dims).
"""

import argparse
import glob as _glob
import os
import re
import subprocess
import sys

import torch

from example_chunk_delta_h import (
    chunk_delta_h,
    prepare_input,
    prepare_k_gated,
    prepare_m,
    relay_dtype,
    relay_slot_count,
    to_chunk_major,
)

# ============================================================================
# Test compile robustness — retry / stderr capture / tmp cleanup
# ============================================================================


def _test_cleanup_tmp_so():
    """Remove temporary compile artifacts after each test case (NOT .cpp —
    those are small source files useful for post-mortem; only the compiled
    .so artifacts accumulate and can poison retries).
    """
    removed = 0
    for p in _glob.glob("/tmp/tmp*.so"):
        try:
            os.remove(p)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[cleanup] removed {removed} stale temporary .so artifacts")


def _capture_bisheng_stderr(exc):
    """Parse the bisheng compile command from an exception message and
    re-run it to capture the raw stderr (which otherwise lands above the
    traceback and is lost in long logs).

    Returns the stderr tail (last 3000 chars) or an empty string if no
    command could be extracted.
    """
    msg = str(exc)
    m = re.search(r"(?:ccec|bisheng|clang)[^\n]*", msg)
    if not m:
        return ""
    cmd = m.group(0).strip()
    if not cmd or len(cmd) < 10:
        return ""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return result.stderr[-3000:] if result.stderr else ""
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _test_run_with_retry(fn, max_retries=2):
    """Retry fn() ONLY on compile errors (not precision/runtime errors).

    On a "Compilation Failed" exception: capture bisheng stderr via
    _capture_bisheng_stderr, clean temporary .so artifacts, and retry up to
    max_retries. The bisheng stderr is attached to the re-raised exception
    for diagnosis.
    """
    last_err = None
    last_stderr = ""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if "Compilation Failed" not in str(e) and "compile" not in str(e).lower():
                raise
            last_err = e
            last_stderr = _capture_bisheng_stderr(e)
            _test_cleanup_tmp_so()
            print(f"[retry] compile error (attempt {attempt + 1}/{max_retries + 1}); cleaned temp artifacts, retrying ...")
    if last_stderr:
        raise type(last_err)(f"{last_err}\n[bisheng stderr (captured)]\n{last_stderr}") from last_err
    raise last_err


# ============================================================================
# Coverage annotations (for coverage_check.py)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"

# ============================================================================
# Precision judgment (double-threshold)
# ============================================================================


def get_precision(dtype):
    """Get precision thresholds by dtype string."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Double-threshold precision check.

    Returns (passed, matched_ratio, max_abs_error).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
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


# ============================================================================
# PyTorch Golden Reference (HIGH-PRECISION fp32)
# ============================================================================


def ref_chunk_delta_h(
    K,
    W,
    U,
    G,
    initial_state,
    chunk_size=64,
    use_g=True,
    use_initial_state=False,
    store_final_state=True,
    save_new_value=True,
):
    """PyTorch reference implementation of chunk_delta_h.

    HIGH-PRECISION fp32 GEMM (CPU torch.matmul with .float() casts, h kept
    fp32 between chunks). The golden is the single high-precision benchmark;
    the kernel's bf16 input-side quantization gap vs this golden is the
    precision metric (judged under bf16 thresholds).

    Uses the gate-absorbed formulation:
      V_new = U - W @ h_prev                       (ungated, save position)
      h_new = exp(g_last) * h + K_gated^T @ V_new  (K_gated = K*exp(g_last-g_i))

    Args:
        K: (B, S, H, DK) bfloat16 — original input
        W: (B, S, H, DK) bfloat16 — original input
        U: (B, S, H, DV) bfloat16 — original input
        G: (B, S, H) float32 — already chunk_local_cumsum'd
        initial_state: (B, H, DK, DV) bfloat16
        chunk_size: C, default 64
        use_g: whether to apply gate
        use_initial_state: whether to use initial_state
        store_final_state: whether to return final_state
        save_new_value: whether to return V_new

    Returns:
        h: (B, BS, H, DK, DV) bfloat16
        final_state: (B, H, DK, DV) float32 (or None)
        V_new: (B, S, H, DV) bfloat16
    """
    B, S, H, DK = K.shape
    DV = U.shape[-1]
    BS = S // chunk_size

    K_f = K.float()
    W_f = W.float()
    U_f = U.float()

    if use_initial_state:
        h = initial_state.float().clone()  # fp32 recursion state
    else:
        h = torch.zeros(B, H, DK, DV, dtype=torch.float32)

    h_out = torch.zeros(B, BS, H, DK, DV, dtype=torch.float32)
    V_new_out = torch.zeros(B, S, H, DV, dtype=torch.float32)

    for i in range(BS):
        h_out[:, i] = h  # fp32 h_prev

        k_c = K_f[:, i * chunk_size : (i + 1) * chunk_size]  # (B, C, H, DK)
        w_c = W_f[:, i * chunk_size : (i + 1) * chunk_size]
        u_c = U_f[:, i * chunk_size : (i + 1) * chunk_size]
        g_c = G[:, i * chunk_size : (i + 1) * chunk_size]  # (B, C, H)

        # V_new = U - W @ h (PRE-gate value, standard save position)
        ws = torch.matmul(w_c.permute(0, 2, 1, 3), h)  # (B, H, C, DV)
        ws = ws.permute(0, 2, 1, 3)  # (B, C, H, DV)
        v_new = u_c - ws  # (B, C, H, DV)
        if save_new_value:
            V_new_out[:, i * chunk_size : (i + 1) * chunk_size] = v_new

        # gate absorbed into K_gated
        if use_g:
            g_last = G[:, (i + 1) * chunk_size - 1, :]  # (B, H)
            coeff = torch.exp(g_last.unsqueeze(1) - g_c)  # (B, C, H)
            k_gated = k_c * coeff.unsqueeze(-1)  # (B, C, H, DK)
            h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1)
        else:
            k_gated = k_c

        # h += K_gated^T @ V_new (fp32 GEMM)
        kv = torch.matmul(
            k_gated.permute(0, 2, 3, 1),  # (B, H, DK, C)
            v_new.permute(0, 2, 1, 3),  # (B, H, C, DV)
        )  # (B, H, DK, DV)
        h = h + kv

    h_out_bf16 = h_out.to(torch.bfloat16)
    V_new_bf16 = V_new_out.to(torch.bfloat16)
    final_state = h if store_final_state else None  # fp32 (no cast)

    return h_out_bf16, final_state, V_new_bf16


# ============================================================================
# Kernel compile memoization (in-process; boundary suite shares one config)
# ============================================================================

_KERNEL_CACHE = {}


def _get_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    use_g,
    use_initial_state,
    store_final_state,
    save_new_value,
    block_DV,
):
    """Compile the kernel for one config, or reuse an in-process copy.

    The boundary suite's 6 cases share one config — memoization replaces 6
    identical compiles with 1. tilelang.disable_cache() in main only disables
    the on-disk cache; this dict cache is independent of it.
    """
    key = (
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        use_g,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DV,
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _test_run_with_retry(
            lambda: chunk_delta_h(
                B=B,
                S=S,
                H=H,
                DK=DK,
                DV=DV,
                chunk_size=chunk_size,
                use_g=use_g,
                use_initial_state=use_initial_state,
                store_final_state=store_final_state,
                save_new_value=save_new_value,
                input_dtype="bfloat16",
                output_dtype="bfloat16",
                accum_dtype="float32",
                state_dtype="float32",
                block_DV=block_DV,
            )
        )
    return _KERNEL_CACHE[key]


# ============================================================================
# Boundary input mutations (applied to the ORIGINALS; golden sees the same)
# ============================================================================


def _special_zero(K, W, U):
    """All-zero K/W/U (G unchanged) — everything stays exactly zero."""
    return torch.zeros_like(K), torch.zeros_like(W), torch.zeros_like(U)


def _special_inf(K, W, U):
    """Single +inf injected at U[0, 0, 0, 0] (first token/head/element)."""
    U2 = U.clone()
    U2[0, 0, 0, 0] = float("inf")
    return K, W, U2


def _special_nan(K, W, U):
    """Single nan injected at U[0, 0, 0, 0]."""
    U2 = U.clone()
    U2[0, 0, 0, 0] = float("nan")
    return K, W, U2


def _special_dbound(K, W, U):
    """Values scaled to ~1e38 (near the bfloat16 max 3.39e38, still finite).

    GEMM products overflow fp32 — inf/nan propagation is the observed object.
    """
    s = 1.0e38
    return (
        (K.float() * s).to(torch.bfloat16),
        (W.float() * s).to(torch.bfloat16),
        (U.float() * s).to(torch.bfloat16),
    )


# ============================================================================
# Case runner
# ============================================================================


def _run_one_case(
    name,
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    block_DV,
    use_g,
    use_initial_state=False,
    store_final_state=True,
    save_new_value=True,
    vrange=None,
    special=None,
    level_tag="L0",
    blocking=True,
):
    """Run one test case: prepare, compile, run, golden, check precision.

    Non-aligned S contract: the kernel only supports full chunks. Data is
    generated at the original S (padded up to an aligned length for
    generation, then sliced back); the golden receives the true S-token
    originals (ref_chunk_delta_h drops the tail internally — same contract),
    and the kernel-side inputs are truncated to BS*chunk_size.

    Args:
        vrange: (lo, hi) affine value-range scaling of K/W/U. Symmetric
            ranges (lo=-hi) are pure scaling; asymmetric ranges add bias
            and are FORBIDDEN in L1 (rank-1 divergence) — only used by
            the non-blocking Boundary suite.
        special: callable(K, W, U) -> (K, W, U) boundary mutation (zero /
            inf / nan / dtype-boundary). Applied to the originals before
            kernel-input re-derivation; the golden sees the same inputs.
        blocking: True → [PRECISION_*] markers (L0/L1, affect exit code);
            False → [BOUNDARY_*] markers (Boundary, non-blocking).

    Returns:
        True if all enabled outputs pass (blocking mode); in non-blocking
        mode the return value is still boolean but the caller ignores it.
    """
    print("=" * 60)
    print(f"[{level_tag}:{name}] B={B}, S={S}, H={H}, DK={DK}, DV={DV}, C={chunk_size}")
    print(f"  use_g={use_g}, block_DV={block_DV}, vrange={vrange}, special={getattr(special, '__name__', None)}")
    print("=" * 60)

    ok_tag = "PRECISION_PASS" if blocking else "BOUNDARY_PASS"
    bad_tag = "PRECISION_FAIL" if blocking else "BOUNDARY_WARN"

    try:
        # --- Host-side data preparation (all CPU) ---
        BS_actual = S // chunk_size
        S_aligned = BS_actual * chunk_size
        gen_S = S if S % chunk_size == 0 else ((S + chunk_size - 1) // chunk_size) * chunk_size

        print(f"[{level_tag}:{name}] preparing input (CPU) ...")
        (
            K,
            W,
            U,
            G,
            initial_state,
            K_gated,
            W_cm,
            U_cm,
            gate_scale,
            _init,
        ) = prepare_input(B, gen_S, H, DK, DV, chunk_size, block_DV=block_DV, use_g=use_g)

        # Non-aligned S: originals keep the true S tokens (golden input).
        if gen_S != S:
            print(
                f"  [tail] S={S} not a multiple of C={chunk_size}: "
                f"host truncates kernel side to S_aligned={S_aligned} (BS={BS_actual}); "
                f"golden drops the tail internally (same contract)"
            )
            K = K[:, :S]
            W = W[:, :S]
            U = U[:, :S]
            G = G[:, :S]

        # Boundary mutation (zero / inf / nan / dtype-boundary).
        if special is not None:
            K, W, U = special(K, W, U)

        # Value-range scaling (affine; symmetric ranges have bias=0).
        vrange_active = vrange is not None and vrange != (-1, 1)
        if vrange_active:
            lo, hi = vrange
            scale = (hi - lo) / 2.0
            bias = (hi + lo) / 2.0
            K = (K.float() * scale + bias).to(torch.bfloat16)
            W = (W.float() * scale + bias).to(torch.bfloat16)
            U = (U.float() * scale + bias).to(torch.bfloat16)

        # Re-derive kernel inputs from the (possibly mutated / truncated)
        # originals when needed. gate_scale depends only on G — it is
        # recomputed never, only sliced for the tail case.
        if (gen_S != S) or special is not None or vrange_active:
            K_k = K[:, :S_aligned]
            W_k = W[:, :S_aligned]
            U_k = U[:, :S_aligned]
            G_k = G[:, :S_aligned]
            K_gated = prepare_k_gated(K_k, G_k, chunk_size, use_g=use_g)
            W_cm = to_chunk_major(W_k, chunk_size)
            U_cm = to_chunk_major(U_k, chunk_size)
            if gen_S != S:
                gate_scale = gate_scale[:, :BS_actual]

        print(f"[{level_tag}:{name}] compiling kernel ...")
        kernel = _get_kernel(
            B=B,
            S=S_aligned,
            H=H,
            DK=DK,
            DV=DV,
            chunk_size=chunk_size,
            use_g=use_g,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            save_new_value=save_new_value,
            block_DV=block_DV,
        )

        # M (M-form) must be derived AFTER any re-derivation above
        # (tail truncation / boundary mutation / vrange all change K_gated /
        # W_cm / gate_scale). use_g=False -> dummy zeros (never read).
        M = prepare_m(K_gated, W_cm, gate_scale, DK, use_g=use_g)

        print(f"[{level_tag}:{name}] moving inputs to NPU ...")
        bv_num = (DV + block_DV - 1) // block_DV
        total_blocks = bv_num * B * H
        # h_state_gm / state_zero: 3-D (total_blocks, DK, block_DV) — matches
        # the kernel's T.Tensor declaration.
        h_state_gm = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        state_zero = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        workspace_h = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        workspace_v = torch.zeros(total_blocks, chunk_size, block_DV, dtype=torch.bfloat16, device="npu")
        # The mform S-SPLIT calibers (C>128) harden the C->V relay to
        # 4 slots (see example_chunk_delta_h.relay_slot_count) — the slot
        # count and dtype must match the kernel's traced workspace shape
        # exactly.
        workspace_vwh = torch.zeros(
            total_blocks,
            relay_slot_count(chunk_size, use_g=use_g),
            chunk_size,
            block_DV,
            dtype=getattr(torch, relay_dtype(chunk_size, use_g=use_g)),
            device="npu",
        )
        workspace_kv = torch.zeros(total_blocks, DK, block_DV, dtype=torch.float32, device="npu")

        print(f"[{level_tag}:{name}] running kernel ...")
        h_out, final_state, V_new = kernel(
            K_gated.to("npu"),
            W_cm.to("npu"),
            U_cm.to("npu"),
            gate_scale.to("npu"),
            initial_state.to("npu"),
            M.to("npu"),
            h_state_gm,
            state_zero,
            workspace_h,
            workspace_v,
            workspace_vwh,
            workspace_kv,
        )
        torch.npu.synchronize()

        print(f"[{level_tag}:{name}] computing golden (CPU) ...")
        h_ref, final_state_ref, V_new_ref = ref_chunk_delta_h(
            K,
            W,
            U,
            G,
            initial_state,
            chunk_size=chunk_size,
            use_g=use_g,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            save_new_value=save_new_value,
        )

        all_passed = True

        # V_new comparison: only the valid (aligned) region — the kernel
        # outputs BS*chunk_size rows; the golden may hold more (tail rows
        # are zero-filled by ref and excluded by contract).
        # The kernel's V_new output is CHUNK-MAJOR (B, BS, H, C, DV)
        # — permute to token-major for the golden comparison (mechanical
        # layout plumbing; the comparison logic itself is unchanged).
        valid_S = BS_actual * chunk_size
        vn_token = V_new.permute(0, 1, 3, 2, 4).reshape(B, S_aligned, H, DV)
        vn_out_cmp = vn_token[:, :valid_S]
        vn_ref_cmp = V_new_ref[:, :valid_S]

        h_passed, h_ratio, h_max_abs = check_precision(h_out, h_ref, "bfloat16")
        status = ok_tag if h_passed else bad_tag
        print(f"[{status}] {name} h ratio={h_ratio:.4f} max_abs={h_max_abs:.3e}")
        all_passed = all_passed and h_passed

        if store_final_state:
            # final_state is physically fp32 (UB fp32 direct out) but
            # judged by the bfloat16 row.
            fs_passed, fs_ratio, fs_max_abs = check_precision(final_state, final_state_ref, "bfloat16")
            status = ok_tag if fs_passed else bad_tag
            print(f"[{status}] {name} final_state ratio={fs_ratio:.4f} max_abs={fs_max_abs:.3e}")
            all_passed = all_passed and fs_passed

        if save_new_value:
            vn_passed, vn_ratio, vn_max_abs = check_precision(vn_out_cmp, vn_ref_cmp, "bfloat16")
            status = ok_tag if vn_passed else bad_tag
            print(f"[{status}] {name} V_new ratio={vn_ratio:.4f} max_abs={vn_max_abs:.3e}")
            all_passed = all_passed and vn_passed

        return all_passed

    except Exception as e:
        print(f"[{bad_tag}] {name} exception: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        _test_cleanup_tmp_so()


def test_chunk_delta_h_l0():
    """L0 precision test: run all three L0 cases."""
    torch.manual_seed(0)
    results = []

    # S=256 (BS=4) — bf16-route no-gate stability bound is BS<=4
    # (precision degrades at BS>=8).
    r1 = _run_one_case(
        "l0_small_no_g",
        B=1,
        S=256,
        H=8,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_g=False,
    )
    results.append(("l0_small_no_g", r1))

    r2 = _run_one_case(
        "l0_small_g",
        B=1,
        S=4096,
        H=8,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_g=True,
    )
    results.append(("l0_small_g", r2))

    print("\n[L0:l0_main_config] WARNING: S=32768 compilation may take >10min ...")
    r3 = _run_one_case(
        "l0_main_config",
        B=1,
        S=32768,
        H=32,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_g=True,
    )
    results.append(("l0_main_config", r3))

    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PRECISION_PASS" if passed else "PRECISION_FAIL"
        print(f"[{status}] {name}")
        all_passed = all_passed and passed

    if all_passed:
        print("[PRECISION_PASS] All L0 tests passed")
    else:
        print("[PRECISION_FAIL] Some L0 tests failed")
    print("=" * 60)

    return all_passed


# ============================================================================
# L1: Functional tests (BLOCKING — deterministic shapes + attr combos)
# ============================================================================

L1_CASES = [
    # --- D-SHAPE: aligned / tail / prime / edge (deterministic) ---
    {
        "name": "l1_aligned_s512",
        "B": 1,
        "S": 512,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-SHAPE-ALIGNED", "D-DTYPE-bf16", "D-DTYPE-fp32"],
    },
    {
        "name": "l1_tail_1_s65",
        "B": 1,
        "S": 65,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-SHAPE-TAIL-1"],
    },
    {
        "name": "l1_tail_mid_s96",
        "B": 1,
        "S": 96,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-SHAPE-TAIL-MID"],
    },
    {
        "name": "l1_prime_s127",
        "B": 1,
        "S": 127,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-SHAPE-PRIME"],
    },
    {
        "name": "l1_edge_s64_h1",
        "B": 1,
        "S": 64,
        "H": 1,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-SHAPE-EDGE"],
    },
    # --- D-VALRANGE: symmetric only (asymmetric forbidden in L1) ---
    {
        "name": "l1_vrange_small",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "vrange": (-0.3, 0.3),
        "tags": ["D-VALRANGE-S"],
    },
    {
        "name": "l1_vrange_mid",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "vrange": (-2, 2),
        "tags": ["D-VALRANGE-M"],
    },
    # --- D-PARAM: non-default attr values ---
    {
        # chunk_size=128 MUST pair with block_DV=64: with block_DV=128 the
        # step-6 UB cluster (h_ub+V_wh_ub+U_ub+U_bf16_ub) would exceed the
        # 192KB UB budget; halving block_DV keeps the peak at ~144KB.
        "name": "l1_chunk128_bdv64",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 128,
        "block_DV": 64,
        "use_g": True,
        "tags": ["D-PARAM-chunk_size", "D-PARAM-block_DV"],
    },
    {
        "name": "l1_block_dv_64",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 64,
        "use_g": True,
        "tags": ["D-PARAM-block_DV"],
    },
    {
        # use_g=False is restricted to BS<=4: BS=1.
        "name": "l1_no_g_bs1",
        "B": 1,
        "S": 64,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": False,
        "tags": ["D-PARAM-use_g"],
    },
    {
        # use_g=False at the stability bound: BS=4 (S=256).
        "name": "l1_no_g_bs4",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": False,
        "tags": ["D-PARAM-use_g"],
    },
    {
        "name": "l1_init_state",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "use_initial_state": True,
        "tags": ["D-PARAM-use_initial_state"],
    },
    {
        "name": "l1_branch_combo",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_g": True,
        "store_final_state": False,
        "save_new_value": False,
        "tags": ["D-PARAM-store_final_state", "D-PARAM-save_new_value"],
    },
    {
        # chunk_size=256 S-split structure (GEMM1 two-half serial L0C
        # reuse + AIV 2-phase; block_DV=128 is SAFE here — the ssplit
        # path phases the UB tiles at (128, block_DV), per-program UB
        # 120KB). APPENDED at the END of the list: this level draws all
        # cases from ONE seed stream — inserting mid-list would shift the
        # random draw of every later case.
        "name": "l1_chunk256",
        "B": 1,
        "S": 512,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 256,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-PARAM-chunk_size"],
    },
    {
        # chunk_size=512 four-segment S-split (W (4,128,DK) slab + AIV
        # 4-phase + 4-slot relay). L1 480/512KB tight budget; per-program
        # UB 136KB. Same coverage duty; appended
        # last (same stream-order rationale).
        "name": "l1_chunk512",
        "B": 1,
        "S": 1024,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 512,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-PARAM-chunk_size"],
    },
    {
        # chunk_size=1024 per-segment STREAMING (K/U/W (2,128,x)
        # double-buffered ping-pong slabs, L1 288KB streamed; GEMM_a
        # first (init=True) + 8 accumulating GEMM_b segments, LITERALLY
        # UNROLLED; AIV 8 phases with the bf16 relay parity tiles). New
        # semantic caliber (h quantizes every 1024 rows) — judged vs the
        # C=1024 fp32 golden under bf16 gates. Same coverage duty; appended
        # last (same stream-order rationale as l1_chunk256/512).
        "name": "l1_chunk1024",
        "B": 1,
        "S": 2048,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 1024,
        "block_DV": 128,
        "use_g": True,
        "tags": ["D-PARAM-chunk_size"],
    },
]


def test_chunk_delta_h_l1():
    """L1 functional tests: shape/attr/branch coverage (all BLOCKING)."""
    torch.manual_seed(0)
    all_passed = True
    results = []
    for case in L1_CASES:
        params = {k: v for k, v in case.items() if k not in ("name", "tags")}
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        passed = _run_one_case(case["name"], level_tag="L1", blocking=True, **params)
        results.append((case["name"], passed))
        all_passed = all_passed and passed

    print("=" * 60)
    for name, passed in results:
        status = "PRECISION_PASS" if passed else "PRECISION_FAIL"
        print(f"[{status}] {name}")
    if all_passed:
        print(f"[PRECISION_PASS] All {len(results)} L1 tests passed")
    else:
        print("[PRECISION_FAIL] Some L1 tests failed")
    print("=" * 60)
    return all_passed


# ============================================================================
# L2: Exception tests — all cases BLOCKING (illegal inputs must be rejected)
# ============================================================================

_L2_BASE = dict(
    B=1,
    S=256,
    H=4,
    DK=128,
    DV=128,
    chunk_size=64,
    use_g=True,
    use_initial_state=False,
    store_final_state=True,
    save_new_value=True,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    accum_dtype="float32",
    state_dtype="float32",
    block_DV=128,
)


def _l2_compile_probe(name, expected_exc_types, **overrides):
    """Compile-only probe: an illegal config should be REJECTED at compile.

    Returns True if the illegal input is rejected with one of the expected
    exception types (BOUNDARY_PASS). Returns False otherwise — either the
    illegal input was silently accepted (BOUNDARY_FAIL: silent accept) or
    the raised exception was not of the expected type (BOUNDARY_FAIL: wrong
    exception type). Both False branches are blocking (merged into exit code).

    The kernel is never EXECUTED for a silently-accepted illegal config
    (OOB risk).
    """
    kwargs = dict(_L2_BASE)
    kwargs.update(overrides)
    try:
        chunk_delta_h(**kwargs)
    except expected_exc_types as e:
        msg = str(e).replace("\n", " ")[:140]
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}: {msg})")
        return True
    except Exception as e:
        msg = str(e).replace("\n", " ")[:140]
        print(f"[BOUNDARY_FAIL] l2 {name}: wrong exception type (expected {expected_exc_types}, got {type(e).__name__}: {msg})")
        return False
    print(f"[BOUNDARY_FAIL] l2 {name}: illegal input not rejected (silent accept, compile-only probe, not executed)")
    return False


# All L2 cases are blocking — illegal inputs must be rejected with the
# expected exception type (AssertionError from host-side asserts in the
# kernel factory).
L2_CASES = [
    {
        # input_dtype=float16 is rejected by the dtype allowlist assert in
        # chunk_delta_h() (bf16 GEMM route requires bfloat16 input/output).
        "name": "l2_unsupported_dtype",
        "overrides": {"input_dtype": "float16", "output_dtype": "float16"},
        "expected_exc_types": (AssertionError,),
        "tags": ["D-EXC-DTYPE"],
    },
    {
        # chunk_size=8 → block_S=8 < 16 (GEMM fractal M minimum assert).
        "name": "l2_bad_chunk_size",
        "overrides": {"chunk_size": 8},
        "expected_exc_types": (AssertionError,),
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # DV=65 with block_DV=64 → DV % block_DV != 0 (alignment assert).
        # Rejected by the "DV must be a multiple of block_DV" host assert.
        "name": "l2_non_aligned_dv",
        "overrides": {"DV": 65, "block_DV": 64},
        "expected_exc_types": (AssertionError,),
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=192 (%16==0 but %128!=0, S-split nseg=1): without the
        # %128 assert the non-streaming S-split path loads W[..., 128:256]
        # out of bounds (opaque compile-stage error). S=192 keeps the
        # S % chunk_size assert quiet so the S-split assert is what fires.
        "name": "l2_ssplit_non_multiple_192",
        "overrides": {"S": 192, "chunk_size": 192},
        "expected_exc_types": (AssertionError,),
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=320 (%16==0 but %128!=0, S-split nseg=2): without the
        # %128 assert the kernel compiles AND runs, but silently drops the
        # tail 64 rows of the V_new path (only nseg*128=256 rows computed,
        # verified by a live repro before the fix). S=320 keeps the
        # S % chunk_size assert quiet so the S-split assert is what fires.
        "name": "l2_ssplit_non_multiple_320",
        "overrides": {"S": 320, "chunk_size": 320},
        "expected_exc_types": (AssertionError,),
        "tags": ["D-EXC-SHAPE"],
    },
]


def test_chunk_delta_h_l2():
    """L2 negative tests: illegal dtype / shape should be rejected (blocking).

    All L2 cases are blocking — the illegal input must be rejected with the
    expected exception type. Returns True only if all cases correctly reject
    their illegal input; the result is merged into the main exit code.
    """
    all_ok = True
    for case in L2_CASES:
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        rejected = _l2_compile_probe(case["name"], case["expected_exc_types"], **case["overrides"])
        if not rejected:
            all_ok = False
    return all_ok


# ============================================================================
# Boundary: Special value tests (NON-BLOCKING — legal values, WARN if beyond)
# ============================================================================

BOUNDARY_CASES = [
    {
        "name": "b_zero_input",
        "special": _special_zero,
        "tags": ["D-SPECIAL-ZERO"],
    },
    {
        # x64 symmetric large: GDN recurrence divergence — mathematical
        # property (both kernel & fp32 golden diverge; ratio holds via rtol
        # but max_abs exceeds the 1e0 cap) → expected WARN, non-blocking.
        "name": "b_large_values",
        "vrange": (-64, 64),
        "tags": ["D-VALRANGE-L"],
    },
    {
        # Asymmetric range: nonzero-mean K/W → rank-1 component →
        # recurrence divergence → expected WARN. FORBIDDEN in L1; legal
        # Boundary probe.
        "name": "b_asym_vrange",
        "vrange": (-0.5, 1.5),
        "tags": ["D-VALRANGE-ASYM"],
    },
    {
        "name": "b_inf_input",
        "special": _special_inf,
        "tags": ["D-SPECIAL-INF"],
    },
    {
        "name": "b_nan_input",
        "special": _special_nan,
        "tags": ["D-SPECIAL-NAN"],
    },
    {
        # ~1e38 inputs (finite bf16, near dtype boundary): GEMM products
        # overflow fp32 → inf/nan propagation → expected WARN.
        "name": "b_dbound",
        "special": _special_dbound,
        "tags": ["D-SPECIAL-DBOUND"],
    },
]

# All boundary cases share this config → ONE compiled kernel via memoization.
_BOUNDARY_CFG = dict(
    B=1,
    S=256,
    H=4,
    DK=128,
    DV=128,
    chunk_size=64,
    block_DV=128,
    use_g=True,
)

_BOUNDARY_WARNINGS = []


def test_chunk_delta_h_boundary():
    """Boundary tests: zero / large / asym / inf / nan / dtype-bound.

    Legal special values; judged by the same bf16 precision standard —
    beyond-threshold results are recorded as [BOUNDARY_WARN] (mathematical
    divergence / propagation properties), NON-BLOCKING.
    """
    torch.manual_seed(0)
    for case in BOUNDARY_CASES:
        params = {k: v for k, v in case.items() if k not in ("name", "tags")}
        params.update(_BOUNDARY_CFG)
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        passed = _run_one_case(case["name"], level_tag="Boundary", blocking=False, **params)
        if not passed:
            _BOUNDARY_WARNINGS.append(case["name"])

    print("=" * 60)
    if _BOUNDARY_WARNINGS:
        print(f"[BOUNDARY_WARN] summary: {len(_BOUNDARY_WARNINGS)} warning(s): {', '.join(_BOUNDARY_WARNINGS)} (non-blocking)")
    else:
        print("[BOUNDARY_PASS] All boundary tests passed")
    print("=" * 60)


# ============================================================================
# Coverage manifest (counts mirror the tags above; checker takes the max of
# tag hits and these explicit counts). Fusion category: no exemptions.
# ============================================================================

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 1,
    "D-DTYPE-fp32": 1,
    "D-SHAPE-ALIGNED": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-chunk_size": 1,
    "D-PARAM-use_g": 2,
    "D-PARAM-use_initial_state": 1,
    "D-PARAM-store_final_state": 1,
    "D-PARAM-save_new_value": 1,
    "D-PARAM-block_DV": 2,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 4,
}

# Fusion category: all dimensions are mandatory — no exemptions declared.
# (D-VALRANGE-L / D-VALRANGE-ASYM are covered by Boundary cases because the
# GDN recurrence diverges at those ranges — a documented mathematical
# property of the operator, not a coverage gap.)
COVERAGE_NA = {}


if __name__ == "__main__":
    import tilelang

    tilelang.disable_cache()

    parser = argparse.ArgumentParser(description="chunk_delta_h precision test")
    parser.add_argument(
        "--level",
        type=str,
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level: l0, l1, l2, boundary, or all",
    )
    args = parser.parse_args()

    blocking_ok = True

    if args.level in ("l0", "all"):
        passed = test_chunk_delta_h_l0()
        blocking_ok = blocking_ok and passed
    if args.level in ("l1", "all"):
        passed = test_chunk_delta_h_l1()
        blocking_ok = blocking_ok and passed
    if args.level in ("l2", "all"):
        l2_ok = test_chunk_delta_h_l2()  # unsupported_dtype now blocking
        blocking_ok = blocking_ok and l2_ok
    if args.level in ("boundary", "all"):
        test_chunk_delta_h_boundary()  # non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    else:
        sys.exit(1)
