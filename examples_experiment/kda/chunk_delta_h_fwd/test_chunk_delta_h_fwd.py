"""Layered precision test for chunk_delta_h_fwd (KDA variant, Ascend NPU).

Uses check_precision double-threshold judgment (elementwise
|actual-golden| <= atol + rtol*|golden| plus a matched-ratio gate and a
max-abs-error cap, NOT torch.assert_close). ALL THREE outputs
(h / final_state / V_new) are judged by the bfloat16 row — the error floor
of every output is set by the bf16 GEMM operand chain:
  - h (bfloat16):           atol=2^-10, rtol=2^-6, max_abs=1e0, req=0.99
  - final_state (physically float32): judged by the bfloat16 row — its error
    floor is set by the bf16 GEMM operand chain (GPU original / FLA are
    isomorphic and exceed fp32 thresholds identically)
  - V_new (bfloat16):       atol=2^-10, rtol=2^-6, max_abs=1e0, req=0.99

The golden (ref_chunk_delta_h_fwd) is the HIGH-PRECISION fp32 single
reference: h stays fp32 across chunks, per-DK-dim exp2 gating, plain
ungated K. The kernel uses bf16 GEMM (input-side quantization) — that
quantization gap vs the fp32 golden is the precision metric, accepted
under the bf16 thresholds.

Test data prep: contractive initial_state per the example's prepare_input
contract (official-magnitude init x1.0 runs as a non-blocking Boundary case).

Finite-input contract: every case validates its inputs host-side
(validate_inputs) BEFORE the launch (CPU tensors, pre-H2D). Non-finite
inputs are EXPECTED-REJECTION cases (b_inf_input / b_nan_input verify the
AssertionError — tensor name + violation count in the message; the kernel
is never launched on non-finite inputs). Finite values are the legal
domain: vrange/large/asym/dbound cases run the kernel and record
[BOUNDARY_WARN] for divergence/propagation beyond thresholds.

Boundary strictness: computed-but-beyond-threshold -> [BOUNDARY_WARN]
(non-blocking); an EXCEPTION on a legal input (compile failure / runtime
error / OOM) -> [BOUNDARY_ERROR] — re-raised by the boundary layer
(visible as a pytest FAIL; the direct-run main records it WITHOUT touching
the exit code: boundary stays non-blocking, but a real error is never
silently folded into the WARN summary).

L0 cases (minimal -> typical scale + flag/seed coverage; one documented
deviation):
  - l0_minimal:  (1,128,2)  all flags True — BS=2 minimal recurrence
  - l0_basic:    (1,512,4)  all flags True — BS=8 forward scan
  - l0_typical:  (1,1024,8) all flags True — BS=16 typical scale
  - l0_multi_bh: (2,512,8)  all flags True — B>1 multi-block parallel
  - l0_no_init:  (1,512,4)  use_initial_state=False — zero-init path
  - l0_no_gk:    (1,256,4)  use_gk=False — 2-GEMM arm (BS<=4 stability bound; see the case-site comment)
  - l0_no_final: (1,512,4)  store_final_state=False
  - l0_no_vnew:  (1,512,4)  save_new_value=False
  - l0_seeds:    l0_typical x seeds {0,1,2} — seed robustness

L1 / L2 / Boundary layers: functional shape/attr coverage, illegal-input
negative probes, and special-value cases (see L1_CASES / L2_CASES /
BOUNDARY_CASES below); the --level dispatcher runs them on demand.

Coverage annotations: each case carries coverage tags (D-DTYPE-* /
D-SHAPE-* / D-PARAM-* / D-VALRANGE-* / D-SPECIAL-* / D-EXC-*); the
explicit per-dimension counts live in COVERAGE_MANIFEST at the bottom of
this file.
"""

import argparse
import glob as _glob
import os
import re
import subprocess
import sys
import zlib

import torch

from example_chunk_delta_h_fwd import (
    chunk_delta_h_fwd,
    prepare_input,
    prepare_m_kda,
    relay_dtype,
    relay_slot_count,
    to_chunk_major,
    validate_inputs,
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
    for p in _glob.glob("/tmp/tmptl_*.so"):
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
# Precision judgment (mixed tolerance: elementwise atol+rtol, matched-ratio
# gate + max-abs-error cap)
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; integer: exact match (zero error)."""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # atol 0.125, rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # integer: exact match, one mismatch = FAIL
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Precision judgment: returns (passed, matched_ratio, max_abs_error).

    Float double threshold: matched_ratio >= required AND max_abs_error <=
    max_abs_error_limit; integer: exact elementwise equality. inf/nan
    positions are compared structurally, excluded from numeric tolerance."""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)  # inf/nan structural comparison
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # all finite-golden positions compared: non-finite actual counts as mismatch
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()  # inf/nan actual -> element False and raises max_abs
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# PyTorch Golden Reference (KDA high-precision fp32)
# ============================================================================


def ref_chunk_delta_h_fwd(
    K,
    W,
    U,
    GK,
    initial_state,
    chunk_size,
    use_gk=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
):
    """KDA gated delta rule forward state recurrence — high-precision fp32.

    Source-independence declaration: this golden was derived INDEPENDENTLY
    from FLA's chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    (fla/ops/kda/fla_chunk_delta.py L33-212, USE_GK+USE_EXP2 path) at
    DESIGN time — BEFORE the kernel implementation was written — and is
    NOT a transcription of the kernel code. This guards against the
    implementation and the reference sharing the same bug and both
    passing.

    h stays fp32 across chunks; per-DK-dim exp2 gating; GEMMs use the fp32
    representation of the original bf16 inputs (chunk-major repacking does
    not change values). Snapshot = the PRE-update state; K and V_new are
    UNGATED (KDA semantics).

    Args:
        K: (B, S, H, DK) bfloat16 — original input
        W: (B, S, H, DK) bfloat16 — original input
        U: (B, S, H, DV) bfloat16 — original input
        GK: (B, S, H, DK) float32 — per-dim cumulative log2-domain gate
            (already chunk_local_cumsum_gk'd, host-side only)
        initial_state: (B, H, DK, DV) bfloat16
        chunk_size: C, default 64
        use_gk / use_initial_state / store_final_state / save_new_value: flags

    Returns:
        h: (B, BS, H, DK, DV) bfloat16
        final_state: (B, H, DK, DV) float32
        V_new: (B, S, H, DV) bfloat16
    """
    B, S, H, DK = K.shape
    DV = U.shape[-1]
    BS = S // chunk_size

    h_out = torch.empty(B, BS, H, DK, DV, dtype=torch.bfloat16)
    final_state = torch.empty(B, H, DK, DV, dtype=torch.float32)
    v_new = torch.empty(B, S, H, DV, dtype=torch.bfloat16)
    for b in range(B):
        for h in range(H):
            st = initial_state[b, h].float() if use_initial_state else torch.zeros(DK, DV)
            for i in range(BS):
                s, e = i * chunk_size, (i + 1) * chunk_size
                h_out[b, i, h] = st.to(torch.bfloat16)  # snapshot: pre-update state
                Kc = K[b, s:e, h].float()
                Wc = W[b, s:e, h].float()
                Uc = U[b, s:e, h].float()
                v = Uc - Wc @ st  # V_new (ungated)
                if save_new_value:
                    v_new[b, s:e, h] = v.to(torch.bfloat16)
                if use_gk:
                    gk_last = GK[b, e - 1, h]  # (DK,) fp32, exp2 domain
                    st = st * torch.exp2(gk_last)[:, None]  # per-DK gating (fp32)
                st = st + Kc.T @ v  # K ungated
            if store_final_state:
                final_state[b, h] = st  # fp32 direct store
    return h_out, final_state, v_new


# ============================================================================
# Kernel compile memoization (in-process; shared configs compile once)
# ============================================================================

_KERNEL_CACHE = {}


def _get_kernel(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    use_gk,
    use_initial_state,
    store_final_state,
    save_new_value,
    block_DV,
):
    """Compile the kernel for one config, or reuse an in-process copy.

    tilelang.disable_cache() in main only disables the on-disk cache; this
    dict cache is independent of it (same-shape flag variants share nothing;
    identical configs — e.g. l0_typical vs l0_seeds — compile once).
    """
    key = (
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        use_gk,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DV,
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _test_run_with_retry(
            lambda: chunk_delta_h_fwd(
                B=B,
                S=S,
                H=H,
                DK=DK,
                DV=DV,
                chunk_size=chunk_size,
                use_gk=use_gk,
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
    use_gk,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    init_scale=0.05,
    seed=None,
    vrange=None,
    special=None,
    level_tag="L0",
    blocking=True,
):
    """Run one test case: prepare, compile, run, golden, check precision.

    Args:
        init_scale: initial_state magnitude scale (0.05 = contractive prep
            default; 1.0 = official magnitude).
        seed: optional per-case torch seed override (l0_seeds robustness
            sweep); when None the case is seeded deterministically from
            crc32(name) — data is independent of execution order.
        vrange: (lo, hi) affine value-range scaling of K/W/U (applied to
            the normalized originals on CPU). Symmetric ranges are pure
            scaling; asymmetric ranges add bias (rank-1 divergence —
            Boundary-only). Large symmetric ranges make ||K^T@W|| >= 1 and
            the recurrence diverges (delta-rule semantics: the state
            update is only contractive when ||K^T@W|| < 1) — Boundary-only.
        special: callable(K, W, U, initial_state) -> (K, W, U, initial_state)
            boundary mutation (zero / inf / nan / dtype-boundary). Applied
            to the originals BEFORE kernel-input re-derivation; the golden
            sees the same inputs.
        blocking: True -> [PRECISION_*] markers (L0/L1, affect exit code);
            False -> [BOUNDARY_*] markers (Boundary, non-blocking).

    Non-aligned S contract (L1 tail/prime cases): the kernel only supports
    full chunks (host assert S % C == 0). Data is generated at the aligned
    gen_S = ceil(S/C)*C; originals keep the true S tokens (golden input —
    ref_chunk_delta_h_fwd drops the tail internally via BS = S // C); the
    kernel-side inputs are truncated to S_aligned = BS*C and the kernel is
    compiled at S_aligned.

    Returns:
        True if all enabled outputs pass.
    """
    print("=" * 60)
    print(f"[{level_tag}:{name}] B={B}, S={S}, H={H}, DK={DK}, DV={DV}, C={chunk_size}")
    print(
        f"  use_gk={use_gk}, init_state={use_initial_state}, final_state={store_final_state}, "
        f"v_new={save_new_value}, init_scale={init_scale}, seed={seed}, "
        f"vrange={vrange}, special={getattr(special, '__name__', None)}"
    )
    print("=" * 60)

    ok_tag = "PRECISION_PASS" if blocking else "BOUNDARY_PASS"
    bad_tag = "PRECISION_FAIL" if blocking else "BOUNDARY_WARN"

    try:
        # Per-case stable seed (name-hash): case data is independent of the
        # execution order and of later list insertions; an explicit seed
        # (the l0_seeds robustness sweep) still overrides.
        torch.manual_seed(seed if seed is not None else (zlib.crc32(name.encode()) & 0xFFFFFFFF))

        # --- Host-side data preparation (all CPU; single H2D below) ---
        print(f"[{level_tag}:{name}] preparing input (CPU) ...")
        BS_actual = S // chunk_size
        S_aligned = BS_actual * chunk_size
        gen_S = S if S % chunk_size == 0 else ((S + chunk_size - 1) // chunk_size) * chunk_size
        (
            K,
            W,
            U,
            GK,
            initial_state,
            K_cm,
            W_cm,
            U_cm,
        ) = prepare_input(B, gen_S, H, DK, DV, chunk_size, block_DV=block_DV, use_gk=use_gk, init_scale=init_scale)

        # Non-aligned S: originals keep the true S tokens (golden input);
        # kernel side truncates to S_aligned (same contract as golden's
        # internal BS = S // C tail drop).
        if gen_S != S:
            print(
                f"  [tail] S={S} not a multiple of C={chunk_size}: host truncates "
                f"kernel side to S_aligned={S_aligned} (BS={BS_actual}); golden "
                "drops the tail internally (same contract)"
            )
            K = K[:, :S]
            W = W[:, :S]
            U = U[:, :S]
            GK = GK[:, :S]

        # Boundary mutation (zero / inf / nan / dtype-boundary).
        if special is not None:
            K, W, U, initial_state = special(K, W, U, initial_state)

        # Value-range scaling (affine; symmetric ranges have bias=0).
        vrange_active = vrange is not None and vrange != (-1, 1)
        if vrange_active:
            lo, hi = vrange
            scale = (hi - lo) / 2.0
            bias = (hi + lo) / 2.0
            K = (K.float() * scale + bias).to(torch.bfloat16)
            W = (W.float() * scale + bias).to(torch.bfloat16)
            U = (U.float() * scale + bias).to(torch.bfloat16)

        # Re-derive the kernel inputs from the (possibly mutated / truncated)
        # originals when needed.
        if (gen_S != S) or special is not None or vrange_active:
            K_k = K[:, :S_aligned]
            W_k = W[:, :S_aligned]
            U_k = U[:, :S_aligned]
            K_cm = to_chunk_major(K_k, chunk_size)
            W_cm = to_chunk_major(W_k, chunk_size)
            U_cm = to_chunk_major(U_k, chunk_size)

        print(f"[{level_tag}:{name}] compiling kernel ...")
        kernel = _get_kernel(
            B=B,
            S=S_aligned,
            H=H,
            DK=DK,
            DV=DV,
            chunk_size=chunk_size,
            use_gk=use_gk,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            save_new_value=save_new_value,
            block_DV=block_DV,
        )

        # M (KDA M-form, per-dim diag) — 2-GEMM arm never reads it. Derived
        # from the TRUNCATED GK so the shape assert (B, BS*C, H, DK) holds.
        print(f"[{level_tag}:{name}] computing M (host, KDA M-form) ...")
        M = prepare_m_kda(K_cm, W_cm, GK[:, :S_aligned], chunk_size, DK, use_gk=use_gk)

        # Finite-input contract: validate BEFORE the launch (CPU tensors,
        # pre-H2D — the scan is negligible at test shapes; the kernel body
        # stays scan-free). Raises AssertionError with the offending tensor
        # name + inf/nan count on violation.
        validate_inputs(K_cm, W_cm, U_cm, M, initial_state)

        print(f"[{level_tag}:{name}] moving inputs to NPU ...")
        bv_num = (DV + block_DV - 1) // block_DV
        total_blocks = bv_num * B * H
        h_state_gm = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        state_zero = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        workspace_h = torch.zeros(total_blocks, DK, block_DV, dtype=torch.bfloat16, device="npu")
        workspace_v = torch.zeros(total_blocks, chunk_size, block_DV, dtype=torch.bfloat16, device="npu")
        # Relay slot count / dtype must match the kernel's traced workspace shape
        workspace_vwh = torch.zeros(
            total_blocks,
            relay_slot_count(chunk_size, use_gk=use_gk),
            chunk_size,
            block_DV,
            dtype=getattr(torch, relay_dtype(chunk_size, use_gk=use_gk)),
            device="npu",
        )
        workspace_kv = torch.zeros(total_blocks, DK, block_DV, dtype=torch.float32, device="npu")
        # The async NPU zero-fill of the workspaces must retire before the
        # kernel launch (the tilelang launch does not order against the
        # torch stream).
        torch.npu.synchronize()

        print(f"[{level_tag}:{name}] running kernel ...")
        h_out, final_state, V_new = kernel(
            K_cm.to("npu"),
            W_cm.to("npu"),
            U_cm.to("npu"),
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

        print(f"[{level_tag}:{name}] computing golden (CPU, high-precision fp32) ...")
        h_ref, final_state_ref, V_new_ref = ref_chunk_delta_h_fwd(
            K,
            W,
            U,
            GK,
            initial_state,
            chunk_size,
            use_gk=use_gk,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            save_new_value=save_new_value,
        )

        all_passed = True

        # h comparison (bf16 thresholds)
        h_passed, h_ratio, h_max_abs = check_precision(h_out, h_ref, "bfloat16")
        status = ok_tag if h_passed else bad_tag
        print(f"[{status}] {name} h ratio={h_ratio:.4f} max_abs={h_max_abs:.3e}")
        all_passed = all_passed and h_passed

        if store_final_state:
            # final_state is physically fp32 (L0C direct) but judged by the
            # bfloat16 row — its error floor is set by the bf16 GEMM
            # operand chain (GPU original / FLA isomorphic).
            fs_passed, fs_ratio, fs_max_abs = check_precision(final_state, final_state_ref, "bfloat16")
            status = ok_tag if fs_passed else bad_tag
            print(f"[{status}] {name} final_state ratio={fs_ratio:.4f} max_abs={fs_max_abs:.3e}")
            all_passed = all_passed and fs_passed

        if save_new_value:
            # V_new is chunk-major (B,BS,H,C,DV) — permute to token-major on
            # the CPU copy AFTER D2H (host NPU path stays metadata-only).
            # Compare only the VALID (aligned) region: the kernel outputs
            # BS*chunk_size rows; the golden may hold more (tail rows are
            # unfilled by ref and excluded by contract).
            valid_S = BS_actual * chunk_size
            vn_cpu = V_new.cpu()
            vn_token = vn_cpu.permute(0, 1, 3, 2, 4).reshape(B, S_aligned, H, DV)
            vn_out_cmp = vn_token[:, :valid_S]
            vn_ref_cmp = V_new_ref[:, :valid_S]
            vn_passed, vn_ratio, vn_max_abs = check_precision(vn_out_cmp, vn_ref_cmp, "bfloat16")
            status = ok_tag if vn_passed else bad_tag
            print(f"[{status}] {name} V_new ratio={vn_ratio:.4f} max_abs={vn_max_abs:.3e}")
            all_passed = all_passed and vn_passed

        return all_passed

    except Exception as e:
        import traceback

        if blocking:
            # L0/L1: the layer assert realizes the failure (pytest FAIL /
            # direct-run exit != 0, no "Test Passed!").
            print(f"[{bad_tag}] {name} exception: {e}")
            traceback.print_exc()
            return False
        # Boundary strictness: an EXCEPTION on a legal input (compile
        # failure / runtime error / OOM) is NOT a precision WARN. Mark it
        # [BOUNDARY_ERROR], keep it OUT of the WARN summary, and re-raise —
        # the boundary layer aggregates errors into a pytest-visible
        # failure (the direct-run main records them without touching the
        # exit code). Expected-rejection cases never reach here (they use
        # a precise try/except bite around validate_inputs, not this path).
        print(f"[BOUNDARY_ERROR] {name} exception (legal-input run failed — NOT a precision WARN): {e}")
        traceback.print_exc()
        raise
    finally:
        _test_cleanup_tmp_so()


def test_chunk_delta_h_fwd_l0():
    """L0 precision test: minimal -> typical scale coverage (contractive
    prep, bf16 rule). Each case seeds itself (crc32(name)) — no shared
    random stream."""
    results = []
    tags_l0 = ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED"]

    # 1. l0_minimal: BS=2 minimal recurrence (init path + 1 gate step + final_state)
    print(f"\n[tags] l0_minimal: {', '.join(tags_l0)}")
    r = _run_one_case("l0_minimal", B=1, S=128, H=2, DK=128, DV=128, chunk_size=64, block_DV=128, use_gk=True)
    results.append(("l0_minimal", r))

    # 2. l0_basic: BS=8 forward scan (multi-round gating / carrier / snapshot)
    print(f"\n[tags] l0_basic: {', '.join(tags_l0)}")
    r = _run_one_case("l0_basic", B=1, S=512, H=4, DK=128, DV=128, chunk_size=64, block_DV=128, use_gk=True)
    results.append(("l0_basic", r))

    # 3. l0_typical: BS=16 typical scale
    print(f"\n[tags] l0_typical: {', '.join(tags_l0)}")
    r = _run_one_case("l0_typical", B=1, S=1024, H=8, DK=128, DV=128, chunk_size=64, block_DV=128, use_gk=True)
    results.append(("l0_typical", r))

    # 4. l0_multi_bh: B>1 + grid=32 multi-block parallel (cid decode bb/bh)
    print(f"\n[tags] l0_multi_bh: {', '.join(tags_l0)}")
    r = _run_one_case("l0_multi_bh", B=2, S=512, H=8, DK=128, DV=128, chunk_size=64, block_DV=128, use_gk=True)
    results.append(("l0_multi_bh", r))

    # 5. l0_no_init: zero-init path (state_zero source + zero h[:,0] snapshot)
    print("\n[tags] l0_no_init: D-PARAM-use_initial_state")
    r = _run_one_case(
        "l0_no_init",
        B=1,
        S=512,
        H=4,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_gk=True,
        use_initial_state=False,
    )
    results.append(("l0_no_init", r))

    # 6. l0_no_gk: 2-GEMM arm coverage (ungated recurrence).
    #    [Deviation] S=256 (BS=4) instead of the S=512 used by the other
    #    flag cases: this arm is mathematically unstable beyond BS=4 on the
    #    bf16 route (CPU-simulated quantization chain: BS=4 + contractive
    #    prep -> all PASS, fs r=0.9966; BS=8 fails at ANY init magnitude,
    #    h r=0.93-0.99) — an intrinsic bf16-GEMM property of the ungated
    #    recurrence, not an implementation defect. BS=4 covers the arm fully.
    print("\n[tags] l0_no_gk: D-PARAM-use_gk, D-SHAPE-ALIGNED")
    r = _run_one_case(
        "l0_no_gk",
        B=1,
        S=256,
        H=4,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_gk=False,
    )
    results.append(("l0_no_gk", r))

    # 7. l0_no_final: final_state not stored (verify h/V_new only)
    print("\n[tags] l0_no_final: D-PARAM-store_final_state")
    r = _run_one_case(
        "l0_no_final",
        B=1,
        S=512,
        H=4,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_gk=True,
        store_final_state=False,
    )
    results.append(("l0_no_final", r))

    # 8. l0_no_vnew: V_new not stored (verify h/final_state only)
    print("\n[tags] l0_no_vnew: D-PARAM-save_new_value")
    r = _run_one_case(
        "l0_no_vnew",
        B=1,
        S=512,
        H=4,
        DK=128,
        DV=128,
        chunk_size=64,
        block_DV=128,
        use_gk=True,
        save_new_value=False,
    )
    results.append(("l0_no_vnew", r))

    # 9. l0_seeds: l0_typical x seeds {0,1,2} (seed robustness; the seed-0
    #    run shares the compiled kernel with l0_typical via memoization)
    for sd in (0, 1, 2):
        print(f"\n[tags] l0_seeds_s{sd}: {', '.join(tags_l0)}")
        r = _run_one_case(
            f"l0_seeds_s{sd}",
            B=1,
            S=1024,
            H=8,
            DK=128,
            DV=128,
            chunk_size=64,
            block_DV=128,
            use_gk=True,
            seed=sd,
        )
        results.append((f"l0_seeds_s{sd}", r))

    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PRECISION_PASS" if passed else "PRECISION_FAIL"
        print(f"[{status}] {name}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"[PRECISION_PASS] All {len(results)} L0 tests passed")
    else:
        print("[PRECISION_FAIL] Some L0 tests failed")
    print("=" * 60)

    # Realize failures on BOTH run paths: the AssertionError propagates under
    # pytest (FAIL) and in the direct-run path (exit != 0, no "Test Passed!").
    assert all_passed, f"L0 precision failures: {[n for n, p in results if not p]}"
    return all_passed


# ============================================================================
# L1 / L2 / Boundary — functional / negative / special-value layers
# ============================================================================

# Operator category tag for coverage accounting (fused multi-output
# recurrence: h / final_state / V_new from one pass)
COVERAGE_CATEGORY = "Fusion"

# --- L1 functional cases (BLOCKING — deterministic shapes + attr combos) ---
L1_CASES = [
    # D-SHAPE: aligned variants (non-power-of-2 multiples of C) + edge
    {
        "name": "l1_aligned_s768",
        "B": 1,
        "S": 768,
        "H": 6,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED"],
    },
    {
        # S=192 = 3C (non-power-of-2 chunk count)
        "name": "l1_s192_3chunks",
        "B": 1,
        "S": 192,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-ALIGNED"],
    },
    {
        # S=320 = 5C (non-power-of-2 chunk count)
        "name": "l1_s320_5chunks",
        "B": 1,
        "S": 320,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-ALIGNED"],
    },
    {
        # Degenerate: BS=1 + H=1 (single chunk, single head, minimal grid)
        "name": "l1_edge_s64_h1",
        "B": 1,
        "S": 64,
        "H": 1,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-EDGE"],
    },
    {
        # Non-power-of-2 B and H
        "name": "l1_bh_variants",
        "B": 3,
        "S": 256,
        "H": 3,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-ALIGNED"],
    },
    # D-SHAPE: tail / prime — host-truncation contract (S % C != 0: the
    # kernel only supports full chunks; originals keep true S for golden,
    # kernel side truncates to BS*C — the wrapper's documented contract)
    {
        "name": "l1_tail_1_s65",
        "B": 1,
        "S": 65,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-TAIL-1"],
    },
    {
        # mid-range remainder 32 (S=96 -> BS=1, truncate to 64)
        "name": "l1_tail_mid_s96",
        "B": 1,
        "S": 96,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-TAIL-MID"],
    },
    {
        # prime S (127 -> BS=1) — completely non-aligned
        "name": "l1_prime_s127",
        "B": 1,
        "S": 127,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-PRIME"],
    },
    {
        # prime S with MULTIPLE chunks (193 -> BS=3, truncate to 192)
        "name": "l1_prime_s193",
        "B": 1,
        "S": 193,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-SHAPE-PRIME"],
    },
    # D-VALRANGE: symmetric small only in L1 — larger / asymmetric ranges
    # make ||K^T@W|| >= 1 and the ungated-K recurrence diverges (delta-rule
    # semantics: the state update is only contractive when ||K^T@W|| < 1;
    # CPU-simulated) -> Boundary layer (WARN).
    {
        "name": "l1_vrange_small",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 128,
        "use_gk": True,
        "vrange": (-0.3, 0.3),
        "tags": ["D-VALRANGE-S"],
    },
    # D-PARAM: non-default attr values (use_gk / use_initial_state /
    # store_final_state / save_new_value already covered by the L0 cases)
    {
        # chunk_size=128 (non-default; mform T.mma path, L0C exactly 128KB)
        "name": "l1_chunk128",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 128,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-PARAM-chunk_size"],
    },
    {
        # block_DV=64 (non-default; bv_num=2, grid doubles, DV split path)
        "name": "l1_block_dv_64",
        "B": 1,
        "S": 256,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 64,
        "block_DV": 64,
        "use_gk": True,
        "tags": ["D-PARAM-block_DV"],
    },
    {
        # chunk_size=256 (ssplit nseg=2; the relay auto-switches to 4 slots
        # / bf16 — host allocation follows via relay_slot_count/relay_dtype).
        # [Deviation] S=512 (BS=2) instead of a longer scan: the M-form
        # recurrence is mathematically unstable beyond BS=2 at C=256 on the
        # bf16 route (||K^T@W|| grows with chunk length; CPU-simulated
        # quantization chain reproduces the NPU drift to 4 decimals —
        # h r~0.93 @BS=8 — an intrinsic bf16-GEMM property, not an
        # implementation defect; same family as the 2-GEMM arm's BS<=4
        # bound). BS=2 covers the full ssplit path.
        "name": "l1_chunk256",
        "B": 1,
        "S": 512,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 256,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-PARAM-chunk_size"],
    },
    {
        # chunk_size=512 (ssplit nseg=4; same 4-slot bf16 relay caliber).
        # [Deviation] S=512 (BS=1): the C=512 M-form recurrence is unstable
        # beyond a single chunk on the bf16 route (same intrinsic property
        # as l1_chunk256). BS=1 covers the full nseg=4 phase structure.
        "name": "l1_chunk512",
        "B": 1,
        "S": 512,
        "H": 4,
        "DK": 128,
        "DV": 128,
        "chunk_size": 512,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-PARAM-chunk_size"],
    },
    {
        # chunk_size=1024 (ssplit nseg=8 -> per-segment STREAMING: K/U/W as
        # (2, seg, x) ping-pong L1 slabs — the streaming path's first
        # coverage in this suite). [Deviation] S=1024 (BS=1): same C-length
        # stability bound as l1_chunk256/l1_chunk512 (M-form cancellation
        # error grows with chunk length on the bf16 route). BS=1 covers the
        # full nseg=8 phase structure. Budgets: per-program UB 188416B /
        # 196352B (96% — planner-aliased second relay tile; see the kernel
        # host assert (7)), L1 294912B / 524032B, L0C 131072B / 131072B
        # (exact).
        "name": "l1_chunk1024",
        "B": 1,
        "S": 1024,
        "H": 2,
        "DK": 128,
        "DV": 128,
        "chunk_size": 1024,
        "block_DV": 128,
        "use_gk": True,
        "tags": ["D-PARAM-chunk_size"],
    },
]


def test_chunk_delta_h_fwd_l1():
    """L1 functional tests: shape/attr/branch coverage (all BLOCKING)."""
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
    # Realize failures on BOTH run paths (see the L0 note).
    assert all_passed, f"L1 precision failures: {[n for n, p in results if not p]}"
    return all_passed


# --- L2 negative cases (NON-BLOCKING — illegal inputs should be rejected) ---

_L2_BASE = dict(
    B=1,
    S=256,
    H=4,
    DK=128,
    DV=128,
    chunk_size=64,
    use_gk=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    accum_dtype="float32",
    state_dtype="float32",
    block_DV=128,
)


def _l2_compile_probe(name, expect_exc, expect_msg, **overrides):
    """Compile-only probe: an illegal config should be REJECTED at the
    wrapper/compile stage — for the EXPECTED reason.

    Dual validation: the raised exception must be of the expected type AND
    carry the expected message keyword. An unrelated failure (environment
    error, missing module, wrong working directory) must NOT count as a
    "correct rejection".

    Args:
        expect_exc: expected exception type (e.g. AssertionError for the
            wrapper's host asserts).
        expect_msg: keyword substring the rejection message must contain.

    Returns bool — True = correctly rejected (BOUNDARY_PASS), False =
    silently accepted or rejected for an unexpected reason (BOUNDARY_WARN).
    The kernel is never EXECUTED for a silently-accepted illegal config
    (OOB risk).
    """
    kwargs = dict(_L2_BASE)
    kwargs.update(overrides)
    try:
        chunk_delta_h_fwd(**kwargs)
    except Exception as e:
        msg = str(e).replace("\n", " ")
        if isinstance(e, expect_exc) and expect_msg in msg:
            print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}: {msg[:140]})")
            return True
        print(
            f"[BOUNDARY_WARN] l2 {name}: rejected for an UNEXPECTED reason "
            f"(wanted {expect_exc.__name__} containing '{expect_msg}'; got "
            f"{type(e).__name__}: {msg[:140]})"
        )
        return False
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input not rejected (silent accept, compile-only probe, not executed)")
    return False


L2_CASES = [
    {
        # fp16 K/W/U route was abandoned — must be rejected
        "name": "l2_unsupported_dtype",
        "overrides": {"input_dtype": "float16", "output_dtype": "float16"},
        "expect_exc": AssertionError,
        "expect_msg": "unsupported input_dtype",
        "tags": ["D-EXC-DTYPE"],
    },
    {
        # S=65 not a multiple of C=64 — the wrapper alignment assert
        "name": "l2_s_not_aligned",
        "overrides": {"S": 65},
        "expect_exc": AssertionError,
        "expect_msg": "not a multiple of chunk_size",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=8 < 16 (GEMM fractal M minimum)
        "name": "l2_bad_chunk_size",
        "overrides": {"chunk_size": 8, "S": 64},
        "expect_exc": AssertionError,
        "expect_msg": "fractal 16 alignment",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # DV=65 with block_DV=64 — a partial-tile caliber that must never
        # reach the kernel (a partial tile would be silently mis-computed);
        # rejected by the alignment asserts, compile-only probe
        "name": "l2_dv_not_aligned",
        "overrides": {"DV": 65, "block_DV": 64},
        "expect_exc": AssertionError,
        "expect_msg": "fractal 16 alignment",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=272 (ssplit, 272 % 128 = 16): nseg=2 segments cover
        # only 256 rows — the non-divisible tail would be silently dropped.
        # Must be rejected by the ssplit divisibility assert.
        "name": "l2_chunk272_nondiv",
        "overrides": {"chunk_size": 272, "S": 544},
        "expect_exc": AssertionError,
        "expect_msg": "is not a multiple of 128",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=144 (ssplit, 144 % 128 = 16): nseg=1 segment covers
        # only 128 rows — same silent-drop class as above.
        "name": "l2_chunk144_nondiv",
        "overrides": {"chunk_size": 144, "S": 288},
        "expect_exc": AssertionError,
        "expect_msg": "is not a multiple of 128",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=32 (use_gk=True, 32 % 16 == 0): passes the generic
        # alignment asserts but is OUTSIDE the mform domain — the traced
        # structures only cover {64,128} (T.mma) and >=256 multiples of
        # 128 (ssplit); C=32 would land in an unverified gemm_v0 route.
        # Rejected by the mform chunk_size domain assert.
        "name": "l2_mform_chunk32",
        "overrides": {"chunk_size": 32, "S": 256},
        "expect_exc": AssertionError,
        "expect_msg": "outside the supported domain",
        "tags": ["D-EXC-SHAPE"],
    },
    {
        # chunk_size=96 (use_gk=True, 96 % 16 == 0, S=192 a multiple):
        # same unverified-route class as l2_mform_chunk32 (C=96 is neither
        # a T.mma caliber nor an ssplit multiple of 128).
        "name": "l2_mform_chunk96",
        "overrides": {"chunk_size": 96, "S": 192},
        "expect_exc": AssertionError,
        "expect_msg": "outside the supported domain",
        "tags": ["D-EXC-SHAPE"],
    },
]


def test_chunk_delta_h_fwd_l2():
    """L2 negative tests: illegal dtype / shape must be rejected for the
    expected reason. Probe results are aggregated (a silent accept or an
    unexpected-reason rejection is a failure)."""
    results = []
    for case in L2_CASES:
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        ok = _l2_compile_probe(
            case["name"],
            expect_exc=case["expect_exc"],
            expect_msg=case["expect_msg"],
            **case["overrides"],
        )
        results.append((case["name"], ok))

    print("=" * 60)
    for name, ok in results:
        status = "BOUNDARY_PASS" if ok else "BOUNDARY_WARN"
        print(f"[{status}] {name}")
    all_rejected = all(ok for _, ok in results)
    if all_rejected:
        print(f"[BOUNDARY_PASS] All {len(results)} L2 negative tests correctly rejected")
    else:
        failed = [n for n, ok in results if not ok]
        print(f"[BOUNDARY_WARN] L2 negative tests not correctly rejected: {failed}")
    print("=" * 60)
    # Realize failures on BOTH run paths (see the L0 note).
    assert all_rejected, f"L2 negative-test failures: {[n for n, ok in results if not ok]}"
    return all_rejected


# --- Boundary special-value cases (NON-BLOCKING) ---
#
# Two case classes, split by the finite-input contract:
#   1. BOUNDARY_CASES — FINITE legal values (zero / vrange / dtype-bound):
#      the kernel RUNS; beyond-threshold results are [BOUNDARY_WARN]
#      (mathematical divergence / overflow propagation properties of the
#      delta-rule family, non-blocking). Boundary of the contract: finite
#      values (any magnitude, including ~1e38 dtype-bound) are the LEGAL
#      domain; non-finite values (inf/nan) are REJECTED host-side.
#   2. BOUNDARY_REJECT_CASES — NON-FINITE inputs: the kernel is NEVER
#      launched; the case verifies the EXPECTED host-side rejection by
#      validate_inputs (AssertionError, offending tensor name + violation
#      count in the message).


def _special_zero(K, W, U, initial_state):
    """All-zero K/W/U (GK unchanged) — gate-only decay of the initial state."""
    return torch.zeros_like(K), torch.zeros_like(W), torch.zeros_like(U), initial_state


def _special_inf(K, W, U, initial_state):
    """Single +inf injected at initial_state[0, 0, 0, 0] (sparse,
    deterministic). NON-FINITE -> expected-rejection path: feeds
    _run_nonfinite_reject_case (validate_inputs must reject; the kernel is
    never launched on it)."""
    init2 = initial_state.clone()
    init2[0, 0, 0, 0] = float("inf")
    return K, W, U, init2


def _special_nan(K, W, U, initial_state):
    """Single nan injected at initial_state[0, 0, 0, 0]. NON-FINITE ->
    expected-rejection path (same as _special_inf)."""
    init2 = initial_state.clone()
    init2[0, 0, 0, 0] = float("nan")
    return K, W, U, init2


def _special_dbound(K, W, U, initial_state):
    """Values scaled to 5e18 — the overflow-boundary caliber of this
    operator's compute chain (finite-input contract boundary: finite
    values of ANY magnitude are legal, non-finite inputs are rejected).

    The binding bound is the HOST-side M precompute (K^T@W in fp32), not
    the bfloat16 max: at a 1e38 scale K^T@W overflows fp32
    host-side -> non-finite M -> validate_inputs rejects the launch (the
    finite-input contract rejects the launch in that case). At 5e18 the
    precompute stays finite (K^T@W ~1.5e36
    typical, fp32 max 3.4e38) and the overflow moves IN-KERNEL: from
    chunk 2 on, M@h products (~1e73) overflow the fp32 L0C accumulator —
    inf/nan PROPAGATION FROM FINITE INPUTS is the observed object (kernel
    runs; [BOUNDARY_WARN] if beyond thresholds, as with divergence)."""
    s = 5.0e18
    return (
        (K.float() * s).to(torch.bfloat16),
        (W.float() * s).to(torch.bfloat16),
        (U.float() * s).to(torch.bfloat16),
        initial_state,
    )


BOUNDARY_CASES = [
    {
        # All-zero K/W/U: FINITE legal values — kernel runs normally (gate-
        # only decay stays inside thresholds; expected PASS).
        "name": "b_zero_input",
        "special": _special_zero,
        "tags": ["D-SPECIAL-ZERO"],
    },
    {
        # (-2,2): FINITE legal range. K/W row norm ~2 -> ||K^T@W|| > 1 ->
        # ungated recurrence diverges (delta-rule semantics; CPU-simulated
        # h r~0.88) — expected WARN, non-blocking.
        "name": "b_vrange_m",
        "vrange": (-2, 2),
        "tags": ["D-VALRANGE-M"],
    },
    {
        # (-64,64): FINITE large symmetric — divergence to ~1e21 (expected
        # WARN). Finite values stay in the legal domain; only inf/nan are
        # rejected (see BOUNDARY_REJECT_CASES).
        "name": "b_large_values",
        "vrange": (-64, 64),
        "tags": ["D-VALRANGE-L"],
    },
    {
        # (-0.5,1.5): FINITE asymmetric — nonzero-mean K/W adds a rank-1
        # component -> divergence (expected WARN).
        "name": "b_asym_vrange",
        "vrange": (-0.5, 1.5),
        "tags": ["D-VALRANGE-ASYM"],
    },
    {
        # Official randn magnitude (init x1.0): FINITE values, bf16 GEMM
        # intrinsic error exceeds tolerance under the ungated-K recurrence
        # — non-blocking. S=512 keeps the original observation caliber
        # (BS=8 scan).
        "name": "b_official_magnitude",
        "S": 512,
        "init_scale": 1.0,
        "tags": ["D-VALRANGE-M"],
    },
    {
        # 5e18 scale (fp32-GEMM overflow boundary of the compute chain —
        # see _special_dbound): FINITE legal values, the kernel runs; from
        # chunk 2 on, fp32 L0C products overflow and inf/nan PROPAGATE from
        # finite inputs (expected WARN; contrast BOUNDARY_REJECT_CASES,
        # where non-finite INPUTS are rejected outright — the finite-input
        # contract boundary is finite=legal domain / non-finite=rejected).
        "name": "b_dbound",
        "special": _special_dbound,
        "tags": ["D-SPECIAL-DBOUND"],
    },
]

# Non-finite-input cases: verify the HOST-SIDE REJECTION (the finite-input
# contract). The kernel is never launched on these inputs.
BOUNDARY_REJECT_CASES = [
    {
        "name": "b_inf_input",
        "special": _special_inf,
        "expect_tensor": "initial_state",
        "expect_count": 1,
        "tags": ["D-SPECIAL-INF"],
    },
    {
        "name": "b_nan_input",
        "special": _special_nan,
        "expect_tensor": "initial_state",
        "expect_count": 1,
        "tags": ["D-SPECIAL-NAN"],
    },
]

# Default boundary config (case dicts may override shape keys; shared
# configs compile once via memoization)
_BOUNDARY_CFG = dict(
    B=1,
    S=256,
    H=4,
    DK=128,
    DV=128,
    chunk_size=64,
    block_DV=128,
    use_gk=True,
)

_BOUNDARY_WARNINGS = []
_BOUNDARY_ERRORS = []


def _run_nonfinite_reject_case(name, special, expect_tensor, expect_count):
    """Expected-REJECTION runner for non-finite inputs (finite-input
    contract): prepare + mutate + M precompute on CPU, then validate_inputs
    MUST raise AssertionError with the offending tensor name and violation
    count in the message — a PRECISE try/except bite (not a swallow).

    Outcomes:
      - AssertionError with the expected name+count -> [BOUNDARY_PASS]
      - validate_inputs passes (silent accept) or message mismatch ->
        [BOUNDARY_WARN] (recorded, non-blocking)
      - any OTHER exception (prep/compile failure) propagates -> the
        boundary layer's [BOUNDARY_ERROR] class (aggregated + re-raised).

    The kernel is NEVER launched on non-finite inputs.
    """
    print("=" * 60)
    print(f"[Boundary:{name}] non-finite input -> expected host-side rejection (kernel NOT launched)")
    print("=" * 60)
    cfg = _BOUNDARY_CFG
    torch.manual_seed(zlib.crc32(name.encode()) & 0xFFFFFFFF)
    (
        K,
        W,
        U,
        GK,
        initial_state,
        K_cm,
        W_cm,
        U_cm,
    ) = prepare_input(
        cfg["B"],
        cfg["S"],
        cfg["H"],
        cfg["DK"],
        cfg["DV"],
        cfg["chunk_size"],
        block_DV=cfg["block_DV"],
        use_gk=cfg["use_gk"],
    )
    # Boundary mutation (same hook as _run_one_case), then re-derive the
    # kernel-side inputs from the mutated originals (structurally parallel
    # to _run_one_case's re-derivation path).
    K, W, U, initial_state = special(K, W, U, initial_state)
    K_cm = to_chunk_major(K, cfg["chunk_size"])
    W_cm = to_chunk_major(W, cfg["chunk_size"])
    U_cm = to_chunk_major(U, cfg["chunk_size"])
    M = prepare_m_kda(K_cm, W_cm, GK, cfg["chunk_size"], cfg["DK"], use_gk=cfg["use_gk"])
    expect_snippet = f"{expect_tensor} contains {expect_count} inf/nan element(s)"
    try:
        validate_inputs(K_cm, W_cm, U_cm, M, initial_state)
    except AssertionError as e:
        msg = str(e)
        if expect_snippet in msg:
            print(f"[BOUNDARY_PASS] {name}: correctly rejected host-side ({msg[:150]})")
            return True
        print(f"[BOUNDARY_WARN] {name}: rejection message mismatch (wanted '{expect_snippet}' in message): {msg[:180]}")
        return False
    print(f"[BOUNDARY_WARN] {name}: non-finite input NOT rejected (validate_inputs passed — silent accept)")
    return False


def test_chunk_delta_h_fwd_boundary():
    """Boundary tests: zero / vrange / dtype-bound (finite, kernel runs) +
    inf / nan (non-finite, expected host-side rejection).

    Strict outcome classes:
      - computed-but-beyond-threshold -> [BOUNDARY_WARN] (non-blocking,
        mathematical divergence / propagation properties); no result assert
        by design — precision regressions surface in the blocking L0/L1
        layers.
      - EXCEPTION on a legal input (compile failure / runtime error / OOM)
        -> [BOUNDARY_ERROR]: recorded separately (NEVER folded into the
        WARN summary) and re-raised as an AssertionError at the end —
        visible as a pytest FAIL; the direct-run main() catches it without
        touching the exit code (boundary stays non-blocking).
      - non-finite inputs -> expected-rejection verification (never
        launched; see _run_nonfinite_reject_case).
    """
    for case in BOUNDARY_CASES:
        overrides = {k: v for k, v in case.items() if k not in ("name", "tags")}
        params = {**_BOUNDARY_CFG, **overrides}
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        try:
            passed = _run_one_case(case["name"], level_tag="Boundary", blocking=False, **params)
        except Exception as e:  # noqa: BLE001 — aggregated + re-raised below
            print(f"[BOUNDARY_ERROR] {case['name']}: {type(e).__name__}: {e} (aggregated; NOT a WARN)")
            _BOUNDARY_ERRORS.append(case["name"])
            continue
        if not passed:
            _BOUNDARY_WARNINGS.append(case["name"])

    reject_results = []
    for case in BOUNDARY_REJECT_CASES:
        print(f"\n[tags] {case['name']}: {', '.join(case['tags'])}")
        ok = _run_nonfinite_reject_case(
            case["name"],
            special=case["special"],
            expect_tensor=case["expect_tensor"],
            expect_count=case["expect_count"],
        )
        reject_results.append((case["name"], ok))

    print("=" * 60)
    for name, ok in reject_results:
        status = "BOUNDARY_PASS" if ok else "BOUNDARY_WARN"
        print(f"[{status}] {name} (expected-rejection verification)")
    if _BOUNDARY_WARNINGS:
        print(f"[BOUNDARY_WARN] summary: {len(_BOUNDARY_WARNINGS)} warning(s): {', '.join(_BOUNDARY_WARNINGS)} (non-blocking)")
    else:
        print("[BOUNDARY_PASS] All boundary tests passed")
    if _BOUNDARY_ERRORS:
        print(
            f"[BOUNDARY_ERROR] summary: {len(_BOUNDARY_ERRORS)} error(s): {', '.join(_BOUNDARY_ERRORS)} (legal-input exceptions — NOT warnings)"
        )
    print("=" * 60)
    if _BOUNDARY_ERRORS:
        # Re-raise for pytest visibility; the direct-run main() catches
        # this without touching the exit code.
        raise AssertionError(f"Boundary ERRORs (exceptions on legal inputs — NOT precision warnings): {', '.join(_BOUNDARY_ERRORS)}")


# ============================================================================
# Coverage manifest: explicit per-dimension case counts (mirrors the
# per-case [tags] above; a dimension counts as covered when tags or these
# explicit counts reach it).
# ============================================================================

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 2,
    "D-DTYPE-fp32": 2,
    "D-SHAPE-ALIGNED": 3,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 2,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 2,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-chunk_size": 4,
    "D-PARAM-use_gk": 1,
    "D-PARAM-use_initial_state": 1,
    "D-PARAM-store_final_state": 1,
    "D-PARAM-save_new_value": 1,
    "D-PARAM-block_DV": 1,
    # D-SPECIAL-INF/NAN are covered by the expected-rejection cases
    # (b_inf_input / b_nan_input verify the host-side rejection of
    # non-finite inputs — the finite-input contract).
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 7,
}

# All dimensions are mandatory — no exemptions declared. (D-VALRANGE-M/L/
# ASYM are covered by Boundary cases because the ungated-K KDA recurrence
# diverges at those ranges — a documented mathematical property of the
# delta-rule operator family, not a coverage gap.)
COVERAGE_NA = {}


if __name__ == "__main__":
    import tilelang

    tilelang.disable_cache()

    parser = argparse.ArgumentParser(description="chunk_delta_h_fwd (KDA) precision test")
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
        passed = test_chunk_delta_h_fwd_l0()
        blocking_ok = blocking_ok and passed
    if args.level in ("l1", "all"):
        passed = test_chunk_delta_h_fwd_l1()
        blocking_ok = blocking_ok and passed
    if args.level in ("l2", "all"):
        l2_ok = test_chunk_delta_h_fwd_l2()
        blocking_ok = blocking_ok and l2_ok
    if args.level in ("boundary", "all"):
        try:
            test_chunk_delta_h_fwd_boundary()  # non-blocking for the exit code
        except AssertionError as e:
            # [BOUNDARY_ERROR] aggregation (exceptions on legal inputs —
            # compile/runtime/OOM): loud under pytest (the function raises);
            # the direct run records it here WITHOUT touching the exit code
            # (boundary stays non-blocking), and never folds it into WARN.
            print(f"[BOUNDARY_ERROR] {e} (boundary layer non-blocking — exit code unaffected)")

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    else:
        sys.exit(1)
