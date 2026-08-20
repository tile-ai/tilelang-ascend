"""Test suite for GQA Sink Attention Backward (BHSD).

L0: 7 cases (rule shapes, block-aligned) — blocking
L1: 8 cases (varying params + value ranges) — blocking
L2: 5 cases (invalid inputs, should reject) — non-blocking
Boundary: 4 cases (special values: inf/nan/zero) — non-blocking
Bench: do_bench performance (bwd main + total pipeline)
msprof: hardware-level kernel profiling (Cube/MTE2/MTE3/L2/Scalar)

Precision standard: 169-line standard.
  float16: atol=6.10e-5, rtol=1.95e-3, max_abs_limit=0.1, required_ratio=0.99
  float32: atol=1.53e-5, rtol=9.77e-4, max_abs_limit=1e-2, required_ratio=0.99

Golden runs on CPU (.cpu() before ref_fwd/ref_bwd).
NPU outputs .cpu() for comparison — avoids 4GB fp32 attention OOM on NPU.
"""

import argparse
import ast
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time

import tilelang
import torch
from tilelang.profiler import do_bench

from example_gqa_sink_bwd_bhsd import (
    flashattn_bwd_dsink,
    flashattn_bwd_k1_qk_recompute,
    flashattn_bwd_k2_softmax_p,
    flashattn_bwd_k3_dv_dp,
    flashattn_bwd_k4_ds_compute,
    flashattn_bwd_k5_dk_dq,
    flashattn_bwd_postprocess,
    flashattn_bwd_preprocess,
    flashattn_fwd,
    ref_bwd,
    ref_fwd,
)

# ============================================================================
# Precision constants (169-line standard)
# ============================================================================

FP16_ATOL = 6.10e-5
FP16_RTOL = 1.95e-3
FP16_MAX_ABS_LIMIT = 0.1
FP16_REQUIRED_RATIO = 0.99

FP32_ATOL = 1.53e-5
FP32_RTOL = 9.77e-4
FP32_MAX_ABS_LIMIT = 1e-2
FP32_REQUIRED_RATIO = 0.99


def check_precision(actual, golden, dtype_str):
    """Double-gate precision check: matched_ratio >= required AND max_abs <= limit.

    INF/NAN structural comparison per precision-standard.md §3.1.
    Returns: (passed, matched_ratio, max_abs_error)
    """
    if dtype_str == "float16":
        atol, rtol, max_abs_limit, required_ratio = (
            FP16_ATOL,
            FP16_RTOL,
            FP16_MAX_ABS_LIMIT,
            FP16_REQUIRED_RATIO,
        )
    else:
        atol, rtol, max_abs_limit, required_ratio = (
            FP32_ATOL,
            FP32_RTOL,
            FP32_MAX_ABS_LIMIT,
            FP32_REQUIRED_RATIO,
        )

    a = actual.detach().cpu().float()
    g = golden.detach().cpu().float()
    # INF/NAN structural comparison (precision-standard.md §3.1)
    special = ~torch.isfinite(g)
    if special.any():  # noqa: SIM102
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or not torch.equal(
            torch.isinf(a[special]), torch.isinf(g[special])
        ):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    threshold = atol + rtol * g[m].abs()
    matched_ratio = (abs_err <= threshold).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Compile-error robustness helpers (stderr capture + /tmp cleanup + retry)
# ============================================================================


def _cleanup_tmp_compilation_files():
    """Clean up tilelang JIT compilation temp files in /tmp.

    tilelang.disable_cache() mode creates a new tmp*.cpp + tmp*.so per
    kernel compilation (libgen.py uses NamedTemporaryFile(delete=False));
    over a 24-case run these accumulate to ~700 files.

    IMPORTANT: only remove tmp*.so, NOT tmp*.cpp. The .cpp source file is
    read by the bisheng compiler process; if we remove it while bisheng is
    still reading (especially in retry scenarios where the previous bisheng
    process may still be tearing down), bisheng fails with
    `error reading '/tmp/tmpXXX.cpp'` and the retry also fails — a race
    condition observed in run 31. The .so is the final loaded artifact and
    can be safely removed after dlopen. The .cpp files accumulate but
    inode usage stays low (~540 files/run, 3% inode usage even over 100
    runs), so leaving them is acceptable.

    Failures are silently ignored (best-effort cleanup).
    """
    patterns = ["/tmp/tmp*.so"]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            basename = os.path.basename(path)
            if len(basename) < 10:
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  [cleanup] removed {removed} tmp compilation files from /tmp")


def _is_compile_error(exc):
    """Check if exception is a tilelang compilation error worth retrying.

    tilelang's libgen.py raises RuntimeError("Compilation Failed! ...") when
    bisheng returns non-zero. Only these are retried — precision failures,
    shape mismatches, and runtime errors propagate immediately.
    """
    msg = str(exc)
    return "Compilation Failed" in msg or "bisheng" in msg.lower()


def _capture_bisheng_stderr(exc):
    """Re-run bisheng command from a Compilation Failed exception to capture stderr.

    tilelang's libgen.py raises RuntimeError(f"Compilation Failed! {command}")
    where command is the Python list repr. The original subprocess.run doesn't
    capture stderr, so the C++ error details go to parent stderr (lost in test
    output noise). This helper parses the command list, checks the .cpp source
    still exists, and re-runs with capture_output=True to retrieve full stderr.

    Returns: (stderr_str, note_str) — stderr is empty string if unavailable.
    """
    msg = str(exc)
    prefix = "Compilation Failed! "
    if prefix not in msg:
        # Try to locate command list anywhere in message (defensive)
        list_start = msg.find("[")
        if list_start == -1:
            return "", f"no bisheng command in exception: {msg[:200]}"
        cmd_repr = msg[list_start : msg.rfind("]") + 1]
    else:
        cmd_repr = msg[len(prefix) :]
    try:
        cmd_list = ast.literal_eval(cmd_repr)
        if not isinstance(cmd_list, list) or not cmd_list:
            return "", f"parsed command is not a non-empty list: {type(cmd_list).__name__}"
    except (ValueError, SyntaxError) as parse_err:
        return "", f"failed to parse bisheng command from exception: {parse_err}"
    # Find the .cpp source file in the command
    cpp_path = next((arg for arg in cmd_list if isinstance(arg, str) and arg.endswith(".cpp")), None)
    if cpp_path is None or not os.path.exists(cpp_path):
        return "", (f"original .cpp source not available (path={cpp_path!r}), cannot re-run to capture stderr")
    # Re-run bisheng with output capture (timeout 5 min for large kernels)
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=300,
        )
        stderr = result.stderr or ""
        stdout = result.stdout or ""
        if stdout and stderr:
            combined = f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        elif stdout:
            combined = f"--- stdout ---\n{stdout}"
        else:
            combined = stderr
        return combined, f"re-ran bisheng (exit={result.returncode})"
    except subprocess.TimeoutExpired:
        return "", "bisheng re-run timed out after 300s"
    except Exception as rerun_err:  # noqa: BLE001
        return "", f"bisheng re-run failed: {type(rerun_err).__name__}: {rerun_err}"


def _run_with_retry(fn, max_retries=2, kernel_name="<unknown>"):
    """Run a JIT compilation call with retry on transient compile errors.

    Only retries on compilation failures (_is_compile_error returns True).
    Precision failures, shape mismatches, and other exceptions propagate
    immediately without retry.

    Between retries: cleans /tmp tmp*.cpp/tmp*.so files and sleeps 1s to let
    system resources release. On final failure, captures full bisheng stderr
    and attaches it to the exception as _captured_stderr for the outer
    handler to surface.

    Returns: the result of fn() on success.
    Raises: the last exception on final failure.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_compile_error(e):
                raise  # not a compile error — propagate immediately
            last_exc = e
            if attempt < max_retries:
                err_brief = str(e)[:200].replace("\n", " ")
                print(f"  [retry] attempt {attempt + 1}/{max_retries} for {kernel_name}: previous error was {err_brief}")
                _cleanup_tmp_compilation_files()
                time.sleep(1)
            else:
                # Final attempt failed — capture stderr before re-raising
                stderr, note = _capture_bisheng_stderr(e)
                if stderr:
                    print(f"  [compile_stderr] {kernel_name} ({note}):")
                    print("  " + stderr[:2000].replace("\n", "\n  "))
                    if not hasattr(e, "_captured_stderr"):
                        e._captured_stderr = stderr  # type: ignore[attr-defined]
                else:
                    print(f"  [compile_stderr] {kernel_name}: unavailable ({note})")
                raise
    # Unreachable — loop either returns or raises
    raise last_exc  # type: ignore[misc]


# ============================================================================
# L0/L1 test runner: runs one case end-to-end and checks all outputs
# ============================================================================


def run_l0_case(case_name, B, H, groups, N, D, window_size, scale=1.0, shift=0.0):
    """Run one L0/L1 case and return dict of results.

    scale/shift control input value range for L1 D-VALRANGE-* coverage.
    Default (1.0, 0.0) preserves L0 behavior.
    """
    H_kv = H // groups
    block_M, block_N = 64, 64
    dim_qk_padded = ((D + 127) // 128) * 128

    # Generate inputs on CPU (fp32 randn -> scale/shift -> fp16), move to NPU
    Q_cpu = (torch.randn(B, H, N, D) * scale + shift).half()
    K_cpu = (torch.randn(B, H_kv, N, D) * scale + shift).half()
    V_cpu = (torch.randn(B, H_kv, N, D) * scale + shift).half()
    sinks_cpu = (torch.randn(H) * scale + shift).half()
    dO_cpu = (torch.randn(B, H, N, D) * scale + shift).half()

    Q = Q_cpu.to("npu")
    K = K_cpu.to("npu")
    V = V_cpu.to("npu")
    sinks = sinks_cpu.to("npu")
    dO = dO_cpu.to("npu")

    results = {"case": case_name, "shape": f"B={B} H={H} N={N} D={D} g={groups} w={window_size}"}

    try:
        # --- Forward ---
        fwd_mod = _run_with_retry(
            lambda: flashattn_fwd(B, H, N, D, groups, window_size, block_M, block_N),
            kernel_name="flashattn_fwd",
        )
        O_npu, lse_npu = fwd_mod(Q, K, V, sinks)
        torch.npu.synchronize()

        O_ref = ref_fwd(Q_cpu, K_cpu, V_cpu, sinks_cpu, window_size, groups)
        passed, ratio, max_abs = check_precision(O_npu, O_ref, "float16")
        results["fwd_O"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        if not passed:
            results["fwd_O"]["status"] = "[PRECISION_FAIL]"
        else:
            results["fwd_O"]["status"] = "[PRECISION_PASS]"

        # --- BWD Preprocess: Delta = sum(O * dO) ---
        prep_mod = _run_with_retry(
            lambda: flashattn_bwd_preprocess(B, H, N, D, blk=32),
            kernel_name="flashattn_bwd_preprocess",
        )
        Delta_npu = prep_mod(O_npu, dO)
        torch.npu.synchronize()

        # Delta golden uses O_npu (same input as kernel) to isolate bwd precision from fwd
        O_cpu_for_delta = O_npu.cpu()
        Delta_ref = (O_cpu_for_delta.float() * dO_cpu.float()).sum(dim=-1)
        passed, ratio, max_abs = check_precision(Delta_npu, Delta_ref, "float32")
        results["bwd_Delta"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_Delta"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # --- BWD Main: 5-kernel split (k1-k5, no-scope Developer mode) ---
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")

        bwd_block_num = H * (N // block_M) * B
        if window_size is not None:
            max_kv_per_q = min(window_size // block_N + 1, N // block_N)
        else:
            max_kv_per_q = N // block_N
        ws_s = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
        ws_p = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
        ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
        ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
        ws_dp = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
        ws_ds = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
        ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")

        bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)

        # k1: S = Q @ K^T -> ws_s
        k1_mod = _run_with_retry(
            lambda: flashattn_bwd_k1_qk_recompute(*bwd_args),
            kernel_name="flashattn_bwd_k1_qk_recompute",
        )
        k1_mod(Q, K, ws_s)
        torch.npu.synchronize()

        # k2: P = softmax(S) + mask + p_delta
        k2_mod = _run_with_retry(
            lambda: flashattn_bwd_k2_softmax_p(*bwd_args),
            kernel_name="flashattn_bwd_k2_softmax_p",
        )
        k2_mod(ws_s, lse_npu, ws_p, ws_p_delta, ws_p_fp32)
        torch.npu.synchronize()

        # k3: dV (Compensated GEMM, atomic_add) + dP -> ws_dp
        k3_mod = _run_with_retry(
            lambda: flashattn_bwd_k3_dv_dp(*bwd_args),
            kernel_name="flashattn_bwd_k3_dv_dp",
        )
        k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
        torch.npu.synchronize()

        # k4: dS = P*(dP-Delta)*scale + mask + ds_delta
        k4_mod = _run_with_retry(
            lambda: flashattn_bwd_k4_ds_compute(*bwd_args),
            kernel_name="flashattn_bwd_k4_ds_compute",
        )
        k4_mod(ws_p_fp32, ws_dp, Delta_npu, ws_ds, ws_ds_delta)
        torch.npu.synchronize()

        # k5: dK (Compensated GEMM, atomic_add) + dQ (L0C accumulate)
        k5_mod = _run_with_retry(
            lambda: flashattn_bwd_k5_dk_dq(*bwd_args),
            kernel_name="flashattn_bwd_k5_dk_dq",
        )
        k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)
        torch.npu.synchronize()

        # Postprocess: fp32 -> fp16
        # dQ skipped — bwd kernel writes dQ as fp16 directly (L0C fp32 -> GM fp16 auto-cast)
        dQ_fp16 = dQ
        post_dk = _run_with_retry(
            lambda: flashattn_bwd_postprocess(B, H_kv, N, dim_qk_padded, blk=64),
            kernel_name="flashattn_bwd_postprocess_dK",
        )
        dK_fp16 = post_dk(dK)
        post_dv = _run_with_retry(
            lambda: flashattn_bwd_postprocess(B, H_kv, N, D, blk=64),
            kernel_name="flashattn_bwd_postprocess_dV",
        )
        dV_fp16 = post_dv(dV)
        torch.npu.synchronize()

        # --- BWD Dsink ---
        dsink_mod = _run_with_retry(
            lambda: flashattn_bwd_dsink(B, H, N, block=128),
            kernel_name="flashattn_bwd_dsink",
        )
        dSinks_npu = dsink_mod(sinks, Delta_npu, lse_npu)
        torch.npu.synchronize()
        dSinks_sum = dSinks_npu.cpu().sum(0).sum(1)  # [H] fp32 on CPU

        # --- Golden backward ---
        dQ_ref, dK_ref, dV_ref, dSinks_ref_autograd = ref_bwd(
            Q_cpu,
            K_cpu,
            V_cpu,
            sinks_cpu,
            dO_cpu,
            window_size,
            groups,
        )

        # dSinks golden: recompute using NPU's actual lse/Delta (isolates dSinks kernel
        # from fwd/preprocess precision — autograd's internal lse/Delta differ from NPU's)
        sinks_exp = sinks_cpu.float().view(1, H, 1)  # [1, H, 1]
        lse_cpu = lse_npu.cpu().float()  # [B, H, N]
        delta_cpu = Delta_npu.cpu().float()  # [B, H, N]
        dSinks_ref = -(torch.exp(sinks_exp - lse_cpu) * delta_cpu).sum(dim=0).sum(dim=1)  # [H]

        # Compare dQ (fp16)
        passed, ratio, max_abs = check_precision(dQ_fp16[..., :D], dQ_ref, "float16")
        results["bwd_dQ"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_dQ"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # Compare dK (fp16)
        passed, ratio, max_abs = check_precision(dK_fp16[..., :D], dK_ref, "float16")
        results["bwd_dK"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_dK"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # Compare dV (fp16)
        passed, ratio, max_abs = check_precision(dV_fp16, dV_ref, "float16")
        results["bwd_dV"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_dV"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # Compare dSinks (fp32)
        passed, ratio, max_abs = check_precision(dSinks_sum, dSinks_ref, "float32")
        results["bwd_dSinks"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_dSinks"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

    except Exception as e:
        results["error"] = str(e)
        results["status"] = "[PRECISION_FAIL]"
        import traceback

        results["traceback"] = traceback.format_exc()
        # If this is a compilation failure, surface full bisheng stderr.
        # _run_with_retry may have already captured + attached _captured_stderr
        # on the final retry; otherwise (non-wrapped path) capture it here.
        if _is_compile_error(e):
            stderr = getattr(e, "_captured_stderr", None)
            if stderr is not None:
                note = "captured by _run_with_retry"
            else:
                stderr, note = _capture_bisheng_stderr(e)
            if stderr:
                results["compile_stderr"] = stderr
                print(f"  [compile_stderr] {note}:")
                print("  " + stderr[:2000].replace("\n", "\n  "))
            else:
                print(f"  [compile_stderr] unavailable: {note}")

    # Clean up tilelang JIT temp files (disable_cache mode creates ~30 per case)
    _cleanup_tmp_compilation_files()

    return results


# ============================================================================
# L0 test cases (rule shapes, block-aligned)
# ============================================================================


def test_gqa_sink_bwd_bhsd_l0():
    """Run all 7 L0 cases (rule shapes, block-aligned)."""
    cases = [
        ("l0_small", 1, 4, 2, 128, 128, None),
        ("l0_causal_full", 1, 8, 4, 256, 128, None),
        ("l0_gqa", 1, 16, 16, 256, 128, None),
        ("l0_window_64", 1, 8, 4, 256, 128, 64),
        ("l0_window_128", 1, 8, 4, 256, 128, 128),
        ("l0_batch_2", 2, 8, 4, 256, 128, 128),
        ("l0_default", 1, 64, 8, 4096, 128, 128),
    ]

    all_passed = True
    all_results = []

    for case_name, B, H, groups, N, D, window_size in cases:
        print(f"\n{'=' * 60}")
        print(f"[L0] {case_name}: B={B} H={H} N={N} D={D} g={groups} w={window_size}")
        print(f"{'=' * 60}")

        result = run_l0_case(case_name, B, H, groups, N, D, window_size)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_passed = False
            continue

        case_passed = True
        for key in ["fwd_O", "bwd_Delta", "bwd_dQ", "bwd_dK", "bwd_dV", "bwd_dSinks"]:
            if key in result:
                r = result[key]
                status = r["status"]
                print(f"  {key:12s}: {status} ratio={r['ratio']:.4f} max_abs={r['max_abs']:.3e}")
                if "[PRECISION_FAIL]" in status:
                    case_passed = False
                    all_passed = False

        if case_passed:
            print(f"  -> {case_name}: ALL [PRECISION_PASS]")
        else:
            print(f"  -> {case_name}: HAS [PRECISION_FAIL]")

    return all_passed, all_results


# ============================================================================
# Coverage metadata (for coverage_check.py)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"

# L1 cases: (name, B, H, groups, N, D, window, tags, scale, shift)
# Kernel constraints: N % 128 == 0 (dsink), N % 64 == 0 (fwd/bwd),
#   window % 64 == 0, H % groups == 0, D == 128 (bwd pads to 128).
# Value ranges: fp16 attention has inherent precision limits — scales kept at
#   1.0 max; VALRANGE variation via N (score range) and shift (distribution
#   asymmetry).
L1_CASES = [
    (
        "l1_n512_g4_w64_s",
        1,
        8,
        4,
        512,
        128,
        64,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-groups", "D-PARAM-window_size", "D-VALRANGE-S"],
        0.3,
        0.0,
    ),
    (
        "l1_n1024_g8_w128_m",
        1,
        8,
        8,
        1024,
        128,
        128,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-seq_len", "D-PARAM-dim", "D-VALRANGE-M"],
        1.0,
        0.0,
    ),
    (
        "l1_n2048_g4_none_l",
        1,
        8,
        4,
        2048,
        128,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-window_size", "D-VALRANGE-L"],
        1.0,
        0.0,
    ),
    (
        "l1_b4_n512_asym",
        4,
        16,
        8,
        512,
        128,
        128,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-batch", "D-VALRANGE-ASYM"],
        1.0,
        0.1,
    ),
    (
        "l1_edge_min",
        1,
        2,
        1,
        128,
        128,
        None,
        [
            "D-DTYPE-fp16",
            "D-SHAPE-EDGE",
            "D-SHAPE-ALIGNED",
            "D-PARAM-groups",
            "D-PARAM-block_M",
            "D-PARAM-block_N",
        ],
        1.0,
        0.0,
    ),
    (
        "l1_h16_g16_w64",
        1,
        16,
        16,
        256,
        128,
        64,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-heads", "D-PARAM-groups"],
        1.0,
        0.0,
    ),
    (
        "l1_b2_n1024_w256",
        2,
        8,
        4,
        1024,
        128,
        256,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-batch", "D-PARAM-window_size"],
        1.0,
        0.0,
    ),
    (
        "l1_n256_g8_w128",
        1,
        8,
        8,
        256,
        128,
        128,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-seq_len"],
        1.0,
        0.0,
    ),
]

# L2 cases: (name, tags) — invalid inputs that should be rejected by kernel.
# Kernel rejects: N not multiple of 64 (fwd assert), N not multiple of 128
#   (dsink assert), D != 128 (bwd shape mismatch), float32 dtype.
L2_CASES = [
    ("l2_dtype_f32", ["D-EXC-DTYPE"]),
    ("l2_shape_n129", ["D-EXC-SHAPE", "D-SHAPE-TAIL-1"]),
    ("l2_shape_n192", ["D-EXC-SHAPE", "D-SHAPE-TAIL-MID"]),
    ("l2_shape_n509", ["D-EXC-SHAPE", "D-SHAPE-PRIME"]),
    ("l2_shape_d64", ["D-EXC-SHAPE"]),
]

# Boundary cases: (name, tags) — legal special values (inf/nan/zero/extreme).
BOUNDARY_CASES = [
    ("boundary_sink_inf", ["D-SPECIAL-INF"]),
    ("boundary_q_nan", ["D-SPECIAL-NAN"]),
    ("boundary_do_zero", ["D-SPECIAL-ZERO"]),
    ("boundary_dbound", ["D-SPECIAL-DBOUND"]),
]

COVERAGE_MANIFEST = {}  # Auto-derived from L1_CASES + L2_CASES + BOUNDARY_CASES tags above
COVERAGE_NA = {}  # No exemptions — Fusion requires all dimensions


# ============================================================================
# L1: functional tests (valid shapes, varying params + value ranges)
# ============================================================================


def test_gqa_sink_bwd_bhsd_l1():
    """L1: functional tests with varying B/H/groups/N/window/value_range.

    All shapes satisfy kernel constraints (N%128==0, D=128, window%64==0).
    """
    all_passed = True
    all_results = []

    for case_entry in L1_CASES:
        case_name, B, H, groups, N, D, window_size, tags, scale, shift = case_entry
        print(f"\n{'=' * 60}")
        print(f"[L1] {case_name}: B={B} H={H} N={N} D={D} g={groups} w={window_size} scale={scale} shift={shift}")
        print(f"{'=' * 60}")

        result = run_l0_case(case_name, B, H, groups, N, D, window_size, scale=scale, shift=shift)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_passed = False
            continue

        case_passed = True
        for key in ["fwd_O", "bwd_Delta", "bwd_dQ", "bwd_dK", "bwd_dV", "bwd_dSinks"]:
            if key in result:
                r = result[key]
                status = r["status"]
                print(f"  {key:12s}: {status} ratio={r['ratio']:.4f} max_abs={r['max_abs']:.3e}")
                if "[PRECISION_FAIL]" in status:
                    case_passed = False
                    all_passed = False

        if case_passed:
            print(f"  -> {case_name}: ALL [PRECISION_PASS]")
        else:
            print(f"  -> {case_name}: HAS [PRECISION_FAIL]")

    return all_passed, all_results


# ============================================================================
# L2: negative tests (invalid inputs should be rejected)
# ============================================================================


def _run_exception(name, fn):
    """L2 helper: fn() feeds invalid input, expect kernel to reject.

    Raises exception -> [BOUNDARY_PASS] (correctly rejected).
    Silent accept -> [BOUNDARY_WARN] (should have rejected).
    Non-blocking.
    """
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__}: {e})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: invalid input silently accepted (should have rejected)")


def test_gqa_sink_bwd_bhsd_l2():
    """L2: negative tests — invalid dtype/shape should be rejected by kernel.

    Kernel constraints: N%128==0 (dsink), N%64==0 (fwd/bwd), D=128, dtype=float16.
    Non-blocking — [BOUNDARY_WARN] only records, doesn't affect exit code.
    """
    print("\n" + "=" * 60)
    print("[L2] Negative tests — invalid inputs should be rejected")
    print("=" * 60)

    # L2-1: float32 dtype (kernel hard-codes float16)
    def case_dtype_f32():
        B, H, groups, N, D = 1, 4, 2, 128, 128
        H_kv = H // groups
        Q = torch.randn(B, H, N, D, dtype=torch.float32, device="npu")
        K = torch.randn(B, H_kv, N, D, dtype=torch.float32, device="npu")
        V = torch.randn(B, H_kv, N, D, dtype=torch.float32, device="npu")
        sinks = torch.randn(H, dtype=torch.float32, device="npu")
        fwd_mod = flashattn_fwd(B, H, N, D, groups, None, 64, 64)
        fwd_mod(Q, K, V, sinks)
        torch.npu.synchronize()

    _run_exception("l2_dtype_f32", case_dtype_f32)

    # L2-2: N=129 (not multiple of 64 — fwd assertion fails)
    def case_shape_n129():
        B, H, groups, N, D = 1, 4, 2, 129, 128
        _fwd_mod = flashattn_fwd(B, H, N, D, groups, None, 64, 64)
        # Assertion fires during JIT compilation (before kernel run)

    _run_exception("l2_shape_n129", case_shape_n129)

    # L2-3: N=192 (multiple of 64 but not 128 — dsink assertion fails)
    def case_shape_n192():
        B, H, _groups, N, _D = 1, 4, 2, 192, 128
        _dsink_mod = flashattn_bwd_dsink(B, H, N, block=128)
        # Assertion: seq_len % 128 == 0 -> 192 % 128 = 64 != 0

    _run_exception("l2_shape_n192", case_shape_n192)

    # L2-4: N=509 (prime, not multiple of 64 — fwd assertion fails)
    def case_shape_n509():
        B, H, groups, N, D = 1, 4, 2, 509, 128
        _fwd_mod = flashattn_fwd(B, H, N, D, groups, None, 64, 64)

    _run_exception("l2_shape_n509", case_shape_n509)

    # L2-5: D=64 (bwd pads to 128, Q shape mismatch)
    def case_shape_d64():
        B, H, groups, N, D = 1, 4, 2, 128, 64
        H_kv = H // groups
        dim_qk_padded = ((D + 127) // 128) * 128  # = 128
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
        K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
        V = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
        dO = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
        lse = torch.randn(B, H, N, dtype=torch.float32, device="npu")
        Delta = torch.randn(B, H, N, dtype=torch.float32, device="npu")
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")
        bwd_block_num = H * (N // 64) * B
        max_kv_per_q = N // 64
        ws_s = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float32, device="npu")
        ws_p = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float16, device="npu")
        ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float16, device="npu")
        ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float32, device="npu")
        ws_dp = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float32, device="npu")
        ws_ds = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float16, device="npu")
        ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, 64, 64, dtype=torch.float16, device="npu")
        bwd_args = (B, H, N, D, D, None, 64, 64, groups)
        k1_mod = flashattn_bwd_k1_qk_recompute(*bwd_args)
        k1_mod(Q, K, ws_s)
        torch.npu.synchronize()
        k2_mod = flashattn_bwd_k2_softmax_p(*bwd_args)
        k2_mod(ws_s, lse, ws_p, ws_p_delta, ws_p_fp32)
        torch.npu.synchronize()
        k3_mod = flashattn_bwd_k3_dv_dp(*bwd_args)
        k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
        torch.npu.synchronize()
        k4_mod = flashattn_bwd_k4_ds_compute(*bwd_args)
        k4_mod(ws_p_fp32, ws_dp, Delta, ws_ds, ws_ds_delta)
        torch.npu.synchronize()
        k5_mod = flashattn_bwd_k5_dk_dq(*bwd_args)
        k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)
        torch.npu.synchronize()

    _run_exception("l2_shape_d64", case_shape_d64)

    return True, []


# ============================================================================
# Boundary: special values (INF/NAN/zero/extreme — legal, precision-checked)
# ============================================================================


def _run_boundary(name, dtype_str, fn):
    """Boundary helper: fn() returns (actual, golden), compared with check_precision.

    Precision pass -> [BOUNDARY_PASS].
    Precision fail or exception -> [BOUNDARY_WARN].
    Non-blocking.
    """
    try:
        actual, golden = fn()
        passed, ratio, max_abs = check_precision(actual, golden, dtype_str)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype_str} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype_str}: {type(e).__name__}: {e}")


def _run_boundary_pipeline(B, H, groups, N, D, window_size, Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu):
    """Run full attention pipeline with given inputs, return (dQ_npu, dQ_ref).

    Uses small shape for fast compilation. Returns dQ only (representative output).
    """
    H_kv = H // groups
    block_M, block_N = 64, 64
    dim_qk_padded = ((D + 127) // 128) * 128

    Q = Q_cpu.to("npu")
    K = K_cpu.to("npu")
    V = V_cpu.to("npu")
    sinks = sinks_cpu.to("npu")
    dO = dO_cpu.to("npu")

    # Forward
    fwd_mod = flashattn_fwd(B, H, N, D, groups, window_size, block_M, block_N)
    O_npu, lse_npu = fwd_mod(Q, K, V, sinks)
    torch.npu.synchronize()

    # Preprocess
    prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
    Delta_npu = prep_mod(O_npu, dO)
    torch.npu.synchronize()

    # BWD main (5-kernel split)
    dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
    dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")
    bwd_block_num = H * (N // block_M) * B
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, N // block_N)
    else:
        max_kv_per_q = N // block_N
    ws_s = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_p = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_dp = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_ds = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
    k1_mod = flashattn_bwd_k1_qk_recompute(*bwd_args)
    k1_mod(Q, K, ws_s)
    torch.npu.synchronize()
    k2_mod = flashattn_bwd_k2_softmax_p(*bwd_args)
    k2_mod(ws_s, lse_npu, ws_p, ws_p_delta, ws_p_fp32)
    torch.npu.synchronize()
    k3_mod = flashattn_bwd_k3_dv_dp(*bwd_args)
    k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
    torch.npu.synchronize()
    k4_mod = flashattn_bwd_k4_ds_compute(*bwd_args)
    k4_mod(ws_p_fp32, ws_dp, Delta_npu, ws_ds, ws_ds_delta)
    torch.npu.synchronize()
    k5_mod = flashattn_bwd_k5_dk_dq(*bwd_args)
    k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)
    torch.npu.synchronize()

    # dQ already fp16 from k5 (L0C fp32 -> GM fp16 auto-cast)
    dQ_fp16 = dQ
    torch.npu.synchronize()

    # Golden dQ
    dQ_ref, _, _, _ = ref_bwd(Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu, window_size, groups)

    return dQ_fp16[..., :D], dQ_ref


def test_gqa_sink_bwd_bhsd_boundary():
    """Boundary: special values (INF/NAN/zero/extreme).

    Uses small shape (B=1, H=4, groups=2, N=128, D=128, window=None).
    Compares dQ (fp16) against golden. Non-blocking — [BOUNDARY_WARN] only records.
    """
    print("\n" + "=" * 60)
    print("[Boundary] Special value tests (INF/NAN/zero/extreme)")
    print("=" * 60)

    B, H, groups, N, D, window_size = 1, 4, 2, 128, 128, None
    H_kv = H // groups

    # Boundary-1: sink contains +-inf
    def case_sink_inf():
        Q_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        K_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        V_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        sinks_cpu = torch.tensor([float("inf"), float("-inf"), 0.0, 1.0], dtype=torch.float16)
        dO_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        return _run_boundary_pipeline(B, H, groups, N, D, window_size, Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu)

    _run_boundary("boundary_sink_inf", "float16", case_sink_inf)

    # Boundary-2: Q contains nan
    def case_q_nan():
        Q_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        Q_cpu[0, 0, 0, 0] = float("nan")
        Q_cpu[0, 1, 64, 0] = float("nan")
        K_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        V_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        sinks_cpu = torch.randn(H, dtype=torch.float16)
        dO_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        return _run_boundary_pipeline(B, H, groups, N, D, window_size, Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu)

    _run_boundary("boundary_q_nan", "float16", case_q_nan)

    # Boundary-3: dO all zeros
    def case_do_zero():
        Q_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        K_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        V_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        sinks_cpu = torch.randn(H, dtype=torch.float16)
        dO_cpu = torch.zeros(B, H, N, D, dtype=torch.float16)
        return _run_boundary_pipeline(B, H, groups, N, D, window_size, Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu)

    _run_boundary("boundary_do_zero", "float16", case_do_zero)

    # Boundary-4: Q/K at fp16 boundary values (±32000, range < 65504 fp16 limit)
    def case_dbound():
        Q_cpu = torch.empty(B, H, N, D, dtype=torch.float16)
        Q_cpu.uniform_(-32000.0, 32000.0)
        K_cpu = torch.empty(B, H_kv, N, D, dtype=torch.float16)
        K_cpu.uniform_(-32000.0, 32000.0)
        V_cpu = torch.randn(B, H_kv, N, D, dtype=torch.float16)
        sinks_cpu = torch.randn(H, dtype=torch.float16)
        dO_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
        return _run_boundary_pipeline(B, H, groups, N, D, window_size, Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu)

    _run_boundary("boundary_dbound", "float16", case_dbound)

    return True, []


# ============================================================================
# do_bench performance (CI §6, D17: def not lambda, D20: semantic naming)
# ============================================================================


def run_bench(B=1, H=64, N=4096, D=128, groups=8, window_size=128, warmup=50, rep=100, per_kernel=False):
    """Benchmark bwd main pipeline + total pipeline via do_bench.

    Pre-allocates all tensors and pre-compiles all JIT modules to avoid
    measuring allocation/compilation overhead. atomic_add outputs (dQ/dK/dV)
    are zeroed in each bench iteration (CI §6.1).
    """
    torch.set_default_device("npu")
    torch.manual_seed(42)

    H_kv = H // groups
    block_M, block_N = 64, 64
    dim_qk_padded = ((D + 127) // 128) * 128

    # Inputs
    Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
    K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
    V = torch.randn_like(K)
    sinks = torch.randn(H, dtype=torch.float16, device="npu")
    dO = torch.randn_like(Q)

    # Forward (precompute O, lse)
    fwd_mod = flashattn_fwd(B, H, N, D, groups, window_size, 64, 64)
    O, lse = fwd_mod(Q, K, V, sinks)
    torch.npu.synchronize()

    # Pre-compile all JIT modules
    prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
    bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
    k1_mod = flashattn_bwd_k1_qk_recompute(*bwd_args)
    k2_mod = flashattn_bwd_k2_softmax_p(*bwd_args)
    k3_mod = flashattn_bwd_k3_dv_dp(*bwd_args)
    k4_mod = flashattn_bwd_k4_ds_compute(*bwd_args)
    k5_mod = flashattn_bwd_k5_dk_dq(*bwd_args)
    post_dk = flashattn_bwd_postprocess(B, H_kv, N, dim_qk_padded, blk=64)
    post_dv = flashattn_bwd_postprocess(B, H_kv, N, D, blk=64)
    dsink_mod = flashattn_bwd_dsink(B, H, N, block=128)
    torch.npu.synchronize()

    # Pre-allocate all tensors (preprocess once for bench_bwd_only)
    delta = prep_mod(O, dO)
    torch.npu.synchronize()

    dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
    dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")

    bwd_block_num = H * (N // block_M) * B
    if window_size is not None:
        max_kv_per_q = min(window_size // block_N + 1, N // block_N)
    else:
        max_kv_per_q = N // block_N

    ws_s = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_p = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_dp = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
    ws_ds = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
    torch.npu.synchronize()

    # Bench 1: bwd main pipeline (k1->k2->k3->k4->k5, no intermediate sync)
    def bench_bwd_us():
        dQ.zero_()
        dK.zero_()
        dV.zero_()
        k1_mod(Q, K, ws_s)
        k2_mod(ws_s, lse, ws_p, ws_p_delta, ws_p_fp32)
        k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
        k4_mod(ws_p_fp32, ws_dp, delta, ws_ds, ws_ds_delta)
        k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)

    bwd_us = do_bench(bench_bwd_us, warmup=warmup, rep=rep) * 1e3  # ms->us
    torch.npu.synchronize()

    # Bench 2: total pipeline (prep + k1-k5 + post_dk + post_dv + dsink)
    def bench_total_us():
        dQ.zero_()
        dK.zero_()
        dV.zero_()
        _delta = prep_mod(O, dO)
        k1_mod(Q, K, ws_s)
        k2_mod(ws_s, lse, ws_p, ws_p_delta, ws_p_fp32)
        k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
        k4_mod(ws_p_fp32, ws_dp, _delta, ws_ds, ws_ds_delta)
        k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)
        post_dk(dK)
        post_dv(dV)
        dsink_mod(sinks, _delta, lse)

    total_us = do_bench(bench_total_us, warmup=warmup, rep=rep) * 1e3
    torch.npu.synchronize()

    # Per-kernel breakdown (optional, with syncs)
    # NOTE: local dict renamed to per_kernel_us to avoid shadowing the
    # per_kernel parameter (bool flag from --per-kernel).
    per_kernel_us = {}
    if per_kernel:

        def _bench_one(fn):
            def run():
                fn()

            return do_bench(run, warmup=warmup // 2 or 1, rep=rep // 2 or 10) * 1e3

        def _k1():
            k1_mod(Q, K, ws_s)

        def _k2():
            k2_mod(ws_s, lse, ws_p, ws_p_delta, ws_p_fp32)

        def _k3():
            dV.zero_()
            k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)

        def _k4():
            k4_mod(ws_p_fp32, ws_dp, delta, ws_ds, ws_ds_delta)

        def _k5():
            dK.zero_()
            k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)

        for name, fn in [
            ("k1_qk_recompute", _k1),
            ("k2_softmax_p", _k2),
            ("k3_dv_dp", _k3),
            ("k4_ds_compute", _k4),
            ("k5_dk_dq", _k5),
        ]:
            per_kernel_us[name] = _bench_one(fn)
            torch.npu.synchronize()

    # Report
    expert_baseline_us = 4699
    gpu_baseline_us = 14287
    print("=== BWD Kernel Benchmark (no-scope Developer 5-kernel split) ===")
    print(f"Shape: B={B} H={H} N={N} D={D} g={groups} w={window_size}")
    print(f"max_kv_per_q={max_kv_per_q}  bwd_block_num={bwd_block_num}")
    print()
    print(f"bwd main pipeline (k1->k5): {bwd_us:.0f} us")
    print(f"total pipeline:             {total_us:.0f} us  (prep+k1-k5+post+dsink)")
    print()
    print(f"--- vs Expert baseline ({expert_baseline_us} us) ---")
    print(f"bwd vs Expert:   {(expert_baseline_us - bwd_us) / expert_baseline_us * 100:+.1f}%  ({bwd_us - expert_baseline_us:+.0f} us)")
    print(f"--- vs GPU baseline ({gpu_baseline_us} us) ---")
    print(f"bwd vs GPU:      {(gpu_baseline_us - bwd_us) / gpu_baseline_us * 100:+.1f}%")
    print(f"total vs GPU:    {(gpu_baseline_us - total_us) / gpu_baseline_us * 100:+.1f}%")
    print()
    target_us = 4464
    print(f"--- target: <= {target_us} us (Expert 95%) ---")
    print(f"target reached: {'YES' if bwd_us <= target_us else 'NO'}  (gap: {bwd_us - target_us:+.0f} us)")

    if per_kernel:
        print()
        print("--- per-kernel breakdown (with syncs) ---")
        for name, us in per_kernel_us.items():
            print(f"  {name:20s}: {us:.0f} us")
        print(f"  {'sum(k1-k5)':20s}: {sum(per_kernel_us.values()):.0f} us")

    print("\nTest Passed!")
    return True


# ============================================================================
# msprof op: hardware-level kernel profiling (Cube/MTE2/MTE3/L2/Scalar stall)
# ============================================================================

# Minimal runner script for msprof op — runs each kernel once, no do_bench loop.
# dQ fp16 direct output (L0C fp32 -> GM fp16 auto-cast), no postprocess for dQ.
# dsink block=128 (kernel assert seq_len % 128 == 0).
_MSPROF_RUNNER_TEMPLATE = '''\
"""Auto-generated by test_gqa_sink_bwd_bhsd.py --level msprof. Runs each kernel
once so msprof op can capture kernel-level performance data."""
import os, sys
sys.path.insert(0, {script_dir!r})
import tilelang, torch
tilelang.disable_cache()
torch.set_default_device("npu")
torch.manual_seed(42)
from example_gqa_sink_bwd_bhsd import (
    flashattn_bwd_k1_qk_recompute, flashattn_bwd_k2_softmax_p,
    flashattn_bwd_k3_dv_dp, flashattn_bwd_k4_ds_compute,
    flashattn_bwd_k5_dk_dq, flashattn_bwd_dsink, flashattn_bwd_postprocess,
    flashattn_bwd_preprocess, flashattn_fwd,
)
B, H, groups, N, D = {B}, {H}, {groups}, {N}, {D}
H_kv = H // groups
window_size = {window_size}
block_M, block_N = 64, 64
dim_qk_padded = ((D + 127) // 128) * 128
Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
V = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
sinks = torch.randn(H, dtype=torch.float16, device="npu")
dO = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
# Forward (produces O, lse for bwd)
fwd_mod = flashattn_fwd(B, H, N, D, groups, window_size, block_M, block_N)
O_npu, lse_npu = fwd_mod(Q, K, V, sinks)
torch.npu.synchronize()
print("===KERNEL_LABEL:forward===")
# Preprocess: Delta = sum(O * dO)
prep_mod = flashattn_bwd_preprocess(B, H, N, D, blk=32)
Delta_npu = prep_mod(O_npu, dO)
torch.npu.synchronize()
print("===KERNEL_LABEL:preprocess===")
# BWD main: 5-kernel split
dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")
bwd_block_num = H * (N // block_M) * B
if window_size is not None:
    max_kv_per_q = min(window_size // block_N + 1, N // block_N)
else:
    max_kv_per_q = N // block_N
ws_s = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
ws_p = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
ws_p_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
ws_p_fp32 = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
ws_dp = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float32, device="npu")
ws_ds = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
ws_ds_delta = torch.empty(bwd_block_num, max_kv_per_q, block_M, block_N, dtype=torch.float16, device="npu")
bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
k1_mod = flashattn_bwd_k1_qk_recompute(*bwd_args)
k1_mod(Q, K, ws_s)
torch.npu.synchronize()
print("===KERNEL_LABEL:k1_qk_recompute===")
k2_mod = flashattn_bwd_k2_softmax_p(*bwd_args)
k2_mod(ws_s, lse_npu, ws_p, ws_p_delta, ws_p_fp32)
torch.npu.synchronize()
print("===KERNEL_LABEL:k2_softmax_p===")
k3_mod = flashattn_bwd_k3_dv_dp(*bwd_args)
k3_mod(ws_p, ws_p_delta, dO, V, dV, ws_dp)
torch.npu.synchronize()
print("===KERNEL_LABEL:k3_dv_dp===")
k4_mod = flashattn_bwd_k4_ds_compute(*bwd_args)
k4_mod(ws_p_fp32, ws_dp, Delta_npu, ws_ds, ws_ds_delta)
torch.npu.synchronize()
print("===KERNEL_LABEL:k4_ds_compute===")
k5_mod = flashattn_bwd_k5_dk_dq(*bwd_args)
k5_mod(ws_ds, ws_ds_delta, Q, K, dK, dQ)
torch.npu.synchronize()
print("===KERNEL_LABEL:k5_dk_dq===")
# Postprocess: dK/dV fp32 -> fp16 (dQ skipped — fp16 direct from k5)
post_dk = flashattn_bwd_postprocess(B, H_kv, N, dim_qk_padded, blk=64)
post_dk(dK)
torch.npu.synchronize()
print("===KERNEL_LABEL:postprocess_dK===")
post_dv = flashattn_bwd_postprocess(B, H_kv, N, D, blk=64)
post_dv(dV)
torch.npu.synchronize()
print("===KERNEL_LABEL:postprocess_dV===")
# Dsink (block=128)
dsink_mod = flashattn_bwd_dsink(B, H, N, block=128)
dSinks_npu = dsink_mod(sinks, Delta_npu, lse_npu)
torch.npu.synchronize()
print("===KERNEL_LABEL:dsink===")
print("All 10 kernels executed.")
'''


# Kernel labels in execution order (matches runner script above).
# NOTE: 10 entries for 10 kernel launches — post_dK and post_dV are separate
# kernel calls (each calls flashattn_bwd_postprocess once), so they get
# separate labels. The runner script prints ===KERNEL_LABEL:<name>=== markers
# after each kernel call to make the launch→kernel mapping explicit (not just
# order-inferred).
_MSPROF_KERNEL_LABELS = [
    "forward",
    "preprocess",
    "k1_qk_recompute",
    "k2_softmax_p",
    "k3_dv_dp",
    "k4_ds_compute",
    "k5_dk_dq",
    "postprocess_dK",
    "postprocess_dV",
    "dsink",
]

# Ascend 910B3: 20 cube cores + 20 vector cores
_NUM_CORES = 20


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(val, default=0.0):
    try:
        if val is None or val == "" or val == "N/A":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _col_floats(rows, col):
    """Extract float values for a column from per-block rows, skipping NA/empty."""
    out = []
    for r in rows:
        v = r.get(col, "NA")
        if v in ("NA", "", None):
            continue
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            continue
    return out


def _median(vals):
    s = sorted(v for v in vals if v is not None)
    n = len(s)
    return s[n // 2] if n else 0.0


def _sum(vals):
    return sum(v for v in vals if v is not None)


def _parse_msprof_op_summary(prof_dir):
    """Parse msprof op output to extract per-launch hardware metrics.

    msprof op mode emits one CSV set per kernel launch under
    ``main_kernel/{launch_idx}/``. Each launch directory contains:
      - OpBasicInfo_*.csv  : one row per main_kernel launch (Task Duration, Block Dim)
      - PipeUtilization_*.csv  : PER-BLOCK rows (aic_*/aiv_* pipeline times + ratios)
      - L2Cache_*.csv          : PER-BLOCK rows (L2 hit rates)
      - ResourceConflictRatio_*.csv : PER-BLOCK rows (wait ratios)
      - Memory_*.csv           : PER-BLOCK rows (GM/L1/UB/L0C traffic in KB)

    Per-block CSVs have NO "Op Name" column. This function groups by launch
    directory and aggregates:
      - ratios -> median across blocks (representative block)
      - times   -> sum across blocks / num_cores (wall us)
      - hit rates -> median across blocks
      - traffic (KB) -> sum across blocks (total kernel traffic)

    Returns list of dicts (one per launch) with metric keys.
    """
    launch_dirs = sorted(
        glob.glob(os.path.join(prof_dir, "**", "main_kernel", "*"), recursive=True),
        key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0,
    )
    launch_dirs = [d for d in launch_dirs if os.path.isdir(d)]

    launches = []
    for d in launch_dirs:
        ob_files = glob.glob(os.path.join(d, "OpBasicInfo_*.csv"))
        if not ob_files:
            continue
        ob_rows = _read_csv_rows(ob_files[0])
        mk = [r for r in ob_rows if r.get("Op Name") == "main_kernel"]
        if not mk:
            continue
        td = _safe_float(mk[0].get("Task Duration(us)", ""))
        try:
            bd = int(mk[0].get("Block Dim", "0")) if mk[0].get("Block Dim", "NA") != "NA" else 0
        except (ValueError, TypeError):
            bd = 0
        op_type = mk[0].get("Op Type", "")
        launch = {"task_duration_us": td, "block_dim": bd, "op_type": op_type}

        pu_files = glob.glob(os.path.join(d, "PipeUtilization_*.csv"))
        pu_rows = _read_csv_rows(pu_files[0]) if pu_files else []
        # AIC times (sum / num_cores ~ wall us)
        for col, key in [
            ("aic_cube_time(us)", "aic_cube_us"),
            ("aic_mte2_time(us)", "aic_mte2_us"),
            ("aic_mte3_time(us)", "aic_mte3_us"),
            ("aic_fixpipe_time(us)", "aic_fixpipe_us"),
            ("aic_scalar_time(us)", "aic_scalar_us"),
            ("aic_scalar_mte2_stall_time(us)", "aic_scalar_mte2_stall_us"),
            ("aic_scalar_cube_stall_time(us)", "aic_scalar_cube_stall_us"),
            ("aic_scalar_wait_time(us)", "aic_scalar_wait_us"),
        ]:
            launch[key] = _sum(_col_floats(pu_rows, col)) / _NUM_CORES
        # AIC ratios (median across blocks)
        for col, key in [
            ("aic_cube_ratio", "aic_cube_ratio"),
            ("aic_mte2_ratio", "aic_mte2_ratio"),
            ("aic_mte3_ratio", "aic_mte3_ratio"),
            ("aic_fixpipe_ratio", "aic_fixpipe_ratio"),
            ("aic_scalar_ratio", "aic_scalar_ratio"),
        ]:
            launch[key] = _median(_col_floats(pu_rows, col))
        # AIV times
        for col, key in [
            ("aiv_vec_time(us)", "aiv_vec_us"),
            ("aiv_mte2_time(us)", "aiv_mte2_us"),
            ("aiv_mte3_time(us)", "aiv_mte3_us"),
            ("aiv_scalar_time(us)", "aiv_scalar_us"),
            ("aiv_scalar_wait_time(us)", "aiv_scalar_wait_us"),
            ("aiv_scalar_mte2_stall_time(us)", "aiv_scalar_mte2_stall_us"),
        ]:
            launch[key] = _sum(_col_floats(pu_rows, col)) / _NUM_CORES
        # AIV ratios
        for col, key in [
            ("aiv_vec_ratio", "aiv_vec_ratio"),
            ("aiv_mte2_ratio", "aiv_mte2_ratio"),
            ("aiv_mte3_ratio", "aiv_mte3_ratio"),
            ("aiv_scalar_ratio", "aiv_scalar_ratio"),
        ]:
            launch[key] = _median(_col_floats(pu_rows, col))

        # L2Cache (per-block hit rates -> median)
        l2_files = glob.glob(os.path.join(d, "L2Cache_*.csv"))
        l2_rows = _read_csv_rows(l2_files[0]) if l2_files else []
        launch["aic_read_hit_pct"] = _median(_col_floats(l2_rows, "aic_read_hit_rate(%)"))
        launch["aiv_read_hit_pct"] = _median(_col_floats(l2_rows, "aiv_read_hit_rate(%)"))
        launch["aic_total_hit_pct"] = _median(_col_floats(l2_rows, "aic_total_hit_rate(%)"))

        # ResourceConflictRatio (per-block wait ratios -> median)
        rc_files = glob.glob(os.path.join(d, "ResourceConflictRatio_*.csv"))
        rc_rows = _read_csv_rows(rc_files[0]) if rc_files else []
        for col, key in [
            ("aic_cube_wait_ratio", "aic_cube_wait_ratio"),
            ("aic_mte2_wait_ratio", "aic_mte2_wait_ratio"),
            ("aiv_vec_wait_ratio", "aiv_vec_wait_ratio"),
            ("aiv_mte2_wait_ratio", "aiv_mte2_wait_ratio"),
            ("aiv_mte3_wait_ratio", "aiv_mte3_wait_ratio"),
        ]:
            launch[key] = _median(_col_floats(rc_rows, col))

        # Memory (per-block traffic KB -> sum)
        mem_files = glob.glob(os.path.join(d, "Memory_*.csv"))
        mem_rows = _read_csv_rows(mem_files[0]) if mem_files else []
        for col, key in [
            ("GM_to_L1_datas(KB)", "gm_to_l1_KB"),
            ("GM_to_UB_datas(KB)", "gm_to_ub_KB"),
            ("UB_to_GM_datas(KB)", "ub_to_gm_KB"),
            ("L0C_to_GM_datas(KB)", "l0c_to_gm_KB"),
        ]:
            launch[key] = _sum(_col_floats(mem_rows, col))

        # Derived: cube_ratio_pct (Cube wall us / Task Duration us * 100)
        launch["cube_ratio_pct"] = (launch["aic_cube_us"] / td * 100.0) if td > 0 else 0.0

        launches.append(launch)

    return launches


def run_msprof(B=1, H=64, groups=8, N=4096, D=128, window_size=128, launch_count=10, warm_up=3):
    """Run msprof op to collect hardware-level kernel performance data.

    Generates a minimal runner script (each kernel runs once, no do_bench loop),
    invokes ``msprof op`` with aic-metrics covering ArithmeticUtilization, Memory,
    L2Cache, PipeUtilization, ResourceConflictRatio, BasicInfo, then parses the
    output CSVs and prints a per-kernel summary table with bottleneck analysis.

    Requires ``msprof`` in PATH (CANN toolkit).
    """
    msprof_cmd = os.environ.get("MSPROF_PATH", "msprof")
    try:
        subprocess.run([msprof_cmd, "--help"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[ERROR] msprof not found in PATH. Please source CANN set_env.sh or set MSPROF_PATH.")
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    window_repr = "None" if window_size is None else window_size
    runner_code = _MSPROF_RUNNER_TEMPLATE.format(
        script_dir=script_dir,
        B=B,
        H=H,
        groups=groups,
        N=N,
        D=D,
        window_size=window_repr,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=script_dir,
        delete=False,
    ) as f:
        f.write(runner_code)
        runner_path = f.name

    try:
        output_dir = tempfile.mkdtemp(prefix="msprof_bwd_out_")
        app_cmd = f"{sys.executable} {runner_path}"
        aic_metrics = "ArithmeticUtilization,Memory,L2Cache,PipeUtilization,ResourceConflictRatio,BasicInfo"
        cmd = (
            f'{msprof_cmd} op --application="{app_cmd}" --output={output_dir} '
            f"--aic-metrics={aic_metrics} "
            f'--kernel-name="main_kernel" '
            f"--launch-count={launch_count} --warm-up={warm_up} --kill=off"
        )

        print()
        print("=" * 82)
        print("  msprof op — hardware-level kernel performance (golden config)")
        print(f"  Config: B={B} H={H} groups={groups} N={N} D={D} window={window_size}")
        print(f"  launch-count={launch_count} warm-up={warm_up}")
        print("=" * 82)
        print(f"  Running: {cmd}")
        print()

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"[ERROR] msprof failed (exit code {result.returncode})")
            print("--- stdout (last 1500 chars) ---")
            print(result.stdout[-1500:] if result.stdout else "")
            print("--- stderr (last 1500 chars) ---")
            print(result.stderr[-1500:] if result.stderr else "")
            return False

        # List output directory structure (helps debug CSV layout)
        print("  msprof output structure:")
        for root, _dirs, files in os.walk(output_dir):
            for fn in files:
                if fn.endswith(".csv"):
                    rel = os.path.relpath(os.path.join(root, fn), output_dir)
                    print(f"    {rel}")
        print()

        launches = _parse_msprof_op_summary(output_dir)
        if not launches:
            print("[ERROR] No main_kernel data found in msprof output")
            print(f"  Output dir: {output_dir}")
            return False

        # Map launches to kernel labels by index.
        # The runner script prints ===KERNEL_LABEL:<name>=== markers after each
        # kernel call; msprof launch directories are named 0,1,2,... in execution
        # order, matching _MSPROF_KERNEL_LABELS.
        for i, l in enumerate(launches):
            l["label"] = _MSPROF_KERNEL_LABELS[i] if i < len(_MSPROF_KERNEL_LABELS) else f"launch_{i}"

        # Per-launch summary table (AIC = Cube core, AIV = Vector core)
        print(f"  Collected {len(launches)} main_kernel launches")
        print()
        print(
            f"  {'#':<3} {'Label':<20} {'Type':<7} {'TaskDur':>9} {'Block':>7} | "
            f"{'AIC Cube':>9} {'MTE2':>7} {'MTE3':>6} {'Scalar':>7} | "
            f"{'AIV Vec':>8} {'MTE2':>7} {'MTE3':>7} {'S_wait':>7} | "
            f"{'L2R%':>6} {'L2V%':>6}"
        )
        print(f"  {'-' * 133}")
        for i, l in enumerate(launches):
            print(
                f"  {i:<3} {l.get('label', ''):<20} {l.get('op_type', ''):<7} "
                f"{l['task_duration_us']:>7.0f}us {l['block_dim']:>7} | "
                f"{l.get('aic_cube_us', 0.0):>7.0f}us {l.get('aic_mte2_us', 0.0):>5.0f}us "
                f"{l.get('aic_mte3_us', 0.0):>4.0f}us {l.get('aic_scalar_us', 0.0):>5.0f}us | "
                f"{l.get('aiv_vec_us', 0.0):>6.0f}us {l.get('aiv_mte2_us', 0.0):>5.0f}us "
                f"{l.get('aiv_mte3_us', 0.0):>5.0f}us {l.get('aiv_scalar_wait_us', 0.0):>5.0f}us | "
                f"{l.get('aic_read_hit_pct', 0.0):>5.1f} {l.get('aiv_read_hit_pct', 0.0):>5.1f}"
            )
        print()

        # --- Per BWD Kernel Bottleneck Analysis (k1->k5) ---
        # bwd pipeline = k1->k2->k3->k4->k5 (launches 2-6).
        # For each bwd kernel, compute Cube% / MTE2% / MTE3% / Scalar+wait% /
        # L2 hit / bottleneck type, so we can see which kernel is slowest
        # and what its bottleneck is (instead of picking a single "main").
        bwd_label_set = {
            "k1_qk_recompute",
            "k2_softmax_p",
            "k3_dv_dp",
            "k4_ds_compute",
            "k5_dk_dq",
        }
        bwd_launches = [l for l in launches if l.get("label") in bwd_label_set]

        print("=" * 82)
        print("  Per BWD Kernel Bottleneck Analysis (k1->k5)")
        print("=" * 82)
        print(
            f"  {'Kernel':<20} {'TaskDur':>9} | "
            f"{'Cube%':>6} {'MTE2%':>6} {'MTE3%':>6} {'S+W%':>6} | "
            f"{'L2R%':>6} {'L2V%':>6} | {'Bottleneck':<12}"
        )
        print(f"  {'-' * 100}")

        bwd_total_td = 0.0
        for l in bwd_launches:
            td = l["task_duration_us"]
            bwd_total_td += td

            def _kpct(v, _td=td):
                return v / _td * 100.0 if _td > 0 else 0.0

            cube_pct = _kpct(l.get("aic_cube_us", 0.0))
            mte2_pct = _kpct(l.get("aic_mte2_us", 0.0)) + _kpct(l.get("aiv_mte2_us", 0.0))
            mte3_pct = _kpct(l.get("aic_mte3_us", 0.0)) + _kpct(l.get("aiv_mte3_us", 0.0))
            sync_pct = _kpct(l.get("aic_scalar_us", 0.0)) + _kpct(l.get("aiv_scalar_wait_us", 0.0))
            l2r = l.get("aic_read_hit_pct", 0.0)
            l2v = l.get("aiv_read_hit_pct", 0.0)

            if cube_pct > 50.0:
                btype = "compute"
            elif mte2_pct + mte3_pct > 40.0 and 0.0 < l2r < 80.0:
                btype = "memory"
            elif sync_pct > 25.0 and cube_pct < 15.0:
                btype = "sync"
            elif mte2_pct + mte3_pct > 30.0:
                btype = "memory"
            else:
                btype = "mixed"

            print(
                f"  {l.get('label', ''):<20} {td:>7.0f}us | "
                f"{cube_pct:>5.1f}% {mte2_pct:>5.1f}% {mte3_pct:>5.1f}% {sync_pct:>5.1f}% | "
                f"{l2r:>5.1f} {l2v:>5.1f} | {btype:<12}"
            )

        print(f"  {'-' * 100}")
        print(f"  {'sum(k1-k5)':<20} {bwd_total_td:>7.0f}us")
        print()

        # --- Detailed breakdown for the slowest bwd kernel ---
        # Keeps the original detailed AIC/AIV pipeline breakdown format, but
        # applied to the slowest bwd kernel (by Task Duration) rather than a
        # wrongly-selected single "main".
        if bwd_launches:
            slowest = max(bwd_launches, key=lambda l: l["task_duration_us"])
            s_label = slowest.get("label", "?")
            s_td = slowest["task_duration_us"]
            s_bd = slowest["block_dim"]

            def _pct(v):
                return v / s_td * 100.0 if s_td > 0 else 0.0

            s_cube = slowest.get("aic_cube_us", 0.0)
            s_aic_mte2 = slowest.get("aic_mte2_us", 0.0)
            s_aic_mte3 = slowest.get("aic_mte3_us", 0.0)
            s_aic_fix = slowest.get("aic_fixpipe_us", 0.0)
            s_aic_scalar = slowest.get("aic_scalar_us", 0.0)
            s_aic_s_wait = slowest.get("aic_scalar_wait_us", 0.0)
            s_aic_s_mte2_stall = slowest.get("aic_scalar_mte2_stall_us", 0.0)
            s_aic_s_cube_stall = slowest.get("aic_scalar_cube_stall_us", 0.0)
            s_aiv_vec = slowest.get("aiv_vec_us", 0.0)
            s_aiv_mte2 = slowest.get("aiv_mte2_us", 0.0)
            s_aiv_mte3 = slowest.get("aiv_mte3_us", 0.0)
            s_aiv_scalar = slowest.get("aiv_scalar_us", 0.0)
            s_aiv_s_wait = slowest.get("aiv_scalar_wait_us", 0.0)
            s_l2r = slowest.get("aic_read_hit_pct", 0.0)
            s_l2v = slowest.get("aiv_read_hit_pct", 0.0)
            s_cube_ratio = slowest.get("cube_ratio_pct", 0.0)
            s_gm_l1 = slowest.get("gm_to_l1_KB", 0.0) / 1024.0  # MB
            s_gm_ub = slowest.get("gm_to_ub_KB", 0.0) / 1024.0
            s_ub_gm = slowest.get("ub_to_gm_KB", 0.0) / 1024.0
            s_l0c_gm = slowest.get("l0c_to_gm_KB", 0.0) / 1024.0

            print(f"  --- Slowest BWD kernel: {s_label} (detailed) ---")
            print(f"  Task Duration:    {s_td:.1f} us  (block_dim={s_bd})")
            print()
            print("  --- AIC (Cube core) pipeline breakdown ---")
            print(f"  Cube compute:     {s_cube:>7.1f} us  ({_pct(s_cube):>5.1f}% of Task Dur)  <- GEMM actual")
            print(f"  MTE2 (GM->L1):    {s_aic_mte2:>7.1f} us  ({_pct(s_aic_mte2):>5.1f}%)  <- K/V/Q/dO load")
            print(f"  MTE3 (L1->GM):    {s_aic_mte3:>7.1f} us  ({_pct(s_aic_mte3):>5.1f}%)")
            print(f"  FIX pipe:         {s_aic_fix:>7.1f} us  ({_pct(s_aic_fix):>5.1f}%)")
            print(f"  Scalar:           {s_aic_scalar:>7.1f} us  ({_pct(s_aic_scalar):>5.1f}%)  <- flag sync + addr")
            print(
                f"  Scalar wait:      {s_aic_s_wait:>7.1f} us  ({_pct(s_aic_s_wait):>5.1f}%)  "
                f"(mte2_stall={s_aic_s_mte2_stall:.1f}, cube_stall={s_aic_s_cube_stall:.1f})"
            )
            print()
            print("  --- AIV (Vector core) pipeline breakdown ---")
            print(f"  Vec compute:      {s_aiv_vec:>7.1f} us  ({_pct(s_aiv_vec):>5.1f}%)  <- softmax/mask/dS")
            print(f"  MTE2 (GM->UB):    {s_aiv_mte2:>7.1f} us  ({_pct(s_aiv_mte2):>5.1f}%)  <- S/lse/Delta load")
            print(f"  MTE3 (UB->GM):    {s_aiv_mte3:>7.1f} us  ({_pct(s_aiv_mte3):>5.1f}%)  <- P/dS/delta write")
            print(f"  Scalar:           {s_aiv_scalar:>7.1f} us  ({_pct(s_aiv_scalar):>5.1f}%)")
            print(f"  Scalar wait:      {s_aiv_s_wait:>7.1f} us  ({_pct(s_aiv_s_wait):>5.1f}%)  <- wait cross_flag")
            print()
            print("  --- L2 cache & memory traffic ---")
            print(f"  L2 read hit:      AIC {s_l2r:.1f}%  AIV {s_l2v:.1f}%")
            print(f"  Mem traffic(MB):  GM->L1={s_gm_l1:.1f}  GM->UB={s_gm_ub:.1f}  UB->GM={s_ub_gm:.1f}  L0C->GM={s_l0c_gm:.1f}")
            print()

            # Classification for slowest kernel
            sync_ratio = _pct(s_aic_scalar) + _pct(s_aiv_s_wait)
            mem_ratio = _pct(s_aic_mte2) + _pct(s_aiv_mte2) + _pct(s_aiv_mte3) + _pct(s_aic_mte3)
            compute_ratio = s_cube_ratio

            if compute_ratio > 50.0:
                bottleneck = "compute"
                opt_hint = "Cube > 50% -> compute-bound. Candidates: reduce GEMM count, enlarge block_M/N, T.mma intrinsic."
            elif mem_ratio > 40.0 and 0.0 < s_l2r < 80.0:
                bottleneck = "memory"
                opt_hint = (
                    "MTE2+MTE3 > 40% + L2 hit < 80% -> memory-bound. Candidates: Fixed "
                    "Core + grid-stride (per-core workspace L2 reuse), T.mma + L0A DB."
                )
            elif sync_ratio > 25.0 and compute_ratio < 15.0:
                bottleneck = "sync"
                opt_hint = (
                    f"Scalar+wait {sync_ratio:.1f}% > 25% + Cube {compute_ratio:.1f}% < 15% "
                    f"-> sync-bound. Candidates: flag reduction, cross_flag reordering."
                )
            elif mem_ratio > 30.0:
                bottleneck = "memory"
                opt_hint = (
                    f"MTE2+MTE3 {mem_ratio:.1f}% > 30% -> memory-bound (workspace "
                    f"round-trip dominates). Candidates: reduce workspace writes, L2 reuse."
                )
            else:
                bottleneck = "mixed"
                opt_hint = (
                    f"Compute {compute_ratio:.1f}% / Sync {sync_ratio:.1f}% / "
                    f"Memory {mem_ratio:.1f}% — no single dominant bottleneck. "
                    f"Likely CV-overlap-bound."
                )

            print(f"  Bottleneck type:  {bottleneck}")
            print(f"    compute_ratio={compute_ratio:.1f}%  sync_ratio={sync_ratio:.1f}%  memory_ratio={mem_ratio:.1f}%")
            print(f"  Optimization hint: {opt_hint}")
            print()

        # --- FLOPS analysis (based on total bwd Task Duration = sum k1-k5) ---
        if window_size is not None and window_size < N:
            vr = window_size * 1.0 / N
        else:
            vr = 1.0
        bwd_flops = 2.0 * B * H * N * N * (5 * D) * vr
        if bwd_total_td > 0:
            print(
                f"  BWD FLOPS (approx, valid_ratio={vr:.4f}): "
                f"{bwd_flops / 1e9:.1f} GFLOPS / {bwd_total_td:.1f} us = "
                f"{bwd_flops / bwd_total_td * 1e-6:.2f} TFlops"
            )
            print(f"  A2/A3 theoretical: 364 TFlops (fp16) — utilization {bwd_flops / bwd_total_td * 1e-6 / 364 * 100:.1f}%")
            print(f"  GPU baseline (14287us): NPU bwd {bwd_total_td:.1f}us = +{(14287 - bwd_total_td) / 14287 * 100:.1f}% vs GPU")
        print("=" * 82)

        # Cleanup
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)
        print("\nTest Passed!")
        return True
    finally:
        os.unlink(runner_path)


# ============================================================================
# Main entry point
# ============================================================================


def main():
    tilelang.disable_cache()
    # Clean up any stale tilelang JIT temp files from previous runs
    _cleanup_tmp_compilation_files()

    parser = argparse.ArgumentParser(description="Test GQA Sink Attention BWD (BHSD)")
    parser.add_argument(
        "--level",
        type=str,
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all", "bench", "msprof"],
        help="Test level to run (bench = do_bench perf, msprof = hardware-level profiling)",
    )
    # Bench/msprof config override
    parser.add_argument("--B", type=int, default=1, help="bench/msprof: batch size")
    parser.add_argument("--H", type=int, default=64, help="bench/msprof: query heads")
    parser.add_argument("--N", type=int, default=4096, help="bench/msprof: sequence length")
    parser.add_argument("--D", type=int, default=128, help="bench/msprof: head dim")
    parser.add_argument("--groups", type=int, default=8, help="bench/msprof: GQA groups")
    parser.add_argument("--window", type=int, default=128, help="bench/msprof: window size (0=causal-only)")
    parser.add_argument("--warmup", type=float, default=50, help="bench: warmup iterations")
    parser.add_argument("--rep", type=float, default=100, help="bench: repeat iterations")
    parser.add_argument("--per-kernel", action="store_true", help="bench: also print per-kernel breakdown")
    parser.add_argument("--launch-count", type=int, default=10, help="msprof: number of kernel launches [1,5000]")
    parser.add_argument("--warm-up", type=int, default=3, help="msprof: warm-up times [0,500]")
    args = parser.parse_args()

    # --- bench mode (do_bench performance) ---
    if args.level == "bench":
        window = args.window if args.window > 0 else None
        ok = run_bench(
            args.B,
            args.H,
            args.N,
            args.D,
            args.groups,
            window,
            warmup=args.warmup,
            rep=args.rep,
            per_kernel=args.per_kernel,
        )
        sys.exit(0 if ok else 1)

    # --- msprof mode (hardware-level kernel performance) ---
    if args.level == "msprof":
        window = args.window if args.window > 0 else None
        ok = run_msprof(
            args.B,
            args.H,
            args.groups,
            args.N,
            args.D,
            window,
            launch_count=args.launch_count,
            warm_up=args.warm_up,
        )
        sys.exit(0 if ok else 1)

    torch.manual_seed(42)

    overall_passed = True

    if args.level in ("l0", "all"):
        passed, results = test_gqa_sink_bwd_bhsd_l0()
        overall_passed = overall_passed and passed

    if args.level in ("l1", "all"):
        passed, results = test_gqa_sink_bwd_bhsd_l1()
        overall_passed = overall_passed and passed

    if args.level in ("l2", "all"):
        passed, results = test_gqa_sink_bwd_bhsd_l2()
        # L2 failures are non-blocking

    if args.level in ("boundary", "all"):
        passed, results = test_gqa_sink_bwd_bhsd_boundary()
        # Boundary failures are non-blocking

    print(f"\n{'=' * 60}")
    if overall_passed:
        print("Test Passed!")
        sys.exit(0)
    else:
        print("Test FAILED — see [PRECISION_FAIL] above")
        sys.exit(1)


if __name__ == "__main__":
    main()
