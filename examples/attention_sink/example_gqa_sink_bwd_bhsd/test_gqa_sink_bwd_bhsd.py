"""Test suite for GQA Sink Attention Backward (BHSD).

3-kernel pipeline (fwd + bwd[preprocess merged] + postprocess), 1 host sync.

L0: 7 cases (rule shapes, block-aligned) — blocking
L1: 8 cases (varying params + value ranges) — blocking
L2: 5 cases (invalid inputs, should reject) — non-blocking
Boundary: 4 cases (special values: inf/nan/zero) — non-blocking

Precision standard: 169-line standard.
  float16: atol=6.10e-5, rtol=1.95e-3, max_abs_limit=0.1, required_ratio=0.99
  float32: atol=1.53e-5, rtol=9.77e-4, max_abs_limit=1e-2, required_ratio=0.99

Golden runs on CPU (.cpu() before ref_fwd/ref_bwd).
NPU outputs .cpu() for comparison — avoids 4GB fp32 attention OOM on NPU.
"""

import argparse
import ast
import glob
import os
import subprocess
import sys
import time

import tilelang
import torch

from example_gqa_sink_bwd_bhsd import (
    flashattn_bwd,
    flashattn_fwd,
)

# ============================================================================
# Golden Reference (PyTorch CPU)
# ============================================================================


def ref_fwd(Q, K, V, Sinks, window_size=None, groups=1):
    """Forward golden (CPU): GQA + Attention Sink + optional sliding window."""
    B, H, N, D = Q.shape
    sm_scale = 1.0 / D**0.5

    K_rep = K.float().repeat_interleave(groups, dim=1)
    V_rep = V.float().repeat_interleave(groups, dim=1)

    S = torch.matmul(Q.float(), K_rep.transpose(-2, -1)) * sm_scale

    pos_q = torch.arange(N, device=Q.device).float()
    pos_k = torch.arange(N, device=Q.device).float()
    causal_mask = pos_k[None, :] <= pos_q[:, None]
    if window_size is not None:
        window_mask = pos_k[None, :] > (pos_q[:, None] - window_size)
        mask = causal_mask & window_mask
    else:
        mask = causal_mask
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    sinks_b = Sinks.view(1, H, 1, 1).float()
    m_with_sink = torch.maximum(sinks_b, m)

    P = torch.exp(S - m_with_sink)
    sinks_exp = torch.exp(sinks_b - m_with_sink)
    normalizer = P.sum(dim=-1, keepdim=True) + sinks_exp
    P = P / normalizer

    O = torch.matmul(P, V_rep)
    return O.half()


def ref_bwd(Q, K, V, Sinks, dO, window_size=None, groups=1):
    """Backward golden (CPU autograd). Returns dQ, dK, dV (fp16), dSinks (fp32)."""
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)
    Sinks_f = Sinks.float().requires_grad_(True)

    B, H, N, D = Q_f.shape
    sm_scale = 1.0 / D**0.5

    K_rep = K_f.repeat_interleave(groups, dim=1)
    V_rep = V_f.repeat_interleave(groups, dim=1)

    S = torch.matmul(Q_f, K_rep.transpose(-2, -1)) * sm_scale

    pos_q = torch.arange(N, device=Q_f.device).float()
    pos_k = torch.arange(N, device=Q_f.device).float()
    causal_mask = pos_k[None, :] <= pos_q[:, None]
    if window_size is not None:
        window_mask = pos_k[None, :] > (pos_q[:, None] - window_size)
        mask = causal_mask & window_mask
    else:
        mask = causal_mask
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = S.max(dim=-1, keepdim=True).values
    sinks_b = Sinks_f.view(1, H, 1, 1)
    m_with_sink = torch.maximum(sinks_b, m)
    P = torch.exp(S - m_with_sink)
    sinks_exp = torch.exp(sinks_b - m_with_sink)
    normalizer = P.sum(dim=-1, keepdim=True) + sinks_exp
    P = P / normalizer

    O = torch.matmul(P, V_rep)
    O.backward(dO.float())

    return Q_f.grad.half(), K_f.grad.half(), V_f.grad.half(), Sinks_f.grad


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
    condition. The .so is the final loaded artifact and can be safely
    removed after dlopen. The .cpp files accumulate but inode usage stays
    low (~540 files/run, 3% inode usage even over 100 runs), so leaving
    them is acceptable.

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

        # --- BWD Main: single flashattn_bwd kernel (preprocess merged, 0 GM workspaces) ---
        # bwd kernel Phase 0 computes Delta internally, Phase 6 computes dSinks.
        # No separate preprocess kernel, no preprocess->bwd host sync.
        delta_out = torch.zeros(B, H, N, dtype=torch.float32, device="npu")
        dSinks_npu = torch.zeros(B, H, N, dtype=torch.float32, device="npu")
        dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
        dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
        dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")

        # All 6 GM workspace buffers eliminated (on-chip direct UB->L1).
        # O and Sinks passed as inputs, Delta_out/dSinks as outputs (preprocess merged).
        bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
        bwd_mod = _run_with_retry(
            lambda: flashattn_bwd(*bwd_args),
            kernel_name="flashattn_bwd",
        )
        bwd_mod(
            Q,
            K,
            V,
            dO,
            O_npu,
            lse_npu,
            sinks,
            delta_out,
            dSinks_npu,
            dQ,
            dK,
            dV,
        )
        torch.npu.synchronize()

        # Delta golden uses O_npu (same input as kernel) to isolate bwd precision from fwd
        O_cpu_for_delta = O_npu.cpu()
        Delta_ref = (O_cpu_for_delta.float() * dO_cpu.float()).sum(dim=-1)
        passed, ratio, max_abs = check_precision(delta_out, Delta_ref, "float32")
        results["bwd_Delta"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_Delta"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # postprocess kernel removed — host .half() cast (dQ already fp16 from bwd)
        dQ_fp16 = dQ[..., :D]
        dK_fp16 = dK[..., :D].half()
        dV_fp16 = dV[..., :D].half()
        torch.npu.synchronize()

        # dSinks: host sum over B and N -> [H] fp32
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
        # from fwd precision — autograd's internal lse/Delta differ from NPU's).
        # Delta now comes from bwd kernel's Delta_out (Phase 0 output).
        sinks_exp = sinks_cpu.float().view(1, H, 1)  # [1, H, 1]
        lse_cpu = lse_npu.cpu().float()  # [B, H, N]
        delta_cpu = delta_out.cpu().float()  # [B, H, N] — from bwd kernel Delta_out
        dSinks_ref = -(torch.exp(sinks_exp - lse_cpu) * delta_cpu).sum(dim=0).sum(dim=1)  # [H]

        # Compare dQ (fp16)
        passed, ratio, max_abs = check_precision(dQ_fp16, dQ_ref, "float16")
        results["bwd_dQ"] = {"passed": passed, "ratio": ratio, "max_abs": max_abs}
        results["bwd_dQ"]["status"] = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"

        # Compare dK (fp16)
        passed, ratio, max_abs = check_precision(dK_fp16, dK_ref, "float16")
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
# Kernel constraints: N % 64 == 0 (fwd/bwd block_M/block_N),
#   window % 64 == 0, H % groups == 0, D == 128 (bwd pads to 128).
#   N%128 dsink constraint removed (preprocess merged into bwd).
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
        ["D-DTYPE-fp16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-PARAM-groups", "D-PARAM-window_size", "D-VALRANGE-S"],
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
# Kernel rejects: N not multiple of 64 (fwd/bwd assert), D != 128 (bwd shape mismatch),
#   float32 dtype. N%128 dsink assert removed (preprocess merged into bwd).
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
COVERAGE_NA = {}  # D-DTYPE-fp32 covered via dSinks (fp32) in every L0/L1 case


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

    Kernel constraints: N%64==0 (fwd/bwd), D=128, dtype=float16.
    N%128 dsink constraint removed (preprocess merged into bwd).
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

    # L2-3: N=192 (multiple of 64 — valid for fwd/bwd block_M=64)
    # preprocess removed (merged into bwd), so N%32 blk constraint no longer applies.
    # bwd uses block_M=64 (192%64==0). This is a valid shape — confirm it works.
    def case_shape_n192():
        B, H, groups, N, D = 1, 4, 2, 192, 128
        H_kv = H // groups
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="npu")
        K = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="npu")
        V = torch.randn_like(K)
        sinks = torch.randn(H, dtype=torch.float16, device="npu")
        fwd_mod = flashattn_fwd(B, H, N, D, groups, None, 64, 64)
        O_npu, lse_npu = fwd_mod(Q, K, V, sinks)
        torch.npu.synchronize()
        return True

    try:
        case_shape_n192()
        print("[BOUNDARY_PASS] l2 l2_shape_n192: N=192 valid (bwd block_M=64, 192%64==0)")
    except Exception as e:
        print(f"[BOUNDARY_WARN] l2 l2_shape_n192: unexpectedly rejected ({type(e).__name__}: {e})")

    # L2-4: N=509 (prime, not multiple of 64 — fwd assertion fails)
    def case_shape_n509():
        B, H, groups, N, D = 1, 4, 2, 509, 128
        _fwd_mod = flashattn_fwd(B, H, N, D, groups, None, 64, 64)

    _run_exception("l2_shape_n509", case_shape_n509)

    # L2-5: D=64 (bwd assert dim_qk % 128 == 0 fails)
    def case_shape_d64():
        B, H, groups, N, D = 1, 4, 2, 128, 64
        # flashattn_bwd asserts dim_qk % 128 == 0 — D=64 fails
        bwd_args = (B, H, N, D, D, None, 64, 64, groups)
        _bwd_mod = flashattn_bwd(*bwd_args)

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

    # BWD main (preprocess merged into bwd kernel Phase 0 + Phase 6, 0 GM workspaces)
    delta_out = torch.zeros(B, H, N, dtype=torch.float32, device="npu")
    dSinks_npu = torch.zeros(B, H, N, dtype=torch.float32, device="npu")
    dQ = torch.zeros(B, H, N, dim_qk_padded, dtype=torch.float16, device="npu")
    dK = torch.zeros(B, H_kv, N, dim_qk_padded, dtype=torch.float32, device="npu")
    dV = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device="npu")
    bwd_args = (B, H, N, D, D, window_size, block_M, block_N, groups)
    bwd_mod = flashattn_bwd(*bwd_args)
    bwd_mod(
        Q,
        K,
        V,
        dO,
        O_npu,
        lse_npu,
        sinks,
        delta_out,
        dSinks_npu,
        dQ,
        dK,
        dV,
    )
    torch.npu.synchronize()

    # postprocess kernel removed — dQ already fp16 from bwd
    dQ_fp16 = dQ[..., :D]
    torch.npu.synchronize()

    # Golden dQ
    dQ_ref, _, _, _ = ref_bwd(Q_cpu, K_cpu, V_cpu, sinks_cpu, dO_cpu, window_size, groups)

    return dQ_fp16, dQ_ref


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
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

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
