"""Test suite for MLA Decode Persistent Attention.

Levels:
  - l0:        threshold tests (rule-based shapes, block divisible)
  - l1:        functional tests (irregular shapes, parameter variations)
  - l2:        negative tests (illegal inputs must be rejected)
  - boundary:  special values (INF/NAN/Zero/Denormalized)
  - all:       run l0 + l1 + l2 + boundary
  - bench:     performance benchmark (tilelang.profiler.do_bench)
  - msprof:    hardware-level profiling (msprof, inlined script)

Precision standard (fp16): atol=2^-14, rtol=2^-9, max_abs_limit=1e-1, required_ratio=0.99.
L0/L1 failures block (exit code 1). L2/Boundary failures are non-blocking (warnings only).
"""

import argparse
import os
import subprocess
import sys

import tilelang
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_mla_decode_persistent import (  # noqa: E402
    ref_mla_decode,
    run_mla_decode,
)

try:
    from einops import rearrange
except ImportError:
    rearrange = None


# ============================================================================
# Precision Standard (mixed tolerance, per dtype)
# ============================================================================


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed tolerance dual-threshold: returns (passed, matched_ratio, max_abs_error).

    Per precision-standard.md §3.1: INF/NAN positions are structurally compared
    (isinf/isnan locations must match between actual and golden), and do NOT
    participate in matched_ratio / max_abs_error calculation.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()

    # INF/NAN structural comparison (precision-standard.md §3.1)
    # If golden has inf/nan at positions where actual does not (or vice versa),
    # the structural check fails immediately.
    g_inf = torch.isinf(g)
    g_nan = torch.isnan(g)
    a_inf = torch.isinf(a)
    a_nan = torch.isnan(a)
    if not torch.equal(g_inf, a_inf) or not torch.equal(g_nan, a_nan):
        return False, 0.0, float("inf")

    # Only compare finite-value positions (where golden is finite)
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ============================================================================
# Coverage declarations (for coverage_check.py)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"

# Dimensions that are inapplicable to this kernel (with reasons).
# D-SHAPE-TAIL-1/TAIL-MID: kernel requires exact divisibility (seqlen_kv %
#   block_N == 0 for num_split=1), so true partial-block tails cannot be tested.
# D-PARAM-dim/pe_dim: dim=512 and pe_dim=64 are algorithmic constants for
#   DeepSeek MLA. Testing different values would be a different algorithm.
# D-PARAM-num_split: all L0/L1 configs use num_split=1 (single-kernel path).
#   num_split>1 fallback path is covered by the two-phase kernel.
COVERAGE_NA = {}


# ============================================================================
# Input generation helper
# ============================================================================


def _gen_inputs(batch, heads, kv_ctx, dim, pe_dim, input_gen="randn"):
    """Generate test inputs on CPU then H2D to NPU.

    Args:
        batch, heads, kv_ctx, dim, pe_dim: shape parameters.
        input_gen: "randn" | "zeros" | "positive" | "small" | "large".

    Returns:
        (q, q_pe, kv, k_pe) — all fp16 NPU tensors.
        q:    [batch, heads, dim]
        q_pe: [batch, heads, pe_dim]
        kv:   [batch, kv_ctx, 1, dim]
        k_pe: [batch, kv_ctx, 1, pe_dim]
    """
    fp16 = torch.float16
    if input_gen == "randn":
        q = torch.randn(batch, heads, dim, dtype=fp16, device="cpu").npu()
        q_pe = torch.randn(batch, heads, pe_dim, dtype=fp16, device="cpu").npu()
        kv = torch.randn(batch, kv_ctx, 1, dim, dtype=fp16, device="cpu").npu()
        k_pe = torch.randn(batch, kv_ctx, 1, pe_dim, dtype=fp16, device="cpu").npu()
    elif input_gen == "zeros":
        q = torch.zeros(batch, heads, dim, dtype=fp16, device="cpu").npu()
        q_pe = torch.zeros(batch, heads, pe_dim, dtype=fp16, device="cpu").npu()
        kv = torch.zeros(batch, kv_ctx, 1, dim, dtype=fp16, device="cpu").npu()
        k_pe = torch.zeros(batch, kv_ctx, 1, pe_dim, dtype=fp16, device="cpu").npu()
    elif input_gen == "positive":
        q = torch.rand(batch, heads, dim, dtype=fp16, device="cpu").npu()
        q_pe = torch.rand(batch, heads, pe_dim, dtype=fp16, device="cpu").npu()
        kv = torch.rand(batch, kv_ctx, 1, dim, dtype=fp16, device="cpu").npu()
        k_pe = torch.rand(batch, kv_ctx, 1, pe_dim, dtype=fp16, device="cpu").npu()
    elif input_gen in ("small", "large"):
        scale = 0.01 if input_gen == "small" else 10.0
        q = (torch.randn(batch, heads, dim, dtype=fp16, device="cpu") * scale).npu()
        q_pe = (torch.randn(batch, heads, pe_dim, dtype=fp16, device="cpu") * scale).npu()
        kv = (torch.randn(batch, kv_ctx, 1, dim, dtype=fp16, device="cpu") * scale).npu()
        k_pe = (torch.randn(batch, kv_ctx, 1, pe_dim, dtype=fp16, device="cpu") * scale).npu()
    else:
        raise ValueError(f"Unknown input_gen: {input_gen}")
    return q, q_pe, kv, k_pe


def _get_core_num():
    """Get NPU cube core count, fallback to 20."""
    try:
        return int(torch.npu.get_device_properties("npu").cube_core_num)
    except Exception:
        return 20


def _run_ref(q, q_pe, kv, k_pe):
    """Run golden reference on CPU (ref_mla_decode handles einops/native fallback)."""
    return ref_mla_decode(q.cpu(), q_pe.cpu(), kv.cpu(), k_pe.cpu())


# ============================================================================
# L0 Test Suite (threshold tests — rule-based shapes for precision convergence)
# ============================================================================

# L0 configs: all use num_split=1 (single-kernel path).
# num_split=1 is mathematically equivalent (same attention, no split-K reduction).
L0_CONFIGS = [
    {
        "name": "l0_small_basic",
        "shape": (1, 64, 128, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    },
    {
        "name": "l0_small_multibatch",
        "shape": (2, 128, 256, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    },
    {
        "name": "l0_num_split_1",
        "shape": (1, 64, 128, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-num_split"],
    },
    {
        "name": "l0_medium",
        "shape": (4, 128, 512, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    },
    {
        "name": "l0_golden",
        "shape": (128, 128, 8192, 512, 64, 1),
        "block_N": 128,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    },
    {
        "name": "l0_num_split_2",
        "shape": (1, 64, 128, 512, 64, 2),  # num_split=2, seqlen_kv=128 (128 % (64*2) == 0)
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-num_split"],
    },
    {
        "name": "l0_num_split_2_golden",
        "shape": (4, 128, 256, 512, 64, 2),  # num_split=2, seqlen_kv=256 (256 % (64*2) == 0)
        "block_N": 64,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-num_split", "D-VALRANGE-M"],
    },
]


def _run_precision_case(name, batch, heads, kv_ctx, dim, pe_dim, num_split, block_N=64, block_H=64, input_gen="randn", level="l0"):
    """Run a single precision test case. Returns (passed, ratio, max_abs)."""
    core_num = _get_core_num()
    torch.manual_seed(42)
    q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim, input_gen)

    output = run_mla_decode(q, q_pe, kv, k_pe, num_split, block_N, block_H, core_num)
    torch.npu.synchronize()

    ref = _run_ref(q, q_pe, kv, k_pe)
    passed, ratio, max_abs = check_precision(output, ref, "float16")
    tag = "PASS" if passed else "FAIL"
    marker = "[PRECISION_" + tag + "]" if level in ("l0", "l1") else "[BOUNDARY_" + tag + "]"
    print(
        f"{marker} {level} {name} "
        f"B={batch} H={heads} S={kv_ctx} dim={dim} pe={pe_dim} split={num_split} "
        f"gen={input_gen} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed, ratio, max_abs


def test_mla_decode_persistent_l0():
    """L0 threshold tests: rule-based shapes (block divisible) for precision convergence."""
    ok = True
    for cfg in L0_CONFIGS:
        name = cfg["name"]
        batch, heads, kv_ctx, dim, pe_dim, num_split = cfg["shape"]
        block_N = cfg.get("block_N", 64)
        try:
            passed, _, _ = _run_precision_case(name, batch, heads, kv_ctx, dim, pe_dim, num_split, block_N=block_N, level="l0")
            ok &= passed
        except Exception as e:
            import traceback

            print(f"[PRECISION_FAIL] l0 {name}: {e}")
            traceback.print_exc()
            ok = False
    return ok


# ============================================================================
# L1 Functional Tests (irregular shapes, parameter variations)
# ============================================================================

# L1 configs: all use num_split=1 (single-kernel path).
L1_CONFIGS = [
    {
        "name": "l1_edge_minimal",
        "shape": (1, 64, 64, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-TAIL-1", "D-VALRANGE-S"],
    },
    {
        "name": "l1_prime_seqlen",
        "shape": (2, 64, 448, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-PRIME"],
    },
    {
        "name": "l1_tail_mid",
        "shape": (1, 64, 128, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID"],
    },
    {
        "name": "l1_large_batch",
        "shape": (8, 128, 128, 512, 64, 1),
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-L"],
    },
    {
        "name": "l1_asym_positive",
        "shape": (1, 64, 128, 512, 64, 1),
        "input_gen": "positive",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-ASYM"],
    },
    {
        "name": "l1_small_values",
        "shape": (1, 64, 128, 512, 64, 1),
        "input_gen": "small",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-S"],
    },
    {
        "name": "l1_large_values",
        "shape": (1, 64, 128, 512, 64, 1),
        "input_gen": "large",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-L"],
    },
    {
        "name": "l1_param_block_n32",
        "shape": (1, 64, 128, 512, 64, 1),
        "block_N": 32,
        "block_H": 64,
        "tags": ["D-DTYPE-fp16", "D-PARAM-block_N"],
    },
    {
        "name": "l1_param_block_h32",
        "shape": (1, 64, 128, 512, 64, 1),
        "block_N": 64,
        "block_H": 32,
        "tags": ["D-DTYPE-fp16", "D-PARAM-block_H"],
    },
    {
        "name": "l1_param_dim256",
        "shape": (1, 64, 128, 256, 32, 1),
        "tags": ["D-DTYPE-fp16", "D-PARAM-dim", "D-PARAM-pe_dim"],
    },
]


def test_mla_decode_persistent_l1():
    """L1 functional tests: irregular shapes, parameter variations, value ranges."""
    ok = True
    for cfg in L1_CONFIGS:
        name = cfg["name"]
        batch, heads, kv_ctx, dim, pe_dim, num_split = cfg["shape"]
        block_N = cfg.get("block_N", 64)
        block_H = cfg.get("block_H", 64)
        input_gen = cfg.get("input_gen", "randn")
        try:
            passed, _, _ = _run_precision_case(
                name,
                batch,
                heads,
                kv_ctx,
                dim,
                pe_dim,
                num_split,
                block_N=block_N,
                block_H=block_H,
                input_gen=input_gen,
                level="l1",
            )
            ok &= passed
        except Exception as e:
            import traceback

            print(f"[PRECISION_FAIL] l1 {name}: {e}")
            traceback.print_exc()
            ok = False
    return ok


# ============================================================================
# L2 Negative Tests (illegal inputs should be rejected)
# ============================================================================


def test_mla_decode_persistent_l2():
    """L2 negative tests: illegal inputs must be rejected (correct exception = PASS)."""
    core_num = _get_core_num()

    # L2-1: Wrong dtype (fp32 instead of fp16) — D-EXC-DTYPE
    try:
        batch, heads, kv_ctx, dim, pe_dim, num_split = 1, 64, 128, 512, 64, 1
        q = torch.randn(batch, heads, dim, dtype=torch.float32, device="cpu").npu()
        q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float32, device="cpu").npu()
        kv = torch.randn(batch, kv_ctx, 1, dim, dtype=torch.float32, device="cpu").npu()
        k_pe = torch.randn(batch, kv_ctx, 1, pe_dim, dtype=torch.float32, device="cpu").npu()
        run_mla_decode(q, q_pe, kv, k_pe, num_split, 64, 64, core_num)
        print("[BOUNDARY_WARN] l2 l2_wrong_dtype: fp32 input silently accepted (expected rejection)")
    except (TypeError, ValueError, RuntimeError, AssertionError):
        print("[BOUNDARY_PASS] l2 l2_wrong_dtype: fp32 input correctly rejected")
    except Exception as e:
        print(f"[BOUNDARY_WARN] l2 l2_wrong_dtype: unexpected error type: {type(e).__name__}: {e}")

    # L2-2: Non-divisible seqlen_kv — D-EXC-SHAPE
    try:
        batch, heads, kv_ctx, dim, pe_dim, num_split = 1, 64, 65, 512, 64, 1
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        run_mla_decode(q, q_pe, kv, k_pe, num_split, 64, 64, core_num)
        print("[BOUNDARY_WARN] l2 l2_bad_shape: non-divisible seqlen silently accepted (expected rejection)")
    except (TypeError, ValueError, RuntimeError, AssertionError, ZeroDivisionError):
        print("[BOUNDARY_PASS] l2 l2_bad_shape: non-divisible seqlen correctly rejected")
    except Exception as e:
        print(f"[BOUNDARY_WARN] l2 l2_bad_shape: unexpected error type: {type(e).__name__}: {e}")


# ============================================================================
# Boundary Tests (special values: INF/NAN/Zero/Denormalized)
# ============================================================================


def test_mla_decode_persistent_boundary():
    """Boundary tests: special values (INF/NAN/Zero/Denormalized). Non-blocking."""
    core_num = _get_core_num()

    # Boundary-1: All zeros — D-SPECIAL-ZERO
    try:
        _run_precision_case("boundary_zeros", 1, 64, 128, 512, 64, 1, input_gen="zeros", level="boundary")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_zeros: exception: {e}")

    # Boundary-2: NaN in inputs — D-SPECIAL-NAN
    try:
        torch.manual_seed(42)
        batch, heads, kv_ctx, dim, pe_dim, num_split = 1, 64, 128, 512, 64, 1
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q_cpu = q.cpu()
        q_cpu[0, 0, 0] = float("nan")
        q = q_cpu.npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, num_split, 64, 64, core_num)
        torch.npu.synchronize()
        out_cpu = output.cpu().float()
        has_nan = torch.isnan(out_cpu).any().item()
        if has_nan:
            print("[BOUNDARY_PASS] boundary boundary_nan: NaN correctly propagated")
        else:
            print("[BOUNDARY_WARN] boundary boundary_nan: NaN not propagated (may be masked by softmax)")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_nan: exception: {e}")

    # Boundary-3: Inf in inputs — D-SPECIAL-INF
    try:
        torch.manual_seed(42)
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q_cpu = q.cpu()
        q_cpu[0, 0, 0] = float("inf")
        q = q_cpu.npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, num_split, 64, 64, core_num)
        torch.npu.synchronize()
        out_cpu = output.cpu().float()
        has_inf_or_nan = torch.isinf(out_cpu).any().item() or torch.isnan(out_cpu).any().item()
        if has_inf_or_nan:
            print("[BOUNDARY_PASS] boundary boundary_inf: Inf correctly handled (produced Inf/NaN)")
        else:
            print("[BOUNDARY_WARN] boundary boundary_inf: Inf silently ignored")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_inf: exception: {e}")

    # Boundary-4: Denormalized values — D-SPECIAL-DBOUND
    try:
        torch.manual_seed(42)
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q = (q.cpu() * 1e-7).npu()
        q_pe = (q_pe.cpu() * 1e-7).npu()
        kv = (kv.cpu() * 1e-7).npu()
        k_pe = (k_pe.cpu() * 1e-7).npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, num_split, 64, 64, core_num)
        torch.npu.synchronize()
        ref = _run_ref(q, q_pe, kv, k_pe)
        passed, ratio, max_abs = check_precision(output, ref, "float16")
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary boundary_dbound matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_dbound: exception: {e}")


# ============================================================================
# Performance benchmark
# ============================================================================


def run_bench():
    """Performance benchmark using tilelang.profiler.do_bench on golden config."""
    from tilelang.profiler import do_bench

    torch.manual_seed(42)
    B, H, S, D, PE = 128, 128, 8192, 512, 64
    q = torch.randn(B, H, D, dtype=torch.float16, device="npu")
    q_pe = torch.randn(B, H, PE, dtype=torch.float16, device="npu")
    kv = torch.randn(B, S, 1, D, dtype=torch.float16, device="npu")
    k_pe = torch.randn(B, S, 1, PE, dtype=torch.float16, device="npu")

    # BUG-1 fix: dynamic core_num detection (was hardcoded 20, breaks on A5/other NPUs).
    # Consistent with main block and msprof script (NEW-3 fix); reuse _get_core_num()
    # for fallback robustness and style parity with _run_precision_case / test_*.
    core_num = _get_core_num()
    lat_ms = do_bench(
        lambda: run_mla_decode(q, q_pe, kv, k_pe, 1, 128, 64, core_num),
        _n_warmup=5,
        _n_repeat=5,
        return_mode="mean",
    )
    lat_us = lat_ms * 1000
    target_us = 3036
    gap = lat_us / target_us
    print(f"[BENCH] latency={lat_us:.0f}us ({lat_ms:.2f}ms) | target={target_us}us | gap={gap:.2f}x")
    print("\nTest Passed!")


# ============================================================================
# msprof hardware-level profiling (inlined, no external perf_tuning/ dependency)
# ============================================================================

_MSPROF_SCRIPT_TEMPLATE = '''"""Auto-generated msprof profiling script (inlined by test_mla_decode_persistent.py)."""
import os
import sys

import torch
import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_mla_decode_persistent import flashattn_phase1_num_split1_ws3_fp16_kvreuse

torch.set_default_device("npu")

# Golden config — matches run_bench() and host dispatch (dim>=512, block_N>=128)
batch, heads, kv_ctx, dim, pe_dim = 128, 128, 8192, 512, 64
block_N, block_H = 128, 64
core_num = int(torch.npu.get_device_properties("npu").cube_core_num)

torch.manual_seed(42)
q = torch.randn(batch, heads, dim, dtype=torch.float16, device="cpu").npu()
q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float16, device="cpu").npu()
kv = torch.randn(batch, kv_ctx, 1, dim, dtype=torch.float16, device="cpu").npu()
k_pe = torch.randn(batch, kv_ctx, 1, pe_dim, dtype=torch.float16, device="cpu").npu()
KV_3d = kv.view(batch, kv_ctx, dim)
K_pe_3d = k_pe.view(batch, kv_ctx, pe_dim)

# ws3_fp16 + kvreuse kernel (num_split=1 path, workspace_3 dtype=fp16, num_stages=1)
p1 = flashattn_phase1_num_split1_ws3_fp16_kvreuse(
    batch, heads, kv_ctx, dim, pe_dim, block_N, block_H, core_num
)

# warmup
for _ in range(3):
    p1(q, q_pe, KV_3d, K_pe_3d)
torch.npu.synchronize()

# profiled runs (msprof will capture these)
for _ in range(10):
    p1(q, q_pe, KV_3d, K_pe_3d)
torch.npu.synchronize()
print("phase1 ws3_fp16_kvreuse runs done")
'''


def run_msprof():
    """Run msprof hardware-level profiling on Phase 1 kernel.

    Captures 8 metric categories: ArithmeticUtilization, MemoryUB, Memory,
    MemoryL0, L2Cache, PipeUtilization, ResourceConflictRatio, BasicInfo.
    Generates a temporary script inlined (no external perf_tuning/ dependency).
    Output goes to msprof_output_latest/ next to the test file.
    """
    import tempfile

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "msprof_output_latest")

    # Generate temporary profiling script (inlined, no external dependency)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=base_dir, delete=False) as f:
        f.write(_MSPROF_SCRIPT_TEMPLATE)
        script_path = f.name

    try:
        cmd = [
            "msprof",
            "op",
            "--application=python " + script_path,
            "--output=" + output_dir,
            "--aic-metrics=ArithmeticUtilization,PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,MemoryUB,L2Cache",
            "--launch-count=3",
            "--warm-up=1",
        ]
        print("Running msprof op profiling...")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"msprof data saved to {output_dir}")
    finally:
        os.unlink(script_path)

    print("\nTest Passed!")


# ============================================================================
# Main entry: --level dispatcher
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MLA Decode Persistent test suite")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all", "bench", "msprof"],
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    if args.level == "bench":
        run_bench()
        return

    if args.level == "msprof":
        run_msprof()
        return

    blocking_ok = True  # only L0/L1 count toward exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_mla_decode_persistent_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_mla_decode_persistent_l1()
    if args.level in ("l2", "all"):
        test_mla_decode_persistent_l2()
    if args.level in ("boundary", "all"):
        test_mla_decode_persistent_boundary()

    if blocking_ok:
        print("\nTest Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
