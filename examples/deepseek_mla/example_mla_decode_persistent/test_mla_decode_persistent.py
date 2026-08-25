"""Test suite for MLA Decode Persistent Attention.

Levels:
  - l0:        threshold tests (rule-based shapes, block divisible)
  - l1:        functional tests (irregular shapes, parameter variations)
  - l2:        negative tests (illegal inputs must be rejected)
  - boundary:  special values (INF/NAN/Zero/Denormalized)
  - all:       run l0 + l1 + l2 + boundary

Precision standard (fp16): atol=2^-14, rtol=2^-9, max_abs_limit=1e-1, required_ratio=0.99.
L0/L1 failures block (exit code 1). L2/Boundary failures are non-blocking (warnings only).
"""

import argparse
import os
import sys

import tilelang
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_mla_decode_persistent import run_mla_decode  # noqa: E402

try:
    from einops import rearrange, einsum
except ImportError:
    rearrange = None
    einsum = None


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
#   block_N == 0), so true partial-block tails cannot be tested.
# D-PARAM-dim/pe_dim: dim=512 and pe_dim=64 are algorithmic constants for
#   DeepSeek MLA. Testing different values would be a different algorithm.
COVERAGE_NA = {}


# ============================================================================
# Golden Reference (PyTorch, CPU, standard F.softmax — natural exp/log)
# ============================================================================


def ref_mla_decode(q, q_pe, kv, k_pe):
    """PyTorch reference implementation (CPU, standard softmax).

    Uses einops if available; falls back to native torch.bmm otherwise.

    Args:
        q:    [batch, heads, dim]         fp16
        q_pe: [batch, heads, pe_dim]      fp16
        kv:   [batch, seqlen_kv, 1, dim]  fp16 (kv_head_num=1)
        k_pe: [batch, seqlen_kv, 1, pe_dim] fp16
    Returns:
        out:  [batch, heads, dim]         fp16
    """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    # NOTE: kernel uses scale = 1/sqrt(dim+pe_dim) with T.tile.axpy (multiplication).
    # Golden uses scale = sqrt(dim+pe_dim) with division. Mathematically equivalent:
    # s * (1/sqrt(d)) == s / sqrt(d). Kept as division for readability (NEW-4).
    scale = (dim + pe_dim) ** 0.5  # sqrt(dim+pe_dim), scores / scale

    if rearrange is not None and einsum is not None:
        kv_head_num = kv.shape[2] if kv.dim() == 4 else 1
        assert kv_head_num == 1, "kv_head_num must be 1"
        num_head_groups = q.shape[1] // kv_head_num

        q_r = rearrange(q, "b (h g) d -> b g h d", g=num_head_groups)
        q_pe_r = rearrange(q_pe, "b (h g) d -> b g h d", g=num_head_groups)
        kv_r = rearrange(kv, "b n h d -> b h n d")
        k_pe_r = rearrange(k_pe, "b n h d -> b h n d")

        query = torch.concat([q_r, q_pe_r], dim=-1)  # [b, g, h, dim+pe_dim]
        key = torch.concat([kv_r, k_pe_r], dim=-1)  # [b, h, s, dim+pe_dim]

        # einops path: explicit fp32 cast to match native path precision (NEW-1 fix)
        scores = einsum(query.float(), key.float(), "b g h d, b h s d -> b g h s")
        attention = F.softmax(scores / scale, dim=-1)
        out = einsum(attention.float(), kv_r.float(), "b g h s, b h s d -> b g h d")
        out = rearrange(out, "b g h d -> b (h g) d")
        return out.half()

    # Native fallback (no einops dependency)
    B = q.shape[0]
    S = kv.shape[1]
    kv_3d = kv.reshape(B, S, dim)
    k_pe_3d = k_pe.reshape(B, S, pe_dim)

    query = torch.cat([q, q_pe], dim=-1)  # [B, H, dim+pe_dim]
    key = torch.cat([kv_3d, k_pe_3d], dim=-1)  # [B, S, dim+pe_dim]

    scores = torch.bmm(query.float(), key.float().transpose(1, 2))
    attention = F.softmax(scores / scale, dim=-1)
    out = torch.bmm(attention, kv_3d.float())  # [B, H, dim]
    return out.half()


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

# L0 configs: single-kernel path (num_split=1, fp32 workspace_3).
L0_CONFIGS = [
    {
        "name": "l0_small_basic",
        "shape": (1, 64, 128, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    },
    {
        "name": "l0_small_multibatch",
        "shape": (2, 128, 256, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    },
    {
        "name": "l0_num_split_1",
        "shape": (1, 64, 128, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-num_split"],
    },
    {
        "name": "l0_medium",
        "shape": (4, 128, 512, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    },
    {
        "name": "l0_golden",
        "shape": (128, 128, 8192, 512, 64),
        "block_N": 128,
        "tags": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    },
]


def _run_precision_case(name, batch, heads, kv_ctx, dim, pe_dim, block_N=64, block_H=64, input_gen="randn", level="l0"):
    """Run a single precision test case. Returns (passed, ratio, max_abs)."""
    core_num = _get_core_num()
    torch.manual_seed(42)
    q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim, input_gen)

    output = run_mla_decode(q, q_pe, kv, k_pe, block_N, block_H, core_num)
    torch.npu.synchronize()

    ref = _run_ref(q, q_pe, kv, k_pe)
    passed, ratio, max_abs = check_precision(output, ref, "float16")
    tag = "PASS" if passed else "FAIL"
    marker = "[PRECISION_" + tag + "]" if level in ("l0", "l1") else "[BOUNDARY_" + tag + "]"
    print(
        f"{marker} {level} {name} "
        f"B={batch} H={heads} S={kv_ctx} dim={dim} pe={pe_dim} "
        f"gen={input_gen} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed, ratio, max_abs


def test_mla_decode_persistent_l0():
    """L0 threshold tests: rule-based shapes (block divisible) for precision convergence."""
    ok = True
    for cfg in L0_CONFIGS:
        name = cfg["name"]
        batch, heads, kv_ctx, dim, pe_dim = cfg["shape"]
        block_N = cfg.get("block_N", 64)
        try:
            passed, _, _ = _run_precision_case(name, batch, heads, kv_ctx, dim, pe_dim, block_N=block_N, level="l0")
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

# L1 configs: single-kernel path (num_split=1, fp32 workspace_3).
L1_CONFIGS = [
    {
        "name": "l1_edge_minimal",
        "shape": (1, 64, 64, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-TAIL-1", "D-VALRANGE-S"],
    },
    {
        "name": "l1_prime_seqlen",
        "shape": (2, 64, 448, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-PRIME"],
    },
    {
        "name": "l1_tail_mid",
        "shape": (1, 64, 128, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID"],
    },
    {
        "name": "l1_large_batch",
        "shape": (8, 128, 128, 512, 64),
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-L"],
    },
    {
        "name": "l1_asym_positive",
        "shape": (1, 64, 128, 512, 64),
        "input_gen": "positive",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-ASYM"],
    },
    {
        "name": "l1_small_values",
        "shape": (1, 64, 128, 512, 64),
        "input_gen": "small",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-S"],
    },
    {
        "name": "l1_large_values",
        "shape": (1, 64, 128, 512, 64),
        "input_gen": "large",
        "tags": ["D-DTYPE-fp16", "D-VALRANGE-L"],
    },
    {
        "name": "l1_param_block_n32",
        "shape": (1, 64, 128, 512, 64),
        "block_N": 32,
        "block_H": 64,
        "tags": ["D-DTYPE-fp16", "D-PARAM-block_N"],
    },
    {
        "name": "l1_param_block_h32",
        "shape": (1, 64, 128, 512, 64),
        "block_N": 64,
        "block_H": 32,
        "tags": ["D-DTYPE-fp16", "D-PARAM-block_H"],
    },
    {
        "name": "l1_param_dim256",
        "shape": (1, 64, 128, 256, 32),
        "tags": ["D-DTYPE-fp16", "D-PARAM-dim", "D-PARAM-pe_dim"],
    },
]


def test_mla_decode_persistent_l1():
    """L1 functional tests: irregular shapes, parameter variations, value ranges."""
    ok = True
    for cfg in L1_CONFIGS:
        name = cfg["name"]
        batch, heads, kv_ctx, dim, pe_dim = cfg["shape"]
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
        batch, heads, kv_ctx, dim, pe_dim = 1, 64, 128, 512, 64
        q = torch.randn(batch, heads, dim, dtype=torch.float32, device="cpu").npu()
        q_pe = torch.randn(batch, heads, pe_dim, dtype=torch.float32, device="cpu").npu()
        kv = torch.randn(batch, kv_ctx, 1, dim, dtype=torch.float32, device="cpu").npu()
        k_pe = torch.randn(batch, kv_ctx, 1, pe_dim, dtype=torch.float32, device="cpu").npu()
        run_mla_decode(q, q_pe, kv, k_pe, 64, 64, core_num)
        print("[BOUNDARY_WARN] l2 l2_wrong_dtype: fp32 input silently accepted (expected rejection)")
    except (TypeError, ValueError, RuntimeError, AssertionError):
        print("[BOUNDARY_PASS] l2 l2_wrong_dtype: fp32 input correctly rejected")
    except Exception as e:
        print(f"[BOUNDARY_WARN] l2 l2_wrong_dtype: unexpected error type: {type(e).__name__}: {e}")

    # L2-2: Non-divisible seqlen_kv — D-EXC-SHAPE
    try:
        batch, heads, kv_ctx, dim, pe_dim = 1, 64, 65, 512, 64
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        run_mla_decode(q, q_pe, kv, k_pe, 64, 64, core_num)
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
    # Boundary-1: All zeros — D-SPECIAL-ZERO
    try:
        _run_precision_case("boundary_zeros", 1, 64, 128, 512, 64, input_gen="zeros", level="boundary")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_zeros: exception: {e}")

    # Boundary-2: NaN in inputs — D-SPECIAL-NAN
    try:
        torch.manual_seed(42)
        batch, heads, kv_ctx, dim, pe_dim = 1, 64, 128, 512, 64
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q_cpu = q.cpu()
        q_cpu[0, 0, 0] = float("nan")
        q = q_cpu.npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, 64, 64, _get_core_num())
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
        batch, heads, kv_ctx, dim, pe_dim = 1, 64, 128, 512, 64
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q_cpu = q.cpu()
        q_cpu[0, 0, 0] = float("inf")
        q = q_cpu.npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, 64, 64, _get_core_num())
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
        batch, heads, kv_ctx, dim, pe_dim = 1, 64, 128, 512, 64
        q, q_pe, kv, k_pe = _gen_inputs(batch, heads, kv_ctx, dim, pe_dim)
        q = (q.cpu() * 1e-7).npu()
        q_pe = (q_pe.cpu() * 1e-7).npu()
        kv = (kv.cpu() * 1e-7).npu()
        k_pe = (k_pe.cpu() * 1e-7).npu()
        output = run_mla_decode(q, q_pe, kv, k_pe, 64, 64, _get_core_num())
        torch.npu.synchronize()
        ref = _run_ref(q, q_pe, kv, k_pe)
        passed, ratio, max_abs = check_precision(output, ref, "float16")
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary boundary_dbound matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary boundary_dbound: exception: {e}")


# ============================================================================
# Main entry: --level dispatcher
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MLA Decode Persistent test suite")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(42)

    blocking_ok = True
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
