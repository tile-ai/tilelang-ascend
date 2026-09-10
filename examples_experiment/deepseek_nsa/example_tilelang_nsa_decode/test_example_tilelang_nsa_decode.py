"""NSA Decode layered precision test suite: L0/L1/L2/Boundary.

This file is the single precision-test entry for `example_tilelang_nsa_decode.py`.
It embeds:
  - ref_nsa_decode:     PyTorch CPU golden reference (moved from operator file).
  - check_precision:    mixed-tolerance dual-gate check (precision-standard.md §4.1).
  - test_nsa_l0/l1/l2/boundary: layered test cases (L0/L1 blocking, L2/Boundary non-blocking).
  - main(--level):      unified dispatch + exit code.

Run:
  python test_example_tilelang_nsa_decode.py --level all     # full precision suite
  python test_example_tilelang_nsa_decode.py --level l0      # L0 threshold only
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel + helpers from sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_tilelang_nsa_decode import (  # noqa: E402
    nsa_decode,
    run_kernel,
)


# =============================================================================
# Golden reference (CPU, from GPU source reference.py naive_nsa_simple_inference)
# =============================================================================
def ref_nsa_decode(q, k, v, block_indices, block_counts, block_size=32):
    """PyTorch CPU reference. scale = D^-0.5 (no log2(e)). fp32 -> fp16 output."""
    scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    G = HQ // H
    BS = block_size
    S = block_indices.shape[-1]

    # GQA expand: H -> HQ
    k, v = k.repeat_interleave(G, dim=2), v.repeat_interleave(G, dim=2)
    block_indices = block_indices.repeat_interleave(G, dim=2)
    block_counts = block_counts.repeat_interleave(G, dim=2)

    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, HQ).to(q.device)

    q, k, v = map(lambda x: x.float(), (q, k, v))
    o = torch.zeros_like(q)
    B, seq_q = q.shape[:2]  # seq_q=1 (decode); avoid shadowing module-level T

    for i in range(B):
        q_b, k_b, v_b, i_b, s_b = q[i], k[i], v[i], block_indices[i], block_counts[i]
        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        i_b = i_b.view(seq_q, block_indices.shape[2], -1).transpose(1, 2)

        q_i = q_b[0] * scale
        i_i = i_b[0]
        s_i = s_b[0]

        # Gather K/V (clamp invalid indices, mask zeros out contribution)
        i_i_clamped = i_i.clamp(0, k_b.shape[0] - 1)
        h_idx = torch.arange(HQ).unsqueeze(0).expand(S * BS, HQ)
        k_i = k_b[i_i_clamped, h_idx]
        v_i = v_b[i_i_clamped, h_idx]

        attn = torch.einsum("h d, n h d -> n h", q_i, k_i)
        attn = attn.masked_fill((c >= s_i), float("-inf"))
        attn = torch.softmax(attn, dim=0)
        result = torch.einsum("n h, n h v -> h v", attn, v_i)

        # block_counts=0 guard: NaN -> 0
        valid = (s_i > 0).unsqueeze(-1)
        o[i, 0] = torch.where(valid, result, torch.zeros_like(result))

    return o.to(dtype)


# =============================================================================
# Precision standard (mixed tolerance dual-gate, precision-standard.md §4.1)
# =============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Int: exact match (0 error).
    Matches .agents/skills/tilelang-op-test-design/references/precision-standard.md §4.1.
    """
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate: return (passed, matched_ratio, max_abs_error).

    Pass condition: matched_ratio >= required AND max_abs_error <= max_abs_error_limit.
    inf/nan positions: structural compare (not counted in numeric tolerance).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
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
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# =============================================================================
# Test input generation
# =============================================================================
def gen_test_inputs(B, SEQ_LEN, H, HQ, D, S, block_size, dtype, seed=0):
    """Generate Q, K, V, block_indices, block_counts on CPU.

    block_indices: sentinel SEQ_LEN = invalid (padding).
    block_counts: number of valid blocks per (b, h).
    """
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn((B, 1, HQ, D), dtype=dtype, generator=g)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)

    block_indices = torch.full((B, 1, H, S), SEQ_LEN, dtype=torch.long)
    for b in range(B):
        for h in range(H):
            num_blocks = max(1, SEQ_LEN // block_size)
            picks = torch.randperm(num_blocks, generator=g)[:S]
            block_indices[b, 0, h, : len(picks)] = picks
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.full((B, 1, H), S, dtype=torch.long)

    return Q, K, V, block_indices, block_counts


# =============================================================================
# L0 threshold tests — blocking
# =============================================================================
def _run_kernel_and_check(level, name, B, SEQ_LEN, H, HQ, D, S, BS, dtype, seed, **kwargs):
    """Run kernel + golden + check_precision, print [PRECISION_PASS/FAIL]. Returns ok."""
    try:
        g = torch.Generator().manual_seed(seed)
        Q = torch.randn((B, 1, HQ, D), dtype=dtype, generator=g)
        K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)
        V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)

        block_idx = kwargs.get("block_idx", 0)
        block_counts_val = kwargs.get("block_counts", S)
        block_indices_list = kwargs.get("block_indices_list")
        gen_fn = kwargs.get("gen_fn")

        if gen_fn is not None:
            Q = gen_fn((B, 1, HQ, D))
            K = gen_fn((B, SEQ_LEN, H, D))
            V = gen_fn((B, SEQ_LEN, H, D))

        block_indices = torch.full((B, 1, H, S), SEQ_LEN, dtype=torch.long)
        if block_indices_list is not None:
            for i, idx in enumerate(block_indices_list[:block_counts_val]):
                block_indices[:, :, :, i] = idx
        else:
            block_indices[:, :, :, :block_counts_val] = block_idx
        block_counts = torch.full((B, 1, H), float(block_counts_val), dtype=torch.float32)

        ref = ref_nsa_decode(Q, K, V, block_indices, block_counts, BS)
        out = run_kernel(Q, K, V, block_indices, block_counts, BS, SEQ_LEN)

        dtype_str = str(dtype).split(".")[-1]
        passed, ratio, max_abs = check_precision(out, ref, dtype_str)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {level} {name} B={B} seq={SEQ_LEN} S={S} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {level} {name}: {e}")
        return False


def test_nsa_l0():
    """L0 threshold tests: regular shapes, precision convergence."""
    dtype = torch.float16
    ok = True
    for name, B, SEQ_LEN, H, HQ, D, S, BS, seed in [
        ("l0_smoke", 1, 64, 1, 16, 16, 1, 32, 0),
        ("l0_standard", 2, 64, 1, 16, 16, 1, 32, 0),
        ("l0_perf_target", 2, 64, 1, 16, 16, 1, 32, 0),
    ]:
        ok &= _run_kernel_and_check("l0", name, B, SEQ_LEN, H, HQ, D, S, BS, dtype, seed)
    return ok


# =============================================================================
# L1 functional tests — blocking
# =============================================================================
def _gen_uniform(low, high):
    def fn(shape):
        return (torch.rand(shape, dtype=torch.float32) * (high - low) + low).to(torch.float16)

    return fn


def test_nsa_l1():
    """L1 functional tests: shape/value/param coverage."""
    dtype = torch.float16
    H, HQ, D, BS = 1, 16, 16, 32
    ok = True

    # Shape variants
    ok &= _run_kernel_and_check("l1", "aligned", 4, 128, H, HQ, D, 1, BS, dtype, 0, block_idx=1)
    ok &= _run_kernel_and_check("l1", "edge", 1, 32, H, HQ, D, 1, BS, dtype, 0, block_idx=0)
    ok &= _run_kernel_and_check("l1", "tail1", 2, 33, H, HQ, D, 1, BS, dtype, 0, block_idx=0)
    ok &= _run_kernel_and_check("l1", "tailmid", 2, 48, H, HQ, D, 1, BS, dtype, 0, block_idx=0)
    ok &= _run_kernel_and_check("l1", "prime", 2, 97, H, HQ, D, 1, BS, dtype, 0, block_idx=0)

    # Value range variants
    ok &= _run_kernel_and_check("l1", "valrange_s", 2, 64, H, HQ, D, 1, BS, dtype, 0, gen_fn=_gen_uniform(-1, 1))
    ok &= _run_kernel_and_check("l1", "valrange_m", 2, 64, H, HQ, D, 1, BS, dtype, 0, gen_fn=_gen_uniform(-10, 10))
    ok &= _run_kernel_and_check("l1", "valrange_l", 2, 64, H, HQ, D, 1, BS, dtype, 0, gen_fn=_gen_uniform(-50, 50))
    ok &= _run_kernel_and_check("l1", "valrange_asym", 2, 64, H, HQ, D, 1, BS, dtype, 0, gen_fn=_gen_uniform(-5, 10))

    # block_counts variants
    ok &= _run_kernel_and_check("l1", "bc0", 2, 64, H, HQ, D, 1, BS, dtype, 0, block_idx=64, block_counts=0)
    ok &= _run_kernel_and_check("l1", "bc_pad", 1, 64, H, HQ, D, 1, BS, dtype, 0, block_idx=64, block_counts=0)
    ok &= _run_kernel_and_check("l1", "tail_bc0", 2, 48, H, HQ, D, 1, BS, dtype, 0, block_idx=48, block_counts=0)

    # S>1 variants
    ok &= _run_kernel_and_check("l1", "s2_full", 2, 64, H, HQ, D, 2, BS, dtype, 0, block_counts=2, block_indices_list=[0, 1])
    ok &= _run_kernel_and_check("l1", "s2_partial", 2, 64, H, HQ, D, 2, BS, dtype, 0, block_counts=1, block_indices_list=[0, 1])
    ok &= _run_kernel_and_check("l1", "s4_full", 2, 128, H, HQ, D, 4, BS, dtype, 0, block_counts=4, block_indices_list=[0, 1, 2, 3])

    # Model typical config (DeepSeek-V3 NSA paper: D=128, S=16, BS=64)
    ok &= _run_kernel_and_check(
        "l1", "typical_d128_s16", 1, 1024, H, HQ, 128, 16, 64, dtype, 0, block_counts=16, block_indices_list=list(range(16))
    )
    ok &= _run_kernel_and_check(
        "l1", "typical_d128_s8", 1, 512, H, HQ, 128, 8, 64, dtype, 0, block_counts=8, block_indices_list=list(range(8))
    )

    return ok


# =============================================================================
# L2 exception tests — blocking (returns bool, merged into exit code)
# =============================================================================
def _run_exception(name, fn, expected_exc=(AssertionError, ValueError, TypeError)):
    """L2: fn() feeds invalid input; expects rejection of specific exception type.

    Returns True if rejected with expected exception, False otherwise.
    Non-expected exceptions are treated as failures (not silently accepted).
    """
    try:
        fn()
    except expected_exc as e:
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__})")
        return True
    except Exception as e:
        print(f"[BOUNDARY_FAIL] l2 {name}: wrong exception type ({type(e).__name__})")
        return False
    print(f"[BOUNDARY_FAIL] l2 {name}: invalid input not rejected")
    return False


def test_nsa_l2():
    """L2 exception tests: unsupported dtype / invalid shape. Blocking (returns bool)."""

    # float32 input (kernel hardcodes float16) — expect ValueError or TypeError
    def _test_fp32_input():
        B, SEQ_LEN, H, HQ, D, S, BS = 1, 64, 1, 16, 16, 1, 32
        kernel = nsa_decode(B, SEQ_LEN, HQ, H, D, S, BS)
        q_f32 = torch.randn(1, 1, 16, 16, dtype=torch.float32, device="cpu").npu()
        k_f32 = torch.randn(1, 64, 1, 16, dtype=torch.float32, device="cpu").npu()
        v_f32 = torch.randn(1, 64, 1, 16, dtype=torch.float32, device="cpu").npu()
        ri = torch.zeros(1, 1, 1, S * BS, dtype=torch.int32, device="cpu").npu()
        bc = torch.full((1, 1, 1), float(S), dtype=torch.float32, device="cpu").npu()
        # workspace auto-allocated by framework via workspace_idx=[6,7,8,9,10]
        kernel(q_f32, k_f32, v_f32, ri, bc)
        torch.npu.synchronize()

    ok = True
    ok &= _run_exception("fp32_input_unsupported", _test_fp32_input, expected_exc=(ValueError, TypeError))

    # HQ=8 (G=8 < 16, violates L0C fractal) — expect AssertionError
    def _test_hq8():
        nsa_decode(1, 64, 8, 1, 16, 1, 32)

    ok &= _run_exception("hq8_rejected", _test_hq8, expected_exc=AssertionError)

    # HQ=17 (G=17 odd, violates vid split) — expect AssertionError
    def _test_hq17():
        nsa_decode(1, 64, 17, 1, 16, 1, 32)

    ok &= _run_exception("hq17_rejected", _test_hq17, expected_exc=AssertionError)
    return ok


# =============================================================================
# Boundary edge/special-value tests — non-blocking
# =============================================================================
def _run_boundary(name, dtype, fn):
    """Boundary: fn() returns (out, ref). Compare by precision standard; fail=WARN."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, str(dtype).split(".")[-1])
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name}: {e}")


def _boundary_run(dtype, seed, value_fn):
    """Run boundary case: create special-value inputs, run kernel + golden, return (out, ref)."""
    B, SEQ_LEN, H, HQ, D, S, BS = 1, 64, 1, 16, 16, 1, 32
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn((B, 1, HQ, D), dtype=dtype, generator=g)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)
    Q, K, V = value_fn(Q, K, V)

    block_indices = torch.zeros((B, 1, H, S), dtype=torch.long)
    block_counts = torch.full((B, 1, H), S, dtype=torch.long)

    ref = ref_nsa_decode(Q, K, V, block_indices, block_counts, BS)
    out = run_kernel(Q, K, V, block_indices, block_counts, BS, SEQ_LEN)
    return out.cpu(), ref


def test_nsa_boundary():
    """Boundary tests: INF/NAN/extreme/all-zero. Non-blocking."""
    dtype = torch.float16
    _run_boundary(
        "all_zero", dtype, lambda: _boundary_run(dtype, 0, lambda q, k, v: (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)))
    )
    _run_boundary("with_inf", dtype, lambda: _boundary_run(dtype, 1, lambda q, k, v: (q, k.clone().fill_(float("inf")), v)))
    _run_boundary("with_nan", dtype, lambda: _boundary_run(dtype, 2, lambda q, k, v: (q, k.clone().fill_(float("nan")), v)))
    _run_boundary("dtype_bound", dtype, lambda: _boundary_run(dtype, 3, lambda q, k, v: (q, k.clone().fill_(65504.0), v)))

    # block_counts=0 -> output all zeros (not NaN)
    def _bc0_run():
        B, SEQ_LEN, H, HQ, D, S, BS = 1, 64, 1, 16, 16, 1, 32
        Q = torch.randn((B, 1, HQ, D), dtype=dtype)
        K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype)
        V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype)
        bi = torch.full((B, 1, H, S), 64, dtype=torch.long)
        bc = torch.full((B, 1, H), 0, dtype=torch.long)
        ref = ref_nsa_decode(Q, K, V, bi, bc, BS)
        out = run_kernel(Q, K, V, bi, bc, BS, SEQ_LEN)
        return out.cpu(), ref

    _run_boundary("bc0_zero", dtype, _bc0_run)


# =============================================================================
# Main: --level dispatch + exit code
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSA Decode layered precision test suite (Ascend)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()

    blocking_ok = True
    if args.level in ("l0", "all"):
        blocking_ok &= test_nsa_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_nsa_l1()
    if args.level in ("l2", "all"):
        blocking_ok &= test_nsa_l2()
    if args.level in ("boundary", "all"):
        test_nsa_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
