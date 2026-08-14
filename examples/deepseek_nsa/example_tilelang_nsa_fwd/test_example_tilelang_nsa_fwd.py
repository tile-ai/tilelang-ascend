"""NSA Forward layered test suite: L0/L1/L2/Boundary + msprof op.

This file is the single test entry for `example_tilelang_nsa_fwd.py`. It embeds:
  - prepare_inputs:   host-side pre-gather K_selected/V_selected/CausalMask (CPU).
  - golden_nsa_fwd:   PyTorch CPU reference (independent of prepare_inputs).
  - check_precision:  mixed-tolerance dual-gate check (precision-standard.md §4.1).
  - test_nsa_l0/l1/l2/boundary: layered test cases (L0/L1 block, L2/Boundary non-block).
  - run_msprof:       msprof op hardware-level profiling (optimization analysis).
  - main(--level):    unified dispatch + exit code.

Run:
  python test_example_tilelang_nsa_fwd.py --level all     # full precision suite
  python test_example_tilelang_nsa_fwd.py --level msprof  # msprof op profiling
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_tilelang_nsa_fwd import native_sparse_attention  # noqa: E402


# =============================================================================
# Host-side preprocessing (CPU, no aclnn dependency)
# =============================================================================
def prepare_inputs(Q, K, V, block_indices, block_counts, block_size, S, is_causal=True, bs_pad=None):
    """Pre-gather K_selected / V_selected / CausalMask on CPU.

    Returns:
        K_selected/V_selected: [B, T, H, S*bs_pad, D] fp16 (real tokens + padding zeros).
        CausalMask:            [B, T, H, S*bs_pad]    fp32 (0.0 visible / -2^30 masked).

    Invalid blocks (s >= block_counts) and padding tokens (bs_pad - BS) leave K/V as
    zeros and mask as -2^30, so the kernel's softmax effectively ignores them.
    """
    B, T, HQ, D = Q.shape
    H = K.shape[2]
    BS = block_size
    assert T >= BS, f"seq_len ({T}) must be >= block_size ({BS}) to allow at least one valid block"
    if bs_pad is None:
        bs_pad = max(BS, 64)
    KV_LEN = S * bs_pad
    NEG_INF = -(2.0**30)

    K_selected = torch.zeros(B, T, H, KV_LEN, D, dtype=Q.dtype)
    V_selected = torch.zeros(B, T, H, KV_LEN, D, dtype=Q.dtype)
    mask = torch.full((B, T, H, KV_LEN), NEG_INF, dtype=torch.float32)

    for b in range(B):
        for t in range(T):
            for h in range(H):
                bc_val = block_counts[b, t, h].item() if isinstance(block_counts, torch.Tensor) else block_counts
                for s in range(S):
                    if s >= bc_val:
                        continue
                    bi = block_indices[b, t, h, s].item()
                    for j in range(BS):
                        pos = bi * BS + j
                        if 0 <= pos < T:
                            K_selected[b, t, h, s * BS + j, :] = K[b, pos, h, :]
                            V_selected[b, t, h, s * BS + j, :] = V[b, pos, h, :]
                            if (not is_causal) or pos <= t:
                                mask[b, t, h, s * BS + j] = 0.0

    return K_selected, V_selected, mask


# =============================================================================
# Golden reference (CPU, based on naive_nsa; independent of prepare_inputs)
# =============================================================================
def golden_nsa_fwd(Q, K, V, block_indices, block_counts, block_size=32, scale=0.1, is_causal=True):
    """PyTorch CPU reference based on naive_nsa (g_slc=g_swa=ones, window_size=0).

    Uses original Q/K/V/block_indices directly (no pre-gather). Single-pass softmax
    matches the NPU kernel. scale is NOT multiplied by log2(e) — NPU uses T.exp,
    GPU uses T.exp2 * log2(e), which are mathematically equivalent.
    """
    B, T, HQ, D = Q.shape
    H = K.shape[2]
    G = HQ // H
    S = block_indices.shape[-1]
    BS = block_size

    dtype = Q.dtype
    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    Output = torch.zeros(B, T, HQ, D, dtype=torch.float32)

    for b in range(B):
        for t in range(T):
            for h in range(H):
                # Collect valid positions for this (b, t, h).
                positions = []
                bc_val = block_counts[b, t, h].item() if isinstance(block_counts, torch.Tensor) else block_counts
                for s in range(S):
                    if s >= bc_val:
                        continue
                    bi = block_indices[b, t, h, s].item()
                    for j in range(BS):
                        pos = bi * BS + j
                        if is_causal and pos > t:
                            continue
                        if pos < 0 or pos >= T:
                            continue
                        positions.append(pos)

                if len(positions) == 0:
                    continue

                k_gathered = K_f[b, positions, h, :]
                v_gathered = V_f[b, positions, h, :]

                # Per query head in the group: standard attention.
                for g in range(G):
                    hq = h * G + g
                    q = Q_f[b, t, hq, :]
                    scores = torch.matmul(q, k_gathered.T) * scale
                    attn = torch.softmax(scores, dim=0)
                    Output[b, t, hq, :] = torch.matmul(attn, v_gathered)

    return Output.to(dtype)


# =============================================================================
# Precision standard (mixed tolerance dual-gate, precision-standard.md §4.1)
# =============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Int: exact match (0 error).
    Matches .agents/skills/tilelang-op-test-design/references/precision-standard.md §4.1.
    """
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),  # atol 9.77e-4, rtol 1.56e-2
        "float32": (2**-16, 2**-10, 1e-2, 0.99),  # atol 1.53e-5, rtol 9.77e-4
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),  # same as float32
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),  # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),  # atol 0.125,  rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # integer exact match
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Mixed-tolerance dual-gate: return (passed, matched_ratio, max_abs_error).

    Pass condition: matched_ratio >= required AND max_abs_error <= max_abs_error_limit.
    inf/nan positions: structural compare (not counted in numeric tolerance).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    # Integer: exact match.
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    # inf/nan structural compare (precision-standard.md §3.1).
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # golden finite positions: full numeric compare
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# =============================================================================
# Test input generation (matches GPU test, CPU)
# =============================================================================
def gen_test_inputs(B, SEQ_LEN, H, HQ, D, S, block_size, dtype, seed=0):
    """Generate Q, K, V, block_indices, block_counts on CPU (matches GPU test)."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, generator=g)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, generator=g)

    # block_indices: sentinel SEQ_LEN = invalid; sorted for deterministic gather order.
    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long)
    block_counts = torch.zeros((B, SEQ_LEN, H), dtype=torch.long)
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                i_i = torch.randperm(max(1, t // block_size), generator=g)[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
                block_counts[b, t, h] = (block_indices[b, t, h] != SEQ_LEN).sum().item()
    block_indices = block_indices.sort(-1)[0]

    return Q, K, V, block_indices, block_counts


def _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D):
    """Reshape [B,T,H,...] → [B*T*H, ...] and expand mask to G groups."""
    K_sel_3d = K_sel.reshape(B * T * H, KV_LEN, D)
    V_sel_3d = V_sel.reshape(B * T * H, KV_LEN, D)
    # mask is per (b,t,h); broadcast to all G query heads in the group (mask is shared).
    mask_3d = mask.reshape(B * T * H, KV_LEN).unsqueeze(1).expand(-1, G, -1).contiguous()
    return K_sel_3d, V_sel_3d, mask_3d


# =============================================================================
# L0 threshold tests (DESIGN.md section 9.2) — blocking
# =============================================================================
def _run_kernel_and_check(level, name, B, T, H, HQ, D, S, BS, BS_pad, scale, dtype, seed, vrange, is_causal=True):
    """Run kernel + golden + check_precision, print [PRECISION_PASS/FAIL]. Returns ok."""
    try:
        Q, K, V, bi, bc = gen_test_inputs(B, T, H, HQ, D, S, BS, dtype, seed)
        # Optional value-range rescaling (vrange=(lo, hi)): randn~N(0,1) linearly rescaled to [lo, hi].
        # NOTE: this is NOT a uniform distribution — the shape stays Gaussian, only the range is mapped.
        if vrange is not None:
            lo, hi = vrange
            scale_factor = (hi - lo) / 2.0
            shift = (hi + lo) / 2.0
            Q = (Q.float() * scale_factor + shift).to(dtype)
            K = (K.float() * scale_factor + shift).to(dtype)
            V = (V.float() * scale_factor + shift).to(dtype)
        K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, is_causal=is_causal, bs_pad=BS_pad)
        KV_LEN = S * BS_pad
        G = HQ // H
        K_sel_3d, V_sel_3d, mask_3d = _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D)
        kernel = native_sparse_attention(
            batch=B,
            seq_len=T,
            head_kv=H,
            heads=HQ,
            dim=D,
            selected_blocks=S,
            bs_pad=BS_pad,
            scale=scale,
        )
        out = kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
        torch.npu.synchronize()
        ref = golden_nsa_fwd(Q, K, V, bi, bc, block_size=BS, scale=scale, is_causal=is_causal)
        # Convert torch.dtype (e.g. torch.float16) to precision-standard dtype string.
        dtype_str = str(dtype).split(".")[-1]
        passed, ratio, max_abs = check_precision(out, ref, dtype_str)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] {level} {name} B={B} T={T} S={S} scale={scale} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed
    except Exception as e:
        import traceback

        print(f"[PRECISION_FAIL] {level} {name}: {e}")
        traceback.print_exc()
        return False


def test_nsa_l0():
    """L0 threshold tests: regular shapes (block-divisible), precision convergence."""
    BS_pad = 64
    dtype = torch.float16
    # (name, B, T, H, HQ, D, S, BS, scale, seed)
    test_configs = [
        ("l0_nsa_basic", 2, 64, 1, 16, 32, 1, 32, 0.1, 0),
        ("l0_nsa_scale_02", 2, 64, 1, 16, 32, 1, 32, 0.2, 1),  # scale=0.2 covers different scale
    ]
    ok = True
    for name, B, T, H, HQ, D, S, BS, scale, seed in test_configs:
        ok &= _run_kernel_and_check("l0", name, B, T, H, HQ, D, S, BS, BS_pad, scale, dtype, seed, None)
    return ok


# =============================================================================
# L1 functional tests — blocking
# =============================================================================
# (name, B, T, H, HQ, D, S, BS, scale, seed, vrange)
L1_CASES = [
    ("l1_aligned", 2, 64, 1, 16, 32, 1, 32, 0.1, 10, (-1, 1)),  # aligned T%BS=0
    ("l1_tail1", 2, 65, 1, 16, 32, 1, 32, 0.1, 11, (-1, 1)),  # T%BS=1
    ("l1_tail_mid", 2, 80, 1, 16, 32, 1, 32, 0.1, 12, (-1, 1)),  # T%BS=16
    ("l1_prime", 2, 67, 1, 16, 32, 1, 32, 0.1, 13, (-1, 1)),  # T prime
    ("l1_edge", 1, 32, 1, 16, 32, 1, 32, 0.1, 14, (-1, 1)),  # T=BS minimal
    ("l1_s2", 2, 64, 1, 16, 32, 2, 32, 0.1, 15, (-1, 1)),  # S=2
    ("l1_scale_02", 2, 64, 1, 16, 32, 1, 32, 0.2, 16, (-1, 1)),  # scale=0.2
    ("l1_valrange_m", 2, 64, 1, 16, 32, 1, 32, 0.1, 17, (-10, 10)),  # medium range
    ("l1_valrange_l", 2, 64, 1, 16, 32, 1, 32, 0.1, 18, (-50, 50)),  # large range
    ("l1_valrange_aym", 2, 64, 1, 16, 32, 1, 32, 0.1, 19, (-5, 10)),  # asymmetric range
]


def test_nsa_l1():
    """L1 functional tests: shape/value/param coverage, incl. irregular shapes."""
    BS_pad = 64
    dtype = torch.float16
    ok = True
    for name, B, T, H, HQ, D, S, BS, scale, seed, vrange in L1_CASES:
        ok &= _run_kernel_and_check("l1", name, B, T, H, HQ, D, S, BS, BS_pad, scale, dtype, seed, vrange)
    # Extra param-coverage cases (non-default values).
    ok &= _run_kernel_and_check("l1", "l1_block_size_64", 2, 64, 1, 16, 32, 1, 64, 64, 0.1, dtype, 20, (-1, 1))  # BS=64, BS_pad=64
    ok &= _run_kernel_and_check(
        "l1", "l1_no_causal", 2, 64, 1, 16, 32, 1, 32, 64, 0.1, dtype, 21, (-1, 1), is_causal=False
    )  # is_causal=False
    ok &= _run_kernel_and_check("l1", "l1_bs_pad_128", 2, 64, 1, 16, 32, 1, 32, 128, 0.1, dtype, 22, (-1, 1))  # bs_pad=128
    ok &= _run_kernel_and_check("l1", "l1_dim_64", 2, 64, 1, 16, 64, 1, 32, 64, 0.1, dtype, 23, (-1, 1))  # D=64
    ok &= _run_kernel_and_check("l1", "l1_groups_32", 2, 64, 1, 32, 32, 1, 32, 64, 0.1, dtype, 24, (-1, 1))  # HQ=32, G=32
    return ok


# =============================================================================
# L2 exception tests — non-blocking (invalid inputs must be rejected)
# =============================================================================
def _run_exception(name, fn):
    """L2: fn() feeds invalid input; expects rejection. Raises=PASS, silently accepts=WARN."""
    try:
        fn()
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(e).__name__})")
        return
    print(f"[BOUNDARY_WARN] l2 {name}: invalid input not rejected (silently accepted)")


def test_nsa_l2():
    """L2 exception tests: unsupported dtype / invalid shape / odd GQA. Non-blocking."""
    BS_pad = 64

    # D-EXC-DTYPE: float32 input (kernel hardcodes float16).
    def _test_fp32_input():
        B, T, H, HQ, D, S, BS = 2, 64, 1, 16, 32, 1, 32
        Q, K, V, bi, bc = gen_test_inputs(B, T, H, HQ, D, S, BS, torch.float32, 0)
        K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, bs_pad=BS_pad)
        KV_LEN = S * BS_pad
        G = HQ // H
        K_sel_3d, V_sel_3d, mask_3d = _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D)
        kernel = native_sparse_attention(
            batch=B,
            seq_len=T,
            head_kv=H,
            heads=HQ,
            dim=D,
            selected_blocks=S,
            bs_pad=BS_pad,
            scale=0.1,
        )
        kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
        torch.npu.synchronize()

    _run_exception("fp32_input_unsupported", _test_fp32_input)

    # D-EXC-SHAPE: T < block_size (no valid block to select).
    def _test_short_seq():
        B, T, H, HQ, D, S, BS = 2, 16, 1, 16, 32, 1, 32  # T=16 < BS=32
        Q = torch.randn(B, T, HQ, D, dtype=torch.float16)
        K = torch.randn(B, T, H, D, dtype=torch.float16)
        V = torch.randn(B, T, H, D, dtype=torch.float16)
        bi = torch.full((B, T, H, S), T, dtype=torch.long)
        bc = torch.zeros((B, T, H), dtype=torch.long)
        # prepare_inputs asserts T >= BS, rejecting this invalid input.
        prepare_inputs(Q, K, V, bi, bc, BS, S, bs_pad=BS_pad)

    _run_exception("short_seq_no_block", _test_short_seq)

    # D-EXC-GQA: G=odd (vid split requires even G; odd G is rejected by the
    # kernel's assert at JIT compile time, see kernel line 59).
    def _test_odd_gqa():
        B, T, H, HQ, D, S, BS = 2, 64, 1, 17, 32, 1, 32  # HQ=17, H=1, G=17 (odd)
        Q = torch.randn(B, T, HQ, D, dtype=torch.float16)
        K = torch.randn(B, T, H, D, dtype=torch.float16)
        V = torch.randn(B, T, H, D, dtype=torch.float16)
        bi = torch.full((B, T, H, S), 0, dtype=torch.long)  # at least 1 valid block
        bc = torch.ones((B, T, H), dtype=torch.long)
        K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, bs_pad=BS_pad)
        KV_LEN = S * BS_pad
        G = HQ // H  # = 17
        K_sel_3d, V_sel_3d, mask_3d = _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D)
        # The following call must raise AssertionError at JIT trace time
        # (when native_sparse_attention is invoked, not at kernel() call).
        kernel = native_sparse_attention(
            batch=B,
            seq_len=T,
            head_kv=H,
            heads=HQ,
            dim=D,
            selected_blocks=S,
            bs_pad=BS_pad,
            scale=0.1,
        )
        kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
        torch.npu.synchronize()

    _run_exception("odd_gqa_rejected", _test_odd_gqa)


# =============================================================================
# Boundary edge/special-value tests — non-blocking (legal extremes)
# =============================================================================
def _run_boundary(name, dtype, fn):
    """Boundary: fn() returns (out, ref). Compare by precision standard; fail=WARN. Non-blocking."""
    try:
        out, ref = fn()
        passed, ratio, max_abs = check_precision(out, ref, dtype)
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {e}")


def _boundary_run(dtype, seed, value_fn):
    """Run boundary case: create special-value inputs, run kernel + golden, return (out, ref)."""
    B, T, H, HQ, D, S, BS, BS_pad, scale = 2, 64, 1, 16, 32, 1, 32, 64, 0.1
    Q, K, V, bi, bc = gen_test_inputs(B, T, H, HQ, D, S, BS, dtype, seed)
    Q, K, V = value_fn(Q, K, V)
    K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, is_causal=True, bs_pad=BS_pad)
    KV_LEN = S * BS_pad
    G = HQ // H
    K_sel_3d, V_sel_3d, mask_3d = _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D)
    kernel = native_sparse_attention(
        batch=B,
        seq_len=T,
        head_kv=H,
        heads=HQ,
        dim=D,
        selected_blocks=S,
        bs_pad=BS_pad,
        scale=scale,
    )
    out = kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
    torch.npu.synchronize()
    ref = golden_nsa_fwd(Q, K, V, bi, bc, block_size=BS, scale=scale, is_causal=True)
    return out.cpu(), ref


def test_nsa_boundary():
    """Boundary tests: INF/NAN/extreme/all-zero/scale=None. Non-blocking."""
    dtype = torch.float16
    # D-SPECIAL-ZERO: all-zero inputs.
    _run_boundary(
        "all_zero", dtype, lambda: _boundary_run(dtype, 0, lambda q, k, v: (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)))
    )
    # D-SPECIAL-INF: inputs with inf.
    _run_boundary("with_inf", dtype, lambda: _boundary_run(dtype, 1, lambda q, k, v: (q, k.clone().fill_(float("inf")), v)))
    # D-SPECIAL-NAN: inputs with nan.
    _run_boundary("with_nan", dtype, lambda: _boundary_run(dtype, 2, lambda q, k, v: (q, k.clone().fill_(float("nan")), v)))
    # D-SPECIAL-DBOUND: fp16 max value (65504).
    _run_boundary("dtype_bound", dtype, lambda: _boundary_run(dtype, 3, lambda q, k, v: (q, k.clone().fill_(65504.0), v)))

    # scale=None should default to 1/sqrt(dim) inside the kernel.
    def _boundary_scale_none():
        B, T, H, HQ, D, S, BS, BS_pad = 2, 64, 1, 16, 32, 1, 32, 64
        Q, K, V, bi, bc = gen_test_inputs(B, T, H, HQ, D, S, BS, dtype, 30)
        K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, is_causal=True, bs_pad=BS_pad)
        KV_LEN = S * BS_pad
        G = HQ // H
        K_sel_3d, V_sel_3d, mask_3d = _to_3d_inputs(K_sel, V_sel, mask, B, T, H, G, KV_LEN, D)
        kernel = native_sparse_attention(
            batch=B,
            seq_len=T,
            head_kv=H,
            heads=HQ,
            dim=D,
            selected_blocks=S,
            bs_pad=BS_pad,
            scale=None,
        )
        out = kernel(Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu())
        torch.npu.synchronize()
        expected_scale = (1.0 / D) ** 0.5
        ref = golden_nsa_fwd(Q, K, V, bi, bc, block_size=BS, scale=expected_scale, is_causal=True)
        return out.cpu(), ref

    _run_boundary("scale_none_default", dtype, _boundary_scale_none)


# =============================================================================
# msprof: hardware-level profiling (optimization analysis, CI_CHECKLIST §10.4)
# =============================================================================
_MSPROF_TARGET_SCRIPT = """\
\"\"\"Auto-generated msprof target: runs NSA Forward kernel for msprof op capture.\"\"\"
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import tilelang
tilelang.disable_cache()
from example_tilelang_nsa_fwd import native_sparse_attention
from test_example_tilelang_nsa_fwd import gen_test_inputs, prepare_inputs

# golden config (matches DESIGN.md §12)
B, SEQ_LEN, H, HQ, D, S, BS, BS_PAD, SCALE = 2, 64, 1, 16, 32, 1, 32, 64, 0.1
DTYPE = torch.float16
KV_LEN = S * BS_PAD
G = HQ // H

kernel = native_sparse_attention(
    batch=B, seq_len=SEQ_LEN, head_kv=H, heads=HQ, dim=D,
    selected_blocks=S, bs_pad=BS_PAD, scale=SCALE,
)

# Generate inputs on CPU then .npu() to avoid Cast helper kernels polluting msprof.
Q, K, V, bi, bc = gen_test_inputs(B, SEQ_LEN, H, HQ, D, S, BS, DTYPE, seed=0)
K_sel, V_sel, mask = prepare_inputs(Q, K, V, bi, bc, BS, S, is_causal=True, bs_pad=BS_PAD)
K_sel_3d = K_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
V_sel_3d = V_sel.reshape(B * SEQ_LEN * H, KV_LEN, D)
mask_3d = mask.reshape(B * SEQ_LEN * H, KV_LEN).unsqueeze(1).expand(-1, G, -1).contiguous()
Q_npu, K_sel_npu, V_sel_npu, mask_npu = Q.npu(), K_sel_3d.npu(), V_sel_3d.npu(), mask_3d.npu()
torch.npu.synchronize()

# Script-side warmup: 5 iterations to prime the NPU pipeline (not captured by msprof).
for _ in range(5):
    kernel(Q_npu, K_sel_npu, V_sel_npu, mask_npu)
torch.npu.synchronize()
# 30 iterations for msprof capture: msprof --warm-up=3 skips the first 3 launches,
# --launch-count=10 captures the next 10; remaining launches ensure enough samples.
for _ in range(30):
    kernel(Q_npu, K_sel_npu, V_sel_npu, mask_npu)
torch.npu.synchronize()
"""


def run_msprof():
    """Run kernel under msprof op for hardware-level profiling.

    Auto-generates a temporary target script (so this test file is self-contained,
    no external perf_tuning/ dependency). Output saved to ./msprof_output_latest/
    (previous output is cleared first to keep "latest" meaningful). Requires CANN
    msprof tool installed.

    Returns True on success, False on failure (msprof not found / non-zero exit /
    timeout / target script write failure).
    """
    import shutil
    import subprocess

    op_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(op_dir, "msprof_output_latest")

    # Clear previous msprof output so "latest" always means the current run.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    target_path = os.path.join(op_dir, "_msprof_target_auto.py")
    cmd = [
        "msprof",
        "op",
        f"--application=python {target_path}",
        f"--output={output_dir}",
        "--kernel-name=main_kernel",
        "--launch-count=10",
        "--warm-up=3",
    ]
    print(f"[msprof] running: {' '.join(cmd)}")
    success = False
    try:
        # Write auto-generated target script inside try so write failures
        # (read-only dir / disk full / permission) are caught, not crashed.
        with open(target_path, "w") as f:
            f.write(_MSPROF_TARGET_SCRIPT)
        result = subprocess.run(cmd, cwd=op_dir, timeout=600)
        success = result.returncode == 0
        if not success:
            print(f"[msprof] msprof exited with code {result.returncode} (may still have produced data)")
    except FileNotFoundError:
        print("[msprof] msprof command not found, skipping (install CANN msprof tool to enable)")
    except subprocess.TimeoutExpired:
        print("[msprof] msprof timed out after 600s")
    except OSError as e:
        print(f"[msprof] failed to write target script: {e}")
    finally:
        # Clean up the auto-generated target script.
        if os.path.exists(target_path):
            os.remove(target_path)
    if success:
        print(f"msprof data saved to {output_dir}")
    else:
        print("[msprof] no profiling data produced (run failed)")
    return success


# =============================================================================
# Main: --level dispatch + exit code
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSA Forward layered test suite (Ascend)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all", "msprof"],
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()

    if args.level == "msprof":
        try:
            success = run_msprof()
        except Exception as e:
            print(f"[msprof] unexpected error: {e}")
            success = False
        if success:
            print("\nTest Passed!")
            sys.exit(0)
        print("\nTest FAILED!")
        sys.exit(1)

    blocking_ok = True  # only L0/L1 count toward blocking decision
    if args.level in ("l0", "all"):
        blocking_ok &= test_nsa_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_nsa_l1()
    if args.level in ("l2", "all"):
        test_nsa_l2()  # L2 negative: non-blocking
    if args.level in ("boundary", "all"):
        test_nsa_boundary()  # Boundary precision: non-blocking

    if blocking_ok:
        print("Test Passed!")  # L0/L1 all passed (checklist #16)
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
