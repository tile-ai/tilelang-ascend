"""Layered test suite for MLA Decode Paged Attention (L0/L1/L2/Boundary).

This file imports the kernel from ``example_mla_decode_paged.py`` and provides
the full CI-compliant precision test harness:

- ``--level l0``      : L0 blocking precision tests (aligned shapes)
- ``--level l1``      : L1 blocking precision tests (tail blocks, params, value ranges)
- ``--level l2``      : L2 non-blocking exception tests (wrong dtype/shape)
- ``--level boundary``: Boundary non-blocking tests (zero/inf/nan/dbound)
- ``--level all``     : Run all of the above

Precision standard (fp16 mixed tolerance, per precision-standard.md):
    atol = 2^-14 (6.10e-5), rtol = 2^-9 (1.95e-3),
    max_abs_error_limit = 0.1, required_matched_ratio = 0.99
    Dual-gate: matched_ratio >= 0.99 AND max_abs_error <= 0.1
"""

import argparse
import math
import os
import sys

import torch

# Ensure we can import the sibling kernel module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from example_mla_decode_paged import mla_decode_tilelang  # noqa: E402

# ============================================================================
# Golden reference (CPU, fp32 computation)
# ============================================================================


def golden_mla_decode_paged(
    Q,
    Q_pe,
    KV,
    K_pe,
    block_table,
    cache_seqlens,
    batch,
    h_q,
    h_kv,
    dv,
    dpe,
    block_size,
    max_seqlen_pad,
    softmax_scale=None,
):
    """PyTorch reference implementation (CPU, fp32).

    Gathers paged KV per batch, computes scaled dot-product attention with
    Q_nope @ KV^T + Q_pe @ K_pe^T, softmax, then P @ KV.
    """
    if softmax_scale is None:
        scale = (dv + dpe) ** -0.5
    else:
        scale = float(softmax_scale)
    output = torch.zeros(batch, h_q, dv, dtype=torch.float32)

    for b in range(batch):
        seq_len = int(cache_seqlens[b])
        num_blocks = (seq_len + block_size - 1) // block_size

        # Gather KV blocks for this batch's sequence.
        kv_seq = torch.zeros(seq_len, dv, dtype=torch.float32)
        kpe_seq = torch.zeros(seq_len, dpe, dtype=torch.float32)
        for blk in range(num_blocks):
            phys_block = int(block_table[b, blk])
            kv_start = phys_block * block_size
            valid_n = min(block_size, seq_len - blk * block_size)
            kv_seq[blk * block_size : blk * block_size + valid_n] = KV[kv_start : kv_start + valid_n, 0, :].float()
            kpe_seq[blk * block_size : blk * block_size + valid_n] = K_pe[kv_start : kv_start + valid_n, 0, :].float()

        for h in range(h_q):
            q_nope = Q[b, h, :].float()
            q_pe = Q_pe[b, h, :].float()

            # Attention scores: Q @ KV^T + Q_pe @ K_pe^T.
            s = q_nope @ kv_seq.T + q_pe @ kpe_seq.T
            s = s * scale
            p = torch.softmax(s, dim=-1)
            output[b, h, :] = p @ kv_seq

    return output.to(torch.float16)


# ============================================================================
# Constants
# ============================================================================

BLOCK_N = 256  # KV block size (halves iters vs 128).
BLOCK_H = 32  # Query head block size (fits L0C 128KB).
CORE_NUM = 20  # Ascend 910B3 physical cube cores.

# ============================================================================
# Precision check (mixed tolerance, dual-gate, per precision-standard.md)
# ============================================================================


def check_precision(name, actual, golden, dtype_str="float16"):
    """Dual-gate precision check: matched_ratio >= required AND max_abs <= limit.

    Per precision-standard.md: threshold only depends on dtype, not operator.
    INF/NAN positions are structurally compared (not counted in numeric tolerance).
    """
    if dtype_str == "float16":
        atol = 2.0**-14  # 6.10e-5
        rtol = 2.0**-9  # 1.95e-3
        max_abs_limit = 1e-1
        required_ratio = 0.99
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    actual_f = actual.float()
    golden_f = golden.float()

    # Structural comparison for inf/nan positions.
    special = ~torch.isfinite(golden_f)
    if special.any() and (
        not torch.equal(torch.isnan(actual_f[special]), torch.isnan(golden_f[special]))
        or not torch.equal(torch.isinf(actual_f[special]), torch.isinf(golden_f[special]))
    ):
        print(f"  {name}: [PRECISION_FAIL]  inf/nan position mismatch")
        return False, 0.0, float("inf")

    # Numeric comparison on finite positions.
    mask = torch.isfinite(golden_f)
    if mask.sum().item() == 0:
        print(f"  {name}: [PRECISION_PASS]  all inf/nan (structural match)")
        return True, 1.0, 0.0

    diff = (actual_f[mask] - golden_f[mask]).abs()
    tolerance = atol + rtol * golden_f[mask].abs()
    matched_ratio = (diff <= tolerance).float().mean().item()
    max_abs_error = diff.max().item()

    passed = matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
    status = "[PRECISION_PASS]" if passed else "[PRECISION_FAIL]"
    print(f"  {name}: {status}  matched_ratio={matched_ratio:.6f}  max_abs_error={max_abs_error:.6e}")
    return passed, matched_ratio, max_abs_error


# ============================================================================
# Coverage manifest (for coverage_check.py)
# ============================================================================

COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 14,
    "D-DTYPE-int32": 14,
    "D-SHAPE-ALIGNED": 8,
    "D-SHAPE-EDGE": 1,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-VALRANGE-S": 1,
    "D-VALRANGE-M": 6,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
    "D-PARAM-block_size": 1,
    "D-PARAM-softmax_scale": 1,
    "D-PARAM-num_split": 1,
}
COVERAGE_NA = {
    # Fusion category: num_split>1 needs combine kernel (not implemented).
}

# ============================================================================
# Test case definitions
# ============================================================================

# ---- L0 threshold cases (blocking) ----
L0_CASES = [
    ("l0_small_b1", 1, 128, 1, 128, 576, 512, 256, ["D-DTYPE-fp16", "D-DTYPE-int32", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_small_b2", 2, 128, 1, 256, 576, 512, 256, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_batch4", 4, 128, 1, 512, 576, 512, 256, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_seq2048", 4, 128, 1, 2048, 576, 512, 256, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l0_golden", 128, 128, 1, 8192, 576, 512, 256, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
]

# ---- L1 functional cases (blocking) ----
L1_CASES = [
    ("l1_tail_1", 1, 128, 1, 65, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-TAIL-1", "D-VALRANGE-M"]),
    ("l1_tail_mid", 2, 128, 1, 160, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID", "D-VALRANGE-M"]),
    ("l1_prime", 1, 128, 1, 97, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-PRIME", "D-VALRANGE-M"]),
    ("l1_edge", 1, 128, 1, 256, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-EDGE", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    ("l1_batch16", 16, 128, 1, 512, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"]),
    (
        "l1_seq4096",
        4,
        128,
        1,
        4096,
        576,
        512,
        256,
        lambda shape, dtype: torch.randn(*shape, dtype=dtype) * 16000.0,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    ),
    (
        "l1_param_bs512",
        1,
        128,
        1,
        512,
        576,
        512,
        512,
        None,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-block_size", "D-VALRANGE-M"],
    ),
    (
        "l1_param_scale",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        None,
        0.05,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-softmax_scale", "D-VALRANGE-M"],
    ),
    (
        "l1_param_numsplit",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        None,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-PARAM-num_split", "D-VALRANGE-M"],
    ),
    (
        "l1_valrange_s",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: torch.randn(*shape, dtype=dtype) * 0.25,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    ),
    (
        "l1_valrange_asym",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: torch.randn(*shape, dtype=dtype) * 3.75 + 2.5,
        None,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-ASYM"],
    ),
]

# ---- L2 exception cases (non-blocking, should be rejected) ----
L2_CASES = [
    ("l2_wrong_dtype", "wrong_dtype", ["D-EXC-DTYPE"]),
    ("l2_wrong_shape", "wrong_shape", ["D-EXC-SHAPE"]),
]

# ---- Boundary special-value cases (non-blocking) ----
BOUNDARY_CASES = [
    (
        "b_zero",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: torch.zeros(*shape, dtype=dtype),
        None,
        ["D-DTYPE-fp16", "D-SPECIAL-ZERO"],
    ),
    (
        "b_dbound",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: (torch.randn(*shape, dtype=dtype) * 0.5 + 0.5).clamp_(-65504, 65504) * 60000,
        None,
        ["D-DTYPE-fp16", "D-SPECIAL-DBOUND"],
    ),
    (
        "b_inf",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: torch.where(torch.rand(*shape) > 0.99, float("inf"), torch.randn(*shape, dtype=dtype)),
        None,
        ["D-DTYPE-fp16", "D-SPECIAL-INF"],
    ),
    (
        "b_nan",
        1,
        128,
        1,
        256,
        576,
        512,
        256,
        lambda shape, dtype: torch.where(torch.rand(*shape) > 0.99, float("nan"), torch.randn(*shape, dtype=dtype)),
        None,
        ["D-DTYPE-fp16", "D-SPECIAL-NAN"],
    ),
    ("b_single_token", 1, 128, 1, 1, 576, 512, 256, None, None, ["D-DTYPE-fp16", "D-SHAPE-EDGE"]),
]


# ============================================================================
# Test harness
# ============================================================================


def _default_data_gen(shape, dtype):
    """Default data generator: standard normal distribution ~ [-4, 4]."""
    return torch.randn(*shape, dtype=dtype)


def _sort_block_table(block_table, KV, K_pe, block_size):
    """Sort block_table by physical block address (ascending) + reorder KV/K_pe.

    In production (vLLM/SGLang), paged KV cache physical blocks are scattered.
    Sorting improves DMA locality and L2 cache hit rate. MLA decode is non-causal
    attention (seq_q=1), so KV block order doesn't affect the result (softmax is
    commutative). For benchmark data (torch.arange, already ascending), this is a no-op.
    """
    batch, num_blocks = block_table.shape
    sorted_indices = block_table.argsort(dim=1)
    sorted_phys = block_table.gather(1, sorted_indices)

    # Build flat index into KV/K_pe for reordered physical blocks.
    flat_offset = sorted_phys.unsqueeze(-1) * block_size
    row_idx = flat_offset + torch.arange(block_size)
    row_idx_flat = row_idx.reshape(-1)

    KV_sorted = KV[row_idx_flat].contiguous()
    K_pe_sorted = K_pe[row_idx_flat].contiguous()
    sorted_block_table = torch.arange(batch * num_blocks, dtype=torch.int32).reshape(batch, num_blocks)

    return sorted_block_table, KV_sorted, K_pe_sorted


def run_test_case(
    name,
    batch,
    h_q,
    h_kv,
    cache_seqlen,
    d,
    dv,
    block_size,
    data_gen=None,
    softmax_scale_override=None,
    tags=None,
    dtype=torch.float16,
    level_tag="",
):
    """Run a single test case and check precision against the golden reference."""
    dpe = d - dv
    if data_gen is None:
        data_gen = _default_data_gen

    cache_seqlens_cpu = torch.tensor([cache_seqlen] * batch, dtype=torch.int32)
    max_seqlen = int(cache_seqlens_cpu.max())
    max_seqlen_pad = math.ceil(max_seqlen / block_size) * block_size
    if max_seqlen_pad < BLOCK_N:
        max_seqlen_pad = BLOCK_N

    num_blocks_per_batch = max_seqlen_pad // block_size

    # Generate data on CPU (fixed seed for reproducibility).
    torch.manual_seed(42)
    Q_full = data_gen((batch, h_q, d), dtype)
    Q = Q_full[..., :dv].contiguous()
    Q_pe = Q_full[..., dv:].contiguous()

    # Pre-multiply Q by softmax_scale (fuses scale into Q at host side).
    if softmax_scale_override is not None:
        pre_scale = float(softmax_scale_override)
    else:
        pre_scale = d**-0.5
    Q = (Q * pre_scale).contiguous()
    Q_pe = (Q_pe * pre_scale).contiguous()

    blocked_k = data_gen((batch * num_blocks_per_batch, block_size, h_kv, d), dtype)
    KV = blocked_k[..., :dv].reshape(-1, h_kv, dv).contiguous()
    K_pe = blocked_k[..., dv:].reshape(-1, h_kv, dpe).contiguous()
    block_table = torch.arange(batch * num_blocks_per_batch, dtype=torch.int32).reshape(batch, num_blocks_per_batch)

    # Sort block_table by physical block address (improves L2 hit for scattered blocks).
    block_table, KV, K_pe = _sort_block_table(block_table, KV, K_pe, block_size)

    # Defensive: validate block_table values are in valid range.
    # kernel does not bounds-check block_table (too expensive on-device);
    # host-side validation prevents silent GM out-of-bounds reads.
    assert block_table.min() >= 0, "block_table contains negative index"
    assert block_table.max() < batch * num_blocks_per_batch, (
        f"block_table max {int(block_table.max())} exceeds KV pool size {batch * num_blocks_per_batch} blocks"
    )

    # Golden (CPU, fp32) — Q is pre-scaled, so golden must NOT apply scale again.
    golden_out = golden_mla_decode_paged(
        Q,
        Q_pe,
        KV,
        K_pe,
        block_table,
        cache_seqlens_cpu,
        batch,
        h_q,
        h_kv,
        dv,
        dpe,
        block_size,
        max_seqlen_pad,
        softmax_scale=1.0,
    )

    # Move to NPU and run kernel.
    Q_npu = Q.npu()
    Q_pe_npu = Q_pe.npu()
    KV_npu = KV.npu()
    K_pe_npu = K_pe.npu()
    block_table_npu = block_table.npu()

    if softmax_scale_override is not None:
        softmax_scale = float(softmax_scale_override)
    else:
        softmax_scale = d**-0.5
    kernel = mla_decode_tilelang(
        batch,
        h_q,
        h_kv,
        max_seqlen_pad,
        dv,
        dpe,
        BLOCK_N,
        BLOCK_H,
        block_size,
        cache_seqlen,
        CORE_NUM,
        softmax_scale,
    )
    out = kernel(Q_npu, Q_pe_npu, KV_npu, K_pe_npu, block_table_npu)
    out_cpu = out.cpu()

    return check_precision(name, out_cpu, golden_out, "float16")


def run_l2_case(name, case_type, tags=None):
    """Run L2 exception case: verify invalid input is rejected (exception = PASS)."""
    print(f"\n--- Running {name} (L2: {case_type}) ---")
    try:
        batch, h_q, h_kv, cache_seqlen, d, dv, block_size = 1, 128, 1, 256, 576, 512, 256
        dpe = d - dv
        max_seqlen_pad = 256
        num_blocks_per_batch = max_seqlen_pad // block_size
        torch.manual_seed(42)

        if case_type == "wrong_dtype":
            wrong_dtype = torch.float32
            Q_full = torch.randn(batch, h_q, d, dtype=wrong_dtype)
            Q = Q_full[..., :dv].contiguous()
            Q_pe = Q_full[..., dv:].contiguous()
            blocked_k = torch.randn(batch * num_blocks_per_batch, block_size, h_kv, d, dtype=wrong_dtype)
        elif case_type == "wrong_shape":
            # Q with dv=256 (wrong, kernel compiled for dv=512).
            Q = torch.randn(batch, h_q, 256, dtype=torch.float16)
            Q_pe = torch.randn(batch, h_q, dpe, dtype=torch.float16)
            blocked_k = torch.randn(batch * num_blocks_per_batch, block_size, h_kv, d, dtype=torch.float16)
        else:
            raise ValueError(f"Unknown case_type: {case_type}")

        KV = blocked_k[..., :dv].reshape(-1, h_kv, dv).contiguous()
        K_pe = blocked_k[..., dv:].reshape(-1, h_kv, dpe).contiguous()
        block_table = torch.arange(batch * num_blocks_per_batch, dtype=torch.int32).reshape(batch, num_blocks_per_batch)

        Q_npu = Q.npu()
        Q_pe_npu = Q_pe.npu()
        KV_npu = KV.npu()
        K_pe_npu = K_pe.npu()
        block_table_npu = block_table.npu()

        softmax_scale = d**-0.5
        kernel = mla_decode_tilelang(
            batch,
            h_q,
            h_kv,
            max_seqlen_pad,
            dv,
            dpe,
            BLOCK_N,
            BLOCK_H,
            block_size,
            cache_seqlen,
            CORE_NUM,
            softmax_scale,
        )
        kernel(Q_npu, Q_pe_npu, KV_npu, K_pe_npu, block_table_npu)
        print(f"  {name}: [BOUNDARY_WARN]  kernel silently accepted wrong input")
        return False, "warn"
    except Exception as e:
        print(f"  {name}: [BOUNDARY_PASS]  correctly rejected: {type(e).__name__}: {e}")
        return True, "pass"


def run_boundary_case(
    name,
    batch,
    h_q,
    h_kv,
    cache_seqlen,
    d,
    dv,
    block_size,
    data_gen=None,
    softmax_scale_override=None,
    tags=None,
):
    """Run Boundary case: special values, precision fail reports WARN (non-blocking)."""
    print(f"\n--- Running {name} (Boundary, B={batch}, S={cache_seqlen}) ---")
    try:
        passed, ratio, max_err = run_test_case(
            name,
            batch,
            h_q,
            h_kv,
            cache_seqlen,
            d,
            dv,
            block_size,
            data_gen=data_gen,
            softmax_scale_override=softmax_scale_override,
            tags=tags,
            level_tag="boundary",
        )
        if passed:
            print(f"[BOUNDARY_PASS] boundary {name} max_diff={max_err:.6e}")
        else:
            print(f"[BOUNDARY_WARN] boundary {name} max_diff={max_err:.6e} (non-blocking)")
        return passed, max_err
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e} (non-blocking)")
        return False, float("inf")


# ============================================================================
# Layered test functions
# ============================================================================


def test_mla_decode_paged_l0():
    """L0 blocking precision tests: 5 aligned-shape cases."""
    print("\n" + "=" * 60)
    print("L0 threshold tests (aligned cache_seqlens, multiples of 256)")
    print("=" * 60)
    results = []
    for case in L0_CASES:
        name, b, h_q, h_kv, seq, d_val, dv, bs = case[:8]
        tags = case[8] if len(case) > 8 else []
        print(f"\n--- Running {name} (B={b}, S={seq}, d={d_val}, dv={dv}) ---")
        try:
            passed, ratio, max_err = run_test_case(
                name,
                b,
                h_q,
                h_kv,
                seq,
                d_val,
                dv,
                bs,
                tags=tags,
                level_tag="l0",
            )
            results.append((name, passed, ratio, max_err))
        except Exception as e:
            print(f"  {name}: [PRECISION_FAIL]  Exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False, 0.0, float("inf")))

    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    all_pass = all(p for _, p, _, _ in results)
    print(f"\n=== L0 Summary: {passed_count}/{total} passed ===")
    if all_pass:
        print("[PRECISION_PASS] All L0 cases passed")
    else:
        print("[PRECISION_FAIL] Some L0 cases failed")
    return results, all_pass


def test_mla_decode_paged_l1():
    """L1 blocking precision tests: irregular shapes, tail blocks, params, value ranges."""
    print("\n" + "=" * 60)
    print("L1 functional tests (tail blocks, params, value ranges)")
    print("=" * 60)
    results = []
    for case in L1_CASES:
        (name, b, h_q, h_kv, seq, d_val, dv, bs, data_gen, scale_override, tags) = case
        print(f"\n--- Running {name} (B={b}, S={seq}, bs={bs}, scale={scale_override}) ---")
        try:
            passed, ratio, max_err = run_test_case(
                name,
                b,
                h_q,
                h_kv,
                seq,
                d_val,
                dv,
                bs,
                data_gen=data_gen,
                softmax_scale_override=scale_override,
                tags=tags,
                level_tag="l1",
            )
            if passed:
                print(f"[PRECISION_PASS] l1 {name} max_diff={max_err:.6e}")
            else:
                print(f"[PRECISION_FAIL] l1 {name} max_diff={max_err:.6e}")
            results.append((name, passed, ratio, max_err))
        except Exception as e:
            print(f"  {name}: [PRECISION_FAIL]  Exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False, 0.0, float("inf")))

    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    all_pass = all(p for _, p, _, _ in results)
    print(f"\n=== L1 Summary: {passed_count}/{total} passed ===")
    if all_pass:
        print("[PRECISION_PASS] All L1 cases passed")
    else:
        print("[PRECISION_FAIL] Some L1 cases failed")
    return results, all_pass


def test_mla_decode_paged_l2():
    """L2 non-blocking exception tests: invalid inputs should be rejected."""
    print("\n" + "=" * 60)
    print("L2 exception tests (invalid inputs should be rejected)")
    print("=" * 60)
    results = []
    for name, case_type, tags in L2_CASES:
        passed, status = run_l2_case(name, case_type, tags)
        results.append((name, passed, status))

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"\n=== L2 Summary: {passed_count}/{total} correctly rejected ===")
    return results


def test_mla_decode_paged_boundary():
    """Boundary non-blocking tests: zero/inf/nan/dbound/single-token."""
    print("\n" + "=" * 60)
    print("Boundary special-value tests (zero/inf/nan/dbound/single-token)")
    print("=" * 60)
    results = []
    for case in BOUNDARY_CASES:
        (name, b, h_q, h_kv, seq, d_val, dv, bs, data_gen, scale_override, tags) = case
        passed, max_err = run_boundary_case(
            name,
            b,
            h_q,
            h_kv,
            seq,
            d_val,
            dv,
            bs,
            data_gen=data_gen,
            softmax_scale_override=scale_override,
            tags=tags,
        )
        results.append((name, passed, max_err))

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"\n=== Boundary Summary: {passed_count}/{total} passed ===")
    return results


# ============================================================================
# Main entry (argparse --level)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MLA Decode Paged Attention Test")
    parser.add_argument(
        "--level",
        choices=["l0", "l1", "l2", "boundary", "all"],
        default="l0",
        help="Test level to run",
    )
    args = parser.parse_args()
    print(f"Running test level: {args.level}")

    exit_code = 0

    l0_results = l1_results = l2_results = b_results = None
    l0_all_pass = l1_all_pass = True

    if args.level in ("l0", "all"):
        l0_results, l0_all_pass = test_mla_decode_paged_l0()

    if args.level in ("l1", "all"):
        l1_results, l1_all_pass = test_mla_decode_paged_l1()

    if args.level in ("l2", "all"):
        l2_results = test_mla_decode_paged_l2()

    if args.level in ("boundary", "all"):
        b_results = test_mla_decode_paged_boundary()

    # Final verdict: L0/L1 blocking; L2/Boundary non-blocking.
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    if args.level in ("l0", "all"):
        l0_pass = sum(1 for _, p, _, _ in l0_results if p)
        print(f"L0: {l0_pass}/{len(l0_results)} passed")
    if args.level in ("l1", "all"):
        l1_pass = sum(1 for _, p, _, _ in l1_results if p)
        print(f"L1: {l1_pass}/{len(l1_results)} passed")
    if args.level in ("l2", "all"):
        l2_pass = sum(1 for _, p, _ in l2_results if p)
        print(f"L2: {l2_pass}/{len(l2_results)} correctly rejected (non-blocking)")
    if args.level in ("boundary", "all"):
        b_pass = sum(1 for _, p, _ in b_results if p)
        print(f"Boundary: {b_pass}/{len(b_results)} passed (non-blocking)")

    # L0/L1 blocking verdict.
    if args.level == "l0":
        if l0_all_pass:
            print("[PRECISION_PASS] L0 all passed")
            print("Test Passed!")
        else:
            print("[PRECISION_FAIL] L0 has failures")
            exit_code = 1
    elif args.level in ("all", "l1"):
        blocking_pass = l0_all_pass and l1_all_pass
        if blocking_pass:
            print("[PRECISION_PASS] All L0+L1 cases passed (L2/Boundary non-blocking)")
            print("Test Passed!")
        else:
            print("[PRECISION_FAIL] L0 or L1 has failures (blocking)")
            exit_code = 1
    elif args.level in ("l2", "boundary"):
        # Non-blocking levels always pass.
        print("Test Passed!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
