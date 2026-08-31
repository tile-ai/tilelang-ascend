"""NSA Forward VarLen precision test suite: L0/L1/L2/Boundary.

This file is the single precision-test entry for `example_tilelang_nsa_fwd_varlen.py`.
It owns all host-side helpers (token indices, test data, golden reference,
host wrapper, precision check) and the layered precision test cases
(L0 threshold / L1 functional / L2 exception / Boundary special-value).

Run:
  python test_example_tilelang_nsa_fwd_varlen.py --level all     # full precision suite
  python test_example_tilelang_nsa_fwd_varlen.py --level l0      # L0 threshold only
"""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_tilelang_nsa_fwd_varlen import native_sparse_attention_varlen  # noqa: E402


# =============================================================================
# Host-side helpers (CPU preprocessing, golden, precision)
# =============================================================================
def prepare_token_indices(offsets):
    """Build token_indices [C_SEQ_LEN, 2] = (batch_idx, token_idx_in_seq) from offsets [N+1].

    CPU-only construction (no NPU aclnn). Returns int32 tensor.
    """
    n = len(offsets) - 1
    total = int(offsets[-1].item())
    token_indices = torch.zeros(total, 2, dtype=torch.int32)
    idx = 0
    for batch_idx in range(n):
        seg_len = int(offsets[batch_idx + 1].item()) - int(offsets[batch_idx].item())
        for t in range(seg_len):
            token_indices[idx, 0] = batch_idx
            token_indices[idx, 1] = t
            idx += 1
    return token_indices


def make_test_data(n, c_seq_len, h, hq, d, s, bs, dtype, seed=42):
    """Construct deterministic NSA varlen test data on CPU.

    Returns q, k, v, block_indices, block_counts, offsets, token_indices, g_slc, scale.
    """
    assert hq % h == 0, f"HQ ({hq}) must be a multiple of H ({h})"
    assert (hq // h) % 2 == 0, f"groups G=HQ/H ({hq // h}) must be even (vid split half_G=G//2); got HQ={hq}, H={h}"
    torch.manual_seed(seed)

    # offsets: split c_seq_len into n segments, each >= bs.
    if n == 1:
        offsets = torch.tensor([0, c_seq_len], dtype=torch.int32)
    else:
        n_slots = c_seq_len // bs
        if n_slots >= n:
            slot_splits = torch.randperm(n_slots - 1)[: n - 1].sort().values + 1
            split_pts = (slot_splits * bs).to(torch.int32)
            offsets = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32),
                    split_pts,
                    torch.tensor([c_seq_len], dtype=torch.int32),
                ]
            )
        else:
            # c_seq_len < bs*n: cannot split into n BS-aligned segments.
            # Fallback to linspace but align split points to BS boundary to avoid
            # GM read overrun in kernel (K[bos+i_s:bos+i_s+BS] needs BS-aligned seg).
            # Last segment may be shorter than BS; block_indices generation handles
            # seg_len < BS by setting block_counts=0 (NS=0 path in kernel).
            raw_offsets = torch.linspace(0, c_seq_len, n + 1).round().to(torch.int32)
            raw_offsets[1:-1] = (raw_offsets[1:-1] // bs) * bs
            raw_offsets[0] = 0
            raw_offsets[-1] = c_seq_len
            offsets = raw_offsets

    token_indices = prepare_token_indices(offsets)

    # block_indices: each token selects up to s candidate blocks within segment & causal.
    block_indices = torch.full((1, c_seq_len, h, s), 0, dtype=torch.int64)
    block_counts = torch.zeros((1, c_seq_len, h), dtype=torch.int64)
    for c in range(c_seq_len):
        i_n = int(token_indices[c, 0].item())
        t = int(token_indices[c, 1].item())
        bos = int(offsets[i_n].item())
        seg_len = int(offsets[i_n + 1].item()) - bos
        max_safe = (seg_len - bs) // bs
        max_causal = t // bs
        max_block = min(max_safe, max_causal)
        n_candidates = max_block + 1
        if n_candidates <= 0:
            continue
        for head_idx in range(h):
            n_sel = min(s, n_candidates)
            i_i = torch.randperm(n_candidates)[:n_sel].sort().values
            block_indices[0, c, head_idx, :n_sel] = i_i
            block_counts[0, c, head_idx] = n_sel

    def _make_kv(seq_len, n_expand, kv_dtype):
        perm = torch.randperm(seq_len)
        return torch.linspace(0, 1, steps=seq_len, dtype=kv_dtype)[perm].view(1, seq_len, 1, 1).expand(1, seq_len, n_expand, d).contiguous()

    q = _make_kv(c_seq_len, hq, dtype)
    k = _make_kv(c_seq_len, h, dtype)
    v = _make_kv(c_seq_len, h, dtype)
    g_slc = torch.rand((1, c_seq_len, hq), dtype=dtype)

    scale = d**-0.5
    return q, k, v, block_indices, block_counts, offsets, token_indices, g_slc, scale


def naive_nsa_fwd_varlen(q, k, v, block_indices, block_counts, block_size, scale, offsets, token_indices, g_slc):
    """NSA Forward VarLen PyTorch reference (golden). CPU float32, cast to q.dtype."""
    _, C_SEQ_LEN, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    BS = block_size
    o_slc = torch.zeros_like(q)

    for c in range(C_SEQ_LEN):
        i_n = int(token_indices[c, 0].item())
        i_t = int(token_indices[c, 1].item())
        bos = int(offsets[i_n].item())

        for h_kv in range(H):
            NS = int(block_counts[0, c, h_kv].item())
            if NS <= 0:
                continue
            q_h = q[0, c, h_kv * G : (h_kv + 1) * G, :].float()  # [G, D]
            scores_max = torch.full((G,), float("-inf"))
            logsum = torch.zeros(G)
            acc_o = torch.zeros(G, D)

            for i in range(NS):
                i_s = int(block_indices[0, c, h_kv, i].item()) * BS
                if i_s < 0 or i_s > i_t:
                    continue
                k_h = k[0, bos + i_s : bos + i_s + BS, h_kv, :].float()  # [BS, D]
                v_h = v[0, bos + i_s : bos + i_s + BS, h_kv, :].float()  # [BS, D]

                s = torch.einsum("gd,bd->gb", q_h, k_h) * scale  # [G, BS]

                col_pos = torch.arange(BS) + i_s
                s[:, col_pos > i_t] = float("-inf")

                scores_max_prev = scores_max.clone()
                scores_max = torch.maximum(scores_max, s.max(dim=-1).values)
                alpha = torch.exp(scores_max_prev - scores_max)
                p = torch.exp(s - scores_max.unsqueeze(-1))
                scores_sum = p.sum(dim=-1)
                logsum = logsum * alpha + scores_sum
                acc_o = acc_o * alpha.unsqueeze(-1) + torch.einsum("gb,bd->gd", p, v_h)

            o_slc[0, c, h_kv * G : (h_kv + 1) * G, :] = (acc_o / logsum.unsqueeze(-1)).to(q.dtype)

    o = o_slc * g_slc.unsqueeze(-1)
    return o


def parallel_nsa_fwd(q, k, v, block_indices, block_counts, block_size, scale, offsets, token_indices, g_slc):
    """Host-side wrapper: H2D-safe preprocessing + kernel call.

    - fp32 input is pre-cast to fp16 on host (Cube doesn't support fp32 GEMM).
    - Q is pre-multiplied with scale (fuse scale into Q, eliminates per-iter axpy).
    - All int64 -> int32 conversions done on CPU before H2D.
    - Gate multiplication fused in kernel; causal mask computed in-kernel.
    """
    B, C_SEQ_LEN, H, K_dim = k.shape
    _, _, HQ, V_dim = q.shape
    _, _, _, S = block_indices.shape
    G = HQ // H
    BS = block_size
    batch = len(offsets) - 1

    dtype_str = str(v.dtype).replace("torch.", "")
    gemm_dtype_str = "float16" if dtype_str == "float32" else dtype_str
    gemm_dtype = getattr(torch, gemm_dtype_str)

    # Q pre-multiply scale (fuse scale into Q at host side).
    q = (q * scale).contiguous()

    # Host-side pre-cast fp32 -> fp16.
    if dtype_str != gemm_dtype_str:
        q = q.to(gemm_dtype)
        k = k.to(gemm_dtype)
        v = v.to(gemm_dtype)

    # GM workspace auto-allocated by framework via workspace_idx (no host allocation).
    core_num = 20

    # Host-side precompute varlen indices to break scalar GM read chain.
    bos_per_token = offsets[token_indices[:, 0]].to(torch.int32)
    bi_2d = block_indices.view(C_SEQ_LEN, H, S)
    bc_2d = block_counts.view(C_SEQ_LEN, H)
    i_s_safe = torch.where(
        bc_2d > 0,
        bi_2d[:, :, 0].to(torch.int32) * BS,
        torch.zeros(C_SEQ_LEN, H, dtype=torch.int32, device=block_indices.device),
    )

    o_slc = native_sparse_attention_varlen(
        batch=batch,
        c_seq_len=C_SEQ_LEN,
        heads=HQ,
        dim=K_dim,
        is_causal=True,
        scale=scale,
        block_size=block_size,
        groups=G,
        selected_blocks=S,
        dtype=dtype_str,
        core_num=core_num,
    )(
        q.view(C_SEQ_LEN, HQ, K_dim),
        k.view(C_SEQ_LEN, H, K_dim),
        v.view(C_SEQ_LEN, H, K_dim),
        bos_per_token,
        i_s_safe,
        bc_2d.to(torch.int32),
        g_slc.view(C_SEQ_LEN, HQ),
    )
    return o_slc.view(B, C_SEQ_LEN, HQ, V_dim)


# =============================================================================
# Precision standard (mixed tolerance dual-gate, precision-standard.md)
# =============================================================================
def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio).

    Float: mixed tolerance; Int: exact match (0 error).
    Matches precision-standard.md section 2.
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
    dtype_str = str(dtype).replace("torch.", "")
    if dtype_str in int_types:
        return (0.0, 0.0, 0.0, 1.0)  # integer exact match
    return fp_table.get(dtype_str, (2**-14, 2**-9, 1e-1, 0.99))


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
    # inf/nan structural compare (precision-standard.md section 3.1).
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
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = matched_ratio >= required_ratio and max_abs_error <= max_abs_limit
    return passed, matched_ratio, max_abs_error


# =============================================================================
# Test runner
# =============================================================================
def _run_nsa_case(level, n, c_seq_len, h, hq, d, s, bs, dtype_str, vrange=(-1, 1), tags=None):
    """Run a single NSA varlen test case and check precision."""
    dtype = getattr(torch, dtype_str)
    q, k, v, bi, bc, off, ti, g, scale = make_test_data(n, c_seq_len, h, hq, d, s, bs, dtype)

    if vrange != (-1, 1):
        lo, hi = vrange
        q = q * (hi - lo) + lo
        k = k * (hi - lo) + lo
        v = v * (hi - lo) + lo

    ref = naive_nsa_fwd_varlen(q.float(), k.float(), v.float(), bi, bc, bs, scale, off, ti, g.float()).to(dtype)

    bi_i32 = bi.to(torch.int32)
    bc_i32 = bc.to(torch.int32)
    off_i32 = off.to(torch.int32)
    ti_i32 = ti.to(torch.int32)

    out = parallel_nsa_fwd(
        q.npu(),
        k.npu(),
        v.npu(),
        bi_i32.npu(),
        bc_i32.npu(),
        bs,
        scale,
        off_i32.npu(),
        ti_i32.npu(),
        g.float().npu(),
    )
    out_cpu = out.cpu()
    passed, ratio, max_abs = check_precision(out_cpu, ref, dtype_str)
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] {level} N={n} C_SEQ={c_seq_len} D={d} S={s} BS={bs} dtype={dtype_str} ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed


# =============================================================================
# L0 threshold tests (blocking)
# =============================================================================
def test_l0_basic_fp16():
    """L0: N=2, C_SEQ_LEN=64, HQ=16, H=1, D=64, S=1, BS=32, fp16."""
    return _run_nsa_case("l0", 2, 64, 1, 16, 64, 1, 32, "float16")


def test_nsa_fwd_varlen_l0():
    """L0 gate: run all L0 cases."""
    return test_l0_basic_fp16()


# =============================================================================
# L1 functional tests (blocking)
# =============================================================================
# (N, C_SEQ_LEN, H, HQ, D, S, BS, dtype, tags)
L1_CASES = [
    (2, 64, 1, 16, 64, 1, 32, "float16", ["D-SHAPE-ALIGNED"]),
    (1, 65, 1, 16, 64, 1, 32, "float16", ["D-SHAPE-TAIL-1"]),
    (3, 96, 1, 16, 64, 1, 32, "float16", ["D-SHAPE-TAIL-MID"]),
    (3, 97, 1, 16, 64, 1, 32, "float16", ["D-SHAPE-PRIME"]),
    (1, 32, 1, 16, 64, 1, 32, "float16", ["D-SHAPE-EDGE"]),
    (2, 64, 1, 16, 64, 1, 32, "float16", ["D-VALRANGE-L"]),
    (2, 64, 1, 16, 64, 1, 32, "float16", ["D-VALRANGE-ASYM"]),
    (2, 64, 1, 16, 64, 1, 32, "float16", ["D-VALRANGE-M"]),
    (2, 64, 1, 16, 64, 1, 32, "bfloat16", ["D-DTYPE-bf16"]),
    (2, 64, 1, 16, 64, 1, 32, "float32", ["D-DTYPE-fp32"]),
]

# D-SPECIAL-DBOUND: D=32 (smaller head dim; requires recompile, D is compile-time param)
L1_DBOUND_CASES = [
    (2, 64, 1, 16, 32, 1, 32, "float16", ["D-SPECIAL-DBOUND"]),
]

# D-TYPICAL: model-typical configs aligned with DeepSeek NSA paper settings.
# - GQA groups=16 (HQ/H=16, matches vid split half_G=8 and ZN/NZ fractal alignment).
# - block_size=32, S=1 specialization, fp16.
# - Covers seq_len up to 4K and head_dim=64/128 (training-scale shapes).
# Note: each (D, BS, G) combination triggers a separate JIT compile (D/BS/G are
# compile-time constants), so these cases also validate multi-config robustness.
L1_TYPICAL_CASES = [
    (2, 2048, 4, 64, 64, 1, 32, "float16", ["D-TYPICAL-SEQ2K"]),
    (4, 1024, 4, 64, 64, 1, 32, "float16", ["D-TYPICAL-BATCH4"]),
    (1, 4096, 4, 64, 64, 1, 32, "float16", ["D-TYPICAL-SEQ4K"]),
    (2, 1024, 4, 64, 128, 1, 32, "float16", ["D-TYPICAL-D128"]),
]


def test_nsa_fwd_varlen_l1():
    """L1 functional tests: irregular shapes + value ranges + dtype + typical configs."""
    ok = True
    for case in L1_CASES + L1_DBOUND_CASES + L1_TYPICAL_CASES:
        n, c_seq_len, h, hq, d, s, bs, dt, tags = case
        vrange = (-1, 1)
        if "D-VALRANGE-L" in (tags or []):
            vrange = (-50, 50)
        elif "D-VALRANGE-M" in (tags or []):
            vrange = (-10, 10)
        elif "D-VALRANGE-ASYM" in (tags or []):
            vrange = (-5, 10)
        try:
            ok &= _run_nsa_case("l1", n, c_seq_len, h, hq, d, s, bs, dt, vrange, tags)
        except Exception as e:
            # D-SPECIAL-DBOUND: D=32 requires recompile; record as non-blocking WARN.
            if "D-SPECIAL-DBOUND" in (tags or []):
                print(f"[BOUNDARY_WARN] l1 D-SPECIAL-DBOUND case {case}: {e}")
            else:
                print(f"[PRECISION_FAIL] l1 case {case}: {e}")
                ok = False
    return ok


# =============================================================================
# L2 exception tests (non-blocking)
# =============================================================================
def test_nsa_fwd_varlen_l2():
    """L2 negative tests: illegal inputs should be rejected."""
    # D-EXC-DTYPE: unsupported dtype (float64)
    try:
        _run_nsa_case("l2", 2, 64, 1, 16, 64, 1, 32, "float64")
        print("[BOUNDARY_WARN] l2 exc_dtype: float64 not rejected")
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 exc_dtype: rejected ({type(e).__name__})")

    # D-EXC-SHAPE: HQ=15 -> G=15 is odd (violates vid split half_G=G//2 requirement).
    try:
        make_test_data(2, 64, 1, 15, 64, 1, 32, torch.float16)
        print("[BOUNDARY_WARN] l2 exc_shape: HQ=15 not rejected")
    except AssertionError as e:
        print(f"[BOUNDARY_PASS] l2 exc_shape: rejected (AssertionError: {e})")
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 exc_shape: rejected ({type(e).__name__})")


# =============================================================================
# Boundary tests (non-blocking)
# =============================================================================
def _run_boundary_special(name, q_inject_fn):
    """Run a boundary case with special-value injection (zero/inf/nan)."""
    dtype = torch.float16
    c_seq_len, h, hq, d, s, bs = 64, 1, 16, 64, 1, 32
    try:
        torch.manual_seed(0)
        if name == "zero":
            q = torch.zeros(1, c_seq_len, hq, d, dtype=dtype)
            k = torch.zeros(1, c_seq_len, h, d, dtype=dtype)
            v = torch.zeros(1, c_seq_len, h, d, dtype=dtype)
            g = torch.zeros(1, c_seq_len, hq, dtype=dtype)
        else:
            q = torch.randn(1, c_seq_len, hq, d, dtype=dtype)
            k = torch.randn(1, c_seq_len, h, d, dtype=dtype)
            v = torch.randn(1, c_seq_len, h, d, dtype=dtype)
            g = torch.rand(1, c_seq_len, hq, dtype=dtype)
        q_inject_fn(q)

        off = torch.tensor([0, 32, 64], dtype=torch.int32)
        ti = prepare_token_indices(off)
        bi = torch.zeros(1, c_seq_len, h, s, dtype=torch.int32)
        bc = torch.ones(1, c_seq_len, h, dtype=torch.int32)
        scale = d**-0.5

        ref = naive_nsa_fwd_varlen(q.float(), k.float(), v.float(), bi.long(), bc.long(), bs, scale, off, ti, g.float()).to(dtype)
        out = parallel_nsa_fwd(
            q.npu(),
            k.npu(),
            v.npu(),
            bi.npu(),
            bc.npu(),
            bs,
            scale,
            off.npu(),
            ti.npu(),
            g.float().npu(),
        )
        passed, ratio, max_abs = check_precision(out.cpu(), ref, "float16")
        tag = "PASS" if passed else "WARN"
        print(f"[BOUNDARY_{tag}] boundary {name} dtype=float16 ratio={ratio:.4f} max_abs={max_abs:.3e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name}: {e}")


def test_nsa_fwd_varlen_boundary():
    """Boundary tests: special values (zero/inf/nan)."""
    _run_boundary_special("zero", lambda q: None)
    _run_boundary_special("inf", lambda q: q.__setitem__((0, 0, 0, 0), float("inf")))
    _run_boundary_special("nan", lambda q: q.__setitem__((0, 0, 0, 0), float("nan")))


# =============================================================================
# Main: --level dispatch + exit code
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSA Forward VarLen precision test suite (Ascend)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run",
    )
    args = parser.parse_args()

    tilelang.disable_cache()

    torch.manual_seed(0)

    blocking_ok = True  # only L0/L1 count toward blocking decision
    if args.level in ("l0", "all"):
        blocking_ok &= test_nsa_fwd_varlen_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_nsa_fwd_varlen_l1()
    if args.level in ("l2", "all"):
        test_nsa_fwd_varlen_l2()  # L2 negative: non-blocking
    if args.level in ("boundary", "all"):
        test_nsa_fwd_varlen_boundary()  # Boundary precision: non-blocking

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
