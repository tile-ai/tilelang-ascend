"""Sparse FlashAttention operator layered tests: L0/L1/L2/Boundary + main(--level)."""

import argparse
import os
import sys

import tilelang
import torch

# Import kernel from sibling file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_flash_attention import sparse_flash_attention  # noqa: E402

# ========== Coverage declarations (for coverage_check.py) ==========
COVERAGE_CATEGORY = "Fusion"
COVERAGE_MANIFEST = {}
COVERAGE_NA = {}


# ========== Golden reference ==========
def golden_sparse_flash_attention(
    query,
    key,
    value,
    sparseIndices,
    scaleValue,
    inputLayout="BSND",
    is_causal=False,
):
    """PyTorch fp64 reference (cann-bench golden semantics): scatter top-k
    mask -> QK^T * scale -> causal mask -> softmax -> PV; all-masked rows -> 0.
    """
    q = query.double()
    k = key.double()
    v = value.double()
    si = sparseIndices.long()
    if inputLayout == "BSND":
        q = q.permute(0, 2, 1, 3)  # [B, N1, S1, Dk]
        k = k.permute(0, 2, 1, 3)  # [B, N2, S2, Dk]
        v = v.permute(0, 2, 1, 3)  # [B, N2, S2, Dv]
        si = si.permute(0, 2, 1, 3)  # [B, N2, S1, topK]
    B, N1, S1, Dk = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    Dv = v.shape[-1]
    G = N1 // N2
    mask = torch.zeros(B, N2, S1, S2, dtype=torch.bool)
    mask.scatter_(-1, si, True)
    if is_causal:
        s1_idx = torch.arange(S1).unsqueeze(-1)
        s2_idx = torch.arange(S2).unsqueeze(0)
        mask = mask & (s2_idx <= s1_idx + (S2 - S1))
    q_g = q.reshape(B, N2, G, S1, Dk)
    scores = torch.einsum("bngsd,bnkd->bngsk", q_g, k) * scaleValue
    scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    scores = scores - scores_max
    scores.exp_()
    scores = torch.where(all_masked, torch.zeros_like(scores), scores / scores.sum(dim=-1, keepdim=True))
    out = torch.einsum("bngsk,bnkd->bngsd", scores, v).reshape(B, N1, S1, Dv)
    if inputLayout == "BSND":
        return out.permute(0, 2, 1, 3).contiguous()
    return out.contiguous()


# ========== Precision standard (mixed tolerance, by dtype) ==========
def get_precision(dtype_str):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    return fp_table.get(dtype_str, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype_str):
    """Mixed tolerance dual-gate check: return (passed, matched_ratio, max_abs_error).

    Checks inf/nan position consistency, then compares finite values with
    mixed tolerance (atol + rtol*|golden|, matched_ratio, max_abs_error_limit).
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype_str)
    a, g = actual.detach().cpu().float(), golden.detach().cpu().float()

    if not torch.equal(torch.isinf(a), torch.isinf(g)):
        return False, 0.0, float("inf")
    if not torch.equal(torch.isnan(a), torch.isnan(g)):
        return False, 0.0, float("inf")

    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Input generation ==========
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def make_input(B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, seed=42):
    """Random inputs honoring the operator contract:
    value == key[..., :Dv] prefix + distinct topK indices in [0, S2)."""
    dt = DTYPE_MAP[dtype_str]
    g = torch.Generator().manual_seed(seed)
    q = (torch.rand(B, S1, N1, Dk, generator=g) * 2 - 1).to(dt)
    k = (torch.rand(B, S2, N2, Dk, generator=g) * 2 - 1).to(dt)
    v = k[..., :Dv].clone()
    sel = torch.rand(B * S1 * N2, S2, generator=g).argsort(dim=1)[:, :topK]
    si = sel.reshape(B, S1, N2, topK).to(torch.int32)
    if layout == "BNSD":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        si = si.permute(0, 2, 1, 3).contiguous()
    return q, k, v, si


def run_case(name, B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, is_causal=False, seed=42):
    """Run a single case vs the fp64 golden and print the mixed-tolerance result."""
    scale = 1.0 / (Dk**0.5)
    q, k, v, si = make_input(B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, seed=seed)
    out = sparse_flash_attention(
        query=q.npu(),
        key=k.npu(),
        value=v.npu(),
        sparseIndices=si.npu(),
        scaleValue=scale,
        inputLayout=layout,
        is_causal=is_causal,
    )
    torch.npu.synchronize()
    golden = golden_sparse_flash_attention(q, k, v, si, scale, layout, is_causal)
    # golden truncated to the output dtype (cann-bench compare semantics)
    passed, ratio, max_abs = check_precision(out, golden.to(out.dtype), dtype_str)
    tag = "PASS" if passed else "FAIL"
    print(
        f"[PRECISION_{tag}] {name} B={B} S1={S1} S2={S2} N1={N1} N2={N2} "
        f"Dk={Dk} Dv={Dv} topK={topK} layout={layout} causal={is_causal} "
        f"dtype={dtype_str} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
    )
    return passed


# ========== L0 tests: threshold (three wrapper routes + dtype) ==========
def test_sparse_flash_attention_l0():
    """L0 threshold tests: one case per wrapper route (v3 / dense / rev1) + bf16."""
    test_configs = [
        # (name, B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str)
        # v3: topK%256==0, Dv==dim_base, amp<16 (decode shape)
        ("l0_v3_decode_fp16", 16, 1, 1024, 32, 8, 128, 128, 512, "BSND", "float16"),
        # dense: amp = S1*topK/S2 >= 16 (falls back internally on aclnn-less SoCs)
        ("l0_dense_reuse_fp16", 2, 256, 1024, 32, 8, 128, 128, 512, "BSND", "float16"),
        # rev1: topK not divisible by 256 (fallback route)
        ("l0_rev1_fallback_fp16", 1, 16, 2048, 128, 8, 128, 128, 384, "BSND", "float16"),
        # dtype dimension: v3 route in bfloat16
        ("l0_v3_decode_bf16", 16, 1, 1024, 32, 8, 128, 128, 512, "BSND", "bfloat16"),
    ]

    ok = True
    for cfg in test_configs:
        name, dims = cfg[0], cfg[1:]
        try:
            ok &= run_case(name, *dims)
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {e}")
            ok = False
    assert ok, "L0 precision tests failed"


# ========== L1 tests: functional (layout / causal / dims variants) ==========
def test_sparse_flash_attention_l1():
    """L1 functional tests: BNSD layout, causal (v3 + dense + all-masked), Dv<Dk,
    Dk tail, G=1, small topK, odd G."""
    test_configs = [
        # (name, B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, is_causal)
        ("l1_bnsd_layout", 8, 1, 1024, 32, 8, 128, 128, 512, "BNSD", "float16", False),
        ("l1_causal_v3", 2, 16, 512, 32, 8, 128, 128, 256, "BSND", "float16", True),
        ("l1_causal_dense", 2, 256, 1024, 32, 8, 128, 128, 512, "BSND", "float16", True),
        ("l1_dv_less_dk", 1, 16, 1024, 32, 8, 128, 64, 384, "BSND", "float16", False),
        ("l1_dk_tail192", 2, 8, 1024, 32, 8, 192, 128, 512, "BSND", "float16", False),
        ("l1_mha_g1", 2, 64, 2048, 16, 16, 128, 128, 256, "BSND", "float16", False),
        ("l1_topk64", 1, 16, 2048, 128, 8, 128, 128, 64, "BSND", "float16", False),
        # odd G routed to rev1: H_per_block rounds up to even, the tail head
        # must still be processed (regression: N1=17, N2=1 used to drop head 16)
        ("l1_odd_g_rev1", 1, 4, 128, 17, 1, 128, 128, 64, "BSND", "float16", False),
        # S1 > S2 + causal: rows s < S1-S2 are fully masked and must output 0.
        # S2 % 256 != 0 keeps both off the dense route: topK=256 -> v3,
        # topK=128 -> rev1 (covers -inf masking on both gather paths).
        ("l1_all_masked_v3", 1, 512, 384, 8, 8, 128, 128, 256, "BSND", "float16", True),
        ("l1_all_masked_rev1", 1, 512, 384, 8, 8, 128, 128, 128, "BSND", "float16", True),
    ]

    ok = True
    for cfg in test_configs:
        name, dims = cfg[0], cfg[1:]
        try:
            ok &= run_case(name, *dims)
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {e}")
            ok = False

    # Regression (bitmap cache): the same sparseIndices tensor reused across
    # two calls with different S2 must not hit a stale cached bitmap - the
    # cache key has to distinguish S2, otherwise the second call reads a
    # wrong-shaped mask.
    B, S1, N1, N2, Dk, Dv, topK = 2, 256, 32, 8, 128, 128, 512
    scale = 1.0 / (Dk**0.5)
    q, k_small, v_small, si = make_input(B, S1, 512, N1, N2, Dk, Dv, topK, "BSND", "float16", seed=11)
    gen = torch.Generator().manual_seed(12)
    k_big = (torch.rand(B, 1024, N2, Dk, generator=gen) * 2 - 1).to(torch.float16)
    v_big = k_big[..., :Dv].clone()
    # Materialize the loop-invariant device tensors once, outside the loop.
    # The bitmap cache is keyed on sparseIndices.data_ptr(): calling si.npu()
    # inside the loop allocates a fresh device tensor (new data_ptr) on every
    # iteration, so the two calls would never share a cache entry and the
    # S2-in-key regression below would silently not be exercised.
    q_npu, si_npu = q.npu(), si.npu()
    for name, k, v, s2 in (
        ("l1_bitmap_cache_s2_a", k_small, v_small, 512),
        ("l1_bitmap_cache_s2_b", k_big, v_big, 1024),
    ):
        try:
            out = sparse_flash_attention(q_npu, k.npu(), v.npu(), si_npu, scale, "BSND", False)
            torch.npu.synchronize()
            golden = golden_sparse_flash_attention(q, k, v, si, scale, "BSND", False)
            passed, ratio, max_abs = check_precision(out, golden.to(out.dtype), "float16")
            tag = "PASS" if passed else "FAIL"
            print(
                f"[PRECISION_{tag}] {name} B={B} S1={S1} S2={s2} N1={N1} N2={N2} "
                f"Dk={Dk} Dv={Dv} topK={topK} layout=BSND causal=False "
                f"dtype=float16 matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
            )
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] {name}: {type(e).__name__}: {e}")
            ok = False

    assert ok, "L1 functional tests failed"


# ========== L2 tests: negative (invalid inputs must be rejected) ==========
def test_sparse_flash_attention_l2():
    """L2 negative tests: invalid layout / N1%N2 / Dv>Dk / topK>S2 must raise."""

    def _expect_reject(desc, fn):
        try:
            fn()
            print(f"[L2_FAIL] {desc}: silently accepted (should have raised)")
            return False
        except (ValueError, AssertionError, RuntimeError):
            print(f"[L2_PASS] {desc}: correctly rejected")
            return True

    q, k, v, si = make_input(1, 8, 256, 8, 8, 128, 128, 256, "BSND", "float16")
    q_npu, k_npu, v_npu, si_npu = q.npu(), k.npu(), v.npu(), si.npu()
    scale = 1.0 / (128**0.5)

    ok = True
    ok &= _expect_reject(
        "layout_invalid",
        lambda: sparse_flash_attention(q_npu, k_npu, v_npu, si_npu, scale, "BSHD", False),
    )

    q2, k2, v2, si2 = make_input(1, 8, 256, 8, 3, 128, 128, 256, "BSND", "float16")
    ok &= _expect_reject(
        "n1_not_divisible_by_n2",
        lambda: sparse_flash_attention(q2.npu(), k2.npu(), v2.npu(), si2.npu(), scale, "BSND", False),
    )

    v_wide = torch.randn(1, 256, 8, 256, dtype=torch.float16).npu()
    ok &= _expect_reject(
        "dv_greater_than_dk",
        lambda: sparse_flash_attention(q_npu, k_npu, v_wide, si_npu, scale, "BSND", False),
    )

    si_pad = torch.cat([si, si], dim=-1).npu()
    ok &= _expect_reject(
        "topk_greater_than_s2",
        lambda: sparse_flash_attention(q_npu, k_npu, v_npu, si_pad, scale, "BSND", False),
    )

    assert ok, "L2 negative tests failed"


# ========== Boundary tests: special values ==========
def test_sparse_flash_attention_boundary():
    """Boundary special value tests: topK==S2, causal all-masked rows, zero query, large amplitude."""

    def _run_special(name, B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, is_causal, special):
        scale = 1.0 / (Dk**0.5)
        q, k, v, si = make_input(B, S1, S2, N1, N2, Dk, Dv, topK, layout, dtype_str, seed=7)
        if special == "zero_q":
            q = torch.zeros_like(q)
        elif special == "large":
            q = (q.float() * 8.0).to(q.dtype)
            k = (k.float() * 8.0).to(k.dtype)
            v = k[..., :Dv].clone()  # keep the value == key prefix contract
        out = sparse_flash_attention(q.npu(), k.npu(), v.npu(), si.npu(), scale, layout, is_causal)
        torch.npu.synchronize()
        golden = golden_sparse_flash_attention(q, k, v, si, scale, layout, is_causal)
        passed, ratio, max_abs = check_precision(out, golden.to(out.dtype), dtype_str)
        tag = "PASS" if passed else "FAIL"
        print(f"[BOUNDARY_{tag}] {name} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        return passed

    ok = True
    ok &= _run_special("topk_equals_s2", 1, 8, 256, 8, 8, 128, 128, 256, "BSND", "float16", False, None)
    ok &= _run_special("causal_all_masked_rows", 1, 512, 256, 8, 8, 128, 128, 256, "BSND", "float16", True, None)
    ok &= _run_special("zero_query_uniform", 1, 8, 512, 8, 8, 128, 128, 256, "BSND", "float16", False, "zero_q")
    ok &= _run_special("large_amplitude", 1, 8, 512, 8, 8, 128, 128, 256, "BSND", "float16", False, "large")
    assert ok, "boundary special-value tests failed"


# ========== Main: --level dispatch + exit code ==========
def _run_level(fn):
    """Run one level; an AssertionError marks the level failed (pytest runs the
    test functions directly and fails on the same asserts)."""
    try:
        fn()
        return True
    except AssertionError as e:
        print(f"[LEVEL_FAIL] {fn.__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # Disable compile cache to avoid stale artifacts
    torch.manual_seed(0)

    ok = True
    if args.level in ("l0", "all"):
        ok &= _run_level(test_sparse_flash_attention_l0)
    if args.level in ("l1", "all"):
        ok &= _run_level(test_sparse_flash_attention_l1)
    if args.level in ("l2", "all"):
        ok &= _run_level(test_sparse_flash_attention_l2)
    if args.level in ("boundary", "all"):
        ok &= _run_level(test_sparse_flash_attention_boundary)

    if ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
