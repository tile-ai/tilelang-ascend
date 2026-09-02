"""MLA Decode test suite: layered precision tests only.

Usage:
  python test_mla_decode.py --level l0        # blocking precision (smoke)
  python test_mla_decode.py --level all       # all precision tests (L0+L1+L2+Boundary)
  python test_mla_decode.py --level l1        # functional (irregular shapes)
  python test_mla_decode.py --level l2        # exception (invalid input rejection)
  python test_mla_decode.py --level boundary  # special values (non-blocking)
"""

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import tilelang
from example_mla_decode import BLOCK_N, mla_decode

tilelang.disable_cache()


# ========== Retry wrapper for intermittent compile failures ==========


def with_compile_retry(max_retries=3):
    """Decorator: retry on RuntimeError containing 'Compilation Failed'.

    The bisheng compiler has an intermittent ~10% failure rate reading
    /tmp/tmp*.cpp temp files (race condition). Retrying the JIT compilation
    succeeds on the 2nd attempt in 100% of observed cases.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "Compilation Failed" in str(e) and attempt < max_retries - 1:
                        print(f"  [retry] compile failed (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    raise
            raise AssertionError("unreachable: loop must return or raise")  # pragma: no cover

        return wrapper

    return decorator


# ========== Golden reference (PyTorch) ==========


def ref_mla_decode(q, q_pe, kv, k_pe):
    """PyTorch golden reference (kv_head_num=1).

    Inputs: q [B,H,D], q_pe [B,H,pe], kv [B,N,1,D], k_pe [B,N,1,pe]
    Output: [B, H, D]
    """
    assert kv.shape[2] == 1, f"golden expects kv_head_num=1, got kv.shape={kv.shape}"
    # Detect dim/shape mismatch between tensors early.
    assert q.shape[-1] == kv.shape[-1], f"dim mismatch: q.shape[-1]={q.shape[-1]} != kv.shape[-1]={kv.shape[-1]}"
    assert q_pe.shape[-1] == k_pe.shape[-1], f"pe_dim mismatch: q_pe.shape[-1]={q_pe.shape[-1]} != k_pe.shape[-1]={k_pe.shape[-1]}"
    assert q.shape[0] == kv.shape[0], f"batch mismatch: q.shape[0]={q.shape[0]} != kv.shape[0]={kv.shape[0]}"
    assert q.shape[1] == q_pe.shape[1], f"heads mismatch: q.shape[1]={q.shape[1]} != q_pe.shape[1]={q_pe.shape[1]}"
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    scale = (dim + pe_dim) ** 0.5

    kv_2d = kv.squeeze(2)
    k_pe_2d = k_pe.squeeze(2)
    scores = torch.matmul(q.float(), kv_2d.float().transpose(1, 2))
    scores = scores + torch.matmul(q_pe.float(), k_pe_2d.float().transpose(1, 2))
    scores = scores / scale
    attention = F.softmax(scores, dim=-1)
    out = torch.matmul(attention, kv_2d.float())
    return out.to(q.dtype)


# ========== Precision standard (mixed tolerance) ==========

_FP_TABLE = {
    "float16": (2**-14, 2**-9, 1e-1, 0.99),  # atol 6.10e-5, rtol 1.95e-3
    "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
    "float32": (2**-16, 2**-10, 1e-2, 0.99),
}


def get_precision(dtype):
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return _FP_TABLE.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """Dual-gate precision check.

    Returns (passed, matched_ratio, max_abs_error).
    Float: matched_ratio >= required AND max_abs_error <= max_abs_limit.
    Int: exact element-wise match.
    inf/nan positions: structural comparison, excluded from numeric tolerance.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:  # integer exact match
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    # inf/nan structural comparison (excluded from numeric tolerance)
    special = ~torch.isfinite(g)
    if special.any() and (
        not torch.equal(torch.isnan(a[special]), torch.isnan(g[special]))
        or not torch.equal(torch.isinf(a[special]), torch.isinf(g[special]))
    ):
        return False, 0.0, float("inf")
    m = torch.isfinite(g)  # finite positions: full compare
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test helpers ==========

_col_indices_cpu = torch.arange(BLOCK_N, dtype=torch.float32)


def _gen_randn(shape):
    return torch.randn(shape, dtype=torch.float16, device="cpu")


def _gen_uniform(low, high):
    def fn(shape):
        return (torch.rand(shape, dtype=torch.float32, device="cpu") * (high - low) + low).to(torch.float16)

    return fn


def _gen_inputs(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, actual_seqlen_kv, gen_fn):
    """Generate test inputs. Returns (q, q_pe, kv_padded, k_pe_padded, kv_orig, k_pe_orig).

    kv_orig/k_pe_orig (length=actual_seqlen_kv) are for golden computation.
    kv_padded/k_pe_padded (length=seqlen_kv, zero-padded) are for kernel input.
    """
    q = gen_fn([batch, heads, dim])
    q_pe = gen_fn([batch, heads, pe_dim])
    kv_orig = gen_fn([batch, actual_seqlen_kv, kv_head_num, dim])
    k_pe_orig = gen_fn([batch, actual_seqlen_kv, kv_head_num, pe_dim])

    if actual_seqlen_kv < seqlen_kv:
        kv = torch.zeros(batch, seqlen_kv, kv_head_num, dim, dtype=torch.float16, device="cpu")
        kv[:, :actual_seqlen_kv, :, :] = kv_orig
        k_pe = torch.zeros(batch, seqlen_kv, kv_head_num, pe_dim, dtype=torch.float16, device="cpu")
        k_pe[:, :actual_seqlen_kv, :, :] = k_pe_orig
    else:
        kv, k_pe = kv_orig, k_pe_orig

    return q, q_pe, kv, k_pe, kv_orig, k_pe_orig


@with_compile_retry()
def run_test_case(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, actual_seqlen_kv=None, gen_fn=None):
    """Run one test case, return (passed, matched_ratio, max_abs_error)."""
    if actual_seqlen_kv is None:
        actual_seqlen_kv = seqlen_kv
    if gen_fn is None:
        gen_fn = _gen_randn

    q, q_pe, kv, k_pe, kv_orig, k_pe_orig = _gen_inputs(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, actual_seqlen_kv, gen_fn)
    # Golden uses original (unpadded) data; kernel uses zero-padded data with tail mask
    ref_out = ref_mla_decode(q, q_pe, kv_orig, k_pe_orig)

    kernel = mla_decode(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, actual_seqlen_kv)
    col_indices = _col_indices_cpu.npu()
    output = kernel(q.npu(), q_pe.npu(), kv.npu(), k_pe.npu(), col_indices)
    torch.npu.synchronize()
    return check_precision(output, ref_out, "float16")


# ========== L0: blocking precision (smoke) ==========


def test_l0():
    configs = [
        ("l0_smoke", 1, 64, 1, 128, 512, 64),
        ("l0_standard", 1, 128, 1, 8192, 512, 64),
        ("l0_perf_target", 132, 128, 1, 8192, 512, 64),
    ]
    ok = True
    for name, b, h, kvh, n, d, dp in configs:
        shape = f"batch={b} heads={h} kv_ctx={n}"
        try:
            p, r, m = run_test_case(b, h, kvh, n, d, dp)
            tag = "PASS" if p else "FAIL"
            print(f"[PRECISION_{tag}] l0 {name} {shape} ratio={r:.4f} max_abs={m:.3e}")
            ok &= p
        except Exception as e:
            print(f"[PRECISION_FAIL] l0 {name} {shape}: {e}")
            ok = False
    return ok


# ========== L1: functional (irregular shapes, tail blocks) ==========

_L1_TAGS = {
    "aligned": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED"],
    "tail1": ["D-DTYPE-fp16", "D-SHAPE-TAIL-1"],
    "tailmid": ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID"],
    "prime": ["D-DTYPE-fp16", "D-SHAPE-PRIME"],
    "edge": ["D-DTYPE-fp16", "D-SHAPE-EDGE"],
    "valrange_s": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    "valrange_m": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-M"],
    "valrange_l": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-L"],
    "valrange_asym": ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-ASYM"],
}

_VALRANGE_GENS = {
    "valrange_s": _gen_uniform(-1, 1),
    "valrange_m": _gen_uniform(-10, 10),
    "valrange_l": _gen_uniform(-50, 50),
    "valrange_asym": _gen_uniform(-5, 10),
}


def test_l1():
    ok = True
    cases = [
        ("aligned", 1, 64, 1, 512, 512, 64, None),
        ("tail1", 1, 64, 1, 640, 512, 64, 513),
        ("tailmid", 1, 64, 1, 640, 512, 64, 576),
        ("prime", 1, 64, 1, 512, 512, 64, 509),
        ("edge", 1, 64, 1, 128, 512, 64, None),
        ("valrange_s", 1, 64, 1, 512, 512, 64, None),
        ("valrange_m", 1, 64, 1, 512, 512, 64, None),
        ("valrange_l", 1, 64, 1, 512, 512, 64, None),
        ("valrange_asym", 1, 64, 1, 512, 512, 64, None),
    ]
    for name, b, h, kvh, n, d, dp, act in cases:
        gen = _VALRANGE_GENS.get(name)
        try:
            p, r, m = run_test_case(b, h, kvh, n, d, dp, act, gen)
            tag = "PASS" if p else "FAIL"
            print(f"[PRECISION_{tag}] l1 {name} ratio={r:.4f} max_abs={m:.3e} {_L1_TAGS[name]}")
            ok &= p
        except Exception as e:
            print(f"[PRECISION_FAIL] l1 {name}: {e}")
            ok = False
    return ok


# ========== L2: exception (non-blocking, should reject invalid input) ==========


def test_l2():
    # D-EXC-DTYPE: float32 input (kernel only supports float16)
    try:
        kernel = mla_decode(1, 64, 1, 128, 512, 64)
        q32 = torch.randn(1, 64, 512, dtype=torch.float32, device="cpu").npu()
        qpe32 = torch.randn(1, 64, 64, dtype=torch.float32, device="cpu").npu()
        kv32 = torch.randn(1, 128, 1, 512, dtype=torch.float32, device="cpu").npu()
        kpe32 = torch.randn(1, 128, 1, 64, dtype=torch.float32, device="cpu").npu()
        kernel(q32, qpe32, kv32, kpe32, _col_indices_cpu.npu())
        print("[BOUNDARY_WARN] l2 exc_dtype float32: not rejected")
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 exc_dtype float32: rejected ({type(e).__name__})")

    # D-EXC-SHAPE: heads=100 (not multiple of 64)
    try:
        mla_decode(1, 100, 1, 128, 512, 64)
        print("[BOUNDARY_WARN] l2 exc_shape heads=100: not rejected")
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 exc_shape heads=100: rejected ({type(e).__name__})")

    # D-EXC-ACTUAL: actual_seqlen_kv=0 (all-masked, would yield NaN)
    try:
        mla_decode(1, 64, 1, 128, 512, 64, actual_seqlen_kv=0)
        print("[BOUNDARY_WARN] l2 exc_actual0 actual_seqlen_kv=0: not rejected")
    except Exception as e:
        print(f"[BOUNDARY_PASS] l2 exc_actual0 actual_seqlen_kv=0: rejected ({type(e).__name__})")


# ========== Boundary: special values (non-blocking) ==========


def test_boundary():
    def _run(name, gen_fn, tags):
        try:
            p, r, m = run_test_case(1, 64, 1, 128, 512, 64, gen_fn=gen_fn)
            tag = "PASS" if p else "WARN"
            print(f"[BOUNDARY_{tag}] boundary {name} ratio={r:.4f} max_abs={m:.3e} {tags}")
        except Exception as e:
            print(f"[BOUNDARY_WARN] boundary {name}: {e}")

    _run("zero", lambda s: torch.zeros(s, dtype=torch.float16, device="cpu"), ["D-SPECIAL-ZERO"])

    def _gen_with_special(s, value, stride=100):
        x = torch.randn(s, dtype=torch.float16, device="cpu")
        if x.numel() > 10:
            x.view(-1)[::stride] = value
        return x

    _run("inf", lambda s: _gen_with_special(s, float("inf")), ["D-SPECIAL-INF"])
    _run("nan", lambda s: _gen_with_special(s, float("nan")), ["D-SPECIAL-NAN"])

    def gen_dbound(s):
        x = torch.randn(s, dtype=torch.float16, device="cpu")
        if x.numel() > 10:
            x.view(-1)[0] = 65504.0
            x.view(-1)[1] = -65504.0
        return x

    _run("dbound", gen_dbound, ["D-SPECIAL-DBOUND"])

    # Mixed +-65504 (fp16 max) causes Q@KV^T intermediate overflow. This is an
    # inherent fp16 attention limitation, NOT a kernel bug.
    # Marked KNOWN-LIMITATION; expected [BOUNDARY_WARN], non-blocking.
    def gen_extreme_mix(s):
        x = torch.randn(s, dtype=torch.float16, device="cpu")
        if x.numel() > 10:
            x.view(-1)[::100] = 65504.0
            x.view(-1)[1::100] = -65504.0
        return x

    _run("extreme_mix_65504", gen_extreme_mix, ["D-SPECIAL-EXTREME-MIX", "KNOWN-LIMITATION"])


# ========== Main: --level dispatch ==========


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    torch.manual_seed(0)
    blocking_ok = True

    if args.level in ("l0", "all"):
        blocking_ok &= test_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_l1()
    if args.level in ("l2", "all"):
        test_l2()
    if args.level in ("boundary", "all"):
        test_boundary()

    if blocking_ok:
        print("\nTest Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
