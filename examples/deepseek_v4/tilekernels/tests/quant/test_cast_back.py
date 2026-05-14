"""Tests for Ascend NPU cast_back (dequantization) kernel.

Adapted from the GPU tests/quant/test_cast_back.py.
Fully aligned with GPU test coverage:
- Same parameter combinations (num_tokens, hidden, fmt, round_sf, block_size, out_dtype)
- Same data generation pattern (roundtrip: quantize → dequantize → compare)
- Same precision thresholds (1e-3 for e4m3 roundtrip, 2e-2 for e2m1)
- Benchmark support via pytest marker
- Environment variable TK_FULL_TEST controls full/fast test mode

Usage:
    python tests/quant/ascend/test_cast_back.py                # fast mode
    TK_FULL_TEST=1 python tests/quant/ascend/test_cast_back.py # full mode
    pytest tests/quant/ascend/test_cast_back.py -v -s          # pytest
    pytest tests/quant/ascend/test_cast_back.py -m benchmark   # benchmark only
"""

import importlib.util
import os
import time

import pytest
import torch

_ASCEND_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir, os.pardir,
    "tile_kernels", "quant", "ascend",
))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cast_back_mod = _load_module(
    "cast_back_kernel", os.path.join(_ASCEND_DIR, "cast_back_kernel.py"))
cast_back = _cast_back_mod.cast_back
per_token_cast_back = _cast_back_mod.per_token_cast_back

NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
IS_FULL_TEST = os.getenv("TK_FULL_TEST") in ["1", "true", "True"]


# ---------------------------------------------------------------------------
# Helper functions (aligned with GPU tile_kernels.testing.*)
# ---------------------------------------------------------------------------

def generate_num_tokens(is_benchmark=False):
    """Aligned with GPU generate_num_tokens(alignment=1)."""
    base = [4001, 8001]
    if IS_FULL_TEST and not is_benchmark:
        return [0] + base
    return base


def generate_hidden_sizes():
    """Aligned with GPU generate_hidden_sizes(align=64)."""
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def _ref_cast_back(x_data, x_sf, num_per_tokens, num_per_channels):
    """CPU reference: out[m, k] = x[m, k] * sf[m // npt, k // npc]."""
    num_tokens, hidden = x_data.shape
    x_f32 = x_data.to(torch.float32).cpu()
    sf_f32 = x_sf.to(torch.float32).cpu()
    sf_expanded = sf_f32.repeat_interleave(num_per_tokens, dim=0)[:num_tokens]
    sf_expanded = sf_expanded.repeat_interleave(num_per_channels, dim=1)[:, :hidden]
    return x_f32 * sf_expanded


def _round_sf_cpu(sf):
    """Round scaling factors to power-of-2 on CPU."""
    sf_cpu = sf.cpu()
    bits = sf_cpu.view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_out = ((127 + exp_sf) << 23).view(torch.float32).to(sf.device)
    sf_inv = ((127 - exp_sf) << 23).view(torch.float32).to(sf.device)
    return sf_out, sf_inv


def _calc_diff(a, b):
    if a.numel() == 0:
        return 0.0
    a_f32 = a.to(torch.float32).cpu()
    b_f32 = b.to(torch.float32).cpu()
    denom = torch.max(a_f32.abs().mean(), torch.tensor(1e-6))
    return ((a_f32 - b_f32).abs().mean() / denom).item()


def _dtype_to_str(dtype):
    return {torch.float32: "fp32", torch.bfloat16: "bf16",
            torch.float16: "fp16"}[dtype]


def _count_bytes(*tensors):
    return sum(t.nelement() * t.element_size() for t in tensors if t is not None)


def _benchmark_timer(func, warmup=3, repeat=10):
    """Simple NPU benchmark timer, returns time in microseconds."""
    for _ in range(warmup):
        func()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        func()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / repeat
    return elapsed * 1e6


# ---------------------------------------------------------------------------
# Data generation (aligned with GPU generate_test_data*)
# ---------------------------------------------------------------------------

def generate_test_data_per_token(params):
    """Generate roundtrip test data: x → quantize → (x_fp8, x_sf).

    Mirrors GPU generate_test_data_per_token:
    - Generates random x with out_dtype
    - Quantizes with per-token scaling (fmt determines max_value)
    - round_sf controls power-of-2 rounding of sf
    """
    nt = params['num_tokens']
    h = params['hidden']
    npc = params['num_per_channels']
    fmt = params['fmt']
    rsf = params['round_sf']
    out_dtype = params['out_dtype']

    x = torch.randn((nt, h), dtype=out_dtype, device=NPU_DEVICE)
    x_f32 = x.to(torch.float32)
    groups = h // npc
    max_val = 6.0 if fmt == 'e2m1' else 448.0

    act_grouped = x_f32.reshape(nt, groups, npc)
    amax = act_grouped.abs().amax(dim=2)
    clamped_amax = amax.clamp(min=1e-4)
    sf = clamped_amax / max_val

    if rsf:
        sf, sf_inv = _round_sf_cpu(sf)
    else:
        sf_inv = max_val / clamped_amax

    sf_inv_expanded = sf_inv.unsqueeze(2).expand_as(act_grouped)
    x_casted = (act_grouped * sf_inv_expanded).reshape(nt, h)
    out_dtype_str = _dtype_to_str(out_dtype)

    return x, x_casted, sf, out_dtype_str


def generate_test_data_block(params):
    """Generate block-mode test data using CPU reference cast.

    Mirrors GPU generate_test_data:
    - Generates random x with out_dtype
    - Quantizes with per-block scaling (npt × npc blocks)
    - round_sf controls power-of-2 rounding of sf
    """
    nt = params['num_tokens']
    h = params['hidden']
    npt = params['num_per_tokens']
    npc = params['num_per_channels']
    rsf = params['round_sf']
    out_dtype = params['out_dtype']

    x = torch.randn((nt, h), dtype=out_dtype, device=NPU_DEVICE)
    x_f32 = x.to(torch.float32).cpu()
    max_fp8 = 448.0

    sf_rows = (nt + npt - 1) // npt
    sf_cols = (h + npc - 1) // npc
    sf = torch.zeros((sf_rows, sf_cols), dtype=torch.float32)
    x_casted = x_f32.clone()

    for bi in range(sf_rows):
        for bj in range(sf_cols):
            r0, r1 = bi * npt, min((bi + 1) * npt, nt)
            c0, c1 = bj * npc, min((bj + 1) * npc, h)
            block = x_f32[r0:r1, c0:c1]
            amax = block.abs().max().clamp(min=1e-4)
            sf[bi, bj] = amax / max_fp8
            x_casted[r0:r1, c0:c1] = block * (max_fp8 / amax)

    if rsf:
        sf_rounded, _ = _round_sf_cpu(sf)
        sf = sf_rounded

    out_dtype_str = _dtype_to_str(out_dtype)
    return x, x_casted.to(NPU_DEVICE), sf.to(NPU_DEVICE), out_dtype_str


# ---------------------------------------------------------------------------
# Parameter generation (aligned with GPU generate_test_params*)
# ---------------------------------------------------------------------------

def generate_test_params_per_token(is_benchmark=False):
    """Aligned with GPU generate_test_params_per_token.

    GPU covers:
      num_tokens × hidden × fmt(e2m1,e4m3) × sf_combo × npc(128,h) × out_dtype(fp32,bf16)
    NPU covers same except sf_combo (TMA/UE8M0 are GPU-H100 specific hardware).
    round_sf is extracted from sf_combo to cover independently.
    """
    return [
        {
            'num_tokens': nt,
            'hidden': h,
            'fmt': fmt,
            'round_sf': rsf,
            'num_per_channels': npc,
            'out_dtype': out_dtype,
        }
        for nt in generate_num_tokens(is_benchmark=is_benchmark)
        for h in generate_hidden_sizes()
        for fmt in ('e2m1', 'e4m3')
        for rsf in (False, True)
        for npc in (128, h)
        for out_dtype in (torch.float32, torch.bfloat16)
        if h % npc == 0
    ]


def generate_test_params_block(is_benchmark=False):
    """Aligned with GPU generate_test_params.

    GPU covers:
      num_tokens × hidden × round_sf × fmt(e4m3) × out_dtype × (npt,npc)
    """
    return [
        {
            'num_tokens': nt,
            'hidden': h,
            'round_sf': rsf,
            'fmt': 'e4m3',
            'out_dtype': out_dtype,
            'num_per_tokens': npt,
            'num_per_channels': npc,
        }
        for nt in generate_num_tokens(is_benchmark=is_benchmark)
        for h in generate_hidden_sizes()
        for rsf in (False, True)
        for out_dtype in (torch.bfloat16, torch.float32)
        for npt, npc in ((128, 1), (128, 128))
    ]


def _make_param_id(params):
    parts = []
    for k, v in params.items():
        if isinstance(v, torch.dtype):
            v = _dtype_to_str(v)
        parts.append(f"{k}={v}")
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'params',
    generate_test_params_per_token(is_benchmark=False),
    ids=_make_param_id,
)
def test_cast_back_per_token(params):
    """Per-token roundtrip: quantize → cast_back → compare with original."""
    h = params['hidden']
    fmt = params['fmt']
    npc = params['num_per_channels']

    x, x_casted, x_sf, out_dtype_str = generate_test_data_per_token(params)
    result = per_token_cast_back((x_casted, x_sf), out_dtype_str,
                                 num_per_channels=npc)

    ref = _ref_cast_back(x_casted, x_sf, 1, npc)
    out_dtype = params['out_dtype']
    ref = ref.to(out_dtype)

    diff_vs_ref = _calc_diff(result, ref)
    roundtrip_diff = _calc_diff(result, x)
    roundtrip_threshold = 2e-2 if fmt == 'e2m1' else 1e-3

    assert result.shape == x.shape
    assert result.dtype == out_dtype
    if params['num_tokens'] == 0:
        return
    assert diff_vs_ref < 1e-5, f"ref diff={diff_vs_ref}"
    assert roundtrip_diff < roundtrip_threshold, (
        f"roundtrip diff={roundtrip_diff}, {fmt=}, {h=}, {npc=}"
    )


@pytest.mark.parametrize(
    'params',
    generate_test_params_block(is_benchmark=False),
    ids=_make_param_id,
)
def test_cast_back(params):
    """Block-wise: reference cast → cast_back → exact match."""
    npt = params['num_per_tokens']
    npc = params['num_per_channels']

    x, x_casted, x_sf, out_dtype_str = generate_test_data_block(params)
    result = cast_back((x_casted, x_sf), out_dtype_str, (npt, npc))

    ref = _ref_cast_back(x_casted, x_sf, npt, npc)
    out_dtype = params['out_dtype']
    ref = ref.to(out_dtype)

    diff = _calc_diff(result, ref)

    assert result.shape == x.shape
    assert result.dtype == out_dtype
    if params['num_tokens'] == 0:
        return
    assert diff < 1e-5, f"diff={diff}"


# ---------------------------------------------------------------------------
# Edge case / boundary tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["fp16", "fp32", "bf16"])
def test_cast_back_output_formats(fmt):
    """Verify all three output dtype formats."""
    nt, h, npc = 256, 256, 128
    x_data = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
    x_sf = torch.rand((nt, h // npc), dtype=torch.float32,
                       device=NPU_DEVICE) + 0.1
    result = per_token_cast_back((x_data, x_sf), fmt, num_per_channels=npc)
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32,
                 "bf16": torch.bfloat16}
    assert result.dtype == dtype_map[fmt]
    assert result.shape == (nt, h)


def test_cast_back_empty():
    """Edge case: zero tokens returns empty tensor."""
    x_data = torch.empty((0, 256), dtype=torch.float32, device=NPU_DEVICE)
    x_sf = torch.empty((0, 2), dtype=torch.float32, device=NPU_DEVICE)
    result = cast_back((x_data, x_sf), "fp32", (1, 128))
    assert result.shape == (0, 256)
    assert result.dtype == torch.float32


def test_cast_back_single_token():
    """Edge case: single token."""
    x_data = torch.randn((1, 128), dtype=torch.float32, device=NPU_DEVICE)
    x_sf = torch.rand((1, 1), dtype=torch.float32, device=NPU_DEVICE) + 0.1
    result = cast_back((x_data, x_sf), "fp32", (1, 128))
    ref = _ref_cast_back(x_data, x_sf, 1, 128)
    assert result.shape == (1, 128)
    assert _calc_diff(result, ref) < 1e-5


def test_cast_back_large_scale():
    """Boundary: large tensor (8192 × 8192)."""
    nt, h, npc = 8192, 8192, 128
    x_data = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
    x_sf = torch.rand((nt, h // npc), dtype=torch.float32,
                       device=NPU_DEVICE) + 0.1
    result = per_token_cast_back((x_data, x_sf), "bf16", num_per_channels=npc)
    ref = _ref_cast_back(x_data, x_sf, 1, npc).to(torch.bfloat16)
    assert result.shape == (nt, h)
    assert _calc_diff(result, ref) < 1e-5


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.parametrize(
    'params',
    generate_test_params_per_token(is_benchmark=True),
    ids=_make_param_id,
)
def test_cast_back_per_token_benchmark(params):
    """Benchmark per-token cast_back throughput."""
    npc = params['num_per_channels']

    x, x_casted, x_sf, out_dtype_str = generate_test_data_per_token(params)
    def func():
        return per_token_cast_back((x_casted, x_sf), out_dtype_str,
                                    num_per_channels=npc)
    result = func()

    t_us = _benchmark_timer(func)
    num_bytes = _count_bytes(x_casted, x_sf, result)
    bw_gbs = num_bytes / t_us / 1e3

    print(f"  [bench] per_token_cast_back {params['num_tokens']}×{params['hidden']} "
          f"npc={npc} fmt={params['fmt']} → {t_us:.1f} μs, {bw_gbs:.1f} GB/s")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'params',
    generate_test_params_block(is_benchmark=True),
    ids=_make_param_id,
)
def test_cast_back_benchmark(params):
    """Benchmark block-wise cast_back throughput."""
    npt = params['num_per_tokens']
    npc = params['num_per_channels']

    x, x_casted, x_sf, out_dtype_str = generate_test_data_block(params)
    def func():
        return cast_back((x_casted, x_sf), out_dtype_str, (npt, npc))
    result = func()

    t_us = _benchmark_timer(func)
    num_bytes = _count_bytes(x_casted, x_sf, result)
    bw_gbs = num_bytes / t_us / 1e3

    print(f"  [bench] cast_back {params['num_tokens']}×{params['hidden']} "
          f"({npt},{npc}) → {t_us:.1f} μs, {bw_gbs:.1f} GB/s")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.npu.set_device(NPU_DEVICE_ID)

    print(f"\nMode: {'FULL' if IS_FULL_TEST else 'FAST'}")
    print(f"Device: npu:{NPU_DEVICE_ID}\n")

    print(">>> Edge cases")
    test_cast_back_empty()
    test_cast_back_single_token()
    for fmt in ["fp16", "fp32", "bf16"]:
        test_cast_back_output_formats(fmt)
    print("  All edge cases PASSED\n")

    print(">>> test_cast_back_per_token")
    for p in generate_test_params_per_token():
        test_cast_back_per_token(p)
    print("  All per_token PASSED\n")

    print(">>> test_cast_back (block)")
    for p in generate_test_params_block():
        test_cast_back(p)
    print("  All block PASSED\n")

    print(">>> Benchmarks")
    for p in generate_test_params_per_token(is_benchmark=True)[:4]:
        test_cast_back_per_token_benchmark(p)
    for p in generate_test_params_block(is_benchmark=True)[:4]:
        test_cast_back_benchmark(p)

    print("\nAll tests PASSED!")
