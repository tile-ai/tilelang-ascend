"""Tests for Ascend NPU cast_back_e5m6. Aligned with GPU test_cast_back_e5m6.py."""

import importlib.util
import os
import time

import pytest
import torch

_ASCEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "tile_kernels", "quant", "ascend"))


def _load_module(n, f):
    s = importlib.util.spec_from_file_location(n, f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_cb = _load_module("cast_back_e5m6_kernel", os.path.join(_ASCEND_DIR, "cast_back_e5m6_kernel.py"))
cast_back_e5m6 = _cb.cast_back_e5m6
_e5m6 = _load_module("per_token_cast_to_e5m6_kernel", os.path.join(_ASCEND_DIR, "per_token_cast_to_e5m6_kernel.py"))
per_token_cast_to_e5m6 = _e5m6.per_token_cast_to_e5m6

NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
IS_FULL_TEST = os.getenv("TK_FULL_TEST") in ["1", "true", "True"]


def generate_num_tokens(is_benchmark=False):
    base = [4001, 8001]
    if IS_FULL_TEST and not is_benchmark:
        return [0] + base
    return base


def generate_hidden_sizes():
    return [576, 2048, 2560, 3072, 4096, 6144, 7168]


def _calc_diff(a, b):
    if a.numel() == 0:
        return 0.0
    a_f32 = a.to(torch.float32).cpu()
    b_f32 = b.to(torch.float32).cpu()
    return ((a_f32 - b_f32).abs().mean() / torch.max(a_f32.abs().mean(), torch.tensor(1e-6))).item()


def _round_sf_cpu(sf):
    sf_cpu = sf.cpu()
    bits = sf_cpu.view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    return ((127 + exp_sf) << 23).view(torch.float32).to(sf.device), ((127 - exp_sf) << 23).view(torch.float32).to(sf.device)


def _benchmark_timer(func, warmup=3, repeat=10):
    for _ in range(warmup):
        func()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        func()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6


# GPU: num_tokens=gen(), hidden=gen(), npc=hidden, sf_combo x2, out_dtype=(bf16,f32)
def generate_params(is_benchmark=False):
    return [
        {"num_tokens": nt, "hidden": h, "out_fmt": fmt, "round_sf": rsf}
        for nt in generate_num_tokens(is_benchmark) for h in generate_hidden_sizes()
        for fmt in ["fp32", "bf16"] for rsf in [True, False]
    ]


def _pid(p):
    return "-".join(f"{k}={v}" for k, v in p.items())


@pytest.mark.parametrize("params", generate_params(), ids=_pid)
def test_cast_back_e5m6(params):
    nt, h, fmt, rsf = params["num_tokens"], params["hidden"], params["out_fmt"], params["round_sf"]
    x = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
    packed, sf = per_token_cast_to_e5m6(x, h, round_sf=rsf)
    result = cast_back_e5m6((packed, sf), fmt, (1, h))
    out_dtype = torch.bfloat16 if fmt == "bf16" else torch.float32
    assert result.dtype == out_dtype
    assert result.shape == (nt, h)
    diff = _calc_diff(result, x)
    assert diff < 5e-3, f"roundtrip diff={diff}"


def test_empty():
    x = torch.empty((0, 128), dtype=torch.float32, device=NPU_DEVICE)
    packed, sf = per_token_cast_to_e5m6(x, 128)
    result = cast_back_e5m6((packed, sf), 'fp32', (1, 128))
    assert result.shape == (0, 128)


@pytest.mark.benchmark
@pytest.mark.parametrize("params", generate_params(is_benchmark=True), ids=_pid)
def test_benchmark(params):
    nt, h = params["num_tokens"], params["hidden"]
    x = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
    packed, sf = per_token_cast_to_e5m6(x, h, round_sf=params["round_sf"])

    def func():
        return cast_back_e5m6((packed, sf), params["out_fmt"], (1, h))

    func()
    t = _benchmark_timer(func)
    print(f"  [bench] {nt}x{h} fmt={params['out_fmt']} -> {t:.1f}us")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.npu.set_device(NPU_DEVICE_ID)
    test_empty()
    print("Boundary PASSED")
    for p in generate_params():
        test_cast_back_e5m6(p)
    print("All PASSED!")
