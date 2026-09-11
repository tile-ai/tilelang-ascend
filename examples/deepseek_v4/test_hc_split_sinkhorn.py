import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        assert torch.equal(actual, golden), "integer output must match exactly"
        return
    atol, rtol, max_abs_limit, required_ratio = table[dtype]
    actual, golden = actual.float(), golden.float()
    special = ~torch.isfinite(golden)
    assert torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
    assert torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    finite = torch.isfinite(golden)
    if not finite.any():
        return
    errors = (actual[finite] - golden[finite]).abs()
    ratio = (errors <= atol + rtol * golden[finite].abs()).float().mean().item()
    max_abs = errors.max().item()
    assert ratio >= required_ratio and max_abs <= max_abs_limit, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"


def _load_hc_split_sinkhorn_example() -> ModuleType:
    source = Path(__file__).with_name("hc_split_sinkhorn.py")
    spec = importlib.util.spec_from_file_location("_hc_split_sinkhorn_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


def test_hc_split_sinkhorn_accuracy() -> None:
    example = _load_hc_split_sinkhorn_example()

    dtype = torch.float32
    batch = 1
    seq_len = 5
    hc_mult = 4
    sinkhorn_iters = 20
    eps = 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    n = batch * seq_len

    torch.manual_seed(42)
    mixes = torch.rand((n, mix_hc), dtype=dtype, device="npu")
    hc_scale = torch.rand(3, dtype=dtype, device="npu")
    hc_base = torch.rand(mix_hc, dtype=dtype, device="npu")

    pre = torch.empty((n, hc_mult), dtype=dtype, device="npu")
    post = torch.empty((n, hc_mult), dtype=dtype, device="npu")
    comb = torch.empty((n, hc_mult, hc_mult), dtype=dtype, device="npu")
    torch.npu.synchronize()

    kernel = example.hc_split_sinkhorn(hc=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)

    pre, post, comb = kernel(mixes, hc_scale, hc_base)
    pre_ref, post_ref, comb_ref = example.hc_split_sinkhorn_ref(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps,
    )
    torch.npu.synchronize()

    _check_precision(pre, pre_ref, "float32")
    _check_precision(post, post_ref, "float32")
    _check_precision(comb, comb_ref, "float32")
