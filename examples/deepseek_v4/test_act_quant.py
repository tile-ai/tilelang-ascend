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
        if dtype == "int8":
            max_abs = (actual.to(torch.int64) - golden.to(torch.int64)).abs().max().item()
            assert max_abs <= 1, f"integer max_abs_error={max_abs} exceeds 1"
            return
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


def _load_act_quant_example() -> ModuleType:
    source = Path(__file__).with_name("act_quant.py")
    spec = importlib.util.spec_from_file_location("_act_quant_example_for_test", source)
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


def test_act_quant_accuracy() -> None:
    example = _load_act_quant_example()

    m = 128
    n = 1024

    torch.manual_seed(42)
    x_bf16 = torch.randn(m, n, dtype=torch.bfloat16, device="npu")

    kernel = example.act_quant_kernel_int8_optimized(n, block_M=16, block_N=n, round_scale=False)

    actual, scales = kernel(x_bf16)
    expected, expected_scales = example.validate_act_quant_kernel(x_bf16, m, n)
    torch.npu.synchronize()

    _check_precision(actual, expected, "int8")
    _check_precision(scales.reshape(m), expected_scales.reshape(m), "float32")
