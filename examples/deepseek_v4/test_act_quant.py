import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
    import torch

    example = _load_act_quant_example()

    m = 128
    n = 1024

    torch.manual_seed(42)
    x_bf16 = torch.randn(m, n, dtype=torch.bfloat16, device="npu")

    kernel = example.act_quant_kernel_int8_optimized(n, block_M=16, block_N=n, round_scale=False)

    actual, scales = kernel(x_bf16)
    expected, expected_scales = example.validate_act_quant_kernel(x_bf16, m, n)
    torch.npu.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1)
    torch.testing.assert_close(scales.reshape(m), expected_scales.reshape(m), rtol=1e-2, atol=1e-2)
