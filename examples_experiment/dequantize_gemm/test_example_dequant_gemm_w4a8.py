import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


DEQUANT_GEMM_W4A8_CASES = [
    pytest.param(64, 64, 256, id="M64_N64_K256"),
    pytest.param(128, 128, 512, id="M128_N128_K512"),
    pytest.param(256, 256, 1024, id="M256_N256_K1024"),
]


def _load_dequant_gemm_w4a8_example() -> ModuleType:
    source = Path(__file__).with_name("example_dequant_gemm_w4a8.py")
    spec = importlib.util.spec_from_file_location("_dequant_gemm_w4a8_example_for_test", source)
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


@pytest.mark.parametrize(("m", "n", "k"), DEQUANT_GEMM_W4A8_CASES)
def test_dequant_gemm_w4a8_accuracy(m: int, n: int, k: int) -> None:
    import torch

    example = _load_dequant_gemm_w4a8_example()

    example.tilelang.disable_cache()
    torch.manual_seed(42)
    example.test(m, n, k)
