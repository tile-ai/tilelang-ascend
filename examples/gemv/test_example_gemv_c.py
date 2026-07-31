import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_gemv_c_example() -> ModuleType:
    source = Path(__file__).with_name("example_gemv_c.py")
    spec = importlib.util.spec_from_file_location("_gemv_c_example_for_test", source)
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


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize(
    "N,K,block_N,block_K,dtype",
    [
        (1024, 1024, 128, 128, "float16"),
        (1024, 1024, 128, 128, "float32"),
        (64, 64, 16, 16, "float16"),
    ],
)
def test_gemv_c_accuracy(N, K, block_N, block_K, dtype):
    torch.manual_seed(0)
    example = _load_gemv_c_example()
    example.check_case(N, K, block_N, block_K, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
