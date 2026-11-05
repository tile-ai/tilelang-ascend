import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_tail_add_example() -> ModuleType:
    source = Path(__file__).with_name("example_tail_add.py")
    spec = importlib.util.spec_from_file_location("_tail_add_example_for_test", source)
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
    "M,N,block_M,block_N,dtype",
    [
        (34, 130, 32, 32, "float"),
        (34, 130, 32, 32, "float16"),
        (100, 200, 64, 128, "float"),
    ],
)
def test_tail_add_accuracy(M, N, block_M, block_N, dtype):
    torch.manual_seed(0)
    example = _load_tail_add_example()

    func = example.tail_add(M, N, block_M, block_N, dtype=dtype)
    torch_dtype = torch.float32 if dtype == "float" else torch.float16
    a = torch.randn(M, N, dtype=torch_dtype).npu()
    b = torch.randn(M, N, dtype=torch_dtype).npu()
    c = func(a, b)
    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
