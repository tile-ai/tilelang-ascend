import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_hadamard_example() -> ModuleType:
    source = Path(__file__).with_name("example_hadamard_transform.py")
    spec = importlib.util.spec_from_file_location("_hadamard_example_for_test", source)
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


@pytest.fixture(scope="module")
def hadamard_example():
    return _load_hadamard_example()


_CONFIGS = [
    (4, 2048, "float", 2048),
    (4, 4096, "float", 2048),
    (2, 1024, "float", 512),
]


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize("B, N, dtype, block_size", _CONFIGS)
def test_hadamard_accuracy(hadamard_example, B, N, dtype, block_size):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
    transform = hadamard_example.hadamard_transform_complete(B, N, dtype, block_size)
    x = torch.randn(B, N, dtype=torch_dtype).npu()
    torch.npu.synchronize()
    y = transform(x)
    y_ref = hadamard_example.ref_hadamard(x.cpu()).npu()
    rtol = 1e-2 if dtype in ["float16", "bfloat16"] else 1e-3
    atol = 1e-2 if dtype in ["float16", "bfloat16"] else 1e-3
    torch.testing.assert_close(y.cpu(), y_ref.cpu(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
