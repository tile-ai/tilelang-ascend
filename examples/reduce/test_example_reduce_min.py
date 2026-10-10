import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_reduce_min_example() -> ModuleType:
    source = Path(__file__).with_name("example_reduce_min.py")
    spec = importlib.util.spec_from_file_location("_example_reduce_min_for_test", source)
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


@pytest.fixture(autouse=True)
def setup_random_seed():
    torch.manual_seed(0)
    yield


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize(
    "M,N,block_M,dtype",
    [
        (1024, 256, 128, "float"),
    ],
)
def test_reduce_min(M, N, block_M, dtype):
    example = _load_reduce_min_example()

    func = example.reduce_min(M, N, block_M, dtype=dtype)

    a = torch.randn(M, N).npu()
    torch.npu.synchronize()

    c = func(a)

    ref_c = torch.min(a, dim=-1).values

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
