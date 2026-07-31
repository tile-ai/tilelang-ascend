import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_elementwise_add_example() -> ModuleType:
    source = Path(__file__).with_name("elementwise_add.py")
    spec = importlib.util.spec_from_file_location("_elementwise_add_for_test", source)
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
def test_elementwise_add_accuracy():
    module = _load_elementwise_add_example()

    M, N = module.M, module.N
    func = module.func

    torch.manual_seed(0)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()
    torch.npu.synchronize()

    c = func(a, b)

    ref_c = a + b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
