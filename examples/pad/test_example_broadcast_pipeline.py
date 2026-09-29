import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_broadcast_pipeline_example() -> ModuleType:
    source = Path(__file__).with_name("example_broadcast_pipeline.py")
    spec = importlib.util.spec_from_file_location("_broadcast_pipeline_example_for_test", source)
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
def test_broadcast_pipeline_accuracy():
    M = 1024
    N = 256
    block_M = 128
    sub_M = 64

    torch.manual_seed(0)
    example = _load_broadcast_pipeline_example()

    func = example.broadcast_pipeline(M, N, block_M, sub_M)

    a = torch.randn(1, N).npu()
    torch.npu.synchronize()

    c = func(a)
    ref_c = a.expand(M, N)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
