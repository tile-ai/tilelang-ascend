import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_sigmoidv2_slice_example() -> ModuleType:
    source = Path(__file__).with_name("sigmoidv2_slice.py")
    spec = importlib.util.spec_from_file_location("_sigmoidv2_slice_example_for_test", source)
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
def test_sigmoidv2_slice_run():
    _load_sigmoidv2_slice_example()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
