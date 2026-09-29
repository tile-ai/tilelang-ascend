import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_unsorted_segment_sum_example() -> ModuleType:
    source = Path(__file__).with_name("unsorted_segment_sum.py")
    spec = importlib.util.spec_from_file_location("_unsorted_segment_sum_example_for_test", source)
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
    "N,D,num_segments,dtype_str",
    [
        (473, 128, 32, "float16"),
        (97, 512, 8, "float16"),
        (512, 512, 32, "float16"),
        (183, 128, 16, "bfloat16"),
        (25, 256, 4, "bfloat16"),
        (256, 128, 16, "bfloat16"),
    ],
)
def test_unsorted_segment_sum_accuracy(N, D, num_segments, dtype_str):
    example = _load_unsorted_segment_sum_example()
    example._test(N, D, num_segments, dtype_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
