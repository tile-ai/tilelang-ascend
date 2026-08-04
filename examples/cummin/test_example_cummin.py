import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_cummin_example() -> ModuleType:
    source = Path(__file__).with_name("example_cummin.py")
    spec = importlib.util.spec_from_file_location("_cummin_example_for_test", source)
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


_CUMMIN_CASES = [
    (1, [1024, 1024], "float16", -1, [-1, 1], "S-float16-1M-aligned-dim=-1"),
    (2, [2048, 2048], "float32", -1, [-2, 2], "M-float32-4M-aligned-dim=-1"),
    (3, [4096, 4096], "bfloat16", -1, [-3, 3], "M-bfloat16-16M-aligned-dim=-1"),
    (4, [8192, 8192], "int32", -1, [-10000, 10000], "L-int32-67M-aligned-dim=-1"),
    (5, [16384, 16384], "float16", 0, [-100, 100], "L-float16-268M-aligned-dim=0"),
    (6, [8192, 8192], "float32", 1, [-1000, 1000], "L-float32-1G-aligned-dim=1"),
    (7, [1023, 1023], "bfloat16", -1, [-0.1, 0.1], "S-bfloat16-1M-unaligned-dim=-1"),
    (8, [1009, 1021], "float16", 0, [-1, 2], "S-float16-1M-prime-unaligned-dim=0"),
    (9, [1537, 769], "float32", -1, [-5, 10], "S-float32-1M-unaligned-dim=-1"),
    (10, [363, 367, 373], "bfloat16", 1, [-50, 100], "M-bfloat16-50M-3D-dim=1"),
    (11, [2049, 513], "float16", -1, [-65504, 65504], "S-float16-fp16-extreme-dim=-1"),
    (12, [3, 7, 13, 4001], "float32", -1, [-88, 88], "S-float32-4D-dim=-1"),
    (13, [1000003], "bfloat16", -1, [-float("inf"), float("inf")], "S-bfloat16-inf-1D-dim=-1"),
    (14, [11, 13, 17, 67, 67], "float16", 2, [float("nan"), float("nan")], "M-float16-nan-5D-dim=2"),
    (15, [3, 7, 11, 13, 1013], "int32", -1, [0, 0], "M-int32-zero-5D-dim=-1"),
    (16, [512, 2049], "float32", -1, [-0.5, 0.5], "S-float32-unaligned-dim=-1"),
    (17, [255, 8193], "bfloat16", 0, [-1, 3], "S-bfloat16-unaligned-dim=0"),
    (18, [4097, 511], "float16", -1, [-1000, 1000], "S-float16-unaligned-dim=-1"),
    (19, [2, 511, 2049], "float32", 1, [-0.2, 0.2], "S-float32-3D-dim=1"),
    (20, [4, 255, 2049], "bfloat16", -1, [-3, 6], "S-bfloat16-3D-dim=-1"),
]


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize(
    "case_id,shape,dtype,dim,value_range,note",
    _CUMMIN_CASES,
    ids=[f"case_{c[0]}" for c in _CUMMIN_CASES],
)
def test_cummin_accuracy(case_id, shape, dtype, dim, value_range, note):
    torch.manual_seed(0)
    example = _load_cummin_example()

    ok, msg = example._run_case_wrapped(case_id, shape, dtype, dim, value_range, note)
    assert ok, f"case_{case_id} ({note}) FAILED: {msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
