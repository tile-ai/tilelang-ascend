import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROPE_MASK_BWD_LAYOUT_CASES = [
    pytest.param("tnd", id="tnd"),
    pytest.param("bsnd", id="bsnd"),
]

ROPE_MASK_BWD_VARIANT_CASES = [
    pytest.param(dtype, rotary_mode, id=f"{dtype}_{rotary_mode}")
    for rotary_mode in ["interleave", "half"]
    for dtype in ["float16", "bfloat16", "float"]
]


def _load_rope_mask_bwd_example() -> ModuleType:
    source = Path(__file__).with_name("rope_mask_bwd.py")
    spec = importlib.util.spec_from_file_location("_rope_mask_bwd_example_for_test", source)
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


@pytest.mark.parametrize("layout", ROPE_MASK_BWD_LAYOUT_CASES)
@pytest.mark.parametrize(("dtype", "rotary_mode"), ROPE_MASK_BWD_VARIANT_CASES)
def test_rope_mask_bwd_accuracy(layout: str, dtype: str, rotary_mode: str) -> None:
    import torch

    example = _load_rope_mask_bwd_example()

    batch_size = 16
    head_num = 64
    hidden_size = 512
    rope_dim = 256

    torch.manual_seed(42)
    example.tilelang.disable_cache()

    if layout == "tnd":
        example.check_case_tnd(batch_size, head_num, hidden_size, rope_dim, dtype, rotary_mode)
    else:
        batch = 4
        seq_len = batch_size // 4 if batch_size >= 4 else batch_size
        example.check_case_bsnd(batch, seq_len, head_num, hidden_size, rope_dim, dtype, rotary_mode)
