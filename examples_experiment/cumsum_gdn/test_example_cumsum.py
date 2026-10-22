import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_cumsum_example() -> ModuleType:
    source = Path(__file__).with_name("example_cumsum.py")
    spec = importlib.util.spec_from_file_location("_cumsum_example_for_test", source)
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
def cumsum_example():
    return _load_cumsum_example()


_BASIC_CONFIGS = [
    (2, 32, 256, 32, False, True),
    (2, 32, 256, 32, True, True),
    (2, 7, 250, 32, False, False),
    (2, 7, 250, 32, True, False),
    (1, 16, 128, 64, False, True),
    (1, 16, 128, 64, True, True),
    (2, 32, 250, 32, False, False),
    (2, 32, 250, 32, True, False),
    (4, 8, 512, 64, False, True),
    (4, 8, 512, 64, True, True),
]

_FRAGMENT_CONFIGS = [
    (2, 32, 256, 32, False, True, False),
    (2, 32, 256, 32, False, True, True),
    (2, 32, 256, 32, True, True, False),
    (2, 32, 256, 32, True, True, True),
    (2, 7, 250, 32, False, False, True),
    (2, 7, 250, 32, True, False, True),
    (2, 32, 250, 32, False, False, True),
    (2, 32, 250, 32, True, False, True),
    (1, 16, 128, 64, False, True, True),
    (1, 16, 128, 64, True, True, True),
]


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize("B, H, L, C, reverse, head_first", _BASIC_CONFIGS)
def test_cumsum_basic(cumsum_example, B, H, L, C, reverse, head_first):
    torch.manual_seed(0)
    shape = (B, H, L) if head_first else (B, L, H)
    g = torch.randn(shape).npu().to(torch.float)
    g_sum = cumsum_example.chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
    ref_g_sum = cumsum_example.ref_chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
    torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
@pytest.mark.parametrize("B, H, L, C, reverse, head_first, use_fragment", _FRAGMENT_CONFIGS)
def test_cumsum_fragment(cumsum_example, B, H, L, C, reverse, head_first, use_fragment):
    torch.manual_seed(0)
    shape = (B, H, L) if head_first else (B, L, H)
    g = torch.randn(shape).npu().to(torch.float)
    g_sum = cumsum_example.chunk_cumsum(g, C, reverse=reverse, head_first=head_first, use_fragment=use_fragment)
    ref_g_sum = cumsum_example.ref_chunk_cumsum(g, C, reverse=reverse, head_first=head_first)
    torch.testing.assert_close(g_sum.cpu(), ref_g_sum.cpu(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
