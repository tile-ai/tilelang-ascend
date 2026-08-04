import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("example_cumsum.py")
    spec = importlib.util.spec_from_file_location("_cumsum_gdn_example_for_test", source)
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


# Shapes come from the example. Sequence lengths that are not a multiple of the
# chunk (250 against 32) and head counts that are odd are what exercise the tail
# handling, so both stay in.
@pytest.mark.parametrize(
    "batch, heads, seq_len, chunk, reverse, head_first",
    [
        (2, 32, 256, 32, False, True),
        (2, 32, 256, 32, True, True),
        (2, 7, 250, 32, False, False),
        (2, 7, 250, 32, True, False),
        (4, 8, 512, 64, True, True),
    ],
)
def test_chunk_cumsum(batch, heads, seq_len, chunk, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len) if head_first else (batch, seq_len, heads)
    torch.manual_seed(0)
    g = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_cumsum(g, chunk, reverse=reverse, head_first=head_first)
    expected = example.ref_chunk_cumsum(g, chunk, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


# use_fragment selects a different buffer for the running sum; the reference is
# the same either way.
@pytest.mark.parametrize(
    "batch, heads, seq_len, chunk, reverse, head_first",
    [
        (2, 32, 256, 32, False, True),
        (2, 32, 256, 32, True, True),
        (2, 7, 250, 32, False, False),
        (1, 16, 128, 64, True, True),
    ],
)
def test_chunk_cumsum_fragment(batch, heads, seq_len, chunk, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len) if head_first else (batch, seq_len, heads)
    torch.manual_seed(0)
    g = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_cumsum(g, chunk, reverse=reverse, head_first=head_first, use_fragment=True)
    expected = example.ref_chunk_cumsum(g, chunk, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
