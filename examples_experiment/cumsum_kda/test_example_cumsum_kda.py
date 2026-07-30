import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("example_cumsum_kda.py")
    spec = importlib.util.spec_from_file_location("_cumsum_kda_example_for_test", source)
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
# block and head counts that are odd are what exercise the tail handling, so
# both stay in.
@pytest.mark.parametrize(
    "batch, heads, seq_len, block_t, reverse, head_first",
    [
        (1, 8, 128, 32, False, True),
        (1, 8, 130, 32, True, False),
        (1, 7, 130, 32, False, False),
        (2, 16, 256, 64, True, True),
    ],
)
def test_chunk_local_cumsum_scalar(batch, heads, seq_len, block_t, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len) if head_first else (batch, seq_len, heads)
    torch.manual_seed(0)
    s = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_local_cumsum_scalar(s, block_t, reverse=reverse, head_first=head_first)
    expected = example.ref_chunk_local_cumsum_scalar(s, block_t, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch, heads, seq_len, reverse, head_first",
    [
        (1, 8, 128, False, True),
        (1, 8, 130, True, False),
        (1, 7, 130, False, False),
        (2, 16, 256, True, True),
    ],
)
def test_chunk_global_cumsum_scalar(batch, heads, seq_len, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len) if head_first else (batch, seq_len, heads)
    torch.manual_seed(0)
    s = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_global_cumsum_scalar(s, reverse=reverse, head_first=head_first)
    expected = example.ref_chunk_global_cumsum_scalar(s, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch, heads, seq_len, s_dim, block_t, reverse, head_first",
    [
        (1, 8, 128, 16, 32, False, True),
        (1, 8, 130, 17, 32, True, False),
        (1, 7, 130, 17, 32, False, False),
        (2, 16, 256, 32, 64, True, True),
    ],
)
def test_chunk_local_cumsum_vector(batch, heads, seq_len, s_dim, block_t, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len, s_dim) if head_first else (batch, seq_len, heads, s_dim)
    torch.manual_seed(0)
    s = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_local_cumsum_vector(s, block_t, reverse=reverse, head_first=head_first)
    expected = example.ref_chunk_local_cumsum_vector(s, block_t, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch, heads, seq_len, s_dim, reverse, head_first",
    [
        (1, 8, 128, 16, False, True),
        (1, 8, 130, 17, True, False),
        (1, 7, 130, 17, False, False),
        (2, 16, 256, 32, True, True),
    ],
)
def test_chunk_global_cumsum_vector(batch, heads, seq_len, s_dim, reverse, head_first) -> None:
    import torch

    example = _load_example()

    shape = (batch, heads, seq_len, s_dim) if head_first else (batch, seq_len, heads, s_dim)
    torch.manual_seed(0)
    s = torch.randn(shape).npu().to(torch.float)

    actual = example.chunk_global_cumsum_vector(s, reverse=reverse, head_first=head_first)
    expected = example.ref_chunk_global_cumsum_vector(s, reverse=reverse, head_first=head_first)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
