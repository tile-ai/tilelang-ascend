import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("moe_token_utils.py")
    spec = importlib.util.spec_from_file_location("_moe_token_utils_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These are the helpers the permute kernels next to them pick their tile sizes
# with, not a kernel of their own, so there is nothing to compare against a
# golden here: what is checked is the arithmetic each one is relied on for.
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("float32", True), ("float", True), ("float16", False), ("bfloat16", False)],
)
def test_is_fp32_dtype(dtype: str, expected: bool) -> None:
    assert _load_example().is_fp32_dtype(dtype) is expected


@pytest.mark.parametrize("hidden", [7168, 384])
def test_auto_tile_h_keeps_hidden_size(hidden: int) -> None:
    # Both of these fit, so the hidden dimension is taken whole.
    assert _load_example().auto_tile_h(hidden) == hidden


@pytest.mark.parametrize(
    ("tokens", "cores", "kwargs", "expected"),
    [
        (8192, 24, {}, 64),
        (16, 24, {}, 16),
        (8192, 24, {"large_candidates": [128, 64, 32]}, 128),
    ],
    ids=["many_tokens", "fewer_tokens_than_a_tile", "custom_candidates"],
)
def test_auto_tile_t(tokens: int, cores: int, kwargs: dict, expected: int) -> None:
    # More tokens than cores can cover in one tile picks from the candidates;
    # fewer than one tile's worth collapses to the token count itself.
    assert _load_example().auto_tile_t(tokens, cores, **kwargs) == expected


@pytest.mark.parametrize(
    ("target", "expected"),
    [(3, (3, 5)), (8, (8, 5))],
    ids=["already_long_enough", "padded"],
)
def test_pad_first_dim(target: int, expected: tuple) -> None:
    module = _load_example()
    assert tuple(module.pad_first_dim(torch.zeros(3, 5), target).shape) == expected


@pytest.mark.parametrize(
    ("target", "expected"),
    [(5, (3, 5)), (9, (3, 9))],
    ids=["already_wide_enough", "padded"],
)
def test_pad_last_dim(target: int, expected: tuple) -> None:
    module = _load_example()
    assert tuple(module.pad_last_dim(torch.zeros(3, 5), target).shape) == expected
