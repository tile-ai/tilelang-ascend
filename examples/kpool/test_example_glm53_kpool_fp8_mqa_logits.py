"""GLM-5.3 k-pool fp8_mqa_logits tests (Ascend port).

Mirrors the reference test suite of examples/kpool in tile-ai/tilelang:
published geometry, full and partial pool ranges, invalid page entries,
non-multiple ``max_num_pools``, and the strict ``-inf`` mask comparison on
top of the FP32 numeric tolerance.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

GLM53_HEAD_DIM = 128
GLM53_INDEX_HEADS = 32
GLM53_PAGE_SIZE = 64
E4M3_MAX = 448.0


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("example_glm53_kpool_fp8_mqa_logits.py")
    spec = importlib.util.spec_from_file_location("_glm53_kpool_logits_for_test", source)
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


def _make_case(
    num_rows: int,
    max_num_pools: int,
    num_blocks: int,
    table_shape: tuple[int, int],
    starts: list[int],
    ends: list[int],
    table_rows: list[int] | None = None,
    invalid_entry: tuple[int, int, int] | None = None,
    seed: int = 20261020,
):
    """Deterministic inputs on CPU; returns (inputs, expected) via the example golden."""
    torch.manual_seed(seed)
    query = (torch.randn(num_rows, GLM53_INDEX_HEADS, GLM53_HEAD_DIM) * 0.5).to(torch.bfloat16)
    # Non-negative head weights (the gated indexer regime): the 32 head terms
    # then add without cancellation, so the fp32 tolerance holds.
    weights = torch.rand(num_rows, GLM53_INDEX_HEADS) + 0.1
    raw = torch.rand(num_blocks, GLM53_PAGE_SIZE, GLM53_HEAD_DIM) * E4M3_MAX
    k_cache = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_cache = torch.rand(num_blocks, GLM53_PAGE_SIZE) + 0.5

    num_page_rows, max_pages = table_shape
    pool_page_table = torch.arange(num_page_rows * max_pages, dtype=torch.int32).reshape(num_page_rows, max_pages)
    pool_page_table %= num_blocks
    if invalid_entry is not None:
        r, c, value = invalid_entry
        pool_page_table[r, c] = value
    if table_rows is None:
        table_rows = list(range(num_rows))
    page_table_rows = torch.tensor(table_rows, dtype=torch.int32)
    pool_starts = torch.tensor(starts, dtype=torch.int32)
    pool_ends = torch.tensor(ends, dtype=torch.int32)

    inputs = dict(
        query=query,
        k_cache=k_cache,
        scale_cache=scale_cache,
        weights=weights,
        pool_page_table=pool_page_table,
        page_table_rows=page_table_rows,
        pool_starts=pool_starts,
        pool_ends=pool_ends,
    )
    return inputs


def _run(example: ModuleType, inputs, max_num_pools: int):
    expected = example.glm53_kpool_fp8_mqa_logits_reference(
        inputs["query"],
        inputs["k_cache"],
        inputs["scale_cache"],
        inputs["weights"],
        inputs["pool_page_table"],
        inputs["page_table_rows"],
        inputs["pool_starts"],
        inputs["pool_ends"],
        max_num_pools=max_num_pools,
    )
    actual = example.glm53_kpool_fp8_mqa_logits(
        inputs["query"].npu(),
        inputs["k_cache"].npu(),
        inputs["scale_cache"].npu(),
        inputs["weights"].npu(),
        inputs["pool_page_table"].npu(),
        inputs["page_table_rows"].npu(),
        inputs["pool_starts"].npu(),
        inputs["pool_ends"].npu(),
        max_num_pools=max_num_pools,
    )
    torch.npu.synchronize()
    return actual.cpu(), expected


def _assert_logits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    # Strict special-value contract first: the -inf mask must match exactly.
    actual_inf = torch.isinf(actual) & (actual < 0)
    expected_inf = torch.isinf(expected) & (expected < 0)
    assert torch.equal(actual_inf, expected_inf)
    torch.testing.assert_close(actual, expected, rtol=2**-10, atol=2**-16)


def test_glm53_kpool_logits_default_geometry() -> None:
    """Keep the standalone kernel defaults aligned with the public checkpoint."""
    example = _load_example()
    assert example.GLM53_INDEX_HEADS == 32
    assert example.GLM53_HEAD_DIM == 128
    assert example.GLM53_PAGE_SIZE == 64


@pytest.mark.parametrize("num_rows,max_num_pools,num_blocks", [(2, 128, 4), (8, 1024, 16)])
def test_glm53_kpool_logits_full_ranges(num_rows: int, max_num_pools: int, num_blocks: int) -> None:
    """Full valid ranges: every pool finite, one page per 64-pool tile."""
    example = _load_example()
    example.tilelang.disable_cache()

    inputs = _make_case(
        num_rows,
        max_num_pools,
        num_blocks,
        table_shape=(num_rows, max_num_pools // GLM53_PAGE_SIZE),
        starts=[0] * num_rows,
        ends=[max_num_pools] * num_rows,
    )
    actual, expected = _run(example, inputs, max_num_pools)
    _assert_logits(actual, expected)
    assert torch.isfinite(actual).all()


def test_glm53_kpool_logits_partial_ranges() -> None:
    """Partial, empty and page-unaligned ranges must emit exact -inf."""
    example = _load_example()
    example.tilelang.disable_cache()

    inputs = _make_case(
        4,
        512,
        8,
        table_shape=(4, 8),
        starts=[7, 0, 64, 500],
        ends=[500, 64, 64, 512],
    )
    actual, expected = _run(example, inputs, 512)
    _assert_logits(actual, expected)
    assert not torch.isfinite(actual[0, :7]).any()
    assert not torch.isfinite(actual[0, 500:]).any()
    assert not torch.isfinite(actual[2]).all()  # empty range row


def test_glm53_kpool_logits_invalid_page_entry() -> None:
    """An invalid page-table entry for an in-range page must mark that page -inf.

    Upstream rejects such metadata on the host; this port keeps the device-side
    guard as a safety net, so the expectation is built by hand: same case with
    a valid entry, with the invalid page's lanes replaced by -inf.
    """
    example = _load_example()
    example.tilelang.disable_cache()

    inputs = _make_case(
        2,
        256,
        4,
        table_shape=(2, 4),
        starts=[0, 0],
        ends=[256, 256],
        invalid_entry=(1, 1, -1),
    )
    actual, _ = _run(example, inputs, 256)

    valid_inputs = dict(inputs)
    valid_table = inputs["pool_page_table"].clone()
    valid_table[1, 1] = 1
    valid_inputs["pool_page_table"] = valid_table
    _, expected = _run(example, valid_inputs, 256)
    expected[1, 64:128] = -torch.inf

    _assert_logits(actual, expected)
    assert not torch.isfinite(actual[1, 64:128]).any()
    assert torch.isfinite(actual[0]).all()


def test_glm53_kpool_logits_non_multiple_max_pools() -> None:
    """max_num_pools not a multiple of the 64-pool tile uses the padded tail."""
    example = _load_example()
    example.tilelang.disable_cache()

    inputs = _make_case(
        3,
        100,
        4,
        table_shape=(3, 2),
        starts=[0, 5, 36],
        ends=[100, 99, 100],
    )
    actual, expected = _run(example, inputs, 100)
    _assert_logits(actual, expected)
    assert actual.shape == (3, 100)


def test_glm53_kpool_logits_empty_inputs() -> None:
    """Empty row batch and zero-width output take the no-launch path."""
    example = _load_example()
    inputs = _make_case(2, 128, 4, table_shape=(2, 2), starts=[0, 0], ends=[128, 128])

    out = example.glm53_kpool_fp8_mqa_logits(
        *(t.npu() for t in inputs.values()),
        max_num_pools=0,
    )
    assert out.shape == (2, 0)

    out = example.glm53_kpool_fp8_mqa_logits(
        inputs["query"][:0].npu(),
        inputs["k_cache"].npu(),
        inputs["scale_cache"].npu(),
        inputs["weights"][:0].npu(),
        inputs["pool_page_table"].npu(),
        inputs["page_table_rows"][:0].npu(),
        inputs["pool_starts"][:0].npu(),
        inputs["pool_ends"][:0].npu(),
        max_num_pools=128,
    )
    assert out.shape == (0, 128)


def test_glm53_kpool_logits_rejects_bad_metadata() -> None:
    """Reject metadata that would break the kernel contract."""
    example = _load_example()

    inputs = _make_case(2, 128, 4, table_shape=(2, 2), starts=[0, 0], ends=[128, 128])
    args = dict(
        query=inputs["query"].npu(),
        k_cache=inputs["k_cache"].npu(),
        scale_cache=inputs["scale_cache"].npu(),
        weights=inputs["weights"].npu(),
        pool_page_table=inputs["pool_page_table"].npu(),
        page_table_rows=inputs["page_table_rows"].npu(),
        pool_starts=inputs["pool_starts"].npu(),
        pool_ends=inputs["pool_ends"].npu(),
        max_num_pools=128,
    )

    with pytest.raises(TypeError, match="query must be torch.bfloat16"):
        example.glm53_kpool_fp8_mqa_logits(**{**args, "query": args["query"].float()})
    with pytest.raises(TypeError, match="k_cache must be torch.uint8"):
        example.glm53_kpool_fp8_mqa_logits(**{**args, "k_cache": args["k_cache"].to(torch.int16)})
    with pytest.raises(TypeError, match="pool_page_table must be torch.int32"):
        example.glm53_kpool_fp8_mqa_logits(**{**args, "pool_page_table": args["pool_page_table"].to(torch.int64)})
    with pytest.raises(ValueError, match="max_num_pools must be nonnegative"):
        example.glm53_kpool_fp8_mqa_logits(**{**args, "max_num_pools": -1})
    with pytest.raises(ValueError, match="requires num_heads=32"):
        example.glm53_kpool_fp8_mqa_logits(**{**args, "num_heads": 16})
