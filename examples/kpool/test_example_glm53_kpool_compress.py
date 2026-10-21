"""GLM-5.3 k-pool compress tests (Ascend port).

Mirrors the CUDA/ROCm test suite of examples/kpool in tile-ai/tilelang:
published geometry, reordered masked cache writes (round_scale on/off),
implicit all-true mask with the empty-input no-launch path, metadata
validation, plus a parametrized multi-tile sweep over 300/1024 pools
(per-PR set) and the benchmark shapes 4096/16384 pools (low_priority,
scheduled full run) that exercises the block grid and scatter writes
across several cache blocks and bounds the dequantized deviation by one
E4M3 code or one per-pool scale unit (see _assert_fp8_within_bounds).
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_example() -> ModuleType:
    source = Path(__file__).with_name("example_glm53_kpool_compress.py")
    spec = importlib.util.spec_from_file_location("_glm53_kpool_compress_for_test", source)
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


def _make_inputs(example: ModuleType, num_pools: int, seed: int = 20260918):
    """Deterministic GLM-shaped tensors on CPU; callers move what they need."""
    torch.manual_seed(seed)
    slot_k = torch.randn(num_pools, example.GLM53_POOL_SIZE, example.GLM53_HEAD_DIM).to(torch.bfloat16)
    slot_score = torch.randn_like(slot_k)
    ape = torch.randn(example.GLM53_POOL_SIZE, example.GLM53_HEAD_DIM, dtype=torch.float32)
    return slot_k, slot_score, ape


def test_glm53_kpool_published_geometry() -> None:
    """Keep the standalone kernel default aligned with the public checkpoint."""
    example = _load_example()
    assert example.GLM53_POOL_SIZE == 4
    assert example.GLM53_HEAD_DIM == 128


@pytest.mark.parametrize("round_scale", [False, True])
def test_glm53_kpool_compress_reordered_masked_cache_write(round_scale: bool) -> None:
    """Match the reference and preserve every cache slot not actively written."""
    example = _load_example()
    example.tilelang.disable_cache()

    slot_k, slot_score, ape = _make_inputs(example, num_pools=7)
    num_blocks, page_size = 3, 16
    k_cache = torch.full((num_blocks, page_size, example.GLM53_HEAD_DIM), 0x38, dtype=torch.uint8).npu()
    scale_cache = torch.full((num_blocks, page_size), -7.0, dtype=torch.float32).npu()
    original_k = k_cache.clone()
    original_scale = scale_cache.clone()

    # Deliberately reorder physical locations. Masked rows use negative
    # sentinels, which must never be dereferenced by the device kernel.
    loc = torch.tensor([31, -1, 2, 40, 17, -123, 9], dtype=torch.int64)
    write_mask = torch.tensor([True, False, True, True, True, False, True])

    expected_k, expected_scale = example.glm53_kpool_reference(slot_k, slot_score, ape, round_scale=round_scale)
    example.glm53_kpool_compress_and_write_cache(
        slot_k.npu(),
        slot_score.npu(),
        ape.npu(),
        loc.npu(),
        k_cache,
        scale_cache,
        write_mask=write_mask.npu(),
        round_scale=round_scale,
    )
    torch.npu.synchronize()

    flat_k = k_cache.view(-1, example.GLM53_HEAD_DIM).cpu()
    flat_scale = scale_cache.view(-1).cpu()
    active_rows = write_mask.nonzero(as_tuple=False).flatten()
    active_locs = loc[active_rows]
    example._assert_fp8_within_one_ulp(flat_k[active_locs], expected_k[active_rows])
    torch.testing.assert_close(flat_scale[active_locs], expected_scale[active_rows], rtol=2e-3, atol=1e-6)

    untouched = torch.ones(num_blocks * page_size, dtype=torch.bool)
    untouched[active_locs] = False
    assert torch.equal(flat_k[untouched], original_k.view(-1, example.GLM53_HEAD_DIM).cpu()[untouched])
    assert torch.equal(flat_scale[untouched], original_scale.view(-1).cpu()[untouched])


def test_glm53_kpool_compress_default_mask_and_empty_input() -> None:
    """Cover the implicit all-true mask and the no-launch empty-input path."""
    example = _load_example()
    example.tilelang.disable_cache()

    slot_k, slot_score, ape = _make_inputs(example, num_pools=3)
    k_cache = torch.zeros((1, 8, example.GLM53_HEAD_DIM), dtype=torch.uint8).npu()
    scale_cache = torch.zeros((1, 8), dtype=torch.float32).npu()
    loc = torch.tensor([6, 0, 4], dtype=torch.int64)

    expected_k, expected_scale = example.glm53_kpool_reference(slot_k, slot_score, ape)
    example.glm53_kpool_compress_and_write_cache(slot_k.npu(), slot_score.npu(), ape.npu(), loc.npu(), k_cache, scale_cache)
    torch.npu.synchronize()
    example._assert_fp8_within_one_ulp(k_cache.view(-1, example.GLM53_HEAD_DIM)[loc].cpu(), expected_k)
    torch.testing.assert_close(scale_cache.view(-1)[loc].cpu(), expected_scale, rtol=0, atol=0)

    empty_k = slot_k[:0].npu()
    empty_score = slot_score[:0].npu()
    before_k = k_cache.clone()
    before_scale = scale_cache.clone()
    example.glm53_kpool_compress_and_write_cache(empty_k, empty_score, ape.npu(), loc[:0].npu(), k_cache, scale_cache)
    assert torch.equal(k_cache, before_k)
    assert torch.equal(scale_cache, before_scale)


def test_glm53_kpool_compress_rejects_unsafe_metadata() -> None:
    """Reject metadata that could cause ambiguous or out-of-bounds writes."""
    example = _load_example()

    slot_k, slot_score, ape = _make_inputs(example, num_pools=2)
    k_cache = torch.zeros((1, 4, example.GLM53_HEAD_DIM), dtype=torch.uint8).npu()
    scale_cache = torch.zeros((1, 4), dtype=torch.float32).npu()
    wanted = (slot_k.npu(), slot_score.npu(), ape.npu())

    with pytest.raises(ValueError, match="unique"):
        example.glm53_kpool_compress_and_write_cache(*wanted, torch.tensor([2, 2], dtype=torch.int64).npu(), k_cache, scale_cache)

    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        example.glm53_kpool_compress_and_write_cache(*wanted, torch.tensor([0, 4], dtype=torch.int64).npu(), k_cache, scale_cache)

    with pytest.raises(TypeError, match="loc must be torch.int64"):
        example.glm53_kpool_compress_and_write_cache(*wanted, torch.tensor([0, 1], dtype=torch.int32).npu(), k_cache, scale_cache)

    with pytest.raises(ValueError, match="ape must have shape"):
        example.glm53_kpool_compress_and_write_cache(
            slot_k.npu(), slot_score.npu(), ape[:, :-1].npu(), torch.tensor([0, 1], dtype=torch.int64).npu(), k_cache, scale_cache
        )

    with pytest.raises(ValueError, match="head_dim=128"):
        example.glm53_kpool_compress_and_write_cache(
            *wanted, torch.tensor([0, 1], dtype=torch.int64).npu(), k_cache, scale_cache, head_dim=64
        )

    with pytest.raises(ValueError, match="pool_size must be positive"):
        example.glm53_kpool_compress_and_write_cache(
            *wanted, torch.tensor([0, 1], dtype=torch.int64).npu(), k_cache, scale_cache, pool_size=0
        )


def _assert_fp8_within_bounds(
    example: ModuleType,
    actual_u8: torch.Tensor,
    expected_u8: torch.Tensor,
    max_scale_units: float = 1.0,
) -> None:
    """Bound each element by one E4M3 code or ``max_scale_units`` of the per-pool scale.

    The decoded quantized values already carry the scale, so ``delta`` is the
    deviation measured in units of that scale. The byte-exact one-ULP check of
    the reference-mirroring cases is not statistically stable at the element
    counts of the multi-tile sweep: a rare fp32 rounding difference in the
    softmax path crosses a bf16 pooling boundary, and Hadamard amplification
    shifts near-zero rotated dims by several E4M3 codes. A seed sweep over the
    four shapes (90 runs, seeds 0-31/7-22, ~44M elements total) put the worst
    multi-code excursion at 0.25 scale units, so this bound keeps 4x headroom
    while staying orders of magnitude tighter than a structural error.
    """
    assert actual_u8.shape == expected_u8.shape
    finite = (actual_u8 & 0x7F) != 0x7F
    assert bool(finite.all()) and bool(((expected_u8 & 0x7F) != 0x7F).all())
    ulps = (example._ordered_codes(actual_u8) - example._ordered_codes(expected_u8)).abs()
    delta = (actual_u8.contiguous().view(torch.float8_e4m3fn).float() - expected_u8.contiguous().view(torch.float8_e4m3fn).float()).abs()
    beyond = (ulps > 1) & (delta > max_scale_units)
    assert not bool(beyond.any()), (
        f"{int(beyond.sum().item())} element(s) exceed one E4M3 code and {max_scale_units} scale units "
        f"(max {int(ulps.max().item())} codes, {float(delta.max().item()):.4f} scale units)"
    )


# Multi-tile shape sweep (moe/quant/mhc_bwd precedent): CI runs pytest with
# --forked, so every case cold-compiles its kernel from scratch and the per-PR
# cost is one case each. The per-PR set keeps two shapes: 300 pools keeps the
# padded tail tile (5 blocks, 300 = 4 * 64 + 44) and 1024 pools adds a full
# 16-block grid; the benchmark shapes 4096/16384 pools reuse the same kernel
# structure and are marked low_priority for the scheduled full run.
@pytest.mark.parametrize(
    "num_pools",
    [
        pytest.param(300, id="pools300"),
        pytest.param(1024, id="pools1024"),
        pytest.param(4096, id="pools4096", marks=pytest.mark.low_priority),
        pytest.param(16384, id="pools16384", marks=pytest.mark.low_priority),
    ],
)
def test_glm53_kpool_compress_multi_tile_scatter(num_pools: int) -> None:
    """Exercise several device tiles and cache blocks with a full write mask."""
    example = _load_example()
    example.tilelang.disable_cache()

    page_size = 64
    num_blocks = (num_pools + page_size - 1) // page_size
    slot_k, slot_score, ape = _make_inputs(example, num_pools=num_pools, seed=7)
    expected_k, expected_scale = example.glm53_kpool_reference(slot_k, slot_score, ape)

    torch.manual_seed(11)
    loc = torch.randperm(num_blocks * page_size)[:num_pools].to(torch.int64)
    k_cache = torch.full((num_blocks, page_size, example.GLM53_HEAD_DIM), 0x38, dtype=torch.uint8).npu()
    scale_cache = torch.full((num_blocks, page_size), -7.0, dtype=torch.float32).npu()

    example.glm53_kpool_compress_and_write_cache(slot_k.npu(), slot_score.npu(), ape.npu(), loc.npu(), k_cache, scale_cache)
    torch.npu.synchronize()

    flat_k = k_cache.view(-1, example.GLM53_HEAD_DIM).cpu()
    flat_scale = scale_cache.view(-1).cpu()
    _assert_fp8_within_bounds(example, flat_k[loc], expected_k)
    torch.testing.assert_close(flat_scale[loc], expected_scale, rtol=2e-3, atol=1e-6)
