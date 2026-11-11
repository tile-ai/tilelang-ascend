"""Pure host scalar/span checks; importing this module touches no Torch or NPU."""

import math
from numbers import Real
import struct


INT32_MAX = 2**31 - 1


def public_prefill_block_n(batch, query_capacity, heads, cache_capacity, scale):
    """Measured seven-input profile only; no device metadata values are inputs."""
    if (batch, query_capacity, heads, cache_capacity) == (1, 128, 96, 128) and scale == 1 / math.sqrt(192):
        return 128
    return 64


def validated_host_scale(scale, default):
    if scale is None:
        return default
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a finite positive host number, not bool or Tensor")
    try:
        value = float(scale)
    except (OverflowError, ValueError) as error:
        raise ValueError("scale must be finite and positive") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("scale must be finite and positive")
    try:
        as_float32 = struct.unpack("=f", struct.pack("=f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("scale must convert to a finite nonzero float32") from error
    if not math.isfinite(as_float32) or as_float32 <= 0:
        raise ValueError("scale must convert to a finite nonzero float32")
    # Keep the original host float: route equality and factory keys are unchanged.
    return value


def checked_elements(name, *shape):
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError(f"{name} dimensions must be nonnegative host integers")
    elements = math.prod(shape)
    if elements > INT32_MAX:
        raise ValueError(f"{name} dense element span {elements} exceeds signed int32 indexing")
    return elements


def _positive(**dimensions):
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
        raise ValueError("positive host dimensions required: " + ",".join(dimensions))


def validate_projection_spans(batch, query_capacity, heads, *, packed=False):
    _positive(batch=batch, query_capacity=query_capacity, heads=heads)
    if type(packed) is not bool:
        raise TypeError("packed must be bool")
    checked_elements("projection QL/Z", batch, query_capacity, heads, 512)
    checked_elements("packed weight" if packed else "projection weight", heads, 256 if packed else 128, 512)
    # Q/QA/G/Y widths are <=512 and each output is an independent allocation.


def validate_tail_spans(batch, extent, tile):
    _positive(batch=batch, extent=extent, tile=tile)
    checked_elements("tail source", batch, extent, 512)
    checked_elements("tail padded output", batch, tile, 512)


def validate_split_spans(batch, heads, *, block_m=32, split_count=6):
    _positive(batch=batch, heads=heads, block_m=block_m, split_count=split_count)
    tasks = batch * ((heads + block_m - 1) // block_m) * split_count
    checked_elements("decode partial Z", tasks, block_m, 512)
    checked_elements("decode merged Z", batch, heads, 512)
    checked_elements("metadata", batch, 8)
    # Partial maximum/normalizer have the same first axes and a smaller width.
    return tasks


def validate_attention_spans(
    batch, query_capacity, heads, cache_capacity, *, decode=False, block_m=32, block_n=64, split_count=6, max_blocks=20
):
    _positive(batch=batch, query_capacity=query_capacity, heads=heads, cache_capacity=cache_capacity)
    if block_m not in (16, 32) or block_n not in (64, 128):
        raise ValueError("unsupported attention tile")
    if type(max_blocks) is not int or not 1 <= max_blocks <= 24:
        raise ValueError("max_blocks must be in [1,24]")
    if type(split_count) is not int or split_count not in (1, 2, 3, 4, 6, 8):
        raise ValueError("unsupported split_count")
    checked_elements("attention QL/Z", batch, query_capacity, heads, 512)
    checked_elements("latent cache C", batch, cache_capacity, 512)
    checked_elements("metadata", batch, 8)
    # QA/E are independent buffers with width64, bounded by their width512 peers.
    rows = heads if decode else query_capacity * heads
    tasks = (
        validate_split_spans(batch, heads, block_m=block_m, split_count=split_count)
        if decode
        else batch * ((rows + block_m - 1) // block_m)
    )
    if tasks >= INT32_MAX - 23:
        raise ValueError("attention task/wave IDs exceed signed int32")
    if rows % block_m:
        validate_tail_spans(batch, rows, block_m)
    if cache_capacity % block_n:
        validate_tail_spans(batch, cache_capacity, block_n)
    blocks = min(max_blocks, tasks)
    checked_elements("score workspace", blocks, block_m, block_n)
    checked_elements("probability workspace", blocks, block_m, 2 * block_n)
    checked_elements("latent workspace", blocks, block_m, 512)


def validate_chain_spans(
    batch,
    query_capacity,
    heads,
    cache_capacity,
    *,
    packed=False,
    decode=False,
    append_sources=False,
    block_m=32,
    block_n=64,
    split_count=6,
    max_blocks=20,
):
    _positive(batch=batch, heads=heads)
    if any(type(value) is not int or value < 0 for value in (query_capacity, cache_capacity)):
        raise ValueError("query/cache capacities must be nonnegative host integers")
    if query_capacity == 0:
        return  # No kernel or nonempty output allocation; preserve the empty domain.
    if cache_capacity == 0:
        checked_elements("empty-cache output", batch, query_capacity, heads, 128)
        return  # The zero path does not allocate/index QL, weights or cache payload.
    validate_projection_spans(batch, query_capacity, heads, packed=packed)
    validate_attention_spans(
        batch,
        query_capacity,
        heads,
        cache_capacity,
        decode=decode,
        block_m=block_m,
        block_n=block_n,
        split_count=split_count,
        max_blocks=max_blocks,
    )
    if append_sources:
        checked_elements("append rawKV", batch, query_capacity, 576)
