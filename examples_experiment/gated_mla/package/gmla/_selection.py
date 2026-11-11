"""Fixed selected modules. Importing this file does not compile or launch kernels."""

from functools import lru_cache


@lru_cache(maxsize=1)
def selected_components():
    from . import layout, metadata
    from .kernels import decode, packed, prefill, query, query_residual, value, vector, zero

    query_provider = query.make_provider(vector, query_residual, packed, blocks=20, prefill_tile_m=None)
    value_provider = value.make_provider(vector, packed, blocks=20, prefill_tile_m=None)
    # Only this freshly constructed provider is populated; source modules and
    # their cached factories remain untouched.
    for name in ("value_gate", "value_gate_kernel", "value_gate_packed", "value_gate_packed_kernel"):
        setattr(query_provider, name, getattr(value_provider, name))
    query_provider.source_files.update({"value_" + k: v for k, v in value_provider.source_files.items()})
    return query_provider, decode, prefill, layout, zero, metadata


@lru_cache(maxsize=2)
def selected_bundle(target):
    from ._runtime import KernelBundle

    provider, decode, prefill, layout, zero, _ = selected_components()
    return KernelBundle(
        projections=provider,
        query_module=provider,
        decode_module=decode,
        prefill_module=prefill,
        layout_module=layout,
        empty_cache_module=zero,
        target=target,
        block_m=32,
        block_n=64,
        max_blocks=20,
    )


@lru_cache(maxsize=2)
def selected_prefill128_bundle(target):
    """Separate immutable profile bundle; the explicit/base bundle stays N64."""
    from dataclasses import replace

    return replace(selected_bundle(target), block_n=128)


def selected_packed_components(target):
    from ._packed import Components

    provider, decode, prefill, layout, _, metadata = selected_components()
    return Components(provider, provider, decode, prefill, layout, metadata, target=target, block_m=32, block_n=64, max_blocks=20)
