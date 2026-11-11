"""Synchronous packed-cache entry points using only selected package modules."""

import torch

from ._packed import CacheCommittedForwardError, PackedModelAdapter, require_inference
from .runtime import validate_target

__all__ = ["CacheCommittedForwardError", "create_cache", "build_packed_adapter"]


def create_cache(batch, capacity, dtype=torch.float16, *, device="npu:0", target="pto"):
    require_inference()
    validate_target(target)
    import torch_npu  # noqa: F401 -- registers the NPU API before owner use

    from ._cache_owner import CacheOwner

    return CacheOwner(batch, capacity, dtype=dtype, device=device, target=target)


def build_packed_adapter(owner, *, heads):
    require_inference()
    from ._selection import selected_packed_components

    return PackedModelAdapter(owner, heads=heads, components=selected_packed_components(validate_target(owner.target)))
