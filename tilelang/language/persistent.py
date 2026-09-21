# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations

from tvm import tir
from tilelang import _ffi_api


def Persistent(
    domain: list[tir.PrimExpr | int],
    wave_size: tir.PrimExpr | int,
    index: tir.PrimExpr | int,
    group_size: tir.PrimExpr | int = 8,
):
    """Tools to construct persistent for loop.

    Parameters
    ----------
    domain : list[tir.PrimExpr or int]
        Positive extent of each iteration-domain dimension.
    wave_size : tir.PrimExpr or int
        Positive number of workers in one wave.
    index : tir.PrimExpr or int
        Worker index in one wave. Must satisfy ``0 <= index < wave_size``.
    group_size : tir.PrimExpr or int
        Positive grouping factor for the last domain dimension. The last
        extent must be divisible by ``min(group_size, domain[-1])``.
    """
    return _ffi_api.Persistent(domain, wave_size, index, group_size)
