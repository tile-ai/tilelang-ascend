# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from tvm import tir
from tvm.script.ir_builder.tir.frame import ForFrame
from tilelang import _ffi_api
from ._loop_annotations import loop_annotations


def Persistent(
    domain: list[tir.PrimExpr | int],
    wave_size: tir.PrimExpr | int,
    index: tir.PrimExpr | int,
    group_size: tir.PrimExpr | int = 8,
    num_stages: int = 0,
    *,
    annotations: Mapping[str, Any] | None = None,
) -> ForFrame:
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
        Positive grouping factor for the last domain dimension. Non-divisible
        groups are padded and invalid tiles are excluded from the body.
    num_stages : int
        Defaults to zero for serial waves. The legacy planner accepts positive
        values for a provable single full wave without padding. The constraint
        planner also supports statically bounded guarded waves with a matching
        calibrated A3 cost model, subject to its pipeline eligibility checks.
    annotations : Mapping[str, Any], optional
        Additional loop metadata; reserved scheduling keys cannot be injected.

    Notes
    -----
    Internal wave and grouping arithmetic uses int64. Symbolic arguments must
    satisfy the positive-domain and worker-index contracts at runtime, and
    the padded domain and wave index range must fit int64.
    """
    return _ffi_api.Persistent(domain, wave_size, index, group_size, num_stages, loop_annotations(annotations))
