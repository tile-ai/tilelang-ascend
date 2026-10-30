# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from tvm import tir
from tvm.tir import IntImm
from tvm.script.ir_builder.tir.frame import ForFrame
from tilelang import _ffi_api
from ._loop_annotations import loop_annotations


def Pipelined(
    start: tir.PrimExpr,
    stop: tir.PrimExpr = None,
    num_stages: int = 0,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    sync: list[list[int]] | None = None,
    group: list[list[int]] | None = None,
    cross_interval: int = 1,
    *,
    annotations: Mapping[str, Any] | None = None,
) -> ForFrame:
    """Tools to construct pipelined for loop.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.
    stop : PrimExpr
        The maximum value of iteration.
    num_stages : int
        The max number of buffer used between pipeline producers and consumers.
        if num_stages is 0, pipeline will not be enabled.
    cross_interval : int
        The interval for cross-core synchronization. When cross_interval=1,
        sync every iteration. When cross_interval=N, sync every N iterations.
    annotations : Mapping[str, Any], optional
        Additional loop metadata. The bool ``enable_offset`` option is consumed
        by constraint planning; other built-in scheduling keys are reserved.
    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = IntImm(start.dtype, 0) if hasattr(start, "dtype") else 0
    if order is None:
        order = []
    if stage is None:
        stage = []
    if sync is None:
        sync = []
    if group is None:
        group = []
    # type: ignore[attr-defined] # pylint: disable=no-member
    return _ffi_api.Pipelined(
        start,
        stop,
        num_stages,
        order,
        stage,
        sync,
        group,
        cross_interval,
        loop_annotations(annotations, allow_offset=True),
    )
