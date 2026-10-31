# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations
from numbers import Number

from tvm import tirx as tir
from tvm.tirx import IntImm
from tvm.runtime import const
from tvm_ffi import Array
from tilelang.language.loop import Pipelined as _Pipelined

__all__ = ["Pipelined"]


def _pipeline_groups(groups):
    rows = []
    for row in groups:
        values = [const(value) if isinstance(value, Number) else value for value in row]
        if any(not isinstance(value, tir.PrimExpr) for value in values):
            raise TypeError("Pipelined sync/group entries must be PrimExpr values")
        rows.append(Array(values))
    return Array(rows)


def Pipelined(
    start: tir.PrimExpr,
    stop: tir.PrimExpr = None,
    num_stages: int = 0,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    sync: list[list[int]] | None = None,
    group: list[list[int]] | None = None,
    cross_interval: int = 1,
):
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
    # The neutral frame constructor has the same source loop construction and
    # accepts additional annotations. TA's old Object map converted scalar
    # entries to PrimExpr; restore that ABI for its extra nested metadata.
    annotations = {}
    if sync:
        annotations["tl_pipeline_sync"] = _pipeline_groups(sync)
    if group:
        annotations["tl_pipeline_group"] = _pipeline_groups(group)
    if cross_interval > 0:
        annotations["tl_cross_interval"] = IntImm("int32", cross_interval)
    return _Pipelined(start, stop, num_stages, order, stage, annotations)
