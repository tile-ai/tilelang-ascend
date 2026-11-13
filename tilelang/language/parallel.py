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


def Parallel(
    *extents: tir.PrimExpr,
    coalesced_width: int | None = None,
    annotations: Mapping[str, Any] | None = None,
) -> ForFrame:
    """Tools to construct nested parallel for loop.
       This can be used to create element-wise tensor expression.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    coalesced_width : Optional[int]
        The coalesced width of the parallel loop.
    annotations : Mapping[str, Any], optional
        Metadata on the outermost generated loop. ``coalesced_width`` retains
        its legacy behavior on every generated dimension. Reserved scheduling
        controls must use their dedicated supported API.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    annotations = loop_annotations(annotations)
    if coalesced_width is not None:
        if "coalesced_width" in annotations and annotations["coalesced_width"] != coalesced_width:
            raise ValueError("conflicting coalesced_width keyword and loop annotation")
        annotations["coalesced_width"] = coalesced_width
    return _ffi_api.Parallel(extents, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member
