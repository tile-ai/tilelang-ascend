# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Embedded literal data for source-compatible A2/A3 raw code."""

from collections.abc import Mapping
import functools
import math

import numpy as np
from tilelang.language.common import decl_buffer as _decl_buffer

from tilelang import tvm
from tvm import tirx

__all__ = ["allocate_const", "decl_buffer"]


def allocate_const(data, dtype, extents, annotations=None):
    """Embed a constant array and return its pointer (no kernel argument).

    Like TA's allocate_const, cast and reshape the literal at construction time.
    Consume the pointer with ``T._src_code(code, pointer)``; the code must use
    the final declared buffer's literal name. Ordinary BufferLoad/BufferStore
    are rejected: the source Ascend emitter could not compile those accesses.
    """
    try:
        op = tvm.ir.Op.get("tl.a2a3_const_data")
    except Exception as error:
        raise RuntimeError("A2/A3 allocate_const requires a native build with USE_ASCEND=ON.") from error
    if annotations is not None and not isinstance(annotations, (Mapping, tvm.ffi.Map)):
        raise TypeError("A2/A3 allocate_const annotations must be a mapping.")
    shape = []
    for extent in extents:
        if not isinstance(extent, (int, np.integer, tirx.IntImm)):
            raise ValueError("A2/A3 allocate_const requires static integer extents.")
        shape.append(int(extent))
    array = np.asarray(data, dtype=dtype)
    if math.prod(shape) == array.size:
        array = array.reshape(shape)
    payload_shape = array.shape
    # The payload has a fixed byte order, including when serialized on another host.
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    annotation_json = tvm.ir.save_json(tvm.runtime.convert({"annotations": annotations or {}, "extents": shape}))
    args = [str(dtype), array.tobytes().hex(), "", "", annotation_json, *payload_shape]
    # The eager frontend binds opaque expressions using the Python LHS name.
    # Returning a pre-bound Var here would hit its pre-buffer pure-Var shortcut
    # and lose names for declarations at the PrimFunc root.
    return tirx.call_intrin("handle", op, *args)


@functools.wraps(_decl_buffer)
def decl_buffer(*args, **kwargs):
    """Preserve decl_buffer and type only live, proven constant-data handles."""
    data = kwargs.get("data", args[2] if len(args) > 2 else None)
    if isinstance(data, tirx.Var) and not isinstance(data.type_annotation, tvm.ir.PointerType):
        prepare = tvm.ffi.get_global_func("tl.a2a3.PrepareConstBufferData", allow_missing=True)
        if prepare is not None:
            prepare(data)
    return _decl_buffer(*args, **kwargs)
