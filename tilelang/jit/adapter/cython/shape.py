# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Validate the source integer domain of implicit kernel shape arguments."""

from __future__ import annotations

import operator

from tvm import tir
from tvm._ffi.runtime_ctypes import DataType, DataTypeCode


def checked_shape_value(symbol: tir.Var, value: object) -> int:
    """Snapshot a dimension before allocating results or passing the int64 ABI."""
    dtype = DataType(symbol.dtype)
    scalar_integer = dtype.lanes == 1 and dtype.type_code in (DataTypeCode.INT, DataTypeCode.UINT)
    if not scalar_integer or dtype.bits not in (8, 16, 32, 64):
        raise TypeError(f"Dynamic shape {symbol} requires a scalar integer declaration")
    bits = dtype.bits - (dtype.type_code == DataTypeCode.INT)
    maximum = min((1 << bits) - 1, (1 << 63) - 1)
    result = int(operator.index(value))
    if result < 0 or result > maximum:
        description = f"Dynamic shape {symbol}={result} is outside its {symbol.dtype}"
        raise ValueError(f"{description} dimension range [0, {maximum}]")
    return result
