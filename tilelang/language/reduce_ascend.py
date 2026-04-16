from __future__ import annotations

import math
from numbers import Integral

from tvm import ir, tir
from tvm.tir import Buffer, BufferRegion, PrimExpr

from .ascend import _dtype, _retrieve_shape

_REDUCE_KWARG_SENTINEL = object()

__all__ = ["reduce", "reduce_sum", "reduce_max", "reduce_min"]


def _get_buffer_extent(object: Buffer | BufferRegion) -> list[int]:
    return list(_retrieve_shape(object))


def _normalize_reduce_real_shape(real_shape) -> list[int] | None:
    if real_shape is None:
        return None
    if not isinstance(real_shape, (list, tuple)):
        raise TypeError(
            f"real_shape must be provided as a list or tuple, but got {type(real_shape).__name__}"
        )
    if len(real_shape) != 2:
        raise ValueError(f"real_shape must have length 2, but got {real_shape}")
    return list(real_shape)


def _shape_expr_equal(lhs: PrimExpr | int, rhs: PrimExpr | int) -> bool:
    lhs_const = _try_get_const_int(lhs)
    rhs_const = _try_get_const_int(rhs)
    if lhs_const is not None and rhs_const is not None:
        return lhs_const == rhs_const
    if ir.structural_equal(lhs, rhs):
        return True
    if isinstance(lhs, tir.PrimExpr) and isinstance(rhs, tir.PrimExpr):
        return tir.analysis.expr_deep_equal(lhs, rhs)
    return False


def _shape_list_equal(lhs: list[PrimExpr | int], rhs: list[PrimExpr | int]) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(_shape_expr_equal(lhs_item, rhs_item) for lhs_item, rhs_item in zip(lhs, rhs))


def _shape_to_str(shape: list[PrimExpr | int]) -> str:
    return f"[{', '.join(str(dim) for dim in shape)}]"


def _try_get_const_int(value: PrimExpr | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, tir.IntImm):
        return int(value)
    return None


def _validate_reduce_real_shape(
    buffer_extent: list[PrimExpr | int],
    real_shape: list[PrimExpr | int] | None,
) -> list[PrimExpr | int] | None:
    if real_shape is None:
        return None

    validated_real_shape = []
    for axis, value in enumerate(real_shape):
        if isinstance(value, bool):
            raise TypeError(
                f"real_shape[{axis}] must be an integer extent or PrimExpr, but got bool"
            )
        if isinstance(value, Integral):
            if int(value) < 0:
                raise ValueError(
                    f"real_shape[{axis}] must be >= 0, but got {value} for buffer shape "
                    f"{_shape_to_str(buffer_extent)}"
                )
            validated_real_shape.append(int(value))
        elif isinstance(value, tir.PrimExpr):
            const_value = _try_get_const_int(value)
            if const_value is not None and const_value < 0:
                raise ValueError(
                    f"real_shape[{axis}] must be >= 0, but got {const_value} for buffer shape "
                    f"{_shape_to_str(buffer_extent)}"
                )
            validated_real_shape.append(value)
        else:
            raise TypeError(
                f"real_shape[{axis}] must be an integer extent or PrimExpr, but got "
                f"{type(value).__name__}"
            )

    if len(buffer_extent) == 2:
        for axis, (value, extent) in enumerate(zip(validated_real_shape, buffer_extent)):
            value_const = _try_get_const_int(value)
            extent_const = _try_get_const_int(extent)
            if value_const is not None and extent_const is not None and value_const > extent_const:
                raise ValueError(
                    f"real_shape[{axis}]={value_const} exceeds buffer extent {extent_const} "
                    f"for buffer shape {_shape_to_str(buffer_extent)}"
                )

    return validated_real_shape


def _resolve_reduce_logical_shape(
    buffer_extent: list[PrimExpr | int],
    real_shape: list[PrimExpr | int] | None,
) -> list[PrimExpr | int]:
    if len(buffer_extent) == 2 and real_shape is not None:
        logical_shape = []
        for value, extent in zip(real_shape, buffer_extent):
            logical_shape.append(extent if _try_get_const_int(value) == 0 else value)
        return logical_shape
    return list(buffer_extent)


def _shape_can_embed_expected_shape(
    out_extent: list[PrimExpr | int], expected_shape: list[PrimExpr | int]
) -> bool:
    out_index = 0
    expected_index = 0
    while out_index < len(out_extent) and expected_index < len(expected_shape):
        if _shape_expr_equal(out_extent[out_index], expected_shape[expected_index]):
            out_index += 1
            expected_index += 1
            continue
        if _try_get_const_int(out_extent[out_index]) == 1:
            out_index += 1
            continue
        return False

    if expected_index != len(expected_shape):
        return False

    while out_index < len(out_extent):
        if _try_get_const_int(out_extent[out_index]) != 1:
            return False
        out_index += 1
    return True


def _validate_reduce_out_shape(
    buffer_extent: list[PrimExpr | int],
    out_extent: list[PrimExpr | int],
    dim: int,
    real_shape: list[PrimExpr | int] | None,
    *,
    out_is_region: bool = False,
) -> None:
    rank = len(buffer_extent)
    if rank not in (1, 2):
        return

    logical_shape = _resolve_reduce_logical_shape(buffer_extent, real_shape)
    normalized_dim = rank - 1 if dim == -1 else dim

    reduced_shape = list(logical_shape[:normalized_dim]) + list(logical_shape[normalized_dim + 1 :])
    keepdim_shape = (
        list(logical_shape[:normalized_dim]) + [1] + list(logical_shape[normalized_dim + 1 :])
    )

    expected_shapes = [reduced_shape]
    if not _shape_list_equal(keepdim_shape, reduced_shape):
        expected_shapes.append(keepdim_shape)

    # Slice-buffer reduce historically allows the output to keep the physical
    # row-buffer layout while only part of it is logically valid. The backends
    # still lower this form by tracking valid columns separately, so the
    # frontend validation must continue accepting it for compatibility.
    if rank == 2 and real_shape is not None:
        physical_cols = buffer_extent[1]
        reduced_extent = reduced_shape[0] if len(reduced_shape) == 1 else None
        physical_cols_const = _try_get_const_int(physical_cols)
        reduced_extent_const = _try_get_const_int(reduced_extent)
        has_slice_buffer_capacity = (
            reduced_extent_const is None
            or physical_cols_const is None
            or physical_cols_const >= reduced_extent_const
        )
        if has_slice_buffer_capacity:
            expected_shapes.append([physical_cols])
            expected_shapes.append([1, physical_cols])

    out_extent = list(out_extent)
    if any(_shape_list_equal(out_extent, expected_shape) for expected_shape in expected_shapes):
        return
    if out_is_region and any(
        _shape_can_embed_expected_shape(out_extent, expected_shape)
        for expected_shape in expected_shapes
    ):
        return

    expected_shapes_str = " or ".join(_shape_to_str(expected_shape) for expected_shape in expected_shapes)
    raise ValueError(
        "Invalid reduce output shape for Ascend fast-path reduce: "
        f"logical input shape is {_shape_to_str(logical_shape)}, "
        f"physical buffer shape is {_shape_to_str(buffer_extent)}, "
        f"dim is {dim}, output shape is {_shape_to_str(out_extent)}, "
        f"expected {expected_shapes_str}"
    )


def _parse_reduce_optional_args(
    op_name: str,
    args: tuple[object, ...],
    clear=_REDUCE_KWARG_SENTINEL,
    real_shape=_REDUCE_KWARG_SENTINEL,
) -> tuple[bool, list[int] | None]:
    parsed_clear = True
    clear_assigned = False
    if clear is not _REDUCE_KWARG_SENTINEL:
        if not isinstance(clear, bool):
            raise TypeError(f"{op_name} clear must be a bool, but got {type(clear).__name__}")
        parsed_clear = clear
        clear_assigned = True

    parsed_real_shape = None
    real_shape_assigned = False
    if real_shape is not _REDUCE_KWARG_SENTINEL and real_shape is not None:
        parsed_real_shape = _normalize_reduce_real_shape(real_shape)
        real_shape_assigned = True

    if len(args) > 2:
        raise TypeError(
            f"{op_name} accepts at most two extra positional arguments after dim, got {len(args)}"
        )

    for arg in args:
        if isinstance(arg, bool):
            if clear_assigned:
                raise TypeError(f"{op_name} got multiple values for clear")
            parsed_clear = arg
            clear_assigned = True
        elif isinstance(arg, (list, tuple)):
            if real_shape_assigned:
                raise TypeError(f"{op_name} got multiple values for real_shape")
            parsed_real_shape = _normalize_reduce_real_shape(arg)
            real_shape_assigned = True
        else:
            raise TypeError(
                f"{op_name} only accepts bool(clear) or list/tuple(real_shape) "
                f"after dim, but got {type(arg).__name__}"
            )

    return parsed_clear, parsed_real_shape


def _legalize_reduce_dim(buffer_extent: list[int], dim: int) -> int:
    if isinstance(dim, bool) or not isinstance(dim, Integral):
        raise TypeError(f"dim must be an integer axis, but got {type(dim).__name__}")

    rank = len(buffer_extent)
    if rank == 1:
        normalized_dim = dim if dim >= 0 else 1 + dim
        if normalized_dim != 0:
            raise ValueError(
                f"Ascend reduce only supports axis 0/-1 for 1D buffers, "
                f"but got dim={dim} for shape {tuple(buffer_extent)}"
            )
        return -1

    if rank == 2:
        normalized_dim = dim if dim >= 0 else 2 + dim
        if normalized_dim not in (0, 1):
            raise ValueError(
                f"Ascend reduce only supports axis 0/1/-1/-2 for 2D buffers, "
                f"but got dim={dim} for shape {tuple(buffer_extent)}"
            )
        return 0 if normalized_dim == 0 else -1

    if rank == 3:
        normalized_dim = dim if dim >= 0 else 2 + dim
        if normalized_dim not in (0, 1):
            raise ValueError(
                "Ascend reduce on 3D buffers only supports the trailing tile axes "
                f"(0/1/-1/-2), but got dim={dim} for shape {tuple(buffer_extent)}"
            )
        return 0 if normalized_dim == 0 else -1

    raise ValueError(
        f"Ascend reduce only supports 1D buffers, 2D buffers, or 3D trailing-tile buffers, "
        f"but got rank={rank} with shape {tuple(buffer_extent)}"
    )


def _reduce_with_clear(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    reduce_type: str,
    dim: int,
    clear: bool,
    real_shape: list[int] | None,
):
    return reduce(buffer, out, reduce_type, dim, real_shape, clear=clear)


def reduce(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    reduce_type: str,
    dim: int,
    real_shape: list[int] = None,
    clear: bool = True,
):
    """Emit the Ascend fast-path reduce intrinsic for buffers or buffer regions."""
    dtype = _dtype(buffer)

    def _handle_buffer_region(br: BufferRegion, mask):
        bf = br.buffer
        indices = [x.min for x in br.region]
        offset = bf.offset_of(indices)[0]
        extent = [x.extent for x in br.region]
        size_extent = math.prod(extent)
        return bf.access_ptr(mask, offset=offset, extent=size_extent), extent

    if isinstance(buffer, BufferRegion):
        buffer_ptr, buffer_extent = _handle_buffer_region(buffer, "r")
    else:
        buffer_ptr = buffer.access_ptr("r")
        buffer_extent = buffer.shape

    if isinstance(out, BufferRegion):
        out_ptr, out_extent = _handle_buffer_region(out, "w")
    else:
        out_ptr = out.access_ptr("w")
        out_extent = out.shape

    validated_real_shape = _validate_reduce_real_shape(list(buffer_extent), real_shape)
    _validate_reduce_out_shape(
        list(buffer_extent),
        list(out_extent),
        dim,
        validated_real_shape,
        out_is_region=isinstance(out, BufferRegion),
    )

    if validated_real_shape is None:
        validated_real_shape = [0, 0]

    if len(buffer_extent) == 1:
        M = 1
        N = buffer_extent[0]
    elif len(buffer_extent) == 2:
        M = buffer_extent[0] if _try_get_const_int(validated_real_shape[0]) == 0 else validated_real_shape[0]
        N = buffer_extent[1] if _try_get_const_int(validated_real_shape[1]) == 0 else validated_real_shape[1]
    elif len(buffer_extent) == 3:
        M = buffer_extent[1]
        N = buffer_extent[2]
    else:
        raise ValueError(
            f"Unsupported buffer rank {len(buffer_extent)} for Ascend fast-path reduce: {buffer_extent}"
        )
    shape = f"{M}, {N}"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reduce"),
        f"{reduce_type}<{dtype}, {shape}, {dim}>",
        out_ptr,
        buffer_ptr,
        tir.const(clear, "bool"),
    )


def reduce_max(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *args,
    clear=_REDUCE_KWARG_SENTINEL,
    real_shape=_REDUCE_KWARG_SENTINEL,
):
    """Perform a max reduction on the current Ascend fast-path.

    Args:
        buffer: The source buffer or buffer region.
        out: The destination buffer or buffer region.
        dim: Reduce axis for the supported fast-path ranks. 1D buffers support
            0/-1, 2D buffers support 0/1/-1/-2, and 3D buffers only support
            the trailing tile axes 0/1/-1/-2.
        *args: Optional positional compatibility arguments for ``clear`` and
            ``real_shape``.
        clear: Whether to initialize ``out`` before reduction.
        real_shape: Optional logical 2D shape for sliced UB tiles.
    """
    parsed_clear, parsed_real_shape = _parse_reduce_optional_args(
        "reduce_max", args, clear=clear, real_shape=real_shape
    )
    legalized_dim = _legalize_reduce_dim(_get_buffer_extent(buffer), dim)
    return _reduce_with_clear(
        buffer, out, "reduce_max", legalized_dim, parsed_clear, parsed_real_shape
    )


def reduce_min(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *args,
    clear=_REDUCE_KWARG_SENTINEL,
    real_shape=_REDUCE_KWARG_SENTINEL,
):
    """Perform a min reduction on the current Ascend fast-path.

    Args:
        buffer: The source buffer or buffer region.
        out: The destination buffer or buffer region.
        dim: Reduce axis for the supported fast-path ranks. 1D buffers support
            0/-1, 2D buffers support 0/1/-1/-2, and 3D buffers only support
            the trailing tile axes 0/1/-1/-2.
        *args: Optional positional compatibility arguments for ``clear`` and
            ``real_shape``.
        clear: Whether to initialize ``out`` before reduction.
        real_shape: Optional logical 2D shape for sliced UB tiles.
    """
    parsed_clear, parsed_real_shape = _parse_reduce_optional_args(
        "reduce_min", args, clear=clear, real_shape=real_shape
    )
    legalized_dim = _legalize_reduce_dim(_get_buffer_extent(buffer), dim)
    return _reduce_with_clear(
        buffer, out, "reduce_min", legalized_dim, parsed_clear, parsed_real_shape
    )


def reduce_sum(
    buffer: Buffer | BufferRegion,
    out: Buffer | BufferRegion,
    dim: int = -1,
    *args,
    clear=_REDUCE_KWARG_SENTINEL,
    real_shape=_REDUCE_KWARG_SENTINEL,
):
    """Perform a sum reduction on the current Ascend fast-path.

    Args:
        buffer: The source buffer or buffer region.
        out: The destination buffer or buffer region.
        dim: Reduce axis for the supported fast-path ranks. 1D buffers support
            0/-1, 2D buffers support 0/1/-1/-2, and 3D buffers only support
            the trailing tile axes 0/1/-1/-2.
        *args: Optional positional compatibility arguments for ``clear`` and
            ``real_shape``.
        clear: Whether to initialize ``out`` before reduction.
        real_shape: Optional logical 2D shape for sliced UB tiles.
    """
    parsed_clear, parsed_real_shape = _parse_reduce_optional_args(
        "reduce_sum", args, clear=clear, real_shape=real_shape
    )
    legalized_dim = _legalize_reduce_dim(_get_buffer_extent(buffer), dim)
    return _reduce_with_clear(
        buffer, out, "reduce_sum", legalized_dim, parsed_clear, parsed_real_shape
    )
