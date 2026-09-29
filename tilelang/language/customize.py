# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations

import math

import tilelang.language as T
from tilelang.language.ascend import _dtype
from tilelang.language.tir import op
from tvm import DataType, arith, tir
from tvm.tir import Buffer, BufferRegion, PrimExpr, Var


_FRACTAL_SCOPES = {
    "shared.l1",
    "wmma.matrix_a",
    "wmma.matrix_b",
    "wmma.accumulator",
}


def _fractal_inner_shapes(scope: str, dtype: DataType) -> tuple[tuple[int, int], ...]:
    element_bits = dtype.bits * dtype.lanes
    if element_bits < 8 or 256 % element_bits != 0:
        raise ValueError("L1/L0 fractal views require byte-addressable dtypes")
    elements_per_c0 = 256 // element_bits
    if scope == "wmma.matrix_b":
        return ((elements_per_c0, 16),)
    if scope == "wmma.accumulator":
        return ((16, 16),)
    if scope == "shared.l1":
        return ((16, elements_per_c0), (elements_per_c0, 16))
    return ((16, elements_per_c0),)


def _validate_fractal_view(
    src: Buffer,
    shape: list[PrimExpr] | tuple[PrimExpr, ...],
    view_dtype: DataType,
    analyzer: arith.Analyzer,
    api_name: str,
) -> None:
    if len(src.shape) != len(shape) or len(shape) < 2:
        raise ValueError(f"{api_name} requires L1/L0 views to keep the same fractal grid")
    if not all(analyzer.can_prove_equal(src_dim, view_dim) for src_dim, view_dim in zip(src.shape[:-2], shape[:-2])):
        raise ValueError(f"{api_name} requires L1/L0 views to keep leading dimensions unchanged")

    src_inners = _fractal_inner_shapes(src.scope(), DataType(src.dtype))
    view_inners = _fractal_inner_shapes(src.scope(), view_dtype)
    for src_inner, view_inner in zip(src_inners, view_inners):
        if not all(analyzer.can_prove_equal(tir.floormod(dim, block), 0) for dim, block in zip(src.shape[-2:], src_inner)):
            continue
        if not all(analyzer.can_prove_equal(tir.floormod(dim, block), 0) for dim, block in zip(shape[-2:], view_inner)):
            continue
        if all(
            analyzer.can_prove_equal(src_dim // src_block, view_dim // view_block)
            for src_dim, src_block, view_dim, view_block in zip(src.shape[-2:], src_inner, shape[-2:], view_inner)
        ):
            return

    raise ValueError(f"{api_name} cannot preserve the L1/L0 fractal-block grid; use an explicit layout conversion")


def atomic_add(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added

    Returns:
        PrimExpr: Handle to the atomic addition operation
    """
    return T.call_extern("handle", "AtomicAdd", T.address_of(dst), value)


def atomic_addx2(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx2", T.address_of(dst), T.address_of(value))


def atomic_addx4(dst: Buffer, value: PrimExpr) -> PrimExpr:
    """Perform an atomic addition operation with double-width operands.

    Args:
        dst (Buffer): Destination buffer where the atomic addition will be performed
        value (PrimExpr): Value to be atomically added (double-width)

    Returns:
        PrimExpr: Handle to the double-width atomic addition operation
    """
    return T.call_extern("handle", "AtomicAddx4", T.address_of(dst), T.address_of(value))


def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr:
    """Perform a 4-element dot product with accumulation (DP4A).

    Args:
        A (Buffer): First input buffer
        B (Buffer): Second input buffer
        C (Buffer): Accumulation buffer

    Returns:
        PrimExpr: Handle to the DP4A operation
    """
    return T.call_extern("handle", "DP4A", T.address_of(A), T.address_of(B), T.address_of(C))


def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr:
    """Clamps the input value dst between [min_val, max_val]

    Args:
        dst: Input value to be clamped
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Value clamped to the specified range
    """
    dst = T.max(dst, min_val)  # Ensure value is not less than minimum
    dst = T.min(dst, max_val)  # Ensure value is not greater than maximum
    return dst


def _make_whole_storage_view(
    src: Buffer,
    shape: list[PrimExpr] | tuple[PrimExpr, ...] | None,
    dtype: str | DataType | None,
    api_name: str,
) -> Buffer:
    if not isinstance(src, Buffer):
        raise TypeError(f"{api_name} expects a complete tir.Buffer, but got {type(src).__name__}")
    analyzer = arith.Analyzer()
    if len(src.strides) != 0:
        raise ValueError(f"{api_name} does not support buffers with explicit strides")
    if len(src.axis_separators) != 0:
        raise ValueError(f"{api_name} does not support buffers with axis separators")
    if not analyzer.can_prove_equal(src.elem_offset, 0):
        raise ValueError(f"{api_name} does not support a non-zero source elem_offset")

    shape = src.shape if shape is None else shape
    dtype = src.dtype if dtype is None else dtype
    view_dtype = dtype if isinstance(dtype, DataType) else DataType(dtype)
    src_dtype = DataType(src.dtype)
    same_shape = len(src.shape) == len(shape) and all(
        analyzer.can_prove_equal(src_dim, view_dim) for src_dim, view_dim in zip(src.shape, shape)
    )
    src_bits = math.prod(src.shape) * src_dtype.bits * src_dtype.lanes
    view_bits = math.prod(shape) * view_dtype.bits * view_dtype.lanes
    if not analyzer.can_prove_equal(src_bits, view_bits):
        raise ValueError(f"{api_name} requires source and view to have the same total bit size, but got {src_bits} and {view_bits}")

    linear_scopes = {"global", "shared", "shared.ub"}
    if src.scope() in _FRACTAL_SCOPES and (src_dtype != view_dtype or not same_shape):
        _validate_fractal_view(src, shape, view_dtype, analyzer, api_name)
    elif src.scope() not in linear_scopes and src.scope() not in _FRACTAL_SCOPES:
        raise ValueError(f"{api_name} does not support storage scope {src.scope()}")

    return T.Tensor(
        shape,
        view_dtype,
        data=src.data,
        elem_offset=src.elem_offset,
        scope=src.scope(),
        align=src.data_alignment,
        offset_factor=src.offset_factor,
    )


def reshape(src: Buffer, shape: list[PrimExpr] | tuple[PrimExpr, ...]) -> Buffer:
    """Return a zero-copy whole-storage view with a new shape.

    Args:
        src: Complete, compact input buffer.
        shape: New logical shape. Its total bit size must equal that of ``src``.

    Returns:
        A buffer that shares ``src.data`` and preserves ``src.dtype``.

    Notes:
        This is the same whole-storage alias operation as :func:`view`, not a
        data movement. See ``docs/api_docs/T.view.md`` for the public
        constraints.
    """
    return _make_whole_storage_view(src, shape, None, "T.reshape")


def view(
    src: Buffer,
    shape: list[PrimExpr] | tuple[PrimExpr, ...] | None = None,
    dtype: str | DataType | None = None,
) -> Buffer:
    """Return a zero-copy whole-storage shape/dtype view of ``src``.

    Args:
        src: Complete, compact input buffer with zero element offset.
        shape: New logical shape. ``None`` preserves ``src.shape``.
        dtype: New logical dtype. ``None`` preserves ``src.dtype``.

    Returns:
        A buffer that shares ``src.data`` without allocating, copying, or
        numerically converting data.

    Notes:
        A dtype change reinterprets bits; it does not cast values. See the
        canonical API reference in ``docs/api_docs/T.view.md`` for the public
        constraints.

    Raises:
        TypeError: If ``src`` is not a complete :class:`tir.Buffer`.
        ValueError: If the source is not a supported whole-storage buffer, the
            total bit sizes differ, or the requested scope/layout view is not
            supported.
    """
    return _make_whole_storage_view(src, shape, dtype, "T.view")


def npu_gemm(A, B, C, init=False, n_actual=None, unit_flag=None, k_actual=None):
    """NPU GEMM intrinsic. A, B, C can be 2D or higher-order (leading dims must be 1).

    n_actual / unit_flag (both default ``None``): optional trailing args mapping to
    the C++ ``mma`` template's ``n_actual`` (runtime output-column count, <= N) and
    ``unitFlag`` (0b10 accumulate / 0b11 flush, driving the hardware mma->fixpipe
    pipeline). When both are ``None`` the call emits the legacy 6-argument form, so
    every existing caller is byte-for-byte unchanged (the C++ defaults are
    ``n_actual = N`` and ``unitFlag = 0``). Setting ``unit_flag=0b11`` here and on a
    following ``T.copy(L0C->GM, unit_flag=0b11)`` fuses the two, letting the fixpipe
    of one tile overlap the mma of the next across an L0C ping-pong.

    k_actual (default ``None``): runtime contraction length, passed as the C++ mma's
    ``K`` argument and overriding the value derived from ``A``'s last dim. This lets
    the operands stay full buffers while the mma contracts only ``k_actual`` columns.
    Passing a symbolic slice instead (``a_l0[pp, :, 0:k]``) is not an option, since
    ``access_ptr`` would need a concrete extent.
    """

    def legalize_arguments(arg: Buffer | Var):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (tir.Buffer | tir.Var: Input argument to legalize

        Returns:
            tir.Buffer | tir.Var: The legalized argument
        """
        if isinstance(arg, Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    def retrieve_shape(object: Buffer | BufferRegion) -> list[int]:
        if isinstance(object, Buffer):
            return object.shape
        elif isinstance(object, BufferRegion):
            region = object.region
            shape = []
            for r in region:
                shape.append(r.extent)
            return shape
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    A_shape = retrieve_shape(A)
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)

    assert len(C_shape) >= 2, "current only support C as a 2D or higher-order tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(C_shape) > 2:
        for i in range(len(C_shape) - 2):
            assert C_shape[i] == 1, (
                "current only support C as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape[-2], C_shape[-1]
    K = A_shape[-1]
    K_B = B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    def retrieve_ptr(object: Buffer | BufferRegion, access_type: str = "r") -> PrimExpr:
        if isinstance(object, Buffer):
            return object.access_ptr(access_type)
        elif isinstance(object, BufferRegion):
            buffer, region = object.buffer, object.region
            indices = []
            for r in region:
                indices.append(r.min)
            strides = []
            stride = 1
            for s in reversed(buffer.shape):
                strides.insert(0, stride)
                stride *= s
            offset = 0
            for i in range(len(indices)):
                offset += indices[i] * strides[i]
            extent = [x.extent for x in object.region]
            size_extent = math.prod(extent)
            return buffer.access_ptr(access_mask=access_type, offset=offset, extent=size_extent)
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "w" if init is True else "rw")

    # k_actual overrides the K derived from A's last dim, so the operands can stay
    # full buffers while the mma contracts fewer columns. The <M, N> template
    # params are unaffected.
    K_runtime = K if k_actual is None else k_actual

    mma_args = [f"mma<{_dtype(A)}, {_dtype(C)}, {M}, {N}>", Aptr, Bptr, Cptr, init, K_runtime]
    # Trailing args are positional, so n_actual must be materialised (as its no-op
    # default N) whenever unit_flag is set.
    if n_actual is not None or unit_flag is not None:
        mma_args.append(n_actual if n_actual is not None else N)
        mma_args.append(unit_flag if unit_flag is not None else 0)
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_mma"), *mma_args)


def loop_break():
    """Break out of the innermost loop."""
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))  # noqa: F821
