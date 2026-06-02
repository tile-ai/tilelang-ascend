# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations

import tilelang.language as T
from tilelang.language.tir import op
from tvm.tir import PrimExpr, Buffer, BufferRegion, Var
from tvm import tir
from tilelang.language.ascend import _dtype
import math


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


def reshape(src: Buffer, shape: list[PrimExpr]) -> Buffer:
    """Reshapes the input buffer to the specified shape.

    Args:
        src (Buffer): Input buffer to be reshaped
        shape (list[PrimExpr]): New shape for the buffer

    Returns:
        Buffer: A new buffer view with the specified shape
    """
    return T.Buffer(shape, src.dtype, src.data)


def view(src: Buffer, shape: list[PrimExpr] | None = None, dtype: str | None = None) -> Buffer:
    """Views the input buffer with optionally modified shape and dtype.

    Args:
        src (Buffer): Input buffer to be viewed
        shape (list[PrimExpr] | None, optional): New shape for the buffer. Defaults to None.
        dtype (str | None = None, optional): New dtype for the buffer. Defaults to None.

    Returns:
        Buffer: A new buffer view with the specified shape and dtype
    """
    if shape is None:
        shape = src.shape
    if dtype is None:
        dtype = src.dtype
    return T.Buffer(shape, dtype, src.data)


def npu_gemm(A, B, C, init=False):
    """NPU GEMM intrinsic. A, B, C can be 2D or higher-order (leading dims must be 1)."""

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

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_mma"), f"mma<{_dtype(A)}, {_dtype(C)}, {M}, {N}>", Aptr, Bptr, Cptr, init, K)


def npu_gemm_mx(A, B, C, scale_A, scale_B, init=False, scale_dtype: str = "uint8"):
    """NPU MXFP GEMM intrinsic that wraps pto::TMATMUL_MX.

    Args:
        A: FP8 / FP4 data tile stored in L0A (Float8/Float4 dtype).
        B: FP8 / FP4 data tile stored in L0B (Float8/Float4 dtype).
        C: float accumulator stored in L0C (dtype=float32).
        scale_A: E8M0 block-scale tile for A, (M, K/32), typically uint8.
        scale_B: E8M0 block-scale tile for B, (K/32, N), typically uint8.
        init: If True, initialize the accumulator (cOut = A*B);
              if False, accumulate (cOut = cIn + A*B).
        scale_dtype: C++ type name used for the scale tile template
                     argument (default "uint8"; the codegen emits "uint8_t").
                     The pto-isa runtime interprets storage as float8_e8m0_t.

    K must be a multiple of 64. Scale block size is 32.
    """

    def legalize_arguments(arg):
        if isinstance(arg, Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    scale_A = legalize_arguments(scale_A)
    scale_B = legalize_arguments(scale_B)

    def retrieve_shape(obj):
        if isinstance(obj, Buffer):
            return obj.shape
        elif isinstance(obj, BufferRegion):
            return [r.extent for r in obj.region]
        raise ValueError(f"Unsupported argument type: {type(obj)} for buffer {obj}")

    A_shape = retrieve_shape(A)
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)
    Sa_shape = retrieve_shape(scale_A)
    Sb_shape = retrieve_shape(scale_B)

    assert len(C_shape) >= 2
    assert len(A_shape) >= 2
    assert len(B_shape) >= 2

    M = C_shape[-2]
    N = C_shape[-1]
    K = A_shape[-1]  # L0A: (M, K)
    K_B = B_shape[-2]  # L0B: (K, N)
    assert K == K_B, f"MXFP GEMM K shape mismatch: K_A={K}, K_B={K_B}"
    kMXScaleFactor = 32
    if isinstance(K, tir.IntImm):
        assert K.value % 64 == 0, f"MXFP GEMM requires K to be a multiple of 64, got K={K.value}"
        expected_sa_cols = K.value // kMXScaleFactor
        expected_sb_rows = K.value // kMXScaleFactor
        if isinstance(Sa_shape[-1], tir.IntImm):
            assert Sa_shape[-1].value == expected_sa_cols, (
                f"scale_A column mismatch: expected {expected_sa_cols} (K/{kMXScaleFactor}), got {Sa_shape[-1].value}"
            )
        if isinstance(Sb_shape[-2], tir.IntImm):
            assert Sb_shape[-2].value == expected_sb_rows, (
                f"scale_B row mismatch: expected {expected_sb_rows} (K/{kMXScaleFactor}), got {Sb_shape[-2].value}"
            )

    def retrieve_ptr(object, access_type="r"):
        if isinstance(object, Buffer):
            return object.access_ptr(access_type)
        elif isinstance(object, BufferRegion):
            buffer, region = object.buffer, object.region
            indices = [r.min for r in region]
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
        raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "w" if init is True else "rw")
    SaPtr = retrieve_ptr(scale_A, "r")
    SbPtr = retrieve_ptr(scale_B, "r")

    # Map scale_dtype to the C++ type name used in the template string.
    scale_type_map = {
        "uint8": "uint8_t",  # storage type; pto-isa treats as float8_e8m0_t
        "float8_e8m0": "float8_e8m0_t",
    }
    if scale_dtype not in scale_type_map:
        raise ValueError(f"Unsupported scale_dtype: {scale_dtype}. Expected one of {list(scale_type_map.keys())}")
    scale_ctype = scale_type_map[scale_dtype]

    template = f"mma_mxfp<{_dtype(A)}, {_dtype(C)}, {scale_ctype}, {M}, {N}, {K}>"
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_mma_mx"),
        template,
        Aptr,
        Bptr,
        Cptr,
        SaPtr,
        SbPtr,
        init,
    )


def loop_break():
    """Break out of the innermost loop."""
    return T.call_intrin("handle", op.Op.get("tl.loop_break"))  # noqa: F821
