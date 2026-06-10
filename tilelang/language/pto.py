from __future__ import annotations

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import Literal
from tvm import tir

import math


_pipe = Literal["fix", "mte1", "mte2", "mte3", "m", "v", "s"]


def _dtype(buf):
    type_map = {
        "float16": "half",
        "float32": "float",
        "int32": "int",
        "uint32": "uint32_t",
        "bfloat16": "bfloat16_t",
        "uint16": "uint16_t",
        "uint8": "uint8_t",
        "int8": "int8_t",
        "int16": "int16_t",
        "int64": "int64_t",
        "uint64": "uint64_t",
        "e4m3_float8": "float8_e4m3_t",
        "e5m2_float8": "float8_e5m2_t",
    }
    if isinstance(buf, BufferRegion):
        buf = buf.buffer
    return type_map[buf.dtype]


def set_cross_flag(pipe: str, flag: int, mode: int = 2):
    """
    Sets a cross-core synchronization flag.

    This function emits an intrinsic to set a specific hardware event ID (flag)
    for a given pipeline stage. It is used in conjunction with `wait_cross_flag`
    to synchronize logical execution queues that are not standard producer-consumer pairs.

    Args:
        pipe (str): The pipeline stage issuing the set action (e.g., "MTE3", "V").
        flag (int): The event ID index to set.
        mode: hard synchronization modes.
            - 0: among all AICs or all AIVs
            - 1: among all AIVs within the same group.
            - 2: between AICs and AIVs within the same group.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_cross_flag"), pipe.upper(), flag, mode)


def wait_cross_flag(flag: int):
    """
    Waits for a cross-core synchronization flag.

    This function blocks the current execution stream until the specified hardware
    event ID (flag) is set by `set_cross_flag`.

    Args:
        flag (int): The event ID index to wait for.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_cross_flag"), flag)


def barrier_all():
    """
    Inserts a barrier for all pipeline stages.

    This ensures that all instructions in all pipelines (Scalar, Vector, Cube, MTE, etc.)
    issued before this barrier are completed before any subsequent instructions are executed.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_pipe_barrier"), "ALL")


def pipe_barrier(pipe: _pipe):
    """
    Inserts a barrier for a specific pipeline stage.

    This ensures that all instructions in the specified pipeline issued before
    this barrier are completed before proceeding.

    Args:
        pipe (_pipe): The specific pipeline stage to synchronize (e.g., "MTE3", "V").

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_pipe_barrier"), pipe.upper())


def set_flag(src: _pipe, dst: _pipe, eventId: int):
    """
    Sets a synchronization flag from a source pipeline to a destination pipeline.

    This is part of the standard pipeline synchronization mechanism (Set/Wait).
    It indicates that the source pipeline has completed its task for a specific event.

    Args:
        src (_pipe): The source pipeline stage (producer).
        dst (_pipe): The destination pipeline stage (consumer).
        eventId (int): The event ID used for synchronization.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_set_flag"), src.upper(), dst.upper(), eventId)


def wait_flag(src: _pipe, dst: _pipe, eventId: int):
    """
    Waits for a synchronization flag from a source pipeline.

    This instruction blocks the destination pipeline until the source pipeline
    issues the corresponding `set_flag` command for the given event ID.

    Args:
        src (_pipe): The source pipeline stage (producer) to wait for.
        dst (_pipe): The destination pipeline stage (consumer) that is waiting.
        eventId (int): The event ID used for synchronization.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_wait_flag"),
        src.upper(),
        dst.upper(),
        eventId,
    )


def _legalize_arguments(arg: Buffer | Var):
    """Convert let-bound variables to their corresponding buffers.

    Args:
        arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

    Returns:
        Union[tir.Buffer, tir.Var]: The legalized argument
    """
    if isinstance(arg, Var) and T.has_let_value(arg):
        return T.get_let_value(arg).buffer
    return arg


def _retrieve_shape(object: Buffer | BufferRegion) -> list[int]:
    """
    Retrieves the shape of a Buffer or a BufferRegion.

    If the input is a Buffer, it returns the buffer's shape directly.
    If the input is a BufferRegion (a slice of a buffer), it calculates and returns
    the shape based on the extents of the region's ranges.

    Args:
        object (Union[tir.Buffer, tir.BufferRegion]): The object to query for shape.

    Returns:
        List[int]: A list of integers (or PrimExprs) representing the shape of the object.

    Raises:
        ValueError: If the input object type is not supported.
    """
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


def _retrieve_ptr(object: Buffer | BufferRegion, access_type: str = "r") -> PrimExpr:
    """
    Retrieves the access pointer (handle) for a Buffer or BufferRegion.

    For a full Buffer, it returns the pointer to the beginning of the memory.
    For a BufferRegion, it calculates the linear byte offset based on the region's
    start indices and the underlying buffer's shape (assuming compact row-major layout),
    and returns the pointer to the start of the sliced region.

    Args:
        object (Union[tir.Buffer, tir.BufferRegion]): The buffer object or slice.
        access_type (str, optional): The access mask (e.g., "r" for read, "w" for write).
            Defaults to "r".

    Returns:
        tir.PrimExpr: An expression representing the pointer to the data.

    Raises:
        ValueError: If the input object type is not supported.
    """
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
        return buffer.access_ptr(access_mask=access_type, offset=offset)
    else:
        raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")


def gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False):
    """
    Performs a block-level General Matrix Multiplication (GEMM).

    This function computes the matrix product $C = op(A) \\times op(B)$, where $op$ represents
    an optional transpose operation. It calculates the M, N, and K dimensions based on the
    shapes of the input buffers and generates the corresponding hardware intrinsic call.

    Args:
        A (Union[Buffer, BufferRegion]): The input matrix A. Can be a high-dimensional tensor,
            but the last two dimensions are treated as the matrix dimensions.
        B (Union[Buffer, BufferRegion]): The input matrix B. Can be a high-dimensional tensor,
            but the last two dimensions are treated as the matrix dimensions.
        C (Union[Buffer, BufferRegion]): The output matrix C. Must be a 2D tensor (M, N).
        transpose_A (bool, optional): Whether to transpose matrix A. Defaults to False.
        transpose_B (bool, optional): Whether to transpose matrix B. Defaults to False.
        init (bool, optional): Whether to initialize the accumulator matrix C (typically to zero)
            before computation. Defaults to False.

    Returns:
        tvm.tir.Call: A TIR intrinsic call to `tl.ascend_gemm_v0`.
    """
    A = _legalize_arguments(A)
    B = _legalize_arguments(B)
    C = _legalize_arguments(C)

    A_shape = _retrieve_shape(A)
    B_shape = _retrieve_shape(B)
    C_shape = _retrieve_shape(C)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
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

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    Aptr = _retrieve_ptr(A, "r")
    Bptr = _retrieve_ptr(B, "r")
    Cptr = _retrieve_ptr(C, "w" if init is True else "rw")

    # assert _dtype(A) == _dtype(B), f"gemm A and B dtype mismatch: {_dtype(A)} vs {_dtype(B)}"
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gemm_v0"),
        f"gemm_v0<{_dtype(A)}, {_dtype(C)}, {M}, {N}, {K}, {str(transpose_A).lower()}, {str(transpose_B).lower()}>",
        Aptr,
        Bptr,
        Cptr,
        init,
    )


def fill(buffer: Buffer, value: PrimExpr):
    """Fill a buffer or buffer region with a specified value.

    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with

    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    size = math.prod(buffer.shape)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_fill"),
        f"tl::ascend::Fill<{_dtype(buffer)}>",
        buffer.access_ptr("w"),
        value,
        size,
    )


def binary_op(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr | float,
    op: str,
):
    def _handle_buffer_region(br: BufferRegion, mask):
        bf = br.buffer
        indices = [x.min for x in br.region]
        offset = bf.offset_of(indices)[0]

        extent = [x.extent for x in br.region]
        return bf.access_ptr(mask, offset=offset), extent

    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_extent = dst.shape
    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_extent = src0.shape

    size_0 = math.prod(dst_extent)
    size_1 = math.prod(src0_extent)
    assert size_0 == size_1, "size must be same"
    if isinstance(src1, BufferLoad):
        buffer_1 = src1.buffer
        indices_1 = src1.indices
        return T.call_intrin(
            "handle",
            tir.op.Op.get(f"tl.ascend_{op}s"),
            dst_ptr,
            src0_ptr,
            buffer_1.access_ptr("r"),
            indices_1[0],
            size_0,
        )

    elif isinstance(src1, (PrimExpr, float, int)):
        return T.call_intrin("handle", tir.op.Op.get(f"tl.ascend_{op}s"), dst_ptr, src0_ptr, src1, size_0)
    elif isinstance(src1, BufferRegion):
        src1_ptr, src1_extent = _handle_buffer_region(src1, "r")
        size_2 = math.prod(src1_extent)
        assert size_0 == size_2, "size must be same"

        return T.call_intrin(
            "handle",
            tir.op.Op.get(f"tl.ascend_{op}"),
            dst_ptr,
            src0_ptr,
            src1_ptr,
            size_0,
        )
    else:
        return T.call_intrin(
            "handle",
            tir.op.Op.get(f"tl.ascend_{op}"),
            dst_ptr,
            src0_ptr,
            src1.access_ptr("r"),
            size_0,
        )


def add(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise addition: dst = src0 + src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "add")


def sub(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad):
    """Performs element-wise subtraction: dst = src0 - src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "sub")


def mul(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise multiplication: dst = src0 * src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "mul")


def div(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad):
    """Performs element-wise division: dst = src0 / src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "div")


def max(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise maximum: dst = max(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "max")


def min(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise minimum: dst = min(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "min")


def bitwise_and(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise bitwise AND: dst = src0 & src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_and")


def bitwise_or(dst: Buffer, src0: Buffer, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise bitwise OR: dst = src0 | src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_or")


def unary_op(dst: Buffer, src0: Buffer, op: str):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get(f"tl.ascend_{op}"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        size_0,
    )


def exp(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "exp")


def ln(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "log")


def abs(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "abs")


def reciprocal(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "reciprocal")


def sqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "sqrt")


def rsqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "rsqrt")


def relu(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "relu")


def not_tl(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "bitwise_not")


def scalar_op(dst: Buffer, src0: Buffer, scalar_value: PrimExpr, op_tl: str):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get(f"tl.ascend_{op_tl}"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        scalar_value,
        size_0,
    )


def leaky_relu(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    return scalar_op(dst, src0, scalar_value, "leaky_relu")


def axpy(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    return scalar_op(dst, src0, scalar_value, "axpy")


def reduce(out: Buffer, buffer: Buffer, tmp: Buffer, reduce_type: str, dim: int):
    dtype = _dtype(buffer)
    shape = f"{buffer.shape[0]}, {buffer.shape[1]}"
    assert len(buffer.shape) == 2, "current only support buffer as a 2D tensor"
    buffer = buffer.access_ptr("r")
    out = out.access_ptr("w")
    tmp = tmp.access_ptr("r")

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reduce"),
        f"{reduce_type}<{dtype}, {shape}, {dim}>",
        out,
        buffer,
        tmp,
    )


def reduce_max(out: Buffer, buffer: Buffer, tmp: Buffer, dim: int):
    return reduce(out, buffer, tmp, "reduce_max", dim)


def reduce_sum(out: Buffer, buffer: Buffer, tmp: Buffer, dim: int):
    return reduce(out, buffer, tmp, "reduce_sum", dim)
