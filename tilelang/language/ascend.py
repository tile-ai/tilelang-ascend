from __future__ import annotations
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, Var
from typing import Union, Literal  # noqa: F401, UP035
from tvm import tir


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
        "int4": "int4b_t",
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


def wait_cross_flag(flag: int, pipe: _pipe | Literal[""] = ""):
    """
    Waits for a cross-core synchronization flag.

    This function blocks the current execution stream until the specified hardware
    event ID (flag) is set by `set_cross_flag`.

    Args:
        flag (int): The event ID index to wait for.
        pipe (str, optional): The specific execution pipe to wait on (e.g., "mte1", "fix").
            Defaults to "".
            **Note:** This parameter is only supported on the **A5 platform**.
            For other architectures, this must be left as an empty string.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_wait_cross_flag"), flag, pipe)


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


def sync_all():
    """
    Performs a global synchronization across the compute unit (block/core).

    This generally ensures memory consistency and execution synchronization
    across the entire block/core scope.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_sync_all"))


def shmem_put_nbi(dst: Buffer, src: Buffer, nelems: PrimExpr, newPe: PrimExpr):
    """Performs a shmem put nbi operation.

    This intrinsic invokes the underlying implementation to copy from the local GM to the newPe GM

    Args:
        dst: The newPe GM.
        src: The local GM.
        nelems: Number of elements.
        newPe: The rank of dst pe.

    Returns:
        A TVM intrinsic call that performs the shmem put nbi operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_shmem_put_nbi"),
        f"shmem_put_nbi<{_dtype(src)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        nelems,
        newPe,
    )


def shmem_ub_put_nbi(ub: Buffer, dst: Buffer, nelems: PrimExpr, newPe: PrimExpr, strelem: PrimExpr = 0):
    """Performs a shmem ub put nbi operation.

    This intrinsic invokes the underlying implementation to copy from the local UB to the newPe GM

    Args:
        ub: The local UB.
        dst: The newPe GM.
        nelems: Number of elements.
        newPe: The rank of dst pe.

    Returns:
        A TVM intrinsic call that performs the shmem ub put nbi operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_shmem_ub_put_nbi"),
        f"shmem_ub_put_nbi<{_dtype(dst)}>",
        ub.access_ptr("r"),
        dst.access_ptr("w"),
        nelems,
        newPe,
        strelem,
    )


def shmem_get_nbi(dst: Buffer, src: Buffer, nelems: PrimExpr, newPe: PrimExpr):
    """Performs a shmem get nbi operation.

    This intrinsic invokes the underlying implementation to copy from the newPe GM to the local GM

    Args:
        dst: The local GM.
        src: The newPe GM.
        nelems: Number of elements.
        newPe: The rank of dst pe.

    Returns:
        A TVM intrinsic call that performs the shmem get nbi operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_shmem_get_nbi"),
        f"shmem_get_nbi<{_dtype(src)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        nelems,
        newPe,
    )


def shmem_ub_get_nbi(dst: Buffer, src: Buffer, nelems: PrimExpr, newPe: PrimExpr):
    """Performs a shmem ub get nbi operation.

    This intrinsic invokes the underlying implementation to copy from the newPe GM to the local UB

    Args:
        dst: The local UB.
        src: The newPe GM.
        nelems: Number of elements.
        newPe: The rank of dst pe.

    Returns:
        A TVM intrinsic call that performs the shmem ub get nbi operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_shmem_ub_get_nbi"),
        f"shmem_ub_get_nbi<{_dtype(src)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        nelems,
        newPe,
    )


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
    if len(C_shape) > 2:
        for i in range(len(C_shape) - 2):
            assert C_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape[-2], C_shape[-1]
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


_MX_FORMAT_TO_CTYPE = {
    "e4m3": "float8_e4m3_t",
    "e5m2": "float8_e5m2_t",
    "e2m1x2": "float4_e2m1x2_t",
    "e1m2x2": "float4_e1m2x2_t",
}

_MX_FP4_FORMATS = ("e2m1x2", "e1m2x2")


def gemm_mx(A, B, C, scale_a, scale_b, init=False, format=None):
    """
    OCP MX (Microscaling) block GEMM on the A5 Cube unit.

    Computes C = (A * scale_a) @ (B * scale_b) where A, B are per-element low-precision
    data (MXFP8 / MXFP4), and scale_a / scale_b are per-32-K-block e8m0 exponents.

    Args:
        A (Buffer | BufferRegion): MXFP data matrix.
            For MXFP8 (format in {"e4m3", "e5m2"}): shape is (M, K_logical), dtype is uint8
                (each byte holds one FP8 element in storage terms). The buffer is typed
                as uint8 but interpreted as float8_e4m3_t / float8_e5m2_t at the Cube.
            For MXFP4 (format in {"e2m1x2", "e1m2x2"}): shape is (M, K_logical) uint8
                — each uint8 byte packs two FP4 values; only the first K_logical/2 bytes
                in each row carry real data. K_logical must be even.
        B (Buffer | BufferRegion): MXFP data matrix (K_logical, N). Same dtype convention
            as A, mirrored along the K dimension.
        C (Buffer | BufferRegion): Accumulator matrix (M, N), dtype must be float32.
        scale_a (Buffer | BufferRegion): Per-row-per-32-K scale buffer (M, K_logical/32),
            dtype uint8 (e8m0). Layout is logical-K-oriented, independent of data format.
        scale_b (Buffer | BufferRegion): Per-32-K-per-col scale buffer (K_logical/32, N),
            dtype uint8 (e8m0).
        init (bool, optional): When True, clears the C L0C accumulator on first use.
        format (str, optional): One of "e4m3", "e5m2", "e2m1x2", "e1m2x2".
            When None, the format is inferred from the A buffer's dtype when it is
            one of e4m3_float8 / e5m2_float8; otherwise it defaults to "e5m2" (MXFP8).

    Returns:
        tvm.tir.Call: A TIR intrinsic call to `tl.ascend_gemm_mx`.
    """
    A = _legalize_arguments(A)
    B = _legalize_arguments(B)
    C = _legalize_arguments(C)
    scale_a = _legalize_arguments(scale_a)
    scale_b = _legalize_arguments(scale_b)

    A_shape = _retrieve_shape(A)
    B_shape = _retrieve_shape(B)
    C_shape = _retrieve_shape(C)
    sA_shape = _retrieve_shape(scale_a)
    sB_shape = _retrieve_shape(scale_b)

    assert len(C_shape) == 2, "gemm_mx only supports C as a 2D tensor"
    assert len(A_shape) == 2, "gemm_mx only supports A as a 2D tensor"
    assert len(B_shape) == 2, "gemm_mx only supports B as a 2D tensor"
    assert len(sA_shape) == 2, "gemm_mx only supports scale_a as a 2D tensor"
    assert len(sB_shape) == 2, "gemm_mx only supports scale_b as a 2D tensor"

    if format is None:
        buf_dtype = A.buffer.dtype if isinstance(A, BufferRegion) else A.dtype
        if buf_dtype == "e4m3_float8":
            format = "e4m3"
        elif buf_dtype == "e5m2_float8":
            format = "e5m2"
        else:
            format = "e5m2"
    assert format in _MX_FORMAT_TO_CTYPE, f"T.gemm_mx unknown format {format!r}; expected one of {sorted(_MX_FORMAT_TO_CTYPE.keys())}"

    M, N = C_shape
    K = A_shape[-1]
    K_B = B_shape[-2]
    assert K == K_B, f"T.gemm_mx K shape check failed: K_A = {K}, K_B = {K_B}"
    if format in _MX_FP4_FORMATS:
        assert K % 2 == 0, f"T.gemm_mx MXFP4 requires K divisible by 2 (packed bytes), got K={K}"
    assert K % 64 == 0, f"T.gemm_mx requires K divisible by 64, got K={K}"
    assert tuple(sA_shape) == (M, K // 32), f"scale_a shape must be (M={M}, K//32={K // 32}), got {sA_shape}"
    assert tuple(sB_shape) == (K // 32, N), f"scale_b shape must be (K//32={K // 32}, N={N}), got {sB_shape}"

    Aptr = _retrieve_ptr(A, "r")
    Bptr = _retrieve_ptr(B, "r")
    Cptr = _retrieve_ptr(C, "w" if init is True else "rw")
    sAptr = _retrieve_ptr(scale_a, "r")
    sBptr = _retrieve_ptr(scale_b, "r")

    data_type_input = _MX_FORMAT_TO_CTYPE[format]
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gemm_mx"),
        f"gemm_mx<{data_type_input}, {_dtype(C)}, {M}, {N}, {K}>",
        Aptr,
        Bptr,
        Cptr,
        sAptr,
        sBptr,
        init,
    )


def printf(format_str: str, *args):
    """
    Prints formatted output.

    This function processes the format string and arguments (handling string escaping
    and Buffer pointer conversion) before generating the hardware intrinsic call.
    It is commonly used for debugging kernel logic.

    Args:
        format_str (str): The format string (C-style), e.g., "Value: %f\n".
        *args: Variable arguments to be formatted. Buffers are automatically converted
            to their access pointers.

    Returns:
        tvm.tir.Call: A TIR intrinsic call to `tl.ascend_printf`.
    """
    format_str = format_str.replace("%p", "0x%x")
    escaped_format = format_str.encode("unicode_escape").decode("utf-8")

    args_list = list(args)
    for i in range(len(args_list)):
        if isinstance(args_list[i], Buffer):
            args_list[i] = args_list[i].access_ptr("r")
        if isinstance(args_list[i], str):
            args_list[i] = args_list[i].encode("unicode_escape").decode("utf-8")
    new_args = tuple(args_list)

    all_args = (escaped_format,) + new_args
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_printf"), *all_args)


def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple = ()):
    """
    Dumps the data of a specific tensor to the host for debugging.

    It allows inspecting intermediate tensor values during hardware execution.

    Args:
        tensor (Buffer): The target buffer/tensor to dump.
        desc (int): A user-defined descriptor ID (uint32) to identify this dump operation.
        dump_size (int): The size of the data to dump (uint32).
        shape_info (tuple, optional): A tuple describing the shape dimensions of the tensor.
            Defaults to an empty tuple.

    Returns:
        tvm.tir.Call: A TIR intrinsic call to `tl.ascend_dump_tensor`.

    Raises:
        ValueError: If `desc` or `dump_size` are not valid uint32 integers.
    """
    if not isinstance(desc, int) or desc < 0 or desc > 0xFFFFFFFF:
        raise ValueError(f"desc must be uint32, but your desc is {desc}")
    # if not isinstance(dump_size, int) or dump_size < 0 or dump_size > 0xFFFFFFFF:
    #     raise ValueError(f"dump_size must be uint32, but your dump_size is {dump_size}")

    tensor_ptr = tensor.access_ptr("r")
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_dump_tensor"),
        tensor_ptr,
        desc,
        dump_size,
        len(shape_info),
        *shape_info,
    )


def reinterpretcast(dst: Buffer, src: Buffer, casttype: str):

    # return T.call_extern("handle", f"ReinterpretCast", dst.access_ptr("w"), src.access_ptr("r"),
    #                      casttype)
    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_reinterpretcast"), dst.access_ptr("w"), src.access_ptr("r"), casttype)


def set_deq_scale(scale: PrimExpr):
    """
    Sets the dequantization scale factor register.

    This function configures the hardware environment with a specific scaling factor,
    typically used in quantized matrix multiplication or convolution operations
    where results need to be dequantized (e.g., int32 -> fp16).

    Args:
        scale (PrimExpr): The scaling factor value.

    Returns:
        tvm.tir.Call: A TIR intrinsic call to `tl.ascend_set_deq_scale`.
    """
    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_set_deq_scale"), scale)
