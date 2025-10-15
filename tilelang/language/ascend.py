import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import List, Union, Literal

import math


def _dtype(buf):
    type_map = {"float16": "half", "float32": "float", "int32": "int", "uint32": "uint32_t"}
    if isinstance(buf, BufferRegion):
        buf = buf.buffer
    return type_map[buf.dtype]


def wait_cross_flag(flag: int):
    return T.call_extern("handle", "AscendC::CrossCoreWaitFlag", flag)


def set_cross_flag(pipe: str, flag: int):
    return T.call_extern("handle", f"AscendC::CrossCoreSetFlag<0x2, PIPE_{pipe.upper()}>", flag)


def barrier_all():
    return T.call_extern("handle", "AscendC::PipeBarrier<PIPE_ALL>")


def gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False):

    def legalize_arguments(arg: Union[Buffer, Var]):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    def retrieve_shape(object: Union[Buffer, BufferRegion]) -> List[int]:
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

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, \
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, \
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    def retrieve_ptr(object: Union[Buffer, BufferRegion], access_type: str = "r") -> PrimExpr:
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

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "rw")

    # assert _dtype(A) == _dtype(B), f"gemm A and B dtype mismatch: {_dtype(A)} vs {_dtype(B)}"
    return T.call_extern(
        "handle",
        f"tl::ascend::gemm_v0<{_dtype(A)}, {_dtype(C)}, {M}, {N}, {K}, {str(transpose_A).lower()}, {str(transpose_B).lower()}>",
        Aptr, Bptr, Cptr, init)


def gemm_v1(A, B, C, transpose_A=False, transpose_B=False, init=False):

    def legalize_arguments(arg: Union[Buffer, Var]):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    def retrieve_shape(object: Union[Buffer, BufferRegion]) -> List[int]:
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

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, \
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, \
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"

    BLOCK_M, BLOCK_N = C_shape
    if not transpose_A:
        L1_BLOCK_M, L1_BLOCK_K = A_shape
    else:
        L1_BLOCK_K, L1_BLOCK_M = A_shape
    L1_BLOCK_N = A_shape[-2] if transpose_B else B_shape[-2]

    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert K == K_B, f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    def retrieve_ptr(object: Union[Buffer, BufferRegion], access_type: str = "r") -> PrimExpr:
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

    Aptr = retrieve_ptr(A, "r")
    Bptr = retrieve_ptr(B, "r")
    Cptr = retrieve_ptr(C, "rw")

    # assert _dtype(A) == _dtype(B), f"gemm A and B dtype mismatch: {_dtype(A)} vs {_dtype(B)}"
    return T.call_extern(
        "handle",
        f"tl::ascend::gemm_v1<{_dtype(A)}, {_dtype(C)}, {L1_BLOCK_M}, {L1_BLOCK_N}, {L1_BLOCK_K}, {BLOCK_M}, {BLOCK_N}, {L1_BLOCK_K}, {str(transpose_A).lower()}, {str(transpose_B).lower()}>",
        Aptr, Bptr, Cptr, init)


def fill(buffer: Buffer, value: PrimExpr):
    """Fill a buffer or buffer region with a specified value.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with
    
    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    # AscendC::Duplicate(ubOut, value, Len);

    size = math.prod(buffer.shape)
    return T.call_extern("handle", f"AscendC::Duplicate<{_dtype(buffer)}>", buffer.access_ptr("w"),
                         value, size)


def arith_progression(buffer: Buffer, first_value, diff_value, count):
    return T.call_extern("handle", f"AscendC::ArithProgression<{_dtype(buffer)}>",
                         buffer.access_ptr("w"), first_value, diff_value, count)


def sort(dst: BufferRegion, src: Buffer, indices: Buffer, tmp_buffer: Buffer, repeat_time):

    def _handle_buffer_region(br: BufferRegion, mask):
        bf = br.buffer
        indices = [x.min for x in br.region]
        offset = bf.offset_of(indices)[0]
        extent = [x.extent for x in br.region]
        return bf.access_ptr(mask, offset=offset), extent

    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
        dst_size = math.prod(dst_extent)
        return T.call_extern("handle", f"AscendC::Sort<{_dtype(dst)}, true>", dst_ptr,
                             src.access_ptr("r"), indices.access_ptr("r"),
                             tmp_buffer.access_ptr("r"), dst_size, repeat_time)


def merge_sort(dst: Buffer, src: Buffer, block_size, block_num, is_copy):
    return T.call_extern("handle", f"tl::ascend::MergeSort<{_dtype(dst)}>", dst.access_ptr("w"),
                         src.access_ptr("r"), block_size, block_num, is_copy)


def topk(dst: Buffer, src: Buffer, tmp: Buffer, block_size):
    return T.call_extern("handle", f"tl::ascend::TopK<{_dtype(dst)}>", dst.access_ptr("w"),
                         src.access_ptr("r"), tmp.access_ptr("r"), block_size)


def gather_mask(dst: Buffer, src: Buffer, num):
    return T.call_extern("handle", f"tl::ascend::GatherMask<{_dtype(dst)}>", dst.access_ptr("w"),
                         src.access_ptr("r"), num)


def init_sort_buf(buffer: Buffer, num, rsv):
    pass
    return T.call_extern("handle", f"tl::ascend::InitSortBuf<{_dtype(buffer)}>",
                         buffer.access_ptr("w"), rsv, num)


def brcb(dst: Buffer, src: Buffer, repeat_times: PrimExpr, dst_blk_stride: PrimExpr,
         dst_repeat_stride: PrimExpr):
    """AscendC brcb wrapper
    """

    src_size = math.prod(src.shape)
    assert src_size >= (repeat_times * 8), "src size must be not less then repeat_times * 8"

    src_ptr = src.access_ptr("r")
    dst_ptr = dst.access_ptr("w")

    return T.call_extern("handle", f"tl::ascend::brcb<{_dtype(src)}>", dst_ptr, src_ptr,
                         repeat_times, dst_blk_stride, dst_repeat_stride)


def binary_op(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion],
              src1: Union[Buffer, BufferLoad, PrimExpr, float], op: str):

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
        # we only can pass the extra index
        return T.call_extern("handle", f"AscendC::{op}s", dst_ptr, src0_ptr,
                             buffer_1.access_ptr("r"), indices_1[0], size_0)

    elif isinstance(src1, (PrimExpr, float)):
        return T.call_extern("handle", f"AscendC::{op}s", dst_ptr, src0_ptr, src1, size_0)
    else:
        return T.call_extern("handle", f"AscendC::{op}", dst_ptr, src0_ptr, src1.access_ptr("r"),
                             size_0)


def add(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad, PrimExpr]):
    return binary_op(dst, src0, src1, "Add")


def sub(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad]):
    return binary_op(dst, src0, src1, "Sub")


def mul(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad, PrimExpr]):
    return binary_op(dst, src0, src1, "Mul")


def div(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad]):
    return binary_op(dst, src0, src1, "Div")


def max(dst: Buffer, src0: Buffer, src1: Union[Buffer]):
    return binary_op(dst, src0, src1, "Max")


def unary_op(dst: Buffer, src0: Buffer, op: str):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::{op}", dst.access_ptr("w"), src0.access_ptr("r"),
                         size_0)


def exp(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Exp")


def ln(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Ln")


def abs(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Abs")


def reciprocal(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Reciprocal")


def sqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Sqrt")


def rsqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Rsqrt")


def relu(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Relu")


def reduce(out: Buffer, buffer: Buffer, tmp: Buffer, reduce_type: str, dim: int):
    dtype = _dtype(buffer)
    shape = f"{buffer.shape[0]}, {buffer.shape[1]}"
    assert len(buffer.shape) == 2, "current only support buffer as a 2D tensor"

    buffer = buffer.access_ptr("r")
    out = out.access_ptr("w")
    tmp = tmp.access_ptr("r")
    if dim == -1:
        pattern = "AscendC::Pattern::Reduce::AR"
    else:
        pattern = "AscendC::Pattern::Reduce::RA"

    return T.call_extern("handle", f"tl::ascend::{reduce_type}<{dtype}, {shape}, {pattern}>", out,
                         buffer, tmp)


def reduce_max(out: Buffer, buffer: Buffer, tmp: Buffer, dim: int):

    return reduce(out, buffer, tmp, "reduce_max", dim)


def reduce_sum(out: Buffer, buffer: Buffer, tmp: Buffer, dim: int):

    return reduce(out, buffer, tmp, "reduce_sum", dim)


_pipe = Literal["fix", "mte1", "mte2", "mte3", "m", "v"]


def set_flag(src: _pipe, dst: _pipe, eventId: int):
    return T.call_extern("handle",
                         f"AscendC::SetFlag<AscendC::HardEvent::{src.upper()}_{dst.upper()}>",
                         eventId)


def wait_flag(src: _pipe, dst: _pipe, eventId: int):
    return T.call_extern("handle",
                         f"AscendC::WaitFlag<AscendC::HardEvent::{src.upper()}_{dst.upper()}>",
                         eventId)


def pipe_barrier(pipe: _pipe):
    return T.call_extern("handle", f"AscendC::PipeBarrier<PIPE_{pipe.upper()}>")
