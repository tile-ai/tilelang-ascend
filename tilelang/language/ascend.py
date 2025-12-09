import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import List, Union, Literal
import numpy as np

import math


def _dtype(buf):
    type_map = {"float16": "half", "float32": "float", "int32": "int", "uint32": "uint32_t", "bfloat16": "bfloat16_t", "uint16": "uint16_t", "uint8": "uint8_t",
                "int8": "int8_t", "int16": "int16_t", "int64": "int64_t", "uint64": "uint64_t"}
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


def sync_all():
    return T.call_extern("handle", f"AscendC::SyncAll<false>")


def printf(format_str: str, *args):
    format_str =  format_str.replace('%p', '0x%x')
    escaped_format = format_str.encode('unicode_escape').decode('utf-8')

    args_list = list(args)
    for i in range(len(args_list)):
        if isinstance(args_list[i], Buffer):
            args_list[i] = args_list[i].access_ptr("r")
        if isinstance(args_list[i], str):
            args_list[i] = args_list[i].encode('unicode_escape').decode('utf-8')
    new_args = tuple(args_list)

    all_args = (escaped_format, ) + new_args
    return T.call_extern("handle", f"AscendC::PRINTF", *all_args)


def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple=()):
    if not isinstance(desc, int) or desc < 0 or desc > 0xFFFFFFFF:
        raise ValueError(f"desc must be uint32, but your desc is {desc}")
    if not isinstance(dump_size, int) or dump_size < 0 or dump_size > 0xFFFFFFFF:
        raise ValueError(f"dump_size must be uint32, but your dump_size is {dump_size}")

    tensor_ptr = tensor.access_ptr("r")
    if (len(shape_info) == 0):
        return T.call_extern("handle", f"AscendC::DumpTensor", tensor_ptr, desc, dump_size)
    else:
        return T.call_extern("handle", f"tl::ascend::DumpTensor", tensor_ptr, desc, dump_size, len(shape_info), *shape_info)
