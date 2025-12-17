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


def gatherb(dst: Buffer, src0: Buffer, offset: Buffer, repeat_time, dst_blk_stride, dst_rep_stride):
    return T.call_extern("handle", f"tl::ascend::Gatherb<{_dtype(dst)}>", dst.access_ptr("w"),
                         src0.access_ptr("r"), offset.access_ptr("r"), repeat_time, dst_blk_stride, dst_rep_stride)


def select(dst: Union[Buffer, BufferRegion], selMask: Buffer, src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferLoad, PrimExpr], selMode: str):
    def retrieve_shape(object: Union[Buffer, BufferRegion]) -> List[int]:
        if isinstance(object, Buffer):
            return list(object.shape)
        elif isinstance(object, BufferRegion):
            region = object.region
            shape = []
            for r in region:
                shape.append(r.extent)
            return shape
        else:
            raise ValueError(f"Unsupported argument type: {type(object)} for buffer {object}")

    dst_shape = retrieve_shape(dst)
    src0_shape = retrieve_shape(src0)

    assert dst_shape == src0_shape, "dst and src0 must have the same shape"

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

    dst_ptr = retrieve_ptr(dst, "r")
    src0_ptr = retrieve_ptr(src0, "r")
    
    sel_mask_ptr = selMask.access_ptr("r")
    src0_extent = src0_shape

    assert selMode in ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_SCALAR_MODE", "VSEL_TENSOR_TENSOR_MODE"]

    sel_mode = f"AscendC::SELMODE::{selMode}"
    size_0 = math.prod(src0_extent)

    if isinstance(src1, BufferLoad):
        assert selMode in ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_TENSOR_MODE"], "selMode must be VSEL_CMPMASK_SPR or VSEL_TENSOR_TENSOR_MODE"

        src1_type = 0
        buffer_1 = src1.buffer
        indices_1 = src1.indices
        return T.call_extern("handle", f"AscendC::Select", dst_ptr, sel_mask_ptr, src0_ptr, src1_type, buffer_1.access_ptr("r"), indices_1[0], sel_mode, size_0)
    elif isinstance(src1, (PrimExpr, float)):
        assert selMode == "VSEL_TENSOR_SCALAR_MODE", "selMode must be VSEL_TENSOR_SCALAR_MODE"

        src1_type = 1
        return T.call_extern("handle", f"AscendC::Select", dst_ptr, sel_mask_ptr, src0_ptr, src1_type, src1, sel_mode, size_0)
    else:
        assert selMode in ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_TENSOR_MODE"], "selMode must be VSEL_CMPMASK_SPR or VSEL_TENSOR_TENSOR_MODE"

        src1_type = 2
        src1_ptr = src1.access_ptr("r")
        return T.call_extern("handle", f"AscendC::Select", dst_ptr, sel_mask_ptr, src0_ptr, src1_type, src1_ptr, sel_mode, size_0)


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

    elif isinstance(src1, (PrimExpr, float, int)):
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


def min(dst: Buffer, src0: Buffer, src1: Union[Buffer]):
    return binary_op(dst, src0, src1, "Min")

def and_tl(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad, PrimExpr]):
    return binary_op(dst, src0, src1, "And")

def or_tl(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad, PrimExpr]):
    return binary_op(dst, src0, src1, "Or")


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

def not_tl(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Not")


def scalar_op(dst: Buffer, src0: Buffer, scalar_value: PrimExpr, op: str):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::{op}<{_dtype(src0)}>", dst.access_ptr("w"), src0.access_ptr("r"),
                         scalar_value, size_0)


def leaky_relu(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    return scalar_op(dst, src0, scalar_value, "LeakyRelu")


def axpy(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    return scalar_op(dst, src0, scalar_value, "Axpy")


def shiftleft(dst: Buffer, src0: Buffer, scalarValue: PrimExpr):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::ShiftLeft", dst.access_ptr("w"),
                         src0.access_ptr("r"), scalarValue, size_0)


def shiftright(dst: Buffer, src0: Buffer, scalarValue: PrimExpr):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::ShiftRight", dst.access_ptr("w"),
                         src0.access_ptr("r"), scalarValue, size_0)
def sort32(dst: Buffer, src0: Buffer, src1: Buffer):
    repeatTimes = math.prod(src0.shape) // 32
    return T.call_extern("handle", f"AscendC::Sort32", dst.access_ptr("w"),
                         src0.access_ptr("r"), src1.access_ptr("r"), repeatTimes)


def createvecindex(dst: Buffer, firstValue: PrimExpr):
    calCount = math.prod(dst.shape)
    return T.call_extern("handle", f"AscendC::CreateVecIndex", dst.access_ptr("w"),
                         firstValue, calCount)


def transpose(dst: Buffer, src: Buffer):
    return T.call_extern("handle", "AscendC::Transpose", dst.access_ptr("w"), src.access_ptr("r"))


def gather(dst: Buffer, src: Buffer, src_offset: Buffer, src_base_addr: PrimExpr):
    count = math.prod(src.shape)
    return T.call_extern("handle", "AscendC::Gather", dst.access_ptr("w"), src.access_ptr("r"),
                          src_offset.access_ptr("r"), src_base_addr, count)


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


def block_reduce_max(dst: Buffer, src: Buffer, repeat: PrimExpr, mask: PrimExpr, dstPepStride: PrimExpr, srcBlkStride: PrimExpr, srcRepStride: PrimExpr):
    return T.call_extern("handle", "AscendC::BlockReduceMax", dst.access_ptr("w"), src.access_ptr("r"), repeat, mask, dstPepStride, srcBlkStride, srcRepStride)  


def block_reduce_min(dst: Buffer, src: Buffer, repeat: PrimExpr, mask: PrimExpr, dstPepStride: PrimExpr, srcBlkStride: PrimExpr, srcRepStride: PrimExpr):
    return T.call_extern("handle", "AscendC::BlockReduceMin", dst.access_ptr("w"), src.access_ptr("r"), repeat, mask, dstPepStride, srcBlkStride, srcRepStride)    


def block_reduce_sum(dst: Buffer, src: Buffer, repeat: PrimExpr, mask: PrimExpr, dstPepStride: PrimExpr, srcBlkStride: PrimExpr, srcRepStride: PrimExpr):
    return T.call_extern("handle", "AscendC::BlockReduceSum", dst.access_ptr("w"), src.access_ptr("r"), repeat, mask, dstPepStride, srcBlkStride, srcRepStride)    


def compare(dst: Buffer, src0: Buffer, src1: Union[Buffer, BufferLoad, PrimExpr], mode: str):
    assert mode in ["EQ", "NE", "GT", "GE", "LT", "LE"]

    dst_ptr = dst.access_ptr("w")
    dst_extent = dst.shape

    src0_ptr = src0.access_ptr("r")
    src0_extent = src0.shape

    size_0 = math.prod(dst_extent)
    size_1 = math.prod(src0_extent)

    cmp_mode = f"AscendC::CMPMODE::{mode}"
    dst_size = size_1

    if isinstance(src1, BufferLoad):
        buffer_1 = src1.buffer
        indices_1 = src1.indices
        # we only can pass the extra index
        return T.call_extern("handle", f"AscendC::CompareScalar", dst_ptr, src0_ptr,
                             buffer_1.access_ptr("r"), indices_1[0], cmp_mode, dst_size)
    elif isinstance(src1, (PrimExpr, float)):
        return T.call_extern("handle", f"AscendC::CompareScalar", dst_ptr, src0_ptr, src1, cmp_mode, dst_size)
    else:
        return T.call_extern("handle", f"AscendC::Compare", dst_ptr, src0_ptr, src1.access_ptr("r"), cmp_mode, dst_size)


def cast_tl(dst: Buffer, src: Buffer, mode: str, count: PrimExpr):
    assert mode in ["CAST_NONE", "CAST_RINT", "CAST_FLOOR", "CAST_CEIL", "CAST_ROUND", "CAST_TRUNC", "CAST_ODD"]

    round_mode = f"AscendC::RoundMode::{mode}"

    # int32 cast half，roundMode not work，should SetDeqScale(half scale)
    # if (src.dtype == "int32" and dst.dtype == "float16"):
    #     T.call_extern("handle", f"AscendC::SetDeqScale", scale)

    return T.call_extern("handle", f"AscendC::Cast", dst.access_ptr("w"), src.access_ptr("r"), round_mode, count)


def set_deq_scale(scale: PrimExpr):
    return T.call_extern("handle", f"AscendC::SetDeqScale", scale)


def sin(dst: Buffer, src: Buffer, tmp: Buffer):
    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::Sin", dst.access_ptr("w"), src.access_ptr("r"),
                         tmp.access_ptr("r"), size_0)

                
def cos(dst: Buffer, src: Buffer, tmp: Buffer):
    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_extern("handle", f"AscendC::Cos", dst.access_ptr("w"), src.access_ptr("r"),
                         tmp.access_ptr("r"), size_0)

