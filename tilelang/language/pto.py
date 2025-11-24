import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import List, Union, Literal
from tvm import tir

import math


def _dtype(buf):
    type_map = {"float16": "half", "float32": "float", "int32": "int", "uint32": "uint32_t", "bfloat16": "bfloat16_t", "uint16": "uint16_t", "uint8": "uint8_t",
                "int8": "int8_t", "int16": "int16_t", "int64": "int64_t", "uint64": "uint64_t"}
    if isinstance(buf, BufferRegion):
        buf = buf.buffer
    return type_map[buf.dtype]


# Todo: PTO Barrier_all
def barrier_all():
    return T.call_extern("handle", "")


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

    # Transform to Buffer Pointer
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
        f"tl::pto::gemm_v0<{_dtype(A)}, {_dtype(C)}, {M}, {N}, {K}, {str(transpose_A).lower()}, {str(transpose_B).lower()}>",
        Aptr, Bptr, Cptr, init)

def fill(buffer: Buffer, value: PrimExpr):
    """Fill a buffer or buffer region with a specified value.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with
    
    Returns:
        A TVM intrinsic call that performs the fill operation
    """

    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_fill"), f"TEXPANDS", buffer.access_ptr("w"), value)

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
        return T.call_intrin("handle", tir.op.Op.get("tl.ascend_binary_ops"), f"T{op.upper()}S", dst_ptr, src0_ptr,
                             buffer_1.access_ptr("r"), indices_1[0])

    elif isinstance(src1, (PrimExpr, float)):
        return T.call_intrin("handle", tir.op.Op.get("tl.ascend_binary_ops"), f"T{op.upper()}S", dst_ptr, src0_ptr, src1)
    else:
        return T.call_intrin("handle", tir.op.Op.get("tl.ascend_binary_op"), f"T{op.upper()}", dst_ptr, src0_ptr, src1.access_ptr("r"))

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

    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_unary_op"), f"T{op.upper()}", dst.access_ptr("w"), src0.access_ptr("r"))


def exp(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Exp")


def ln(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Log")


def abs(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Abs")


def reciprocal(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "RECIP")


def sqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Sqrt")


def rsqrt(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Rsqrt")


def relu(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Relu")

def not_tl(dst: Buffer, src0: Buffer):
    return unary_op(dst, src0, "Not")

def scalar_op(dst: Buffer, src0: Buffer, scalar_value: PrimExpr, op_name: str):
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_scalar_op"), op_name, dst.access_ptr("w"), src0.access_ptr("r"),
                         scalar_value)


def leaky_relu(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    return scalar_op(dst, src0, scalar_value, "TLRELU")

def reduce(out: Buffer, buffer: Buffer, reduce_type: str, dim: int):
    dtype = _dtype(buffer)
    out_dtype = _dtype(out)
    shape = f"{buffer.shape[0]}, {buffer.shape[1]}"
    assert len(buffer.shape) == 2, "current only support buffer as a 2D tensor"
    assert dtype == out_dtype, "src & dst dtype must be equal"

    buffer = buffer.access_ptr("r")
    out = out.access_ptr("w")
    op_name = "TROWMAX"
    if dim == -1:
        if reduce_type == "max":
            op_name = "TROWMAX"
        else:
            op_name = "TROWSUM"
    else:
        if reduce_type == "max":
            op_name = "TCOLMAX"
        else:
            op_name = "TCOLSUM"

    return T.call_intrin("handle", tir.op.Op.get("tl.ascend_reduce"), op_name, out,
                         buffer)

def reduce_max(out: Buffer, buffer: Buffer, dim: int):

    return reduce(out, buffer, "max", dim)


def reduce_sum(out: Buffer, buffer: Buffer, dim: int):

    return reduce(out, buffer, "sum", dim)