"""The language interface for tl programs."""

import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Var
from typing import List, Union, Optional
from tvm import ir, tir
from tvm import runtime
from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from tilelang import _ffi_api
from .kernel import get_thread_bindings, get_thread_extents, FrameStack
import threading
import os

from tilelang.language.copy import buffer_region_to_tile_region, buffer_load_to_tile_region, region

def _get_extent(data):
    if isinstance(data, tir.Var) and T.has_let_value(data):
        data = T.get_let_value(data)
    result = []
    if isinstance(data, tir.Buffer):
        result = data.shape
    elif isinstance(data, tir.BufferRegion):
        result = [x.extent for x in data.region]
    return result

def _buffer_to_tile_region_with_extent(buffer: tir.Buffer, access_type: str, extent:[]):
    """Convert a TVM buffer to a tile region descriptor.

    Args:
        buffer (tir.Buffer): The buffer to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        extent ([]): buffer extent

    Returns:
        tir.Call: A region descriptor covering the entire buffer
    """
    mins = [0 for _ in buffer.shape]
    return region(T.BufferLoad(buffer, mins), access_type, *extent)

def _to_region(data, access_type, extent):
    if isinstance(data, tir.Var) and T.has_let_value(data):
        data = T.get_let_value(data)
    if isinstance(data, tir.Buffer):
        return _buffer_to_tile_region_with_extent(data, access_type, extent)
    elif isinstance(data, tir.BufferRegion):
        return buffer_region_to_tile_region(data, access_type, extent[-len(data.buffer.shape):])
    elif isinstance(data, tir.IntImm) or isinstance(data, tir.FloatImm):
        return data
    else:
        return buffer_load_to_tile_region(data, access_type, extent[-len(data.buffer.shape):])

def _legalize_dim(buffer: tir.Buffer, dim: int):
    if dim < 0:
        dim = len(buffer.shape) + dim
    return dim

def npuir_copy(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad],
    size: [] = []
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        size ([]): buffer extent

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    if size == []:
        src_extent = _get_extent(src)
        dst_extent = _get_extent(dst)
        assert src_extent or dst_extent, "Can't deduce copy extents from args"
        src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
        dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
        extent = max(src_extent, dst_extent)
    else:
        extent = size
    src = _to_region(src, "r", extent)
    dst = _to_region(dst, "w", extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_copy"), src, dst)

class AscendBinaryOp(object):
    """
    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir binary operation
    """
    def __init__(self, opName, src0, src1, dst):
        self.__opName = opName
        self.__src0 = src0
        self.__src1 = src1
        self.__dst = dst
    def buildTirCall(self):
        src0 = _to_region(self.__src0, "r", _get_extent(self.__src0))
        src1 = _to_region(self.__src1, "r", _get_extent(self.__src1))
        dst = _to_region(self.__dst, "w", _get_extent(self.__dst))
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_" + self.__opName), src0, src1, dst)

"""npuir add at tile-level."""
def npuir_add(A, B, C):
    return AscendBinaryOp("add", A, B, C).buildTirCall()

"""npuir sub at tile-level."""
def npuir_sub(A, B, C):
    return AscendBinaryOp("sub", A, B, C).buildTirCall()

"""npuir mul at tile-level."""
def npuir_mul(A, B, C):
    return AscendBinaryOp("mul", A, B, C).buildTirCall()

"""npuir div at tile-level."""
def npuir_div(A, B, C):
    return AscendBinaryOp("div", A, B, C).buildTirCall()

"""npuir max at tile-level."""
def npuir_max(A, B, C):
    return AscendBinaryOp("max", A, B, C).buildTirCall()

"""npuir min at tile-level."""
def npuir_min(A, B, C):
    return AscendBinaryOp("min", A, B, C).buildTirCall()

"""npuir or at tile-level."""
def npuir_or(A, B, C):
    return AscendBinaryOp("or", A, B, C).buildTirCall()

"""npuir and at tile-level."""
def npuir_and(A, B, C):
    return AscendBinaryOp("and", A, B, C).buildTirCall()

"""npuir xor at tile-level."""
def npuir_xor(A, B, C):
    return AscendBinaryOp("xor", A, B, C).buildTirCall()

"""npuir pow at tile-level."""
def npuir_pow(A, B, C):
    return AscendBinaryOp("pow", A, B, C).buildTirCall()

"""npuir shl at tile-level."""
def npuir_shl(A, B, C):
    return AscendBinaryOp("shl", A, B, C).buildTirCall()


class AscendUnaryOp(object):
    """
    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir unary operation
    """
    def __init__(self, opName, src, dst):
        self.__opName = opName
        self.__src = src
        self.__dst = dst
    def buildTirCall(self):
        src = _to_region(self.__src, "r", _get_extent(self.__src))
        dst = _to_region(self.__dst, "w", _get_extent(self.__dst))
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_" + self.__opName), src, dst)

"""npuir exp at tile-level."""
def npuir_exp(A, B):
    return AscendUnaryOp("exp", A, B).buildTirCall()

"""npuir relu at tile-level."""
def npuir_relu(A, B):
    return AscendUnaryOp("relu", A, B).buildTirCall()

"""npuir ln at tile-level."""
def npuir_ln(A, B):
    return AscendUnaryOp("ln", A, B).buildTirCall()

"""npuir sqrt at tile-level."""
def npuir_sqrt(A, B):
    return AscendUnaryOp("sqrt", A, B).buildTirCall()

"""npuir rsqrt at tile-level."""
def npuir_rsqrt(A, B):
    return AscendUnaryOp("rsqrt", A, B).buildTirCall()

"""npuir abs at tile-level."""
def npuir_abs(A, B):
    return AscendUnaryOp("abs", A, B).buildTirCall()

"""npuir rec at tile-level."""
def npuir_rec(A, B):
    return AscendUnaryOp("rec", A, B).buildTirCall()

"""npuir not at tile-level."""
def npuir_not(A, B):
    return AscendUnaryOp("not", A, B).buildTirCall()

def npuir_select(Cond, A, B, Out):
    """npuir select at tile-level.

    Args:
        Cond (Union[tir.Buffer, tir.Var]): Input argument to legalize
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        Out (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_select operation
    """

    Cond = _to_region(Cond, "r", _get_extent(A))
    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    Out = _to_region(Out, "w", _get_extent(Out))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_select"), Cond, A, B, Out)


def npuir_cmp(A, B, C, cmp_mod):
    """npuir cmp at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_cmp operation
    """

    valid_cmp_mode = {"eq", "ne", "lt", "gt", "ge", "le"}
    assert cmp_mod in valid_cmp_mode, "cmp mode is invalid."

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_cmp"), A, B, C, cmp_mod)

def npuir_shr(A, B, C, round: bool = True):
    """npuir shift right at tile-level.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
    Returns:
        tir.Call: A handle to the npuir_shr operation
    """

    A = _to_region(A, "r", _get_extent(A))
    B = _to_region(B, "r", _get_extent(B))
    C = _to_region(C, "w", _get_extent(C))
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_shr"), A, B, C, round)


def npuir_dot(A: Union[tir.Buffer, tir.Var],
    B: Union[tir.Buffer, tir.Var],
    C: Union[tir.Buffer, tir.Var],
    size: [] = [], initC: bool = False, a_transpose: bool = False, b_transpose: bool = False):
    """npuir dot at tile-level. C = C + A * B.

    Args:
        A (Union[tir.Buffer, tir.Var]): Input argument to legalize
        B (Union[tir.Buffer, tir.Var]): Input argument to legalize
        C (Union[tir.Buffer, tir.Var]): Output argument to legalize
        initC (bool): whether to initialize L0C value to zero (C = A * B)
        a_transpose (bool): Matrix A is transposed before load
        b_transpose (bool): Matrix B is transposed before load
    Returns:
        tir.Call: A handle to the npuir_dot operation
    """

    if size == []:
        A_extent = _get_extent(A)
        B_extent = _get_extent(B)
        C_extent = _get_extent(C)
    else:
        assert len(size) == 3, "size must contains [m, k, n]"
        A_extent = [size[0], size[1]]
        B_extent = [size[1], size[2]]
        C_extent = [size[0], size[2]]

    A = _to_region(A, "r", A_extent)
    B = _to_region(B, "r", B_extent)
    C = _to_region(C, "rw", C_extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_dot"), A, B, C, initC, a_transpose, b_transpose)


def npuir_load_nd2nz(src, dst, size = []):
    """npuir nd2nz-load data from OUT to L1 at tile-level.

    Args:
        src (Union[tir.Buffer, tir.Var]): Input argument to legalize
        dst (Union[tir.Buffer, tir.Var]): Output argument to legalize
        size ([]): buffer extent
    Returns:
        tir.Call: A handle to the npuir_load_nd2nz operation
    """

    src = _to_region(src, "r", _get_extent(src) if size is [] else size)
    dst = _to_region(dst, "w", _get_extent(dst) if size is [] else size)
    # dst_continuous: whether the source data is stored continuously in the destination buffer.
    # It is good to always set dst_continuous to True.
    dst_continuous = True
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_load_nd2nz"), src, dst, dst_continuous)

def npuir_store_nz2nd(src, dst, size=[]):
    """npuir nz2nd-store data from L1 to gm at tile-level.

    Args:
        src (Union[tir.Buffer, tir.Var]): Input argument to legalize
        dst (Union[tir.Buffer, tir.Var]): Output argument to legalize
        size ([]): buffer extent
    Returns:
        tir.Call: A handle to the npuir_load_nd2nz operation
    """

    src = _to_region(src, "r", _get_extent(src) if size is [] else size)
    dst = _to_region(dst, "w", _get_extent(dst) if size is [] else size)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_store_nz2nd"), src, dst)


def npuir_store_fixpipe(src, dst, size = [], enable_nz2nd = False, channel_split = False, pre_relu_mode = ""):
    """npuir nd2nz-load data from OUT to L1 at tile-level.

    Args:
        src (tir.Buffer): Input argument to legalize
        dst (tir.Buffer): Output argument to legalize
        size ([]): buffer extent
        enable_nz2nd (bool): whether enable nz2nd when store to OUT
        channel_split (bool): whether split channel when store to OUT
        pre_relu_mode (str): "", "relu", "leaky_relu", "prelu"
    Returns:
        tir.Call: A handle to the npuir_store_fixpipe operation
    """

    assert((src.dtype == dst.dtype)
           or (src.dtype == "float32" and dst.dtype == "float16")
           or (src.dtype == "float32" and dst.dtype == "bfloat16")
           or (src.dtype == "int32" and dst.dtype == "int8")), \
            "Unexpected pre-quant mode in npuir_store_fixpipe"

    src = _to_region(src, "r", _get_extent(src) if size is [] else size)
    dst = _to_region(dst, "w", _get_extent(dst) if size is [] else size)
    pre_relu_map = {"": 0, "relu": 1, "leaky_relu": 2, "prelu": 3}
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_store_fixpipe"), src, dst,
                           enable_nz2nd, channel_split, pre_relu_map[pre_relu_mode])

def npuir_brc(src, dst):
    """Broadcast a vector or a scalar according to the broadcast axes array

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion, tir.var]): Source vector or scalar
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector

    Raises:
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If input and output shapes do not match for broadcast.

    Returns:
        tir.Call: A handle to the npuir_brc operation
    """
    src_extent = _get_extent(src)
    dst_extent = _get_extent(dst)

    if not isinstance(src, tir.Var):
        assert len(src_extent) == len(
            dst_extent), "The input vector and output vector must have same rank."

        for i in range(0, len(src_extent)):
            if src_extent[i] != 1:
                assert src_extent[i] == dst_extent[
                    i], "The input and output shapes do not match for broadcast."
    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_brc"), src, dst)


def npuir_cast(src, dst, size=[], round_mode="rint"):
    """Performs element-wise operation on N operands and produces a single result.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        round_mode: Round mode (round/rint/floor/ceil/trunc/odd)

    Raises:
        AssertionError: If input is not vector.
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If round mode is invalid.
        AssertionError: If input and output shapes do not match for broadcast.

    Returns:
        tir.Call: A handle to the npuir_cast operation
    """
    broadcast_dims = []
    valid_round_mode = {"round", "rint", "floor", "ceil", "trunc", "odd"}
    src_extent = _get_extent(src) if size == [] else size
    dst_extent = _get_extent(dst) if size == [] else size

    assert not isinstance(src, tir.Var), "The first input is vector-only."
    assert len(src_extent) == len(
        dst_extent), "The input/init operands and result have the same rank."
    assert round_mode in valid_round_mode, "Round mode is invalid."

    for i in range(0, len(src_extent)):
        if src_extent[i] != 1:
            assert src_extent[i] == dst_extent[
                i], "The input and output shapes do not match for broadcast."

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_cast"), src, dst, round_mode)

def _get_tmp_buffer_exp(data):
    if isinstance(data, tir.Buffer):
        return T.alloc_ub(data.shape, data.dtype)
    elif isinstance(data, tir.BufferLoad):
        return T.alloc_ub(data.buffer.shape, data.dtype)
    else:
        raise TypeError(f"Unsupported dst type: {type(data)}")

def _get_tmp_buffer_dev(data):
    if isinstance(data, tir.Buffer):
        return T.alloc_shared(data.shape, data.dtype)
    elif isinstance(data, tir.BufferLoad):
        return T.alloc_shared(data.buffer.shape, data.dtype)
    else:
        raise TypeError(f"Unsupported dst type: {type(data)}")

def npuir_reduce(src, dst, dims:Union[list, tuple, int], reduce_mode, size=[], clear: bool = True):
    """Reduce one or more axes of the source vector according to the reduction axes array, starting from an init value.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        dims: The reduction indices array
        reduce_mode: Reduce mode (sum/prod/max/min/max_with_index_left/max_with_index_right/min_with_index_left/min_with_index_right/any/all/xori/ori/abssum/absmax/none)
        clear (bool): Whether to initialize the output buffer before reduction

    Raises:
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If reduce mode is invalid.
        AssertionError: If The reduction indices array is empty.

    Returns:
        tir.Call: A handle to the npuir_reduce operation
    """
    valid_reduce_mode = {"sum", "prod", "max", "min", "max_with_index_left", "max_with_index_right", "min_with_index_left", "min_with_index_right", "any", "all", "xori", "ori", "abssum", "absmax", "none"}
    valid_reduce_clear_false_mode = {"sum", "max", "min", "abssum", "absmax"}
    valid_reduce_abs_mode = {"abssum", "absmax"}
    reduce_abs_mode_map = {
        "abssum": "sum",
        "absmax": "max",
    }
    if reduce_mode in valid_reduce_abs_mode:
        abs_call = AscendUnaryOp("abs", src, src).buildTirCall()
        T.evaluate(abs_call)
        reduce_mode = reduce_abs_mode_map[reduce_mode]

    if isinstance(dims, int):
        dims = [dims]
    src_extent = _get_extent(src) if size == [] else size.copy()
    if size != []:
        for dim in dims:
            size[dim] = 1
    dst_extent = _get_extent(dst) if size == [] else size.copy()
    assert len(src_extent) == len(
        dst_extent), "The input vector and output vector must have same rank."
    assert reduce_mode in valid_reduce_mode, "Reduce mode is invalid."
    assert len(dims) != 0, "The reduction indices array cannot be empty."

    src_region = _to_region(src, "r", src_extent)
    dst_region = _to_region(dst, "w", dst_extent)
    reduce_dims = ','.join(str(dim) for dim in dims)
    if clear == False:
        redeuce_op_for_clear_map = {
            "sum": "add",
            "max": "max",
            "min": "min",
        }
        TILELANG_ASCEND_MODE = os.environ.get('TILELANG_ASCEND_MODE')
        if TILELANG_ASCEND_MODE is None or \
            TILELANG_ASCEND_MODE.lower().strip() in ['expert', 'exp', 'e']:
            tmp = _get_tmp_buffer_exp(dst)
        else:
            tmp = _get_tmp_buffer_dev(dst)
        tmp_extent = _get_extent(tmp)
        assert len(dst_extent) == len(tmp_extent), "The out vector and tmp vector must have same rank."
        assert reduce_mode in valid_reduce_clear_false_mode, "This mode is not supported when clear is false."
        tmp_region = _to_region(tmp, "w", tmp_extent)
        reduce_call = tir.call_intrin("handle", tir.op.Op.get("tl.npuir_reduce"), src_region, tmp_region, reduce_dims, reduce_mode)
        T.evaluate(reduce_call)
        binary_call = AscendBinaryOp(redeuce_op_for_clear_map[reduce_mode], dst, tmp, dst).buildTirCall()
        T.evaluate(binary_call)
    else:
        reduce_call = tir.call_intrin("handle", tir.op.Op.get("tl.npuir_reduce"), src_region, dst_region, reduce_dims, reduce_mode)
        T.evaluate(reduce_call)

def reduce_max(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce max on input buffer, store the result to output buffer

    Parameters
    ----------
    buffer : Buffer
        The input buffer.
    out : Buffer
        The output buffer.
    dim : int
        The dimension to perform reduce on
    clear : bool
        If set to True, the output buffer will first be initialized to -inf.
    Returns
    -------
    handle : PrimExpr
    """
    dim = _legalize_dim(buffer, dim)
    return npuir_reduce(buffer, out, reduce_mode="max", dims=dim, clear=clear)

def reduce_min(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce min on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be initialized to inf. Defaults to True.

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return npuir_reduce(buffer, out, reduce_mode="min", dims=dim, clear=clear)

def reduce_sum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on
        clear (bool, optional): If True, output buffer will be cleared before reduction.
                              If False, results will be accumulated on existing values.
                              Defaults to True.
    Note: When clear=True, reduce_sum will not compute directly on the output buffer. This is because 
          during warp reduction, the same value would be accumulated multiple times (number of threads 
          in the warp). Therefore, the implementation with clear=True follows these steps:
        1. create a temp buffer with same shape and dtype as out
        2. copy out to temp buffer
        3. call reduce_sum with temp buffer and out
        4. Add temp buffer to out

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return npuir_reduce(buffer, out, reduce_mode="sum", dims=dim, clear=clear)

def reduce_abssum(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1):
    """Perform reduce absolute sum on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return npuir_reduce(buffer, out, reduce_mode="abssum", dims=dim, clear=True)

def reduce_absmax(buffer: tir.Buffer, out: tir.Buffer, dim: int = -1, clear: bool = True):
    """Perform reduce absolute max on input buffer, store the result to output buffer.

    Args:
        buffer (tir.Buffer): The input buffer
        out (tir.Buffer): The output buffer
        dim (int): The dimension to perform reduce on

    Returns:
        tir.Call: Handle to the reduction operation
    """
    dim = _legalize_dim(buffer, dim)
    return npuir_reduce(buffer, out, reduce_mode="absmax", dims=dim, clear=clear)

def npuir_cumsum(src: tir.Buffer, dst: Optional[tir.Buffer] = None, dim: int = 0, reverse: bool = False):
    """Perform cumulative sum on input buffer, store the result to output buffer.

    Args:
        src (tir.Buffer): The input buffer
        dst (tir.Buffer, optional): The output buffer. Defaults to None.
        dim (int, optional): The dimension to perform cumulative sum on. Defaults to 0.
        reverse (bool, optional): Whether to perform reverse cumulative sum. Defaults to False.

    Returns:
        tir.Call: Handle to the cumulative sum operation
    """

    shape = src.shape
    if dim >= len(shape) or dim <= -len(shape):
        raise ValueError(f"Dimension {dim} is out of bounds for buffer with shape {shape}")
    if dim < 0:
        dim = len(shape) + dim

    if dst is None:
        dst = src
    
    src_extent = src.shape
    out_extent = dst.shape
    src_tmp = _to_region(src, "r", src_extent)
    dst_tmp = _to_region(dst, "w", out_extent)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.npuir_cumsum"),
        src_tmp,
        dst_tmp,
        str(dim),
        reverse,
    )
    
def npuir_atomic_add(src, dst, size=[]):
    """Perform atomic add operation on the NPU.

    Args:
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination vector
        src: The value to be added atomically
        size (list, optional): Optional size override for dst

    Returns:
        tir.Call: A handle to the npuir_atomic_add operation
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    dst_extent = _get_extent(dst) if size == [] else size.copy()

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_atomic_add"), src, dst)    

def npuir_atomic_addx4(src, dst, size=[]):
    """Perform atomic add operation with quad-width operands on the NPU.

    Args:
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination vector
        src: The value to be added atomically
        size (list, optional): Optional size override for dst

    Returns:
        tir.Call: A handle to the npuir_atomic_addx4 operation
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    dst_extent = _get_extent(dst) if size == [] else size.copy()

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_atomic_add"), src, dst)
    
def npuir_gather(src, dst, indices:Union[list, tuple], size=[]):
    """Retrieve elements from a tensor/memref according to given indices, and store these elements in another tensor/memref. The gather axis is the last dimension.
 
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        indices: The gather indices array
 
    Returns:
        tir.Call: A handle to the npuir_gather operation
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    indices_extent = _get_extent(indices) if size == [] else size.copy()
    dst_extent = _get_extent(dst) if size == [] else size.copy()
 
    src = _to_region(src, "r", src_extent)
    indices = _to_region(indices, "r", indices_extent)
    dst = _to_region(dst, "w", dst_extent)
 
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_gather"), src, dst, indices)
 
def npuir_interleave(*args, channel_nums: int = 2, size = []):
    """Interleaves the values of N tensors along their last dimension. All tensors must have the same shape.
 
    Args:
        srcs (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vectors
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        channel_nums: The number of channels each input participates in during each interleaving
    
    Raises:
        AssertionError: If input vector and output vector have different shapes.
        AssertionError: If the channel nums array is empty.
 
    Returns:
        tir.Call: A handle to the npuir_interleave operation
    
    Notes:
        Due to hardware limitations, only two vectors are currently supported for interleaving.
    """
    *srcs, dst = args
    srcs_arr = []
 
    dst_size = size[:-1] + [size[-1] * 2] if size !=[] else []
    dst_extent = _get_extent(dst) if size == [] else dst_size.copy()
    dst = _to_region(dst, "w", dst_extent)
 
    for i, src in enumerate(srcs):
        src_extent = _get_extent(src) if size == [] else size.copy()
        assert len(src_extent) == len(
        dst_extent), "The input vector and output vector must have same rank."
        src = _to_region(src, "r", src_extent)
        srcs_arr.append(src)
 
    def _tir_call_intrin(channel_nums, dst, *srcs: tir.PrimExpr):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_interleave"), channel_nums, dst, *srcs)
 
    return _tir_call_intrin(channel_nums, dst, *srcs_arr)
 
def npuir_deinterleave(*args, channel_nums: int = 2, index_mode: str = "ALL_CHANNELS", size=[]):
    """Deinterleave one tensor along the last dimension.
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dsts (Union[tir.Buffer, tir.BufferLoad]): Destination vectors
        channel_nums: The number of channels each input participates in during each interleaving
        index_mode: HIVM deinterleave mode
    
    Raises:
        AssertionError: If deinterleave mode is invalid.
        AssertionError: If the last dimension of the input tensor is not the multiple of channel_nums.
        AssertionError: If input vector and output vector have different ranks.
        AssertionError: If the channel nums array is empty.
 
    Returns:
        tir.Call: A handle to the npuir_deinterleave operation
    """
    src, *dsts = args
    dsts_arr = []
 
    valid_index_mode = {"CHANNEL_0", "CHANNEL_1", "ALL_CHANNELS"}
    assert index_mode in valid_index_mode, "Deinterleave mode is invalid."
    src_extent = _get_extent(src) if size == [] else size.copy()
    assert src_extent[-1] % channel_nums == 0, "The last dimension of the input tensor must be multiple of channel_nums."
    src = _to_region(src, "r", src_extent)
 
    dst_size = size[:-1] + [size[-1] * 0.5] if size !=[] else []
    for i, dst in enumerate(dsts):
        dst_extent = _get_extent(dst) if size == [] else dst_size.copy()
        assert len(src_extent) == len(
            dst_extent), "The input vector and output vector must have same rank."
        dst = _to_region(dst, "w", dst_extent)
        dsts_arr.append(dst)
 
    def _tir_call_intrin(channel_nums, index_mode, src, *dsts: tir.PrimExpr):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_deinterleave"), channel_nums, index_mode, src, *dsts)
 
    return _tir_call_intrin(channel_nums, index_mode, src, *dsts_arr)
 
def npuir_transpose(src, dst, permutation = Union[list, tuple], size=[]):
    """Permutes the dimensions of src according to the given permutation. In other words: dim(dst, i) = dim(src, permutation[i]).
 
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
    
    Raises:
        AssertionError: If input vector and output vector have different ranks.
 
    Returns:
        tir.Call: A handle to the npuir_transpose operation
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    dst_size = []
    if size != []:
        for i in range(len(size)):
            dst_size[i] = size[permutation[i]]
    dst_extent = _get_extent(dst) if size == [] else dst_size.copy()
    assert len(src_extent) == len(
        dst_extent), "The input vector and output vector must have same rank."
    
    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
    permutation_str = ','.join(str(pm) for pm in permutation)
 
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_transpose"), src, dst, permutation_str)

def npuir_arange(dst, strides: Union[list, tuple], offset=0, size=[]):
    """Fill a vector with range 0,1,2... based on strides and offset.
    e.g. offset = 1, strides = [1, 2], tensor/memref shape = [2x4xi32],
    the result is [[1, 3, 5, 7,
                    2, 4, 6, 8]].
 
    Args:
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        strides (Union[list, tuple]): Stride list
 
    Returns:
        tir.Call: A handle to the npuir_arange operation
    """
    dst_extent = _get_extent(dst) if size == [] else size.copy()
    dst = _to_region(dst, "w", dst_extent)
    strides_str = ','.join(str(stride) for stride in strides)

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_arange"), dst, strides_str, offset)

def npuir_concat(*args, size=[]):
    """The concat operation constructs a tensor out of a variadic list of input
    tensors, concatenated along a static dimension number.
 
    Args:
        srcs (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vectors
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        dim: Specifies the dimension along which to concatenate
    
    Raises:
        AssertionError: If input vector and output vector have different ranks.
 
    Returns:
        tir.Call: A handle to the npuir_concat operation
    """
    *srcs, dst, dim = args
    srcs_arr = []
    dst_size = size

    for i, src in enumerate(srcs):
        src_extent = _get_extent(src) if size == [] else size.copy()
        if size == []:
            assert len(src_extent) == len(_get_extent(dst)), "The input vector and output vector must have same rank."
        src = _to_region(src, "r", src_extent)
        srcs_arr.append(src)
        if i == dim and size != []:
            dst_size[i] *= len(srcs)

    dst_extent = _get_extent(dst) if size == [] else dst_size.copy()
    dst = _to_region(dst, "w", dst_extent)

    def _tir_call_intrin(dim, dst, *srcs: tir.PrimExpr):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_concat"), dim, dst, *srcs)
 
    return _tir_call_intrin(dim, dst, *srcs_arr)

def npuir_pad(src, dst, pad_value, low: Union[list, tuple], high: Union[list, tuple], size=[]):
    """Pads the input operand.
 
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
        pad_value: The value to pad
        low: The padding lengths along the start of each dimension(Dynamic)
        high: The padding lengths along the end of each dimension(Dynamic)
        static_low: The padding lengths along the start of each dimension(Static)
        static_high: The padding lengths along the end of each dimension(Static)
 
    Returns:
        tir.Call: A handle to the npuir_pad operation
    
    Notes:
        1. Both low/static_low and high/staitc_high can be negative, but the result tensor dimensions are all non-negative
        2. Not support decomposing multi-dim padding for now.
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    dst_size = []
    if size != []:
        for i in range(len(size)):
            dst_size[i] = size[i] + low[i] + high[i]
            assert dst_sze[i] >= 0, "The result tensor dimensions should be non-negative."
    dst_extent = _get_extent(dst) if size == [] else dst_size.copy()
    assert len(src_extent) == len(
        dst_extent), "The input vector and output vector must have same rank."

    assert len(src_extent) == len(low), "Low pad array should have the same length with input vector."
    assert len(src_extent) == len(high), "High pad array should have the same length with input vector."

    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)

    dynamic = []
    num_dynamic_low = 0
    static_low = []
    static_high = []
    pad_dim = -1
    for idx, (l, h) in enumerate(zip(low, high)):
        if isinstance(l, tir.Var) or isinstance(h, tir.Var) or l != 0 or h != 0:
            if pad_dim < 0:
                pad_dim = idx
            else: 
                raise ValueError("Not support decomposing multi-dim padding for now.")

        if isinstance(l, tir.Var):
            dynamic.append(l)
            static_low.append(0)
            num_dynamic_low += 1
        else:
            static_low.append(l)
        
        if isinstance(h, tir.Var):
            dynamic.append(h)
            static_high.append(0)
        else:
            static_high.append(h)

    s_low_str = ','.join(str(s_l) for s_l in static_low)
    s_high_str = ','.join(str(s_h) for s_h in static_high)

    def _tir_call_intrin(src, dst, pad_value, pad_dim, s_low_str, s_high_str, num_dynamic_low, *dynamic: tir.PrimExpr):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_pad"), src, dst, pad_value, pad_dim, s_low_str, s_high_str, num_dynamic_low, *dynamic)
 
    return _tir_call_intrin(src, dst, pad_value, pad_dim, s_low_str, s_high_str, num_dynamic_low, *dynamic)

def npuir_flip(src, dst, size=[]):
    """Flips a tensor along the last dimension.
 
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination vector
 
    Returns:
        tir.Call: A handle to the npuir_flip operation
    """
    src_extent = _get_extent(src) if size == [] else size.copy()
    dst_extent = _get_extent(dst) if size == [] else size.copy()
 
    src = _to_region(src, "r", src_extent)
    dst = _to_region(dst, "w", dst_extent)
 
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_flip"), src, dst)

def npuir_bitcast(src, dtype, size = []):
    """Reinterprets the bits of a shaped value without changing data.
 
    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source vector
        dtype (str): Data type in the result vector
 
    Raises:
        AssertionError: If src vector data type and converted data type have different bit widths.

    Returns:
        tir.Call: A handle to the npuir_bitcast operation
    """
    src_dtype = runtime.DataType(src.dtype)
    src_extent = _get_extent(src) if size == [] else size.copy()
    src = _to_region(src, "rw", src_extent)

    tir_dtype = runtime.DataType(dtype)
    assert (tir_dtype.bits == src_dtype.bits), "The converted data type should have the same bit width with the src data type."

    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_bitcast"), src, dtype)

def npuir_print(obj: Union[tir.PrimExpr, tir.Buffer], msg: str = "", hex: bool = False) -> tir.PrimExpr:
    """
    A generic print function that handles both TIR buffers and primitive expressions.

    - If the input is a TIR buffer, it prints its values, but only on the first thread (tx=0, ty=0, tz=0).
    - If the input is a TIR primitive expression, it prints its value directly.

    Parameters:
        obj (Union[tir.PrimExpr, tir.Buffer]): The object to print. It can be either a tir.Buffer or tir.PrimExpr.
        msg (str): An optional message to include in the print statement.
        hex (bool): If printing in hex format.

    Returns:
        tir.PrimExpr: The TIR expression for the debug print operation.

    Raises:
        AssertionError: If PrimExpr input is not a variable (constant is not supported).
        AssertionError: If input variable is not integer or float.
        ValueError: If the input buffer scope is unsupported.
        ValueError: If the input object type is unsupported.
    """
    if isinstance(obj, tir.Var):
        assert ("int" in obj.dtype or "float" in obj.dtype), "Only support printing integer/float variables."
        if not msg:
            msg = f"expr<{obj}>"
        # Directly print primitive expressions.
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_debug_print_var"), obj, msg, hex)

    elif isinstance(obj, tir.Buffer):
        scope = obj.scope()
        if scope in {"shared", "shared.dyn", "global"}:
            if not msg:
                msg = f"buffer<{obj.name}, {obj.dtype}>"
            obj_extent = _get_extent(obj)
            obj = _to_region(obj, "r", obj_extent)
            return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_debug_print_buffer_value"), obj, msg, hex)
        else:
            # Unsupported buffer scope.
            raise ValueError(
                f"Unexpected buffer scope: {scope}. Supported scopes are share, share.dyn and global.")
    elif isinstance(obj, BufferLoad) or isinstance(obj, BufferRegion):
        if not msg:
            msg = f"subview<{obj.buffer.name}, {obj.buffer.dtype}>"
        obj_extent = _get_extent(obj)
        obj = _to_region(obj, "r", obj_extent)
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_debug_print_buffer_value"), obj, msg, hex)

    else:
        # Unsupported object type.
        raise ValueError(
            f"Unexpected type: {type(obj)}. Supported types are tir.Buffer, tir.BufferLoad, tir.BufferRegion and tir.PrimExpr.")

_local = threading.local()

def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "resource_specialize_frame_stack"):
        _local.resource_specialize_frame_stack = FrameStack()
    return _local.resource_specialize_frame_stack


@register_object("tl.ResourceSpecializeFrame")
class ResourceSpecializeFrame(TIRFrame):

    def __enter__(self):
        super().__enter__()
        _get_current_stack().push(self)
        self.name = self.frames[0].attr_key

    def __exit__(self, ptype, value, trace):
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> Optional["ResourceSpecializeFrame"]:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def set(self, other, event_id: int = 0):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_set_flag"), self.name, other, event_id)

    def wait(self, other, event_id: int = 0):
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_wait_flag"), other, self.name, event_id)

    def block_barrier(self, id):
        """npuir inter block barrier at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block"), 0, self.name, id)

    def subblock_barrier(self, id):
        """npuir inter subblock barrier at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block"), 1, self.name, id)

    def sync_block_set(self, id):
        """npuir intra block sync at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block_set operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block_set"), 2, self.name, id)

    def sync_block_wait(self, id):
        """npuir intra block sync at tile-level.

        Args:
            id: Flag id
        Returns:
            tir.Call: A handle to the npuir_sync_block_wait operation
        """
        return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_sync_block_wait"), self.name, id)

def ResourceSpecialize(resource: str):
    return _ffi_api.ResourceSpecialize(resource)


rs = ResourceSpecialize


def set_flag(other, event_id: int = 0):
    return ResourceSpecializeFrame.Current().set(other, event_id)


def wait_flag(other, event_id: int = 0):
    return ResourceSpecializeFrame.Current().wait(other, event_id)

def pipe_barrier(pipe):
    return tir.call_intrin("handle", tir.op.Op.get("tl.npuir_pipe_barrier"), pipe)

def block_barrier(id):
    return ResourceSpecializeFrame.Current().block_barrier(id)

def subblock_barrier(id):
    return ResourceSpecializeFrame.Current().subblock_barrier(id)

def sync_block_set(id):
    return ResourceSpecializeFrame.Current().sync_block_set(id)

def sync_block_wait(id):
    return ResourceSpecializeFrame.Current().sync_block_wait(id)

@register_object("tl.ScopeFrame")
class ScopeFrame(TIRFrame):
    """
    ScopeFrame is a custom TIRFrame that manages mix kernel
    and handles the entry and exit of the kernel launch scope.
    """


def Scope(name):
    """Tools to construct a scope frame.

    Parameters
    ----------
    name : str
        A string representing cube-core or vector-core

    Returns
    -------
        The result ScopeFrame.
    Examples:
        >>> T.Scope("Cube")
    """

    return _ffi_api.Scope(name)
