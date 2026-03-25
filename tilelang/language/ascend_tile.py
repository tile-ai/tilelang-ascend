import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Call
from typing import Union  # noqa: UP035
from tvm import tir
from tilelang.language.ascend import _dtype
import functools
import warnings

import math


def deprecated(message=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message or f"{func.__name__} is deprecated and will be removed in future versions.", DeprecationWarning, stacklevel=2
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_buffer_info(
    br: Union[Buffer, BufferRegion],
    mask: str,  # noqa: FA100
) -> tuple[Call, PrimExpr]:
    """
    Unified handling of Buffer and BufferRegion to retrieve the underlying access pointer and total data size.

    Args:
        br: The input Buffer or BufferRegion (slice).
        mask: Access mode (e.g., "r" for read, "w" for write).

    Returns:
        ptr: The underlying access pointer with the correct offset applied (tir.Call).
        size: The total number of elements in the data block (tir.PrimExpr).
    """
    if isinstance(br, BufferRegion):
        real_buffer = br.buffer

        indices = [x.min for x in br.region]
        offset = real_buffer.offset_of(indices)[0]
        ptr = real_buffer.access_ptr(mask, offset=offset)

        size = 1
        for r in br.region:
            size *= r.extent

        return ptr, size
    elif isinstance(br, Buffer):
        ptr = br.access_ptr(mask)
        size = math.prod(br.shape)
        return ptr, size
    else:
        raise TypeError(f"Unsupported type: {type(br)}")


def _handle_buffer_region(br: BufferRegion, mask):
    bf = br.buffer
    indices = [x.min for x in br.region]
    offset = bf.offset_of(indices)[0]
    extent = [x.extent for x in br.region]
    size_extent = math.prod(extent)
    return bf.access_ptr(mask, offset=offset, extent=size_extent), extent


def fill(buffer: Union[Buffer, BufferRegion], value: PrimExpr):  # noqa: FA100
    """Fill a buffer or buffer region with a specified value.

    Args:
        buffer: Either a TVM buffer or buffer region to be filled
        value: The value to fill the buffer with

    Returns:
        A TVM intrinsic call that performs the fill operation
    """
    if isinstance(buffer, BufferRegion):
        buffer_ptr, buffer_extent = _handle_buffer_region(buffer, "w")
        size = math.prod(buffer_extent)
    else:
        buffer_ptr = buffer.access_ptr("w")
        size = math.prod(buffer.shape)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_fill"),
        f"Fill<{_dtype(buffer)}>",
        buffer_ptr,
        value,
        size,
    )


def arith_progression(buffer: Buffer, first_value: PrimExpr, diff_value: PrimExpr, count: PrimExpr):
    """Generates an arithmetic progression sequence in a buffer.

    Args:
        buffer: The destination buffer where the sequence will be stored.
        first_value: The starting value of the arithmetic progression.
        diff_value: The difference (step) between consecutive values.
        count: The number of elements to generate.

    Returns:
        A TVM intrinsic call that performs the arithmetic progression operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_arith_progression"),
        f"ArithProgression<{_dtype(buffer)}>",
        buffer.access_ptr("w"),
        first_value,
        diff_value,
        count,
    )


def sort(
    dst: Union[Buffer, BufferRegion],  # noqa: FA100
    src: Buffer,
    indices: Buffer,
    tmp_buffer: Buffer,
    repeat_time: PrimExpr,
):
    """Sorts elements from the source buffer and stores values and indices.

    This function performs a sort operation on the source buffer, outputting both
    the sorted values to the destination buffer and the original indices to the
    indices buffer.

    Args:
        dst: The destination buffer or buffer region where the sorted values will be stored.
        src: The source buffer containing the data to be sorted.
        indices: The buffer where the original indices of the sorted elements will be stored.
        tmp_buffer: A temporary buffer required by the hardware for the sorting computation.
        repeat_time: The number of iterations or elements to process in the sort operation.

    Returns:
        A TVM intrinsic call that performs the sort operation.
    """
    dst_ptr, dst_size = _get_buffer_info(dst, "w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sort"),
        f"Sort<{_dtype(dst)}, true>",
        dst_ptr,
        src.access_ptr("r"),
        indices.access_ptr("r"),
        tmp_buffer.access_ptr("r"),
        repeat_time,
    )


def merge_sort(
    dst: Buffer,
    src: Buffer,
    block_size: PrimExpr,
    block_num: PrimExpr,
    is_copy: PrimExpr,
):
    """Performs a merge sort operation.

    This intrinsic invokes the underlying implementation to perform merge sort
    on the data blocks.

    Args:
        dst: The destination buffer where the sorted result will be stored.
        src: The source buffer containing the data to be merged or sorted.
        block_size: The number of elements in each block to be merged.
        block_num: The total number of blocks to process.
        is_copy: A boolean flag (0 or 1) indicating whether to copy the data
            without sorting.

    Returns:
        A TVM intrinsic call that performs the merge sort operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_merge_sort"),
        f"MergeSort<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        block_size,
        block_num,
        is_copy,
    )


def topk(dst: Buffer, src: Buffer, tmp: Buffer, block_size: PrimExpr):
    """Performs a TopK operation.

    This intrinsic invokes the underlying implementation to select the top K elements
    from the source data.

    Args:
        dst: The destination buffer where the TopK results will be stored.
        src: The source buffer containing the input data.
        tmp: A temporary buffer used for intermediate calculations during the process.
        block_size: The size of the data block to be processed.

    Returns:
        A TVM intrinsic call that performs the TopK operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_topk"),
        f"TopK<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        tmp.access_ptr("r"),
        block_size,
    )


def gather_mask(dst: Buffer, src: Buffer, src1Pattern: Union[str, Buffer]):  # noqa: FA100
    """Performs a gather mask operation.

    This intrinsic invokes the underlying implementation to perform a gather mask
    operation based on the source data and the specified count.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.
        src1Pattern: The data collection mask has two modes: built‑in fixed mode and user‑defined mode.
                     Currently, only fixed mode is supported.
        When the built-in fixed mode is enabled, the data type of src1Pattern is str, including the following 7 modes:
            - "P0101": Extract elements at even indices.
            - "P1010": Extract elements at odd indices.
            - "P0001": Extract the first element from every four elements.
            - "P0010": Extract the second element from every four elements.
            - "P0100": Extract the third element from every four elements.
            - "P1000": Extract the fourth element from every four elements.
            - "P1111": Extract all elements.
        When the custom mode is enabled, the data type of src1Pattern is Buffer.

    Returns:
        A TVM intrinsic call that performs the gather mask operation.
    """

    if isinstance(src1Pattern, Buffer):
        assert src1Pattern.dtype == "uint32", f"src1Pattern dtype must be uint32, got {src1Pattern.dtype}"

        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_gather_mask"),
            f"GatherMask<{_dtype(dst)}>",
            dst.access_ptr("w"),
            src.access_ptr("r"),
            src1Pattern.access_ptr("r"),
        )
    else:
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_gather_mask"),
            f"GatherMask<{_dtype(dst)}>",
            dst.access_ptr("w"),
            src.access_ptr("r"),
            src1Pattern,
        )


def gatherb(
    dst: Buffer,
    src0: Buffer,
    offset: Buffer,
    repeat_time: PrimExpr,
    dst_blk_stride: PrimExpr,
    dst_rep_stride: PrimExpr,
):
    """Performs a GatherB operation.

    This intrinsic invokes the underlying implementation to gather data from the source
    buffer based on the provided offsets and stores it in the destination buffer,
    using the specified strides to control the memory layout.

    Args:
        dst: The destination buffer where the gathered data will be stored.
        src0: The source buffer containing the data table to be gathered from.
        offset: The buffer containing the offsets or indices for gathering.
        repeat_time: The number of repetitions or blocks to process.
        dst_blk_stride: The stride between elements within a block in the destination buffer.
        dst_rep_stride: The stride between repetitions (blocks) in the destination buffer.

    Returns:
        A TVM intrinsic call that performs the GatherB operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gatherb"),
        f"Gatherb<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        offset.access_ptr("r"),
        repeat_time,
        dst_blk_stride,
        dst_rep_stride,
    )


def select(
    dst: Union[Buffer, BufferRegion],  # noqa: FA100
    selMask: Buffer,
    src0: Union[Buffer, BufferRegion],  # noqa: FA100
    src1: Union[Buffer, BufferLoad, PrimExpr],  # noqa: FA100
    selMode: str,
):
    """Performs an element-wise Select operation based on a mask.

    This intrinsic invokes the underlying Ascend implementation to select elements
    from `src0` or `src1` based on the `selMask` condition and the specified `selMode`,
    storing the result in `dst`.

    Args:
        dst: The destination buffer or buffer region where the result will be stored.
        selMask: The mask buffer that determines which source to select from.
        src0: The first source buffer or buffer region.
        src1: The second source operand. It can be a Buffer (Tensor), a specific
            BufferLoad, or a scalar value (PrimExpr/float).
        selMode: The selection mode string. Must be one of:
            - 'VSEL_CMPMASK_SPR': Select based on compare mask.
            - 'VSEL_TENSOR_SCALAR_MODE': Select between a tensor and a scalar.
            - 'VSEL_TENSOR_TENSOR_MODE': Select between two tensors.

    Returns:
        A TVM intrinsic call that performs the Select operation.
    """

    def retrieve_shape(object: Union[Buffer, BufferRegion]) -> list[int]:  # noqa: FA100
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

    def retrieve_ptr(
        object: Union[Buffer, BufferRegion],
        access_type: str = "r",  # noqa: FA100
    ) -> PrimExpr:
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

    dst_ptr = retrieve_ptr(dst, "r")
    src0_ptr = retrieve_ptr(src0, "r")

    sel_mask_ptr = selMask.access_ptr("r")
    src0_extent = src0_shape

    assert selMode in [
        "VSEL_CMPMASK_SPR",
        "VSEL_TENSOR_SCALAR_MODE",
        "VSEL_TENSOR_TENSOR_MODE",
    ]

    size_0 = math.prod(src0_extent)

    if isinstance(src1, BufferLoad):
        assert selMode in ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_TENSOR_MODE"], "selMode must be VSEL_CMPMASK_SPR or VSEL_TENSOR_TENSOR_MODE"

        src1_type = 0
        buffer_1 = src1.buffer
        indices_1 = src1.indices
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_select"),
            dst_ptr,
            sel_mask_ptr,
            src0_ptr,
            src1_type,
            buffer_1.access_ptr("r"),
            indices_1[0],
            selMode,
            size_0,
        )
    elif isinstance(src1, (PrimExpr, float)):
        assert selMode == "VSEL_TENSOR_SCALAR_MODE", "selMode must be VSEL_TENSOR_SCALAR_MODE"

        src1_type = 1
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_select"),
            dst_ptr,
            sel_mask_ptr,
            src0_ptr,
            src1_type,
            src1,
            selMode,
            size_0,
            _dtype(src0),
            _dtype(selMask),
        )
    else:
        assert selMode in ["VSEL_CMPMASK_SPR", "VSEL_TENSOR_TENSOR_MODE"], "selMode must be VSEL_CMPMASK_SPR or VSEL_TENSOR_TENSOR_MODE"

        src1_type = 2
        src1_ptr = src1.access_ptr("r")
        return tir.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_select"),
            dst_ptr,
            sel_mask_ptr,
            src0_ptr,
            src1_type,
            src1_ptr,
            selMode,
            size_0,
        )


def init_sort_buf(buffer: Buffer, num: PrimExpr, rsv: PrimExpr):
    """Initializes a buffer for sorting operations.

    This intrinsic invokes the underlying implementation to initialize the specified
    buffer, which is typically required as an auxiliary or index buffer for
    hardware sorting instructions.

    Args:
        buffer: The buffer to be initialized.
        num: The number of elements to initialize in the buffer.
        rsv: A reserved parameter or specific initialization value required by
            the hardware API.

    Returns:
        A TVM intrinsic call that performs the buffer initialization.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_init_sort_buf"),
        f"InitSortBuf<{_dtype(buffer)}>",
        buffer.access_ptr("w"),
        rsv,
        num,
    )


@deprecated()
def brcb(dst: Buffer, src: Buffer, repeat_times: PrimExpr, dst_blk_stride: PrimExpr, dst_repeat_stride: PrimExpr):
    """Broadcast repeat copy block intrinsic.

    .. warning::
        **NOT IMPLEMENTED**: The backend code generation (codegen) for this interface
        has **NOT** been implemented yet. **DO NOT USE** this function, as it will
        result in compilation or runtime errors.

    Args:
        dst (Buffer): The destination buffer.
        src (Buffer): The source buffer.
        repeat_times (PrimExpr): The number of times to repeat the operation.
        dst_blk_stride (PrimExpr): The stride between blocks in the destination.
        dst_repeat_stride (PrimExpr): The stride between repetitions in the destination.

    Returns:
        tvm.tir.Call: A TIR external call node representing the operation.
    """

    src_size = math.prod(src.shape)
    assert src_size >= (repeat_times * 8), "src size must be not less then repeat_times * 8"

    src_ptr = src.access_ptr("r")
    dst_ptr = dst.access_ptr("w")

    return T.call_extern("handle", f"tl::ascend::brcb<{_dtype(src)}>", dst_ptr, src_ptr, repeat_times, dst_blk_stride, dst_repeat_stride)


def binary_op(
    dst: Union[Buffer, BufferRegion],  # noqa: FA100
    src0: Union[Buffer, BufferRegion],  # noqa: FA100
    src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr, float],  # noqa: FA100
    op: str,
):

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


def add(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]):  # noqa: FA100
    """Performs element-wise addition: dst = src0 + src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "add")


def sub(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad]):  # noqa: FA100
    """Performs element-wise subtraction: dst = src0 - src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "sub")


def mul(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]):  # noqa: FA100
    """Performs element-wise multiplication: dst = src0 * src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "mul")


def div(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad]):  # noqa: FA100
    """Performs element-wise division: dst = src0 / src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "div")


def max(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]):  # noqa: FA100
    """Performs element-wise maximum: dst = max(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "max")


def min(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]):  # noqa: FA100
    """Performs element-wise minimum: dst = min(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "min")


def bitwise_and(
    dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]
):  # noqa: FA100
    """Performs element-wise bitwise AND: dst = src0 & src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_and")


def bitwise_or(
    dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], src1: Union[Buffer, BufferRegion, BufferLoad, PrimExpr]
):  # noqa: FA100
    """Performs element-wise bitwise OR: dst = src0 | src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_or")


def unary_op(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion], op: str):  # noqa: FA100

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

    # return T.call_extern("handle", f"AscendC::{op}", dst_ptr, src0_ptr, size_0)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(f"tl.ascend_{op}"),
        dst_ptr,
        src0_ptr,
        size_0,
    )


def exp(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise exponential: dst = exp(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "exp")


def sigmoid(dst: Union[Buffer, BufferRegion], src: Union[Buffer, BufferRegion], tmp: Buffer):  # noqa: FA100
    if isinstance(dst, BufferRegion):
        print("test1")
        dst_ptr, buffer_extent = _handle_buffer_region(dst, "w")
        size = math.prod(buffer_extent)
        print("test2")
    else:
        dst_ptr = dst.access_ptr("w")
        size = math.prod(dst.shape)

    if isinstance(src, BufferRegion):
        src_ptr, _ = _handle_buffer_region(src, "r")
    else:
        src_ptr = src.access_ptr("r")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sigmoid"),
        dst_ptr,
        src_ptr,
        tmp.access_ptr("w"),
        size,
    )


def ln(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise natural logarithm: dst = ln(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "ln")


def abs(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise absolute value: dst = abs(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "abs")


def reciprocal(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise reciprocal: dst = 1 / src0.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "reciprocal")


def sqrt(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise square root: dst = sqrt(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "sqrt")


def rsqrt(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise reciprocal square root: dst = 1 / sqrt(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "rsqrt")


def relu(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise Rectified Linear Unit (ReLU): dst = max(0, src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "relu")


def bitwise_not(dst: Union[Buffer, BufferRegion], src0: Union[Buffer, BufferRegion]):  # noqa: FA100
    """Performs element-wise bitwise NOT (inversion): dst = ~src0.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "bitwise_not")


def scalar_op(
    dst: Union[Buffer, BufferRegion],
    src0: Union[Buffer, BufferRegion],
    scalar_value: PrimExpr,
    op_tl: str,  # noqa: FA100
):
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
        size_2 = math.prod(dst_extent)
    else:
        dst_ptr = dst.access_ptr("w")
        size_2 = math.prod(dst.shape)

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
        size_0 = math.prod(src0_extent)
    else:
        src0_ptr = src0.access_ptr("r")
        size_0 = math.prod(src0.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get(f"tl.ascend_{op_tl}"),
        dst_ptr,
        src0_ptr,
        scalar_value,
        size_0,
    )


def leaky_relu(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    """Performs element-wise Leaky ReLU activation.

    Formula: dst = src0 if src0 >= 0 else src0 * scalar_value

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalar_value: The negative slope coefficient.
    """
    return scalar_op(dst, src0, scalar_value, "leaky_relu")


def axpy(dst: Buffer, src0: Buffer, scalar_value: PrimExpr):
    """Performs element-wise AXPY operation: dst = scalar_value * src0 + dst.

    Note: This operation updates the destination buffer in-place by adding
    the scaled source buffer.

    Args:
        dst: The destination buffer (acts as both operand Y and output).
        src0: The source buffer X.
        scalar_value: The scalar alpha.
    """
    return scalar_op(dst, src0, scalar_value, "axpy")


def bitwise_lshift(dst: Buffer, src0: Buffer, scalarValue: PrimExpr):
    """Performs element-wise bitwise left shift: dst = src0 << scalarValue.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalarValue: The number of bits to shift (scalar).
    """
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_bitwise_lshift"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        scalarValue,
        size_0,
    )


def bitwise_rshift(dst: Buffer, src0: Buffer, scalarValue: PrimExpr):
    """Performs element-wise bitwise right shift: dst = src0 >> scalarValue.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalarValue: The number of bits to shift (scalar).
    """
    size_0 = math.prod(src0.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_bitwise_rshift"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        scalarValue,
        size_0,
    )


@deprecated()
def bilinear_interpolation(
    dst: Buffer,
    src0: Buffer,
    src0_offset: Buffer,
    src1: Buffer,
    mask: PrimExpr,
    h_repeat: PrimExpr,
    repeat_mode: bool,
    dst_blk_stride: PrimExpr,
    v_r_offset: PrimExpr,
    v_repeat: PrimExpr,
    shared_tmp_buffer: Buffer,
):
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_bilinear_interpolation"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src0_offset.access_ptr("r"),
        src1.access_ptr("r"),
        mask,
        h_repeat,
        repeat_mode,
        dst_blk_stride,
        v_r_offset,
        v_repeat,
        shared_tmp_buffer.access_ptr("r"),
    )


def _wholereduce(
    reduce_type: str,
    dst: Buffer,
    src: Buffer,
    mask: PrimExpr,
    repeattimes: PrimExpr,
    dstrepstride: PrimExpr,
    srcblkstride: PrimExpr,
    srcrepstride: PrimExpr,
    reduce_order: str = None,
):
    args = [
        dst.access_ptr("w"),
        src.access_ptr("r"),
        mask,
        repeattimes,
        dstrepstride,
        srcblkstride,
        srcrepstride,
    ]

    if reduce_order is not None:
        args.append(reduce_order)

    return tir.call_intrin("handle", tir.op.Op.get(f"tl.ascend_wholereduce{reduce_type}"), *args)


@deprecated()
def wholereducemax(
    dst: Buffer,
    src: Buffer,
    mask: PrimExpr,
    repeattimes: PrimExpr,
    dstrepstride: PrimExpr,
    srcblkstride: PrimExpr,
    srcrepstride: PrimExpr,
    ReduceOrder: str = "ORDER_VALUE_INDEX",
):
    """
    Warning:Currently, this implementation does not support pto target
    """
    return _wholereduce("max", dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride, ReduceOrder)


@deprecated()
def wholereducemin(
    dst: Buffer,
    src: Buffer,
    mask: PrimExpr,
    repeattimes: PrimExpr,
    dstrepstride: PrimExpr,
    srcblkstride: PrimExpr,
    srcrepstride: PrimExpr,
    ReduceOrder: str = "ORDER_VALUE_INDEX",
):
    """
    Warning:Currently, this implementation does not support pto target
    """
    return _wholereduce("min", dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride, ReduceOrder)


@deprecated()
def wholereducesum(
    dst: Buffer, src: Buffer, mask: PrimExpr, repeattimes: PrimExpr, dstrepstride: PrimExpr, srcblkstride: PrimExpr, srcrepstride: PrimExpr
):
    """
    Warning:Currently, this implementation does not support pto target
    """
    return _wholereduce("sum", dst, src, mask, repeattimes, dstrepstride, srcblkstride, srcrepstride)


def sort32(dst: Buffer, src0: Buffer, src1: Buffer):
    """Performs a specific 32-element block sorting operation.

    This intrinsic invokes the underlying implementation to sort data in groups
    of 32 elements.

    Args:
        dst: The destination buffer where the sorted results will be stored.
        src0: The first source buffer containing the data to be sorted.
        src1: The second source buffer (often used for indices or auxiliary data).
    """
    repeatTimes = math.prod(src0.shape) // 32
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sort32"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1.access_ptr("r"),
        repeatTimes,
    )


def createvecindex(dst: Buffer, firstValue: PrimExpr):
    """Generates a vector index sequence.

    This intrinsic fills the destination buffer with a sequence of increasing
    indices starting from `firstValue` (e.g., firstValue, firstValue+1, ...).

    Args:
        dst: The destination buffer to be filled with indices.
        firstValue: The starting value of the index sequence.
    """
    calCount = math.prod(dst.shape)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_createvecindex"),
        f"CreateVecIndex<{_dtype(dst)}>",
        dst.access_ptr("w"),
        firstValue,
        calCount,
    )


def transpose(dst: Buffer, src: Buffer):
    """Performs a matrix transposition operation.

    This intrinsic invokes the underlying implementation to transpose the source
    buffer into the destination buffer.

    Args:
        dst: The destination buffer.
        src: The source buffer to be transposed.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_transpose"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
    )


def gather(dst: Buffer, src: Buffer, src_offset: Buffer, src_base_addr: PrimExpr):
    """Performs a gather operation.

    This intrinsic gathers elements from the source buffer based on the provided
    offsets and a base address, storing the result in the destination buffer.

    Args:
        dst: The destination buffer where the gathered data will be stored.
        src: The source buffer containing the data table.
        src_offset: The buffer containing offsets/indices for gathering.
        src_base_addr: The base address offset to be added to the gather indices.
    """
    count = math.prod(src.shape)
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gather"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        src_offset.access_ptr("r"),
        src_base_addr,
        count,
    )


@deprecated()
def block_reduce_max(
    dst: Buffer,
    src: Buffer,
    repeat: PrimExpr,
    mask: PrimExpr,
    dstPepStride: PrimExpr,
    srcBlkStride: PrimExpr,
    srcRepStride: PrimExpr,
):
    """Performs a block-level reduction max operation.

    This intrinsic invokes the underlying implementation to find the maximum
    value within data blocks from the source buffer.

    Warning:Currently, this implementation does not support pto target

    Args:
        dst: The destination buffer where the results will be stored.
        src: The source buffer containing the data to be reduced.
        repeat: The number of iterations (repeats) to perform.
        mask: The mask parameter to control valid elements in the operation.
        dstPepStride: The stride between destination elements for consecutive repeats.
        srcBlkStride: The stride between source blocks within a single iteration.
        srcRepStride: The stride between source blocks for consecutive repeats.

    Returns:
        A TVM intrinsic call that performs the block reduce max operation.
    """
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_block_reduce_max"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        repeat,
        mask,
        dstPepStride,
        srcBlkStride,
        srcRepStride,
    )


@deprecated()
def block_reduce_min(
    dst: Buffer,
    src: Buffer,
    repeat: PrimExpr,
    mask: PrimExpr,
    dstPepStride: PrimExpr,
    srcBlkStride: PrimExpr,
    srcRepStride: PrimExpr,
):
    """Performs a block-level reduction min operation.

    This intrinsic invokes the underlying implementation to find the minimum
    value within data blocks from the source buffer.

    Warning:Currently,this implementation does not support pto target

    Args:
        dst: The destination buffer where the results will be stored.
        src: The source buffer containing the data to be reduced.
        repeat: The number of iterations (repeats) to perform.
        mask: The mask parameter to control valid elements in the operation.
        dstPepStride: The stride between destination elements for consecutive repeats.
        srcBlkStride: The stride between source blocks within a single iteration.
        srcRepStride: The stride between source blocks for consecutive repeats.

    Returns:
        A TVM intrinsic call that performs the block reduce min operation.
    """
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_block_reduce_min"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        repeat,
        mask,
        dstPepStride,
        srcBlkStride,
        srcRepStride,
    )


@deprecated()
def block_reduce_sum(
    dst: Buffer,
    src: Buffer,
    repeat: PrimExpr,
    mask: PrimExpr,
    dstPepStride: PrimExpr,
    srcBlkStride: PrimExpr,
    srcRepStride: PrimExpr,
):
    """Performs a block-level reduction sum operation.

    This intrinsic invokes the underlying implementation to calculate the sum
    of elements within data blocks from the source buffer.

    Warning:Currently,this implementation does not support pto target

    Args:
        dst: The destination buffer where the results will be stored.
        src: The source buffer containing the data to be reduced.
        repeat: The number of iterations (repeats) to perform.
        mask: The mask parameter to control valid elements in the operation.
        dstPepStride: The stride between destination elements for consecutive repeats.
        srcBlkStride: The stride between source blocks within a single iteration.
        srcRepStride: The stride between source blocks for consecutive repeats.

    Returns:
        A TVM intrinsic call that performs the block reduce sum operation.
    """
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_block_reduce_sum"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        repeat,
        mask,
        dstPepStride,
        srcBlkStride,
        srcRepStride,
    )


def compare(
    dst: Buffer,
    src0: Buffer,
    src1: Union[Buffer, BufferLoad, PrimExpr],
    mode: str,  # noqa: FA100
):
    """Generic dispatch function for element-wise comparison operations.

    This function compares elements between `src0` and `src1` according to the
    specified `mode` and stores the result in `dst`. It supports both
    tensor-tensor and tensor-scalar comparisons.

    Args:
        dst: The destination buffer where the comparison results will be stored.
        src0: The first source buffer.
        src1: The second source operand. It can be a Buffer (for element-wise
            tensor comparison) or a BufferLoad/PrimExpr/float (for tensor-scalar
            comparison).
        mode: The comparison mode string. Supported values:
            - "EQ": Equal to (==)
            - "NE": Not equal to (!=)
            - "GT": Greater than (>)
            - "GE": Greater than or equal to (>=)
            - "LT": Less than (<)
            - "LE": Less than or equal to (<=)

    Returns:
        A TVM intrinsic call that performs the comparison operation.
    """
    assert mode in ["EQ", "NE", "GT", "GE", "LT", "LE"]

    dst_ptr = dst.access_ptr("w")

    src0_ptr = src0.access_ptr("r")
    src0_extent = src0.shape

    size_1 = math.prod(src0_extent)

    dst_size = size_1

    if isinstance(src1, BufferLoad):
        buffer_1 = src1.buffer
        indices_1 = src1.indices
        return T.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_compare_scalar"),
            dst_ptr,
            src0_ptr,
            buffer_1.access_ptr("r"),
            indices_1[0],
            mode,
            dst_size,
        )
    elif isinstance(src1, (PrimExpr, float)):
        return T.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_compare_scalar"),
            dst_ptr,
            src0_ptr,
            src1,
            mode,
            dst_size,
        )
    else:
        return T.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_compare"),
            dst_ptr,
            src0_ptr,
            src1.access_ptr("r"),
            mode,
            dst_size,
        )


def cast(dst: Buffer, src: Buffer, mode: str, count: PrimExpr):
    """Performs element-wise data type conversion with a specified rounding mode.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.
        mode: The rounding mode string. Supported values include:
            - "CAST_NONE": No specific rounding.
            - "CAST_RINT": Round to the nearest integer.
            - "CAST_FLOOR": Round down (towards negative infinity).
            - "CAST_CEIL": Round up (towards positive infinity).
            - "CAST_ROUND": Round to the nearest integer, ties away from zero.
            - "CAST_TRUNC": Truncate (round towards zero).
            - "CAST_ODD": Round to the nearest odd integer.
        count: The number of elements to process.

    Returns:
        A TVM intrinsic call that performs the cast operation.
    """
    assert mode in [
        "CAST_NONE",
        "CAST_RINT",
        "CAST_FLOOR",
        "CAST_CEIL",
        "CAST_ROUND",
        "CAST_TRUNC",
        "CAST_ODD",
    ]

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_cast"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        mode,
        count,
    )


def sin(dst: Buffer, src: Buffer, tmp: Buffer):
    """Performs element-wise sine calculation: dst = sin(src).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.
        tmp: A temporary buffer used for intermediate calculations.

    Returns:
        A TVM intrinsic call that performs the sine operation.
    """
    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sin"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        tmp.access_ptr("r"),
        size_0,
    )


def cos(dst: Buffer, src: Buffer, tmp: Buffer):
    """Performs element-wise cosine calculation: dst = cos(src).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.
        tmp: A temporary buffer used for intermediate calculations.

    Returns:
        A TVM intrinsic call that performs the cosine operation.
    """
    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_cos"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        tmp.access_ptr("r"),
        size_0,
    )


# def clampMax(dst: Buffer, src: Buffer, tmp: Buffer, scalar_value: PrimExpr, count: PrimExpr):
#
#     return min(dst, src, scalar_value)
#
# def clampMin(dst: Buffer, src: Buffer, tmp: Buffer, scalar_value: PrimExpr, count: PrimExpr):
#
#     return max(dst, src, scalar_value)
#
# def round(dst: Buffer, src: Buffer, tmp: Buffer, count: PrimExpr):
#
#     return cast(dst, src, "CAST_ROUND", count)
#
def pow(dst: Buffer, src0: Buffer, src1: Buffer, tmp: Buffer):
    """Performs element-wise power calculation: dst = src0 ^ src1.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The base buffer.
        src1: The exponent buffer.
        tmp: A temporary buffer used for intermediate calculations.

    Returns:
        A TVM intrinsic call that performs the power operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_pow"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1.access_ptr("r"),
        tmp.access_ptr("w"),
    )


def bitwise_xor(dst: Buffer, src0: Buffer, src1: Buffer, tmp: Buffer):
    """Performs element-wise bitwise XOR operation: dst = src0 ^ src1.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The first source operand buffer.
        src1: The second source operand buffer.
        tmp: A temporary buffer used for intermediate calculations.

    Returns:
        A TVM intrinsic call that performs the bitwise XOR operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_bitwise_xor"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1.access_ptr("r"),
        tmp.access_ptr("w"),
    )


def clamp_max(out: Buffer, buffer: Buffer, tmp: Buffer, scalar_value: PrimExpr, count: PrimExpr):
    """_summary_
    Clip tensor elements to no more than scalar_value, replace elements larger than scalar_value with scalar_value,
    keep original values for elements less than or equal to scalar_value
    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        tmp: The second source operand buffer.
        scalar_value: The max scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp_max operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp_max"),
        f"ClampMax<{_dtype(buffer)}>",
        out.access_ptr("w"),
        buffer.access_ptr("r"),
        tmp.access_ptr("r"),
        scalar_value,
        count,
    )


def clamp_min(out: Buffer, buffer: Buffer, tmp: Buffer, scalar_value: PrimExpr, count: PrimExpr):
    """
    Clip tensor elements to no less than v, replace elements smaller than scalar_value with scalar_value,
    keep original values for elements greater than or equal to scalar_value
    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        tmp: The second source operand buffer.
        scalar_value: The min scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp_min operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp_min"),
        f"ClampMin<{_dtype(buffer)}>",
        out.access_ptr("w"),
        buffer.access_ptr("r"),
        tmp.access_ptr("r"),
        scalar_value,
        count,
    )


def clamp(out: Buffer, buffer: Buffer, tmp: Buffer, min_scalar: PrimExpr, max_scalar: PrimExpr, count: PrimExpr):
    """
    Clip tensor elements to [min_scalar, max_scalar] range, replace out-of-bounds values with boundary values
    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        tmp: The second source operand buffer.
        min_scalar: The min scalar value
        max_scalar: The max scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp operation.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp"),
        f"Clamp<{_dtype(buffer)}>",
        out.access_ptr("w"),
        buffer.access_ptr("r"),
        tmp.access_ptr("r"),
        min_scalar,
        max_scalar,
        count,
    )


def round(out: Buffer, buffer: Buffer, tmp: Buffer, count: PrimExpr):

    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.ascend_round"), out.access_ptr("w"), buffer.access_ptr("r"), tmp.access_ptr("r"), count
    )


def broadcast(
    dst: Union[Buffer, BufferRegion],  # noqa: FA100
    src: Union[Buffer, BufferRegion],  # noqa: FA100
    tmp: Union[Buffer, BufferRegion],
):  # noqa: FA100
    """Generates a TIR intrinsic call for the AscendC `Broadcast` operation.

    This function performs a broadcast copy from the source buffer (`src`) to the
    destination buffer (`dst`). It automatically infers the broadcasting axis
    based on the shapes of the input buffers.

    Args:
        dst (tvm.tir.Buffer): The destination buffer. Must be allocated in the
            Unified Buffer (UB). Its shape determines the output size.
        src (tvm.tir.Buffer): The source buffer. Must be allocated in the
            Unified Buffer (UB). Its shape must be compatible with `dst` for broadcasting.
        tmp (tvm.tir.Buffer): The temporary buffer.

    Returns:
        tvm.tir.Call: A TIR intrinsic call node that maps to the C++ `AscendC::Broadcast` API.

    Raises:
        AssertionError: If the input shapes violate the dimension constraints.

    Constraints:
        1. **Rank Consistency**: The number of dimensions (rank) of `src` and `dst` must be identical.
        2. **Supported Dimensions**: Only 1D and 2D tensors are supported. The rank must be 1 or 2.
        3. **Broadcasting Logic**:
            - **Axis 0 (Row Broadcast)**: Inferred if `src.shape[0] == 1` and `dst.shape[0] > 1`.
              The source row is replicated `dst.shape[0]` times.
            - **Axis 1 (Column Broadcast)**: Inferred if `src.shape[1] == 1` and `dst.shape[1] > 1`.
              The source column is replicated `dst.shape[1]` times.
            - **No Broadcast (Copy)**: If shapes are identical, the axis defaults to 0.
    """

    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_extent = dst.shape

    if isinstance(src, BufferRegion):
        src_ptr, src_extent = _handle_buffer_region(src, "r")
    else:
        src_ptr = src.access_ptr("r")
        src_extent = src.shape

    if isinstance(tmp, BufferRegion):
        tmp_ptr, _ = _handle_buffer_region(tmp, "r")
    else:
        tmp_ptr = tmp.access_ptr("r")

    dtype = _dtype(src)

    if len(dst_extent) == 3:
        dst_extent = [dst_extent[1], dst_extent[2]]
    if len(src_extent) == 3:
        src_extent = [src_extent[1], src_extent[2]]
    dim = len(dst_extent)
    assert dim in [1, 2], "Ascend Broadcast only supports dim=1 or dim=2."
    assert len(src_extent) == dim, "Source and Dest dimension must match."

    axis = 0
    if dim == 2:
        if src_extent[0] == 1 and dst_extent[0] != 1:
            axis = 0
        elif src_extent[1] == 1 and dst_extent[1] != 1:
            axis = 1
        else:
            axis = 0
    else:  # dim == 1
        axis = 0

    op_name = "tl.ascend_broadcast"
    template_args = f"{dtype}, {dim}, {axis}, false"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_name),
        f"Broadcast<{template_args}>",
        dst_ptr,
        src_ptr,
        tmp_ptr,
        dim,
        *dst_extent,
        *src_extent,
    )


def sub_experiment(dst: Buffer, src0: Buffer, src1: Buffer, count: PrimExpr):
    """Performs element-wise subtraction(with count function): dst = src0 - src1.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The base buffer.
        src1: The exponent buffer.
        count: The number of elements to process.
    """

    size_0 = math.prod(src0.shape)
    size_1 = math.prod(src1.shape)
    size_2 = math.prod(dst.shape)
    assert size_0 == size_1 == size_2, "size must be same"

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sub_experiment"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1.access_ptr("r"),
        count,
    )


def abs_experiment(dst: Buffer, src: Buffer, count: PrimExpr):
    """Performs element-wise absolute value(with count function): dst = abs(src0).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        count: The number of elements to process.
    """

    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)
    assert size_0 == size_2, "size must be same"

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_abs_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        count,
    )


def mins_experiment(dst: Buffer, src: Buffer, scalarValue: PrimExpr, count: PrimExpr):
    """Performs comparison of each element in the tensor with a scalar.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        scalarValue: The scalar for comparison.
        count: The number of elements to process.
    """
    size_0 = math.prod(src.shape)
    size_2 = math.prod(dst.shape)
    assert size_0 == size_2, "size must be same"

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_mins_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        scalarValue,
        count,
    )


def reduce_sum_experiment(dst: Buffer, src: Buffer, sharedtmp: Buffer, count: PrimExpr):
    """Performs summation of all input data.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        sharedtmp: Used to store intermediate results during instruction execution.
        count: The number of elements to process.
    """

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reducesum_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        sharedtmp.access_ptr("r"),
        count,
    )


def reduce_sum_mask_experiment(dst: Buffer, src: Buffer, sharedtmp: Buffer, mask: PrimExpr, repeatTime: PrimExpr, srcRepStride: PrimExpr):
    """Performs summation of all input data(High-dimensional tensor slicing and computation).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        sharedtmp: Used to store intermediate results during instruction execution.
        mask: Used to control the elements participating in the computation within each iteration.
        repeatTime: Number of iterations.
        srcRepStride: The address step size of the source operand between adjacent iterations.
    """

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reducesum_mask_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        sharedtmp.access_ptr("r"),
        mask,
        repeatTime,
        srcRepStride,
    )


def gathermask_experiment(
    dst: Buffer, src0: Buffer, src1Pattern: Buffer, reduceMode: bool, mask: PrimExpr, GatherMaskParams: list[int], rsvdCnt: PrimExpr
):
    """Performs a gather mask operation(User-defined mode).

    This intrinsic invokes the underlying implementation to perform a gather mask
    operation based on the source data and the specified count.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The source buffer containing the input data.
        src1Pattern: Selects elements from the source operand according to the binary values
                     corresponding to the user-defined input Tensor values and writes them to the destination operand.
        reduceMode: Used to select the mask parameter mode:
                    - "False": Normal mode.
                    - "True": Counter mode.
        mask: Used to control the elements participating in the computation within each iteration.
        GatherMaskParams: A data structure that controls the address step size of operands:
                    - "src0BlockStride": Used to set the address step size between different DataBlocks of src0 in the same iteration, in units of DataBlock.
                    - "repeatTime": Number of iterations.
                    - "src0RepeatStride": Used to set the address step size of src0 between adjacent iterations, in units of DataBlock.
                    - "src1RepeatStride": Used to set the address step size of src1 between adjacent iterations, in units of DataBlock.
        rsvdCnt: The count of elements retained after filtering by this instruction, corresponding to the number of valid elements in dstLocal.
    """

    src0BlockStride = GatherMaskParams[0]
    repeatTime = GatherMaskParams[1]
    src0RepeatStride = GatherMaskParams[2]
    src1RepeatStride = GatherMaskParams[3]
    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gather_mask_experiment"),
        f"GatherMask_experiment<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1Pattern.access_ptr("r"),
        reduceMode,
        mask,
        src0BlockStride,
        repeatTime,
        src0RepeatStride,
        src1RepeatStride,
        rsvdCnt,
    )


def fill_experiment(
    dst: Buffer, value: PrimExpr, mask: list[int], repeatTimes: PrimExpr, dstBlockStride: PrimExpr, dstRepeatStride: PrimExpr
):
    """Fill a buffer or buffer region with a specified value(High-dimensional tensor slicing and computation).

    Args:
        dst: Either a TVM buffer or buffer region to be filled.
        value: The value to fill the buffer with.
        mask: Used to control the elements participating in the computation within each iteration.
        repeatTime: Number of iterations.
        dstBlockStride: Address stride between different DataBlocks for the vector destination operand within a single iteration.
        dstRepeatStride: Address stride of the same DataBlock for the vector destination operand between adjacent iterations.
    """

    mask0 = mask[0]
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_fill_experiment"),
        f"Fill_experiment<{_dtype(dst)}>",
        dst.access_ptr("w"),
        value,
        mask0,
        repeatTimes,
        dstBlockStride,
        dstRepeatStride,
    )


def sum_experiment(dst: Buffer, src: Buffer, sumParams: list[int]):
    """Sum elements along the last dimension (high-level API).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        outter: Get the sum of elements over the last dimension.
        inner: Represents the number of padded elements along the inner axis of the input data,
               inner * sizeof(T) must be an integer multiple of 32 bytes.
        n: Represents the actual number of elements along the inner axis of the input data.
    """

    outter = sumParams[0]
    inner = sumParams[1]
    n = sumParams[2]
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sum_experiment"),
        f"Sum_experiment<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        outter,
        inner,
        n,
    )


def datacachecleanandinvalid_experiment(dst: Buffer, CacheLine: str, DcciDst: str):

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_datacachecleanandinvalid_experiment"),
        f"AscendC::DataCacheCleanAndInvalid<{_dtype(dst)}, AscendC::CacheLine::{CacheLine}, AscendC::DcciDst::{DcciDst}>",
        dst.access_ptr("w"),
    )
