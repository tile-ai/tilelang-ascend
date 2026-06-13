from __future__ import annotations
import tilelang.language as T
from tvm.tir import PrimExpr, Buffer, BufferRegion, BufferLoad, Call
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
    br: Buffer | BufferRegion,
    mask: str,
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


def _handle_buffer_region_2d(br: BufferRegion, mask):
    """Like _handle_buffer_region but flattens ND extents to 2D [rows, cols].

    For a 3D buffer s_ub[T, 32, 512] sliced as s_ub[tile, :, :]:
        extent_nd = [1, 32, 512]  →  extent_2d = [32, 512]

    Leading dimensions are folded into rows; the innermost dimension is kept as cols.
    """
    bf = br.buffer
    indices = [x.min for x in br.region]
    offset = bf.offset_of(indices)[0]
    extent_nd = [x.extent for x in br.region]
    size_extent = math.prod(extent_nd)
    if len(extent_nd) >= 2:
        extent_2d = [math.prod(extent_nd[:-1]), extent_nd[-1]]
    else:
        extent_2d = [1, extent_nd[0]]
    return bf.access_ptr(mask, offset=offset, extent=size_extent), extent_2d


_ATOMIC_ADD_V1_ERR = "T.tile.atomic_add V1 only supports local tensor -> GM atomic add."


def _tile_region(buffer: BufferLoad, access_type: str, *args: PrimExpr):
    access_mask = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.region"),
        buffer,
        access_mask,
        *args,
    )


def _resolve_let_value(data):
    if isinstance(data, tir.Var) and T.has_let_value(data):
        return T.get_let_value(data)
    return data


def _atomic_add_buffer(data, arg_name: str) -> Buffer:
    data = _resolve_let_value(data)
    if isinstance(data, Buffer):
        return data
    if isinstance(data, BufferRegion):
        return data.buffer
    if isinstance(data, BufferLoad):
        return data.buffer
    raise TypeError(f"{_ATOMIC_ADD_V1_ERR} {arg_name} must be a Buffer, BufferRegion, or BufferLoad, got {type(data).__name__}.")


def _atomic_add_scope(data, arg_name: str) -> str:
    buffer = _atomic_add_buffer(data, arg_name)
    scope_attr = getattr(buffer, "scope", None)
    scope = scope_attr() if callable(scope_attr) else scope_attr
    if not isinstance(scope, str):
        raise TypeError(f"{_ATOMIC_ADD_V1_ERR} Cannot determine {arg_name} buffer scope.")
    return scope


def _atomic_add_extent(data):
    data = _resolve_let_value(data)
    if isinstance(data, Buffer):
        return list(data.shape)
    if isinstance(data, BufferRegion):
        return [x.extent for x in data.region]
    return None


def _merge_atomic_add_extents(src_extent, dst_extent):
    import builtins

    if not src_extent and not dst_extent:
        raise ValueError(f"{_ATOMIC_ADD_V1_ERR} Cannot deduce atomic_add extents.")

    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    if len(src_extent) != len(dst_extent):
        max_len = builtins.max(len(src_extent), len(dst_extent))
        src_extent = src_extent + [1] * (max_len - len(src_extent))
        dst_extent = dst_extent + [1] * (max_len - len(dst_extent))

    extent = []
    for src_val, dst_val in zip(src_extent, dst_extent):
        if isinstance(src_val, (int, float)) and isinstance(dst_val, (int, float)):
            extent.append(builtins.max(src_val, dst_val))
        else:
            if not isinstance(src_val, PrimExpr):
                src_val = tir.IntImm("int32", int(src_val))
            if not isinstance(dst_val, PrimExpr):
                dst_val = tir.IntImm("int32", int(dst_val))
            extent.append(tir.max(src_val, dst_val))
    return extent


def _atomic_add_to_tile_region(data, access_type: str, extents: list[PrimExpr]):
    data = _resolve_let_value(data)
    if isinstance(data, Buffer):
        mins = [0 for _ in data.shape]
        return _tile_region(T.BufferLoad(data, mins), access_type, *data.shape)
    if isinstance(data, BufferRegion):
        mins = [x.min for x in data.region]
        region_extents = [x.extent for x in data.region]
        if len(region_extents) < len(extents):
            raise ValueError(f"{_ATOMIC_ADD_V1_ERR} Region rank is smaller than inferred extent rank.")
        return _tile_region(
            T.BufferLoad(data.buffer, mins),
            access_type,
            *region_extents,
        )
    if isinstance(data, BufferLoad):
        indices = data.indices
        if len(indices) > len(extents):
            extents = [1] * (len(indices) - len(extents)) + list(extents)
        if len(indices) != len(extents):
            raise ValueError(f"{_ATOMIC_ADD_V1_ERR} BufferLoad rank does not match inferred extents.")
        return _tile_region(data, access_type, *extents)
    raise TypeError(f"{_ATOMIC_ADD_V1_ERR} Expected a tensor region, got {type(data).__name__}.")


def atomic_add(
    dst: Buffer | BufferRegion | BufferLoad,
    src: Buffer | BufferRegion | BufferLoad,
):
    """Atomically add a local tensor tile into a GM destination tile.

    V1 intentionally models Ascend DMA atomic add only: the destination must be
    GM, and the source must be a local tensor region that can be copied out.
    """
    dst_scope = _atomic_add_scope(dst, "dst")
    src_scope = _atomic_add_scope(src, "src")
    if dst_scope != "global":
        raise ValueError(f"{_ATOMIC_ADD_V1_ERR} dst scope must be global, got {dst_scope}.")
    if src_scope == "global":
        raise ValueError(f"{_ATOMIC_ADD_V1_ERR} src scope must be local, got global.")

    dst_extent = _atomic_add_extent(dst)
    src_extent = _atomic_add_extent(src)
    extent = _merge_atomic_add_extents(src_extent, dst_extent)
    dst_region = _atomic_add_to_tile_region(dst, "w", extent)
    src_region = _atomic_add_to_tile_region(src, "r", extent)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_atomic_add"),
        dst_region,
        src_region,
    )


def fill(buffer: Buffer | BufferRegion, value: PrimExpr):
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


def clear(buffer: Buffer | tir.Var):
    """Clear a buffer or buffer region by filling with zeros.

    Args:
        buffer: The buffer, buffer region, or variable to be cleared.
                If a tir.Var is provided, it will be resolved automatically.

    Returns:
        A TVM intrinsic call that fills the buffer with zeros.
    """
    if isinstance(buffer, tir.Var) and T.has_let_value(buffer):
        buffer_region = T.get_let_value(buffer)
        if isinstance(buffer_region, BufferRegion):
            return fill(buffer_region, 0)
        else:
            raise ValueError(f"Invalid buffer region: {buffer_region}")
    return fill(buffer, 0)


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


def sort(dst: Buffer, src: Buffer, actual_num: PrimExpr):
    """
    Performs a full sort on arbitrarily-lengthed input data with automatic internal
    alignment. Sorts each 32-element block via sort32, then merges all sorted
    blocks via merge_sort to produce the final ordered output.

    The output contains interleaved (value, index) pairs in descending order:
      [val0, idx0, val1, idx1, ...] where idx is the original position (0-based).
    Indices are generated internally; dst must be 2x the size of src.

    Args:
    dst: Destination buffer for interleaved (value, index) pairs. Must have
         at least 2 * aligned_size elements.
    src: Source buffer containing the data to be sorted.
    tmp: Temporary buffer for intermediate sort/merge results (2x the size of src).
    actual_num: The number of valid elements in src. When actual_num is less than
                the buffer size, unused positions are padded with -inf before sorting.
    """
    repeatTimes = (actual_num + 31) // 32  # ceiling to 32-aligned
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sort"),
        f"Sort<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        repeatTimes,
        actual_num,
    )


def merge_sort(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    src2: Buffer | BufferRegion | None = None,
    src3: Buffer | BufferRegion | None = None,
):
    """Performs a 2/3/4-way merge sort operation.

    This intrinsic invokes the underlying implementation to perform merge sort
    on multiple sorted blocks using AscendC::MrgSort hardware API.
    blockLen is calculated from each source buffer size.

    Hardware MrgSort format: 4 floats per element
    - Position 0: sort key (value)
    - Position 1: data (index)
    - Position 2-3: reserved/padding
    - blockLen = number of elements = buffer_size / 4

    Args:
        dst: The destination buffer or buffer region where the merged result will be stored.
        src0: First source buffer or buffer region.
        src1: Second source buffer or buffer region.
        src2: Third source buffer or buffer region (optional, for 3-way or 4-way merge).
        src3: Fourth source buffer or buffer region (optional, for 4-way merge).

    Returns:
        A TVM intrinsic call that performs the merge sort operation.
    """

    def retrieve_shape(object: Buffer | BufferRegion) -> list[int]:
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

    def retrieve_ptr(
        object: Buffer | BufferRegion,
        access_type: str = "r",
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

    src_buffers = [s for s in [src0, src1, src2, src3] if s is not None]
    num_ways = len(src_buffers)

    if num_ways < 2 or num_ways > 4:
        raise ValueError(f"merge_sort requires 2-4 source buffers, got {num_ways}")

    # Calculate blockLen for each source buffer
    # Value-index pair format: 2 floats per element [value, index]
    # blockLen = number of elements = buffer_size / 2
    # Note: Hardware MrgSort has format compatibility issues with this format
    blockLens = []
    for buf in src_buffers:
        buf_size = math.prod(retrieve_shape(buf))
        blockLens.append(buf_size // 2)  # Value-index pair format

    args = (
        [
            f"MergeSort<{_dtype(dst)}>",
            num_ways,
            retrieve_ptr(dst, "w"),
        ]
        + [retrieve_ptr(buf, "r") for buf in src_buffers]
        + blockLens
    )

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_merge_sort"),
        *args,
    )


def topk(dst: Buffer, src: Buffer, K: PrimExpr, actual_num: PrimExpr):
    """Performs a TopK operation by sorting the source data and extracting the top K elements.

    Internally calls Sort on the source data, then copies the top K interleaved
    (value, index) pairs into the destination buffer.

    Args:
        dst: Destination buffer for top K interleaved (value, index) pairs.
             Must have at least 2*K elements.
        src: Source buffer containing the data to find top K from.
             Assumes src has static shape for buffer sizing.
        K: Number of top elements to extract.
        actual_num: The number of valid elements in src (can be symbolic for dynamic shapes).

    Returns:
        A TVM intrinsic call that performs the TopK operation.
    """
    from tvm.arith import Analyzer

    analyzer = Analyzer()
    max_actual_num = 0

    for dim in src.shape:
        dim_simplified = analyzer.simplify(dim)
        if isinstance(dim_simplified, tir.IntImm):
            max_actual_num += dim_simplified.value
        else:
            raise ValueError(
                f"topk requires src buffer with static shape for buffer sizing. "
                f"Found dynamic dimension: {dim}. Please ensure src buffer has compile-time constant shape."
            )

    repeatTimes = (max_actual_num + 31) // 32
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_topk"),
        f"TopK<{_dtype(dst)}>",
        dst.access_ptr("w"),
        src.access_ptr("r"),
        K,
        repeatTimes,
        actual_num,
        max_actual_num,
    )


def gather_mask(dst: Buffer, src: Buffer, src1Pattern: str | Buffer):
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
    dst: Buffer | BufferRegion,
    selMask: Buffer,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferLoad | PrimExpr,
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

    def retrieve_shape(object: Buffer | BufferRegion) -> list[int]:
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
        object: Buffer | BufferRegion,
        access_type: str = "r",
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
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr | float,
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


def add(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise addition: dst = src0 + src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "add")


def sub(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad):
    """Performs element-wise subtraction: dst = src0 - src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "sub")


def mul(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise multiplication: dst = src0 * src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "mul")


def div(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad):
    """Performs element-wise division: dst = src0 / src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer or BufferLoad).
    """
    return binary_op(dst, src0, src1, "div")


def max(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise maximum: dst = max(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "max")


def min(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise minimum: dst = min(src0, src1).

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "min")


def bitwise_and(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise bitwise AND: dst = src0 & src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_and")


def bitwise_or(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion | BufferLoad | PrimExpr):
    """Performs element-wise bitwise OR: dst = src0 | src1.

    Args:
        dst: The destination buffer.
        src0: The first source buffer.
        src1: The second source operand (Buffer, BufferLoad, or Scalar).
    """
    return binary_op(dst, src0, src1, "bitwise_or")


def unary_op(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, op: str):
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


def exp(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise exponential: dst = exp(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "exp")


def sigmoid(dst: Buffer | BufferRegion, src: Buffer | BufferRegion):
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
        size,
    )


def silu(dst: Buffer | BufferRegion, src: Buffer | BufferRegion):
    """Performs element-wise SiLU (Swish) activation: dst = src * sigmoid(src).

    SiLU (Sigmoid Linear Unit) is also known as Swish activation function.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer.

    Returns:
        A TVM intrinsic call that performs the Silu operation.

    Note:
        - Supports data types: half, float (Atlas A2/A3)
        - SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, buffer_extent = _handle_buffer_region(dst, "w")
        size = math.prod(buffer_extent)
    else:
        dst_ptr = dst.access_ptr("w")
        size = math.prod(dst.shape)

    if isinstance(src, BufferRegion):
        src_ptr, _ = _handle_buffer_region(src, "r")
    else:
        src_ptr = src.access_ptr("r")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_silu"),
        dst_ptr,
        src_ptr,
        size,
    )


def ln(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise natural logarithm: dst = ln(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "ln")


def abs(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise absolute value: dst = abs(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "abs")


def reciprocal(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise reciprocal: dst = 1 / src0.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "reciprocal")


def sqrt(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise square root: dst = sqrt(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "sqrt")


def rsqrt(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise reciprocal square root: dst = 1 / sqrt(src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "rsqrt")


def relu(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise Rectified Linear Unit (ReLU): dst = max(0, src0).

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "relu")


def bitwise_not(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion):
    """Performs element-wise bitwise NOT (inversion): dst = ~src0.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
    """
    return unary_op(dst, src0, "bitwise_not")


def scalar_op(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalar_value: PrimExpr,
    op_tl: str,
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


def leaky_relu(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, scalar_value: PrimExpr):  # type: ignore  # noqa: F821
    """Performs element-wise Leaky ReLU activation.

    Formula: dst = src0 if src0 >= 0 else src0 * scalar_value

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalar_value: The negative slope coefficient.
    """
    return scalar_op(dst, src0, scalar_value, "leaky_relu")


def axpy(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, scalar_value: PrimExpr):  # noqa: F821
    """Performs element-wise AXPY operation: dst = scalar_value * src0 + dst.

    Note: This operation updates the destination buffer in-place by adding
    the scaled source buffer.

    Args:
        dst: The destination buffer (acts as both operand Y and output).
        src0: The source buffer X.
        scalar_value: The scalar alpha.
    """
    return scalar_op(dst, src0, scalar_value, "axpy")


def mul_add_dst(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
):
    """Performs element-wise multiply-add: dst = src0 * src1 + dst.

    This operation performs a fused multiply-add where src0 and src1 are multiplied,
    then added to the existing values in dst, with the result stored back in dst.

    Args:
        dst: The destination buffer (also acts as the accumulator input).
             Must be in UB (Unified Buffer) scope.
        src0: The first source buffer for multiplication.
        src1: The second source buffer for multiplication.

    Returns:
        A TVM intrinsic call that performs the MulAddDst operation.

    Note:
        - Supports data types: half, float (Atlas A2/A3)
        - Also supports: int16_t, uint16_t, int32_t, uint32_t (Atlas 200I/500 A2)
        - dst acts as both input (accumulator) and output
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "rw")
    else:
        dst_ptr = dst.access_ptr("rw")
        dst_extent = dst.shape

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_extent = src0.shape

    if isinstance(src1, BufferRegion):
        src1_ptr, src1_extent = _handle_buffer_region(src1, "r")
    else:
        src1_ptr = src1.access_ptr("r")
        src1_extent = src1.shape

    size_dst = math.prod(dst_extent)
    size_src0 = math.prod(src0_extent)
    size_src1 = math.prod(src1_extent)

    assert size_dst == size_src0 == size_src1, "dst, src0, and src1 must have the same size"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_mul_add_dst"),
        dst_ptr,
        src0_ptr,
        src1_ptr,
        size_dst,
    )


def bitwise_lshift(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, scalarValue: PrimExpr):  # noqa: F821
    """Performs element-wise bitwise left shift: dst = src0 << scalarValue.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalarValue: The number of bits to shift (scalar).
    """
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
        tir.op.Op.get("tl.ascend_bitwise_lshift"),
        dst_ptr,
        src0_ptr,
        scalarValue,
        size_0,
    )


def bitwise_rshift(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, scalarValue: PrimExpr):  # noqa: F821
    """Performs element-wise bitwise right shift: dst = src0 >> scalarValue.

    Args:
        dst: The destination buffer.
        src0: The source buffer.
        scalarValue: The number of bits to shift (scalar).
    """
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
        tir.op.Op.get("tl.ascend_bitwise_rshift"),
        dst_ptr,
        src0_ptr,
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


def gather(dst: Buffer | BufferRegion, src: Buffer | BufferRegion, src_offset: Buffer | BufferRegion, src_base_addr: PrimExpr):  # noqa: F821
    """Performs a gather operation.

    This intrinsic gathers elements from the source buffer based on the provided
    offsets and a base address, storing the result in the destination buffer.

    Args:
        dst: The destination buffer where the gathered data will be stored.
        src: The source buffer containing the data table.
        src_offset: The buffer containing offsets/indices for gathering.
        src_base_addr: The base address offset to be added to the gather indices.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, _ = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")

    if isinstance(src, BufferRegion):
        src_ptr, src_extent = _handle_buffer_region(src, "r")
        size = math.prod(src_extent)
    else:
        src_ptr = src.access_ptr("r")
        size = math.prod(src.shape)

    if isinstance(src_offset, BufferRegion):
        src_offset_ptr, _ = _handle_buffer_region(src_offset, "r")
    else:
        src_offset_ptr = src_offset.access_ptr("r")

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_gather"),
        dst_ptr,
        src_ptr,
        src_offset_ptr,
        src_base_addr,
        size,
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
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
    mode: str,  # noqa: F821, FA100
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
    if isinstance(dst, BufferRegion):
        dst_ptr, _ = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_extent = _handle_buffer_region(src0, "r")
        size_0 = math.prod(src0_extent)
    else:
        src0_ptr = src0.access_ptr("r")
        size_0 = math.prod(src0.shape)

    dst_size = size_0

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
    elif isinstance(src1, BufferRegion):
        src1_ptr, _ = _handle_buffer_region(src1, "r")
        return T.call_intrin(
            "handle",
            tir.op.Op.get("tl.ascend_compare"),
            dst_ptr,
            src0_ptr,
            src1_ptr,
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


def cast(dst: Buffer | BufferRegion, src: Buffer | BufferRegion, mode: str, count: PrimExpr):  # noqa: F821
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

    if isinstance(dst, BufferRegion):
        dst_ptr, _ = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")

    if isinstance(src, BufferRegion):
        src_ptr, _ = _handle_buffer_region(src, "r")
    else:
        src_ptr = src.access_ptr("r")

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_cast"),
        dst_ptr,
        src_ptr,
        mode,
        count,
    )


def sin(dst: Buffer | BufferRegion, src: Buffer | BufferRegion):  # noqa: F821
    """Performs element-wise sine calculation: dst = sin(src).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.

    Returns:
        A TVM intrinsic call that performs the sine operation.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
        size_2 = math.prod(dst_extent)
    else:
        dst_ptr = dst.access_ptr("w")
        size_2 = math.prod(dst.shape)

    if isinstance(src, BufferRegion):
        src_ptr, src_extent = _handle_buffer_region(src, "r")
        size_0 = math.prod(src_extent)
    else:
        src_ptr = src.access_ptr("r")
        size_0 = math.prod(src.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_sin"),
        dst_ptr,
        src_ptr,
        size_0,
    )


def cos(dst: Buffer | BufferRegion, src: Buffer | BufferRegion):  # noqa: F821
    """Performs element-wise cosine calculation: dst = cos(src).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The source buffer containing the input data.

    Returns:
        A TVM intrinsic call that performs the cosine operation.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_extent = _handle_buffer_region(dst, "w")
        size_2 = math.prod(dst_extent)
    else:
        dst_ptr = dst.access_ptr("w")
        size_2 = math.prod(dst.shape)

    if isinstance(src, BufferRegion):
        src_ptr, src_extent = _handle_buffer_region(src, "r")
        size_0 = math.prod(src_extent)
    else:
        src_ptr = src.access_ptr("r")
        size_0 = math.prod(src.shape)

    assert size_0 == size_2, "size must be same"

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_cos"),
        dst_ptr,
        src_ptr,
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
def pow(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion):  # noqa: F821
    """Performs element-wise power calculation: dst = src0 ^ src1.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The base buffer.
        src1: The exponent buffer.

    Returns:
        A TVM intrinsic call that performs the power operation.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, _ = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")

    if isinstance(src0, BufferRegion):
        src0_ptr, _ = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")

    if isinstance(src1, BufferRegion):
        src1_ptr, _ = _handle_buffer_region(src1, "r")
    else:
        src1_ptr = src1.access_ptr("r")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_pow"),
        dst_ptr,
        src0_ptr,
        src1_ptr,
    )


def bitwise_xor(dst: Buffer | BufferRegion, src0: Buffer | BufferRegion, src1: Buffer | BufferRegion):  # noqa: F821
    """Performs element-wise bitwise XOR operation: dst = src0 ^ src1.

    Args:
        dst: The destination buffer where the result will be stored.
        src0: The first source operand buffer.
        src1: The second source operand buffer.

    Returns:
        A TVM intrinsic call that performs the bitwise XOR operation.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, _ = _handle_buffer_region(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")

    if isinstance(src0, BufferRegion):
        src0_ptr, _ = _handle_buffer_region(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")

    if isinstance(src1, BufferRegion):
        src1_ptr, _ = _handle_buffer_region(src1, "r")
    else:
        src1_ptr = src1.access_ptr("r")

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_bitwise_xor"), dst_ptr, src0_ptr, src1_ptr)


def clamp_max(out: Buffer | BufferRegion, buffer: Buffer | BufferRegion, scalar_value: PrimExpr, count: PrimExpr):  # noqa: F821
    """_summary_
    Clip tensor elements to no more than scalar_value, replace elements larger than scalar_value with scalar_value,
    keep original values for elements less than or equal to scalar_value

    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        scalar_value: The max scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp_max operation.
    """
    if isinstance(out, BufferRegion):
        out_ptr, _ = _handle_buffer_region(out, "w")
    else:
        out_ptr = out.access_ptr("w")

    if isinstance(buffer, BufferRegion):
        buffer_ptr, _ = _handle_buffer_region(buffer, "r")
    else:
        buffer_ptr = buffer.access_ptr("r")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp_max"),
        f"ClampMax<{_dtype(buffer)}>",
        out_ptr,
        buffer_ptr,
        scalar_value,
        count,
    )


def clamp_min(out: Buffer | BufferRegion, buffer: Buffer | BufferRegion, scalar_value: PrimExpr, count: PrimExpr):  # noqa: F821
    """
    Clip tensor elements to no less than v, replace elements smaller than scalar_value with scalar_value,
    keep original values for elements greater than or equal to scalar_value

    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        scalar_value: The min scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp_min operation.
    """
    if isinstance(out, BufferRegion):
        out_ptr, _ = _handle_buffer_region(out, "w")
    else:
        out_ptr = out.access_ptr("w")

    if isinstance(buffer, BufferRegion):
        buffer_ptr, _ = _handle_buffer_region(buffer, "r")
    else:
        buffer_ptr = buffer.access_ptr("r")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp_min"),
        f"ClampMin<{_dtype(buffer)}>",
        out_ptr,
        buffer_ptr,
        scalar_value,
        count,
    )


def clamp(out: Buffer | BufferRegion, buffer: Buffer | BufferRegion, min_scalar: PrimExpr, max_scalar: PrimExpr, count: PrimExpr):  # noqa: F821
    """
    Clip tensor elements to [min_scalar, max_scalar] range, replace out-of-bounds values with boundary values

    Args:
        out: The destination buffer where the result will be stored.
        buffer: The first source operand buffer.
        min_scalar: The min scalar value
        max_scalar: The max scalar value
        count: The size of tensor out

    Returns:
        A TVM intrinsic call that performs the clamp operation.
    """
    if isinstance(out, BufferRegion):
        out_ptr, _ = _handle_buffer_region(out, "w")
    else:
        out_ptr = out.access_ptr("w")

    if isinstance(buffer, BufferRegion):
        buffer_ptr, _ = _handle_buffer_region(buffer, "r")
    else:
        buffer_ptr = buffer.access_ptr("r")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_clamp"),
        f"Clamp<{_dtype(buffer)}>",
        out_ptr,
        buffer_ptr,
        min_scalar,
        max_scalar,
        count,
    )


def round(out: Buffer | BufferRegion, buffer: Buffer | BufferRegion, count: PrimExpr):  # noqa: F821
    if isinstance(out, BufferRegion):
        out_ptr, _ = _handle_buffer_region(out, "w")
    else:
        out_ptr = out.access_ptr("w")

    if isinstance(buffer, BufferRegion):
        buffer_ptr, _ = _handle_buffer_region(buffer, "r")
    else:
        buffer_ptr = buffer.access_ptr("r")

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_round"), out_ptr, buffer_ptr, count)


def broadcast(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    axis: int | None = None,
):
    """Generates a TIR intrinsic call for the Ascend `Broadcast` operation.

    This function performs a broadcast copy from the source buffer (`src`) to the
    destination buffer (`dst`). It automatically infers the broadcasting axis
    based on the shapes of the input buffers, or uses the explicitly provided axis.

    Args:
        dst: Destination buffer (must be in UB).
        src: Source buffer (must be in UB).
        axis: Broadcasting axis (0 or 1). If None, auto-inferred.

    Returns:
        tvm.tir.Call: Intrinsic call for AscendC Broadcast API.

    Raises:
        ValueError: If shapes are incompatible for broadcasting.
    """
    # --- 1. Extract pointers and shapes ---
    dst_ptr, dst_extent = _handle_buffer_region(dst, "w") if isinstance(dst, BufferRegion) else (dst.access_ptr("w"), dst.shape)
    src_ptr, src_extent = _handle_buffer_region(src, "r") if isinstance(src, BufferRegion) else (src.access_ptr("r"), src.shape)
    dtype = _dtype(src)
    # 3D to 2D conversion
    dst_extent = list(dst_extent)[-2:]
    src_extent = list(src_extent)[-2:]

    dst_dim, src_dim = len(dst_extent), len(src_extent)
    if dst_dim not in [1, 2] or src_dim not in [1, 2]:
        raise ValueError(f"Ascend Broadcast only supports 1D or 2D: dst_dim={dst_dim}, src_dim={src_dim}")

    # --- 2. Normalize 1D src to 2D ---
    if src_dim == 1 and dst_dim == 2:
        if axis == 0 or (axis is None and dst_extent[1] == src_extent[0]):
            src_extent, axis = [1, src_extent[0]], 0
        elif axis == 1 or (axis is None and dst_extent[0] == src_extent[0]):
            src_extent, axis = [src_extent[0], 1], 1
        else:
            raise ValueError(f"Cannot broadcast 1D src {src_extent} to 2D dst {dst_extent}, axis={axis}")
        src_dim = 2

    if src_dim != dst_dim:
        raise ValueError(f"Dimension mismatch: dst_dim={dst_dim} != src_dim={src_dim}")

    # --- 3. Auto-infer axis (2D case) ---
    if axis is None:
        if dst_dim == 2:
            if src_extent[0] == 1 and dst_extent[0] != 1:
                axis = 0
            elif src_extent[1] == 1 and dst_extent[1] != 1:
                axis = 1
            else:
                axis = 0  # No broadcast, default axis=0
        else:
            axis = 0  # 1D case

    # --- 4. Shape validation ---
    if dst_dim == 2:
        if axis not in [0, 1]:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

        if src_extent[axis] == 1:
            # Broadcast case: validate non-broadcast dimension matches
            other_axis = 1 - axis
            if src_extent[other_axis] != dst_extent[other_axis]:
                raise ValueError(
                    f"Broadcast dimension mismatch: src[{other_axis}]={src_extent[other_axis]} "
                    f"!= dst[{other_axis}]={dst_extent[other_axis]}"
                )
        elif src_extent != dst_extent:
            # No broadcast: shapes must match exactly
            raise ValueError(f"Shapes must match when src[{axis}] != 1: src={src_extent}, dst={dst_extent}")
    elif dst_dim == 1:
        # 1D case: axis must be 0
        if axis != 0:
            raise ValueError(f"1D broadcast requires axis=0, got axis={axis}")
        # broadcast requires src[0] == 1, otherwise shapes must match
        if src_extent[0] != 1 and src_extent != dst_extent:
            raise ValueError(f"1D broadcast requires src[0]=1 or shapes must match: src={src_extent}, dst={dst_extent}")

    # --- 5. Generate TIR intrinsic call ---
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_broadcast"),
        f"Broadcast<{dtype}, {dst_dim}, {axis}, false>",
        dst_ptr,
        src_ptr,
        dst_dim,
        *dst_extent,
        *src_extent,
    )


def row_expand_mul(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    tmp: Buffer | BufferRegion | None = None,
):
    """Performs row-wise broadcast multiply: dst[i, j] = src0[i, j] * src1[i].

    This is a fused row-broadcast and multiply operation backed by the PTO
    TROWEXPANDMUL instruction.  Each element in src1 is broadcast across all
    columns of the corresponding row in src0.

    Args:
        dst: Destination buffer (UB, shape [rows, cols]).
        src0: Source data buffer (UB, shape [rows, cols]).
        src1: Per-row scalars. Accepts 1D [R] or 2D [1, R] (row vector);
              both are treated as R scalars matching dst rows.
        tmp: Optional temporary buffer for internal workspace.

    Returns:
        A TVM intrinsic call for the TROWEXPANDMUL operation.
    """
    if isinstance(dst, BufferRegion):
        dst_ptr, dst_shape = _handle_buffer_region_2d(dst, "w")
    else:
        dst_ptr = dst.access_ptr("w")
        dst_shape = list(dst.shape[-2:])

    if isinstance(src0, BufferRegion):
        src0_ptr, src0_shape = _handle_buffer_region_2d(src0, "r")
    else:
        src0_ptr = src0.access_ptr("r")
        src0_shape = list(src0.shape[-2:])

    if isinstance(src1, BufferRegion):
        src1_ptr, src1_nd_extent = _handle_buffer_region(src1, "r")
        src1_full_shape = [src1_nd_extent[-1]] if len(src1_nd_extent) >= 2 else src1_nd_extent
    else:
        src1_ptr = src1.access_ptr("r")
        src1_full_shape = list(src1.shape)

    if len(src1_full_shape) == 1:
        # 1D [R] → R per-row scalars
        src1_len = src1_full_shape[0]
    elif len(src1_full_shape) == 2:
        s0, s1 = src1_full_shape[-2], src1_full_shape[-1]
        if s0 == 1:
            src1_len = s1  # [1, R]
        elif s1 == 1:
            src1_len = s0  # [R, 1]
        else:
            raise ValueError(
                f"src1 must be 1D [R], [1, R], or [R, 1]; got {src1_full_shape}"
            )
    else:
        raise ValueError(
            f"src1 must be 1D or 2D, got shape {src1_full_shape}"
        )

    if len(dst_shape) != 2 or len(src0_shape) != 2:
        raise ValueError("row_expand_mul requires 2D buffers for dst and src0.")

    if dst_shape != src0_shape:
        raise ValueError(
            f"dst and src0 shapes must match: dst={dst_shape}, src0={src0_shape}"
        )

    if src1_len != dst_shape[0]:
        raise ValueError(
            f"src1 scalar count must match dst rows: src1={src1_len}, dst[0]={dst_shape[0]}"
        )

    dtype = _dtype(src0)
    args = [
        f"RowExpandMul<{dtype}>",
        dst_ptr,
        src0_ptr,
        src1_ptr,
    ]
    if tmp is not None:
        if isinstance(tmp, BufferRegion):
            tmp_ptr, _ = _handle_buffer_region(tmp, "rw")
        else:
            tmp_ptr = tmp.access_ptr("rw")
        args.append(tmp_ptr)

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_row_expand_mul"),
        *args,
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


def reduce_sum_experiment(dst: Buffer, src: Buffer, count: PrimExpr):
    """Performs summation of all input data.

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        count: The number of elements to process.
    """

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reducesum_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
        count,
    )


def reduce_sum_mask_experiment(dst: Buffer, src: Buffer, mask: PrimExpr, repeatTime: PrimExpr, srcRepStride: PrimExpr):
    """Performs summation of all input data(High-dimensional tensor slicing and computation).

    Args:
        dst: The destination buffer where the result will be stored.
        src: The base buffer.
        mask: Used to control the elements participating in the computation within each iteration.
        repeatTime: Number of iterations.
        srcRepStride: The address step size of the source operand between adjacent iterations.
    """

    return T.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_reducesum_mask_experiment"),
        dst.access_ptr("w"),
        src.access_ptr("r"),
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
