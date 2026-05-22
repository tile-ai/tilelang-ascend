# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from __future__ import annotations
from enum import IntEnum
from tilelang import language as T
from tvm import arith, ir, tir


def region(buffer: tir.BufferLoad, access_type: str, *args: tir.PrimExpr):
    """Create a memory region descriptor for tile operations.

    Args:
        buffer (tir.BufferLoad): The buffer to create a region for
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        *args (tir.PrimExpr): Extent expressions defining the region size

    Returns:
        tir.Call: A region descriptor for tile operations
    """
    access_type = {"r": 1, "w": 2, "rw": 3}[access_type]
    return tir.call_intrin("handle", tir.op.Op.get("tl.region"), buffer, access_type, *args)


def buffer_to_tile_region(buffer: tir.Buffer, access_type: str):
    """Convert a TVM buffer to a tile region descriptor.

    Args:
        buffer (tir.Buffer): The buffer to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor covering the entire buffer
    """
    mins = [0 for _ in buffer.shape]
    extents = [x for x in buffer.shape]
    return region(T.BufferLoad(buffer, mins), access_type, *extents)


def buffer_load_to_tile_region(load: tir.BufferLoad, access_type: str, extents: list[tir.PrimExpr]):
    """Convert a buffer load operation to a tile region descriptor.

    Args:
        load (tir.BufferLoad): The buffer load operation
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write
        extents (List[tir.PrimExpr]): List of expressions defining the region size

    Returns:
        tir.Call: A region descriptor for the loaded area
    """
    indices = load.indices
    if len(indices) > len(extents):
        # (f"mismatch between indices and extents for buffer load {load}: indices = {indices}, extents = {extents}, "
        # f"region will be expanded in the last 2 dimensions")
        new_extents = []
        for _ in range(len(indices) - len(extents)):
            new_extents.append(1)
        for extent in extents:
            new_extents.append(extent)
        extents = new_extents
    assert len(indices) == len(extents), f"indices = {indices}, extents = {extents}"
    return region(load, access_type, *extents)


def buffer_region_to_tile_region(buffer_region: tir.BufferRegion, access_type: str, extents: list[tir.PrimExpr]):
    """Convert a buffer region to a tile region descriptor.

    Args:
        buffer_region (tir.BufferRegion): The buffer region to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor for the specified buffer region
    """
    mins = [x.min for x in buffer_region.region]
    region_extents = [x.extent for x in buffer_region.region]
    assert len(region_extents) >= len(extents), f"region_extents must be >= extents, region_extents = {region_extents}, extents = {extents}"

    return region(T.BufferLoad(buffer_region.buffer, mins), access_type, *region_extents)


def copy(
    src: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
    dst: tir.Buffer | tir.BufferLoad,
    coalesced_width: int | None = None,
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    assert src_extent or dst_extent, "Can't deduce copy extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    if len(src_extent) != len(dst_extent):
        max_len = max(len(src_extent), len(dst_extent))
        if len(src_extent) < max_len:
            src_extent = src_extent + [1] * (max_len - len(src_extent))
        if len(dst_extent) < max_len:
            dst_extent = dst_extent + [1] * (max_len - len(dst_extent))

    extent = []
    for i in range(len(src_extent)):
        src_val = src_extent[i]
        dst_val = dst_extent[i]

        if isinstance(src_val, (int, float)) and isinstance(dst_val, (int, float)):
            extent.append(max(src_val, dst_val))
        else:
            if not isinstance(src_val, tir.PrimExpr):
                src_val = tir.IntImm("int32", int(src_val))
            if not isinstance(dst_val, tir.PrimExpr):
                dst_val = tir.IntImm("int32", int(dst_val))
            extent.append(tir.max(src_val, dst_val))

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")
    if coalesced_width is not None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst, coalesced_width)
    else:
        return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst)


def c2d_im2col(
    img: tir.Buffer,
    col: tir.Buffer,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
):
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.c2d_im2col"),
        img.access_ptr("r"),
        col.access_ptr("w"),
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
    )


def _is_cross_cv_copy(src: tir.Buffer, dst: tir.Buffer) -> bool:
    """Check if this is a cross-CV copy (UB→L1 or L0C→UB only)."""
    src_scope: str = src.scope()
    dst_scope: str = dst.scope()
    return (
        (src_scope == "shared" and dst_scope == "shared.dyn")  # UB → L1
        or (src_scope == "wmma.accumulator" and dst_scope == "shared")  # L0C → UB
    )


def _is_almost_zero(expr: tir.PrimExpr) -> bool:
    """Check if simplified expression represents a value in [0, 1].

    True if IntImm(0) (exact 2x) or FloorMod(X, 2) (~2x for symbolic M vs M//2).
    """
    if isinstance(expr, tir.IntImm):
        return int(expr.value) == 0
    if isinstance(expr, tir.FloorMod):
        return isinstance(expr.b, tir.IntImm) and int(expr.b.value) == 2
    return False


def _has_2x_ratio(s: tir.PrimExpr, d: tir.PrimExpr) -> bool:
    """Check if s and d have approximately 2x ratio.

    Instead of proving s == d*2 (requires divisibility info TVM can't track),
    computes the difference and checks if it simplifies to something bounded
    to [0, 1]. Works for both IntImm and symbolic PrimExpr.
    """
    analyzer: arith.Analyzer = arith.Analyzer()
    two: tir.IntImm = tir.IntImm("int32", 2)
    return _is_almost_zero(analyzer.simplify(s - d * two)) or _is_almost_zero(analyzer.simplify(d - s * two))


def _is_equal_dim(s: tir.PrimExpr, d: tir.PrimExpr) -> bool:
    """Check if two shape dims are equal, handling both int and PrimExpr."""
    if isinstance(s, (int, tir.IntImm)) and isinstance(d, (int, tir.IntImm)):
        return int(s) == int(d)
    analyzer: arith.Analyzer = arith.Analyzer()
    diff: tir.PrimExpr = analyzer.simplify(s - d)
    return isinstance(diff, tir.IntImm) and int(diff.value) == 0


def _check_cross_cv_shapes(src_shape: list[tir.PrimExpr], dst_shape: list[tir.PrimExpr]) -> None:
    """Check shapes for cross-CV copy, allowing one dim to differ by 2x."""
    if len(src_shape) != len(dst_shape):
        raise ValueError(f"Shape dimension mismatch: {src_shape} vs {dst_shape}")

    diff_count: int = 0
    for s, d in zip(src_shape, dst_shape):
        if _is_equal_dim(s, d):
            continue
        if not _has_2x_ratio(s, d):
            raise ValueError(f"Shape mismatch: {src_shape} vs {dst_shape} (dimension differs, not 2x ratio)")
        diff_count += 1

    if diff_count > 1:
        raise ValueError(f"More than one dimension differs: {src_shape} vs {dst_shape}")


def npu_copy_v2(
    src: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
    dst: tir.Buffer | tir.BufferLoad,
    enable_relu: bool = False,
    transpose: bool | None = False,  # for copy_l1_to_l0 param: tranpose l1
    pad_value: float | int | tir.PrimExpr | None = None,
    tmp: tir.Buffer | tir.BufferLoad | None = None,
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        enable_relu (bool): Whether to enable ReLU. Defaults to False.
        transpose (Optional[bool]): Whether to transpose for copy_l1_to_l0. Defaults to False.
        pad_value (Optional[Union[float, int, tir.PrimExpr]]): Value to fill in UB unused area.
            Supports float, int, tir.FloatImm, tir.IntImm, tir.PrimExpr (e.g., -T.infinity(dtype)).
            Defaults to 0.
        tmp (Optional[Union[tir.Buffer, tir.BufferLoad]]): Temporary buffer for UB->L1 copy
            on A5 platform. Used for ND->Nz format conversion. Defaults to None.
            Only required when copying from UB to L1 on A5.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer) and not transpose:
        if _is_cross_cv_copy(src, dst):
            _check_cross_cv_shapes(src.shape, dst.shape)
        else:
            ir.assert_structural_equal(src.shape, dst.shape)

    # src_shape = src.shape if isinstance(src, tir.Buffer) else src.buffer.shape

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    assert src_extent or dst_extent, "Can't deduce copy extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    if len(src_extent) != len(dst_extent):
        max_len = max(len(src_extent), len(dst_extent))
        if len(src_extent) < max_len:
            src_extent = src_extent + [1] * (max_len - len(src_extent))
        if len(dst_extent) < max_len:
            dst_extent = dst_extent + [1] * (max_len - len(dst_extent))

    extent = []
    for i in range(len(src_extent)):
        src_val = src_extent[i]
        dst_val = dst_extent[i]

        if isinstance(src_val, (int, float)) and isinstance(dst_val, (int, float)):
            extent.append(max(src_val, dst_val))
        else:
            if not isinstance(src_val, tir.PrimExpr):
                src_val = tir.IntImm("int32", int(src_val))
            if not isinstance(dst_val, tir.PrimExpr):
                dst_val = tir.IntImm("int32", int(dst_val))
            extent.append(tir.max(src_val, dst_val))

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent[-len(data.buffer.shape) :])
        else:
            return buffer_load_to_tile_region(data, access_type, extent[-len(data.buffer.shape) :])

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")

    # Handle pad_value parameter
    if pad_value is None:
        pad_value = 0
    if isinstance(pad_value, (tir.FloatImm, tir.IntImm, tir.PrimExpr)):
        pad_value_expr = pad_value
    elif isinstance(pad_value, float):
        pad_value_expr = tir.FloatImm("float32", pad_value)
    else:
        pad_value_expr = tir.IntImm("int32", int(pad_value))

    # Handle tmp parameter (for UB->L1 copy on A5)
    if tmp is None:
        tmp_region = tir.IntImm("int32", 0)
    else:
        tmp_region = _to_region(tmp, "rw")

    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_copy"), src, dst, enable_relu, transpose, pad_value_expr, tmp_region)


class CopyCVMode(IntEnum):
    SingleVec0 = 0
    SingleVec1 = 1
    DualSplitM = 2
    DualSplitN = 3


def copy_cv_experiment(src: tir.Buffer, dst: tir.Buffer, mode: int | CopyCVMode = CopyCVMode.DualSplitM):
    """L0C to UB direct copy using TMOV (PTO A5 only).

    Args:
        dst: Destination buffer (UB, 'shared' scope)
        src: Source buffer (L0C, 'wmma.accumulator' scope)
        mode: CopyCVMode (TMOV AccToVecMode)  (default CopyCVMode.DualSplitM)
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_copy_cv_experiment"),
        src.access_ptr("r"),
        dst.access_ptr("w"),
        int(mode),
    )


def copy_vc_experiment(src: tir.Buffer, dst: tir.Buffer, tmp: tir.Buffer, mode: int = 0):
    """UB to L1 direct copy using TINSERT with ND→NZ conversion (PTO A5 only).

    Args:
        dst: Destination buffer (L1, 'shared.dyn' scope)
        src: Source buffer (UB, 'shared' scope) — ND format
        tmp: Temporary buffer (UB, 'shared' scope) — scratch for NZ conversion
        mode: TINSERT TInsertMode (0=default, 2=SPLIT2, 3=SPLIT4)
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.ascend_copy_vc_experiment"),
        src.access_ptr("r"),
        dst.access_ptr("w"),
        tmp.access_ptr("rw"),
        int(mode),
    )
