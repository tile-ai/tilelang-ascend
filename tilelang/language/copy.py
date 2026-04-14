# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Union, List, Optional
from tilelang import language as T
from tvm import tir


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
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.region"), buffer, access_type, *args
    )


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


def buffer_load_to_tile_region(
    load: tir.BufferLoad, access_type: str, extents: List[tir.PrimExpr]
):
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
    indices_head = []
    for idx in load.indices:
        if isinstance(idx, tir.expr.Ramp):
            indices_head.append(idx.base)
        else:
            indices_head.append(idx)
    return region(T.BufferLoad(load.buffer, indices_head), access_type, *extents)


def buffer_region_to_tile_region(
    buffer_region: tir.BufferRegion, access_type: str, extents: List[tir.PrimExpr]
):
    """Convert a buffer region to a tile region descriptor.

    Args:
        buffer_region (tir.BufferRegion): The buffer region to convert
        access_type (str): Type of access - 'r' for read, 'w' for write, 'rw' for read-write

    Returns:
        tir.Call: A region descriptor for the specified buffer region
    """
    mins = [x.min for x in buffer_region.region]
    region_extents = [x.extent for x in buffer_region.region]
    assert len(region_extents) >= len(extents), (
        f"region_extents must be >= extents, region_extents = {region_extents}, extents = {extents}"
    )

    return region(
        T.BufferLoad(buffer_region.buffer, mins), access_type, *region_extents
    )


def copy(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    coalesced_width: Optional[int] = None,
    size: Optional[List] = None,
):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.
        size (Optional[list], optional): Explicit extent for copy region. Legacy API compatibility.


    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return list(data.shape)
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, tir.BufferLoad):
            # BufferLoad (e.g. B[bx, by]) cannot infer extent from indices alone.
            # Return None so caller uses the other side's extent for block copy.
            return None
        else:
            return None

    def _is_slice(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        return isinstance(data, tir.BufferRegion)

    has_explicit_size = size is not None and len(size) > 0
    if has_explicit_size and (_is_slice(src) or _is_slice(dst)):
        raise ValueError(
            "T.copy: cannot use both slice syntax and the size parameter. "
        )

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    def _borrowed_extent(data, peer_extent):
        if has_explicit_size:
            return list(size)
        if isinstance(data, tir.Buffer):
            return list(data.shape)
        if isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        # BufferLoad only carries a starting point, so when size=... is absent
        # it has to borrow extents from the opposite operand.
        assert peer_extent is not None, (
            "T.copy cannot deduce copy extents from two BufferLoad operands; "
            "use slice syntax on one side or pass size=[...]."
        )
        return list(peer_extent)

    def _to_region(data, access_type, peer_extent, peer_is_slice):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            if not has_explicit_size:
                # When a plain buffer is paired with an explicit slice of the
                # same rank, reuse the peer extents so tail-tile copies like
                # T.copy(A[bx:..., by:...], UB) and T.copy(UB, C[bx:..., by:...])
                # keep matching logical shapes. For rank-mismatch singleton
                # cases, preserve whole-buffer semantics and let the backend
                # perform shape alignment.
                if (
                    peer_is_slice
                    and peer_extent is not None
                    and len(peer_extent) == len(data.shape)
                ):
                    return buffer_load_to_tile_region(
                        T.BufferLoad(data, [0] * len(data.shape)),
                        access_type,
                        list(peer_extent),
                    )
                return buffer_to_tile_region(data, access_type)
            ndim = len(data.shape)
            extent = _borrowed_extent(data, peer_extent)
            trailing = extent[-ndim:] if len(extent) >= ndim else extent
            return buffer_load_to_tile_region(
                T.BufferLoad(data, [0] * ndim), access_type, trailing
            )
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(
                data, access_type, [x.extent for x in data.region]
            )
        else:
            ndim = len(data.buffer.shape)
            extent = _borrowed_extent(data, peer_extent)
            trailing = extent[-ndim:] if len(extent) >= ndim else extent
            return buffer_load_to_tile_region(data, access_type, trailing)

    src = _to_region(src, "r", dst_extent, _is_slice(dst))
    dst = _to_region(dst, "w", src_extent, _is_slice(src))
    if coalesced_width is not None:
        return tir.call_intrin(
            "handle", tir.op.Op.get("tl.copy"), src, dst, coalesced_width
        )
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
