# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TA's coordinate transform, separate from the shared swizzle annotation."""

from tvm import tirx as tir
from tvm.tirx import call_intrin

__all__ = ["npu_use_swizzle", "use_swizzle"]


def npu_use_swizzle(cid, m, n, k, block_m, block_n, off=1, dir=0, in_loop=False):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    # device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    assert m % block_m == 0 and n % block_n == 0, "m and n must be divisible by block_m and block_n"
    return call_intrin(
        "int32",
        tir.op.Op.get("tl.ascend_use_swizzle"),
        f"thread_block_swizzle<{m}, {n}, {k}, {block_m}, {block_n}, {off}, {dir}>",
        cid,
        m // block_m * n // block_n,
    )


# let npu_use_swizzle is aliased to use_swizzle
def use_swizzle(cid, m, n, k, block_m, block_n, off=1, dir=0, in_loop=False):
    """Alias for npu_use_swizzle with proper signature for function hints."""
    return npu_use_swizzle(cid, m, n, k, block_m, block_n, off, dir, in_loop)
