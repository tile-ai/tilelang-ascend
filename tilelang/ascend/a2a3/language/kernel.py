# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2/A3 launch API ported from TileLang-Ascend's language/kernel.py.

The default NPU launch yields ``(cid, vid)``. The explicit ``threads=1|2``
developer form yields ``cid``; ``get_thread_binding()`` reads its ``vid``.
Both forms record vid as the target-neutral ``cthread`` launch dimension.
"""

from __future__ import annotations

from tvm import tirx

from .._native import require_native
from tilelang.jit.exceptions import JITNoBuilderError
from tilelang.language.kernel import (
    FrameStack,
    KernelLaunchFrame,
    get_block_binding,
    get_block_bindings,
    get_block_extent,
    get_block_extents,
    get_thread_binding,
    get_thread_bindings,
    get_thread_extent,
    get_thread_extents,
    kernel_launch_factory,
    launch_kernel,
)

__all__ = [
    "FrameStack",
    "KernelLaunchFrame",
    "Kernel",
    "get_thread_binding",
    "get_thread_bindings",
    "get_thread_extent",
    "get_thread_extents",
    "get_block_binding",
    "get_block_bindings",
    "get_block_extent",
    "get_block_extents",
]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
    is_npu: bool = False,
    pipeline: bool = False,
) -> KernelLaunchFrame:
    """Construct the TA launch, including its two NPU binding forms.

    ``is_npu=True`` takes a single core-grid extent. With no ``threads`` it
    yields ``(cid, vid)`` and uses two vector subcores. Explicit ``threads``
    accepts 1 or 2, or a one-element list/tuple, yields ``cid`` and preserves
    the ``npu_cv_ratio`` function attribute. In either case ``vid`` is a
    ``cthread`` binding, rather than a SIMT thread domain.

    The original CPU and GPU forms remain available through the shared
    neutral launch machinery. ``pipeline`` is retained as TA's reserved
    compatibility argument; the original implementation does not consume it.
    """
    if is_npu:
        kernel_launch = require_native("tl.A2A3KernelLaunch", "Kernel")
        from tilelang.language.eager.builder import Builder

        if Builder.current() is None:
            raise JITNoBuilderError("T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context. No Builder is available.")
        if len(blocks) != 1:
            raise ValueError("NPU kernel must have exactly one block dimension")
        attrs = {}
        if threads is None:
            attrs["tilelang.is_npu_kernel_frame"] = True
        else:
            if isinstance(threads, int):
                threads = [threads]
            elif isinstance(threads, (list, tuple)):
                threads = list(threads)
            else:
                raise TypeError(f"threads must be int, list or tuple, but got {type(threads)}")
            if len(threads) != 1:
                raise ValueError("NPU kernel must have exactly one thread dimension")
            if threads[0] not in (1, 2):
                raise ValueError(f"NPU kernel threads must be 1 or 2, but got {threads[0]}")
            attrs["tilelang.is_npu_kernel_frame_dev_mode"] = True
        # Keep the TA NPU early-return path; prelude is consumed only by the
        # generic launch below in the source implementation.
        return kernel_launch(blocks, threads, attrs)

    if not is_cpu and threads is None:
        threads = 128
    if is_cpu and threads is not None:
        raise ValueError("CPU kernel cannot have block size")
    annotations = {"tilelang.is_cpu_kernel_frame": True} if is_cpu else {}
    return launch_kernel(blocks, threads=threads, prelude=prelude, **annotations)
