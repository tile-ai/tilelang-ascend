# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

from typing import Literal
from pathlib import Path
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from .kernel_cache import KernelCache
from tilelang.env import TILELANG_CLEAR_CACHE

# Create singleton instance of KernelCache
_kernel_cache_instance = KernelCache()


def cached(
    func: PrimFunc = None,
    out_idx: list[int] = None,
    workspace_idx: list[int] = None,
    *args,
    target: str | Target = "auto",
    target_host: str | Target = None,
    platform: str = "auto",
    execution_backend: Literal["dlpack", "ctypes", "cython"] | None = "cython",
    verbose: bool | None = False,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kerne(ls (using KernelCache class).
    """

    return _kernel_cache_instance.cached(
        func,
        out_idx,
        workspace_idx,
        *args,
        target=target,
        target_host=target_host,
        platform=platform,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def get_cache_dir() -> Path:
    """
    Gets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.get_cache_dir()
        PosixPath('/Users/username/.tilelang/cache')
    """
    return _kernel_cache_instance.get_cache_dir()


def set_cache_dir(cache_dir: str):
    """
    Sets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.set_cache_dir("/path/to/cache")
    """
    _kernel_cache_instance.set_cache_dir(cache_dir)


def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()


if TILELANG_CLEAR_CACHE.lower() in ("1", "true", "yes", "on"):
    clear_cache()
