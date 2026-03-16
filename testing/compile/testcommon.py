# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Common utilities for testing/compile. npuir_compile_to_bin compiles npuir/mlir
via tilelang JIT. ascend_mode sets TILELANG_ASCEND_MODE for the duration of a test.
"""
import contextlib
import os

from tilelang.jit.jit_npu import compiler_npu


def npuir_compile_to_bin(npuir_str: str) -> bytes:
    """
    Compile npuir/mlir string to binary via _npuir_to_bin_enable_npu_compile.
    Success iff no exception and return value is non-empty.
    """
    compiler = compiler_npu()
    compiler.mlir_content = npuir_str
    compiler.metadata = {}
    return compiler._npuir_to_bin_enable_npu_compile()


@contextlib.contextmanager
def ascend_mode(mode: str):
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev
