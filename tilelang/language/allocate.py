# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Memory allocation utilities for Tile-AI programs.

This module provides a set of functions for allocating different types of memory buffers
in Tile-AI programs. It wraps TVM's buffer allocation functionality with convenient
interfaces for different memory scopes.

Available allocation functions:
    - alloc_shared: Allocates shared memory buffers for inter-thread communication
    - alloc_fragment: Allocates fragment memory buffers for specialized operations
    - alloc_var: Allocates single-element variable buffers

Each function takes shape and dtype parameters and returns a TVM buffer object
with the appropriate memory scope.
"""

from __future__ import annotations

import tvm
from tvm.script import tir as T
from tvm.tir import PrimExpr
from tvm.script.parser.tir import block_attr
from typing import overload

# from .dtypes import dtype as tl_dtype
from tvm.tir.buffer import Buffer
from tvm.tir.expr import FloatImm, IntImm


def alloc_shared(shape, dtype, scope="shared.dyn"):
    """Allocate a shared memory buffer for inter-thread communication.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "shared.dyn"

    Returns:
        T.Buffer: A TVM buffer object allocated in shared memory
    """
    if dtype == "bool":
        # lei: This is a hack to handle bool type.
        # Because tilelang's merge smem pass cannot merge bool type currently.
        scope = "shared"
    return T.alloc_buffer(shape, dtype, scope=scope)


def alloc_fragment(shape, dtype, scope="local.fragment"):
    """Allocate a fragment memory buffer for specialized operations.

    Args:
        shape (tuple): The shape of the buffer to allocate
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        scope (str, optional): The memory scope. Defaults to "local.fragment"

    Returns:
        T.Buffer: A TVM buffer object allocated in fragment memory
    """
    return T.alloc_buffer(shape, dtype, scope=scope)


@overload
def alloc_var(dtype, init: PrimExpr | int | float, scope: str = "local.var") -> Buffer: ...


@overload
def alloc_var(dtype, scope: str = "local.var", *, init: PrimExpr | int | float | None = None) -> Buffer: ...


def alloc_var(dtype, *args, scope: str = "local.var", init: PrimExpr | int | float | None = None) -> Buffer:
    """Allocate a single-element variable buffer.

    Args:
        dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
        *args: Optional positional arguments. A single positional string is treated
            as the scope for backward compatibility. A single non-string positional
            argument (or keyword ``init``) specifies the initializer. When two
            positional arguments are provided, they are interpreted as
            ``(init, scope)``.
        scope (str, optional): The memory scope. Defaults to "local.var".
            Use as keyword argument for clarity when also providing an initializer.
        init (PrimExpr, optional): The optional initializer value. When provided,
            the generated code will initialize the variable with this value instead
            of defaulting to zero.
    Examples:
        a = T.alloc_var('int32', 1) # var with init 1
        a = T.alloc_var('int32', 'local.var') # var with local.var scope
        a = T.alloc_var('int32', 1, 'local.var') # var with init 1 and local.var scope
        a = T.alloc_var('int32', 'local.var', init=1) # var with init 1 and local.var scope
        a = T.alloc_var('int32', init=1) # var with init 1 and local.var scope
    Returns:
        T.Buffer: A TVM buffer object allocated as a single-element variable
    """
    parsed_scope = scope
    parsed_init = init
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, str) and parsed_init is None and scope == "local.var":
            parsed_scope = arg
        else:
            if parsed_init is not None:
                raise TypeError("Initializer specified multiple times in alloc_var.")
            parsed_init = arg
    elif len(args) == 2:
        if parsed_init is not None:
            raise TypeError("Initializer specified multiple times in alloc_var.")
        parsed_init, parsed_scope_arg = args
        if not isinstance(parsed_scope_arg, str):
            raise TypeError("Scope must be provided as a string in alloc_var.")
        parsed_scope = parsed_scope_arg
    elif len(args) > 2:
        raise TypeError(f"alloc_var expected at most 3 positional arguments but got {len(args) + 1}.")

    if not isinstance(parsed_scope, str):
        raise TypeError("Scope must be a string in alloc_var.")

    buffer = T.alloc_buffer([1], dtype, scope=parsed_scope)
    if parsed_init is not None:
        if isinstance(parsed_init, (int, float, IntImm, FloatImm)):
            init_const = tvm.tir.const(parsed_init, dtype)
            block_attr({"tl.local_var_init": {buffer.data: init_const}})
        else:
            T.buffer_store(buffer, parsed_init, 0)
    return buffer


"""
The following are memory scopes in Ascend.
Here is the correspondence between TIR scopes and Ascend memory scopes:
- shared.dyn -> L1
- wmma.matrix_a -> L0A
- wmma.matrix_b -> L0B
- wmma.accumulator -> L0C
- shared -> UB
"""


def alloc_L1(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="shared.l1")


def alloc_L0A(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="wmma.matrix_a")


def alloc_L0B(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="wmma.matrix_b")


def alloc_L0C(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="wmma.accumulator")


def alloc_ub(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="shared")
