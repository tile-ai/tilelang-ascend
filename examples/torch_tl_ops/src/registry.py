"""
TileLang Ascend Operators - Operator Registry

Manages PyTorch registration for all operators.
"""

import torch
from typing import Dict
from .loader import KernelRegistry
from .ops.base import BaseOp

_kernel_registry = KernelRegistry

_lib_def = None
_lib_impl = None


def _ensure_lib_initialized():
    """Ensure PyTorch operator library is initialized"""
    global _lib_def, _lib_impl
    
    if _lib_def is None:
        _lib_def = torch.library.Library("tl_ascend_ops", "DEF")
        _lib_impl = torch.library.Library("tl_ascend_ops", "IMPL")


def _register_op(op: BaseOp):
    """Register a single operator to PyTorch"""
    _ensure_lib_initialized()
    
    _lib_def.define(op.signature)
    
    def _impl_wrapper(*args, **kwargs):
        return op.impl(*args, **kwargs, registry=_kernel_registry)
    
    _lib_impl.impl(op.name, _impl_wrapper, "PrivateUse1")


def register_all_ops() -> Dict[str, BaseOp]:
    """Register all operators"""
    from .ops.flash_attention import flash_attention_op
    from .ops.gemm import gemm_op
    
    ops = {
        "flash_attention": flash_attention_op,
        "gemm": gemm_op,
    }
    
    for name, op in ops.items():
        _register_op(op)
    
    print(f"✓ Registered {len(ops)} operators: {', '.join(ops.keys())}")
    return ops


def get_kernel_registry():
    """Get kernel registry"""
    return _kernel_registry
