"""
TileLang Ascend Operators Package

Provides PyTorch operator interfaces with offline installation support.

Dependencies:
- torch > 2.6.0
- torch_npu

Usage:
    import tl_ascend_ops
    
    # Method 1: Call via package
    output = tl_ascend_ops.flash_attention(q, k, v)
    c = tl_ascend_ops.gemm(a, b)
    
    # Method 2: Call via torch_npu
    import torch_npu
    output = torch_npu.flash_attention(q, k, v)
    c = torch_npu.gemm(a, b)
    
    # Method 3: Call via torch.ops
    output = torch.ops.tl_ascend_ops.flash_attention(q, k, v)
    torch.ops.tl_ascend_ops.gemm(a, b, c)
"""

from .loader import KernelRegistry
from .registry import register_all_ops, get_kernel_registry

__version__ = "0.1.0"

# Register all operators
_registered_ops = register_all_ops()

# Import operator Python APIs
from .ops.flash_attention import flash_attention_op
from .ops.gemm import gemm_op

flash_attention = flash_attention_op.python_api
gemm = gemm_op.python_api

# Inject into torch_npu module
def _inject_to_torch_npu():
    """Inject operator interfaces into torch_npu module"""
    try:
        import torch_npu
        
        # Dynamically inject all registered operators
        injected_ops = []
        for op_name, op in _registered_ops.items():
            op_func = getattr(op, 'python_api', None)
            if op_func is not None:
                setattr(torch_npu, op_name, op_func)
                injected_ops.append(op_name)
        
        torch_npu.KernelRegistry = KernelRegistry
        print(f"✓ Injected operators into torch_npu: {', '.join(injected_ops)}")
    except ImportError:
        print(f"⚠ torch_npu not installed, skipping injection")

_inject_to_torch_npu()

__all__ = [
    "flash_attention",
    "gemm",
    "KernelRegistry",
    "flash_attention_op",
    "gemm_op",
    "register_all_ops",
    "get_kernel_registry",
]

print(f"✓ tl_ascend_ops loaded successfully")
