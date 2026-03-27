"""
Dynamic Shape GEMM Operator Definition

Supports arbitrary M, N, K dimensions for matrix multiplication
"""

from typing import Optional
import torch
from .base import BaseOp


class GemmOp(BaseOp):
    """Dynamic shape GEMM operator"""
    
    _kernel = None
    
    @property
    def name(self) -> str:
        return "gemm"
    
    @property
    def signature(self) -> str:
        return "gemm(Tensor A, Tensor B, Tensor C) -> Tensor"
    
    def get_kernel(self, registry):
        if GemmOp._kernel is None:
            GemmOp._kernel = registry.get_kernel(self.name)
        return GemmOp._kernel
    
    def impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, registry=None) -> torch.Tensor:
        kernel = self.get_kernel(registry)
        kernel(A, B, C)
        return C
    
    def python_api(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dynamic shape GEMM operator (C = A @ B)
        
        Args:
            A: Input matrix [M, K], NPU tensor, float16
            B: Input matrix [K, N], NPU tensor, float16
            C: Output matrix [M, N], NPU tensor, float16 (optional, auto-allocated if not provided)
        
        Returns:
            Output matrix [M, N], float16
        """
        if A.device.type != "npu":
            raise ValueError("A must be an NPU tensor")
        if B.device.type != "npu":
            raise ValueError("B must be an NPU tensor")
        
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Matrix dimension mismatch: A.shape[1]={K} != B.shape[0]={K2}")
        
        if C is None:
            C = torch.empty(M, N, dtype=torch.float16, device=A.device)
        elif C.device.type != "npu":
            raise ValueError("C must be an NPU tensor")
        elif C.shape != (M, N):
            raise ValueError(f"C shape mismatch: expected ({M}, {N}), got {C.shape}")
        
        return torch.ops.tl_ascend_ops.gemm(A, B, C)


gemm_op = GemmOp()
