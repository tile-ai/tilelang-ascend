"""
Flash Attention Operator Definition

Fixed shape: seq_len=512, dim=128
"""

import torch
from .base import BaseOp


class FlashAttentionOp(BaseOp):
    """Flash Attention operator"""
    
    _kernel = None
    
    @property
    def name(self) -> str:
        return "flash_attention"
    
    @property
    def signature(self) -> str:
        return "flash_attention(Tensor Q, Tensor K, Tensor V) -> Tensor"
    
    def get_kernel(self, registry):
        if FlashAttentionOp._kernel is None:
            FlashAttentionOp._kernel = registry.get_kernel(self.name)
        return FlashAttentionOp._kernel
    
    def impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, registry=None) -> torch.Tensor:
        kernel = self.get_kernel(registry)
        output = kernel(Q, K, V)
        return output
    
    def python_api(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention operator
        
        Args:
            Q: Query tensor [seq_len, dim], NPU tensor, float16
            K: Key tensor [seq_len, dim], NPU tensor, float16
            V: Value tensor [seq_len, dim], NPU tensor, float16
        
        Returns:
            Output tensor [seq_len, dim]
        """
        if Q.device.type != "npu":
            raise ValueError("Q must be an NPU tensor")
        if K.device.type != "npu":
            raise ValueError("K must be an NPU tensor")
        if V.device.type != "npu":
            raise ValueError("V must be an NPU tensor")
        
        seq_len, dim = Q.shape
        if K.shape != (seq_len, dim) or V.shape != (seq_len, dim):
            raise ValueError(f"Shape mismatch: Q={Q.shape}, K={K.shape}, V={V.shape}")
        
        return torch.ops.tl_ascend_ops.flash_attention(Q, K, V)


flash_attention_op = FlashAttentionOp()
