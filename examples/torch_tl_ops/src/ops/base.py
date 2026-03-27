"""
Operator Base Class Definition

All operators need to inherit from this base class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class BaseOp(ABC):
    """Operator base class"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name, used for kernel directory and registration"""
        pass
    
    @property
    @abstractmethod
    def signature(self) -> str:
        """PyTorch operator signature, e.g. 'flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor'"""
        pass
    
    @abstractmethod
    def get_kernel(self, registry) -> Any:
        """Get kernel from registry"""
        pass
    
    @abstractmethod
    def impl(self, *args, **kwargs) -> torch.Tensor:
        """Operator implementation, calls kernel to execute computation"""
        pass
    
    @abstractmethod
    def python_api(self, *args, **kwargs) -> torch.Tensor:
        """Python API interface, includes parameter validation and default value handling"""
        pass
