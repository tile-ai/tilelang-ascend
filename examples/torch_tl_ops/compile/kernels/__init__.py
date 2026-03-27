"""
TileLang Kernel Definitions Directory

Each kernel is defined in a separate file for easy maintenance and extension.
"""

from .flash_attention import compile_flash_attention_kernel
from .gemm import compile_gemm_kernel

KERNEL_REGISTRY = {
    "flash_attention": compile_flash_attention_kernel,
    "gemm": compile_gemm_kernel,
}

__all__ = [
    "compile_flash_attention_kernel",
    "compile_gemm_kernel",
    "KERNEL_REGISTRY",
]
