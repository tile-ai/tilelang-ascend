# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR transforms: transformation passes by dialect.

- mlir: canonicalize, cse, sccp
- tilelangir: cv_split, vectorize
- bishengir: adapt_triton_kernel
"""

from . import mlir, tilelangir, bishengir

__all__ = ["mlir", "tilelangir", "bishengir"]
