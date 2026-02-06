# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""BishengIR dialect transformation passes."""

from tilelang.tilelangir.utils import pass_fn

adapt_triton_kernel = pass_fn("adapt-triton-kernel")
