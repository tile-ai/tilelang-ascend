# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""BishengIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

adapt_triton_kernel = pass_fn("adapt-triton-kernel")
canonicalize_module = pass_fn("canonicalize-module")
append_device_spec = pass_fn("hacc-append-device-spec")
extended_canonicalizer = pass_fn("canonicalize-ext")
arith_to_affine = pass_fn("convert-arith-to-affine")
