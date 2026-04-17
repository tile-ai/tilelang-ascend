# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""MLIR builtin/built-in transformation passes."""

from tilelang.tladapter.utils import pass_fn

arith_to_affine = pass_fn("convert-arith-to-affine")
canonicalize = pass_fn("canonicalize")
canonicalize_ext_func = pass_fn("canonicalize-ext", "func.func")
cse = pass_fn("cse")
inline_scope = pass_fn("inline-scope")
map_for_to_forall = pass_fn("map-for-to-forall", "func.func")
memref_dse = pass_fn("memref-dse", "func.func")
scf_canonicalize_iter_arg = pass_fn("scf-canonicalize-iter-arg", "func.func")
scf_for_loop_canonicalization = pass_fn("scf-for-loop-canonicalization")
sccp = pass_fn("sccp")
