# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""MLIR builtin/built-in transformation passes."""

from tilelang.tladapter.utils import pass_fn

F = "func.func"

canonicalize = pass_fn("canonicalize")
cse = pass_fn("cse")
sccp = pass_fn("sccp")
loop_invariant_code_motion = pass_fn("loop-invariant-code-motion")
loop_invariant_subset_hoisting = pass_fn("loop-invariant-subset-hoisting")
scf_for_loop_canonicalization = pass_fn("scf-for-loop-canonicalization")
scf_canonicalize_iter_arg = pass_fn("scf-canonicalize-iter-arg", op=F)
scf_remove_redundant_loop_init = pass_fn("scf-remove-redundant-loop-init")
map_for_to_forall = pass_fn("map-for-to-forall", op=F)

# bufferization
one_shot_bufferize = pass_fn("one-shot-bufferize")
drop_equivalent_buffer_results = pass_fn("drop-equivalent-buffer-results")

# tensor
propagate_reshape = pass_fn("propagate-reshape", op=F)
fold_tensor_empty = pass_fn("fold-tensor-empty", op=F)
optimize_dps_op_with_yielded_insert_slice = pass_fn(
    "optimize-dps-op-with-yielded-insert-slice", op=F)

# memref
memref_dse = pass_fn("memref-dse", op=F)
fold_alloc_reshape = pass_fn("fold-alloc-reshape", op=F)
lower_memref_ext = pass_fn("lower-memref-ext")

# scope
inline_scope = pass_fn("inline-scope")
