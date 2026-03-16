# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""HIVM dialect transformation passes (decomposed from optimize-hivm-pipeline)."""

from tilelang.tladapter.utils import pass_fn

F = "func.func"

# ── entry ──────────────────────────────────────────────────────────────
init_entry_kernel = pass_fn("hivm-init-entry-kernel", op=F)

# ── pre-bufferization optimisation ─────────────────────────────────────
normalize_matmul = pass_fn("hivm-normalize-matmul")
inline_fixpipe = pass_fn("hivm-inline-fixpipe")
tile_batchmm_into_loop = pass_fn("hivm-tile-batchmm-into-loop", op=F)
insert_load_store_for_mix_cv = pass_fn("hivm-insert-load-store-for-mix-cv", op=F)
insert_nz2nd_for_debug = pass_fn("hivm-insert-nz2nd-for-debug")
insert_workspace_for_mix_cv = pass_fn("insert-workspace-for-mix-cv")
bind_workspace_arg = pass_fn("hivm-bind-workspace-arg", op=F)
infer_func_core_type = pass_fn("hivm-infer-func-core-type")
auto_blockify_parallel_loop = pass_fn("auto-blockify-parallel-loop")
mark_multi_buffer = pass_fn("hivm-mark-multi-buffer", op=F)
inline_otf_broadcast = pass_fn("hivm-inline-otf-broadcast", op=F)
cv_pipelining = pass_fn("cv-pipelining", op=F)
tile_cube_vector_loop = pass_fn("tile-cube-vector-loop")
plan_memory = pass_fn("hivm-plan-memory", op=F)
clone_tensor_empty = pass_fn("hivm-clone-tensor-empty", op=F)
hivm_inline_otf_load_store = pass_fn("hivm-inline-otf-load-store", op=F)

# ── cross-core sync ───────────────────────────────────────────────────
mark_real_core_type = pass_fn("hivm-mark-real-core-type")
inject_block_sync = pass_fn("hivm-inject-block-sync", op=F)

# ── host function insertion ────────────────────────────────────────────
insert_infer_workspace_size_func = pass_fn("hivm-insert-infer-workspace-size-func", op=F)
insert_infer_task_type_func = pass_fn("hivm-insert-infer-task-type-func")

# ── mix kernel ─────────────────────────────────────────────────────────
split_mix_kernel = pass_fn("hivm-split-mix-kernel")
tile_and_bind_sub_block = pass_fn("hivm-bind-sub-block")

# ── bufferisation ──────────────────────────────────────────────────────
opt_single_point = pass_fn("hivm-opt-single-point", op=F)

# ── sync ───────────────────────────────────────────────────────────────
graph_sync_solver = pass_fn("hivm-graph-sync-solver", op=F)
inject_sync = pass_fn("hivm-inject-sync", op=F)

# ── post-bufferisation ────────────────────────────────────────────────
lift_zero_rank = pass_fn("hivm-lift-zero-rank", op=F)
hivm_map_forall_to_blocks = pass_fn("hivm-map-forall-to-blocks", op=F)
hivm_decompose_op = pass_fn("hivm-decompose-op", op=F)
sync_block_hoisting = pass_fn("hivm-sync-block-hoisting", op=F)
bind_sync_block_lock_arg = pass_fn("hivm-bind-sync-block-lock-arg", op=F)
insert_infer_sync_block_lock_num_and_init_func = pass_fn(
    "hivm-insert-infer-sync-block-lock-num-and-init-func", op=F)
sync_block_lock_lowering = pass_fn("hivm-lower-create-sync-block-lock", op=F)
non_contiguous_reshape_to_copy = pass_fn("convert-non-contiguous-reshape-to-copy")
infer_hivm_mem_scope = pass_fn("hivm-infer-mem-scope")
hivm_aggregated_decompose_op = pass_fn("hivm-aggregated-decompose-op", op=F)
hivm_recognize_deinterleave_op = pass_fn("hivm-recognize-deinterleave-op", op=F)

# ── storage alignment ─────────────────────────────────────────────────
align_alloc_size = pass_fn("hivm-align-alloc-size", op=F)
mark_stride_align = pass_fn("hivm-mark-stride-align", op=F)
enable_stride_align = pass_fn("hivm-enable-stride-align", op=F)

# ── buffer sizing ─────────────────────────────────────────────────────
auto_infer_buffer_size = pass_fn("hivm-auto-infer-buffer-size", op=F)
constantize_buffer_size = pass_fn("hivm-constantize-buffer-size", op=F)
set_buffer_size = pass_fn("hivm-set-buffer-size", op=F)
flatten_ops = pass_fn("hivm-flatten-ops", op=F)
reduce_rank_subview = pass_fn("hivm-reduce-rank-subview", op=F)
lift_lowest_stride = pass_fn("hivm-lift-lowest-stride", op=F)
alloc_extra_buffer = pass_fn("hivm-alloc-extra-buffer", op=F)
inline_load_copy = pass_fn("hivm-inline-load-copy", op=F)

# ── lower to loops ────────────────────────────────────────────────────
hivm_lower_to_loops = pass_fn("hivm-lower-to-loops", op=F)
add_ffts_to_sync_block_set_op = pass_fn("hivm-add-ffts-to-syncblocksetop", op=F)
enable_multi_buffer = pass_fn("hivm-enable-multi-buffer", op=F)

# ── data layout ────────────────────────────────────────────────────────
infer_hivm_data_layout = pass_fn("hivm-infer-data-layout", op=F)
