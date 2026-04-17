# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""BishengIR dialect transformation passes."""

from tilelang.tladapter.utils import pass_fn

adapt_triton_kernel = pass_fn("adapt-triton-kernel")
append_device_spec = pass_fn("hacc-append-device-spec")
annotation_lowering = pass_fn("annotation-lowering")
auto_infer_buffer_size = pass_fn("hivm-auto-infer-buffer-size", "func.func")
bind_workspace_arg = pass_fn("hivm-bind-workspace-arg", "func.func")
bind_sync_block_lock_arg = pass_fn("hivm-bind-sync-block-lock-arg", "func.func")
canonicalize_module = pass_fn("canonicalize-module")
canonicalize_ext = pass_fn("canonicalize-ext")
constantize_buffer_size = pass_fn("hivm-constantize-buffer-size", "func.func")
convert_hfusion_to_hivm = pass_fn("convert-hfusion-to-hivm")
convert_non_contiguous_reshape_to_copy = pass_fn(
    "convert-non-contiguous-reshape-to-copy"
)
convert_tensor_to_hivm = pass_fn("convert-tensor-to-hivm")
convert_to_hivm_op = pass_fn("convert-to-hivm-op")
convert_hivm_to_std = pass_fn("convert-hivm-to-std")
enable_hivmc_compatible_print = pass_fn("enable-hivmc-compatible-print")
enable_multi_buffer = pass_fn("hivm-enable-multi-buffer", "func.func")
enable_stride_align = pass_fn("hivm-enable-stride-align", "func.func")
flatten_ops = pass_fn("hivm-flatten-ops", "func.func")
fold_alloc_reshape = pass_fn("fold-alloc-reshape", "func.func")
aggregated_decompose_op = pass_fn("hivm-aggregated-decompose-op", "func.func")
graph_sync_solver = pass_fn("hivm-graph-sync-solver", "func.func")
infer_data_layout = pass_fn("hivm-infer-data-layout", "func.func")
infer_mem_scope = pass_fn("hivm-infer-mem-scope")
infer_workspace_size_func = pass_fn(
    "hivm-insert-infer-workspace-size-func", "func.func"
)
inline_load_copy = pass_fn("hivm-inline-load-copy", "func.func")
init_entry_kernel = pass_fn("hivm-init-entry-kernel", "func.func")
insert_infer_sync_block_lock_num_and_init_func = pass_fn(
    "hivm-insert-infer-sync-block-lock-num-and-init-func", "func.func"
)
insert_init_and_finish_for_debug = pass_fn(
    "hivm-insert-init-and-finish-for-debug", "func.func"
)
lift_lowest_stride = pass_fn("hivm-lift-lowest-stride", "func.func")
lift_zero_rank = pass_fn("hivm-lift-zero-rank", "func.func")
lower_memref_ext = pass_fn("lower-memref-ext")
map_forall_to_blocks = pass_fn("hivm-map-forall-to-blocks", "func.func")
mark_disable_load = pass_fn("hivm-mark-disable-load", "func.func")
mark_multi_buffer = pass_fn("hivm-mark-multi-buffer", "func.func")
mark_stride_align = pass_fn("hivm-mark-stride-align", "func.func")
opt_single_point = pass_fn("hivm-opt-single-point", "func.func")
plan_memory = pass_fn("hivm-plan-memory", "func.func")
add_ffts_to_syncblocksetop = pass_fn("hivm-add-ffts-to-syncblocksetop", "func.func")
align_alloc_size = pass_fn("hivm-align-alloc-size", "func.func")
alloc_extra_buffer = pass_fn("hivm-alloc-extra-buffer", "func.func")
decompose_op = pass_fn("hivm-decompose-op", "func.func")
lower_to_loops = pass_fn("hivm-lower-to-loops", "func.func")
recognize_deinterleave_op = pass_fn("hivm-recognize-deinterleave-op", "func.func")
reduce_rank_subview = pass_fn("hivm-reduce-rank-subview", "func.func")
set_buffer_size = pass_fn("hivm-set-buffer-size", "func.func")
sync_block_lock_lowering = pass_fn(
    "hivm-lower-create-sync-block-lock", "func.func"
)
sync_block_hoisting = pass_fn("hivm-sync-block-hoisting", "func.func")
triton_global_kernel_args_to_hivm_op = pass_fn(
    "triton-global-kernel-args-to-hivm-op", "func.func"
)
