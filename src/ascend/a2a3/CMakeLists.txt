# A2/A3 compiler sources are built with the existing USE_ASCEND gate. They
# transform IR or emit source text and do not include CANN headers or link CANN
# libraries. The parent Ascend CMake file owns the shared runtime/stub choice.
# catlass, shmem, and pto-isa headers belong to the later Bisheng JIT compile,
# so their availability must not gate this native, source-only build.
list(APPEND TILE_LANG_SRCS
  src/ascend/a2a3/ir.cc
  src/ascend/a2a3/codegen/codegen_ascend.cc
  src/ascend/a2a3/codegen/rt_mod_ascend.cc
  src/ascend/a2a3/codegen/codegen_ascend_pto.cc
  src/ascend/a2a3/codegen/rt_mod_ascend_pto.cc
  src/ascend/a2a3/codegen/constant.cc
  src/ascend/a2a3/codegen/utils.cc
  src/ascend/a2a3/target/memory_info.cc
  src/ascend/a2a3/layout/ascend_layout.cc
  src/ascend/a2a3/layout/utils.cc
  src/ascend/a2a3/op/operator.cc
  src/ascend/a2a3/op/parallel.cc
  src/ascend/a2a3/op/ascend.cc
  src/ascend/a2a3/op/builtin.cc
  src/ascend/a2a3/op/elem.cc
  src/ascend/a2a3/op/generic.cc
  src/ascend/a2a3/op/gemm.cc
  src/ascend/a2a3/op/reduce.cc
  src/ascend/a2a3/transform/allocate_tmp_buffer.cc
  src/ascend/a2a3/transform/ascend_collect_buffer_shape.cc
  src/ascend/a2a3/transform/ascend_host.cc
  src/ascend/a2a3/transform/ascend_infer_buffer_scope.cc
  src/ascend/a2a3/transform/ascend_storage_rewrite.cc
  src/ascend/a2a3/transform/ascend_memory_planning.cc
  src/ascend/a2a3/transform/ascend_lower_opaque_block.cc
  src/ascend/a2a3/transform/ascend_lower_parallel_to_vector.cc
  src/ascend/a2a3/transform/ascend_pto_save_buffer_shape.cc
  src/ascend/a2a3/transform/ascend_tail_mask_propagation.cc
  src/ascend/a2a3/transform/ascend_vector_mask.cc
  src/ascend/a2a3/transform/ascend_vid_reduction.cc
  src/ascend/a2a3/transform/ascend_workspace_reduction.cc
  src/ascend/a2a3/transform/layout_inference.cc
  src/ascend/a2a3/transform/lower_tile_op.cc
  src/ascend/a2a3/transform/legalize_safe_memory_access.cc
  src/ascend/a2a3/transform/flatten_buffer.cc
  src/ascend/a2a3/transform/config_index_bitwidth.cc
  src/ascend/a2a3/transform/legalize_vectorized_loop.cc
  src/ascend/a2a3/transform/loop_partition.cc
  src/ascend/a2a3/transform/loop_vectorize.cc
  src/ascend/a2a3/transform/vectorize_loop.cc
  src/ascend/a2a3/transform/ascend_combinecv.cc
  src/ascend/a2a3/transform/ascend_remove_no_op.cc
  src/ascend/a2a3/transform/ascend_simplify.cc
  src/ascend/a2a3/transform/cross_core_pipeline.cc
  src/ascend/a2a3/transform/pipeline_planning.cc
  src/ascend/a2a3/transform/ascend_sync_insert.cc
  src/ascend/a2a3/transform/inject_pipeline.cc
  src/ascend/a2a3/transform/ascend_sync_insert_vs.cc
  src/ascend/a2a3/transform/annotate_device_regions.cc
  src/ascend/a2a3/transform/ascend_vector_instruction_selection.cc
  src/ascend/a2a3/transform/ascend_vector_mask_legalize.cc
)
