# TileLang Coverage Report

**Generated at**: 2026-04-30 14:51:04

---

## Python Coverage (tilelang/ + examples/)

### Summary

TOTAL                                                                            17909   9054    49%

### Module Details

| Name | Stmts | Miss | Cover |
|------|-------|------|-------|
| tilelang/cache/tuner_cache.py | 80 | 80 | 0% |
| tilelang/carver/roller/shape_inference/common.py | 55 | 55 | 0% |
| tilelang/contrib/rocm.py | 102 | 102 | 0% |
| tilelang/intrinsics/mfma_macro_generator.py | 204 | 204 | 0% |
| tilelang/jit/env.py | 26 | 26 | 0% |
| tilelang/language/ast/_ffi_api.py | 2 | 2 | 0% |
| tilelang/language/ast/ir.py | 544 | 544 | 0% |
| tilelang/language/fill.py | 14 | 14 | 0% |
| tilelang/language/parser/entry.py | 60 | 60 | 0% |
| tilelang/language/parser/operation.py | 87 | 87 | 0% |
| tilelang/language/parser/parser.py | 260 | 260 | 0% |
| tilelang/language/pto.py | 171 | 171 | 0% |
| tilelang/quantize/lop3.py | 58 | 58 | 0% |
| tilelang/quantize/quantization.py | 136 | 136 | 0% |
| tilelang/quantize/utils.py | 78 | 78 | 0% |
| tilelang/tools/Analyzer.py | 74 | 74 | 0% |
| tilelang/tools/plot_layout.py | 92 | 92 | 0% |
| tilelang/language/gemm.py | 60 | 55 | 8% |
| tilelang/intrinsics/mma_macro_generator.py | 425 | 380 | 11% |
| tilelang/carver/roller/policy/tensorcore.py | 249 | 222 | 11% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_h.py | 70 | 61 | 13% |
| tilelang/carver/common_schedules.py | 54 | 46 | 15% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_wy_fast.py | 47 | 40 | 15% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_o.py | 52 | 44 | 15% |
| tilelang/contrib/dlpack.py | 19 | 16 | 16% |
| tilelang/carver/arch/driver/cuda_driver.py | 69 | 58 | 16% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_solve_tril.py | 81 | 67 | 17% |
| tilelang/contrib/nvcc.py | 190 | 157 | 17% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_scaled_dot_kkt.py | 39 | 32 | 18% |
| tilelang/contrib/hipcc.py | 50 | 41 | 18% |
| tilelang/language/logical.py | 44 | 36 | 18% |
| tilelang/jit/adapter/dlpack.py | 32 | 26 | 19% |
| tilelang/engine/callback.py | 42 | 33 | 21% |
| tilelang/jit/adapter/wrapper.py | 307 | 240 | 22% |
| tilelang/primitives/gemm/gemm_mma.py | 80 | 62 | 22% |
| tilelang/language/print.py | 79 | 61 | 23% |
| tilelang/carver/arch/arch_base.py | 17 | 13 | 24% |
| tilelang/intrinsics/mfma_layout.py | 83 | 63 | 24% |
| tilelang/contrib/cc.py | 197 | 149 | 24% |
| examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_cumsum.py | 28 | 21 | 25% |
| tilelang/primitives/gemm/base.py | 133 | 98 | 26% |
| tilelang/intrinsics/utils.py | 56 | 41 | 27% |
| tilelang/carver/arch/cdna.py | 26 | 19 | 27% |
| tilelang/carver/arch/cuda.py | 80 | 58 | 28% |
| tilelang/jit/adapter/utils.py | 54 | 39 | 28% |
| tilelang/intrinsics/mma_layout.py | 79 | 57 | 28% |
| tilelang/carver/matmul_analysis.py | 525 | 378 | 28% |
| examples/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py | 262 | 188 | 28% |
| tilelang/carver/template/general_reduce.py | 57 | 40 | 30% |
| tilelang/language/reduce.py | 43 | 30 | 30% |
| examples/flash_attention/flash_attn_bhsd.py | 150 | 104 | 31% |
| tilelang/layout/fragment.py | 51 | 34 | 33% |
| tilelang/carver/template/flashattention.py | 69 | 44 | 36% |
| tilelang/utils/tensor.py | 142 | 89 | 37% |
| examples/sparse_flash_attention/example_sparse_flash_attn_mask.py | 231 | 143 | 38% |
| tilelang/carver/roller/policy/default.py | 441 | 271 | 39% |
| tilelang/jit/adapter/ctypes/adapter.py | 93 | 57 | 39% |
| examples/lightning_indexer/example_lightning_indexer_dynamic_shape.py | 154 | 94 | 39% |
| examples/normalization/rms_norm.py | 59 | 36 | 39% |
| examples/lightning_indexer/example_lightning_indexer.py | 156 | 94 | 40% |
| examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py | 227 | 136 | 40% |
| examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py | 225 | 134 | 40% |
| tilelang/transform/simplify.py | 22 | 13 | 41% |
| tilelang/carver/template/gemv.py | 51 | 30 | 41% |
| tilelang/language/builtin.py | 72 | 42 | 42% |
| tilelang/carver/template/conv.py | 59 | 33 | 44% |
| examples/gemm/example_gemm_intrinsic.py | 95 | 53 | 44% |
| tilelang/carver/utils.py | 140 | 77 | 45% |
| tilelang/engine/lower.py | 122 | 67 | 45% |
| examples/gemm/example_gemm_intrinsic_persistent.py | 90 | 49 | 46% |
| tilelang/carver/arch/cpu.py | 13 | 7 | 46% |
| examples/linear_attention_and_rnn/gdn/gdn_solve_tril.py | 75 | 38 | 49% |
| examples/consal_conv1d/causal_conv1d.py | 206 | 103 | 50% |
| tilelang/carver/analysis.py | 171 | 84 | 51% |
| examples/print/elementwise_print.py | 62 | 30 | 52% |
| examples/elementwise/elementwise_add_pipeline.py | 69 | 33 | 52% |
| examples/simple_fusion/matmul_add.py | 59 | 28 | 53% |
| examples/simple_fusion/matmul_add_infer_scope.py | 59 | 28 | 53% |
| tilelang/language/tir/op.py | 290 | 136 | 53% |
| tilelang/language/warpgroup.py | 28 | 13 | 54% |
| examples/chunk_gated_delta_rule/chunk_gated_delta_rule_varlen.py | 209 | 93 | 56% |
| tilelang/language/tir/entry.py | 32 | 14 | 56% |
| tilelang/language/ascend_tile.py | 516 | 224 | 57% |
| tilelang/carver/roller/bestfit.py | 49 | 21 | 57% |
| tilelang/carver/roller/hint.py | 161 | 68 | 58% |
| tilelang/utils/language.py | 29 | 12 | 59% |
| tilelang/carver/template/elementwise.py | 27 | 11 | 59% |
| examples/gemm/example_gemm_transpose_l1.py | 50 | 20 | 60% |
| tilelang/language/copy.py | 125 | 50 | 60% |
| tilelang/layout/swizzle.py | 5 | 2 | 60% |
| examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py | 28 | 11 | 61% |
| tilelang/language/kernel.py | 138 | 54 | 61% |
| examples/pipeline/gemm_v0_pipeline.py | 52 | 20 | 62% |
| tilelang/carver/roller/rasterization.py | 29 | 11 | 62% |
| tilelang/libinfo.py | 37 | 14 | 62% |
| examples/pad/example_broadcast_pipeline.py | 56 | 21 | 62% |
| examples/reduce/example_reduce_min_pipeline.py | 59 | 22 | 63% |
| tilelang/utils/target.py | 66 | 24 | 64% |
| examples/linear_attention_and_rnn/gdn/gdn_chunk_h.py | 69 | 25 | 64% |
| tilelang/carver/roller/policy/ascend.py | 464 | 168 | 64% |
| examples/grouped_gemm/example_grouped_gemm_fwd_2.py | 108 | 39 | 64% |
| examples/sparse_flash_attention/sfa_golden.py | 144 | 52 | 64% |
| tilelang/language/overrides/parser.py | 123 | 43 | 65% |
| tilelang/carver/roller/policy/common.py | 26 | 9 | 65% |
| examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py | 110 | 38 | 65% |
| examples/linear_attention_and_rnn/gdn/gdn_wy_fast.py | 47 | 16 | 66% |
| tilelang/language/allocate.py | 59 | 20 | 66% |
| tilelang/carver/roller/shape_inference/tir.py | 275 | 92 | 67% |
| examples/gemm/example_gemm.py | 45 | 15 | 67% |
| examples/linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py | 39 | 13 | 67% |
| tilelang/jit/adapter/cython/adapter.py | 222 | 71 | 68% |
| tilelang/carver/arch/ascend.py | 97 | 31 | 68% |
| tilelang/carver/roller/node.py | 473 | 151 | 68% |
| examples/elementwise/elementwise_add.py | 42 | 13 | 69% |
| examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py | 52 | 16 | 69% |
| tilelang/autotuner/capture.py | 39 | 12 | 69% |
| examples/gemm/example_gemm_persistent.py | 46 | 14 | 70% |
| tilelang/layout/layout.py | 33 | 10 | 70% |
| tilelang/language/ascend.py | 116 | 34 | 71% |
| tilelang/common/transform_kind.py | 14 | 4 | 71% |
| tilelang/jit/adapter/base.py | 35 | 10 | 71% |
| tilelang/jit/param.py | 14 | 4 | 71% |
| tilelang/language/proxy.py | 53 | 15 | 72% |
| tilelang/engine/param.py | 46 | 13 | 72% |
| examples/elementwise/setvalue_example.py | 44 | 12 | 73% |
| examples/grouped_gemm/example_grouped_gemm_fwd.py | 94 | 25 | 73% |
| examples/pad/example_broadcast.py | 39 | 10 | 74% |
| examples/reduce/example_reduce_min.py | 39 | 10 | 74% |
| tilelang/autotuner/param.py | 134 | 34 | 75% |
| tilelang/language/reduce_ascend.py | 216 | 53 | 75% |
| tilelang/jit/adapter/libgen.py | 72 | 16 | 78% |
| tilelang/version.py | 23 | 5 | 78% |
| tilelang/carver/template/base.py | 42 | 9 | 79% |
| tilelang/autotuner/tuner.py | 342 | 71 | 79% |
| tilelang/env.py | 145 | 30 | 79% |
| examples/autotune/example_gemm_autotune.py | 30 | 6 | 80% |
| tilelang/utils/deprecated.py | 10 | 2 | 80% |
| tilelang/language/frame.py | 73 | 14 | 81% |
| examples/gemm/example_gemm_tail_block_developer.py | 21 | 4 | 81% |
| tilelang/language/customize.py | 90 | 17 | 81% |
| tilelang/profiler/bench.py | 44 | 7 | 84% |
| tilelang/cache/kernel_cache.py | 81 | 12 | 85% |
| tilelang/jit/kernel.py | 82 | 11 | 87% |
| tilelang/language/parallel.py | 8 | 1 | 88% |
| examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h32_d512.py | 46 | 5 | 89% |
| examples/flash_attention/fa_opt/flash_attn_bhsd_ascendc.py | 41 | 4 | 90% |
| examples/cross_entropy_loss/example_cross_entro.py | 42 | 4 | 90% |
| examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h16_d128.py | 49 | 4 | 92% |
| examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py | 49 | 4 | 92% |
| tilelang/carver/template/matmul.py | 52 | 4 | 92% |
| tilelang/intrinsics/ascend_layout.py | 57 | 4 | 93% |
| examples/activation/sigmoid.py | 15 | 1 | 93% |
| examples/activation/silu.py | 15 | 1 | 93% |
| examples/activation/gelu_grad.py | 17 | 1 | 94% |
| tilelang/language/tir/ir.py | 159 | 7 | 96% |
| tilelang/engine/phase.py | 48 | 2 | 96% |
| examples/pos_embedding/rope.py | 55 | 2 | 96% |
| examples/gemv/example_gemv_c.py | 29 | 1 | 97% |
| examples/linear_attention_and_rnn/linear_attention_causal.py | 58 | 2 | 97% |
| examples/gemv/example_gemv_v.py | 30 | 1 | 97% |
| examples/pos_embedding/rope_mask.py | 63 | 2 | 97% |
| examples/pos_embedding/rms_rope_fused.py | 65 | 2 | 97% |
| examples/quant_batch_matmul/example_quant_matmul.py | 33 | 1 | 97% |
| examples/topk_selector/example_topk_selector.py | 67 | 2 | 97% |
| examples/quant_batch_matmul/example_quant_batch_matmul.py | 34 | 1 | 97% |
| examples/linear_attention_and_rnn/opt_gdn_full.py | 72 | 2 | 97% |
| examples/pos_embedding/rms_rope_fused_mask.py | 73 | 2 | 97% |
| examples/aclgraph/rms_rope_aclgraph.py | 79 | 2 | 97% |
| examples/convolution/example_convolution.py | 45 | 1 | 98% |
| examples/sort/example_merge_sort.py | 93 | 2 | 98% |
| examples/developer_mode/sparse_flash_attn_developer.py | 58 | 1 | 98% |
| examples/sparse_flash_attention/example_sparse_flash_attn.py | 58 | 1 | 98% |
| examples/pipeline/sparse_flash_attn_gqa_pipeline.py | 61 | 1 | 98% |
| examples/sparse_flash_attention/example_sparse_flash_attn_gqa.py | 61 | 1 | 98% |
| examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py | 66 | 1 | 98% |
| examples/flash_attention/paged_flash_attn_bhsd.py | 67 | 1 | 99% |
| examples/pipeline/sparse_flash_attn_gqa_pipeline_pto.py | 67 | 1 | 99% |
| examples/activation/gelu_mul.py | 18 | 0 | 100% |
| examples/activation/sigmoidv2.py | 15 | 0 | 100% |
| examples/activation/sigmoidv2_slice.py | 15 | 0 | 100% |
| examples/activation/swi_glu.py | 22 | 0 | 100% |
| examples/activation/tanh.py | 15 | 0 | 100% |
| examples/autotune/example_gemm_carver.py | 32 | 0 | 100% |
| examples/batch_gemm/batch_gemm.py | 25 | 0 | 100% |
| examples/developer_mode/flash_attn_bshd_developer.py | 30 | 0 | 100% |
| examples/developer_mode/gelu_mul_developer.py | 18 | 0 | 100% |
| examples/developer_mode/gemm_developer.py | 23 | 0 | 100% |
| examples/developer_mode/matmul_add_developer.py | 23 | 0 | 100% |
| examples/flash_attention/flash_attn_bhsd_cc_sync.py | 26 | 0 | 100% |
| examples/gemm/example_gemm_infer_scope.py | 23 | 0 | 100% |
| examples/gemm/example_gemm_pto_developer.py | 23 | 0 | 100% |
| examples/linear_attention_and_rnn/gdn_full.py | 69 | 0 | 100% |
| examples/linear_attention_and_rnn/linear_attention_normalize.py | 38 | 0 | 100% |
| examples/normalization/layer_norm.py | 18 | 0 | 100% |
| examples/pipeline/flash_attn_bshd_pipeline.py | 30 | 0 | 100% |
| examples/pipeline/matmul_add_pipeline.py | 24 | 0 | 100% |
| examples/pos_embedding/rms_norm.py | 31 | 0 | 100% |
| examples/reduce/example_col_reduce_max_slice_buffer.py | 18 | 0 | 100% |
| examples/reduce/example_row_reduce_max_slice_buffer.py | 18 | 0 | 100% |
| examples/softmax/example_online_softmax.py | 18 | 0 | 100% |
| tilelang/_ffi_api.py | 2 | 0 | 100% |
| tilelang/language/memscope.py | 5 | 0 | 100% |
| tilelang/language/persistent.py | 5 | 0 | 100% |
| tilelang/language/pipeline.py | 17 | 0 | 100% |
| tilelang/transform/_ffi_api.py | 2 | 0 | 100% |
| tilelang/transform/pass_config.py | 41 | 0 | 100% |

---

## C++ Coverage (src/)

### Summary

  lines......: 41.2% (8030 of 19498 lines)
  functions..: 44.6% (869 of 1947 functions)

### File Details

| File | Lines | Functions |
|------|-------|----------|
| ir.cc | 59.5% | 57.1% |
| layout/layout.cc | 18.7% | 21.8% |
| layout/layout.h | 31.2% | 37.9% |
| layout/swizzle.h | 50.0% | 66.7% |
| op/ascend.cc | 92.6% | 98.1% |
| op/op.cc | 63.0% | 50.0% |
| target/codegen_ascend.cc | 67.9% | 70.0% |
| target/codegen_ascend_pto.cc | 44.8% | 56.4% |
| target/utils.cc | 17.6% | 36.4% |
| transform/allocate_tmp_buffer.cc | 59.6% | 92.9% |
| transform/ascend_collect_buffer_shape.cc | 94.7% | 92.9% |
| transform/ascend_combinecv.cc | 88.1% | 94.3% |
| transform/ascend_infer_buffer_scope.cc | 81.7% | 90.9% |
| transform/ascend_lower_opaque_block.cc | 51.8% | 69.2% |
| transform/ascend_lower_parallel_to_vector.cc | 58.6% | 84.1% |
| transform/ascend_memory_planning.cc | 89.6% | 91.7% |
| transform/ascend_pto_save_buffer_shape.cc | 61.1% | 71.4% |
| transform/ascend_storage_rewrite.cc | 62.5% | 84.6% |
| transform/ascend_sync_insert.cc | 88.7% | 98.4% |
| transform/common/loop_fusion_utils.h | 34.8% | 85.7% |
| transform/config_index_bitwidth.cc | 19.5% | 22.2% |
| transform/cross_core_pipeline.cc | 93.4% | 98.1% |
| transform/flatten_buffer.cc | 58.4% | 73.1% |
| transform/inject_pipeline.cc | 68.2% | 81.6% |
| transform/layout_inference.cc | 46.4% | 76.9% |
| transform/legalize_safe_memory_access.cc | 49.7% | 83.3% |
| transform/lower_tile_op.cc | 53.1% | 70.0% |
| transform/pipeline_planning.cc | 69.9% | 90.9% |

---

**Note**: Coverage based on executed tests. Files with 0% were not executed.
