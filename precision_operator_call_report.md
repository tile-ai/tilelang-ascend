# Precision Operator Call Report

统计文件数：153

说明：CallMode 表示入口类型；CallerCount 为静态文本调用者数量。

## examples\aclgraph（1 个）

- rms_rope_aclgraph.py：主入口直接运行；潜在调用者 0 个

## examples\activation（10 个）

- gelu_grad.py：可导入函数调用；潜在调用者 0 个
- gelu_mul.py：可导入函数调用；潜在调用者 1 个
- sigmoid.py：可导入函数调用；潜在调用者 15 个
- sigmoidv2.py：可导入函数调用；潜在调用者 1 个
- sigmoidv2_slice.py：可导入函数调用；潜在调用者 0 个
- silu.py：可导入函数调用；潜在调用者 7 个
- swi_glu.py：可导入函数调用；潜在调用者 2 个
- swi_glu_grad.py：可导入函数调用；潜在调用者 0 个
- swi_glu_v2.py：可导入函数调用；潜在调用者 0 个
- tanh.py：可导入函数调用；潜在调用者 2 个

## examples\attention_sink\example_gqa_sink_bwd_bhsd（2 个）

- example_gqa_sink_bwd_bhsd.py：主入口直接运行；潜在调用者 1 个
- test_gqa_sink_bwd_bhsd.py：主入口和测试入口；潜在调用者 0 个

## examples\attention_sink\example_gqa_sink_fwd_varlen（2 个）

- example_gqa_sink_fwd_varlen.py：主入口直接运行；潜在调用者 1 个
- test_gqa_sink_fwd_varlen.py：主入口和测试入口；潜在调用者 0 个

## examples\autotune（2 个）

- example_gemm_autotune.py：可导入函数调用；潜在调用者 0 个
- example_gemm_carver.py：可导入函数调用；潜在调用者 0 个

## examples\batch_gemm（2 个）

- batch_gemm.py：主入口直接运行；潜在调用者 1 个
- test_batch_gemm.py：测试入口运行；潜在调用者 0 个

## examples\cann-bench\cummin（1 个）

- example_cummin.py：主入口和测试入口；潜在调用者 0 个

## examples\cann-bench\gather（1 个）

- gather.py：主入口直接运行；潜在调用者 26 个

## examples\cann-bench\group_norm（1 个）

- example_group_norm.py：主入口直接运行；潜在调用者 0 个

## examples\cann-bench\transpose（2 个）

- test_transpose.py：主入口和测试入口；潜在调用者 1 个
- transpose.py：主入口直接运行；潜在调用者 59 个

## examples\causal_conv1d（3 个）

- causal_conv1d.py：主入口直接运行；潜在调用者 2 个
- causal_conv1d_decode.py：主入口直接运行；潜在调用者 0 个
- causal_conv1d_pto.py：主入口直接运行；潜在调用者 0 个

## examples\compile_flags（1 个）

- compile_flags_example.py：可导入函数调用；潜在调用者 0 个

## examples\cross_entropy_loss（1 个）

- example_cross_entro.py：主入口直接运行；潜在调用者 0 个

## examples\deepseek_v4（9 个）

- act_quant.py：主入口直接运行；潜在调用者 1 个
- hc_split_sinkhorn.py：主入口直接运行；潜在调用者 1 个
- int8_gemm.py：主入口直接运行；潜在调用者 1 个
- lightning_indexer.py：主入口直接运行；潜在调用者 0 个
- sparse_attention.py：主入口直接运行；潜在调用者 16 个
- test_act_quant.py：测试入口运行；潜在调用者 0 个
- test_hc_split_sinkhorn.py：测试入口运行；潜在调用者 0 个
- test_int8_gemm.py：测试入口运行；潜在调用者 0 个
- test_sparse_attention.py：测试入口运行；潜在调用者 0 个

## examples\deepseek_v4\sparse_flash_mla（1 个）

- sparse_flash_mla_golden.py：可导入函数调用；潜在调用者 1 个

## examples\developer_mode（7 个）

- flash_attn_bshd_developer.py：可导入函数调用；潜在调用者 1 个
- gelu_mul_developer.py：可导入函数调用；潜在调用者 0 个
- gemm_developer.py：可导入函数调用；潜在调用者 0 个
- matmul_add_developer.py：可导入函数调用；潜在调用者 0 个
- matmul_add_developer_vec_cast.py：可导入函数调用；潜在调用者 0 个
- sparse_flash_attn_developer.py：可导入函数调用；潜在调用者 0 个
- sparse_flash_attn_developer_vid_reduce.py：可导入函数调用；潜在调用者 0 个

## examples\dispatch_combine（1 个）

- dispatch_combine_shmem.py：主入口直接运行；潜在调用者 0 个

## examples\elementwise（2 个）

- elementwise_add.py：可导入函数调用；潜在调用者 0 个
- elementwise_add_pipeline.py：可导入函数调用；潜在调用者 0 个

## examples\exception_dump_test（1 个）

- example_exception_dump.py：可导入函数调用；潜在调用者 0 个

## examples\flash_attention（4 个）

- flash_attn_bhsd.py：主入口直接运行；潜在调用者 6 个
- flash_attn_bhsd_cc_sync.py：可导入函数调用；潜在调用者 1 个
- paged_flash_attn_bhsd.py：主入口直接运行；潜在调用者 0 个
- test_flash_attn_bhsd.py：主入口和测试入口；潜在调用者 0 个

## examples\flash_attention\fa_opt（4 个）

- flash_attn_bhsd_ascendc.py：主入口直接运行；潜在调用者 1 个
- flash_attn_bhsd_auto_pipeline_h16_d128.py：主入口直接运行；潜在调用者 0 个
- flash_attn_bhsd_auto_pipeline_h32_d512.py：主入口直接运行；潜在调用者 0 个
- flash_attn_bhsd_expert_h16_d128.py：主入口直接运行；潜在调用者 2 个

## examples\fused_sigmoid_gating_delta_rule（1 个）

- fused_sigmoid_gating_delta_rule_varlen.py：主入口直接运行；潜在调用者 0 个

## examples\fusedmoe（2 个）

- example_fusedmoe.py：主入口直接运行；潜在调用者 1 个
- test_fusedmoe.py：主入口和测试入口；潜在调用者 0 个

## examples\gemm（9 个）

- example_gemm.py：可导入函数调用；潜在调用者 0 个
- example_gemm_fp8_pto.py：可导入函数调用；潜在调用者 0 个
- example_gemm_infer_scope.py：可导入函数调用；潜在调用者 0 个
- example_gemm_intrinsic.py：可导入函数调用；潜在调用者 0 个
- example_gemm_intrinsic_persistent.py：主入口直接运行；潜在调用者 0 个
- example_gemm_persistent.py：主入口直接运行；潜在调用者 0 个
- example_gemm_pto_developer.py：可导入函数调用；潜在调用者 0 个
- example_gemm_tail_block_developer.py：可导入函数调用；潜在调用者 0 个
- example_gemm_transpose_l1.py：可导入函数调用；潜在调用者 0 个

## examples\gemm_aot（1 个）

- test_example_gemm.py：可导入函数调用；潜在调用者 0 个

## examples\gemv（2 个）

- example_gemv_c.py：主入口直接运行；潜在调用者 0 个
- example_gemv_v.py：主入口直接运行；潜在调用者 0 个

## examples\generative_recommendation（1 个）

- mtgr_ragged_segment_attention.py：主入口直接运行；潜在调用者 0 个

## examples\gqa_fwd_varlen（3 个）

- gqa_fwd_varlen.py：主入口直接运行；潜在调用者 5 个
- perf_gqa_fwd_varlen.py：主入口直接运行；潜在调用者 1 个
- test_gqa_fwd_varlen.py：主入口和测试入口；潜在调用者 1 个

## examples\HISA（3 个）

- block_sparse_mqa_attn_expert_test.py：主入口和测试入口；潜在调用者 1 个
- block_sparse_mqa_attn_expert_test_for_a5.py：主入口和测试入口；潜在调用者 0 个
- paged_block_sparse_mqa_attn_expert.py：主入口和测试入口；潜在调用者 0 个

## examples\linear_attention_and_rnn（4 个）

- gdn_full.py：可导入函数调用；潜在调用者 0 个
- linear_attention_causal.py：可导入函数调用；潜在调用者 0 个
- linear_attention_normalize.py：可导入函数调用；潜在调用者 0 个
- opt_gdn_full.py：可导入函数调用；潜在调用者 0 个

## examples\linear_attention_and_rnn\gdn（6 个）

- gdn_chunk_cumsum.py：主入口直接运行；潜在调用者 2 个
- gdn_chunk_h.py：主入口直接运行；潜在调用者 2 个
- gdn_chunk_o.py：主入口直接运行；潜在调用者 2 个
- gdn_chunk_scaled_dot_kkt.py：主入口直接运行；潜在调用者 2 个
- gdn_solve_tril.py：主入口直接运行；潜在调用者 2 个
- gdn_wy_fast.py：主入口直接运行；潜在调用者 2 个

## examples\linear_attention_and_rnn\opt_gdn（6 个）

- opt_gdn_chunk_cumsum.py：主入口直接运行；潜在调用者 1 个
- opt_gdn_chunk_h.py：主入口直接运行；潜在调用者 1 个
- opt_gdn_chunk_o.py：主入口直接运行；潜在调用者 1 个
- opt_gdn_chunk_scaled_dot_kkt.py：主入口直接运行；潜在调用者 1 个
- opt_gdn_solve_tril.py：主入口直接运行；潜在调用者 1 个
- opt_gdn_wy_fast.py：主入口直接运行；潜在调用者 1 个

## examples\mha_sink_fwd_bhsd（2 个）

- mha_sink_fwd_bhsd.py：主入口直接运行；潜在调用者 1 个
- test_mha_sink_fwd_bhsd.py：主入口和测试入口；潜在调用者 1 个

## examples\moe_token_permute（4 个）

- moe_token_permute.py：主入口和测试入口；潜在调用者 2 个
- moe_token_permute_grad.py：主入口和测试入口；潜在调用者 0 个
- moe_token_unpermute.py：主入口和测试入口；潜在调用者 1 个
- moe_token_unpermute_grad.py：主入口和测试入口；潜在调用者 0 个

## examples\normalization（2 个）

- layer_norm.py：可导入函数调用；潜在调用者 0 个
- rms_norm.py：可导入函数调用；潜在调用者 9 个

## examples\pad（2 个）

- example_broadcast.py：主入口直接运行；潜在调用者 0 个
- example_broadcast_pipeline.py：主入口直接运行；潜在调用者 0 个

## examples\pipeline（5 个）

- flash_attn_bshd_pipeline.py：可导入函数调用；潜在调用者 0 个
- gemm_v0_pipeline.py：可导入函数调用；潜在调用者 0 个
- matmul_add_pipeline.py：主入口直接运行；潜在调用者 0 个
- sparse_flash_attn_gqa_pipeline.py：可导入函数调用；潜在调用者 0 个
- sparse_flash_attn_gqa_pipeline_pto.py：可导入函数调用；潜在调用者 0 个

## examples\pos_embedding（11 个）

- rms_norm.py：主入口直接运行；潜在调用者 9 个
- rms_rope_fused.py：主入口直接运行；潜在调用者 3 个
- rms_rope_fused_mask.py：主入口直接运行；潜在调用者 1 个
- rope.py：主入口直接运行；潜在调用者 30 个
- rope_mask.py：主入口直接运行；潜在调用者 2 个
- rope_mask_bwd.py：主入口直接运行；潜在调用者 1 个
- test_rms_norm_pos_embedding.py：测试入口运行；潜在调用者 0 个
- test_rms_rope_fused.py：测试入口运行；潜在调用者 1 个
- test_rms_rope_fused_mask.py：测试入口运行；潜在调用者 0 个
- test_rope.py：测试入口运行；潜在调用者 3 个
- test_rope_mask.py：测试入口运行；潜在调用者 1 个

## examples\quant_batch_matmul（2 个）

- example_quant_batch_matmul.py：主入口直接运行；潜在调用者 0 个
- example_quant_matmul.py：主入口直接运行；潜在调用者 0 个

## examples\reduce（4 个）

- example_col_reduce_max_slice_buffer.py：可导入函数调用；潜在调用者 0 个
- example_reduce_min.py：主入口直接运行；潜在调用者 0 个
- example_reduce_min_pipeline.py：主入口直接运行；潜在调用者 0 个
- example_row_reduce_max_slice_buffer.py：可导入函数调用；潜在调用者 0 个

## examples\simple_fusion（2 个）

- matmul_add.py：可导入函数调用；潜在调用者 4 个
- matmul_add_infer_scope.py：可导入函数调用；潜在调用者 0 个

## examples\softmax（1 个）

- example_online_softmax.py：可导入函数调用；潜在调用者 0 个

## examples\sparse_flash_attention（7 个）

- example_sparse_flash_attn.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_dynamic_shape.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_gqa.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_gqa_pto.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_gqa_pto_developer.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_mask.py：可导入函数调用；潜在调用者 0 个
- example_sparse_flash_attn_mask_pa.py：可导入函数调用；潜在调用者 0 个

## examples\sparse_flash_attention\bench_sfa（1 个）

- bench_sfa.py：测试入口运行；潜在调用者 0 个

## examples\tail_mask（1 个）

- example_tail_add.py：主入口直接运行；潜在调用者 0 个

## examples\tile_kernels\mhc（1 个）

- head_compute_mix_kernel.py：主入口和测试入口；潜在调用者 0 个

## examples\tile_kernels\moe（3 个）

- moe_aux_fi.py：主入口和测试入口；潜在调用者 0 个
- moe_topk_gate.py：主入口和测试入口；潜在调用者 0 个
- moe_topk_sum_and_topk_group_idx.py：主入口和测试入口；潜在调用者 0 个

## examples\tile_kernels\quant（1 个）

- per_block_cast_lossless_kernel.py：主入口和测试入口；潜在调用者 0 个

## examples\torch_tl_ascend（2 个）

- test_source.py：主入口直接运行；潜在调用者 0 个
- test_torch.py：主入口直接运行；潜在调用者 0 个

## examples\unsorted_segment_sum（1 个）

- unsorted_segment_sum.py：主入口直接运行；潜在调用者 0 个

## examples\xattention（2 个）

- xattention.py：主入口直接运行；潜在调用者 1 个
- xattention_paged.py：主入口直接运行；潜在调用者 0 个

## examples\xllm_kernels（3 个）

- fused_gdn_gating.py：主入口直接运行；潜在调用者 1 个
- rope.py：主入口直接运行；潜在调用者 30 个
- split_qkv_rmsnorm_mrope.py：主入口直接运行；潜在调用者 1 个

