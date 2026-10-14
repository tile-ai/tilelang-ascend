# 精度修改与低性能示例迁移清单

## 范围

- 基线：`origin/ascendc_pto`
- 当前分支：`precision-and-move-pr`
- 精度相关实现/测试函数所在文件：153 个
- 新增逐文件精度测试：55 个
- 迁移的低性能算子实现：23 个

## 精度标准

所有 `_check_precision`、`check_precision`、`_get_precision` 及关联校验函数按 `.agents/skills/tilelang-op-test-design/references/precision-standard.md` 修正：FP16 使用 `atol=2**-14`、`rtol=2**-9`、`max_abs<=1e-1`、通过率 `>=0.99`；其他浮点、FP8、整数、NaN/Inf 遵循同一标准中的对应规则。

## 新增测试文件（55 个）

- `examples/activation/test_precision_gelu_grad.py`
- `examples/activation/test_precision_gelu_mul.py`
- `examples/activation/test_precision_sigmoid.py`
- `examples/activation/test_precision_sigmoidv2.py`
- `examples/activation/test_precision_sigmoidv2_slice.py`
- `examples/activation/test_precision_silu.py`
- `examples/activation/test_precision_swi_glu.py`
- `examples/activation/test_precision_swi_glu_grad.py`
- `examples/activation/test_precision_swi_glu_v2.py`
- `examples/activation/test_precision_tanh.py`
- `examples/compile_flags/test_precision_compile_flags_example.py`
- `examples/deepseek_v4/sparse_flash_mla/test_precision_sparse_flash_mla_golden.py`
- `examples/developer_mode/test_precision_flash_attn_bshd_developer.py`
- `examples/developer_mode/test_precision_gelu_mul_developer.py`
- `examples/developer_mode/test_precision_matmul_add_developer.py`
- `examples/developer_mode/test_precision_matmul_add_developer_vec_cast.py`
- `examples/elementwise/test_precision_elementwise_add.py`
- `examples/elementwise/test_precision_elementwise_add_pipeline.py`
- `examples/exception_dump_test/test_precision_example_exception_dump.py`
- `examples/flash_attention/test_precision_flash_attn_bhsd_cc_sync.py`
- `examples/gemm/test_precision_example_gemm_fp8_pto.py`
- `examples/gemm/test_precision_example_gemm_intrinsic.py`
- `examples/gemm/test_precision_example_gemm_tail_block_developer.py`
- `examples/gemm_aot/test_precision_test_example_gemm.py`
- `examples/linear_attention_and_rnn/test_precision_gdn_full.py`
- `examples/linear_attention_and_rnn/test_precision_linear_attention_causal.py`
- `examples/linear_attention_and_rnn/test_precision_linear_attention_normalize.py`
- `examples/linear_attention_and_rnn/test_precision_opt_gdn_full.py`
- `examples/normalization/test_precision_layer_norm.py`
- `examples/normalization/test_precision_rms_norm.py`
- `examples/pipeline/test_precision_flash_attn_bshd_pipeline.py`
- `examples/pipeline/test_precision_sparse_flash_attn_gqa_pipeline.py`
- `examples/pipeline/test_precision_sparse_flash_attn_gqa_pipeline_pto.py`
- `examples/reduce/test_precision_example_col_reduce_max_slice_buffer.py`
- `examples/reduce/test_precision_example_row_reduce_max_slice_buffer.py`
- `examples/simple_fusion/test_precision_matmul_add.py`
- `examples/simple_fusion/test_precision_matmul_add_infer_scope.py`
- `examples/softmax/test_precision_example_online_softmax.py`
- `examples/sparse_flash_attention/test_precision_example_sparse_flash_attn_gqa.py`
- `examples/sparse_flash_attention/test_precision_example_sparse_flash_attn_gqa_pto.py`
- `examples/sparse_flash_attention/test_precision_example_sparse_flash_attn_gqa_pto_developer.py`
- `examples_experiment/autotune/test_precision_example_gemm_autotune.py`
- `examples_experiment/autotune/test_precision_example_gemm_carver.py`
- `examples_experiment/developer_mode/test_precision_gemm_developer.py`
- `examples_experiment/developer_mode/test_precision_sparse_flash_attn_developer.py`
- `examples_experiment/developer_mode/test_precision_sparse_flash_attn_developer_vid_reduce.py`
- `examples_experiment/gemm/test_precision_example_gemm.py`
- `examples_experiment/gemm/test_precision_example_gemm_infer_scope.py`
- `examples_experiment/gemm/test_precision_example_gemm_pto_developer.py`
- `examples_experiment/gemm/test_precision_example_gemm_transpose_l1.py`
- `examples_experiment/pipeline/test_precision_gemm_v0_pipeline.py`
- `examples_experiment/sparse_flash_attention/test_precision_example_sparse_flash_attn.py`
- `examples_experiment/sparse_flash_attention/test_precision_example_sparse_flash_attn_dynamic_shape.py`
- `examples_experiment/sparse_flash_attention/test_precision_example_sparse_flash_attn_mask.py`
- `examples_experiment/sparse_flash_attention/test_precision_example_sparse_flash_attn_mask_pa.py`

这些测试调用对应源码中的精度 checker，以零误差基线验证判定逻辑；不替代完整 NPU kernel 端到端测试。

## 文件与函数明细

以下函数名来自当前 PR 相对上游的实际 Python 文件扫描；函数名不代表所有函数均被修改，精度相关函数均已纳入本次校准或审计。

- `examples/HISA/block_sparse_mqa_attn_expert_test.py`：_check_precision, test_block_sparse_mqa_attn
- `examples/HISA/block_sparse_mqa_attn_expert_test_for_a5.py`：_check_precision, test_block_sparse_mqa_attn
- `examples/HISA/paged_block_sparse_mqa_attn_expert.py`：_check_precision, test_paged_block_sparse_mqa_attn
- `examples/aclgraph/rms_rope_aclgraph.py`：_check_precision
- `examples/activation/gelu_grad.py`：_check_precision
- `examples/activation/gelu_mul.py`：_check_precision
- `examples/activation/sigmoid.py`：_check_precision
- `examples/activation/sigmoidv2.py`：_check_precision
- `examples/activation/sigmoidv2_slice.py`：_check_precision
- `examples/activation/silu.py`：_get_precision, _check_precision
- `examples/activation/swi_glu.py`：_check_precision
- `examples/activation/swi_glu_grad.py`：_check_precision
- `examples/activation/swi_glu_v2.py`：_check_precision
- `examples/activation/tanh.py`：_get_precision, _check_precision
- `examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py`：_get_precision, _check_precision
- `examples/attention_sink/example_gqa_sink_bwd_bhsd/test_gqa_sink_bwd_bhsd.py`：get_precision, check_precision, test_gqa_sink_bwd_bhsd_l0, test_gqa_sink_bwd_bhsd_l1, test_gqa_sink_bwd_bhsd_l2, test_gqa_sink_bwd_bhsd_boundary
- `examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py`：_get_precision, _check_precision
- `examples/attention_sink/example_gqa_sink_fwd_varlen/test_gqa_sink_fwd_varlen.py`：_get_precision, _check_precision, test_gqa_sink_fwd_l0, test_gqa_sink_fwd_l1, test_gqa_sink_fwd_l2, test_gqa_sink_fwd_boundary, test_forward
- `examples/batch_gemm/test_batch_gemm.py`：_check_precision, test_batch_gemm_accuracy
- `examples/cann-bench/cummin/example_cummin.py`：test_cummin_all
- `examples/cann-bench/gather/gather.py`：_check_precision
- `examples/cann-bench/transpose/transpose.py`：_check_precision
- `examples/causal_conv1d/causal_conv1d.py`：_check_precision, _run_test
- `examples/causal_conv1d/causal_conv1d_decode.py`：_check_precision, _run_ref_check
- `examples/causal_conv1d/causal_conv1d_pto.py`：_check_precision, _run_test
- `examples/compile_flags/compile_flags_example.py`：_check_precision
- `examples/cross_entropy_loss/example_cross_entro.py`：_check_precision, check_case
- `examples/deepseek_v4/act_quant.py`：_check_precision, test
- `examples/deepseek_v4/hc_split_sinkhorn.py`：_check_precision, test
- `examples/deepseek_v4/int8_gemm.py`：_check_precision, test
- `examples/deepseek_v4/lightning_indexer.py`：_check_precision, _check_result
- `examples/deepseek_v4/sparse_attention.py`：_check_precision, make_random_test_inputs, test
- `examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_golden.py`：_check_precision, check_result, check_lse
- `examples/deepseek_v4/test_act_quant.py`：_check_precision, test_act_quant_accuracy
- `examples/deepseek_v4/test_hc_split_sinkhorn.py`：_check_precision, test_hc_split_sinkhorn_accuracy
- `examples/deepseek_v4/test_int8_gemm.py`：_check_precision, test_int8_gemm_accuracy
- `examples/deepseek_v4/test_sparse_attention.py`：_check_precision, test_sparse_attention_accuracy
- `examples/developer_mode/flash_attn_bshd_developer.py`：_check_precision
- `examples/developer_mode/gelu_mul_developer.py`：_check_precision
- `examples/developer_mode/matmul_add_developer.py`：_check_precision
- `examples/developer_mode/matmul_add_developer_vec_cast.py`：_check_precision
- `examples/dispatch_combine/dispatch_combine_shmem.py`：_check_precision
- `examples/elementwise/elementwise_add.py`：_get_precision, _check_precision
- `examples/elementwise/elementwise_add_pipeline.py`：_check_precision
- `examples/exception_dump_test/example_exception_dump.py`：_check_precision
- `examples/flash_attention/fa_opt/flash_attn_bhsd_ascendc.py`：_check_precision
- `examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h32_d512.py`：_check_precision
- `examples/flash_attention/flash_attn_bhsd.py`：_check_precision
- `examples/flash_attention/flash_attn_bhsd_cc_sync.py`：_check_precision
- `examples/flash_attention/paged_flash_attn_bhsd.py`：_check_precision, check_case
- `examples/flash_attention/test_flash_attn_bhsd.py`：_check_precision, test_flash_attn_bhsd_accuracy
- `examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py`：_check_precision
- `examples/fusedmoe/example_fusedmoe.py`：_check_precision, host_preprocess_for_test
- `examples/fusedmoe/test_fusedmoe.py`：_check_precision, test_fusedmoe_l0, _run_precision, test_fusedmoe_l1, test_fusedmoe_l2, test_fusedmoe_boundary, test_fusedmoe_bench, test_forward
- `examples/gemm/example_gemm_fp8_pto.py`：_check_precision
- `examples/gemm/example_gemm_intrinsic.py`：_check_precision
- `examples/gemm/example_gemm_intrinsic_persistent.py`：_check_precision
- `examples/gemm/example_gemm_tail_block_developer.py`：_check_precision
- `examples/gemm_aot/test_example_gemm.py`：_check_precision
- `examples/gemv/example_gemv_c.py`：_check_precision, check_case
- `examples/generative_recommendation/mtgr_ragged_segment_attention.py`：_check_precision, test
- `examples/gqa_fwd_varlen/gqa_fwd_varlen.py`：_check_precision
- `examples/gqa_fwd_varlen/perf_gqa_fwd_varlen.py`：_check_precision
- `examples/gqa_fwd_varlen/test_gqa_fwd_varlen.py`：_check_precision, test_gqa_fwd_varlen_l0, test_gqa_fwd_varlen_l1, test_gqa_fwd_varlen_l2, test_gqa_fwd_varlen_boundary
- `examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn/gdn_chunk_h.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn/gdn_chunk_o.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn/gdn_chunk_scaled_dot_kkt.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn/gdn_solve_tril.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn/gdn_wy_fast.py`：_check_precision
- `examples/linear_attention_and_rnn/gdn_full.py`：_check_precision
- `examples/linear_attention_and_rnn/linear_attention_causal.py`：_check_precision
- `examples/linear_attention_and_rnn/linear_attention_normalize.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_cumsum.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_h.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_o.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_chunk_scaled_dot_kkt.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_solve_tril.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn/opt_gdn_wy_fast.py`：_check_precision
- `examples/linear_attention_and_rnn/opt_gdn_full.py`：_check_precision
- `examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py`：_check_precision
- `examples/mha_sink_fwd_bhsd/test_mha_sink_fwd_bhsd.py`：_check_precision, test_mha_sink_fwd_bhsd_l0, test_mha_sink_fwd_bhsd_l1, test_mha_sink_fwd_bhsd_l2, test_mha_sink_fwd_bhsd_boundary
- `examples/moe_token_permute/moe_token_permute.py`：_check_precision, test_permute_parameterized, test_permute
- `examples/moe_token_permute/moe_token_permute_grad.py`：_check_precision, test_permute_grad_parameterized, test_permute_grad
- `examples/moe_token_permute/moe_token_unpermute_grad.py`：_check_precision, test_unpermute_grad_parameterized, test_unpermute_grad
- `examples/normalization/layer_norm.py`：_check_precision
- `examples/normalization/rms_norm.py`：_check_precision
- `examples/pad/example_broadcast.py`：_check_precision
- `examples/pad/example_broadcast_pipeline.py`：_check_precision
- `examples/pipeline/flash_attn_bshd_pipeline.py`：_check_precision
- `examples/pipeline/matmul_add_pipeline.py`：_check_precision
- `examples/pipeline/sparse_flash_attn_gqa_pipeline.py`：_check_precision
- `examples/pipeline/sparse_flash_attn_gqa_pipeline_pto.py`：_check_precision
- `examples/pos_embedding/rms_norm.py`：_check_precision
- `examples/pos_embedding/rms_rope_fused.py`：_check_precision
- `examples/pos_embedding/rms_rope_fused_mask.py`：_check_precision
- `examples/pos_embedding/rope.py`：_check_precision
- `examples/pos_embedding/rope_mask.py`：_check_precision
- `examples/pos_embedding/rope_mask_bwd.py`：_check_precision, check_case_tnd, check_case_bsnd
- `examples/pos_embedding/test_rms_norm_pos_embedding.py`：_check_precision, test_rms_norm_accuracy
- `examples/pos_embedding/test_rms_rope_fused.py`：_check_precision, test_rms_rope_fused_accuracy
- `examples/pos_embedding/test_rms_rope_fused_mask.py`：_check_precision, test_rms_rope_fused_mask_accuracy
- `examples/pos_embedding/test_rope.py`：_check_precision, test_rope_accuracy
- `examples/pos_embedding/test_rope_mask.py`：_check_precision, test_rope_mask_accuracy
- `examples/quant_batch_matmul/example_quant_batch_matmul.py`：_check_precision, check_case
- `examples/quant_batch_matmul/example_quant_matmul.py`：_check_precision, check_case
- `examples/reduce/example_col_reduce_max_slice_buffer.py`：_check_precision
- `examples/reduce/example_reduce_min.py`：_check_precision
- `examples/reduce/example_reduce_min_pipeline.py`：_check_precision
- `examples/reduce/example_row_reduce_max_slice_buffer.py`：_check_precision
- `examples/simple_fusion/matmul_add.py`：_check_precision
- `examples/simple_fusion/matmul_add_infer_scope.py`：_check_precision
- `examples/softmax/example_online_softmax.py`：_check_precision
- `examples/sparse_flash_attention/bench_sfa/bench_sfa.py`：_check_precision, test_op
- `examples/sparse_flash_attention/example_sparse_flash_attn_gqa.py`：_check_precision
- `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto.py`：_check_precision
- `examples/sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py`：_check_precision
- `examples/tail_mask/example_tail_add.py`：_check_precision
- `examples/tile_kernels/mhc/head_compute_mix_kernel.py`：_check_precision, test_fwd, test_bwd
- `examples/tile_kernels/moe/moe_aux_fi.py`：_check_precision, generate_test_data, generate_test_params, test_aux_fi
- `examples/tile_kernels/moe/moe_topk_gate.py`：_check_precision, generate_test_params, test_topk_gate, test_topk_gate_backward
- `examples/tile_kernels/moe/moe_topk_sum_and_topk_group_idx.py`：_check_precision, generate_test_params, test_topk_sum_and_topk_group_idx, test_topk_sum_and_topk_group_idx_backward
- `examples/tile_kernels/quant/per_block_cast_lossless_kernel.py`：generate_test_data, generate_test_params, test_per_block_cast_lossless
- `examples/torch_tl_ascend/test_source.py`：_check_precision
- `examples/torch_tl_ascend/test_torch.py`：_check_precision
- `examples/unsorted_segment_sum/unsorted_segment_sum.py`：_check_precision, _test, _test_3d
- `examples/xattention/xattention.py`：_check_precision
- `examples/xattention/xattention_paged.py`：_check_precision
- `examples/xllm_kernels/fused_gdn_gating.py`：_check_precision, _run_ref_check
- `examples/xllm_kernels/rope.py`：_check_precision, _run_ref_check
- `examples/xllm_kernels/split_qkv_rmsnorm_mrope.py`：_check_precision, _run_ref_check
- `examples_experiment/autotune/example_gemm_autotune.py`：manual_check_prog
- `examples_experiment/autotune/example_gemm_carver.py`：manual_check_prog
- `examples_experiment/batch_gemm/batch_gemm.py`：_check_precision
- `examples_experiment/developer_mode/gemm_developer.py`：_check_precision
- `examples_experiment/developer_mode/sparse_flash_attn_developer.py`：_check_precision
- `examples_experiment/developer_mode/sparse_flash_attn_developer_vid_reduce.py`：_check_precision
- `examples_experiment/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h16_d128.py`：_check_precision
- `examples_experiment/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`：_check_precision
- `examples_experiment/gemm/example_gemm.py`：_check_precision
- `examples_experiment/gemm/example_gemm_infer_scope.py`：_check_precision
- `examples_experiment/gemm/example_gemm_persistent.py`：_check_precision
- `examples_experiment/gemm/example_gemm_pto_developer.py`：_check_precision
- `examples_experiment/gemm/example_gemm_transpose_l1.py`：_check_precision
- `examples_experiment/gemv/example_gemv_v.py`：_check_precision, check_case
- `examples_experiment/moe_token_permute/moe_token_unpermute.py`：_check_precision, test_unpermute_parameterized, test_unpermute
- `examples_experiment/pipeline/gemm_v0_pipeline.py`：_check_precision
- `examples_experiment/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_baseline.py`：init_test
- `examples_experiment/sparse_flash_attention/example_sparse_flash_attn.py`：_check_precision
- `examples_experiment/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py`：_check_precision
- `examples_experiment/sparse_flash_attention/example_sparse_flash_attn_mask.py`：_check_precision
- `examples_experiment/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py`：_check_precision
- `examples_experiment/topk_selector/example_topk_selector.py`：check_case

## 迁移文件

23 个低性能算子实现从 `examples/` 移至 `examples_experiment/`；迁移本身不改变实现逻辑。对存在精度测试的算子，测试文件同步移动并更新 `source_path`。

## 验证

```bash
python -m compileall -q examples examples_experiment
python scripts/validate_precision_all.py
python -m pytest -q $(find examples examples_experiment -type f -name 'test_precision_*.py' | sort)
```

CANN 9.1 运行结果：`53 passed, 2 skipped, 0 failed`。无效测试源码路径：0。

## 变更口径与审查说明

- “153 个文件”表示本 PR 相对 `origin/ascendc_pto` 中包含精度校验标记的 Python 文件总数，不表示 153 个文件都新增了测试入口。
- 精度实现函数主要为 `_check_precision`；涉及辅助函数时还包括 `_get_precision`、`check_precision`、`check_result`、`check_lse` 和 `check_case`。
- 新增测试文件统一包含 `load_checker` 与 `test_precision_checker`，并对零误差基线执行判定。
- 23 个低性能算子实现的迁移属于文件位置调整；由于其中部分文件同时有精度修改，在组合 PR 的整体 diff 中可能显示为“修改后重命名”，不能据此判断迁移改变了算法。
- 迁移后仍保留 55 个精度测试，所有测试源码路径均已指向实际文件位置。

## CI 门禁与限制

- 已通过 Python 语法编译检查和本地静态路径检查。
- CANN 9.1 环境下的精度 checker 测试已通过；该结果不等同于所有算子的完整 NPU 端到端执行结果。
- 两个自动调优示例没有独立精度 checker，因此按预期跳过，不应视为失败。
- PR 提交后仍需等待目标仓库 GitHub Actions 完成；当前文档不预先宣称 CI 已通过。
- 若目标仓库 CI 仅扫描 `examples/`，应确认是否需要将 `examples_experiment/` 纳入扫描范围。
- 若项目要求在 `ci/operator_test_manifest.yaml` 登记新增测试，应由维护者确认这些轻量 checker 测试是否需要登记，避免与现有 runner 重复执行。

## 推荐审核结论

本 PR 可以作为“精度校验标准修正 + 低性能示例归档迁移”提交审核。审核时应分别检查：

1. 精度函数的阈值、通过率、特殊值和整数分支；
2. 23 个实现文件是否保持纯迁移语义；
3. 迁移后的测试路径和 CI 收集范围；
4. 完整 NPU 算子测试是否需要在后续 PR 中单独补充。

