# TileLang 性能知识索引

先阅读 [通用实现指南](tilelang_ascend_guide.md)、[模板成熟度](template_status.md) 和 [验证门禁](validation.md)。目标仓库已有实现直接引用仓库内真源；TileLang API 通过实际导入路径定位，不在 Skill 中复制第二份。

结构文档只提供提炼后的优化决策，不索引或要求读取对应历史优化实现；从当前目标 baseline 自主实现并验证。

| 算子族 | 文档 | 代码参考 | 关键优化 |
|---|---|---|---|
| Common | [npu-arch](../../npu-arch/SKILL.md) · [copy](common/datacopy_optimization_design.md) · [tail](common/tail_block_design.md) · [resident](common/ub_resident_design.md) · [task map](common/persistent_task_mapping.md) | [当前 examples](../../../../examples/) | 硬件事实、合并搬运、padding、常驻、多版本、多轴任务映射 |
| Broadcast | [guide](broadcast/broadcast_design.md) | [广播与参数加载](broadcast/broadcast_design.md) | 单轴/多轴、resident input、shape specialization |
| Conversion | [guide](conversion/guide.md) | [Transpose 测试](../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) | UB 转置测试与搬运 API 参考；通用转置需提供已验证 factory |
| Elementwise / Gather | [vector](elementwise/vector_efficiency_design.md) · [tiling/task](elementwise/tiling_task_vector_search.md) · [pipeline](elementwise/double_buffer_design.md) · [indexed short record](elementwise/indexed_short_record.md) | [Elementwise 示例](../../../../examples/elementwise/)、[搬运 API](../../../../tilelang/language/copy_op.py) | Vector、短记录打包、共享字段融合、partial 打包、静态整 tile/JIT、连续搬运、任务展平、指令数据流、2/3 stage |
| MatMul | [guide](matmul/guide.md) | [GEMM 示例](../../../../examples/gemm/) | tile 搜索、L1/L0、流水、resident、Stream-K |
| Reduction | [guide](reduce/guide.md) · [rowwise reduce epilogue](reduce/rowwise_reduce_epilogue.md) · [wide output](reduce/wide_output_tiling.md) · [batched short reduction](reduce/batched_short_reduction.md) · [fixed small state](reduce/fixed_small_state.md) | [reduction strategies（设计）](reduce/templates/) | fp32 状态、full-load、行归约与逐元素写回融合、wide-output tile、recompute、online、SIMD 短归约、小状态常驻、split-axis |
| FlashAttention | [guide](flash_attention/guide.md) | [FlashAttention 示例](../../../../examples/flash_attention/) | Q resident、online softmax、Cube/Vector forwarding |
| Scan | [guide](scan/guide.md) | [scan strategies（设计）](scan/templates/) | row-owner、resident carry、lane scan、三阶段 split |
| RoPE | [guide](rope/guide.md) | [当前 RoPE 示例](../../../../examples/pos_embedding/rope.py) | fp32 pair rotation、layout specialization |
| Scalar | [guide](scalar/guide.md) | 各算子 Python factory | 常量化、地址复用、生命周期、代码体积 |
| Sort/TopK | [guide](sort/radix_sort.md) | [TopK selector](../../../../examples/topk_selector/) | padded load、局部选择、分层 merge |
| Conv | [guide](conv/guide.md) | 当前 `T.gemm_v0` / `T.mma` | tiled im2col/grouped GEMM |
| 通算组合 | [guide](mc2/guide.md) | local GEMM + 正式 distributed runtime | 异步 collective、chunk pipeline |

## 验证要求

- 从目标仓库复制的模板仍须在当前分支重新运行对应测试。
- Scan、RoPE 和布局转换实现先做当前后端 lowering，再做设备精度与端到端性能测试。
- 先按 [template_status.md](template_status.md) 标注成熟度；`PARTIAL`/`DESIGN_ONLY` 不得列为可直接拷贝模板。
- fp16/bf16 reduction、norm、softmax 和 GEMM 默认使用 fp32 统计或累加状态。
- 尾块、动态 shape、NaN/Inf、极小任务和多阶段 workspace 必须纳入测试矩阵。

## Common

- 核数、buffer 与架构能力：[npu-arch](../../npu-arch/SKILL.md) 和 [编译器有效容量](common/hardware_resource_discovery.md)，复用已确认的 SoC/NpuArch 证据，区分平台可用量和编译器预算。
- 固定核数、任务波次与每核尾差：[任务映射](common/task_mapping.md)。
- 自动/手动同步、C/V handoff：[流水与同步](common/pipeline_and_cv_overlap.md)。
- Host pad/contiguous、非连续 stride、D2H 与数据相关同步：[Host 与设备数据流](common/host_device_dataflow.md)。

## 专题检索

- 多行 tile 和原参数 staging：[Broadcast](broadcast/broadcast_design.md)。
- 归约遍数融合与 online 递推：[Online Softmax](reduce/vector_reduce_pass_fusion.md)。
- dtype 特定硬件路径：[dtype 适配](conversion/dtype_adaptation.md)。
- 多 batch、双尾块及 kernel→planner 审计：[多阶段连续化](conversion/multi_stage_contiguous.md)。
- 正交轴向量化及 NaN/整数精度：[Scan](scan/orthogonal_axis_vectorization.md)。
- 历史参数和负结果：[GEMM 历史案例](matmul/historical_cases.md)。
- 性能评审：[性能关注项](performance-antipatterns.md)，再按命中项读取专题。
