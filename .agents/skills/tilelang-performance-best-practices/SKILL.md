---
name: tilelang-performance-best-practices
description: TileLang 性能最佳实践与反模式检查。按算子族提供 Tiling、搬运、流水、同步、归约、GEMM 和布局优化的适用条件及源码依据，供性能方案分析、实现和代码评审按需查阅。
---

# TileLang 性能最佳实践

按以下优先级使用事实来源：

1. 当前目标算子、当前仓库的 `examples/`、测试和公开接口约束。
2. 实际导入的 TileLang 框架源码、`examples/` 与 `testing/`；用实际 Python 导入结果核对运行版本与参考源码是否一致。
3. `references/` 中按成熟度分类的实现参考、策略元数据和设计文档。

## 工作流

1. 在当前目标工程和实际 TileLang 源码的 `examples/` 查找同类算子，优先复用当前分支已经验证的调度与 API。
2. 定位实际 TileLang 源码并核对实际导入版本；在 `examples/`、`testing/`、API 定义和当前后端 lowering 中核对不确定接口，禁止按目录名猜测安装版本。
3. 容量、核数或架构影响实现时，阅读 [硬件资源与编译器有效容量发现](references/common/hardware_resource_discovery.md)，按名称加载 [npu-arch](../npu-arch/SKILL.md)，传入已确认的 SoC/NpuArch 与目标设备证据，记录已确认的硬件资源来源，并从实际 TileLang lowering 核对平台用户可用量、编译器预算与预留。
4. 阅读 [优化点索引](references/index.md)、[模板成熟度](references/template_status.md) 和对应算子族文档；只有 `VERIFIED` 或 `PRODUCTION_REFERENCE` 可作为直接复用候选。
5. 先实现精度正确的最小路径，再根据基准数据优化。不得通过放宽容差、降低 reference 精度或跳过 shape 获得“通过”。
6. 由实施阶段先运行当前后端定向精度测试（独立 pytest 入口示例）：

   ```bash
   pytest <test_file> -x
   ```

7. 定向精度测试全部通过后运行相关完整测试套；合入前按项目要求运行扩展测试。性能比较保持输入、dtype、warmup、repeat、设备和并发度一致。
8. 报告容差、通过数、首个失败、覆盖 shape、kernel latency 和基线；没有实测数据时只称“设计建议”。
9. 区分数值失败、编译失败和仓库主动抛出的 `NotImplementedError`。未实现路径必须如实记录并补齐实现，不能放宽容差、跳过测试或把此前通过的子集称为全量通过。

## 验证入口

- 当前仓库的算子测试、`testing/python/` 与各算子族链接的现有示例用于验证；沿用目标的真实编译和测试入口，不假设固定 CLI。

## 代码准入规则

- Kernel 实现只使用 Python TileLang DSL；tiling、dispatch 与数学 reference 使用普通 Python。先从目标仓库同类算子实现 派生，再按 [模板成熟度](references/template_status.md) 选择 bundled reference。
- 编译后端沿用目标工程已经确认的配置；不为复用模板擅自切换 target，不把其他后端的环境变量或参数写入当前调用链。
- Ascend `T.Kernel` 的网格、`threads`、`cid/vid` 和 workspace 规则按当前仓库 [公共语言参考](../../../docs/language_ref/primitives.md) 核对，不套用其他后端的线程域规则。
- 执行域优先沿用同类已验证实现：规则逐元素通常使用 `T.Parallel` 或 `T.tile.*`，Norm/Reduction 复用当前 `T.reduce_*` 路径；离散索引、复杂分支或原子操作按当前 API 与同类源码核对。执行域变更必须通过精度与性能实测。
- GEMM 优先沿用已验证的 `T.gemm_v0` / `T.mma` 结构与布局；其他布局不是 API 禁止项，但必须通过目标版本 lowering 与设备验证。首个有效 K 子 tile 清零 fp32 L0C，后续子 tile 累加。
- fp16/bf16 的归约、归一化、softmax 统计量和矩阵乘以 fp32 累加为起点。内部混合精度、HF32 和近似指令可实验，但必须保持原测试、容差和公开约定，并以正确性与同口径性能决定是否保留。
- Vector、Cube 与混合核分别使用硬件查询确认的可用 AIV、AIC 及配对资源；构建模板时显式传入核数，小任务核数不超过独立 task 数。
- Elementwise、gather/scatter 或布局转换优化必须按 [Tiling、任务与 Vector 数据流搜索](references/elementwise/tiling_task_vector_search.md) 建立任务树和有来源的 tile 边界。物理路线按完整数据路径分类：planar/scratch 中间 payload 属于 materialized transform，不能代替直接连续 load 后的寄存器重排。将 tile/task、数据流、精度和流水作为正交轴，并由同一实际激活候选裁决有交互的组合。
- UB 预算必须区分已确认的硬件规格、平台用户可用容量、当前 lowering 的编译器预算与预留和显式 buffer footprint；尤其在改变执行侧、cast、scratch 或临时 buffer 后重新计算。预算同时包含 buffer version、对齐 padding、常驻数据和有明确依据的安全余量。GM 尾块只 copy `valid`，UB 仍按完整 SIMD/DMA footprint 分配。
- 流水任务必须按 [多版本流水设计](references/elementwise/double_buffer_design.md) 建立跨迭代依赖；所有仍存活的 buffer 都要版本化，自动分析不适用时改用显式 stage storage。是否生效以生成代码及同口径 latency/overlap 为准。
- 多轴 tile 域按 [Persistent 多轴任务映射](references/common/persistent_task_mapping.md) 核对，避免热循环中不必要的扁平索引解码。
- 归约计算较轻且非归约连续轴较宽时，按 [Reduction 宽输出轴 Tiling](references/reduce/wide_output_tiling.md) 评估合并连续输出 tile。
- 以公开接口合约确定 shape 矩阵；接口不支持任意尾块时只覆盖允许余数类，不把新 fallback 当成既有要求。同时检查 forward/backward、动态 shape、极小/极大 shape、NaN/Inf 语义和空任务边界。

## 算子族路由

| 算子族 | 首选模式 | 源码参考（复用前验证） |
|---|---|---|
| Elementwise / Quant | Persistent + UB staging + Vector；短记录索引变换读取 [结构参考](references/elementwise/indexed_short_record.md) | [Elementwise 示例](../../../examples/elementwise/)；Quant 按目标 API 和 dtype 核验 |
| Rowwise Reduction + Elementwise Epilogue | 精确归约轴 UB + 常驻广播参数 + SIMD fp32 归约；必须同时评估单行流水与跨行批处理，读取 [行归约与逐元素写回](references/reduce/rowwise_reduce_epilogue.md) 和 [批量短归约](references/reduce/batched_short_reduction.md) | 按目标算子验证 |
| Reduction / Norm / Softmax | fp32 累加、分层归约；按结构核对 [批量短归约](references/reduce/batched_short_reduction.md) 或 [固定小状态](references/reduce/fixed_small_state.md) | [Normalization 示例](../../../examples/normalization/) |
| Transpose / Gather | 按源窗口跨度与密度选择连续 load + 已验证重排、Vector gather 或临时物化；不支持的路径再评估 UB-local fallback | [Transpose 测试](../../../testing/python/language/test_tilelang_ascend_language_transpose.py)、[copy_op.py](../../../tilelang/language/copy_op.py)；通用转置实现按目标算子核对 |
| MatMul | L1/L0C、已验证布局、K 维流水；Stream-K/full-load/SWAT 为候选 | [GEMM 示例](../../../examples/gemm/) |
| FlashAttention | Cube/Vector 数据流、在线 softmax、fp32 状态 | [FlashAttention 示例](../../../examples/flash_attention/) |
| TopK / irregular | 当前 `T.tile.*`、`T.Parallel` 或已验证索引路径 | [TopK selector 示例](../../../examples/topk_selector/)、[排序专题](references/sort/radix_sort.md) |

## Reference 路由

从 [references/index.md](references/index.md) 选择算子族，并以 [template_status.md](references/template_status.md) 判断可用性。`EXECUTABLE_BASELINE` 只用于精度起点，`PARTIAL`/`DESIGN_ONLY` 不得描述为可直接拷贝的优化实现；状态升级必须同时留下 lowering、精度和性能证据。
