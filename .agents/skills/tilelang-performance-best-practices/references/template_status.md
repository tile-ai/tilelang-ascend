# 模板成熟度与兼容性

先按本表判断代码可用性，再读取算子族文档。`PRODUCTION_REFERENCE` 是特征门控状态：当前 kernel 与 launcher 必须包含本表摘要及链接文档的核心结构；缺失时将当前路径降为 `EXECUTABLE_BASELINE`，历史优化知识只作候选，不继承性能结论。文件存在不代表优化策略已经实现。

结构文档中的“已实现并实测”是知识证据等级，不属于代码模板成熟度，也不决定候选实施优先级；迁入当前实现仍须核对适用条件并重新验证。

| 状态 | 使用规则 |
|---|---|
| `PRODUCTION_REFERENCE` | 真源位于当前仓库；直接引用生产文件，不在 Skill 复制 kernel |
| `VERIFIED` | 在记录的 A2/A3 版本完成当前后端 lowering、设备运行和精度校验；迁入其他版本仍须复验 |
| `EXECUTABLE_BASELINE` | 可执行的正确性起点，没有性能结论，不得直接进入生产 dispatch |
| `PARTIAL` | 只有部分策略结构或退化为 baseline，必须补全并验证 |
| `DESIGN_ONLY` | 只有设计、tiling 元数据或 Python helper，不是 TileLang kernel |

## Agent 展示分类

成熟度描述实现完成度，不单独代表性能排名。Agent 必须同时检查本表“已验证范围或缺口”和对应算子族文档，再使用以下展示分类：

- ✅ **可直接拷贝优化模式**：`PRODUCTION_REFERENCE` 或当前版本 `VERIFIED` 的 TileLang `.py`，并且优化结构已经真实实现。可以复用实现骨架；只有存在适用于当前比较口径的性能证据时，才能称为高性能模板或预期更快。
- ⚠️ **可执行基线或部分实现**：`EXECUTABLE_BASELINE`、`PARTIAL`，或虽可运行但没有对应优化结构/性能证据的实现。只作为正确性起点、实现参考或待补全候选。
- ❌ **仅设计参考**：`DESIGN_ONLY`，或没有当前版本可执行 TileLang kernel 的资料。不得进入直接实施方案。

不得仅因存在完整 `.py` 文件就判为“可直接拷贝优化模式”；也不得仅因代码位于本 Skill 中就宣称性能优于目标仓库当前实现。

## 本库参考的当前证据范围

本次整理进行了文档与源码核对，没有运行 NPU 编译、精度或性能测试；因此没有为下列参考新授予 `VERIFIED` 状态，也没有将历史数字移贴到当前实现。

| 知识入口 | 参考性质 | 使用前重点核验 |
|---|---|---|
| [MatMul](matmul/guide.md) | 当前 API/示例链接 + 带条件历史实验 | gemm_v0/mma 参数、分层 K、buffer ownership、布局和精度 |
| [FlashAttention](flash_attention/guide.md) | 当前 expert/auto_pipeline 实现链接 + C/V 设计 | 版本差异、mask/GQA、stride、workspace、尾块和多次调用 |
| [RoPE](rope/guide.md) | 当前实现与历史优化路径 | 索引范围、layout、dtype、原位语义和尾行 |
| [Elementwise](elementwise/guide.md) | 向量化与流水结构参考 | 实际 API、slot 生命周期、数值等价和有效范围 |
| [Broadcast](broadcast/guide.md) | 多行 tile 与参数广播方法 | 真实 stride、padding、输出归属和 buffer 预算 |
| [Reduction](reduce/guide.md) | 当前归约/online 示例及算法分析 | 轴、输出形状、在线递推和数值稳定 |
| [Scan](scan/guide.md) | 当前示例与正交轴设计 | 依赖、int32 精确性、NaN/tie 和完整 host 成本 |
| [Conversion](conversion/guide.md) | dtype、布局和多阶段设计 | bit-exact、batch 边界、每 stage 合同及完整调用链 |

专题中的历史性能记录只说明原作者观察，缺少完整设备/版本/日志时明确列出缺口。不把历史负收益变成永久禁止，也不把 API 回归示例当作性能模板。

## 新增设计专题状态

本库新增的算法策略、流程与结构文档没有随附 A2/A3 可执行 kernel 的，均为 `DESIGN_ONLY`；包括对应的 Reduction/Scan 策略文档、Stream-K、SWAT、Conv 组合与通算组合设计。原文的“已实现并实测”标签不能跨平台继承，专题会注明目标验证缺口。当前仓库真源可另有完整证据，按实际实现独立判断，不因设计文档未实现而否定它。

实际运行前核对 `python -c 'import tilelang; print(tilelang.__file__, tilelang.__version__)'`。版本或源码路径不同即视为未验证组合。

## 状态升级门禁

把 `PARTIAL` 或 `DESIGN_ONLY` 升为 `VERIFIED` 时同时记录：目标硬件、CANN、TileLang 与当前仓库版本、完整命令、shape/dtype、容差、lowering 状态、首个失败、性能口径和原始结果文件。没有同条件实测数据时只称“实现候选”或“正确性 baseline”。
