---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。触发：设计算子、生成 design.md、算子方案设计、新算子开发、算子实现方案。"
---

# TileLang-Ascend 算子设计文档生成

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-Ascend 算子设计文档（`design.md`），涵盖以下核心决策：

- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：Golden 函数与 L0 门槛测试计划（完整分层套件 L1/L2/Boundary 由 tilelang-op-test-design 生成）

---

## 2. 输入要求

### 必需信息

| 字段 | 说明 |
|------|------|
| 算子名称 | 如 `softmax`、`layer_norm`、`flash_attention` |
| 数学公式 | 算子的数学表达，如 $\text{softmax}(x_i) = e^{x_i} / \sum e^{x_j}$ |
| 输入张量规格 | shape、dtype |
| 输出张量规格 | shape、dtype |
| 编程模式偏好 | Developer / Expert / 混合 |
| **迁移算子路径** ⭐ | 原算子文件路径（迁移时必需），用于获取 golden 实现 |
| **输出形状** ⭐ | 原算子输出 shape（迁移时必需），如 `(N, M)` 或 `(M, N)` |

**迁移算子时必须提供原算子路径和输出形状**，否则无法证明迁移正确性。Golden 实现一致性要求详见 [tilelang-op-generate SKILL.md §8 Checklist #9-#10](../tilelang-op-generate/SKILL.md)。

**提问规则（必须严格遵守）**：
1. **优先使用调用方传入的字段**：若调用方（如 `@tilelang-op-orchestrator` 通过 analyst 传入 `op_requirements` 结构）已经提供了字段值，**全部跳过提问**，直接进入技术约束检测和 design 生成
2. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
3. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
4. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
5. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

**⚠️ 当被 orchestrator → analyst Subagent 链路调度时**：
- analyst 会把 orchestrator 在 Primary 上下文预检收集到的 `op_requirements` 完整传入
- 此时 5 个必需字段应当全部已 provided，跳过整个提问环节
- 若 skill 仍发现字段歧义或缺漏，**不要**在当前 Subagent 上下文调用 `AskUserQuestion`（透传不到真实用户），而是让 analyst 返回 `partial_input` + 缺失字段名给 orchestrator，由 orchestrator 在 Primary 上下文追问

### 推荐信息

| 字段 | 说明 |
|------|------|
| 典型配置 | 常用的 shape 组合与优先级 |
| 参考实现 | PyTorch / NumPy 参考代码 |
| 性能目标 | 目标吞吐量或延迟 |
| 动态轴说明 | 哪些维度在运行时变化 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 3. 技术约束（必须遵守）

本项目为 TileLang-Ascend（华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。

**生成 design.md 前必须执行强制检测**：三维 Kernel、threads 参数、动态循环边界、GPU 专用 API、GEMM 非整除、L0C 溢出等。

详细已知限制清单、强制检测规则、警告输出模板见 [references/ascend-constraints.md](references/ascend-constraints.md)。

---

## 4. 工作流程

### Phase 1：输入解析与算子特征分析

1. 解析算子名称与数学公式
2. 验证必需字段是否完整
3. 分析算子特征：
   - **计算类型判定**：
     - 纯 Vector（element-wise / reduction）→ 仅需 UB
     - 纯 Cube（仅 matmul）→ 需要 L1 + L0A/L0B/L0C
     - 混合（matmul + element-wise 后处理）→ 核间流水线，需要 CV 融合
     - **Host 预处理**：如 im2col 等 Python 侧预处理步骤，标明在 design 的 §1 和 §4 中
   - **复杂度级别**：
     - 单步（如 element-wise add）→ 无循环、单次搬运
     - 多步（如 softmax = max + sub + exp + sum + div）→ 多次计算、可能需要中间缓冲
     - 融合（如 flash attention = GEMM + softmax + GEMM）→ 核间协作、流水线
   - **动态 shape 判定**：是否存在运行时才确定的维度
4. **非整除场景预判**：检查输入 shape 是否可能不被 block size 整除。GEMM 类算子的 `M // block_M` 和 `N // block_N` 在 `M < block_M` 或 `N < block_N` 时产生零 block 或不完整 tile，必须在设计中明确处理策略（host 侧 zero-padding + crop，或 Kernel 内动态 block size）

### Phase 2：信息收集

**必须执行强制步骤 0：搜索本项目同类实现**。详细工具调用、信息收集步骤、禁止行为见 [references/info-sources.md](references/info-sources.md)。

### Phase 3：生成 design.md

基于 [examples/design-template.md](examples/design-template.md) 模板，填充所有章节：

1. 概述
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略（**必含：非整除时 padding+crop 策略，或 Kernel 内动态 block 方案**）
6. 循环与调度结构
7. 同步策略
8. CV 融合设计（**按模式分支**：Developer 默认消除 workspace/vid——`threads=2` + 片上直连，不产出 workspace 规格；仅 Expert/混合或复杂场景回退才设计 workspace + `workspace_idx`。详见 design-template.md §8.2）
9. 验证方案（Golden + **L0 门槛测试计划**；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-test-design`，不在此枚举）
10. 风险点与注意事项
11. 交付清单

### Phase 4：质量自检

按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。

### Phase 5：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 6：输出

将 `design.md` 输出到当前目录或用户指定路径。若文件已存在，询问是否覆盖。

---

## 5. 算子特征分析决策树

详细决策树（Ascend 版）、平台识别、API 映射规则、NPU 硬件约束（分形限制 / 对齐要求 / 存储大小上限）见 [references/decision-tree.md](references/decision-tree.md)。

---

## 6. 信息源优先级

信息源优先级表与冲突处理原则见 [references/info-sources.md](references/info-sources.md)。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户未提供数学公式 | 提问补全，给出常见算子公式作为参考 |
| 必需字段缺失 | 列出缺失项，逐一提问 |
| API 查询无结果 | 标注为「需扩展」，在风险点中说明 |
| 目标文件已存在 | 询问用户是否覆盖或另存 |
| 算子过于复杂 | 建议拆分为多个子算子分别设计 |

---

## 8. 完成报告

文档生成完成后，按 [examples/completion-report-template.md](examples/completion-report-template.md) 输出报告。

---

## 9. 生成算子

完成报告后，询问用户是否根据此报告生成对应算子代码。

---

## 子目录索引

- [references/ascend-constraints.md](references/ascend-constraints.md) — 技术约束清单、强制检测规则、警告输出格式
- [references/decision-tree.md](references/decision-tree.md) — 算子特征分析决策树、平台识别、NPU 硬件约束、API 映射规则
- [references/quality-checklist.md](references/quality-checklist.md) — 19 项质量自检清单
- [references/info-sources.md](references/info-sources.md) — 信息收集步骤、信息源优先级、冲突处理原则
- [examples/design-template.md](examples/design-template.md) — design.md 完整模板
- [examples/completion-report-template.md](examples/completion-report-template.md) — 完成报告输出模板
