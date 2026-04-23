---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。触发：设计算子、生成 design.md、算子方案设计、新算子开发、算子实现方案。"
---

# TileLang-Ascend 算子设计文档生成

---

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-Ascend 算子设计文档（`design.md`），涵盖以下核心决策：

- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：Golden 函数与多级测试计划

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

**提问规则（必须严格遵守）**：
1. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
2. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
3. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
4. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

### 推荐信息

| 字段 | 说明 |
|------|------|
| 典型配置 | 常用的 shape 组合与优先级 |
| 参考实现 | PyTorch / NumPy 参考代码 |
| 性能目标 | 目标吞吐量或延迟 |
| 动态轴说明 | 哪些维度在运行时变化 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 3. 工作流程

### Phase 1：输入解析与算子特征分析

1. 解析算子名称与数学公式
2. 验证必需字段是否完整
3. 分析算子特征：
   - **计算类型判定**：
     - 纯 Vector（element-wise / reduction）→ 仅需 UB
     - 纯 Cube（含 matmul）→ 需要 L1 + L0A/L0B/L0C
     - 混合（matmul + element-wise 后处理）→ 核间流水线
   - **复杂度级别**：
     - 单步（如 element-wise add）→ 无循环、单次搬运
     - 多步（如 softmax = max + sub + exp + sum + div）→ 多次计算、可能需要中间缓冲
     - 融合（如 flash attention = GEMM + softmax + GEMM）→ 核间协作、流水线
   - **动态 shape 判定**：是否存在运行时才确定的维度

### Phase 2：信息收集

1. 查阅 `examples/` 中同类算子实现
2. 查阅 [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) 确认 API 可用性和用法
3. 查阅 [tilelang-expert-to-developer SKILL.md](../tilelang-expert-to-developer/SKILL.md) 确认编程模式和 pass_configs 配置
4. 判断 API 可用性时，必须同时核对公开导出路径（如 `tilelang/language/__init__.py`）与 lowering / codegen 实现（如 `src/op/`、`src/target/`），不能仅凭 `_ascend.py` / `_cuda.py` 文件名推断
5. 如有参考实现，分析其计算步骤

### Phase 3：生成 design.md

基于 `templates/design-template.md` 模板，填充所有章节：

1. 概述
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略
6. 循环与调度结构
7. 同步策略
8. 验证方案
9. 风险点与注意事项
10. 交付清单

### Phase 4：质量自检

按照 §5 中的自检清单逐项检查，确保文档质量。

### Phase 5：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 6：输出

将 `design.md` 输出到当前目录或用户指定路径。若文件已存在，询问是否覆盖。

---

## 4. 算子特征分析决策树

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
算子数学公式
├─ 含 matmul / @ / 矩阵乘
│   ├─ 仅 matmul → 纯 Cube
│   │   模式: Expert (手动管理 L0)
│   │   API: T.gemm_v0 / T.gemm_v1 / T.mma
│   │   内存: GM→L1→L0A/L0B→L0C→UB→GM
│   │
│   └─ matmul + element-wise 后处理 → 混合（融合算子）
│       模式: Expert + 核间流水线
│       API: T.gemm + T.tile.* / T.Parallel
│       内存: Cube 核 L0C→UB 交给 Vector 核处理
│       同步: T.set_cross_flag / T.wait_cross_flag
│
├─ 纯 element-wise（逐元素运算）
│   ├─ 单步运算 → Developer 模式优先
│   │   API: T.Parallel + 算术符号
│   │   内存: GM→UB→GM
│   │
│   └─ 多步运算（如 softmax、layer_norm）
│       ├─ 需要精细控制 buffer 分配和复用 → Expert 模式
│       │   API: T.reduce_* + T.tile.* + T.alloc_ub
│       │
│       └─ 无需精细内存控制 → Developer 模式
│           API: T.Parallel 内链式运算 + T.reduce_*
│
└─ 含归约（reduce_sum / reduce_max / reduce_min）
    两种模式均可使用 T.reduce_*，选择依据：
    ├─ 简单归约（如单步 reduce_sum）→ Developer 模式
    └─ 归约 + 多步后续计算 + 需精细 buffer 管理 → Expert 模式
    API: T.reduce_sum / T.reduce_max / T.reduce_min
    内存: GM→UB→GM
```

---

## 5. 质量自检清单

生成 `design.md` 后，逐项检查：

| # | 检查项 | 是否必须通过 |
|---|--------|-------------|
| 1 | **编程模式有明确结论和理由**：不是笼统的「视情况而定」 | ✅ 必须 |
| 2 | **API 映射具体到函数名和参数**：不是「使用相关 API」 | ✅ 必须 |
| 3 | **内存搬运路径完整**：从 GM 到计算再到 GM 的每一步都有说明 | ✅ 必须 |
| 4 | **Tiling 策略有约束分析**：解释了为什么选择该 Block/Tile 大小 | ⭕ 推荐 |
| 5 | **同步策略与编程模式匹配**：Developer 用自动同步、Expert 标明手动同步点 | ⭕ 推荐 |
| 6 | **验证方案覆盖典型配置**：不是「待补充」 | ⭕ 推荐 |
| 7 | **无占位符或模糊描述**：无 `{placeholder}`、TODO、「待补充」（已确认的除外） | ✅ 必须 |

**通过条件**：必须项全部通过，推荐项至少通过 2/3。

---

## 6. 信息源优先级

| 优先级 | 信息源 | 用途 |
|--------|--------|------|
| 1 | `docs/TileLang-Ascend Programming Guide.md` | 权威 API 说明和编程指南 |
| 2 | [tilelang-api-best-practices SKILL.md](../tilelang-api-best-practices/SKILL.md) | API 用法速查和最佳实践 |
| 3 | [tilelang-expert-to-developer SKILL.md](../tilelang-expert-to-developer/SKILL.md) | 编程模式选择和 pass_configs 配置 |
| 4 | `tilelang/language/__init__.py` + `tilelang/language/*.py` | 公开 API 导出关系与前端定义 |
| 5 | `src/op/` + `src/target/` | lowering 与后端实现状态 |
| 6 | `examples/` 示例代码 | 实际 API 用法和编程模式参考 |
| 7 | `testing/python/language/` | 边界用法和测试模式参考 |

**冲突处理**：当信息源之间矛盾时，以 `docs/` 为准。若 `docs/` 未覆盖，以 `tilelang/language/` 源码实际实现为准。

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

文档生成完成后，输出以下格式的报告：

```
## 设计文档生成报告

- 算子: {算子名称}
- 编程模式: {Developer / Expert / 混合}
- 计算类型: {纯 Vector / 纯 Cube / 混合}
- 输出路径: {文件路径}

### 自检结果
1. 编程模式选型: ✅ / ❌
2. API 映射具体性: ✅ / ❌
3. 内存搬运完整性: ✅ / ❌
4. Tiling 约束分析: ✅ / ❌
5. 同步策略匹配: ✅ / ❌
6. 验证方案覆盖: ✅ / ❌
7. 无占位符: ✅ / ❌

### 待确认项
- {列出需要用户进一步确认的内容}
```

## 9. 生成算子
完成报告后，询问用户是否根据此报告生成对应算子代码