---
name: tilelang-pass-design
description: "根据 Pass 需求生成 TileLang-Ascend Pass 设计文档（pass-design.md）。涵盖 Pass 定位分析（Phase 1/2归属、Pipeline位置、依赖关系）、IR 变换设计、C++ 实现方案、测试方案、风险分析等。触发关键词：设计 Pass、Pass 设计文档、写 Pass、实现 Pass、添加 Pass、新建 Pass、Pass 设计、开发 Pass。"
---

# TileLang-Ascend Pass 设计文档生成 Skill

---

## 1. 目标

根据 Pass 需求信息，生成一份完整的 TileLang-Ascend Pass 设计文档（`pass-design.md`），涵盖以下核心决策：

- **Pass 定位**：Phase 1 / Phase 2 归属、Pipeline 具体位置
- **依赖分析**：上游 Pass 输入、下游 Pass 输出、数据流传递
- **IR 变换设计**：输入 IR 结构、输出 IR 结构、变换逻辑
- **实现方案**：C++ 类设计、核心方法、Python Wrapper、Pipeline 集成
- **测试方案**：功能测试、依赖测试、边界测试
- **风险分析**：已知约束、常见错误、与其他 Pass 的交互

---

## 2. 输入要求

### 必需信息

| 字段 | 说明 |
|------|------|
| Pass 名称 | 如 `BufferReuseOptimizer`、`L0CLayoutOptimization` |
| 功能描述 | Pass 要解决的问题和目标 |
| IR 变换类型 | 修改 IR / 收集信息 / 验证 IR |
| 平台范围 | 平台无关 / Ascend 特定 |

**提问规则（必须严格遵守）**：
1. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
2. **按表格顺序依次询问**：Pass 名称 → 功能描述 → IR 变换类型 → 平台范围
3. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个

### 推荐信息

| 字段 | 说明 |
|------|------|
| 参考 Pass | 可参考的现有 Pass 名称 |
| 输入数据依赖 | Pass 需要哪些 attrs（如 `buffer_shapess`、`address_map`） |
| 输出数据供给 | Pass 产生哪些 attrs 供下游使用 |
| 性能目标 | Pass 对编译时间或生成代码性能的影响 |

---

## 3. 工作流程

### Phase 1：需求澄清

1. 解析用户提供的 Pass 需求信息
2. 检查必需字段是否完整
3. **按顺序逐一提问补全缺失字段**（每次只问一个）

### Phase 2：信息收集

1. **查阅 Pass 定位参考资料**：
   - `tilelang-pass-workflow-analyzer/references/pass-pipeline-overview.md` - Pipeline 架构
   - `tilelang-pass-workflow-analyzer/references/new-pass-placement-guide.md` - 定位决策流程
   - `tilelang-pass-workflow-analyzer/references/pass-dependency-graph.md` - 依赖关系

2. **查阅 Pass 实现参考资料**：
   - `tilelang-pass-analyzer/references/pass-registry-ascend.md` - 类似 Pass 实现
   - `tilelang-pass-analyzer/references/ir-examples.md` - IR 变换示例格式

3. **查阅本 skill 的实现模式参考**：
   - `references/pass-impl-patterns.md` - C++ 类模板、注册方式

### Phase 3：Pass 定位分析

按照 `new-pass-placement-guide.md` 的决策流程：

**Step 1：分析 Pass 功能**
- Pass 的主要功能是什么？
- Pass 修改 IR 还是收集信息？
- Pass 是平台无关还是 Ascend 特定？
- Pass 属于哪类优化（Lowering / 合法化 / 内存 / 流水线 / 同步 / 其他）？

**Step 2：分析依赖关系**
- Pass 需要哪些输入数据（attrs）？
- 这些数据由哪个 Pass 产生？
- Pass 产生哪些输出数据？
- 这些数据由哪个 Pass 使用？

**Step 3：确定阶段归属**
- DSL Lowering / IR 合法化 → Phase 1
- 硬件优化 / 内存优化 / 同步优化 → Phase 2
- 输入数据来自 Phase 1 → 可在 Phase 1 或 Phase 2
- 输入数据来自 Phase 2 → 必须在 Phase 2

**Step 4：确定具体位置**
- 依赖优先：Pass 必须在产生其输入数据的 Pass 后执行
- 供给优先：Pass 必须在使用其输出数据的 Pass 前执行
- 功能相邻：将 Pass 放在功能相似的 Pass 附近

### Phase 4：生成 pass-design.md

基于 `templates/pass-design-template.md` 模板，填充所有章节：

1. 概述
2. Pass 定位
3. IR 变换设计
4. 实现方案
5. 测试方案
6. 风险点与注意事项
7. 交付清单

### Phase 5：质量自检

按照 §5 中的自检清单逐项检查，确保文档质量。

### Phase 6：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 7：输出与关联引导

1. 将 `pass-design.md` 输出到当前目录或用户指定路径
2. 若文件已存在，询问是否覆盖
3. **关联引导**：提示用户可使用相关 skill 查看详细信息

---

## 4. 定位决策速查表

### 阶段归属速查

| Pass 功能特征 | 阶段归属 | 理由 |
|-------------|---------|------|
| DSL Lowering | Phase 1 | 将高级 DSL 转换为底层 IR |
| IR 合法化 | Phase 1 | 确保 lowered IR 符合规范 |
| 信息收集 | Phase 1 或 Phase 2 | 根据收集时机决定 |
| 硬件优化 | Phase 2 | 针对 Ascend 硬件特性优化 |
| 内存优化 | Phase 2 | 利用硬件内存层级优化 |
| 流水线优化 | Phase 2 | 多核流水线规划 |
| 同步优化 | Phase 2 | 多核同步策略 |

### 典型定位场景

| Pass 功能特征 | 推荐位置 |
|-------------|---------|
| IR 合法化检查 | Phase 1 末尾 |
| 新 Tile 操作 Lowering | Phase 1，`LowerTileOp` 后 |
| 内存优化 | Phase 2，`AscendStorageRewrite` 后，`AscendMemoryPlanning` 前 |
| 同步优化 | Phase 2，`AscendMemoryPlanning` 后，`AscendSyncInsert` 前 |
| 流水线优化 | Phase 2，`PipelinePlanning` 后 |
| 信息收集 | 根据收集内容类型决定 |

### 关键数据依赖

| 输入数据 | 产生 Pass | 阶段 |
|---------|-----------|------|
| `buffer scope` | `AscendInferBufferScope` | Phase 1 |
| `buffer_shapess` | `CollectBufferShapes` | Phase 1 |
| `address_map` | `AscendMemoryPlanning` | Phase 2 |
| `size_map` | `AscendMemoryPlanning` | Phase 2 |

---

## 5. 质量自检清单

生成 `pass-design.md` 后，逐项检查：

| # | 检查项 | 是否必须通过 |
|---|--------|-------------|
| 1 | **阶段归属有明确结论和理由**：不是「视情况而定」 | ✅ 必须 |
| 2 | **Pipeline 位置具体到步骤**：如「Phase 2 步骤 13，在 X Pass 后」 | ✅ 必须 |
| 3 | **依赖关系完整**：上游 Pass + 下游 Pass + 数据传递 | ✅ 必须 |
| 4 | **IR 变换有输入/输出示例**：伪 IR 格式，标注变化点 | ✅ 必须 |
| 5 | **C++ 类名和核心方法明确**：具体到类名和关键方法名 | ✅ 必须 |
| 6 | **Pipeline 集成代码完整**：包含集成到 `phase.py` 的代码片段 | ✅ 必须 |
| 7 | **无占位符或模糊描述**：无 `{placeholder}`、TODO、「待补充」 | ✅ 必须 |

**通过条件**：所有必须项全部通过。

---

## 6. 信息源优先级

| 优先级 | 信息源 | 用途 |
|--------|--------|------|
| 1 | `tilelang-pass-workflow-analyzer/references/` | Pass 定位、依赖分析 |
| 2 | `tilelang-pass-analyzer/references/` | Pass 实现参考、IR 变换示例 |
| 3 | `references/pass-impl-patterns.md`（本 skill） | C++ 类模板、注册方式 |
| 4 | `tilelang/engine/phase.py` | Pipeline 集成参考 |
| 5 | `src/transform/*.cc` | 典型实现模式（仅在追问时使用） |

**冲突处理**：当信息源之间矛盾时，以 `pass-pipeline-overview.md` 和 `new-pass-placement-guide.md` 为准。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户未提供 Pass 名称 | 提问补全 |
| 必需字段缺失 | 列出缺失项，逐一提问 |
| 无法确定阶段归属 | 分析依赖关系后给出推荐方案，标注为「需确认」 |
| 目标文件已存在 | 询问用户是否覆盖或另存 |
| Pass 功能过于复杂 | 建议拆分为多个子 Pass 分别设计 |

---

## 8. 完成报告与关联引导

文档生成完成后，输出以下格式的报告：

```
## Pass 设计文档生成报告

- Pass 名称: {Pass 名称}
- 阶段归属: {Phase 1 / Phase 2}
- Pipeline 位置: {具体位置描述}
- IR 变换类型: {修改 IR / 收集信息 / 验证 IR}
- 输出路径: {文件路径}

### 自检结果
1. 阶段归属明确: ✅ / ❌
2. Pipeline 位置具体: ✅ / ❌
3. 依赖关系完整: ✅ / ❌
4. IR 变换示例清晰: ✅ / ❌
5. C++ 实现方案明确: ✅ / ❌
6. Pipeline 集成代码完整: ✅ / ❌
7. 无占位符: ✅ / ❌

### 待确认项
- {列出需要用户进一步确认的内容}

### 后续步骤建议
1. 查看详细 Pass 工作流：使用 **tilelang-pass-workflow-analyzer** skill
2. 查看类似 Pass 实现：使用 **tilelang-pass-analyzer** skill
3. 开始实现 Pass：使用 **tilelang-pass-generate** skill（待创建）
```

---

## 9. 参考资料

本 skill 依赖以下参考文件：

| 文件 | 路径 | 用途 |
|------|------|------|
| Pass 定位指南 | `tilelang-pass-workflow-analyzer/references/new-pass-placement-guide.md` | 定位决策流程 |
| Pipeline 架构 | `tilelang-pass-workflow-analyzer/references/pass-pipeline-overview.md` | 阶段划分、Pass 列表 |
| 依赖关系图 | `tilelang-pass-workflow-analyzer/references/pass-dependency-graph.md` | 数据流分析 |
| Pass 注册表 | `tilelang-pass-analyzer/references/pass-registry-ascend.md` | 类似 Pass 参考 |
| IR 示例 | `tilelang-pass-analyzer/references/ir-examples.md` | IR 变换格式 |
| 实现模式 | `references/pass-impl-patterns.md`（本 skill） | C++ 模板代码 |

---

## 10. 与其他 Skill 的关系

| Skill | 关系 | 使用时机 |
|------|------|----------|
| `tilelang-pass-workflow-analyzer` | 依赖 | 查阅 Pipeline 架构、Pass 定位、依赖关系 |
| `tilelang-pass-analyzer` | 依赖 | 查阅现有 Pass 实现细节、IR 变换示例 |
| `tilelang-pass-generate`（待创建） | 后续 | 根据设计文档生成 Pass 代码 |