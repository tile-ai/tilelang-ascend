---
name: tilelang-pass-workflow-analyzer
description: TileLang Ascend Pass 工作流分析。用于理解 Pass 之间的关系、执行顺序、数据依赖，以及帮助定位新 Pass 应该在哪里添加。触发时机：用户询问 Pass pipeline、Pass 执行顺序、Pass 之间的关系、如何添加新 Pass、Pass 依赖关系、"Pass 工作流"、"Pass 顺序"、"在哪里加 Pass" 等关键词时。
---

# TileLang Pass 工作流分析 Skill

## ⭐ 核心约束：分阶段信息获取

**严格遵守以下信息获取优先级：**

| 查询阶段 | 信息来源 | 工具使用 |
|---------|---------|---------|
| **理解现有模型** | ✅ **先读对应 reference** | `read(references/*.md)` |
| **给出精确顺序或修改建议** | ✅ **再核对当前源码** | `read(tilelang/engine/phase.py)` |

**执行规则：**
```
用户首次提问 Pass 工作流
  ↓
1. 立即读取 reference 文件（pass-pipeline-overview.md / pass-dependency-graph.md）
  ↓
2. 若只解释概念，基于 reference 回答
  ↓
3. 若要给出精确顺序、插入点或改代码，核对当前 phase.py 与相关 pass 源码
```

**禁止行为：**
- ❌ 跳过 reference，重新发明一套依赖模型
- ❌ 在没有核对当前 `phase.py` 时给出可执行的插入点
- ❌ 把 reference 中的固定步骤号当成比源码更高的权威

---

## 核心目标

这个 skill 用于：
1. **串联 Pass 关系** - 展示 Pass 之间的依赖关系和数据流传递
2. **解释 Pipeline 架构** - 说明两阶段编译流程及其设计原理
3. **定位新 Pass** - 帮助用户确定新 Pass 应该添加在 Pipeline 的哪个位置

**与 tilelang-pass-analyzer 的区别：**
- `tilelang-pass-analyzer`：侧重于单个 Pass 的功能、原理、实现细节
- `tilelang-pass-workflow-analyzer`：侧重于 Pass 之间的关系、整体工作流、新 Pass 定位

## 工作流程

当用户提出与 Pass 工作流相关的询问时，按照以下步骤响应：

### Step 1: 识别用户意图

判断用户是想要：
- **了解整体架构** → 提供 Pipeline overview
- **理解 Pass 关系** → 展示依赖关系图
- **添加新 Pass** → 提供定位指南和建议
- **查询特定阶段** → 定位到具体阶段并解释

### Step 2: 选择合适的分析方式

根据意图，使用以下工具：

| 用户意图 | 参考文档 | 分析方式 |
|---------|---------|---------|
| 了解整体架构 | `references/pass-pipeline-overview.md` | 展示两阶段架构图，解释设计原理 |
| 理解 Pass 关系 | `references/pass-dependency-graph.md` | 展示依赖关系图，说明数据传递 |
| 添加新 Pass | `references/new-pass-placement-guide.md` | 分析 Pass 功能特征，给出定位建议 |
| 查询特定阶段 | 根据阶段名称查找 | 定位到具体 Pass，展示上下游关系 |

### Step 3: 提供分析报告

输出格式：

```markdown
# Pass 工作流分析报告

## 用户问题
[用户的具体问题]

## 分析结果

### [根据意图提供对应内容]

### 关键要点
- [要点 1]
- [要点 2]

### 建议操作
[如果用户要添加新 Pass，给出具体建议]
```

## 两阶段 Pipeline 架构概要

TileLang-Ascend 采用两阶段编译流程：

```
Python DSL (@tilelang.jit)
    ↓
[Phase 1: LowerAndLegalize] ← 前端标准化、Lowering
    - 目标：将高级 DSL 转换为标准化、可分析的 TIR
    - 特点：前端规范化与必要的 Ascend early lowering
    ↓
[Phase 2: OptimizeForTarget] ← 后端优化、平台特化
    - 目标：针对 Ascend 硬件特性的优化
    - 特点：硬件相关、性能导向
    ↓
CANN 工具链 → NPU 执行
```

**详细架构请查阅：** `references/pass-pipeline-overview.md`

## Pass 依赖关系核心概念

### 数据传递方式

Pass 之间既通过 `PrimFunc` attrs，也通过结构性 IR contract 传递数据：

| 数据名称 | Attr 键 | 产生 Pass | 消费 Pass |
|---------|---------|-----------|-----------|
| Buffer Scope | buffer annotations | `AscendInferBufferScope` | `CrossCorePipeline`, `CombineCV` |
| Initial shapes | `initial_buffer_shapes` | `BufferShapeCollector` | `Flatten2DBuffer` |
| Final logical shapes | `logic_buffer_shapes` | `Flatten2DBuffer` | `AscendMemoryPlanning` |
| Address Map | `address_map` | `AscendMemoryPlanning` | `AscendSyncInsert`, `AscendSyncInsertVS` |
| Size Map | `size_map` | `AscendMemoryPlanning` | `AscendSyncInsert`, `AscendSyncInsertVS` |

storage scope、resource scope、lowered calls、loop annotations 和 selected terminals 都是结构性
contract；即使 attrs 未变，后续 rewrite 仍可能使分析结果失效。

**详细依赖关系请查阅：** `references/pass-dependency-graph.md`

## 新 Pass 定位指南核心原则

### 定位原则

添加新 Pass 时，遵循以下原则：

1. **功能归属** - Pass 的功能属于哪个阶段？
   - 前端 Lowering → Phase 1 (LowerAndLegalize)
   - 后端优化 → Phase 2 (OptimizeForTarget)

2. **数据依赖** - Pass 需要哪些输入数据？
   - 识别上游 Pass 产生的 attrs
   - 确保上游 Pass 已经执行

3. **输出供给** - Pass 产生哪些输出数据？
   - 识别下游 Pass 需要的 attrs
   - 确保下游 Pass 可以访问

4. **目标范围** - Pass 对哪些 target 生效？
   - target specificity 决定 gate，不单独决定阶段
   - 阶段和精确位置由输入/输出 IR contract 决定

### 不可跨越的后端边界

具体场景不要在入口 skill 中硬编码。统一从
`references/new-pass-placement-guide.md` 推导，并核对当前 `phase.py`。特别是：

- buffer/address/access rewrite 必须在 MemoryPlanning 前完成；
- memory/pipeline-affecting call 必须在 SyncInsert 前生成并补齐 OperationConfig；所有
  hardware-call/scope rewrite 必须在 ResourceScopeVerify 前完成；
- managed Vector semantic rewrite 必须在 InstructionSelection 前完成；
- MaskLegalize 后不得再运行 TIR-transforming pass。

**详细定位指南请查阅：** `references/new-pass-placement-guide.md`

## 参考资料

当需要详细信息时，查阅以下文档：

- **Pipeline 架构详解**：`references/pass-pipeline-overview.md`（两阶段架构、每个阶段的 Pass 列表、设计原理）
- **依赖关系图**：`references/pass-dependency-graph.md`（数据流、依赖链、关键 Pass 之间的关系）
- **新 Pass 定位指南**：`references/new-pass-placement-guide.md`（定位原则、典型场景、案例分析）
- **已有 Pass 注册表**：`.agents/skills/tilelang-pass-analyzer/references/pass-registry-ascend.md`（所有 Pass 的详细信息）

## 输出报告模板

当用户询问 Pass 工作流时，使用以下模板生成报告：

```markdown
# Pass 工作流分析报告

## 问题类型
[整体架构 / Pass 关系 / 新 Pass 定位 / 特定阶段查询]

## 核心分析

### [根据问题类型提供对应分析]

#### [具体分析内容]
- [要点 1]
- [要点 2]
- [要点 3]

## 数据流分析

### 输入依赖
- [上游 Pass 1]: [提供的数据 1]
- [上游 Pass 2]: [提供的数据 2]

### 输出供给
- [下游 Pass 1]: [需要的数据 1]
- [下游 Pass 2]: [需要的数据 2]

## 定位建议

### 推荐位置
[具体位置描述，如 "Phase 2，在 Pass X 和 Pass Y 之间"]

### 理由
1. [理由 1]
2. [理由 2]

### 需要注意的依赖
- [依赖说明 1]
- [依赖说明 2]

## 实现建议

### 参考实现
可以参考以下 Pass 的实现：
- [Pass A]: [文件路径]
- [Pass B]: [文件路径]

### 关键步骤
1. C++ 实现：[说明]
2. Python Wrapper：[说明]
3. Pipeline 集成：[说明]

## 相关文件路径
- Pipeline 定义：`tilelang/engine/phase.py`
- Pass 注册：`src/transform/*.cc`
- Python API：`tilelang/transform/__init__.py`
```

## 使用示例

### 示例 1: 用户询问整体架构

**用户问题**: "整个 Pass pipeline 是怎么组织的？"

**响应策略**:
1. 读取 `references/pass-pipeline-overview.md`
2. 展示两阶段架构图
3. 解释 Phase 1 和 Phase 2 的设计原理
4. 提供关键 Pass 的简要说明

### 示例 2: 用户询问 Pass 关系

**用户问题**: "AscendMemoryPlanning 和 AscendSyncInsert 有什么关系？"

**响应策略**:
1. 读取 `references/pass-dependency-graph.md`
2. 展示数据流：`AscendMemoryPlanning` → `address_map/size_map` → `AscendSyncInsert` / `VS`
3. 解释依赖关系：两个 Sync pass 都需要 Memory Planning 的最终地址映射
4. 说明如果要在它们之间添加新 Pass，需要注意数据传递

### 示例 3: 用户想添加新 Pass

**用户问题**: "我想加一个 Pass 来优化 L0C buffer 的数据布局，应该加在哪里？"

**响应策略**:
1. 读取 `references/new-pass-placement-guide.md`
2. 分析 Pass 功能特征：
   - 功能：优化 buffer 数据布局
   - 依赖：需要知道 buffer scope (L0C)
   - 平台：Ascend 特定
3. 判断它会修改 buffer layout/address contract，因此要求在 MemoryPlanning 前
4. 读取当前 `phase.py`，再选满足输入依赖且不跨该边界的精确相邻 Pass
5. 给出实现建议、被消费/失效的 facts 和参考 Pass

## 注意事项

1. **区分单个 Pass 分析 vs 工作流分析**
   - 用户询问单个 Pass 功能 → 使用 `tilelang-pass-analyzer` skill
   - 用户询问 Pass 关系/工作流 → 使用此 skill

2. **优先展示关键信息**
   - 避免一次性输出过多细节
   - 先展示概要，再根据用户追问提供详细信息

3. **定位建议需明确**
   - 提供相邻锚点（如 "after X, before Y"），不要依赖易漂移的步骤号
   - 解释理由，不要只给出结论

4. **关联已有文档**
   - 充分利用 `tilelang-pass-analyzer` skill 的注册表文档
   - 当用户询问特定 Pass 时，引导其查看详细注册信息
