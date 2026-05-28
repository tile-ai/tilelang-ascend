# TileLang Pass Agents Guide

本文件为 AI Agent 在本代码仓库中进行 TileLang-Ascend Pass 开发、修改与重构提供统一工作流指导。

## 目标

本文件关注 Pass 相关任务的总体执行流程，覆盖以下场景：

- 新增 Pass
- 修改已有 Pass 行为
- 重构已有 Pass 实现
- 调整 Pass 在 Pipeline 中的位置或依赖关系
- 添加 Pass 设计文档、实现代码与验证

本文件不替代具体 skill，而是负责把多个 pass skill 串成一条完整流程。

Pass 在仓库中的实现通常分为两层：

- **C++ Pass**：位于 `src/transform/*.cc`，承载核心 IR 变换逻辑
- **Python 封装**：位于 `tilelang/transform/__init__.py`，通过 FFI 暴露给编译流水线

因此，Pass 任务通常不是单文件任务，而是围绕“分析 + 设计 + 实现 + 封装 + 接入编译流程 + 验证”的联动任务。

---

## 可用 Pass Skills

当前 Pass 相关 skill 分工如下：

| Skill | 职责 | 触发时机 |
|------|------|----------|
| `tilelang-pass-analyzer` | 分析单个 Pass 的功能、原理、差异、分类 | 用户询问某个 Pass 是做什么的，或要对比/分类查询时 |
| `tilelang-pass-workflow-analyzer` | 分析 Pass 工作流、顺序、依赖关系、定位新 Pass | 用户询问 Pass pipeline、依赖、顺序、插入位置时 |
| `tilelang-pass-design` | 生成 Pass 设计文档 | 用户要设计 Pass、写 pass-design.md、明确方案时 |
| `tilelang-pass-generate` | 根据设计文档生成 Pass **实现侧**最终代码（先输出 `pass-impl-skeleton.md` 框架文档，再落 C++/Python wrapper/pass_config/phase.py，最后做最小冒烟验证）。**不生成 UT/ST**，UT/ST 由独立的 Pass 测试生成 skill 处理 | 用户要根据设计文档开始实现 Pass、修改 Pass、重构 Pass 时 |

---

## 核心原则

严格遵循以下核心原则。

### 原则 1：先判断任务类型，再选择 skill

收到 Pass 任务后，先判断属于哪一类：

- **分析类**：理解已有 Pass 功能、原理、差异
- **工作流类**：理解 Pipeline 顺序、依赖、插入位置
- **设计类**：形成 Pass 方案文档
- **执行类**：真正修改代码，实现新增、修改、重构与验证

禁止在任务类型尚未明确时直接开始大范围读源码或改代码。

### 原则 2：先找到最直接的切入点，再逐步补上下文

Pass 任务应先从最具体、最直接的位置开始看：

- 指定了 Pass 名称 → 先围绕该 Pass
- 指定了文件 → 先围绕该文件
- 指定了行为问题 → 先围绕真正控制这个行为的 Pass
- 指定了 Pipeline 位置 → 先围绕相邻 Pass 和依赖链

禁止一开始就广泛扫描 `src/transform/` 全目录。

### 原则 3：新增、修改、重构分开处理

三类任务的策略不同，禁止混为一谈：

- **新增 Pass**：先定位 Phase 和位置，再设计，再实现，再接入 Pipeline
- **修改 Pass**：先确认当前行为和目标行为，再做最小修改
- **重构 Pass**：优先保持语义不变，先做结构整理，再做必要行为调整

### 原则 4：Pass 改动通常是多文件联动

Pass 开发常常不止修改一个 `.cc` 文件，还可能需要同步修改：

- `src/transform/<pass_name>.cc`
- `tilelang/transform/__init__.py`
- `tilelang/engine/phase.py`
- `tilelang/transform/pass_config.py`
- `testing/python/` 或相关示例测试
- pass 设计文档或参考文档

禁止只改 C++ 实现而忽略 Python 封装、phase 集成或验证入口。

### 原则 5：优先解决真正的问题，不做只遮住问题的改动

遇到 Pass 问题时，优先看清楚问题真正出在哪里：

- 控制行为的核心 Visit 方法
- 影响 IR 变换的属性、注解或中间信息
- Pipeline 中真正决定顺序的阶段
- 上下游 Pass 的数据依赖

禁止只靠增加特例分支把问题暂时规避，除非已经确认这就是最小且稳定的修复方式。

### 原则 6：每次改动后立刻做最小范围验证

完成第一处实质性修改后，必须立刻做一次最小范围验证：

- 受影响范围内最小的一条测试
- 最小范围的编译或构建检查
- 最基本的语法或导入检查
- 如果没有可执行检查，至少确认路径和引用没有问题

禁止连续做多轮补丁后再一起验证。

---

## 补充工程约束

以下约束适合在执行新增、修改、重构 Pass 时统一遵守。

### 约束 1：先理解编译流水线，再开始实现

- 在写或改 Pass 之前，先确认它位于哪个阶段，以及上下游 Pass 是谁
- 需要位置和依赖时，优先通过 `tilelang-pass-workflow-analyzer` 相关资料确认
- 禁止在未理解 Pipeline 顺序和依赖关系时盲目修改 IR

### 约束 2：优先参考同类 Pass 实现

- 在 `src/transform/` 中找到功能最相似的 Pass 作为模板
- 复用现有的 Visit 模式、注册方式、中间属性传递方式和代码风格
- 禁止脱离现有模式从空白重新发明实现框架

### 约束 3：尽量不修改 TVM 原生 Pass

- 优先通过新增或调整 TileLang 自己的 Pass 解决问题
- 如果必须修改 `tir.transform.*` 或其他 TVM 原生逻辑，需要明确说明原因、影响范围和替代方案为何不可行
- 禁止仅为局部便利直接改写 TVM 原生 Pass 行为

### 约束 4：新增 Pass 保持功能正交

- 每个新增 Pass 应只负责一种明确的 IR 变换或信息收集职责
- 新增前先确认现有 Pass 是否已经覆盖该功能，避免职责重叠
- 如需扩展已有功能，优先评估在现有 Pass 中做增量修改，而不是新增冗余 Pass

### 约束 5：测试分层要与改动范围匹配

- 每个新 Pass 至少应覆盖核心 IR 变换路径和关键边界条件
- 修改已有 Pass 时，至少补一条能覆盖目标行为变化的回归验证
- 对局部改动，先跑最小范围验证；对影响面较大的改动，再补更广的算子或集成验证
- 若变更影响多个算子或通用 pipeline，可进一步运行更广范围的验证，例如相关测试集或 `examples/bench_test.sh`

> 测试**编写**由独立的 Pass 测试生成 skill（待创建）负责。`tilelang-pass-generate` 只做实现侧代码 + 冒烟验证，并在收尾报告里把上述测试需求列为「测试待补清单」，作为给测试 skill 的输入。

---

## 总体执行流程

所有 Pass 任务统一按以下阶段执行。

### 阶段一：任务归类

先明确当前任务属于哪一种：

1. **新增 Pass**
2. **修改已有 Pass**
3. **重构已有 Pass**
4. **仅分析，不落代码**

分类规则：

- 用户要“加一个 Pass” → 新增 Pass
- 用户要“修改某个 Pass 行为” → 修改已有 Pass
- 用户要“整理实现结构，但尽量不改行为” → 重构已有 Pass
- 用户只问作用、顺序、依赖 → 分析类任务

### 阶段二：选择合适 skill

根据任务类型匹配 skill：

| 任务类型 | 优先 skill | 输出 |
|---------|-----------|------|
| Pass 功能分析 | `tilelang-pass-analyzer` | 功能分析报告 |
| Pass 工作流分析 | `tilelang-pass-workflow-analyzer` | 工作流/依赖分析报告 |
| Pass 方案设计 | `tilelang-pass-design` | `pass-design.md` |
| Pass 代码执行 | `tilelang-pass-generate` | `pass-impl-skeleton.md`（框架）+ 实现侧代码（C++/Python wrapper/pass_config/phase.py）+ 最小冒烟验证；**不含 UT/ST** |
| Pass 测试生成 | Pass 测试生成 skill（**待创建**） | UT / ST 测试代码（接续 `tilelang-pass-generate` 与设计文档 §5）|

若用户目标是“最终把 Pass 改好”，则整体流程通常为：

1. 先分析现状
2. 再确认工作流位置
3. 再产出设计文档
4. 最后执行实现与验证

### 阶段三：最小上下文收集

#### 新增 Pass

按以下顺序收集信息：

1. `tilelang-pass-workflow-analyzer` 的参考资料，明确 Phase 与插入位置
2. `tilelang-pass-analyzer` 的注册表和相似 Pass 资料
3. 已有设计文档或新生成的 `pass-design.md`
4. 最相近的一个或两个实现文件

#### 修改已有 Pass

按以下顺序收集信息：

1. 对应 Pass 的参考资料或设计文档
2. 当前负责核心逻辑的 `.cc` 文件
3. 对应 Python 封装和 `phase.py` 调用位置
4. 受影响的测试或示例

#### 重构已有 Pass

按以下顺序收集信息：

1. 当前 Pass 实现文件
2. 注册入口、Python 封装、phase 集成位置
3. 现有测试覆盖范围
4. 必要时再查看相邻 Pass 或公共工具类

### 阶段四：形成设计或简版执行计划

#### 需要完整设计文档的场景

- 新增 Pass
- 影响 Pipeline 位置或关键中间数据传递
- 改动涉及多个 Pass 协作
- 行为变化较大，无法通过小补丁表达

此时优先使用 `tilelang-pass-design` 生成 `pass-design.md`。

#### 可直接进入执行的场景

- 小范围 bugfix
- 已有清晰设计文档
- 仅重构代码结构，不改变对外行为

即使直接进入执行，也必须先写出简版执行计划，至少包含：

1. 目标 Pass
2. 目标行为
3. 预计修改文件
4. 最小验证方式

### 阶段五：执行实现

执行阶段由 `tilelang-pass-generate` 为主，必须覆盖以下检查项。

#### 新增 Pass 检查项（实现侧由 `tilelang-pass-generate` 完成）

1. 新建或补充 `src/transform/<pass_name>.cc`
2. 确认 `TVM_REGISTER_GLOBAL("tl.transform.<PassName>")`
3. 增加 `tilelang/transform/__init__.py` Python 封装
4. 必要时增加 `pass_config.py` 配置键
5. 将 Pass 接入 `tilelang/engine/phase.py`
6. 补充相关设计文档或参考资料
7. **测试由独立的 Pass 测试生成 skill（待创建）补充**，本步骤只在收尾报告里列出测试待补清单，不在 `tilelang-pass-generate` 内写测试

#### 修改已有 Pass 检查项

1. 先定位真正控制当前行为的方法或关键数据结构
2. 做最小行为修改
3. 检查是否影响中间属性、注解、循环结构或 buffer 访问
4. 检查 phase 中上下游 Pass 是否仍然兼容
5. **回归测试由 Pass 测试生成 skill 补充**，本阶段在报告里写明需要覆盖的目标行为差异

#### 重构已有 Pass 检查项

1. 默认保持注册名和外部调用入口不变
2. 优先抽辅助函数、辅助类、局部工具方法
3. 避免在同一次重构中混入大行为改动
4. 每次重构后立即做局部验证
5. 如修改接口，必须同步修改 Python 封装、phase 与文档

### 阶段六：验证与收尾

`tilelang-pass-generate` 阶段只做不依赖 UT/ST 的冒烟验证：

1. 导入冒烟（Python 封装 ↔ C++ 注册一致）
2. 跨文件命名 grep 一致（`PassName` 三处一致、配置键两处一致）
3. 已有最小 example 跑通，确认 pipeline 没有因为新 Pass 接入而崩
4. （可选）本机能跑则做最小构建冒烟

UT/ST 测试相关验证由后续独立的 Pass 测试生成 skill 处理。

收尾输出必须至少包含：

- 任务类型：新增 / 修改 / 重构
- 关键改动文件
- Pass 所在阶段与位置
- 是否影响上下游依赖
- 已完成的验证
- 剩余风险或待确认项

---

## 任务类型专用约束

### 新增 Pass 约束

- 必须先明确属于 Phase 1 还是 Phase 2
- 必须说明插入到哪个 Pass 前后
- 必须说明输入依赖和输出供给
- 必须先确认现有 Pass 未覆盖目标功能，避免职责重叠
- 未明确位置前，不得直接写实现

### 修改 Pass 约束

- 必须先说明当前行为与目标行为差异
- 必须先找到真正控制这个行为的代码位置
- 不得先广泛重写整个 pass

### 重构 Pass 约束

- 若用户未明确允许行为变化，默认语义保持不变
- 不得同时做结构重构和大范围功能重写
- 不得无验证地移动公共工具逻辑

---

## 常见工作流模板

### 模板一：新增 Pass

1. 使用 `tilelang-pass-workflow-analyzer` 确定位置
2. 使用 `tilelang-pass-analyzer` 找到相似 Pass
3. 使用 `tilelang-pass-design` 产出 `pass-design.md`
4. 使用 `tilelang-pass-generate` 完成实现与接入
5. 运行最小范围验证并总结风险

### 模板二：修改已有 Pass

1. 使用 `tilelang-pass-analyzer` 理解当前 Pass
2. 必要时用 `tilelang-pass-workflow-analyzer` 确认上下游依赖
3. 如改动较大，先补 `pass-design.md`
4. 使用 `tilelang-pass-generate` 做最小修改
5. 运行回归验证

### 模板三：重构已有 Pass

1. 先阅读当前实现和对外入口
2. 明确重构范围与语义保持边界
3. 如需要，补充简版设计说明
4. 使用 `tilelang-pass-generate` 执行重构
5. 分步验证，避免一次性大改

---

## 常见错误

- 只改 `src/transform/*.cc`，忘记 Python 封装或 `phase.py`
- 未确认依赖关系就调整 Pass 顺序
- 未补测试就修改 IR 变换行为
- 将“新增 Pass”和“修改已有 Pass”混成同一类任务
- 重构时顺手改语义，导致难以定位回归
- 一开始就扫大量源码，后面才发现其实参考资料已经足够

---

## 文件检查清单

执行 Pass 任务时，优先检查以下文件或目录：

- `src/transform/`
- `src/transform/common/`
- `tilelang/transform/__init__.py`
- `tilelang/transform/pass_config.py`
- `tilelang/engine/phase.py`
- `testing/python/`
- `.agents/skills/tilelang-pass-analyzer/references/`
- `.agents/skills/tilelang-pass-workflow-analyzer/references/`
- `.agents/skills/tilelang-pass-design/`
- `.agents/skills/tilelang-pass-generate/`
- `.agents/skills/tilelang-pass-generate/templates/pass-impl-skeleton-template.md`
- `.agents/skills/tilelang-pass-generate/references/code-generation-checklist.md`
- `.agents/skills/tilelang-pass-generate/references/integration-points.md`

---

## 使用示例

### 示例 1：用户要新增 Pass

用户问题：

> “我想加一个优化 L0C 布局的 Pass。”

推荐流程：

1. 先用 `tilelang-pass-workflow-analyzer` 确认应位于哪一阶段
2. 再用 `tilelang-pass-analyzer` 找现有相似 Pass
3. 再用 `tilelang-pass-design` 形成设计文档
4. 最后用 `tilelang-pass-generate` 落代码并验证

### 示例 2：用户要修改已有 Pass

用户问题：

> “AscendSyncInsert 现在插入的同步过多，帮我改一下。”

推荐流程：

1. 先分析 `AscendSyncInsert` 当前逻辑
2. 再确认它和 `AscendMemoryPlanning` 的依赖关系
3. 如策略变化较大，先形成设计说明
4. 再进行最小修改和回归验证

### 示例 3：用户要重构已有 Pass

用户问题：

> “把 CombineCV 里面的逻辑拆开整理一下，但先不要改行为。”

推荐流程：

1. 先确认外部接口和 phase 位置保持不变
2. 只做结构性拆分
3. 每次改动后做最小范围验证