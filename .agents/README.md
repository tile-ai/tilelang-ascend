# TileLang-Ascend OpenCode 工作流

通过 AI 代理与专家技能，自动完成华为昇腾 NPU 上的 TileLang 算子开发全流程：

```
环境预检 → 算子设计 → 代码实现+测试+精度调试（一站式） → （可选）性能调优
```

本仓库为 [OpenCode](https://opencode.ai) 预配置了项目规范（[AGENTS.md](../AGENTS.md)）、专家技能（Skills）和协作代理（Agents），开箱即用。

---

## 快速开始

在本仓库目录下启动 OpenCode，通过以下任一方式描述你的目标：

### 方式一：数学公式描述

```
开发一个 softmax 算子，公式是 softmax(x_i) = exp(x_i) / sum(exp(x_j))，
输入是 [B, N] 的 float16 张量，沿最后一维做归一化。
```

### 方式二：参考同类算子

```
参考 examples/normalization/example_layer_norm.py 的结构，
开发一个 RMSNorm 算子。
```

### 方式三：提供 design.md 草稿

```
请根据 ./examples/my_op/draft_design.md 中的设计草稿，
开发对应的 TileLang-Ascend 算子。
```

OpenCode 会自动加载项目规范，触发环境预检，编排 3 阶段流程。**唯一需要你确认的两件事**：编程模式（Developer / Expert / 混合）和精度通过后是否需要性能调优。

---

## 使用方式

### OpenCode

在 tilelang-ascend 仓库主目录启动 OpenCode：

#### 方式一：直接对 Orchestrator Agent 描述需求（推荐）

按 `Tab` 键切换到 `tilelang-op-orchestrator` 代理，输入开发任务：

```
开发一个 softmax 算子，公式是 exp(x_i) / sum(exp(x_j))，输入 [B, N] float16
```

Orchestrator 会自动驱动 3 阶段状态机：
**环境预检 → Stage 1 设计 → Stage 2 实现+测试+精度调试（一站式） → (Stage 3 性能调优)**

#### 方式二：直接调用 Skill（跳过编排）

如果只想用某个 skill 的能力，不走完整编排：

```
/tilelang-op-design          # 只生成 DESIGN.md
/tilelang-op-generate        # 只生成 kernel 代码
/tilelang-perf-optimization  # 只做性能优化分析
/tilelang-env-check          # 单独跑环境检查
```

绕过 Orchestrator 时，状态文件不会被写入。

#### 续跑与恢复

中断后再次对 Orchestrator 描述算子名即可：

```
继续 softmax 的开发
```

Orchestrator 从 `examples/softmax/.orchestrator_state.json` 自动续跑（环境预检不重复执行）。

---

### Claude Code（实验性）

Claude Code 使用不同的目录结构，需要先迁移项目配置：

```bash
# 1. 创建 Claude Code 目录结构
mkdir -p .claude/skills .claude/agents

# 2. 复制项目指令文件
cp AGENTS.md CLAUDE.md

# 3. 复制 Skills 和 Agents
cp -r .agents/skills/* .claude/skills/
cp -r .opencode/agents/* .claude/agents/
```

> **注意**：本仓库目前尚未配置 Claude Code 的 hook/lint 机制（pypto 仓库的 `.agents/settings.json` 在 tilelang-ascend 还不可用）。Claude Code 下的状态机维护与门禁校验都由 Orchestrator 自身的 Read/Write 操作完成。

启动后直接在对话中使用斜杠命令或自然语言：

```
/tilelang-op-design 开发一个 softmax 算子
```

或指定 Agent 启动：

```bash
claude --agent tilelang-op-orchestrator
```

---

### 使用建议

| 场景 | 推荐方式 |
|:---|:---|
| 完整算子开发（含精度+性能） | Orchestrator Agent |
| 仅需 design.md / kernel 代码 | 直接调用 `tilelang-op-design` / `tilelang-op-generate` |
| Pass 分析与开发 | 直接调用 `tilelang-pass-*` 系列（不走 Orchestrator） |
| 环境配置 / 调试 | 直接调用 `tilelang-env-check` / `tilelang-error-fixer` |
| 续跑中断的算子 | Orchestrator Agent，提算子名即可 |

---

## 核心架构

### AGENTS.md — 项目规范

[AGENTS.md](../AGENTS.md) 是 OpenCode 的项目级自定义指令文件，在本仓库使用 OpenCode 时自动加载生效。

该文件定义了：

- **Skills 索引**：按触发时机分类的技能清单
- **6 项核心原则**：不要凭记忆猜 API、从示例入手、注意双层修改、遵循硬件内存层级、优先复用定位问题而非重写、新增算子必须创建独立目录
- **Developer / Expert 模式对照**
- **分阶段开发指南**（需求分析 → 实现 → 测试 → 调试 → 测试编写）
- **算子开发编排体系**（与 Agents 的对接说明）

> 进一步了解：[OpenCode 自定义规则文档](https://opencode.ai/docs/zh-cn/rules/)

---

### Agents — 协作代理

代理定义在 [`.opencode/agents/`](../.opencode/agents/) 目录下，负责编排和隔离执行算子开发流程。

| 代理 | 模式 | 职责 |
|:---|:---|:---|
| [tilelang-op-orchestrator](../.opencode/agents/tilelang-op-orchestrator.md) | Primary | 3 阶段状态机、工件门禁、Subagent 调度、设计回退控制、环境预检 |
| [tilelang-op-analyst](../.opencode/agents/tilelang-op-analyst.md) | Subagent | Stage 1 算子设计（含需求理解、设计回退） |
| [tilelang-op-developer](../.opencode/agents/tilelang-op-developer.md) | Subagent | Stage 2 一站式：代码实现 + 测试 + 精度调试（通过 mode 区分 first_impl / retry_impl / precision_fix；含 `[DESIGN_ERROR]` 识别） |
| [tilelang-op-perf-tuner](../.opencode/agents/tilelang-op-perf-tuner.md) | Subagent | Stage 3 性能调优（可选） |

**Orchestrator 状态机**：

```
[env-check]                          一次性，env_check_passed=true 后跳过
    ↓
Stage 1: 算子设计 → @tilelang-op-analyst
    ↓                                 产物：DESIGN.md
Stage 2: 代码实现 + 测试 + 精度调试（一站式）→ @tilelang-op-developer
    │                                 每次 attempt 由 mode 区分：
    │                                 - first_impl   (attempt 1)
    │                                 - retry_impl   (运行失败后)
    │                                 - precision_fix (精度失败后)
    │                                 上限 5 次 attempt
    │
    ├─ PRECISION_FAIL ──→ 留在 Stage 2，下次 mode=precision_fix
    ├─ DESIGN_ERROR ────→ 设计回退到 Stage 1
    └─ PRECISION_PASS ──┐
                        ↓
             ┌── 询问用户是否需要性能调优 ──┐
             │                              │
             ↓ 不需要                       ↓ 需要 + 提供调优信息
          SUCCESS                  Stage 3: 性能调优 → @tilelang-op-perf-tuner
                                           产物：perf_tuning/
                                           ↓
                                       SUCCESS
```

> **关键特性**：
> - **Stage 2 一站式**：原"代码实现"和"精度修复"合并为单一阶段，Orchestrator 通过 `mode` 字段控制每次调度语义
> - **Stage 3 是可选阶段**，由用户在精度通过后主动确认
> - **设计回退机制**：Developer 在 Stage 2 中可标记 `[DESIGN_ERROR]` 触发回退到 Stage 1 重新设计（不设次数上限）
> - **环境预检**：一次性通过即可，续跑不重复
> - **状态文件**：仅 Orchestrator 可写 `.orchestrator_state.json`，Subagent 不得读写

详细规范见 [.opencode/agents/README.md](../.opencode/agents/README.md)。

---

### Skills — 专家技能

技能定义在 [`.agents/skills/`](skills/) 目录下，每个 skill 包含一个 `SKILL.md` 文件描述完整执行流程。

**调用方式**：

- **自动匹配** — 描述目标，OpenCode 根据 AGENTS.md 的 Skills 表自动选择
- **斜杠命令** — 明确指定：`/tilelang-op-design`
- **自然语言点名** — `请使用 tilelang-op-generate 技能`

> 进一步了解：[OpenCode Skills 文档](https://opencode.ai/docs/zh-cn/skills/)

---

## 技能详解

按场景快速定位：[算子开发](#算子开发与编排) · [API 参考](#api-参考与编程模式) · [环境与调试](#环境与调试) · [Pass 分析](#pass-分析与开发) · [元 Skill](#元-skill-管理与质量)

### 算子开发与编排

#### `tilelang-op-design` — 算子设计文档生成

**适用场景**：根据算子需求生成完整的 design.md（11 章节）

**你需要提供**：算子名称、数学公式、输入输出规格、编程模式偏好（Developer / Expert / 混合）

**你会得到**：design.md，包含编程模式选型、API 映射、内存层级规划、Tiling 策略、Loop 结构、同步策略、验证方案、风险点等

**强制前置**：必须先查 `examples/` 同类实现，执行技术约束检测（三维 Kernel / threads / 动态边界 / L0C 容量 / GEMM 非整除）

#### `tilelang-op-generate` — 算子代码生成

**适用场景**：根据 design.md 生成完整的 kernel 实现 + 测试入口

**你需要提供**：design.md 路径（或由 Orchestrator 自动传入）

**你会得到**：单一文件 `example_{op}.py`（含 `@tilelang.jit` kernel + 内嵌 PyTorch golden + **用户指定 shape** 的 test 用例 + main 块）

**关键约束**：优先复用 `examples/` 同类实现，冲突时以 examples > design.md > docs 为准

#### `tilelang-perf-optimization` — 性能分析与优化

**适用场景**：精度通过后的性能调优，包含瓶颈定位、迭代调优、精度复验

**核心 6 步**：精度校验 → 性能采集 → 算子类型判断 → 优化实施 → 精度验证 → 效果验证

**你会得到**：性能优化报告 + `perf_tuning/` 目录（含基线、备份、跨轮日志）

#### `tilelang-ascend-tile-api` — Tile API 端到端开发

**适用场景**：新增或封装 `T.tile.xxx` 小 API（如 `T.tile.exp`、`T.tile.add`）

**工作范围**：Python 前端（`ascend_tile.py`）→ C++ op（`src/op/`）→ lowering → Ascend C helper → codegen → 测试 + 文档

**注意**：这不是算子开发，而是**语言原语扩展**，会同时修改 Python 与 C++ 代码

---

### API 参考与编程模式

#### `tilelang-custom-skill/tilelang-api-best-practices` — API 速查与最佳实践

**适用场景**：写 kernel 时查阅 API 用法、最佳实践

**主要内容**：
- Kernel 定义、内存分配、数据搬运
- 计算原语：GEMM、归约、Tile 扩展操作
- 调度、同步与调试 API

#### `tilelang-custom-skill/tilelang-expert-to-developer` — 模式选择与转换

**适用场景**：判断使用 Developer / Expert / 混合模式，或在两种模式间转换实现

**你会得到**：模式对照表 + `pass_configs` 配置指南 + 转换示例

---

### 环境与调试

#### `tilelang-custom-skill/tilelang-env-check` — 环境检查

**适用场景**：开始算子开发前的环境验证（Orchestrator 会自动触发一次性预检）

**检查内容**：
- Python 包：torch / torch_npu (>= 2.6.0)
- CANN：ASCEND_HOME_PATH + 版本 (>= 8.3)
- 代码仓库：子模块完整性
- 编译产物：build 目录
- 环境变量：`source set_env.sh`

**自动修复**：子模块缺失、编译产物缺失、环境变量未设置（torch / CANN 版本问题需手动）

#### `tilelang-custom-skill/tilelang-debug-helper` — 调试辅助

**适用场景**：算子运行异常需要调试，配置 GDB 或查看生成的 Ascend C 代码

**你会得到**：GDB 配置 + `T.printf` / `T.dump_tensor` 使用方法 + build 目录定位

#### `tilelang-custom-skill/tilelang-error-fixer` — 错误诊断与修复

**适用场景**：编译错误、运行时错误的系统化排查

**工作流程**：错误分类 → 定位错误行 → 比对 API 文档与 examples → 给出修复方案

#### `tilelang-custom-skill/tilelang-submodule-pull` — 子模块拉取

**适用场景**：env-check 检测到子模块不完整时自动调用（也可手动）

**子模块**：tvm、cutlass 等

---

### Pass 分析与开发

> 与算子开发独立的工作流，主要用于编译器 Pass 维度的工作。

#### `tilelang-pass-analyzer` — Pass 功能分析

**适用场景**：理解某个 Pass 的功能、对比多个 Pass、查询 Pass 分类

**触发关键词**："XX pass 是干什么的"、"分析 XX pass"、"XX 和 YY pass 的区别"

#### `tilelang-pass-workflow-analyzer` — Pass 工作流分析

**适用场景**：查询 Pass 执行顺序、依赖关系、新 Pass 应该添加到哪里

**触发关键词**："pass 的工作流程"、"Pass 执行顺序"、"Pass 依赖关系"、"如何添加新 Pass"

#### `tilelang-pass-design` — Pass 设计

**适用场景**：设计一个新 Pass 的方案

**你会得到**：Pass 设计文档，包含目标、IR 变换规则、与其他 Pass 的依赖关系、测试方案

---

### 元 Skill（管理与质量）

#### `skill-creator` — Skill 创建

**适用场景**：根据需求创建一个新的 skill 目录与 SKILL.md

#### `skill-journal` — Skill 使用日志

**适用场景**：记录某个 skill 的使用情况、问题反馈、改进建议

#### `tilelang-skill-review` — Skill 质量评审

**适用场景**：审计某个 skill 是否遵循规范、发布前评估

#### `tilelang-custom-skill/tilelang-review-skill` — 代码评审

**适用场景**：对算子实现或 Pass 代码做评审

#### `tilelang-custom-skill/tilelang-github-operations` — GitHub 操作

**适用场景**：创建 PR、管理 issue、查询 commit 等

---

## 产物目录约定

新建算子在 `examples/{op}/` 下：

```
examples/{op}/
├── DESIGN.md                              # Stage 1
├── example_{op}.py                        # Stage 2-3 单一交付文件（@tilelang.jit kernel + 内嵌 golden + 用户指定 shape 的 test 用例 + main）
├── README.md                              # Stage 2（可选）
├── debug_log.md                           # 每次 Subagent 调度追加一条记录
├── perf_tuning/                           # Stage 3（仅当用户启用）
│   ├── baseline_iter{N}.json
│   ├── {op}_impl_iter{N}_before.py
│   └── perf_log.md
├── history_version/                       # 两类备份
│   ├── design_rev{N}.md                   # 设计回退备份
│   └── {op}_impl_s2_attempt{N}.py         # Stage 2 precision_fix 备份
└── .orchestrator_state.json               # 状态文件（仅 Orchestrator 写）
```

---

## 常见问题

<details>
<summary><b>AGENTS.md、Skills 和 Agents 有什么区别？</b></summary>

| 维度 | AGENTS.md | Skills | Agents |
|:---|:---|:---|:---|
| 作用 | 项目级自定义规范 | 特定任务的执行流程 | 编排和隔离执行复杂任务 |
| 加载方式 | 自动加载，对所有对话生效 | 按需加载，调用时才生效 | Orchestrator 主导，Subagent 被调度 |
| 内容 | 通用开发规范和原则 | 具体任务的步骤、工具、验证标准 | 状态机、工件契约、重试策略 |
| 执行模式 | 规则约束 | 直接执行 | Primary 编排 + Subagent 隔离执行 |

三者配合：AGENTS.md 定义"怎么做才对"，Skills 定义"怎么一步步做完"，Agents 定义"怎么编排和隔离执行"。

</details>

<details>
<summary><b>什么时候用 Orchestrator，什么时候直接用 Skill？</b></summary>

- **完整算子开发**：使用 `tilelang-op-orchestrator` agent，自动驱动 3 阶段
- **单步任务**：直接调用对应 Skill，如只需 design.md 就调用 `tilelang-op-design`
- **调试修复**：直接调用调试类 Skill，如 `tilelang-error-fixer`、`tilelang-debug-helper`
- **Pass 开发**：直接调用 `tilelang-pass-*` 系列（不走 Orchestrator）

</details>

<details>
<summary><b>Stage 3 性能调优为什么是可选的？</b></summary>

实际开发场景中，很多算子只需要功能正确，并不一定追求极致性能。Orchestrator 在精度通过后会**主动询问**：

```
softmax 算子精度已通过。是否需要进行性能调优？
```

- 你回 "不需要" → 直接 `SUCCESS`，流程结束
- 你回 "需要" → 继续询问性能调优必要信息（目标类型、目标数值、baseline、shape、噪声阈值、迭代上限），追加写入 DESIGN.md → 进入 Stage 3

这避免了为不需要性能的算子白白消耗 10 轮性能迭代。

</details>

<details>
<summary><b>什么是设计回退（DESIGN_ERROR）？</b></summary>

DESIGN.md 不视为**不可质疑的硬性约束**。实际开发中可能出现：

- 设计选用的 API 在 `tilelang/language/` 中无导出
- Tiling 策略导致 L0C 容量溢出（> 128KB）
- 内存层级路径不可实现
- 同步策略与编程模式冲突
- 循环边界依赖动态 tensor 值（违反 Ascend 限制）

Developer 在 Stage 2 中识别到这些情形时，会在输出加 `[DESIGN_ERROR]` 标记。Orchestrator 据此：

1. 备份当前 DESIGN.md 到 `history_version/design_rev{N}.md`
2. 把错误原因 + 历史回退备份传给 analyst
3. analyst 用 `revision` 模式生成新 design（必须有明确的差异化调整）
4. 重新进入 Stage 2 实现

**不设次数上限**，以最终精度通过为准。死循环风险由 Stage 2/3 自身的 5 次重试上限兜底。

</details>

<details>
<summary><b>状态文件 .orchestrator_state.json 是什么？能手动改吗？</b></summary>

每个算子在 `examples/{op}/.orchestrator_state.json` 维护一份独立状态：

```json
{
  "operator_name": "softmax",
  "env_check_passed": true,
  "current_stage": 2,
  "stage_status": {"1": "completed", "2": "in_progress"},
  "stage_retry_count": {"1": 0, "2": 1, "3": 0},
  "design_revision_count": 0,
  "perf_tuning_requested": null,
  "last_updated": "2026-05-19T..."
}
```

- **仅 Orchestrator 可写**，Subagent 不得读写
- 中断后续跑：再次对 Orchestrator 提算子名即可
- 想从头开始：删除该文件
- **不要手动修改字段**：会破坏门禁校验逻辑

本环境**无专用 `state_transition` 工具**，由 Orchestrator 通过 Read/Write 手动操作 JSON（详见 [orchestrator §状态写入接口](../.opencode/agents/tilelang-op-orchestrator.md)）。

</details>

<details>
<summary><b>其他 AI 工具兼容性</b></summary>

本项目主要支持 [OpenCode](https://opencode.ai)，理论上也可在 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)、Cursor、Codex 等工具中使用。

**Claude Code 目录结构映射**：

| 组件 | OpenCode | Claude Code |
|:---|:---|:---|
| 项目指令 | `AGENTS.md` | `CLAUDE.md` |
| Skills | `.agents/skills/` | `.claude/skills/` |
| Agents | `.opencode/agents/` | `.claude/agents/` |

**当前已知差异**：
- 本仓库目前**未配置 hook/plugin**（pypto 仓库的 `.opencode/plugins/` 在 tilelang-ascend 还没有），状态机维护、门禁校验、Stage 备份均由 Orchestrator 自身的 Read/Write 完成
- 对 Subagent 输出的三态判定（`[PRECISION_PASS]` / `[PRECISION_FAIL]` / `[DESIGN_ERROR]`）依赖 Orchestrator 主动检查标准输出

</details>

---

## 相关文档

- [AGENTS.md](../AGENTS.md) — 项目级开发规范
- [.opencode/agents/README.md](../.opencode/agents/README.md) — 代理体系技术细节
- [TileLang-Ascend Programming Guide](../docs/TileLang-Ascend%20Programming%20Guide.md) — 编程指南
- [.agents/skills/](skills/) — Skill 库源文件
