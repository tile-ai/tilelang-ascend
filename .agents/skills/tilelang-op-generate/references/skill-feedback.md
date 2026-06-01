# Skill 反馈采集

## 目录

- [1. 触发权归属](#1-触发权归属)
- [2. 触发时机](#2-触发时机)
- [3. 步骤 1：枚举本次查阅过的所有 skill](#3-步骤-1枚举本次查阅过的所有-skill)
- [4. 步骤 2：针对每个 skill 反思（逐个过）](#4-步骤-2针对每个-skill-反思逐个过)
- [5. 步骤 3：写 journal 文件](#5-步骤-3写-journal-文件)
- [6. 自检](#6-自检)
- [7. 完成报告](#7-完成报告)

---

## 1. 触发权归属

> **本节的触发权归属取决于调用模式：**
>
> | 调用模式 | 由谁负责 skill 反馈采集 |
> |---------|----------------------|
> | 通过 `tilelang-op-orchestrator` 编排（推荐） | **由 orchestrator 在流程结束（SUCCESS / BLOCKED_*）时统一执行**，本 skill 不主动触发。详见 [.opencode/agents/tilelang-op-orchestrator.md §流程结束反思采集](../../../../.opencode/agents/tilelang-op-orchestrator.md) |
> | 单独调用本 skill（`/tilelang-op-generate`，跳过编排） | **由调用者在算子调试通过后手动触发**，按下文流程执行 |
>
> 为什么分开：orchestrator 模式下本 skill 在 Subagent 隔离上下文中被多次调度，单次调度结束 ≠ "全流程结束"。让本 skill 自己触发反思会导致 ① 每次调用都做一次 → 浪费；② 看不到其他 Subagent 用过什么 skill → 反思不全。因此 orchestrator 模式下交给 orchestrator 在全流程视野下统一采集。
>
> **若你（developer subagent）在 orchestrator 模式下被调度本 skill，直接跳过本节即可**——orchestrator 会在最终阶段做反思。但你**仍然应该在 `debug_log.md` 里如实记录本次调度的 changes / error_summary / next_hint**，这是 orchestrator 反思的核心数据源。

本节是 **skill 自适应更新机制**的采集端，**仅在单独调用模式下适用**。每次算子开发流程跑完后，必须把"哪些 skill 没讲清楚 / 被现实打脸 / 凭经验补的内容"写到 `.agents/skill-journal/`，由 `/tilelang-skill-review` 后续聚合评审。

**注意**：本节覆盖**整个开发链路**用到的所有 skill，不只是 op-design / op-generate。

## 2. 触发时机

满足以下任一条件后立即执行：
- 算子代码已生成且至少跑通过一次（即使精度不达标但能编译）
- 用户明确表示"本次开发结束"或"暂时到这"
- 调试中卡了很久（即使没跑通也要把过程中的发现写下来，type 标 `unclear_workflow`）

## 3. 步骤 1：枚举本次查阅过的所有 skill

回顾整个开发会话，列出**实际打开 / 引用 / 跳转过**的所有 skill 路径（相对 `.agents/skills/`），不只是 op-design 和 op-generate。常见包含：

| skill | 何时会被查阅 |
|-------|-------------|
| `tilelang-op-design` | 设计阶段全程 |
| `tilelang-op-generate` | 生成阶段全程（即本 skill 自身）|
| `tilelang-custom-skill/tilelang-api-best-practices` | 查 API 用法 / 参数 |
| `tilelang-custom-skill/tilelang-expert-to-developer` | 决定模式 / pass_configs |
| `tilelang-custom-skill/tilelang-debug-helper` | 调试报错 |
| `tilelang-custom-skill/tilelang-error-fixer` | 修编译/运行错误 |
| `tilelang-ascend-tile-api` | 查 T.tile.* 系列 |
| 其它 | 任何被 grep / read 过的 SKILL.md |

**规则**：宁可多列，不可漏列。漏列会导致那个 skill 的反馈永远收不上来。

## 4. 步骤 2：针对每个 skill 反思（逐个过）

对**每一个**在步骤 1 列出的 skill，按以下四问逐项检查：

1. 该 skill 讲清楚的事项里，**有哪些被现实打脸**？（如说"支持 X"实际不支持）
2. 我**凭经验补了**它没讲的什么内容？（如自己加了个对齐处理）
3. 它的**示例 / API 描述是否过时**？（如示例 shape 写错、API 签名变了）
4. 它的**工作流步骤是否漏了关键检查**？（如没说"先 grep examples/"）

每个 yes 的发现 = 一条 entry。**没有发现也要记录**（写空 entries），便于统计 skill 的"完美命中率"。

## 5. 步骤 3：写 journal 文件

按 `.agents/skill-journal/README.md` 的 schema，写到：

```
.agents/skill-journal/{op}-{YYYYMMDD-HHMMSS}.md
```

frontmatter 的 `skills_consulted` 字段必须包含步骤 1 的完整列表。

每条 entry 包含 `target_skill / target_artifact / target_section / type / severity / status:pending / observation / evidence / proposed_change`，字段含义见 [skill-journal README](../../skill-journal/README.md)。

**`target_artifact` 分流准则（取值 / 三段式触发条件 / type 默认映射 / 写入禁止清单）的权威定义**见 [tilelang-skill-review/references/entry-schema.md §5](../../tilelang-skill-review/references/entry-schema.md)，必须严格遵循。

## 6. 自检

写完 journal 后逐项检查：

| # | 检查项 | 必须通过 |
|---|--------|---------|
| 1 | `skills_consulted` 包含本次查阅的所有 skill | ✅ |
| 2 | 至少 50% 的 `skills_consulted` 在 entries 中至少出现一次（避免只反思 op-generate 自己）| ✅ |
| 3 | 每条 entry 的 `evidence` 都有具体报错/代码/文件引用 | ✅ |
| 4 | 没有重复 entry（同 `target_skill + target_artifact + target_section + type` 只出现一次） | ✅ |
| 5 | `severity=high` 的 entry 都附带了具体踩坑过程 | ⭕ |

## 7. 完成报告

写完 journal 后输出：

```
## Skill 反馈采集报告

- Journal 文件: .agents/skill-journal/{op}-{timestamp}.md
- 查阅的 skill 数量: N
- 写入 entries 数量: M
- 按 skill 分布:
  - tilelang-op-design: 3
  - tilelang-custom-skill/tilelang-api-best-practices: 2
  - ...
- 提示: 运行 /tilelang-skill-review 进入评审流程
```
