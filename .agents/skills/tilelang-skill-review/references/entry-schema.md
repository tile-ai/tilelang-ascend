# Journal Entry 解析格式约定

## 目录

- [1. Entry 块标准结构](#1-entry-块标准结构)
- [2. 解析规则](#2-解析规则)
- [3. source 字段处理](#3-source-字段处理)
- [4. 容错](#4-容错)
- [5. target_artifact 分流（权威定义）](#5-target_artifact-分流权威定义)

---

## 1. Entry 块标准结构

journal 文件 entry 块的标准结构（与 `skill-journal/README.md` 一致）：

```
## Entry eN
- **target_skill**: <path>
- **target_artifact**: <skill|troubleshooting>   # 可选，缺省 skill
- **target_section**: <section>
- **type**: <type>
- **severity**: <severity>
- **status**: <status>
- **source**: <agent|developer>           # 可选，缺省 agent

**Observation**:
<text, 可多行>

**Evidence**:
<text, 可多行>

**Proposed change**:
<text, 可多行>

---
```

## 2. 解析规则

解析时按 `^## Entry e\d+` 切块；每块内字段用 `^- \*\*(\w+)\*\*: (.+)$` 提取；文本段用 `^\*\*(Observation|Evidence|Proposed change)\*\*:$` 起始，到下一个 `**X**:` 或块结束为止。

## 3. source 字段处理

- entry 无 `source` 字段 → 视为 `agent`
- 来自 `manual-{date}.md` 的 entry 应当带 `source: developer`；若缺失，按文件名补全

## 4. 容错

- 字段顺序可乱
- 文本段可空
- 多余空行无视
- frontmatter 用 `---` 包围，按 YAML 解析

遇到不合规的 entry：跳过 + 在评审表上方提示 `⚠️ skipped N malformed entries: {file:line}`。

## 5. target_artifact 分流（权威定义）

> 任何写 entry 的地方（orchestrator 反思采集 / op-generate skill 反馈 / skill-review apply 落盘）都必须按本节执行分流。**本文件是 target_artifact 规则的唯一权威源**，其它文档只能引用本节，不得复述。

### 取值与改动目标

| 取值 | 触发条件 | 改动目标 |
|------|---------|---------|
| `skill`（默认） | 反馈针对 skill 的**规则 / 流程 / 决策树 / 代码示例**层面 | `.agents/skills/{target_skill}/SKILL.md` |
| `troubleshooting` | 反馈是**具体错误 → 具体解决方案**，可独立成条目 | `.agents/skills/{target_skill}/references/troubleshooting.md` |

### `troubleshooting` 必须同时满足

① `observation` 含具体的错误信息 / 错误码 / 错误堆栈片段；② `proposed_change` 形如"症状 X → 改动 Y"；③ 改动可独立成"症状-原因-解决"三段式条目，不依赖 SKILL.md 上下文。

### type ↔ artifact 默认映射

仅作参考，最终以"同时满足条件"为准：

- `skill` 类：`missing_constraint` / `wrong_api_signature` / `outdated_example` / `missing_api_doc` / `unclear_workflow` / `mode_misjudgment` / `pass_config_gap`
- `troubleshooting` 类：`runtime_error_recipe` / `error_code_workaround` / `known_bug_avoidance`

### 写入与应用时的禁止清单

- ❌ 把所有 `target_skill` 全填成 `tilelang-op-generate`（懒得分类的常见错误）
- ❌ 把所有 entry 都填成 `target_artifact: skill`（懒得分流）——编译/运行错误带具体错误码的 entry 几乎都应该是 `troubleshooting`
- ❌ 漏写 evidence（无证据的提案会被 `/tilelang-skill-review` 直接拒）
- ❌ 在 journal 里直接写完整修订后的 SKILL.md 段落（review skill 在 apply 阶段才生成具体修改文本）

### 同桶聚合时的一致性要求

同一桶聚合项内**所有 entry 的 target_artifact 必须一致**。若不一致，按桶内多数派决定，并在 Apply 报告中提示分裂情况，让用户人工复核。
