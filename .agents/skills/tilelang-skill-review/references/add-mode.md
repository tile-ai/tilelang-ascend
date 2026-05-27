# 添加模式工作流（`add` / `add <text>`）

供**开发者主动**写反馈，区别于 op-generate §6 中 agent 自动反思的批量采集。

## 目录

- [1. 两种调用形式](#1-两种调用形式)
- [2. 公共前置](#2-公共前置)
- [3. 交互式（`add`）](#3-交互式add)
- [4. 快速式（`add <text>`）](#4-快速式add-text)
- [5. 落盘](#5-落盘)
- [6. 完成报告](#6-完成报告)
- [7. 注意事项](#7-注意事项)

---

## 1. 两种调用形式

| 形式 | 适用 |
|------|------|
| `add` | 交互式：逐个问 7 个字段，UX 友好 |
| `add <free text>` | 快速式：自由文本 → 自动补全字段 → 让开发者确认 |

## 2. 公共前置

1. `glob .agents/skills/**/SKILL.md` 建立 **target_skill 候选列表**
2. 计算今日文件路径：`.agents/skill-journal/manual-{YYYYMMDD}.md`
3. 如该文件已存在，读取最大 entry id（`^## Entry e(\d+)` 取最大值），新 entry 用 `e{max+1}`；不存在则从 `e1` 开始

## 3. 交互式（`add`）

按以下顺序用 AskUserQuestion **一次一个**地问，每个问题给出选项（target_skill / target_artifact / type / severity 用 multipleChoice，其它用 freeForm）：

| 序号 | 字段 | 形式 | 选项 |
|------|------|------|------|
| 1/8 | `target_skill` | multipleChoice + Other | 由 §2 步骤 1 的候选列表生成 |
| 2/8 | `target_artifact` | multipleChoice | `skill`（默认，改规则/流程） / `troubleshooting`（具体错误 → 解决方案，独立条目） |
| 3/8 | `target_section` | freeForm | `artifact=skill` 填 `§N.N` 或小节标题；`artifact=troubleshooting` 填具体故障条目标题 |
| 4/8 | `type` | multipleChoice | skill 类：`missing_constraint` / `wrong_api_signature` / `outdated_example` / `missing_api_doc` / `unclear_workflow` / `mode_misjudgment` / `pass_config_gap`；troubleshooting 类：`runtime_error_recipe` / `error_code_workaround` / `known_bug_avoidance`；其他：`other` |
| 5/8 | `severity` | multipleChoice | `high` / `medium` / `low` |
| 6/8 | `observation` | freeForm | 一句话描述 |
| 7/8 | `evidence` | freeForm (可空) | 报错 / 代码 / 文件引用 |
| 8/8 | `proposed_change` | freeForm | 具体改动提案 |

收齐后跳到 §5 落盘。

## 4. 快速式（`add <text>`）

### 步骤 1：基于自由文本自动补全

以下匹配规则**按顺序**应用：

| 信号 | 推断 |
|------|------|
| 文本含 `T.gemm` / `T.copy` / `T.tile.*` / `T.alloc_*` 等 API 名 | `target_skill = tilelang-custom-skill/tilelang-api-best-practices` |
| 文本含 "模式" / "Developer" / "Expert" / "pass_config" | `target_skill = tilelang-custom-skill/tilelang-expert-to-developer` |
| 文本含 "设计" / "design" / "选型" | `target_skill = tilelang-op-design` |
| 文本含 "生成代码" / "实现" / "kernel" | `target_skill = tilelang-op-generate` |
| 文本含 "调试" / "报错" / "error" | `target_skill = tilelang-custom-skill/tilelang-debug-helper` |
| 文本含 "参数顺序" / "签名" / "参数错" | `type = wrong_api_signature` |
| 文本含具体错误码（`error code 0x...` / `ErrCode F...` / `Errcode: F...`） | `target_artifact = troubleshooting`，`type = error_code_workaround` |
| 文本含具体错误信息（含 `Memory allocation failed` / `aicore error` / `shape mismatch` 等）+ "解决"/"绕过"/"workaround" | `target_artifact = troubleshooting`，`type = runtime_error_recipe` |
| 文本含 "已知 bug" / "规避" / "workaround" | `target_artifact = troubleshooting`，`type = known_bug_avoidance` |
| 其他默认 | `target_artifact = skill` |
| 文本含 "示例" / "例子" + 错/不对/过时 | `type = outdated_example` |
| 文本含 "没说" / "漏" / "缺" / "限制" | `type = missing_constraint` |
| 文本含 "工作流" / "步骤" / "顺序" | `type = unclear_workflow` |
| 无法确定 | `target_skill = Unknown` / `type = other`，置信度标低 |

未明确指出的 severity 默认 `medium`。`observation` 取原文本前 80 字符。`evidence` / `proposed_change` 留空待确认。

### 步骤 2：确认表格

输出：

```
基于你的描述，自动补全如下，请确认或修正：

| 字段             | 自动填值                                              | 信心 |
|------------------|------------------------------------------------------|------|
| target_skill     | tilelang-custom-skill/tilelang-api-best-practices    | high |
| target_section   | (空)                                                 | -    |
| type             | wrong_api_signature                                  | high |
| severity         | medium (默认)                                        | low  |
| observation      | T.gemm_v0 示例参数顺序写反了，C 和 B 颠倒            | -    |
| evidence         | (空)                                                 | -    |
| proposed_change  | (空)                                                 | -    |

回复:
  ok                                → 用上面这些直接落盘
  fix severity=high                 → 修单个字段
  fix evidence="..." section="..."  → 批量补充
  detail                            → 改成交互式逐项问
  cancel                            → 取消，不落盘
```

### 步骤 3：解析用户回复

| 用户输入 | 行为 |
|----------|------|
| `ok` | 用当前字段值落盘 |
| `fix <key>=<value> [<key>=<value>...]` | 更新对应字段，重新输出确认表格直到用户回复 `ok` |
| `detail` | 进入 §3 交互式流程（已填字段作为默认值） |
| `cancel` | 不落盘，结束 |
| 其它 | 视为 freeForm 补充，给出再次确认表格 |

## 5. 落盘

读取 `manual-{YYYYMMDD}.md`：

**情形 A：文件不存在** —— 新建：

```markdown
---
created: <当前 ISO8601>
source: developer
---

# Manual Skill Feedback - <YYYY-MM-DD>

## Entry e1
- **target_skill**: ...
- **target_section**: ...
- **type**: ...
- **severity**: ...
- **status**: pending
- **source**: developer

**Observation**:
...

**Evidence**:
...

**Proposed change**:
...

---
```

**情形 B：文件已存在** —— 在末尾追加新 Entry 块（编号 = 已有最大 id + 1），entry 内同样包含 `source: developer` 字段。**不**改 frontmatter。

## 6. 完成报告

```
✅ 已添加 entry e2 到 .agents/skill-journal/manual-20260511.md
   target: tilelang-custom-skill/tilelang-api-best-practices §T.gemm_v0 用法示例
   severity: high   type: wrong_api_signature   source: developer
   提示：下次运行 /tilelang-skill-review 会聚合到评审表
```

## 7. 注意事项

- **同一会话内可连续 `add`**：每条都追加到当日 manual 文件
- **不要在 add 阶段去 Edit 目标 SKILL.md**：add 只写 journal，apply 才改 skill
- **若用户输入与已有 entry 完全重复**（同 target_skill + target_artifact + target_section + type + observation），提示并询问是否仍要写入
