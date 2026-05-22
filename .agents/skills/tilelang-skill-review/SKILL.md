---
name: tilelang-skill-review
description: "聚合 .agents/skill-journal/ 中算子开发反馈，生成 skill 改进建议表，开发者命令行勾选后应用到对应 SKILL.md。覆盖所有 .agents/skills/ 下的 skill，不限于 op-design / op-generate。触发：skill review、skill 评审、检查 skill 反馈、应用 skill 改动、tilelang skill 改进、skill 反馈。"
---

# Skill Review — TileLang-Ascend Skill 改进评审

聚合算子开发过程中产生的反馈，生成可勾选的改进建议表，让开发者控制哪些落到 SKILL.md。

---

## 1. 适用场景

| 场景 | 输入 | 输出 |
|------|------|------|
| 周期性评审 | `.agents/skill-journal/*.md` 中所有 `status: pending` 的 entry | 分组表格 + 评审快照 |
| 应用改动 | `apply 1,3,5` | 修改对应 SKILL.md，更新 entry 状态 |
| 拒绝改动 | `reject 2,4` | 仅更新 entry 状态为 `rejected` |
| 状态查询 | `status` | 各 skill 的 pending / applied / rejected 计数 |

---

## 2. 核心约束

- **永远不直接修改 SKILL.md，除非用户用 `apply` 明确勾选**
- **评审范围覆盖全部 skill**：`glob .agents/skills/**/SKILL.md` 自动发现，不硬编码
- **rejected 的 entry 不删除**：保留供后续频次累计判断（"反复被拒但反复出现"是改 skill 的强信号）
- **所有更改最终落在文本文件**：可 `git diff` / `git checkout` 回退，不需要数据库

---

## 3. 输入参数解析

skill 调用时若带 args，按以下规则解析：

| args 形式 | 含义 |
|-----------|------|
| 空 / 无参数 | 进入**评审模式**：扫描、聚合、输出表格、写快照 |
| `apply N[,N...]` | **应用模式**：对评审表中编号 N 的行应用改动 |
| `apply all` | 应用全部 pending 改动（高风险，需二次确认）|
| `reject N[,N...]` | **拒绝模式**：仅标记为 rejected，不改 SKILL.md |
| `add` | **添加模式（交互式）**：开发者主动反馈，逐个问 7 个字段后写入 `manual-{date}.md` |
| `add <text>` | **添加模式（快速式）**：吃自由文本，自动补全字段后让开发者确认 |
| `status` | 列出每个 skill 的 pending/applied/rejected 计数 |

若 args 含义不明，进入评审模式并提示可用命令。

---

## 4. 评审模式工作流

### 步骤 1：发现所有 skill

```
glob .agents/skills/**/SKILL.md
```

建立 `skill_path -> SKILL.md 绝对路径` 映射，备后续 apply 用。

### 步骤 2：扫描 journal

```
glob .agents/skill-journal/*.md   # 排除 README.md 和 reviews/ 目录
```

读取每个 journal 文件，提取：
- frontmatter（`op / created / skills_consulted`）
- 所有 entry 块（按 `## Entry eN` 切分）
- 每个 entry 的字段（target_skill / target_artifact / target_section / type / severity / status / observation / evidence / proposed_change）。`target_artifact` 缺省视为 `skill`

**只处理 `status: pending` 的 entry**，其余跳过。

### 步骤 3：聚合 & 排序

按 `(target_skill, target_artifact, target_section, type)` 四元组分桶。**`target_artifact` 必须参与分桶**——同一 target_skill 下，改 SKILL.md 和改 troubleshooting.md 是不同的改动单位，不可合并。同一桶内的 entry 合并：
- 频次 = entry 数量
- 严重度 = 取最高（high > medium > low）
- 来源 = 同桶内含 developer 来源就标 `👤+🤖`（混合）；纯 developer 标 `👤`；纯 agent 标 `🤖`
- 证据 = 合并所有 evidence（用 `; ` 分隔，仅保留前 3 条）
- 提案 = 取频次最高的 proposed_change，其它列为补充

排序优先级：

```
score = severity_weight * frequency
severity_weight: high=3, medium=2, low=1
```

按 score 降序排列。

### 步骤 4：输出表格

按 `target_skill` 分组输出。每组内 score 降序。**来源**列用图标区分：`🤖` agent 自动反思 / `👤` 开发者手填 / `👤+🤖` 两者都有。

```
================================================================
Skill 改进建议评审 (生成时间: 2026-05-11 16:30)
================================================================

── tilelang-op-design ────────────────────────────────────────
| #  | 📄artifact | 章节         | 类型                | 严重 | 频次 | 来源 | 提案摘要                              | 证据 entry          |
|----|-----------|--------------|--------------------|------|------|------|---------------------------------------|---------------------|
| 1  | skill     | §2.5.1       | missing_constraint | high | 3    | 🤖    | 加 UB 容量限制 192KB                  | softmax/e1, gemm/e2 |
| 2  | skill     | §4.1         | mode_misjudgment   | med  | 2    | 🤖    | 修正"混合算子可用 Developer 单模式"    | flash/e1, conv/e3   |

── tilelang-op-generate ──────────────────────────────────────
| #  | 📄artifact     | 章节                       | 类型                  | 严重 | 频次 | 来源 | 提案摘要                              | 证据 entry          |
|----|---------------|----------------------------|----------------------|------|------|------|---------------------------------------|---------------------|
| 3  | skill         | §3 步骤 3                  | outdated_example     | med  | 1    | 🤖    | 修正 broadcast 索引示例 shape         | softmax/e3          |
| 4  | troubleshooting | 编译时错误.内存分配失败    | runtime_error_recipe | high | 2    | 🤖    | 追加：减小 block_M 到 64               | softmax/e1, gemm/e4 |

── tilelang-custom-skill/tilelang-api-best-practices ────────
| #  | 章节         | 类型                | 严重 | 频次 | 来源 | 提案摘要                              | 证据 entry          |
|----|--------------|--------------------|------|------|------|---------------------------------------|---------------------|
| 4  | T.gemm_v0    | wrong_api_signature| high | 1    | 👤    | 调换示例第 2、3 参数顺序              | manual-20260511/e1  |
| 5  | T.copy 索引  | outdated_example   | med  | 2    | 👤+🤖 | 修正越界示例                          | gemm/e3, manual-20260511/e2 |

================================================================

下一步:
  apply 1,3      → 应用第 1、3 条
  reject 2       → 拒绝第 2 条
  apply all      → 应用全部
  add            → 添加新反馈（交互式）
  add "..."      → 添加新反馈（快速式）
  status         → 查看 pending 计数

快照已写入: .agents/skill-journal/reviews/review-2026-05-11.md
```

### 步骤 5：写评审快照

```
.agents/skill-journal/reviews/review-{YYYY-MM-DD}.md
```

快照内容 = 上面表格 + 每条聚合项对应的 source entry 完整文本。**这一步必做**，让后续 apply 命令能按编号定位回原 entry。

如果同一天已经有 review 文件，追加到末尾（用 `## 评审会话 HH:MM` 二级标题分隔）。

---

## 5. 应用模式工作流（`apply N,...`）

### 步骤 1：定位

读取最近一份 `.agents/skill-journal/reviews/review-{date}.md`，找到编号 N 对应的聚合项。

### 步骤 2：解析改动目标

从聚合项提取：
- `target_skill_path`（解析 `target_skill` 得到 skill 目录绝对路径）
- **`target_artifact`**（`skill` / `troubleshooting`，决定最终改哪个文件）
- `target_section`
- `proposed_change`
- 关联的所有原 entry id（`{op}/eN`）

#### 解析 target_artifact 到目标文件

| `target_artifact` | 目标文件 | 文件不存在时 |
|-------------------|---------|-------------|
| `skill`（缺省） | `{target_skill_path}/SKILL.md` | 报错跳过（SKILL.md 必须存在） |
| `troubleshooting` | `{target_skill_path}/references/troubleshooting.md` | **自动创建**：先 `mkdir -p references/`，再用骨架模板创建 troubleshooting.md（含一级标题 + 编译/运行/精度三个章节），然后追加新条目 |

> 同一桶聚合项内**所有 entry 的 target_artifact 必须一致**。若不一致，按桶内多数派决定，并在 Apply 报告中提示分裂情况，让用户人工复核。

### 步骤 3：起草具体编辑

#### 分流 A：`target_artifact = skill` （改 SKILL.md）

读取 SKILL.md 的目标章节。基于 `proposed_change` + 原 entry 的 `evidence`，起草具体的 Edit 操作（旧文本 → 新文本）。

**起草原则**：
- 优先**最小修改**：能加一行表格条目就不改整段
- **保留原有措辞和示例**：除非是 `outdated_example` / `wrong_api_signature` 类型
- **遵循目标 skill 的写作风格**：表格用表格、列表用列表，参考前后文
- **不引入新章节**，除非聚合项明确指向"missing section"

#### 分流 B：`target_artifact = troubleshooting` （改 references/troubleshooting.md）

读取 troubleshooting.md（若不存在按上文骨架模板创建）。基于 `proposed_change` + `evidence` 追加或更新故障条目。

**条目格式**（统一规范）：

```markdown
### N. {简短症状标题}

**错误信息**:
\`\`\`
{从 evidence 提取的真实错误片段}
\`\`\`

**原因**: {一句话说明根因}

**解决方案**:
1. {步骤 1，含具体代码示例}
2. {步骤 2}

**触发条件**: {可选，什么情况下会出现}
**关联 skill / 章节**: {可选，回链到对应 SKILL.md 章节}
```

**起草原则**：
- 一个 entry → 一个独立条目（独立编号），不要把多个 entry 揉成一段
- 错误信息必须从 evidence 摘抄**原始片段**，不要润色或泛化
- 解决方案必须**可直接复制粘贴**的代码 / 命令
- 已存在相同症状条目（按错误信息片段匹配）→ 优先合并，不重复加条目

### 步骤 4：应用并更新状态

1. 用 Edit 工具修改目标文件（SKILL.md 或 troubleshooting.md，按 target_artifact 决定）
2. 在所有关联原 entry 文件中，把对应 entry 的 `**status**: pending` 改成 `**status**: applied`
3. 在评审快照里给该编号加上 ✅ 标记，并记录实际改动的文件路径

### 步骤 5：报告

```
## Apply 报告

应用项: 1, 3, 4
- [1] tilelang-op-design §2.5.1 (skill): ✅ 已添加 UB 容量限制行 (3 个 entry 标记 applied)
- [3] tilelang-op-generate §3 (skill): ✅ 已修正 broadcast 示例 (1 个 entry 标记 applied)
- [4] tilelang-op-generate / 编译时错误.内存分配失败 (troubleshooting): ✅ 已追加 case "block_M 过大 → 砍到 64" (2 个 entry 标记 applied)

跳过项: 0

涉及文件:
- .agents/skills/tilelang-op-design/SKILL.md
- .agents/skills/tilelang-op-generate/SKILL.md
- .agents/skills/tilelang-op-generate/references/troubleshooting.md

建议: git diff .agents/skills/ 确认改动是否符合预期
```

### 起草冲突处理

| 情况 | 处理 |
|------|------|
| 目标章节找不到（artifact=skill） | 跳过该编号，在报告中标记 ⚠️ "section not found"，entry 维持 pending |
| troubleshooting.md 不存在（artifact=troubleshooting） | 按骨架模板自动创建（编译时错误 / 运行时错误 / 精度错误 三个一级子章节），再追加条目 |
| 桶内 target_artifact 不一致 | 按多数派决定，在报告中标记 ⚠️ "artifact split"，让用户人工复核少数派 entry |
| proposed_change 太模糊无法落地 | 跳过，标记 ⚠️ "ambiguous proposal"，建议用户手动改 |
| 同一编号的多个原 entry 提案彼此矛盾 | 取频次最高的，其它在报告中提示 |
| Edit 工具找不到 old_string（章节内容已被其它 apply 改过） | 重新读文件后重试一次，仍失败则跳过 |
| troubleshooting.md 已有相同症状条目 | 优先**合并**到既有条目（追加解决方案或触发条件），不重复加新编号 |

---

## 6. 拒绝模式工作流（`reject N,...`）

简单直接：

1. 定位编号 N 对应的所有原 entry
2. 把每个 entry 的 `**status**: pending` 改成 `**status**: rejected`
3. 在评审快照里给该编号加上 ❌ 标记
4. 报告:
   ```
   ## Reject 报告
   拒绝项: 2, 4 (共 5 个 entry 标记 rejected)
   ```

**不**修改任何 SKILL.md。

---

## 7. 添加模式工作流（`add` / `add <text>`）

供**开发者主动**写反馈，区别于 op-generate §6 中 agent 自动反思的批量采集。两种调用形式：

| 形式 | 适用 |
|------|------|
| `add` | 交互式：逐个问 7 个字段，UX 友好 |
| `add <free text>` | 快速式：自由文本 → 自动补全字段 → 让开发者确认 |

### 7.1 公共前置

1. `glob .agents/skills/**/SKILL.md` 建立 **target_skill 候选列表**
2. 计算今日文件路径：`.agents/skill-journal/manual-{YYYYMMDD}.md`
3. 如该文件已存在，读取最大 entry id（`^## Entry e(\d+)` 取最大值），新 entry 用 `e{max+1}`；不存在则从 `e1` 开始

### 7.2 交互式（`add`）

按以下顺序用 AskUserQuestion **一次一个**地问，每个问题给出选项（target_skill / target_artifact / type / severity 用 multipleChoice，其它用 freeForm）：

| 序号 | 字段 | 形式 | 选项 |
|------|------|------|------|
| 1/8 | `target_skill` | multipleChoice + Other | 由 §7.1 步骤 1 的候选列表生成 |
| 2/8 | `target_artifact` | multipleChoice | `skill`（默认，改规则/流程） / `troubleshooting`（具体错误 → 解决方案，独立条目） |
| 3/8 | `target_section` | freeForm | `artifact=skill` 填 `§N.N` 或小节标题；`artifact=troubleshooting` 填具体故障条目标题 |
| 4/8 | `type` | multipleChoice | skill 类：`missing_constraint` / `wrong_api_signature` / `outdated_example` / `missing_api_doc` / `unclear_workflow` / `mode_misjudgment` / `pass_config_gap`；troubleshooting 类：`runtime_error_recipe` / `error_code_workaround` / `known_bug_avoidance`；其他：`other` |
| 5/8 | `severity` | multipleChoice | `high` / `medium` / `low` |
| 6/8 | `observation` | freeForm | 一句话描述 |
| 7/8 | `evidence` | freeForm (可空) | 报错 / 代码 / 文件引用 |
| 8/8 | `proposed_change` | freeForm | 具体改动提案 |

收齐后跳到 §7.4 落盘。

### 7.3 快速式（`add <text>`）

#### 步骤 1：基于自由文本自动补全

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

#### 步骤 2：确认表格

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

#### 步骤 3：解析用户回复

| 用户输入 | 行为 |
|----------|------|
| `ok` | 用当前字段值落盘 |
| `fix <key>=<value> [<key>=<value>...]` | 更新对应字段，重新输出确认表格直到用户回复 `ok` |
| `detail` | 进入 §7.2 交互式流程（已填字段作为默认值） |
| `cancel` | 不落盘，结束 |
| 其它 | 视为 freeForm 补充，给出再次确认表格 |

### 7.4 落盘

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

### 7.5 完成报告

```
✅ 已添加 entry e2 到 .agents/skill-journal/manual-20260511.md
   target: tilelang-custom-skill/tilelang-api-best-practices §T.gemm_v0 用法示例
   severity: high   type: wrong_api_signature   source: developer
   提示：下次运行 /tilelang-skill-review 会聚合到评审表
```

### 7.6 注意事项

- **同一会话内可连续 `add`**：每条都追加到当日 manual 文件
- **不要在 add 阶段去 Edit 目标 SKILL.md**：add 只写 journal，apply 才改 skill
- **若用户输入与已有 entry 完全重复**（同 target_skill + target_artifact + target_section + type + observation），提示并询问是否仍要写入

---

## 8. 状态查询模式（`status`）

```
glob .agents/skill-journal/*.md
```

统计每个 skill 的：
- pending 计数
- applied 计数
- rejected 计数

输出：

```
## Skill Journal 状态

| Skill                                                 | pending | applied | rejected |
|-------------------------------------------------------|---------|---------|----------|
| tilelang-op-design                                    | 5       | 12      | 3        |
| tilelang-op-generate                                  | 3       | 8       | 2        |
| tilelang-custom-skill/tilelang-api-best-practices     | 2       | 4       | 0        |
| ...                                                   |         |         |          |

最近一次评审: .agents/skill-journal/reviews/review-2026-05-09.md
```

---

## 9. 解析格式约定

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

解析时按 `^## Entry e\d+` 切块；每块内字段用 `^- \*\*(\w+)\*\*: (.+)$` 提取；文本段用 `^\*\*(Observation|Evidence|Proposed change)\*\*:$` 起始，到下一个 `**X**:` 或块结束为止。

**`source` 字段处理**：
- entry 无 `source` 字段 → 视为 `agent`
- 来自 `manual-{date}.md` 的 entry 应当带 `source: developer`；若缺失，按文件名补全

**容错**：
- 字段顺序可乱
- 文本段可空
- 多余空行无视
- frontmatter 用 `---` 包围，按 YAML 解析

遇到不合规的 entry：跳过 + 在评审表上方提示 `⚠️ skipped N malformed entries: {file:line}`。

---

## 10. 安全检查清单

每次 apply 后自检：

| # | 检查项 | 必须 |
|---|--------|------|
| 1 | 修改的文件路径都在 `.agents/skills/**/SKILL.md` 范围内 | ✅ |
| 2 | 没有改 README / 模板 / examples | ✅ |
| 3 | 每个被 apply 的编号都有对应 entry 状态变更 | ✅ |
| 4 | 评审快照中已标记 ✅ / ❌ | ✅ |
| 5 | 报告里列出了所有受影响的 SKILL.md 路径 | ✅ |

---

## 11. 错误处理

| 场景 | 处理 |
|------|------|
| skill-journal/ 不存在 | 评审/apply/reject/status 模式下提示 "尚无反馈，先跑一次 op-generate 流程"；add 模式下自动创建目录后继续 |
| 无 pending entry | 输出空表格 + "全部已处理"，提示运行 `status` 查看历史 |
| reviews/ 子目录不存在 | 自动创建后写入 |
| journal 文件 frontmatter 缺字段 | 跳过该文件，提示 |
| `apply N` 中 N 超出范围 | 报告该编号无效，继续处理其它编号 |
| 同一会话内多次 apply 同一编号 | 第二次起跳过（已标记 applied） |
| `add` 时无法 glob 出任何 skill | 提示 "未发现 .agents/skills/ 下任何 SKILL.md，请检查工作目录"，退出 |
| `add <text>` 自动补全置信度全部 low | 跳过确认表格，直接进入交互式（§7.2）以避免错误填充 |
| `add` 时检测到完全重复的 entry | 输出已有 entry id，询问 "仍要写入吗 (y/N)"，默认拒绝 |
