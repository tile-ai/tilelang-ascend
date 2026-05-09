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
- 每个 entry 的字段（target_skill / target_section / type / severity / status / observation / evidence / proposed_change）

**只处理 `status: pending` 的 entry**，其余跳过。

### 步骤 3：聚合 & 排序

按 `(target_skill, target_section, type)` 三元组分桶。同一桶内的 entry 合并：
- 频次 = entry 数量
- 严重度 = 取最高（high > medium > low）
- 证据 = 合并所有 evidence（用 `; ` 分隔，仅保留前 3 条）
- 提案 = 取频次最高的 proposed_change，其它列为补充

排序优先级：

```
score = severity_weight * frequency
severity_weight: high=3, medium=2, low=1
```

按 score 降序排列。

### 步骤 4：输出表格

按 `target_skill` 分组输出。每组内 score 降序。

```
================================================================
Skill 改进建议评审 (生成时间: 2026-05-09 14:30)
================================================================

── tilelang-op-design ────────────────────────────────────────
| #  | 章节         | 类型                | 严重 | 频次 | 提案摘要                              | 证据 entry          |
|----|--------------|--------------------|------|------|---------------------------------------|---------------------|
| 1  | §2.5.1       | missing_constraint | high | 3    | 加 UB 容量限制 192KB                  | softmax/e1, gemm/e2 |
| 2  | §4.1         | mode_misjudgment   | med  | 2    | 修正"混合算子可用 Developer 单模式"    | flash/e1, conv/e3   |

── tilelang-op-generate ──────────────────────────────────────
| #  | 章节         | 类型                | 严重 | 频次 | 提案摘要                              | 证据 entry          |
|----|--------------|--------------------|------|------|---------------------------------------|---------------------|
| 3  | §3 步骤 3    | outdated_example   | med  | 1    | 修正 broadcast 索引示例 shape         | softmax/e3          |

── tilelang-custom-skill/tilelang-api-best-practices ────────
| #  | 章节         | 类型                | 严重 | 频次 | 提案摘要                              | 证据 entry          |
|----|--------------|--------------------|------|------|---------------------------------------|---------------------|
| 4  | T.gemm_v0    | wrong_api_signature| high | 1    | 修正参数顺序: (A,B,C,trans_A,init)    | gemm/e1             |

================================================================

下一步:
  apply 1,3      → 应用第 1、3 条
  reject 2       → 拒绝第 2 条
  apply all      → 应用全部
  status         → 查看 pending 计数

快照已写入: .agents/skill-journal/reviews/review-2026-05-09.md
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
- `target_skill_path`（→ 对应 SKILL.md 绝对路径）
- `target_section`
- `proposed_change`
- 关联的所有原 entry id（`{op}/eN`）

### 步骤 3：起草具体编辑

读取 SKILL.md 的目标章节。基于 `proposed_change` + 原 entry 的 `evidence`，起草具体的 Edit 操作（旧文本 → 新文本）。

**起草原则**：
- 优先**最小修改**：能加一行表格条目就不改整段
- **保留原有措辞和示例**：除非是 `outdated_example`/`wrong_api_signature` 类型
- **遵循目标 skill 的写作风格**：表格用表格、列表用列表，参考前后文
- **不引入新章节**，除非聚合项明确指向"missing section"

### 步骤 4：应用并更新状态

1. 用 Edit 工具修改 SKILL.md
2. 在所有关联原 entry 文件中，把对应 entry 的 `**status**: pending` 改成 `**status**: applied`
3. 在评审快照里给该编号加上 ✅ 标记

### 步骤 5：报告

```
## Apply 报告

应用项: 1, 3
- [1] tilelang-op-design §2.5.1: ✅ 已添加 UB 容量限制行 (3 个 entry 标记 applied)
- [3] tilelang-op-generate §3: ✅ 已修正 broadcast 示例 (1 个 entry 标记 applied)

跳过项: 0

涉及文件:
- .agents/skills/tilelang-op-design/SKILL.md
- .agents/skills/tilelang-op-generate/SKILL.md

建议: git diff .agents/skills/ 确认改动是否符合预期
```

### 起草冲突处理

| 情况 | 处理 |
|------|------|
| 目标章节找不到 | 跳过该编号，在报告中标记 ⚠️ "section not found"，entry 维持 pending |
| proposed_change 太模糊无法落地 | 跳过，标记 ⚠️ "ambiguous proposal"，建议用户手动改 |
| 同一编号的多个原 entry 提案彼此矛盾 | 取频次最高的，其它在报告中提示 |
| Edit 工具找不到 old_string（章节内容已被其它 apply 改过） | 重新读文件后重试一次，仍失败则跳过 |

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

## 7. 状态查询模式（`status`）

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

## 8. 解析格式约定

journal 文件 entry 块的标准结构（与 `skill-journal/README.md` 一致）：

```
## Entry eN
- **target_skill**: <path>
- **target_section**: <section>
- **type**: <type>
- **severity**: <severity>
- **status**: <status>

**Observation**:
<text, 可多行>

**Evidence**:
<text, 可多行>

**Proposed change**:
<text, 可多行>

---
```

解析时按 `^## Entry e\d+` 切块；每块内字段用 `^- \*\*(\w+)\*\*: (.+)$` 提取；文本段用 `^\*\*(Observation|Evidence|Proposed change)\*\*:$` 起始，到下一个 `**X**:` 或块结束为止。

**容错**：
- 字段顺序可乱
- 文本段可空
- 多余空行无视
- frontmatter 用 `---` 包围，按 YAML 解析

遇到不合规的 entry：跳过 + 在评审表上方提示 `⚠️ skipped N malformed entries: {file:line}`。

---

## 9. 安全检查清单

每次 apply 后自检：

| # | 检查项 | 必须 |
|---|--------|------|
| 1 | 修改的文件路径都在 `.agents/skills/**/SKILL.md` 范围内 | ✅ |
| 2 | 没有改 README / 模板 / examples | ✅ |
| 3 | 每个被 apply 的编号都有对应 entry 状态变更 | ✅ |
| 4 | 评审快照中已标记 ✅ / ❌ | ✅ |
| 5 | 报告里列出了所有受影响的 SKILL.md 路径 | ✅ |

---

## 10. 错误处理

| 场景 | 处理 |
|------|------|
| skill-journal/ 不存在 | 提示 "尚无反馈，先跑一次 op-generate 流程" 并退出 |
| 无 pending entry | 输出空表格 + "全部已处理"，提示运行 `status` 查看历史 |
| reviews/ 子目录不存在 | 自动创建后写入 |
| journal 文件 frontmatter 缺字段 | 跳过该文件，提示 |
| `apply N` 中 N 超出范围 | 报告该编号无效，继续处理其它编号 |
| 同一会话内多次 apply 同一编号 | 第二次起跳过（已标记 applied） |
