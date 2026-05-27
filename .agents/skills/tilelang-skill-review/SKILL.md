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

> **entry 解析细节**（字段正则、source 字段处理、容错规则）查 [references/entry-schema.md](references/entry-schema.md)。

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

### 步骤 4：输出表格 & 写评审快照

按 `target_skill` 分组输出。完整表格格式范例、快照写入规则见 [examples/review-output.md](examples/review-output.md)。

---

## 5. 应用模式工作流（`apply N,...`）

5 步流程：定位 → 解析改动目标（artifact=skill 改 SKILL.md / artifact=troubleshooting 改 references/troubleshooting.md）→ 起草编辑 → 应用并更新 entry 状态 → 报告。

> **应用 apply 命令时**查 [references/apply-mode.md](references/apply-mode.md)：含目标文件映射表、SKILL.md / troubleshooting.md 分流起草原则、troubleshooting 条目格式、Apply 报告范例、起草冲突 7 类处理表。

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

供**开发者主动**写反馈，区别于 op-generate §6 中 agent 自动反思的批量采集。

- `add`：交互式，逐个问 7 个字段
- `add <text>`：快速式，吃自由文本，自动补全后确认

> **触发 add 命令时**查 [references/add-mode.md](references/add-mode.md)：含公共前置、交互式 8 字段问答表、快速式信号匹配表、确认表格格式、落盘 schema、完成报告。

---

## 8. 状态查询模式（`status`）

```
glob .agents/skill-journal/*.md
```

统计每个 skill 的 pending / applied / rejected 计数。

输出范例见 [examples/review-output.md §3](examples/review-output.md)。

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
| skill-journal/ 不存在 | 评审/apply/reject/status 模式下提示 "尚无反馈，先跑一次 op-generate 流程"；add 模式下自动创建目录后继续 |
| 无 pending entry | 输出空表格 + "全部已处理"，提示运行 `status` 查看历史 |
| reviews/ 子目录不存在 | 自动创建后写入 |
| journal 文件 frontmatter 缺字段 | 跳过该文件，提示 |
| `apply N` 中 N 超出范围 | 报告该编号无效，继续处理其它编号 |
| 同一会话内多次 apply 同一编号 | 第二次起跳过（已标记 applied） |
| `add` 时无法 glob 出任何 skill | 提示 "未发现 .agents/skills/ 下任何 SKILL.md，请检查工作目录"，退出 |
| `add <text>` 自动补全置信度全部 low | 跳过确认表格，直接进入交互式（§7）以避免错误填充 |
| `add` 时检测到完全重复的 entry | 输出已有 entry id，询问 "仍要写入吗 (y/N)"，默认拒绝 |

---

## 子目录索引

- [references/apply-mode.md](references/apply-mode.md) — apply 命令的 5 步流程详情（artifact 分流、起草原则、冲突处理）
- [references/add-mode.md](references/add-mode.md) — add 命令两种调用形式的完整规则
- [references/entry-schema.md](references/entry-schema.md) — journal entry 结构、解析正则、source 字段、容错
- [examples/review-output.md](examples/review-output.md) — 评审表格 / 评审快照 / 状态查询的输出范例
