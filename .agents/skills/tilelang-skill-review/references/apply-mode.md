# 应用模式工作流（`apply N,...`）

## 目录

- [1. 步骤 1：定位](#1-步骤-1定位)
- [2. 步骤 2：解析改动目标](#2-步骤-2解析改动目标)
- [3. 步骤 3：起草具体编辑](#3-步骤-3起草具体编辑)
- [4. 步骤 4：应用并更新状态](#4-步骤-4应用并更新状态)
- [5. 步骤 5：报告](#5-步骤-5报告)
- [6. 起草冲突处理](#6-起草冲突处理)

---

## 1. 步骤 1：定位

读取最近一份 `.agents/skill-journal/reviews/review-{date}.md`，找到编号 N 对应的聚合项。

## 2. 步骤 2：解析改动目标

从聚合项提取：
- `target_skill_path`（解析 `target_skill` 得到 skill 目录绝对路径）
- **`target_artifact`**（`skill` / `troubleshooting`，决定最终改哪个文件；分流准则的权威定义在 SKILL 子目录索引下的 `entry-schema.md §5`）
- `target_section`
- `proposed_change`
- 关联的所有原 entry id（`{op}/eN`）

### 解析 target_artifact 到目标文件

| `target_artifact` | 目标文件 | 文件不存在时 |
|-------------------|---------|-------------|
| `skill`（缺省） | `{target_skill_path}/SKILL.md` | 报错跳过（SKILL.md 必须存在） |
| `troubleshooting` | `{target_skill_path}/references/troubleshooting.md` | **自动创建**：先 `mkdir -p references/`，再用骨架模板创建 troubleshooting.md（含一级标题 + 编译/运行/精度三个章节），然后追加新条目 |

> 同一桶聚合项内**所有 entry 的 target_artifact 必须一致**。若不一致，按桶内多数派决定，并在 Apply 报告中提示分裂情况，让用户人工复核。

## 3. 步骤 3：起草具体编辑

### 分流 A：`target_artifact = skill` （改 SKILL.md）

读取 SKILL.md 的目标章节。基于 `proposed_change` + 原 entry 的 `evidence`，起草具体的 Edit 操作（旧文本 → 新文本）。

**起草原则**：
- 优先**最小修改**：能加一行表格条目就不改整段
- **保留原有措辞和示例**：除非是 `outdated_example` / `wrong_api_signature` 类型
- **遵循目标 skill 的写作风格**：表格用表格、列表用列表，参考前后文
- **不引入新章节**，除非聚合项明确指向"missing section"

### 分流 B：`target_artifact = troubleshooting` （改 references/troubleshooting.md）

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

## 4. 步骤 4：应用并更新状态

1. 用 Edit 工具修改目标文件（SKILL.md 或 troubleshooting.md，按 target_artifact 决定）
2. 在所有关联原 entry 文件中，把对应 entry 的 `**status**: pending` 改成 `**status**: applied`
3. 在评审快照里给该编号加上 ✅ 标记，并记录实际改动的文件路径

## 5. 步骤 5：报告

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

## 6. 起草冲突处理

| 情况 | 处理 |
|------|------|
| 目标章节找不到（artifact=skill） | 跳过该编号，在报告中标记 ⚠️ "section not found"，entry 维持 pending |
| troubleshooting.md 不存在（artifact=troubleshooting） | 按骨架模板自动创建（编译时错误 / 运行时错误 / 精度错误 三个一级子章节），再追加条目 |
| 桶内 target_artifact 不一致 | 按多数派决定，在报告中标记 ⚠️ "artifact split"，让用户人工复核少数派 entry |
| proposed_change 太模糊无法落地 | 跳过，标记 ⚠️ "ambiguous proposal"，建议用户手动改 |
| 同一编号的多个原 entry 提案彼此矛盾 | 取频次最高的，其它在报告中提示 |
| Edit 工具找不到 old_string（章节内容已被其它 apply 改过） | 重新读文件后重试一次，仍失败则跳过 |
| troubleshooting.md 已有相同症状条目 | 优先**合并**到既有条目（追加解决方案或触发条件），不重复加新编号 |
