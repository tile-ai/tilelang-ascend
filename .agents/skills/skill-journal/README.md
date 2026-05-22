# Skill Journal

存储每次算子开发过程中收集的 **skill 改进反馈**。

整体流程：

```
op-generate 完成后 ──► 写 journal/{op}-{timestamp}.md (status=pending 的 entries)
        │
        ▼
开发者运行 /tilelang-skill-review
        │
        ▼
聚合所有 pending entries ──► 输出表格 + reviews/review-{date}.md
        │
        ▼
开发者命令行勾选: apply 1,3,5  /  reject 2,4
        │
        ▼
review skill 修改对应 SKILL.md，更新 entry.status
```

## 目录结构

```
.agents/skill-journal/
├── README.md                    # 本文件
├── {op}-{timestamp}.md          # 每次算子开发产生一个 journal 文件
└── reviews/                     # 评审快照（自动创建）
    └── review-{date}.md
```

## Journal 文件格式

每个 journal 文件由 frontmatter + 多个 entry 组成。entry 状态在 entry 内部维护，**文件本身不移动**。

```markdown
---
op: softmax
created: 2026-05-09T14:30:00
skills_consulted:
  - tilelang-op-design
  - tilelang-op-generate
  - tilelang-custom-skill/tilelang-api-best-practices
  - tilelang-custom-skill/tilelang-expert-to-developer
---

# Skill Feedback - softmax

## Entry e1
- **target_skill**: tilelang-op-design
- **target_artifact**: skill
- **target_section**: §2.5.1 已知限制
- **type**: missing_constraint
- **severity**: high
- **status**: pending

**Observation**:
设计时没说 T.alloc_ub 的总大小上限，跑 example 才发现 UB 只有 192KB。

**Evidence**:
报错 `Memory allocation failed required: 245760`。把 block_M 从 128 砍到 64 后通过。

**Proposed change**:
在 §2.5.1 表格新增一行 "UB 容量限制 192KB / 单 block buffer 总和不可超" 并给出修正方法。

---

## Entry e2
- **target_skill**: tilelang-custom-skill/tilelang-api-best-practices
- **target_section**: T.tile.broadcast 用法
- **type**: outdated_example
- **severity**: medium
- **status**: pending

...
```

### 开发者手填的 manual 文件（`manual-{YYYYMMDD}.md`）

由 `/tilelang-skill-review add` 写入，同一天追加到同一文件，frontmatter 简化：

```markdown
---
created: 2026-05-11T16:23:00
source: developer
---

# Manual Skill Feedback - 2026-05-11

## Entry e1
- **target_skill**: tilelang-custom-skill/tilelang-api-best-practices
- **target_artifact**: skill
- **target_section**: T.gemm_v0 用法示例
- **type**: wrong_api_signature
- **severity**: high
- **status**: pending
- **source**: developer

**Observation**: ...
**Evidence**: ...
**Proposed change**: ...

---
```

## 字段说明

### Frontmatter

| 字段 | 说明 |
|------|------|
| `op` | 算子名（小写下划线，如 `softmax`、`flash_attention`）|
| `created` | ISO8601 时间戳 |
| `skills_consulted` | 本次开发**实际查阅过**的所有 skill 路径列表，相对 `.agents/skills/` |

### Entry 字段

| 字段 | 说明 |
|------|------|
| `target_skill` | 目标 skill 路径（相对 `.agents/skills/`），可指向任意 skill |
| `target_artifact` | **`skill` / `troubleshooting`**，决定本条 entry 在 apply 时改哪个文件（详见下方「Artifact 分流准则」），缺省 `skill` |
| `target_section` | 目标章节标识：`target_artifact=skill` 用 `§N.N` 或具体小节标题；`target_artifact=troubleshooting` 用具体故障条目标题（如 `"编译时错误.内存分配失败"`） |
| `type` | 见下方"类型词表" |
| `severity` | `high` / `medium` / `low` |
| `status` | `pending` / `applied` / `rejected`，由 review skill 维护 |
| `source` | `agent` / `developer`，缺省 `agent`。`developer` 表示由 `/tilelang-skill-review add` 添加的人工反馈 |
| `observation` | 一句话描述发现的问题 |
| `evidence` | 报错信息 / 实际代码 / 调试过程的具体证据 |
| `proposed_change` | 具体的改动提案（一两句话），让 reviewer 能直接判断是否值得改 |

### 类型词表

| type | 默认 artifact | 含义 | 例子 |
|------|--------------|------|------|
| `missing_constraint` | `skill` | skill 没讲到的硬约束 | UB 容量、对齐要求、不支持的形参组合 |
| `wrong_api_signature` | `skill` | API 签名/参数描述与实际不符 | `T.gemm_v0` 参数顺序错 |
| `outdated_example` | `skill` | 示例代码已经跑不通或不是最佳写法 | broadcast 索引示例 shape 写错 |
| `missing_api_doc` | `skill` | 完全没提到的 API | `T.tile.exp` 没收录 |
| `unclear_workflow` | `skill` | 工作流步骤模糊或漏检查 | 没说"先搜 examples/" 的强制顺序 |
| `mode_misjudgment` | `skill` | 编程模式选型描述误导 | 把混合算子说成可用 Developer 单模式 |
| `pass_config_gap` | `skill` | pass_configs 配置说明不全 | 没提 `AUTO_CV_SYNC` 必须配 `AUTO_CV_COMBINE` |
| `runtime_error_recipe` | `troubleshooting` | 具体错误信息有可复用的解决方案 | `Memory allocation failed required: 245760` → 减小 block_M |
| `error_code_workaround` | `troubleshooting` | 具体错误码 / 异常的规避方法 | `aicore error code 0x80004001` → 检查 tile 对齐 |
| `known_bug_avoidance` | `troubleshooting` | 已知框架 bug 的规避手法 | view 后 `inplace=True` 精度异常 → 改 `inplace=False` |
| `other` | `skill` | 不属于以上 | |

### Artifact 分流准则

每条 entry 必须显式或隐式指定 `target_artifact`：

| 取值 | 改动目标文件 | 内容形式 |
|------|-------------|---------|
| `skill`（默认） | `.agents/skills/{target_skill}/SKILL.md` | 规则 / 流程 / 决策树 / 代码示例修订 |
| `troubleshooting` | `.agents/skills/{target_skill}/references/troubleshooting.md`（不存在则自动创建） | "症状 → 原因 → 解决"型独立故障条目 |

**判定方法**（同时满足才填 `troubleshooting`）：

1. `observation` 含**具体的错误信息 / 错误码 / 错误堆栈片段**
2. `proposed_change` 形如"症状 X → 改动 Y"，可以独立成条目
3. 改动**不依赖 SKILL.md 上下文**就能被未来读者查阅

否则一律填 `skill`（默认值）。

**简化判别**：能不能形成"症状—原因—解决"三段式独立条目？能 → `troubleshooting`；不能（修的是规则、原则、流程描述、代码示例） → `skill`。

### 严重度判定

| severity | 标准 |
|----------|------|
| `high` | 不改会导致后续算子开发踩同样的坑，或编译/运行失败 |
| `medium` | 不改会让生成的代码不是最佳实践 |
| `low` | 措辞优化、补充示例 |

## 命名规范

journal 文件按来源区分命名模式：

| 模式 | 来源 | 说明 |
|------|------|------|
| `{op}-{YYYYMMDD-HHMMSS}.md` | agent（op-generate §6 自动反思） | 每次算子开发一个文件，时间戳精确到秒避免冲突，frontmatter 含 `op` 和 `skills_consulted` |
| `manual-{YYYYMMDD}.md` | developer（`/tilelang-skill-review add`） | **同一天追加**到同一个文件，frontmatter 含 `source: developer`，无 `op` / `skills_consulted` 字段 |

时间戳用本地时间，与 frontmatter 的 ISO8601 时间一致即可。

## 注意事项

- **不要在 journal 里写解决方案的完整代码**：journal 只记录"skill 哪里需要改"，具体修改文本由 review skill 在 apply 阶段产出
- **同一问题不要重复写**：写之前 grep 一下现有 journal，避免 e1 和 e2 是同一件事
- **拒绝的 entry 也保留**：`status=rejected` 的 entry 留着，下次同主题再出现时频次会累计，便于发现"反复被拒但反复出现"的争议项
