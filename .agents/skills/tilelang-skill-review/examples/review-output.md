# 评审表格输出范例

## 1. 完整评审输出

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

## 2. 评审快照写入规则

```
.agents/skill-journal/reviews/review-{YYYY-MM-DD}.md
```

快照内容 = 上面表格 + 每条聚合项对应的 source entry 完整文本。**这一步必做**，让后续 apply 命令能按编号定位回原 entry。

如果同一天已经有 review 文件，追加到末尾（用 `## 评审会话 HH:MM` 二级标题分隔）。

## 3. 状态查询输出范例

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
