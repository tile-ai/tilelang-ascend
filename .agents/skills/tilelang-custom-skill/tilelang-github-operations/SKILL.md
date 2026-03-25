---
name: tilelang-github-operations
description: |
  GitHub 操作指南集合。支持的操作：(1) PR 工作流：提交代码、创建 Pull Request；(2) GitHub CLI 配置：安装、认证、Token 管理。触发关键词：PR、pull request、push、commit、gh 命令、GitHub CLI、提交代码、创建 PR 等。
---

# GitHub 操作指南

本技能提供 GitHub 操作的完整指导，帮助 AI Agent 正确执行 GitHub 相关任务。

## 支持的操作

根据用户需求，阅读对应的操作指南：

| 操作 | 说明 | 指南文档 |
|------|------|----------|
| **GitHub CLI 配置** | 安装和配置 GitHub CLI，包括认证和 Token 管理 | [gh-cli-guide.md](gh-cli-guide.md) |
| **PR 工作流** | 提交代码并创建 Pull Request，包括 commit、push、创建 PR 的完整流程 | [pr-workflow-guide.md](pr-workflow-guide.md) |

## 使用方式

1. **识别用户意图**：根据用户请求确定需要执行的操作
2. **阅读对应指南**：打开上表中对应的指南文档，按照文档执行
3. **逐步执行**：严格遵循指南中的步骤和注意事项

## 快速导航

### 场景 → 操作

| 用户场景 | 执行操作 |
|----------|----------|
| 首次使用 GitHub CLI、认证失败、需配置 Token | 阅读 `gh-cli-guide.md` |
| 提交代码变更、创建 PR、查看 PR 状态 | 阅读 `pr-workflow-guide.md` |

## 扩展指南

添加新操作时需完成以下步骤：

1. 在本目录下创建新的 `.md` 文档
2. 在「支持的操作」表格中添加新条目
3. 在「场景 → 操作」表格中添加对应场景
4. **更新头部 description 元数据**，添加新操作的名称和关键词（AI Agent 通过 description 决定是否调用此技能）

---

> **重要提示**：执行任何 GitHub 操作前，请务必仔细阅读对应指南文档中的「注意事项」部分。