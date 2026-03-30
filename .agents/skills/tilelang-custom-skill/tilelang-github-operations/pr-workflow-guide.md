# AI Agent 使用 GitHub CLI 提交 PR 操作指南

> 本指南面向 AI Agent，描述如何将本地修改提交到远程仓库并创建 PR。

---

## 完整流程

### 1. 确认目标仓库和分支

如果用户没有明确指定目标仓库和分支，**必须先询问用户**：

```
请问您希望提交到哪个仓库和分支？
- 默认选项：tile-ai/tilelang-ascend 的 ascendc_pto 分支
- 或指定其他仓库/分支
```

**默认配置**：
- 仓库：`tile-ai/tilelang-ascend`（https://github.com/tile-ai/tilelang-ascend）
- 分支：`ascendc_pto`（该仓库的默认分支）

**分支推断规则**：
- 用户指定仓库 + 分支 → 使用用户指定的分支
- 用户仅指定仓库，未指定分支 → 使用该仓库的默认分支（通常为 `main` 或 `master`）
- 用户均未指定 → 使用默认配置（`tile-ai/tilelang-ascend` 的 `ascendc_pto` 分支）

---

### 2. 查看当前状态

```bash
# 查看当前分支和修改状态
git status

# 查看已暂存和未暂存的修改内容
git diff --staged
git diff

# 查看最近的 commit 历史（了解 commit 风格）
git log --oneline -10

# 查看远程仓库配置
git remote -v
```

---

### 3. 展示待提交文件列表（重要）

在执行任何 git add 或 commit 操作之前，**必须向用户展示待提交的文件列表**：

```bash
# 查看所有变更文件
git status --short
```

向用户确认：

```
以下文件将被提交：
- 文件1
- 文件2
- ...

请确认是否继续？如有不希望提交的文件，请告知。
```

等待用户确认后再继续。如有不需要提交的文件，只 add 需要的文件。

---

### 4. 暂存文件

根据用户确认的文件列表，使用 git add：

```bash
# 添加指定文件
git add <file1> <file2> ...

# 或添加整个目录
git add <directory>/

# 避免使用 git add . 或 git add -A，防止意外提交
```

---

### 5. 编写 Commit 描述

#### Commit Message 格式

```
[<type>] <subject> 🤖

<body>
```

> **重要**：所有由 AI Agent 提交的 commit，标题末尾必须添加 `🤖` 标识。

#### Type 类型

| Type | 说明 |
|------|------|
| Feat | 新功能 |
| Fix | Bug 修复 |
| Docs | 文档更新 |
| Refactor | 重构代码 |
| Example | 添加样例 |
| Test | 添加测试 |
| Chore | 构建/工具相关 |
| BugFix | Bug 修复（另一种写法） |
| Feature | 新功能（另一种写法） |
| WIP | 开发中（Work In Progress） |

#### 示例

```
[Feat] Add vector operator for Softmax 🤖

- Implement softmax kernel using TileLang Ascend
- Add test cases for different input shapes
- Update documentation
```

#### 创建 Commit

```bash
git commit -m "<commit message>"
```

---

### 6. 推送分支到远程

```bash
# 推送当前分支到远程仓库（-u 设置上游跟踪）
git push -u <remote> <branch-name>

# 示例：推送到 origin
git push -u origin ascendc_pto
```

如果分支是新分支，GitHub 会提示创建 PR 的链接。

---

### 7. 创建 Pull Request

#### 使用 gh pr create

```bash
gh pr create \
  --repo <owner/repo> \
  --base <target-branch> \
  --head <source-branch> \
  --title "<PR title>" \
  --body "<PR description>"
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--repo` | 目标仓库（如 `tile-ai/tilelang-ascend`），当前仓库可省略 |
| `--base` | 目标分支（PR 合并到哪个分支） |
| `--head` | 源分支（当前分支） |
| `--title` | PR 标题 |
| `--body` | PR 描述 |

#### PR 标题格式

```
[<type>] <subject> 🤖
```

> **重要**：所有由 AI Agent 提交的 PR，标题末尾必须添加 `🤖` 标识，标明这是来自 AI 助手的贡献。

#### PR Body 模板

```markdown
## Summary

<1-3 句话描述这个 PR 做了什么>

## Changes

- 变更点1
- 变更点2
- 变更点3

## Testing

<如何测试，或测试结果>
```

#### 示例

```bash
gh pr create \
  --base ascendc_pto \
  --head agent-dev \
  --title "[Feat] Add vector operator for Softmax 🤖" \
  --body "$(cat <<'EOF'
## Summary

Add softmax vector operator implementation for Ascend NPU.

## Changes

- Add `examples/softmax/softmax.py` with kernel implementation
- Add test cases for various input shapes
- Update documentation

## Testing

Run `python examples/softmax/softmax.py` to verify correctness.
EOF
)"
```

---

### 8. 确认 PR 创建成功并总结

PR 创建成功后，**必须向用户展示 PR 的基本信息**：

```bash
# 查看 PR 详情
gh pr view <PR-number> --repo <owner/repo>

# 查看修改统计
git diff <target-branch>...HEAD --stat
```

#### 展示内容模板

```
## PR #<number> 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | <PR 标题> |
| **状态** | OPEN |
| **作者** | <作者> |
| **目标分支** | <owner/repo>: <base-branch> ← <head-branch> |
| **链接** | <PR URL> |

### 代码变更统计

- **修改文件**: X 个
- **新增行数**: +XXX
- **删除行数**: -XXX

### 内容概述

<简要描述 PR 的主要内容>
```

---

## 常用 gh 命令

```bash
# 查看认证状态
gh auth status

# 查看当前仓库的 PR 列表
gh pr list

# 查看 PR 详情
gh pr view <number>

# 查看 PR 的 checks 状态
gh pr checks <number>

# 合并 PR（需要权限）
gh pr merge <number>
```

---

## 注意事项

1. **检查 gh 工具可用性** - 创建 PR 前确认 `gh`（GitHub CLI）已安装并完成认证。若工具不存在，请参考 [gh-cli-guide.md](./gh-cli-guide.md) 进行安装和配置
2. **始终确认目标仓库和分支** - 默认为 `tile-ai/tilelang-ascend` 的 `ascendc_pto` 分支；若用户仅指定仓库，默认分支通常为 `main`
3. **提交前展示文件列表** - 让用户确认，避免提交意外文件
4. **不要提交敏感文件** - 如 `.env`、`credentials.json`、密钥等
5. **commit message 要清晰** - 说明改动了什么，为什么改动
6. **检查 .gitignore** - 确保不需要的文件已被忽略
7. **国内网络不稳定** - GitHub 操作失败时请重试
8. **AI Agent 标识** - 所有 commit 和 PR 标题末尾必须添加 `🤖` 标识
9. **PR 创建后总结** - 向用户展示 PR 基本信息、代码变更统计、内容概述

---

## 完整示例流程

```bash
# 1. 查看状态
git status
git diff --staged && git diff
git log --oneline -5

# 2. 确认文件（展示给用户）
git status --short

# 3. 暂存确认的文件
git add examples/softmax/

# 4. 提交（注意 🤖 标识）
git commit -m "[Feat] Add softmax vector operator 🤖"

# 5. 推送
git push -u origin agent-dev

# 6. 创建 PR（注意 🤖 标识）
gh pr create \
  --repo tile-ai/tilelang-ascend \
  --base ascendc_pto \
  --head agent-dev \
  --title "[Feat] Add softmax vector operator 🤖" \
  --body "$(cat <<'EOF'
## Summary

Add softmax kernel implementation for Ascend NPU.

## Changes

- Add softmax.py with TileLang kernel
- Add test cases

## Testing

Run `python examples/softmax/softmax.py`
EOF
)"

# 7. 查看 PR 详情并展示给用户
gh pr view <number> --repo tile-ai/tilelang-ascend
git diff origin/ascendc_pto...HEAD --stat
```