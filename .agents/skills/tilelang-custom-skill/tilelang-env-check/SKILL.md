---
name: tilelang-env-check
description: TileLang-Ascend 环境检查与配置验证技能。检查代码仓库完整性、编译安装状态、环境变量配置，并运行简单测试验证环境。发现问题会自动调用相关 skill 进行修复，并按依赖顺序重新执行后续步骤。触发关键词："环境检查"、"检查环境"、"验证环境"、"环境配置"、"环境搭建"、"env check"、"check environment"、"verify environment"、"setup environment"。
---

# TileLang-Ascend 环境检查

## 概述

本技能用于验证 TileLang-Ascend 开发环境是否正确配置。包括：

1. **代码仓库完整性检查**：验证代码和子模块是否完整拉取
2. **编译安装检查**：验证是否已成功编译安装
3. **环境变量检查**：验证必要的环境变量是否设置

**重要特性**：发现问题时，会自动调用相关 skill 进行修复，并按依赖顺序重新执行后续步骤，无需用户手动干预。

## 触发条件

当用户提到以下关键词时触发：
- 环境检查、检查环境、验证环境
- 环境配置、环境搭建
- env check、check environment、verify environment、setup environment

## 依赖关系说明

环境配置三步骤之间存在严格的依赖关系：

```
子模块完整 ──→ 编译成功 ──→ 环境变量设置
```

**关键规则**：

| 问题场景 | 需要执行的操作 |
|---------|---------------|
| 子模块缺失 | ① 拉取子模块 → ② **重新编译** → ③ 设置环境变量 |
| 无编译产物（子模块完整） | ① 编译 → ② 设置环境变量 |
| 环境变量未设置 | 设置环境变量 |

**重要**：如果子模块缺失，即使之前存在编译产物，也必须重新编译，因为旧产物是基于不完整代码生成的。

## 自动修复策略

当检查发现问题时，AI agent 应主动执行以下修复操作，并按依赖顺序执行：

| 问题类型 | 自动修复动作 | 后续必须操作 |
|---------|-------------|-------------|
| 子模块不完整 | 调用 `tilelang-submodule-pull` skill | 必须重新编译 + 设置环境变量 |
| 编译产物不存在 | 执行 `bash install_ascend.sh` | 必须设置环境变量 |
| 环境变量未设置 | 执行 `source set_env.sh` | 无 |

## 检查流程

### 第一步：检查代码仓库完整性

检查以下内容：
- 主仓库是否存在
- 子模块是否完整拉取和初始化（使用 `git submodule status` 检查完整性）

```bash
# 检查主仓库
git status

# 检查子模块完整性（推荐方式）
git submodule status

# 输出格式说明：
# <commit-hash> <path>     - 正常，子模块完整
# -<commit-hash> <path>     - 未初始化，需要拉取
# +<commit-hash> <path>     - commit 与索引不一致，可能下载不完整
# (空)                      - 子模块不存在
```

**完整性检测逻辑**：
- 使用 `git submodule status` 获取状态前缀
- `-` 前缀：未初始化
- `+` 前缀：commit 不一致，下载不完整
- 目录存在但为空：已初始化但未检出
- 无前缀且有内容：完整

检查失败时统一标记为 `submodule_incomplete`，后续由 AI agent 调用 `tilelang-submodule-pull` skill 进行修复。

**如果子模块不完整，AI agent 应执行以下操作序列**：

1. **立即调用 `tilelang-submodule-pull` skill 拉取子模块**：
   ```
   skill(name="tilelang-submodule-pull")
   ```

2. **子模块拉取完成后，必须重新编译**（即使之前有编译产物）：
   ```bash
   bash install_ascend.sh
   ```

3. **编译完成后，必须设置环境变量**：
   ```bash
   source set_env.sh
   ```

### 第二步：检查编译安装状态

检查以下内容：
- `build/` 目录是否存在
- 编译产物是否存在（检查关键库文件）

```bash
# 检查 build 目录
ls -la build/

# 检查编译产物（根据实际情况调整）
ls -la build/*.so 2>/dev/null || echo "No .so files found"
```

**如果检查失败，AI agent 应执行以下操作序列**：

1. **执行编译安装**：
   ```bash
   bash install_ascend.sh
   ```

2. **编译完成后，必须设置环境变量**：
   ```bash
   source set_env.sh
   ```

### 第三步：检查环境变量

检查以下环境变量：
- `TL_ROOT`：TileLang 根目录
- `PYTHONPATH`：Python 路径
- `ACL_OP_INIT_MODE`：ACL 操作初始化模式

```bash
echo "TL_ROOT: $TL_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "ACL_OP_INIT_MODE: $ACL_OP_INIT_MODE"
```

**如果检查失败，AI agent 应执行环境设置**：
```bash
source set_env.sh
```

## 脚本路径

| 文件 | 路径 | 说明 |
|-----|------|------|
| 环境检查脚本 | `.agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/check_env.sh` | 主检查脚本 |
| 快速验证脚本 | `.agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/quick_verify.py` | 最小化TileLang测试脚本 |

### 快速验证脚本

执行简单测试验证环境是否可用：
```bash
source set_env.sh
python .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/quick_verify.py
```

**测试失败处理**：直接执行子模块修复流程（调用 `tilelang-submodule-pull` → 编译 → 设置环境变量 → 测试），跳过重新检查。

## 使用方法

### 方法一：使用检查脚本

```bash
bash .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/check_env.sh
```

### 方法二：AI Agent 执行检查并自动修复

当用户请求环境检查时，AI agent 应：

1. 运行检查脚本或手动执行检查步骤
2. 根据检查结果，发现问题时主动调用相关 skill/执行命令进行修复
3. **按依赖顺序执行修复**（子模块 → 编译 → 环境变量）
4. 修复完成后重新检查，直到所有检查项通过
5. 告知用户环境状态

## 输出格式

脚本会输出详细的检查结果，格式如下：

```
========================================
TileLang-Ascend 环境检查
========================================

[1/4] 检查代码仓库完整性...
✓ Git 仓库存在
✓ 子模块 tvm 存在
✓ 子模块 cutlass 存在
✓ 所有子模块完整

[2/4] 检查编译安装状态...
✓ build 目录存在
✓ 编译产物存在

[3/3] 检查环境变量...
✓ TL_ROOT 已设置
✓ PYTHONPATH 已设置
✓ ACL_OP_INIT_MODE 已设置

========================================
✓ 环境检查通过！所有配置正确。
========================================
```

## 完整工作流（包含测试验证）

```
开始
  ↓
[1] 检查代码仓库完整性
  ├─ 失败 → 执行子模块修复流程 [A]
  │
  └─ 成功 → 继续
  ↓
[2] 检查编译安装状态
  ├─ 失败 → 执行编译修复流程 [B]
  │
  └─ 成功 → 继续
  ↓
[3] 检查环境变量
  ├─ 失败 → 执行 source set_env.sh → 继续 [4]
  └─ 成功 → 继续
  ↓
[4] 运行测试验证
  ├─ 成功 → ✓ 环境正确，告知用户
  │
  └─ 失败 → 执行子模块修复流程 [A]（跳过检查，直接修复）
  ↓
结束

[A] 子模块修复流程（测试失败时直接执行此流程）：
    调用 tilelang-submodule-pull → 编译 → 设置环境变量 → 测试

[B] 编译修复流程：
    bash install_ascend.sh → 设置环境变量 → 测试
```

**关键逻辑**：测试失败时，直接执行子模块修复流程 [A]，跳过重新检查，因为这样可以确保环境肯定正确。

## AI Agent 执行指南

当触发此 skill 时，AI agent 应按以下步骤执行：

### 步骤 1：执行检查

使用 Bash 工具运行检查脚本：
```bash
bash .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/check_env.sh
```

### 步骤 2：解析结果并按依赖顺序自动修复

根据检查脚本的输出结果：

**场景 A：子模块不完整**

如果显示 "✗ 子模块不完整"：
1. 立即使用 Skill 工具调用 `tilelang-submodule-pull`
2. 等待子模块拉取完成
3. **必须重新编译**：`bash install_ascend.sh`（即使之前有编译产物）
4. **必须设置环境变量**：`source set_env.sh`
5. 跳转到 **步骤 5** 运行测试验证

**场景 B：编译产物不存在（子模块完整）**

如果显示 build 目录不存在或编译产物不存在，但子模块完整：
1. 执行编译：`bash install_ascend.sh`
2. **必须设置环境变量**：`source set_env.sh`
3. 跳转到 **步骤 5** 运行测试验证

**场景 C：环境变量未设置**

如果仅显示环境变量未设置：
1. 执行：`source set_env.sh`
2. 跳转到 **步骤 5** 运行测试验证

**场景 D：所有检查通过**

直接跳转到 **步骤 5** 运行测试验证。

### 步骤 5：运行测试验证

执行简单测试脚本验证环境是否真正可用：
```bash
source set_env.sh
python .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/quick_verify.py
```

**测试结果处理**：

| 测试结果 | 处理动作 |
|---------|---------|
| ✓ TileLang 环境验证通过! | 告知用户 "环境已正确配置，可以开始使用" |
| 测试失败 | **直接执行子模块修复流程**，跳过重新检查 |

**测试失败时的处理流程**：
1. 调用 `tilelang-submodule-pull` skill 拉取子模块
2. 编译：`bash install_ascend.sh`
3. 设置环境变量：`source set_env.sh`
4. 再次运行测试验证

**重要**：测试失败时，不要重新检查代码仓库完整性，直接执行子模块修复流程，这样可以确保环境肯定正确。

### 步骤 6：告知用户

- 如果测试通过：告知用户 "环境已正确配置，可以开始使用"
- 如果多次重试后仍失败：调用 `tilelang-error-fixer` skill 进行诊断

## 常见问题

### 问题1：子模块拉取失败

**自动处理**：
1. 调用 `tilelang-submodule-pull` skill
2. 拉取成功后重新编译
3. 设置环境变量

### 问题2：编译失败

**自动处理**：
1. 执行 `bash install_ascend.sh`
2. 如果失败则调用 `tilelang-error-fixer` skill 诊断
3. 编译成功后设置环境变量

### 问题3：环境变量未设置

**自动处理**：执行 `source set_env.sh`

### 问题4：测试运行失败

**自动处理**：
1. **直接执行子模块修复流程**，跳过重新检查：
2. 调用 `tilelang-submodule-pull` skill 拉取子模块
3. 编译：`bash install_ascend.sh`
4. 设置环境变量：`source set_env.sh`
5. 再次运行测试验证

**注意**：测试失败时，不需要重新检查代码仓库完整性，直接执行修复流程即可确保环境正确。

## 注意事项

1. 每次新开终端都需要重新设置环境变量（运行 `source set_env.sh`）
2. 编译安装只需执行一次，除非代码有更新
3. **如果子模块缺失，必须重新编译，旧产物不可用**
4. AI agent 应主动修复问题，无需等待用户手动操作
5. 测试用例需要 NPU 设备可用

## 相关 Skills

- `tilelang-submodule-pull`：自动拉取代码和子模块（子模块缺失时调用）
- `tilelang-debug-helper`：调试帮助
- `tilelang-error-fixer`：错误诊断与修复（测试失败时调用）