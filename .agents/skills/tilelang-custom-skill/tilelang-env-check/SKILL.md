---
name: tilelang-env-check
description: TileLang-Ascend 环境检查与配置验证技能。检查代码仓库完整性、编译安装状态、环境变量配置，并运行简单测试验证环境。发现问题会自动调用相关 skill 进行修复，并按依赖顺序重新执行后续步骤。触发关键词："环境检查"、"检查环境"、"验证环境"、"环境配置"、"环境搭建"、"env check"、"check environment"、"verify environment"、"setup environment"。
---

# TileLang-Ascend 环境检查

## 概述

本技能用于验证 TileLang-Ascend 开发环境是否正确配置。包括：

0. **Python 包依赖检查**：检查 torch 和 torch_npu 是否已安装且版本满足要求（torch/torch_npu >= 2.6.0）
1. **CANN 环境检查**：检查 ASCEND_HOME_PATH 环境变量是否设置且版本满足要求（CANN >= 8.3）
2. **代码仓库完整性检查**：验证代码和子模块是否完整拉取
3. **编译安装检查**：验证是否已成功编译安装
4. **环境变量检查**：验证必要的环境变量是否设置

**重要特性**：
- 发现问题时，会自动调用相关 skill 进行修复，并按依赖顺序重新执行后续步骤
- **所有问题都会打印提示给用户**，包括会被自动修复的问题，让用户知道之前存在这个问题

**前置检查说明**：步骤 0-1 为前置检查，**只检查不修复**，检查完成后统一告知用户结果，然后继续后续检查流程。

## 触发条件

当用户提到以下关键词时触发：
- 环境检查、检查环境、验证环境
- 环境配置、环境搭建
- env check、check environment、verify environment、setup environment

## 版本要求

| 检查项 | 最低版本要求 | 说明 |
|-------|-------------|------|
| torch | >= 2.6.0 | PyTorch 基础库 |
| torch_npu | >= 2.6.0 | 昇腾 NPU 支持 |
| CANN | >= 8.3 | 昇腾计算架构 |

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

**重要**：修复前必须告知用户问题存在，修复后再告知用户已修复。

## 检查流程

### 前置步骤零：检查 Python 包依赖

检查 torch 和 torch_npu 包是否已安装且版本满足要求：

```bash
# 检查 torch 和 torch_npu 版本
pip list 2>/dev/null | grep -E "^torch\s|^torch_npu\s" || pip3 list 2>/dev/null | grep -E "^torch\s|^torch_npu\s"
```

**检查逻辑**：
- 同时存在 torch 和 torch_npu 且版本 >= 2.6.0：前置检查通过
- 缺失 torch：告知用户 "✗ 未安装 torch 包"
- 缺失 torch_npu：告知用户 "✗ 未安装 torch_npu 包"
- torch 版本 < 2.6.0：告知用户 "✗ torch 版本过低 (当前版本: X.X.X)，需要 >= 2.6.0"
- torch_npu 版本 < 2.6.0：告知用户 "✗ torch_npu 版本过低 (当前版本: X.X.X)，需要 >= 2.6.0"
- 两者都缺失：告知用户 "✗ 未安装 torch 和 torch_npu 包"

**版本比较方法**：
使用 Python 进行版本比较：
```bash
python3 -c "
import re
def check_version(pkg_line, min_ver):
    match = re.search(r'(\d+\.\d+\.\d+)', pkg_line)
    if match:
        ver = match.group(1)
        parts = [int(x) for x in ver.split('.')]
        min_parts = [int(x) for x in min_ver.split('.')]
        return ver, parts >= min_parts[:len(parts)]
    return None, False

torch_line = 'torch                     2.5.0'  # 示例
torch_npu_line = 'torch_npu                 2.5.0'  # 示例

torch_ver, torch_ok = check_version(torch_line, '2.6.0')
torch_npu_ver, torch_npu_ok = check_version(torch_npu_line, '2.6.0')

if not torch_ok:
    print(f'torch 版本过低: {torch_ver}')
if not torch_npu_ok:
    print(f'torch_npu 版本过低: {torch_npu_ver}')
"
```

**重要**：此检查只报告结果，**不自动修复**。检查完成后继续下一步。

### 前置步骤一：检查 CANN 环境变量和版本

检查 ASCEND_HOME_PATH 环境变量是否设置且版本满足要求：

```bash
# 检查 ASCEND_HOME_PATH 和版本
if [ -n "$ASCEND_HOME_PATH" ]; then
    echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
    # 从路径中提取版本号（如 cann-8.5.0）
    cann_ver=$(echo "$ASCEND_HOME_PATH" | grep -oP 'cann-\d+\.\d+' | sed 's/cann-//')
    echo "CANN 版本: $cann_ver"
else
    echo "ASCEND_HOME_PATH 未设置"
fi
```

**检查逻辑**：
- ASCEND_HOME_PATH 存在且不为空，且 CANN 版本 >= 8.3：前置检查通过
- ASCEND_HOME_PATH 未设置或为空：告知用户 "✗ CANN 包路径未 source"
- CANN 版本 < 8.3：告知用户 "✗ CANN 版本过低 (当前版本: X.X)，需要 >= 8.3"

**版本提取方法**：
CANN 版本通常从 ASCEND_HOME_PATH 路径中提取，如 `/home/user/Ascend/cann-8.5.0` 中的 `8.5`。

**重要**：此检查只报告结果，**不自动修复**。检查完成后继续下一步。

### 前置检查结果汇总

在完成前置步骤 0 和 1 后，AI agent 应汇总结果告知用户：

```
========================================
前置环境检查结果
========================================
[Python 包] torch: ✓ 已安装 (版本 X.X.X) / ✗ 未安装 / ✗ 版本过低 (X.X.X < 2.6.0)
[Python 包] torch_npu: ✓ 已安装 (版本 X.X.X) / ✗ 未安装 / ✗ 版本过低 (X.X.X < 2.6.0)
[CANN 环境] ASCEND_HOME_PATH: ✓ 已设置 (版本 X.X) / ✗ 未设置 / ✗ 版本过低 (X.X < 8.3)
========================================
```

如果前置检查全部通过，告知用户 "前置环境检查通过，继续后续检查..."；
如果前置检查存在问题，告知用户具体问题项，然后询问用户是否继续后续检查。

### 第二步：检查代码仓库完整性

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

**问题提示要求**：
如果子模块不完整，**必须先告知用户问题存在**：
```
✗ 发现问题：子模块不完整，正在自动修复...
```

然后执行修复流程，修复完成后告知用户：
```
✓ 问题已修复：子模块已完整拉取
```

**如果子模块不完整，AI agent 应执行以下操作序列**：

1. **告知用户问题存在**："✗ 发现问题：子模块不完整，正在自动修复..."
2. **立即调用 `tilelang-submodule-pull` skill 拉取子模块**：
   ```
   skill(name="tilelang-submodule-pull")
   ```
3. **告知用户修复结果**："✓ 问题已修复：子模块已完整拉取"
4. **告知用户需要重新编译**："由于子模块更新，需要重新编译..."
5. **子模块拉取完成后，必须重新编译**（即使之前有编译产物）：
   ```bash
   bash install_ascend.sh
   ```
6. **编译完成后，必须设置环境变量**：
   ```bash
   source set_env.sh
   ```

### 第三步：检查编译安装状态

检查以下内容：
- `build/` 目录是否存在
- 编译产物是否存在（检查关键库文件）

```bash
# 检查 build 目录
ls -la build/

# 检查编译产物（根据实际情况调整）
ls -la build/*.so 2>/dev/null || echo "No .so files found"
```

**问题提示要求**：
如果编译产物不存在，**必须先告知用户问题存在**：
```
✗ 发现问题：编译产物不存在，正在自动修复...
```

然后执行修复流程，修复完成后告知用户：
```
✓ 问题已修复：编译产物已生成
```

**如果检查失败，AI agent 应执行以下操作序列**：

1. **告知用户问题存在**："✗ 发现问题：编译产物不存在，正在自动修复..."
2. **执行编译安装**：
   ```bash
   bash install_ascend.sh
   ```
3. **告知用户修复结果**："✓ 问题已修复：编译产物已生成"
4. **编译完成后，必须设置环境变量**：
   ```bash
   source set_env.sh
   ```

### 第四步：检查环境变量

检查以下环境变量：
- `TL_ROOT`：TileLang 根目录
- `PYTHONPATH`：Python 路径
- `ACL_OP_INIT_MODE`：ACL 操作初始化模式

```bash
echo "TL_ROOT: $TL_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "ACL_OP_INIT_MODE: $ACL_OP_INIT_MODE"
```

**问题提示要求**：
如果环境变量未设置，**必须先告知用户问题存在**：
```
✗ 发现问题：环境变量未设置，正在自动修复...
```

然后执行修复流程，修复完成后告知用户：
```
✓ 问题已修复：环境变量已设置
```

**如果检查失败，AI agent 应执行环境设置**：
1. **告知用户问题存在**："✗ 发现问题：环境变量未设置，正在自动修复..."
2. 执行：`source set_env.sh`
3. **告知用户修复结果**："✓ 问题已修复：环境变量已设置"

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
2. **发现问题时，先告知用户问题存在，再执行修复**
3. 根据检查结果，发现问题时主动调用相关 skill/执行命令进行修复
4. **按依赖顺序执行修复**（子模块 → 编译 → 环境变量）
5. 修复完成后告知用户修复结果
6. 修复完成后重新检查，直到所有检查项通过
7. 告知用户环境状态

## 输出格式

脚本会输出详细的检查结果，格式如下：

```
========================================
TileLang-Ascend 环境检查
========================================

[前置] 检查 Python 包依赖...
✓ torch 已安装 (版本 2.7.1 >= 2.6.0)
✓ torch_npu 已安装 (版本 2.7.1 >= 2.6.0)

[前置] 检查 CANN 环境变量...
✓ ASCEND_HOME_PATH 已设置 (CANN 版本 8.5 >= 8.3)

[1/4] 检查代码仓库完整性...
✓ Git 仓库存在
✓ 子模块 tvm 存在
✓ 子模块 cutlass 存在
✓ 所有子模块完整

[2/4] 检查编译安装状态...
✓ build 目录存在
✓ 编译产物存在

[3/4] 检查环境变量...
✓ TL_ROOT 已设置
✓ PYTHONPATH 已设置
✓ ACL_OP_INIT_MODE 已设置

========================================
✓ 环境检查通过！所有配置正确。
========================================
```

**发现问题时的输出格式**：

```
========================================
TileLang-Ascend 环境检查
========================================

[前置] 检查 Python 包依赖...
✓ torch 已安装 (版本 2.7.1 >= 2.6.0)
✓ torch_npu 已安装 (版本 2.7.1 >= 2.6.0)

[前置] 检查 CANN 环境变量...
✓ ASCEND_HOME_PATH 已设置 (CANN 版本 8.5 >= 8.3)

[1/4] 检查代码仓库完整性...
✓ Git 仓库存在
✗ 发现问题：子模块不完整，正在自动修复...
[执行 tilelang-submodule-pull skill]
✓ 问题已修复：子模块已完整拉取

[2/4] 检查编译安装状态...
✗ 发现问题：编译产物不存在，正在自动修复...
[执行 bash install_ascend.sh]
✓ 问题已修复：编译产物已生成

[3/4] 检查环境变量...
✗ 发现问题：环境变量未设置，正在自动修复...
[执行 source set_env.sh]
✓ 问题已修复：环境变量已设置

========================================
✓ 环境检查通过！所有配置已修复完成。
========================================
```

## 完整工作流（包含测试验证）

```
开始
  ↓
[前置0] 检查 Python 包依赖 (torch + torch_npu >= 2.6.0)
  ├─ 缺失/版本过低 → 告知用户问题，询问是否继续
  └─ 正常 → 继续
  ↓
[前置1] 检查 CANN 环境变量 (ASCEND_HOME_PATH, CANN >= 8.3)
  ├─ 未设置/版本过低 → 告知用户问题，询问是否继续
  └─ 正常 → 继续
  ↓
[前置结果汇总] 告知用户前置检查结果
  ↓
[2] 检查代码仓库完整性
  ├─ 失败 → 告知用户问题 → 执行子模块修复流程 [A] → 告知用户修复结果
  │
  └─ 成功 → 继续
  ↓
[3] 检查编译安装状态
  ├─ 失败 → 告知用户问题 → 执行编译修复流程 [B] → 告知用户修复结果
  │
  └─ 成功 → 继续
  ↓
[4] 检查环境变量
  ├─ 失败 → 告知用户问题 → 执行 source set_env.sh → 告知用户修复结果 → 继续 [5]
  └─ 成功 → 继续
  ↓
[5] 运行测试验证
  ├─ 成功 → ✓ 环境正确，告知用户
  │
  └─ 失败 → 告知用户问题 → 执行子模块修复流程 [A]（跳过检查，直接修复）→ 告知用户修复结果
  ↓
结束

[A] 子模块修复流程（测试失败时直接执行此流程）：
    告知用户问题 → 调用 tilelang-submodule-pull → 编译 → 设置环境变量 → 测试 → 告知用户修复结果

[B] 编译修复流程：
    告知用户问题 → bash install_ascend.sh → 设置环境变量 → 测试 → 告知用户修复结果
```

**关键逻辑**：
1. 前置检查 [0-1] 只检查不修复，完成后统一告知用户结果
2. **所有问题都要告知用户**，包括会被自动修复的问题
3. 修复前告知问题，修复后告知结果
4. 测试失败时，直接执行子模块修复流程 [A]，跳过重新检查

## AI Agent 执行指南

当触发此 skill 时，AI agent 应按以下步骤执行：

### 步骤 0：检查 Python 包依赖

使用 Bash 工具检查 torch 和 torch_npu：
```bash
pip list 2>/dev/null | grep -E "^torch\s|^torch_npu\s" || pip3 list 2>/dev/null | grep -E "^torch\s|^torch_npu\s"
```

**结果处理**：
- 同时存在 torch 和 torch_npu 且版本 >= 2.6.0：标记为通过
- 缺失 torch：告知用户 "✗ 未安装 torch 包"
- 缺失 torch_npu：告知用户 "✗ 未安装 torch_npu 包"
- torch 版本 < 2.6.0：告知用户 "✗ torch 版本过低 (当前版本: X.X.X)，需要 >= 2.6.0"
- torch_npu 版本 < 2.6.0：告知用户 "✗ torch_npu 版本过低 (当前版本: X.X.X)，需要 >= 2.6.0"
- 两者都缺失：告知用户 "✗ 未安装 torch 和 torch_npu 包"

**重要**：此检查只报告结果，**不自动修复**。

### 步骤 1：检查 CANN 环境变量和版本

使用 Bash 工具检查 ASCEND_HOME_PATH：
```bash
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
# 从路径中提取版本号
echo "$ASCEND_HOME_PATH" | grep -oP 'cann-\d+\.\d+' | sed 's/cann-//'
```

**结果处理**：
- 存在且不为空，且版本 >= 8.3：标记为通过
- 未设置或为空：告知用户 "✗ CANN 包路径未 source，请先 source CANN 环境变量"
- CANN 版本 < 8.3：告知用户 "✗ CANN 版本过低 (当前版本: X.X)，需要 >= 8.3"

**重要**：此检查只报告结果，**不自动修复**。

### 步骤 1.5：前置检查结果汇总

汇总步骤 0 和 1 的结果，告知用户：

```
========================================
前置环境检查结果
========================================
[Python 包] torch: ✓ 已安装 (版本 X.X.X) / ✗ 未安装 / ✗ 版本过低 (X.X.X < 2.6.0)
[Python 包] torch_npu: ✓ 已安装 (版本 X.X.X) / ✗ 未安装 / ✗ 版本过低 (X.X.X < 2.6.0)
[CANN 环境] ASCEND_HOME_PATH: ✓ 已设置 (版本 X.X) / ✗ 未设置 / ✗ 版本过低 (X.X < 8.3)
========================================
```

如果前置检查存在问题，询问用户是否继续后续检查。

### 步骤 2：执行代码仓库检查

使用 Bash 工具运行检查脚本：
```bash
bash .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/check_env.sh
```

### 步骤 3：解析结果并按依赖顺序自动修复

根据检查脚本的输出结果：

**场景 A：子模块不完整**

如果显示 "✗ 子模块不完整"：
1. **告知用户问题存在**："✗ 发现问题：子模块不完整，正在自动修复..."
2. 立即使用 Skill 工具调用 `tilelang-submodule-pull`
3. 等待子模块拉取完成
4. **告知用户修复结果**："✓ 问题已修复：子模块已完整拉取"
5. **告知用户需要重新编译**："由于子模块更新，需要重新编译..."
6. **必须重新编译**：`bash install_ascend.sh`（即使之前有编译产物）
7. **必须设置环境变量**：`source set_env.sh`
8. 跳转到 **步骤 4** 运行测试验证

**场景 B：编译产物不存在（子模块完整）**

如果显示 build 目录不存在或编译产物不存在，但子模块完整：
1. **告知用户问题存在**："✗ 发现问题：编译产物不存在，正在自动修复..."
2. 执行编译：`bash install_ascend.sh`
3. **告知用户修复结果**："✓ 问题已修复：编译产物已生成"
4. **必须设置环境变量**：`source set_env.sh`
5. 跳转到 **步骤 4** 运行测试验证

**场景 C：环境变量未设置**

如果仅显示环境变量未设置：
1. **告知用户问题存在**："✗ 发现问题：环境变量未设置，正在自动修复..."
2. 执行：`source set_env.sh`
3. **告知用户修复结果**："✓ 问题已修复：环境变量已设置"
4. 跳转到 **步骤 4** 运行测试验证

**场景 D：所有检查通过**

直接跳转到 **步骤 4** 运行测试验证。

### 步骤 4：运行测试验证

执行简单测试脚本验证环境是否真正可用：
```bash
source set_env.sh
python .agents/skills/tilelang-custom-skill/tilelang-env-check/scripts/quick_verify.py
```

**测试结果处理**：

| 测试结果 | 处理动作 |
|---------|---------|
| ✓ TileLang 环境验证通过! | 告知用户 "环境已正确配置，可以开始使用" |
| 测试失败 | **告知用户问题存在，然后执行子模块修复流程**，跳过重新检查 |

**测试失败时的处理流程**：
1. **告知用户问题存在**："✗ 发现问题：测试验证失败，正在自动修复..."
2. 调用 `tilelang-submodule-pull` skill 拉取子模块
3. 编译：`bash install_ascend.sh`
4. 设置环境变量：`source set_env.sh`
5. **告知用户修复结果**："✓ 问题已修复：环境已重新配置"
6. 再次运行测试验证

**重要**：测试失败时，不要重新检查代码仓库完整性，直接执行子模块修复流程，这样可以确保环境肯定正确。

### 步骤 5：告知用户

- 如果测试通过：告知用户 "✓ 环境已正确配置，可以开始使用"
- 如果多次重试后仍失败：调用 `tilelang-error-fixer` skill 进行诊断

## 常见问题

### 问题0：Python 包缺失或版本过低（torch 或 torch_npu）

**处理方式**：只报告结果，不自动修复。告知用户：
- 缺失 torch："✗ 请安装 torch 包：pip install torch"
- 缺失 torch_npu："✗ 请安装 torch_npu 包：pip install torch-npu"
- 两者都缺失："✗ 请安装 torch 和 torch_npu 包"
- torch 版本过低："✗ torch 版本过低 (当前: X.X.X)，请升级到 >= 2.6.0：pip install --upgrade torch"
- torch_npu 版本过低："✗ torch_npu 版本过低 (当前: X.X.X)，请升级到 >= 2.6.0：pip install --upgrade torch-npu"

### 问题1：CANN 环境变量未设置或版本过低

**处理方式**：只报告结果，不自动修复。告知用户：
- 未设置："✗ 请先 source CANN 环境变量（通常为 source /usr/local/Ascend/ascend-toolkit/set_env.sh）"
- 版本过低："✗ CANN 版本过低 (当前: X.X)，请升级到 >= 8.3"

### 问题2：子模块不完整

**处理方式**：
1. **告知用户问题存在**："✗ 发现问题：子模块不完整"
2. 自动修复：调用 `tilelang-submodule-pull` skill
3. 拉取成功后重新编译
4. 设置环境变量
5. **告知用户修复结果**："✓ 问题已修复：子模块已完整拉取"

### 问题3：编译产物不存在

**处理方式**：
1. **告知用户问题存在**："✗ 发现问题：编译产物不存在"
2. 自动修复：执行 `bash install_ascend.sh`
3. 如果失败则调用 `tilelang-error-fixer` skill 诊断
4. 编译成功后设置环境变量
5. **告知用户修复结果**："✓ 问题已修复：编译产物已生成"

### 问题4：环境变量未设置

**处理方式**：
1. **告知用户问题存在**："✗ 发现问题：环境变量未设置"
2. 自动修复：执行 `source set_env.sh`
3. **告知用户修复结果**："✓ 问题已修复：环境变量已设置"

### 问题5：测试运行失败

**处理方式**：
1. **告知用户问题存在**："✗ 发现问题：测试验证失败"
2. **直接执行子模块修复流程**，跳过重新检查：
3. 调用 `tilelang-submodule-pull` skill 拉取子模块
4. 编译：`bash install_ascend.sh`
5. 设置环境变量：`source set_env.sh`
6. 再次运行测试验证
7. **告知用户修复结果**："✓ 问题已修复：环境已重新配置"

**注意**：测试失败时，不需要重新检查代码仓库完整性，直接执行修复流程即可确保环境正确。

## 注意事项

1. 前置检查（Python 包、CANN 环境变量）只报告结果，**不自动修复**
2. **所有问题都要告知用户**，包括会被自动修复的问题
3. **修复前告知问题，修复后告知结果**
4. 每次新开终端都需要重新设置环境变量（运行 `source set_env.sh`）
5. 编译安装只需执行一次，除非代码有更新
6. **如果子模块缺失，必须重新编译，旧产物不可用**
7. AI agent 应主动修复问题（前置检查除外），无需等待用户手动操作
8. 测试用例需要 NPU 设备可用

## 相关 Skills

- `tilelang-submodule-pull`：自动拉取代码和子模块（子模块缺失时调用）
- `tilelang-debug-helper`：调试帮助
- `tilelang-error-fixer`：错误诊断与修复（测试失败时调用）