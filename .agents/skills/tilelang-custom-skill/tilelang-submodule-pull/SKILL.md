---
name: tilelang-submodule-pull
description: 自动拉取 tilelang 仓库及其三方仓代码。提供定时拉取脚本，支持 git pull --recurse-submodules 和 git submodule update --init --recursive 两种方式，自动检测错误并重试。当用户提到"重试拉取三方库"、"自动重试拉取"、"拉取子模块"、"更新三方库"、"重新拉取三方库"、"自动拉取代码"等关键词时触发此技能。
---

# Tilelang 三方仓自动拉取

## 概述

本技能提供自动拉取 tilelang 仓库及其三方仓代码的能力：
- **定时拉取**：每隔一小时执行一次拉取
- **错误处理**：自动检测网络错误、访问失败等问题
- **多种方式**：支持两种拉取方式，自动切换
- **实时输出**：拉取过程实时打印到终端和日志文件
- **超时保护**：后台运行超过10小时自动停止，避免忘记关闭

## 脚本路径

所有路径都是相对于当前项目根目录的相对路径：

假设当前项目在 `/mnt/workspace/tilelang-ascend`，则：

| 文件 | 路径 | 说明 |
|-----|------|------|
| 拉取脚本 | `.agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh` | 主脚本 |
| 日志文件 | `.agents/skills/tilelang-custom-skill/tilelang-submodule-pull/logs/git_pull.log` | 拉取日志 |

脚本会自动检测项目根目录，无需手动配置。

## 使用方式

### 1. 直接运行（前台）

```bash
bash .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh
```

### 2. 后台运行

```bash
nohup .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh &
```

### 3. 查看日志

```bash
tail -f .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/logs/git_pull.log
```

## 工作流程

```
1. 配置 git 镜像源（使用 ghfast.top 加速）
2. 检查运行时间是否超过10小时（仅后台运行）
    ↓ 超过10小时
3. 自动停止脚本并记录日志
    ↓ 未超时
4. 同时执行两个命令：
   - git submodule update --init --recursive
   - git pull --recurse-submodules
    ↓ 任一失败
5. 等待 1 小时后重试整个流程
    ↓ 两个都成功
6. 停止脚本
```

## 错误检测

脚本会自动检测以下错误：

- `Could not access`：无法访问子模块
- `error`：一般错误
- `fatal`：致命错误
- `Failed to clone`：克隆失败
- `unable to access`：无法访问 URL

## 脚本内容

```bash
#!/bin/bash

# SKILL目录
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SKILL_DIR/logs"
LOG_FILE="$LOG_DIR/git_pull.log"
# 项目根目录（SKILL目录的父目录的父目录）
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SKILL_DIR")")")"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 配置 git 镜像源
echo "配置 git 镜像源..." | tee -a "$LOG_FILE"
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/" 2>&1 | tee -a "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

# 记录脚本开始时间（用于10小时超时检查）
START_TIME=$(date +%s)
MAX_RUNTIME_SECONDS=36000

while true; do
    # 检查运行时间是否超过10小时
    CURRENT_TIME=$(date +%s)
    RUNTIME=$((CURRENT_TIME - START_TIME))
    if [ "$RUNTIME" -ge "$MAX_RUNTIME_SECONDS" ]; then
        echo "========================================" | tee -a "$LOG_FILE"
        echo "⚠ 脚本运行超过10小时，自动停止: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        echo "已运行时间: $((RUNTIME / 3600))小时 $((RUNTIME % 3600 / 60))分钟" | tee -a "$LOG_FILE"
        exit 0
    fi
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "开始拉取: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    
    # 执行两个命令
    echo "执行两个命令..." | tee -a "$LOG_FILE"
    
    # 命令1: git submodule update --init --recursive
    echo "1. 执行命令\`git submodule update --init --recursive\`" | tee -a "$LOG_FILE"
    TMP_FILE1=$(mktemp)
    git submodule update --init --recursive 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE1"
    OUTPUT1=$(cat "$TMP_FILE1")
    rm "$TMP_FILE1"
    
    # 命令2: git pull --recurse-submodules
    echo "2. 执行命令\`git pull --recurse-submodules\`" | tee -a "$LOG_FILE"
    TMP_FILE2=$(mktemp)
    git pull --recurse-submodules 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE2"
    OUTPUT2=$(cat "$TMP_FILE2")
    rm "$TMP_FILE2"
    
    # 检查命令1是否有错误
    HAS_ERROR1=false
    if echo "$OUTPUT1" | grep -q "Could not access\|error\|fatal\|Failed to clone\|unable to access"; then
        echo "✗ git submodule update --init --recursive 失败" | tee -a "$LOG_FILE"
        HAS_ERROR1=true
    else
        echo "✓ git submodule update --init --recursive 成功" | tee -a "$LOG_FILE"
    fi
    
    # 检查命令2是否有错误
    HAS_ERROR2=false
    if echo "$OUTPUT2" | grep -q "Could not access\|error\|fatal\|Failed to clone\|unable to access"; then
        echo "✗ git pull --recurse-submodules 失败" | tee -a "$LOG_FILE"
        HAS_ERROR2=true
    else
        echo "✓ git pull --recurse-submodules 成功" | tee -a "$LOG_FILE"
    fi
    
    # 如果任一命令失败，等待1小时后重试
    if [ "$HAS_ERROR1" = true ] || [ "$HAS_ERROR2" = true ]; then
        echo "✗ 拉取失败: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        echo "等待 1 小时后重试..."
        sleep 3600
        continue
    fi
    
    # 两个命令都成功
    echo "✓ 所有代码已成功拉取: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    echo "脚本停止"
    break
done
```

## 注意事项

1. 脚本会在所有代码成功拉取后自动停止
2. 如果网络不稳定，脚本会自动重试
3. 所有输出都会同时显示在终端和日志文件中
4. 日志文件会记录每次拉取的详细过程
