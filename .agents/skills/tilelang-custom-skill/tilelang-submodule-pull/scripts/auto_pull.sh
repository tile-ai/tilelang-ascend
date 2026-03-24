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
    TMP_FILE1=$(mktemp)
    git submodule update --init --recursive 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE1"
    OUTPUT1=$(cat "$TMP_FILE1")
    rm "$TMP_FILE1"
    
    # 命令2: git pull --recurse-submodules
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
