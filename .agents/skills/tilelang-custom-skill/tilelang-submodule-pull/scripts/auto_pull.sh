#!/bin/bash

# SKILL目录
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SKILL_DIR/logs"
LOG_FILE="$LOG_DIR/git_pull.log"
# tilelang-ascend工作目录
WORK_DIR="/mnt/workspace/tilelang-ascend"

# 创建日志目录
mkdir -p "$LOG_DIR"

cd "$WORK_DIR" || exit 1

while true; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "开始拉取: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    
    # 同时执行两个命令
    echo "同时执行两个命令..." | tee -a "$LOG_FILE"
    
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
