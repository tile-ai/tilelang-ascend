#!/bin/bash

# SKILL directory
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SKILL_DIR/logs_DIR"
LOG_FILE="$LOG_DIR/git_pull.log"
# Project root directory (parent of parent of SKILL directory)
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SKILL_DIR")")")"

# Create log directory
mkdir -p "$LOG_DIR"

# Configure git mirror source
echo "Configuring git mirror source..." | tee -a "$LOG_FILE"
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/" 2>&1 | tee -a "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

# Record script start time (for 10-hour timeout check)
START_TIME=$(date +%s)
MAX_RUNTIME_SECONDS=36000

while true; do
    # Check if running time exceeds 10 hours
    CURRENT_TIME=$(date +%s)
    RUNTIME=$((CURRENT_TIME - START_TIME))
    if [ "$RUNTIME" -ge "$MAX_RUNTIME_SECONDS" ]; then
        echo "========================================" | tee -a "$LOG_FILE"
        echo "⚠ Script running for over 10 hours, auto-stopping: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        echo "Runtime: $((RUNTIME / 3600)) hours $((RUNTIME % 3600 / 60)) minutes" | tee -a "$LOG_FILE"
        exit 0
    fi
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting pull: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    
    # Execute two commands
    echo "Executing two commands..." | tee -a "$LOG_FILE"
    
    # Command 1: git submodule update --init --recursive
    TMP_FILE1=$(mktemp)
    git submodule update --init --recursive 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE1"
    OUTPUT1=$(cat "$TMP_FILE1")
    rm "$TMP_FILE1"
    
    # Command 2: git pull --recurse-submodules
    TMP_FILE2=$(mktemp)
    git pull --recurse-submodules 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE2"
    OUTPUT2=$(cat "$TMP_FILE2")
    rm "$TMP_FILE2"
    
    # Check if command 1 has errors
    HAS_ERROR1=false
    if echo "$OUTPUT1" | grep -q "Could not access\|error\|fatal\|Failed to clone\|unable to access"; then
        echo "✗ git submodule update --init --recursive failed" | tee -a "$LOG_FILE"
        HAS_ERROR1=true
    else
        echo "✓ git submodule update --init --recursive succeeded" | tee -a "$LOG_FILE"
    fi
    
    # Check if command 2 has errors
    HAS_ERROR2=false
    if echo "$OUTPUT2" | grep -q "Could not access\|error\|fatal\|Failed to clone\|unable to access"; then
        echo "✗ git pull --recurse-submodules failed" | tee -a "$LOG_FILE"
        HAS_ERROR2=true
    else
        echo "✓ git pull --recurse-submodules succeeded" | tee -a "$LOG_FILE"
    fi
    
    # If either command fails, wait 1 hour then retry
    if [ "$HAS_ERROR1" = true ] || [ "$HAS_ERROR2" = true ]; then
        echo "✗ Pull failed: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        echo "Waiting 1 hour before retry..."
        sleep 3600
        continue
    fi
    
    # Both commands succeeded
    echo "✓ All code successfully pulled: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    echo "Script stopped"
    break
done

