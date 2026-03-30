---
name: tilelang-submodule-pull
description: Automatically pull tilelang repository and its third-party code. Provides scheduled pull script supporting git pull --recurse-submodules and git submodule update --init --recursive with automatic error detection and retry. Triggers when user mentions "重新拉取三方库", "自动重拉", "重新拉取子模块", "auto retry pull", "pull submodules", "update third-party libs", "retry pulling third-party libs", "auto pull code" or similar keywords.
---

# Tilelang Third-Party Repository Auto-Pull

## Overview

This skill provides automatic pulling of tilelang repository and its third-party code:
- **Scheduled Pull**: Executes pull every hour
- **Error Handling**: Automatically detects network errors, access failures, etc.
- **Multiple Methods**: Supports two pull methods with automatic switching
- **Real-time Output**: Pull process printed to terminal and log file in real-time
- **Timeout Protection**: Automatically stops after 10 hours of background running to prevent forgetting to close

## Script Paths

All paths are relative to the current project root directory:

Assuming the current project is at `/mnt/workspace/tilelang-ascend`, then:

| File | Path | Description |
|-----|------|------|
| Pull Script | `.agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh` | Main script |
| Log File | `.agents/skills/tilelang-custom-skill/tilelang-submodule-pull/logs/git_pull.log` | Pull log |

The script automatically detects the project root directory, no manual configuration needed.

## Usage

### 1. Direct Run (Foreground)

```bash
bash .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh
```

### 2. Background Run

```bash
nohup .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/scripts/auto_pull.sh &
```

### 3. View Logs

```bash
tail -f .agents/skills/tilelang-custom-skill/tilelang-submodule-pull/logs/git_pull.log
```

## Workflow

```
1. Configure git mirror source (using ghfast.top for acceleration)
2. Check if running time exceeds 10 hours (background run only)
    ↓ Exceeds 10 hours
3. Automatically stop script and log
    ↓ Not exceeded
4. Execute two commands simultaneously:
   - git submodule update --init --recursive
   - git pull --recurse-submodules
    ↓ If either fails
5. Wait 1 hour then retry the entire process
    ↓ Both succeed
6. Stop script
```

## Error Detection

The script automatically detects the following errors:

- `Could not access`: Cannot access submodule
- `error`: General error
- `fatal`: Fatal error
- `Failed to clone`: Clone failed
- `unable to access`: Cannot access URL

## Script Content

```bash
#!/bin/bash

# SKILL directory
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$SKILL_DIR/logs"
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
    echo "1. Executing command \`git submodule update --init --recursive\`" | tee -a "$LOG_FILE"
    TMP_FILE1=$(mktemp)
    git submodule update --init --recursive 2>&1 | tee -a "$LOG_FILE" | tee "$TMP_FILE1"
    OUTPUT1=$(cat "$TMP_FILE1")
    rm "$TMP_FILE1"
    
    # Command 2: git pull --recurse-submodules
    echo "2. Executing command \`git pull --recurse-submodules\`" | tee -a "$LOG_FILE"
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
```

## Notes

1. The script automatically stops after all code is successfully pulled
2. If the network is unstable, the script will automatically retry
3. All output is displayed in both terminal and log file
4. The log file records detailed process of each pull

