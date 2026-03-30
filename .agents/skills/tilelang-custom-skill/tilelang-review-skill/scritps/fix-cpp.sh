#!/usr/bin/env bash
# Fix C++ file format issues using clang-format

set -euo pipefail

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Get changed files from git status
# Using --porcelain to get machine-readable output
# Format: "XY filename" where XY are status codes (M=modified, A=added, R=renamed, C=copied, ?=untracked)
# For renamed files: "R  old -> new", we extract the new filename
# Exclude deleted files (D in status)
CHANGED_FILES=$(git status --porcelain --untracked-files=all 2>/dev/null | grep -vE '^\s*D\s' | awk '{
    if ($1 ~ /^R/) {
        print $4  # renamed file: "R  old -> new", $4 is new filename
    } else if ($1 ~ /^[MADRC?]/ || $1 ~ /^\?\?/) {
        print $2  # normal files: "XY filename", $2 is filename
    }
}' || echo "")

CPP_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(c|cc|cpp|h|hpp|hh|icc)$' || true)

if [ -z "$CPP_FILES" ]; then
    echo "No C++ files to fix."
    exit 0
fi

# Convert to array
FILE_ARRAY=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$REPO_ROOT/$file" ]; then
        FILE_ARRAY+=("$REPO_ROOT/$file")
    fi
done <<< "$CPP_FILES"

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo "No C++ files to fix."
    exit 0
fi

# Check if clang-format is available
if ! command -v clang-format &>/dev/null; then
    echo "Error: clang-format not found. Please install clang-format."
    exit 1
fi

cd "$REPO_ROOT"

echo "Fixing ${#FILE_ARRAY[@]} C++ file(s)..."

# Run clang-format -i with style from .clang-format file
clang-format -i --style=file "${FILE_ARRAY[@]}"

echo "C++ files have been fixed."
