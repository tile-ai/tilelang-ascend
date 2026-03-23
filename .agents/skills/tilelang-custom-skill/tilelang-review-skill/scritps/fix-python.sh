#!/usr/bin/env bash
# Fix Python file format issues using ruff

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

PYTHON_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(py|pyi)$' || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files to fix."
    exit 0
fi

# Convert to array
FILE_ARRAY=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$REPO_ROOT/$file" ]; then
        FILE_ARRAY+=("$REPO_ROOT/$file")
    fi
done <<< "$PYTHON_FILES"

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo "No Python files to fix."
    exit 0
fi

cd "$REPO_ROOT"

echo "Fixing ${#FILE_ARRAY[@]} Python file(s)..."

# Run ruff check --fix
echo "Running ruff check --fix..."
ruff check --fix "${FILE_ARRAY[@]}" || true

# Run ruff format
echo "Running ruff format..."
ruff format "${FILE_ARRAY[@]}"

echo "Python files have been fixed."
