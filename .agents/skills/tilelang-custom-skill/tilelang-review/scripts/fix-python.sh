#!/usr/bin/env bash
# Fix Python file format issues using ruff
#
# Usage:
#   fix-python.sh          # Fix changed files only (default)
#   fix-python.sh --all    # Fix all files in the repository

set -euo pipefail

# Parse arguments
CHECK_ALL=false
for arg in "$@"; do
    case $arg in
        --all)
            CHECK_ALL=true
            shift
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"

cd "$REPO_ROOT"

# Get files to fix based on mode
if [ "$CHECK_ALL" = true ]; then
    # Get all Python files in the repository
    PYTHON_FILES=$(find . -type f \( -name "*.py" -o -name "*.pyi" \) \
        ! -path "./.git/*" \
        ! -path "./build/*" \
        ! -path "./.venv/*" \
        ! -path "./venv/*" \
        ! -path "./__pycache__/*" \
        ! -path "./.tox/*" \
        ! -path "./node_modules/*" \
        2>/dev/null | sed 's|^\./||' || true)
else
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
fi

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files to fix."
    exit 0
fi

# Convert to array
FILE_ARRAY=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$REPO_ROOT/$file" ]; then
        FILE_ARRAY+=("$file")
    fi
done <<< "$PYTHON_FILES"

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo "No Python files to fix."
    exit 0
fi

echo "Fixing ${#FILE_ARRAY[@]} Python file(s)..."

# Run ruff check --fix
echo "Running ruff check --fix..."
ruff check --fix "${FILE_ARRAY[@]}" || true

# Run ruff format
echo "Running ruff format..."
ruff format "${FILE_ARRAY[@]}"

echo "Python files have been fixed."