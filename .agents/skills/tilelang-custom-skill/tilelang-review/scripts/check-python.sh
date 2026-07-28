#!/usr/bin/env bash
# Check Python files for format issues using ruff
# Outputs results in JSON format for processing
#
# Usage:
#   check-python.sh          # Check changed files only (default)
#   check-python.sh --all    # Check all files in the repository

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

# Check if ruff is installed
if ! command -v ruff &>/dev/null; then
    echo '{"error": "ruff not found", "message": "Installing ruff...", "install_command": "curl -LsSf https://astral.sh/ruff/install.sh | sh"}'
    exit 0
fi

cd "$REPO_ROOT"

# Get files to check based on mode
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

    # Filter for Python files only
    PYTHON_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(py|pyi)$' || true)
fi

if [ -z "$PYTHON_FILES" ]; then
    echo '{"issues": [], "format_issues": [], "files_checked": 0}'
    exit 0
fi

# Convert to array for processing
FILE_ARRAY=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$REPO_ROOT/$file" ]; then
        FILE_ARRAY+=("$file")
    fi
done <<< "$PYTHON_FILES"

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo '{"issues": [], "format_issues": [], "files_checked": 0}'
    exit 0
fi

# Run ruff check (linting)
# Note: ruff check returns non-zero exit code when issues found, but still outputs valid JSON
RUFF_CHECK_OUTPUT=$(ruff check --output-format=json "${FILE_ARRAY[@]}" 2>/dev/null || true)
# If output is empty, default to empty array
if [ -z "$RUFF_CHECK_OUTPUT" ]; then
    RUFF_CHECK_OUTPUT="[]"
fi

# Run ruff format check (formatting)
RUFF_FORMAT_OUTPUT=$(ruff format --check "${FILE_ARRAY[@]}" 2>&1 || true)

# Build JSON output
echo "{"
echo "  \"issues\": $RUFF_CHECK_OUTPUT,"

# Parse format output
if echo "$RUFF_FORMAT_OUTPUT" | grep -qi "reformat"; then
    # Extract files that need formatting from "Would reformat: filename" lines
    FORMAT_FILES=$(echo "$RUFF_FORMAT_OUTPUT" | grep -i "Would reformat:" | awk '{print "  \"" $NF "\""}' | paste -sd ',' -)
    if [ -n "$FORMAT_FILES" ]; then
        echo "  \"format_issues\": [$FORMAT_FILES],"
    else
        echo "  \"format_issues\": [],"
    fi
else
    echo "  \"format_issues\": [],"
fi

echo "  \"files_checked\": ${#FILE_ARRAY[@]}"
echo "}"