#!/usr/bin/env bash
# Check C++ files for format issues using clang-format
# Outputs results in JSON format for processing
#
# Usage:
#   check-cpp.sh          # Check changed files only (default)
#   check-cpp.sh --all    # Check all files in the repository

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

# Check if clang-format is installed
if ! command -v clang-format &>/dev/null; then
    echo '{"error": "clang-format not found", "message": "Please install clang-format", "install_command": "sudo apt-get install clang-format-18 || brew install clang-format@18 || pip install clang-format==18.1.8"}'
    exit 0
fi

cd "$REPO_ROOT"

# Get files to check based on mode
if [ "$CHECK_ALL" = true ]; then
    # Get all C++ files in the repository
    CPP_FILES=$(find . -type f \( -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cxx" -o -name "*.h" -o -name "*.hpp" -o -name "*.hh" -o -name "*.icc" \) \
        ! -path "./.git/*" \
        ! -path "./build/*" \
        ! -path "./third_party/*" \
        ! -path "./.venv/*" \
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

    # Filter for C++ files only
    CPP_FILES=$(echo "$CHANGED_FILES" | grep -E '\.(c|cc|cpp|cxx|h|hpp|hh|icc)$' || true)
fi

if [ -z "$CPP_FILES" ]; then
    echo '{"issues": [], "files_checked": 0}'
    exit 0
fi

# Convert to array and check existence
FILE_ARRAY=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$REPO_ROOT/$file" ]; then
        FILE_ARRAY+=("$file")
    fi
done <<< "$CPP_FILES"

if [ ${#FILE_ARRAY[@]} -eq 0 ]; then
    echo '{"issues": [], "files_checked": 0}'
    exit 0
fi

# Run clang-format in dry-run mode
# Collect files that need formatting
NEEDS_FORMATTING=()

for file in "${FILE_ARRAY[@]}"; do
    # Run clang-format and compare
    if ! clang-format --dry-run --Werror "$file" >/dev/null 2>&1; then
        # Generate diff for this file
        DIFF=$(clang-format --style=file "$file" | diff -u "$file" - 2>/dev/null || true)

        if [ -n "$DIFF" ]; then
            NEEDS_FORMATTING+=("$file")
        fi
    fi
done

# Build JSON output
echo "{"
echo "  \"issues\": ["

FIRST=true
for file in "${NEEDS_FORMATTING[@]}"; do
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ","
    fi
    printf '    {"file": "%s", "message": "File needs formatting"}' "$file"
done

echo ""
echo "  ],"
echo "  \"files_checked\": ${#FILE_ARRAY[@]}"
echo "}"