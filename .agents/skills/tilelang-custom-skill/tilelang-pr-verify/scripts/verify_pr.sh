#!/bin/bash

# ================= Usage =================
# bash verify_pr.sh --pr <url|number> [--repo <owner/repo>] [--backend <auto|ascendc|pto|both|ascendc,pto>]
#                   [--project-root <path>] [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]]
#                   [--max-jobs N] [--output-dir <path>] [--task-timeout <seconds>] [--build-timeout <seconds>]
#                   [--pytest-timeout <seconds>]
#
# Options:
#   --pr <url|number>           PR URL (https://github.com/owner/repo/pull/N) or plain number
#   --repo <owner/repo>         Required when --pr is a plain number
#   --backend <...>             Compilation backend (default: auto)
#                               Single: auto | ascendc | pto
#                               Multi:  both | ascendc,pto (comma-separated, order preserved)
#                               Multi-backend runs each backend for before/after; each backend gets
#                               its own subdirectory (logs, Excel, Markdown report).
#   --project-root <path>       Project root directory (defaults to cwd)
#   --skip-aclgraph[=true|false]  Skip aclgraph examples (default: true)
#   --skip-pytest[=true|false]  Skip pytest phase (default: true)
#   --max-jobs N                Max parallel jobs (default: 8)
#   --output-dir <path>         Output directory path (default: <cwd>/tmp/pr_verify_<timestamp>_<pr_number>)
#   --task-timeout <secs>       Per-task timeout forwarded to run_examples.sh (default: 600).
#                               A hung example is killed and marked [TIMEOUT]. 0 disables.
#   --build-timeout <secs>      Timeout for each build phase (make / install_ascend.sh) (default: 1800).
#                               A hung build is killed (SIGTERM, then SIGKILL after 60s grace). 0 disables.
#   --pytest-timeout <secs>     Overall pytest phase timeout forwarded to run_examples.sh (default: 1800).
#                               Independent of --task-timeout; pytest runs hundreds of cases and needs a
#                               larger budget than a single bench script. 0 disables.
# ================= ========== =================

# ================= Parameter Defaults =================
PR_INPUT=""
REPO_INPUT=""
BACKEND="auto"
PROJECT_ROOT_ARG=""
SKIP_ACLGRAPH=true
SKIP_PYTEST=true
MAX_JOBS=8
OUTPUT_DIR=""
TASK_TIMEOUT=600
BUILD_TIMEOUT=1800
PYTEST_TIMEOUT=1800

# ================= Parameter Parsing =================
while [[ $# -gt 0 ]]; do
    case $1 in
        --pr)
            PR_INPUT="$2"
            shift 2
            ;;
        --repo)
            REPO_INPUT="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            # Validate: single value, "both", or comma-separated list
            if [[ "$BACKEND" != "auto" && "$BACKEND" != "ascendc" && "$BACKEND" != "pto" \
                  && "$BACKEND" != "both" && "$BACKEND" != *,* ]]; then
                echo "Error: --backend must be 'auto', 'ascendc', 'pto', 'both', or comma-separated (e.g. 'ascendc,pto')" >&2
                exit 1
            fi
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT_ARG="$2"
            shift 2
            ;;
        --skip-aclgraph|--skip-aclgraph=true)
            SKIP_ACLGRAPH=true
            shift
            ;;
        --skip-aclgraph=false)
            SKIP_ACLGRAPH=false
            shift
            ;;
        --skip-pytest|--skip-pytest=true)
            SKIP_PYTEST=true
            shift
            ;;
        --skip-pytest=false)
            SKIP_PYTEST=false
            shift
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task-timeout)
            TASK_TIMEOUT="$2"
            if ! [[ "$TASK_TIMEOUT" =~ ^[0-9]+$ ]]; then
                echo "Error: --task-timeout must be a non-negative integer" >&2
                exit 1
            fi
            shift 2
            ;;
        --build-timeout)
            BUILD_TIMEOUT="$2"
            if ! [[ "$BUILD_TIMEOUT" =~ ^[0-9]+$ ]]; then
                echo "Error: --build-timeout must be a non-negative integer" >&2
                exit 1
            fi
            shift 2
            ;;
        --pytest-timeout)
            PYTEST_TIMEOUT="$2"
            if ! [[ "$PYTEST_TIMEOUT" =~ ^[0-9]+$ ]]; then
                echo "Error: --pytest-timeout must be a non-negative integer" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$PR_INPUT" ]]; then
    echo "Error: --pr is required (PR URL or number)" >&2
    exit 1
fi

# ================= Resolve Backend List =================
if [[ "$BACKEND" == "both" ]]; then
    BACKENDS=("ascendc" "pto")
    IS_MULTI=true
elif [[ "$BACKEND" == *,* ]]; then
    IFS=',' read -ra BACKENDS <<< "$BACKEND"
    for b in "${BACKENDS[@]}"; do
        if [[ "$b" != "auto" && "$b" != "ascendc" && "$b" != "pto" ]]; then
            echo "Error: invalid backend '$b' in '--backend $BACKEND'" >&2
            echo "  Allowed: auto, ascendc, pto (comma-separated) or 'both'" >&2
            exit 1
        fi
    done
    IS_MULTI=true
else
    BACKENDS=("$BACKEND")
    IS_MULTI=false
fi

# ================= Path Setup =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
SKILLS_DIR="$(dirname "$SKILL_DIR")"
RUN_EXAMPLES_PATH="$SKILLS_DIR/tilelang-run-examples/scripts"

if [[ ! -f "$RUN_EXAMPLES_PATH/run_examples.sh" ]]; then
    echo "Error: run_examples.sh not found at $RUN_EXAMPLES_PATH" >&2
    echo "  Ensure tilelang-run-examples skill is installed" >&2
    exit 1
fi

if [[ -n "$PROJECT_ROOT_ARG" ]]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT_ARG" && pwd)"
else
    PROJECT_ROOT="$(pwd)"
fi

if [[ ! -f "$PROJECT_ROOT/set_env.sh" ]]; then
    echo "Error: Cannot find set_env.sh in PROJECT_ROOT=$PROJECT_ROOT" >&2
    echo "  Use --project-root <path> to specify the project root directory" >&2
    exit 1
fi

# ================= PR Input Parsing =================
parse_pr_input() {
    local input="$1"
    if [[ "$input" =~ ^https?://github\.com/([^/]+)/([^/]+)/pull/([0-9]+) ]]; then
        PR_OWNER="${BASH_REMATCH[1]}"
        PR_REPO="${BASH_REMATCH[2]}"
        PR_NUMBER="${BASH_REMATCH[3]}"
        PR_REPO_FULL="${PR_OWNER}/${PR_REPO}"
        return 0
    fi
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        PR_NUMBER="$input"
        if [[ -z "$REPO_INPUT" ]]; then
            echo "Error: --repo <owner/repo> is required when --pr is a plain number" >&2
            exit 1
        fi
        PR_REPO_FULL="$REPO_INPUT"
        PR_OWNER="${REPO_INPUT%%/*}"
        PR_REPO="${REPO_INPUT#*/}"
        return 0
    fi
    echo "Error: Invalid PR input: $input" >&2
    echo "  Expected: https://github.com/owner/repo/pull/N or a plain number" >&2
    exit 1
}

parse_pr_input "$PR_INPUT"

echo "====================================="
echo "PR Verify Configuration"
echo "  PR: ${PR_REPO_FULL}#${PR_NUMBER}"
if [[ "$IS_MULTI" == true ]]; then
    echo "  Backends: ${BACKENDS[*]} (multi)"
else
    echo "  Backend: ${BACKENDS[0]}"
fi
echo "  Skip aclgraph: $SKIP_ACLGRAPH"
echo "  Pytest: $([ "$SKIP_PYTEST" = true ] && echo "skipped (--skip-pytest, default)" || echo "enabled (--skip-pytest=false)")"
echo "  Max jobs: $MAX_JOBS"
if [[ "$TASK_TIMEOUT" -gt 0 ]]; then
    echo "  Task timeout: ${TASK_TIMEOUT}s"
else
    echo "  Task timeout: disabled"
fi
if [[ "$BUILD_TIMEOUT" -gt 0 ]]; then
    echo "  Build timeout: ${BUILD_TIMEOUT}s"
else
    echo "  Build timeout: disabled"
fi
if [[ "$PYTEST_TIMEOUT" -gt 0 ]]; then
    echo "  Pytest timeout: ${PYTEST_TIMEOUT}s"
else
    echo "  Pytest timeout: disabled"
fi
echo "  Project root: $PROJECT_ROOT"
echo "====================================="

# ================= Network Retry Helpers =================
# Git stall-detection opts: lowSpeedLimit=1KB/s, lowSpeedTime=<N>s
# Injected via `git -c` per-command; does not modify global/repo config.
# Exponential backoff: 2->4->8->16->30s (capped at 30s) between retries.

# run_net_retry <hard_timeout_secs> <max_retries> <desc> -- <cmd...>
# Streams command output (prefixed). Progress to stdout. Returns exit code.
# hard_timeout=0 disables hard timeout (stall detection via git -c still applies).
run_net_retry() {
    local hard_timeout="$1" max_retries="$2" desc="$3"
    shift 3
    [[ "$1" == "--" ]] && shift
    local attempt=1 ec
    while [[ $attempt -le $max_retries ]]; do
        echo "  [$attempt/$max_retries] $desc ..."
        if [[ "$hard_timeout" -gt 0 ]]; then
            timeout --kill-after 5s "$hard_timeout" "$@" 2>&1 | sed 's/^/    /'
            ec=${PIPESTATUS[0]}
        else
            "$@" 2>&1 | sed 's/^/    /'
            ec=${PIPESTATUS[0]}
        fi
        [[ $ec -eq 0 ]] && return 0
        if [[ $ec -eq 124 ]]; then
            echo "  [$attempt/$max_retries] TIMEOUT after ${hard_timeout}s"
        else
            echo "  [$attempt/$max_retries] FAILED (exit $ec)"
        fi
        attempt=$((attempt + 1))
        if [[ $attempt -le $max_retries ]]; then
            local backoff=$((2 ** (attempt - 1)))
            [[ $backoff -gt 30 ]] && backoff=30
            echo "  Retrying in ${backoff}s..."
            sleep "$backoff"
        fi
    done
    echo "  Error: $desc failed after $max_retries attempts"
    return 1
}

# run_net_retry_capture <hard_timeout_secs> <max_retries> <desc> -- <cmd...>
# Captures command output. On success, prints output to stdout (for $(...) capture).
# Progress and error messages to stderr. Returns exit code.
run_net_retry_capture() {
    local hard_timeout="$1" max_retries="$2" desc="$3"
    shift 3
    [[ "$1" == "--" ]] && shift
    local attempt=1 out ec
    while [[ $attempt -le $max_retries ]]; do
        echo "  [$attempt/$max_retries] $desc ..." >&2
        if [[ "$hard_timeout" -gt 0 ]]; then
            out=$(timeout --kill-after 5s "$hard_timeout" "$@" 2>&1)
            ec=$?
        else
            out=$("$@" 2>&1)
            ec=$?
        fi
        [[ $ec -eq 0 ]] && { printf '%s' "$out"; return 0; }
        if [[ $ec -eq 124 ]]; then
            echo "  [$attempt/$max_retries] TIMEOUT after ${hard_timeout}s" >&2
        else
            echo "  [$attempt/$max_retries] FAILED (exit $ec)" >&2
        fi
        echo "$out" | tail -3 | sed 's/^/    /' >&2
        attempt=$((attempt + 1))
        if [[ $attempt -le $max_retries ]]; then
            local backoff=$((2 ** (attempt - 1)))
            [[ $backoff -gt 30 ]] && backoff=30
            echo "  Retrying in ${backoff}s..." >&2
            sleep "$backoff"
        fi
    done
    echo "  Error: $desc failed after $max_retries attempts" >&2
    return 1
}

# ================= PR Metadata via gh =================
if ! command -v gh &>/dev/null; then
    echo "Error: gh (GitHub CLI) not found. Install: https://cli.github.com/" >&2
    exit 1
fi

echo ""
echo "Fetching PR metadata..."
PR_JSON=$(run_net_retry_capture 30 5 "fetch PR metadata" -- \
    gh pr view "$PR_NUMBER" --repo "$PR_REPO_FULL" \
    --json baseRefOid,headRefOid,baseRefName,title,url) || exit 1

BASE_REF_OID=$(echo "$PR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['baseRefOid'])")
HEAD_REF_OID=$(echo "$PR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['headRefOid'])")
BASE_REF_NAME=$(echo "$PR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['baseRefName'])")
PR_TITLE=$(echo "$PR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['title'])")
PR_URL=$(echo "$PR_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['url'])")

echo "  Title: $PR_TITLE"
echo "  URL: $PR_URL"
echo "  Base: $BASE_REF_NAME ($BASE_REF_OID)"
echo "  Head: $HEAD_REF_OID"

# ================= Remote Mapping =================
map_remote() {
    local owner_repo="$1"
    local remote
    remote=$(git -C "$PROJECT_ROOT" remote -v | grep -E "github\.com[/:]${owner_repo}(\.git|/|$| )" | head -1 | awk '{print $1}')
    echo "$remote"
}

REMOTE=$(map_remote "$PR_REPO_FULL")
if [[ -z "$REMOTE" ]]; then
    echo "Error: No git remote found for $PR_REPO_FULL" >&2
    echo "  Available remotes:" >&2
    git -C "$PROJECT_ROOT" remote -v >&2
    exit 1
fi
echo "  Remote: $REMOTE -> $PR_REPO_FULL"

# ================= Fetch Commits =================
echo ""
echo "Fetching commits..."
run_net_retry 120 5 "fetch PR head (pull/${PR_NUMBER}/head)" -- \
    git -C "$PROJECT_ROOT" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=10 \
    fetch "$REMOTE" "pull/${PR_NUMBER}/head" || exit 1
run_net_retry 120 5 "fetch base branch ($BASE_REF_NAME)" -- \
    git -C "$PROJECT_ROOT" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=10 \
    fetch "$REMOTE" "$BASE_REF_NAME" || exit 1

if ! git -C "$PROJECT_ROOT" cat-file -e "${HEAD_REF_OID}^{commit}" 2>/dev/null; then
    echo "Error: Head commit $HEAD_REF_OID not available locally" >&2
    exit 1
fi

if ! git -C "$PROJECT_ROOT" cat-file -e "${BASE_REF_OID}^{commit}" 2>/dev/null; then
    echo "  Base commit not found locally, trying all remotes..."
    for r in $(git -C "$PROJECT_ROOT" remote); do
        timeout --kill-after 5s 60 \
            git -C "$PROJECT_ROOT" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=10 \
            fetch "$r" 2>/dev/null || true
        if git -C "$PROJECT_ROOT" cat-file -e "${BASE_REF_OID}^{commit}" 2>/dev/null; then
            echo "  Found via remote: $r"
            break
        fi
    done
    if ! git -C "$PROJECT_ROOT" cat-file -e "${BASE_REF_OID}^{commit}" 2>/dev/null; then
        echo "Error: Base commit $BASE_REF_OID not available after fetching all remotes" >&2
        exit 1
    fi
fi

BEFORE_SHA=$(git -C "$PROJECT_ROOT" merge-base "$BASE_REF_OID" "$HEAD_REF_OID")
AFTER_SHA="$HEAD_REF_OID"

echo "  BEFORE (merge-base): $BEFORE_SHA"
echo "  AFTER  (head):       $AFTER_SHA"

# ================= Rebuild Detection =================
CHANGED_FILES=$(git -C "$PROJECT_ROOT" diff --name-only "$BEFORE_SHA" "$AFTER_SHA")

NEEDS_REBUILD=false
NEEDS_SUBMODULE_UPDATE=false

while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    if [[ "$file" =~ ^src/.*\.(cc|cpp|cxx|c|h|hpp)$ ]] || \
       [[ "$file" == "CMakeLists.txt" ]] || \
       [[ "$file" =~ ^cmake/.*\.cmake$ ]] || \
       [[ "$file" == "build/config.cmake" ]]; then
        NEEDS_REBUILD=true
    fi
    if [[ "$file" =~ ^3rdparty/ ]]; then
        NEEDS_SUBMODULE_UPDATE=true
        NEEDS_REBUILD=true
    fi
done <<< "$CHANGED_FILES"

echo ""
echo "Change Analysis:"
echo "  Changed files: $(echo "$CHANGED_FILES" | grep -c .)"
echo "  Needs rebuild: $NEEDS_REBUILD"
echo "  Needs submodule update: $NEEDS_SUBMODULE_UPDATE"
if [[ "$NEEDS_REBUILD" == true ]]; then
    echo "  Rebuild-triggering files:"
    echo "$CHANGED_FILES" | grep -E '^src/.*\.(cc|cpp|cxx|c|h|hpp)$|^CMakeLists\.txt$|^cmake/.*\.cmake$|^build/config\.cmake$|^3rdparty/' | sed 's/^/    /'
fi

# ================= Output Directory =================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(pwd)/tmp/pr_verify_${TIMESTAMP}_${PR_NUMBER}"
fi
mkdir -p "$OUTPUT_DIR"
# Normalize to absolute path: do_run() does `(cd "$OUTPUT_DIR" && ... | tee "$log_path")`,
# so a relative --output-dir would resolve log_path against the new cwd and write to a
# non-existent nested path (e.g. my_dir/my_dir/ascendc/before.log).
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Protect OUTPUT_DIR (and pr_verify_* logs) from `git stash --include-untracked`
# in save_state(): empty/untracked dirs inside the repo would be removed by stash
# and not restored on pop (git cannot store empty dirs). Adding the pattern to
# .git/info/exclude makes them "ignored" so stash skips them.
EXCLUDE_FILE="$PROJECT_ROOT/.git/info/exclude"
if [[ -f "$EXCLUDE_FILE" ]] && ! grep -q '^pr_verify_' "$EXCLUDE_FILE"; then
    printf '\n# tilelang-pr-verify outputs\npr_verify_*\n' >> "$EXCLUDE_FILE"
fi

echo ""
echo "Output directory: $OUTPUT_DIR"

# ================= State Protection =================
ORIG_BRANCH=""
ORIG_SHA=""
STASHED=false
RESTORED=false

save_state() {
    cd "$PROJECT_ROOT"
    ORIG_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    ORIG_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")

    if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo "  Stashing uncommitted changes..."
        # Compare refs/stash before and after the push: `git stash push` may
        # create no new stash entry when changes are gitignored or otherwise
        # not stashable. Without this guard, STASHED would be set to true
        # unconditionally, and restore_state() would `git stash pop` a
        # pre-existing user stash — causing data loss. Only mark STASHED when
        # the stash ref actually changed.
        local stash_before stash_after
        stash_before=$(git rev-parse --verify refs/stash 2>/dev/null || echo "")
        git stash push --include-untracked -m "pr_verify_stash_${TIMESTAMP}" 2>&1 | sed 's/^/  /'
        stash_after=$(git rev-parse --verify refs/stash 2>/dev/null || echo "")
        if [[ "$stash_before" != "$stash_after" ]]; then
            STASHED=true
        else
            echo "  (no stash created — nothing stashed, skipping pop on exit)"
        fi
    fi
}

restore_state() {
    if [[ "$RESTORED" == true ]]; then
        return
    fi
    RESTORED=true

    cd "$PROJECT_ROOT" 2>/dev/null || return
    if [[ -n "$ORIG_BRANCH" ]]; then
        git checkout "$ORIG_BRANCH" >/dev/null 2>&1
    elif [[ -n "$ORIG_SHA" ]]; then
        git checkout --detach "$ORIG_SHA" >/dev/null 2>&1
    fi

    if [[ "$STASHED" == true ]]; then
        echo "  Restoring stashed changes..."
        if ! git stash pop >/dev/null 2>&1; then
            echo "  WARNING: 'git stash pop' failed (possible conflicts from build artifacts)." >&2
            echo "  Your changes are preserved in the stash. Run 'git stash list' to locate and resolve manually." >&2
        fi
    fi
}

trap restore_state EXIT INT TERM

# ================= Helper: Build & Run =================
do_build() {
    if [[ "$NEEDS_REBUILD" != true ]]; then
        echo "  Skipping rebuild (no C++ changes detected)"
        return 0
    fi

    echo "  Rebuilding (incremental)..."
    if [ -d "build" ] && [ -f "build/Makefile" ]; then
        local build_output
        if [[ "$BUILD_TIMEOUT" -gt 0 ]]; then
            build_output=$(cd build && timeout --kill-after 60s "$BUILD_TIMEOUT" make -j$(nproc) 2>&1)
        else
            build_output=$(cd build && make -j$(nproc) 2>&1)
        fi
        local build_exit=$?
        echo "$build_output" | tail -5 | sed 's/^/  /'
        if [[ $build_exit -eq 124 ]]; then
            echo "  Build TIMEOUT after ${BUILD_TIMEOUT}s" >&2
            return 1
        fi
        if [[ $build_exit -ne 0 ]]; then
            echo "  Incremental make failed (exit $build_exit), trying install_ascend.sh --enable-incremental..."
            if [[ "$BUILD_TIMEOUT" -gt 0 ]]; then
                timeout --kill-after 60s "$BUILD_TIMEOUT" bash install_ascend.sh --enable-incremental 2>&1 | tail -10 | sed 's/^/  /'
            else
                bash install_ascend.sh --enable-incremental 2>&1 | tail -10 | sed 's/^/  /'
            fi
            local inc_exit=${PIPESTATUS[0]}
            if [[ $inc_exit -eq 124 ]]; then
                echo "  install_ascend.sh (incremental) TIMEOUT after ${BUILD_TIMEOUT}s" >&2
                return 1
            fi
            if [[ $inc_exit -ne 0 ]]; then
                echo "Error: Build failed" >&2
                return 1
            fi
        fi
    else
        echo "  build/ not found or no Makefile, running install_ascend.sh..."
        if [[ "$BUILD_TIMEOUT" -gt 0 ]]; then
            timeout --kill-after 60s "$BUILD_TIMEOUT" bash install_ascend.sh 2>&1 | tail -10 | sed 's/^/  /'
        else
            bash install_ascend.sh 2>&1 | tail -10 | sed 's/^/  /'
        fi
        local full_exit=${PIPESTATUS[0]}
        if [[ $full_exit -eq 124 ]]; then
            echo "  install_ascend.sh TIMEOUT after ${BUILD_TIMEOUT}s" >&2
            return 1
        fi
        if [[ $full_exit -ne 0 ]]; then
            echo "Error: Build failed" >&2
            return 1
        fi
    fi
    return 0
}

do_run() {
    local label="$1"
    local backend="$2"
    local log_path="$3"

    echo "  Clearing kernel cache..."
    rm -rf ~/.tilelang/cache

    echo "  Running examples (backend=$backend)..."
    local skip_flag
    if [[ "$SKIP_ACLGRAPH" == true ]]; then
        skip_flag="--skip-aclgraph"
    else
        skip_flag="--skip-aclgraph=false"
    fi

    local pytest_flag
    if [[ "$SKIP_PYTEST" == true ]]; then
        pytest_flag="--skip-pytest"
    else
        pytest_flag="--skip-pytest=false"
    fi

    mkdir -p "$(dirname "$log_path")"

    (cd "$OUTPUT_DIR" && bash "$RUN_EXAMPLES_PATH/run_examples.sh" \
        --backend "$backend" \
        --project-root "$PROJECT_ROOT" \
        $skip_flag \
        $pytest_flag \
        --max-jobs "$MAX_JOBS" \
        --task-timeout "$TASK_TIMEOUT" \
        --pytest-timeout "$PYTEST_TIMEOUT" \
        2>&1 | tee "$log_path") || true

    local summary
    summary=$(grep -E "^Total: " "$log_path" | tail -1)
    if [[ -n "$summary" ]]; then
        echo "  $label summary: $summary"
    else
        echo "  $label summary: (not found in log)"
    fi
}

build_and_run() {
    local sha="$1"
    local label="$2"
    local backend="$3"
    local log_path="$4"

    echo ""
    echo "====================================="
    echo "Phase: $label ($sha) [backend=$backend]"
    echo "====================================="

    cd "$PROJECT_ROOT"
    git checkout --detach "$sha" 2>&1 | sed 's/^/  /'

    if [[ "$NEEDS_SUBMODULE_UPDATE" == true ]]; then
        echo "  Updating submodules..."
        run_net_retry 600 3 "submodule update --init --recursive" -- \
            git -C "$PROJECT_ROOT" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=15 \
            submodule update --init --recursive || exit 1
    fi

    do_build || exit 1
    do_run "$label" "$backend" "$log_path"
}

# ================= Phase 1: BEFORE (merge-base) =================
echo ""
echo "Saving current git state..."
save_state
echo "  Original branch: ${ORIG_BRANCH:-detached HEAD}"
echo "  Original SHA: $ORIG_SHA"

TOTAL_START_EPOCH=$(date +%s)

for b in "${BACKENDS[@]}"; do
    if [[ "$IS_MULTI" == true ]]; then
        before_log="$OUTPUT_DIR/$b/before.log"
    else
        before_log="$OUTPUT_DIR/before.log"
    fi
    build_and_run "$BEFORE_SHA" "BEFORE (merge-base)" "$b" "$before_log"
done

# ================= Phase 2: AFTER (head) =================
for b in "${BACKENDS[@]}"; do
    if [[ "$IS_MULTI" == true ]]; then
        after_log="$OUTPUT_DIR/$b/after.log"
    else
        after_log="$OUTPUT_DIR/after.log"
    fi
    build_and_run "$AFTER_SHA" "AFTER (head)" "$b" "$after_log"
done

# ================= Restore State =================
TOTAL_END_EPOCH=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END_EPOCH - TOTAL_START_EPOCH))

echo ""
echo "====================================="
echo "Restoring original git state..."
restore_state
trap - EXIT INT TERM
echo "  Restored to: ${ORIG_BRANCH:-$ORIG_SHA}"

# ================= Excel Export =================
echo ""
echo "====================================="
echo "Exporting Excel report (before=Round1, after=Round2)..."
for b in "${BACKENDS[@]}"; do
    if [[ "$IS_MULTI" == true ]]; then
        b_dir="$OUTPUT_DIR/$b"
    else
        b_dir="$OUTPUT_DIR"
    fi
    echo "  [$b] Exporting..."
    (cd "$b_dir" && \
        python3 "$RUN_EXAMPLES_PATH/export_to_excel.py" --log before.log --backend "$b" --output-dir . 2>&1 | sed 's/^/    /' && \
        python3 "$RUN_EXAMPLES_PATH/export_to_excel.py" --log after.log --backend "$b" --output-dir . 2>&1 | sed 's/^/    /')
done

# ================= Report Generation =================
echo ""
echo "====================================="
echo "Generating Markdown report..."
for b in "${BACKENDS[@]}"; do
    if [[ "$IS_MULTI" == true ]]; then
        b_dir="$OUTPUT_DIR/$b"
        report_file="$b_dir/pr_verify_report.md"
    else
        b_dir="$OUTPUT_DIR"
        report_file="$b_dir/pr_verify_report.md"
    fi
    echo "  [$b] Generating report..."
    python3 "$SCRIPT_DIR/generate_report.py" \
        --before-log "$b_dir/before.log" \
        --after-log "$b_dir/after.log" \
        --pr-url "$PR_URL" \
        --pr-title "$PR_TITLE" \
        --backend "$b" \
        --before-sha "$BEFORE_SHA" \
        --after-sha "$AFTER_SHA" \
        --needs-rebuild "$NEEDS_REBUILD" \
        --output "$report_file" 2>&1 | sed 's/^/    /'
done

# ================= Multi-backend Summary Report =================
if [[ "$IS_MULTI" == true ]]; then
    summary_report="$OUTPUT_DIR/pr_verify_report.md"
    echo "  [summary] Generating multi-backend summary report..."
    python3 "$SCRIPT_DIR/generate_report.py" --multi \
        --output-dir "$OUTPUT_DIR" \
        --pr-url "$PR_URL" \
        --pr-title "$PR_TITLE" \
        --before-sha "$BEFORE_SHA" \
        --after-sha "$AFTER_SHA" \
        --needs-rebuild "$NEEDS_REBUILD" \
        --output "$summary_report" 2>&1 | sed 's/^/    /'
fi

# ================= Final Summary =================
echo ""
echo "====================================="
echo "PR Verify Complete!"
echo "====================================="
echo "  PR: $PR_TITLE"
echo "  PR URL: $PR_URL"
if [[ "$IS_MULTI" == true ]]; then
    echo "  Backends: ${BACKENDS[*]}"
else
    echo "  Backend: ${BACKENDS[0]}"
fi
echo "  Before (merge-base): $BEFORE_SHA"
echo "  After (head): $AFTER_SHA"
echo "  Total elapsed: $(printf '%dm%02ds' $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60)))"
echo ""
echo "  Reports:"
if [[ "$IS_MULTI" == true ]]; then
    echo "    [summary]"
    echo "      Markdown: $OUTPUT_DIR/pr_verify_report.md"
fi
for b in "${BACKENDS[@]}"; do
    if [[ "$IS_MULTI" == true ]]; then
        b_dir="$OUTPUT_DIR/$b"
        echo "    [$b]"
        echo "      Excel:    $b_dir/run_examples_results.xlsx"
        echo "      Markdown: $b_dir/pr_verify_report.md"
        echo "      Logs:     $b_dir/before.log, $b_dir/after.log"
    else
        echo "    Excel:    $OUTPUT_DIR/run_examples_results.xlsx"
        echo "    Markdown: $OUTPUT_DIR/pr_verify_report.md"
        echo "    Logs:     $OUTPUT_DIR/before.log, $OUTPUT_DIR/after.log"
    fi
done
echo "====================================="
