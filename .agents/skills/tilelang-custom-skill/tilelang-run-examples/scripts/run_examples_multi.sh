#!/bin/bash

# ================= Usage =================
# bash run_examples_multi.sh --backend <both|ascendc,pto|auto|ascendc|pto>
#                            [--output-dir <path>] [other run_examples.sh options...]
#
# Wrapper around run_examples.sh that runs one or more backends in sequence.
#
# --backend values:
#   both                 Run ascendc then pto
#   ascendc,pto          Run ascendc then pto (comma-separated, order preserved)
#   auto|ascendc|pto     Single backend (delegates to run_examples.sh directly)
#
# --output-dir <path>    Directory for per-backend log files (default: ./tmp)
#                        Each backend's output is teed to <output-dir>/run_<backend>.log
#                        This option is NOT forwarded to run_examples.sh (which doesn't support it).
#
# All other options (--skip-aclgraph, --skip-pytest, --dirs, --max-jobs, --project-root,
# --task-timeout, etc.) are forwarded verbatim to each run_examples.sh invocation.
#
# Exit code: 0 only if all backends pass; 1 if any backend has failures.
# ================= ========== =================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXAMPLES="$SCRIPT_DIR/run_examples.sh"

if [[ ! -f "$RUN_EXAMPLES" ]]; then
    echo "Error: run_examples.sh not found at $RUN_EXAMPLES" >&2
    exit 1
fi

BACKEND_INPUT=""
OUTPUT_DIR="./tmp"
PASSED_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND_INPUT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            PASSED_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore empty array if no extra args (avoids unbound variable under set -u).
if [[ ${#PASSED_ARGS[@]} -eq 0 ]]; then
    PASSED_ARGS=()
fi

# ================= Resolve Backend List =================
if [[ -z "$BACKEND_INPUT" || "$BACKEND_INPUT" == "auto" || "$BACKEND_INPUT" == "ascendc" || "$BACKEND_INPUT" == "pto" ]]; then
    # Single backend: delegate directly, preserving original behavior exactly.
    exec "$RUN_EXAMPLES" --backend "${BACKEND_INPUT:-auto}" "${PASSED_ARGS[@]}"
fi

if [[ "$BACKEND_INPUT" == "both" ]]; then
    BACKENDS=("ascendc" "pto")
else
    IFS=',' read -ra BACKENDS <<< "$BACKEND_INPUT"
    for b in "${BACKENDS[@]}"; do
        if [[ "$b" != "auto" && "$b" != "ascendc" && "$b" != "pto" ]]; then
            echo "Error: invalid backend '$b' in '--backend $BACKEND_INPUT'" >&2
            echo "  Allowed: auto, ascendc, pto (comma-separated) or 'both'" >&2
            exit 1
        fi
    done
fi

mkdir -p "$OUTPUT_DIR"

OVERALL_EXIT=0

echo "====================================="
echo "Multi-Backend Run"
echo "  Backends: ${BACKENDS[*]}"
echo "  Output dir: $OUTPUT_DIR"
echo "====================================="

for b in "${BACKENDS[@]}"; do
    echo ""
    echo "#####################################"
    echo "# Backend: $b"
    echo "#####################################"

    LOG="$OUTPUT_DIR/run_${b}.log"

    # run_examples.sh exits non-zero on failures; don't abort the loop.
    bash "$RUN_EXAMPLES" --backend "$b" "${PASSED_ARGS[@]}" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}

    if [[ $rc -ne 0 ]]; then
        OVERALL_EXIT=1
    fi
done

# ================= Summary =================
echo ""
echo "====================================="
echo "Multi-Backend Summary"
echo "====================================="
for b in "${BACKENDS[@]}"; do
    LOG="$OUTPUT_DIR/run_${b}.log"
    summary=$(grep -E "^Total: [0-9]+ \| Passed: [0-9]+ \| Failed: [0-9]+" "$LOG" | tail -1)
    if [[ -n "$summary" ]]; then
        echo "  [$b] $summary"
    else
        echo "  [$b] (summary not found in $LOG)"
    fi
done
echo "====================================="

exit $OVERALL_EXIT
