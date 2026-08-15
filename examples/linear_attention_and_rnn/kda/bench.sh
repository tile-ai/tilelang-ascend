#!/usr/bin/env bash
#
# bench.sh -- operator-level benchmark for the KDA L1 chunkwise forward pass.
#
# Collects on-board Ascend profiling data with `msprof op` for each of the six
# KDA L1 stage kernels and for the full six-stage pipeline, then prints one row
# per target.  Every number in the report comes from the profiler run that has
# just happened: no latency, throughput or speedup value is baked into this
# script, and nothing is printed when the profiler produced no data.
#
# Usage:
#   bash bench.sh                              # all six stages + full pipeline
#   bash bench.sh --list                       # list the targets and exit
#   bash bench.sh --only cumsum,chunk_o        # profile a subset of stages
#   bash bench.sh --skip-pipeline              # stages only
#   bash bench.sh --warmup 10 --launch-count 50
#   bash bench.sh --output ./msprof_output --python python3
#   bash bench.sh --kernel-name 'main_kernel'  # override the profiled symbol
#
# Environment variables (command line flags take precedence):
#   KDA_WARMUP KDA_LAUNCH_COUNT KDA_OUTPUT KDA_PYTHON KDA_TIMEOUT
#   KDA_KERNEL_NAME KDA_AIC_METRICS
#
# Requirements:
#   `msprof` must be on PATH.  Source the CANN environment first, e.g.
#       source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   If msprof is missing this script prints an error and exits non-zero; it
#   never falls back to host-side timing and never invents a number.

set -euo pipefail

# ======================== defaults ========================

WARMUP="${KDA_WARMUP:-5}"
LAUNCH_COUNT="${KDA_LAUNCH_COUNT:-20}"
OUTPUT_DIR="${KDA_OUTPUT:-./msprof_output}"
PYTHON_BIN="${KDA_PYTHON:-python3}"
TIMEOUT_S="${KDA_TIMEOUT:-1800}"
AIC_METRICS="${KDA_AIC_METRICS:-Default}"

# All six stage kernels declare their TileLang prim_func as `main`, and the
# Ascend codegen emits the device symbol as "<global_symbol>_kernel"
# (src/target/codegen_ascend.cc).  Hence one shared kernel name for every
# target.  Override with --kernel-name if the symbol ever changes.
KERNEL_NAME="${KDA_KERNEL_NAME:-main_kernel}"

ONLY=""
LIST_ONLY=false
SKIP_PIPELINE=false
KEEP_RAW=false

# ======================== argument parsing ========================

print_help() {
    sed -n '3,28p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --warmup)         WARMUP="$2"; shift 2 ;;
        --launch-count)   LAUNCH_COUNT="$2"; shift 2 ;;
        --output)         OUTPUT_DIR="$2"; shift 2 ;;
        --python)         PYTHON_BIN="$2"; shift 2 ;;
        --timeout)        TIMEOUT_S="$2"; shift 2 ;;
        --kernel-name)    KERNEL_NAME="$2"; shift 2 ;;
        --aic-metrics)    AIC_METRICS="$2"; shift 2 ;;
        --only)           ONLY="$2"; shift 2 ;;
        --list)           LIST_ONLY=true; shift ;;
        --skip-pipeline)  SKIP_PIPELINE=true; shift ;;
        --keep-raw)       KEEP_RAW=true; shift ;;
        -h|--help)        print_help; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; echo "Run 'bash bench.sh --help' for usage." >&2; exit 2 ;;
    esac
done

# ======================== target table ========================

# Format: "id|script|description"
# Order matches the stage order of the pipeline.
TARGETS=(
    "cumsum|kda_chunk_cumsum.py|stage 1/6  chunk-local cumulative log gate G"
    "kkt|kda_chunk_scaled_dot_kkt.py|stage 2/6  strict-lower decayed Gram matrix L"
    "solve_tril|kda_solve_tril.py|stage 3/6  unit lower triangular inverse A"
    "wy_fast|kda_wy_fast.py|stage 4/6  UT transform W and U"
    "chunk_h|kda_chunk_h.py|stage 5/6  per-chunk entry states and V'"
    "chunk_o|kda_chunk_o.py|stage 6/6  output O"
)
# kda_full.py sits one level up, next to gdn_full.py, mirroring the upstream
# layout: the stage kernels live in this directory, the driver above it.
PIPELINE_TARGET="pipeline|../kda_full.py|full six-stage forward pass"

# ======================== paths ========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# The stage modules import each other by bare module name, so the script
# directory has to be importable.  Also make the enclosing TileLang checkout
# importable when this example sits inside one (examples/<group>/kda/ -> root
# is three levels up); silently skip when it does not look like a checkout.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
CANDIDATE_ROOT="$(cd "$SCRIPT_DIR/../../.." 2>/dev/null && pwd || true)"
if [ -n "$CANDIDATE_ROOT" ] && [ -d "$CANDIDATE_ROOT/tilelang" ]; then
    export PYTHONPATH="$CANDIDATE_ROOT:$PYTHONPATH"
fi

# ======================== --list short circuit ========================

if $LIST_ONLY; then
    echo "KDA L1 benchmark targets (id | script | description):"
    for entry in "${TARGETS[@]}"; do
        id="${entry%%|*}"; rest="${entry#*|}"
        script="${rest%%|*}"; desc="${rest#*|}"
        printf "  %-12s %-32s %s\n" "$id" "$script" "$desc"
    done
    id="${PIPELINE_TARGET%%|*}"; rest="${PIPELINE_TARGET#*|}"
    script="${rest%%|*}"; desc="${rest#*|}"
    printf "  %-12s %-32s %s\n" "$id" "$script" "$desc"
    exit 0
fi

# ======================== environment checks ========================

if ! command -v msprof >/dev/null 2>&1; then
    cat >&2 <<'MSG'
Error: `msprof` was not found on PATH.

This benchmark reports on-board profiler measurements only.  Without msprof
there is nothing to measure, and this script will not substitute host-side
timing or any placeholder value.

Fix: source the CANN environment on the Ascend box, for example
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
then run this script again.
MSG
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: python interpreter '$PYTHON_BIN' not found on PATH." >&2
    echo "Pass a different one with --python, or set KDA_PYTHON." >&2
    exit 1
fi

HAVE_TIMEOUT=false
TIMEOUT_NOTE=" (ignored: 'timeout' is not on PATH)"
if command -v timeout >/dev/null 2>&1; then
    HAVE_TIMEOUT=true
    TIMEOUT_NOTE=""
fi

# ======================== target selection ========================

selected=()
if [ -n "$ONLY" ]; then
    # Comma separated ids; validate every one of them before running anything.
    IFS=',' read -r -a wanted <<< "$ONLY"
    for w in "${wanted[@]}"; do
        w="$(echo "$w" | tr -d '[:space:]')"
        [ -z "$w" ] && continue
        found=false
        for entry in "${TARGETS[@]}"; do
            if [ "${entry%%|*}" = "$w" ]; then
                selected+=("$entry"); found=true; break
            fi
        done
        if [ "${PIPELINE_TARGET%%|*}" = "$w" ]; then
            selected+=("$PIPELINE_TARGET"); found=true
        fi
        if ! $found; then
            echo "Error: unknown target id '$w'. Run 'bash bench.sh --list'." >&2
            exit 2
        fi
    done
else
    selected=("${TARGETS[@]}")
    if ! $SKIP_PIPELINE; then
        selected+=("$PIPELINE_TARGET")
    fi
fi

if [ "${#selected[@]}" -eq 0 ]; then
    echo "Error: no targets selected." >&2
    exit 2
fi

for entry in "${selected[@]}"; do
    rest="${entry#*|}"; script="${rest%%|*}"
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "Error: expected script '$script' relative to bench.sh, but it is missing." >&2
        exit 1
    fi
done

# ======================== msprof result parser ========================

# Reads whatever msprof left behind for one target and emits a single line:
#     mean|min|max|count|source
# with "NA" for the values and 0 for the count when nothing usable was found.
#
# Two sources are tried, in order:
#   1. any CSV under the msprof output directory that carries a per-task
#      duration column (OpBasicInfo / op_summary / op_statistic and friends);
#   2. the captured stdout of msprof, which prints a "Task Duration(us)" line.
# The CSV schema is not stable across CANN releases, so columns are matched by
# a case-insensitive substring of the header rather than by exact name.
parse_msprof_result() {
    local out_dir="$1" log_file="$2" pattern="$3"
    "$PYTHON_BIN" - "$out_dir" "$log_file" "$pattern" <<'PY_PARSER'
import csv
import os
import re
import sys

out_dir, log_file, pattern = sys.argv[1], sys.argv[2], sys.argv[3]

# Per-launch duration columns, most specific first.
DUR_KEYS = (
    "task duration(us)",
    "task duration",
    "aicore time(us)",
    "aicore_time(us)",
    "aiv time(us)",
    "aiv_time(us)",
    "task time(us)",
    "duration(us)",
)
# Aggregate column: usable only together with a launch count.
TOTAL_KEYS = ("total time(us)", "total_time(us)")
COUNT_KEYS = ("count", "total count", "task count")
NAME_KEYS = ("op name", "op type", "kernel name", "kernel_name", "name")

# Preference order for candidate CSV files.
FILE_RANK = ("opbasicinfo", "op_summary", "opsummary", "op_statistic", "opstatistic")


def norm(s):
    return (s or "").strip().lower()


def find_col(header, keys):
    lowered = [norm(h) for h in header]
    for key in keys:
        for i, h in enumerate(lowered):
            if h == key:
                return i
    for key in keys:
        for i, h in enumerate(lowered):
            if key in h:
                return i
    return -1


def to_float(raw):
    raw = (raw or "").strip().replace(",", "")
    if not raw or raw.upper() in ("N/A", "NA", "NAN", "-"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def name_matches(cell, pat):
    cell, pat = norm(cell), norm(pat)
    if not pat:
        return True
    for alt in pat.split("|"):
        alt = alt.strip().rstrip("*")
        if alt and alt in cell:
            return True
    return False


def csv_candidates(root):
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".csv"):
                hits.append(os.path.join(dirpath, fn))

    def rank(path):
        base = os.path.basename(path).lower()
        for i, tag in enumerate(FILE_RANK):
            if tag in base:
                return i
        return len(FILE_RANK)

    hits.sort(key=lambda p: (rank(p), p))
    return hits


def values_from_csv(path):
    """Return (values, filtered_by_name) or (None, False)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return None, False
    if len(rows) < 2:
        return None, False

    header = rows[0]
    body = [r for r in rows[1:] if any((c or "").strip() for c in r)]
    if not body:
        return None, False

    name_idx = find_col(header, NAME_KEYS)
    dur_idx = find_col(header, DUR_KEYS)

    def collect(require_name):
        out = []
        for row in body:
            if require_name and name_idx >= 0 and name_idx < len(row):
                if not name_matches(row[name_idx], pattern):
                    continue
            if dur_idx >= 0 and dur_idx < len(row):
                val = to_float(row[dur_idx])
                if val is not None:
                    out.append(val)
        return out

    if dur_idx >= 0:
        vals = collect(True)
        if vals:
            return vals, (name_idx >= 0)
        vals = collect(False)
        if vals:
            return vals, False
        return None, False

    # No per-launch column: derive an average from total time / launch count.
    tot_idx = find_col(header, TOTAL_KEYS)
    cnt_idx = find_col(header, COUNT_KEYS)
    if tot_idx < 0 or cnt_idx < 0:
        return None, False
    derived = []
    for row in body:
        if name_idx >= 0 and name_idx < len(row):
            if not name_matches(row[name_idx], pattern):
                continue
        if tot_idx >= len(row) or cnt_idx >= len(row):
            continue
        total, count = to_float(row[tot_idx]), to_float(row[cnt_idx])
        if total is not None and count:
            derived.append(total / count)
    if derived:
        return derived, (name_idx >= 0)
    return None, False


def values_from_log(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    hits = re.findall(r"[Tt]ask\s+[Dd]uration\s*\(us\)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    vals = [float(h) for h in hits]
    return vals or None


values, source, filtered = None, "none", False

if os.path.isdir(out_dir):
    for path in csv_candidates(out_dir):
        vals, was_filtered = values_from_csv(path)
        if vals:
            values = vals
            filtered = was_filtered
            source = os.path.relpath(path, out_dir)
            break

if values is None and os.path.isfile(log_file):
    vals = values_from_log(log_file)
    if vals:
        values, source, filtered = vals, "msprof stdout", True

if not values:
    print("NA|NA|NA|0|none")
    sys.exit(0)

mean = sum(values) / len(values)
tag = source if filtered else source + " (unfiltered)"
print("%.3f|%.3f|%.3f|%d|%s" % (mean, min(values), max(values), len(values), tag))
PY_PARSER
}

# ======================== msprof runner ========================

# run_msprof_op <script> <out_dir> ; returns non-zero when msprof itself failed.
run_msprof_op() {
    local script="$1" out_dir="$2"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    chmod 700 "$out_dir" 2>/dev/null || true

    local log_file="$out_dir/msprof.log"
    local -a cmd=(
        msprof op
        "--kernel-name=$KERNEL_NAME"
        "--warm-up=$WARMUP"
        "--launch-count=$LAUNCH_COUNT"
        "--aic-metrics=$AIC_METRICS"
        "--output=$out_dir"
        "$PYTHON_BIN" "$script"
    )
    if $HAVE_TIMEOUT; then
        cmd=(timeout "$TIMEOUT_S" "${cmd[@]}")
    fi

    echo "  > msprof op (warm-up=$WARMUP launch-count=$LAUNCH_COUNT kernel=$KERNEL_NAME)"
    local rc=0
    "${cmd[@]}" >"$log_file" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  [X] msprof op exited with status $rc; see $log_file"
        tail -n 8 "$log_file" 2>/dev/null | sed 's/^/      /' || true
        return 1
    fi
    # Give the profiler a moment to flush its CSV files to disk.
    sleep 1
    return 0
}

# ======================== main ========================

mkdir -p "$OUTPUT_DIR"
SUMMARY_CSV="$OUTPUT_DIR/kda_bench.csv"
echo "target,script,mean_us,min_us,max_us,samples,source,status" > "$SUMMARY_CSV"

echo "================================================================"
echo "KDA L1 benchmark -- msprof op, on-board collection"
echo "================================================================"
echo "Script dir   : $SCRIPT_DIR"
echo "Python       : $PYTHON_BIN"
echo "Kernel name  : $KERNEL_NAME"
echo "Warm-up      : $WARMUP"
echo "Launch count : $LAUNCH_COUNT"
echo "aic-metrics  : $AIC_METRICS"
echo "Output dir   : $OUTPUT_DIR"
echo "Timeout      : ${TIMEOUT_S}s${TIMEOUT_NOTE}"
echo "PYTHONPATH   : $PYTHONPATH"
echo "================================================================"
echo "Note: all six stage kernels expose the same device symbol"
echo "      ('$KERNEL_NAME'), so a profile of the full pipeline cannot"
echo "      attribute time per stage by name.  That is what the per-stage"
echo "      runs below are for; the pipeline row is an aggregate."
echo "================================================================"

declare -a ROWS=()
ok_count=0
fail_count=0

for entry in "${selected[@]}"; do
    id="${entry%%|*}"; rest="${entry#*|}"
    script="${rest%%|*}"; desc="${rest#*|}"

    echo ""
    echo "[$id] $script -- $desc"

    target_dir="$OUTPUT_DIR/$id"
    status="ok"
    mean="NA"; vmin="NA"; vmax="NA"; samples="0"; source="none"

    if run_msprof_op "$script" "$target_dir"; then
        parsed="$(parse_msprof_result "$target_dir" "$target_dir/msprof.log" "$KERNEL_NAME" || true)"
        IFS='|' read -r mean vmin vmax samples source <<< "${parsed:-NA|NA|NA|0|none}"
        if [ "$mean" = "NA" ] || [ "${samples:-0}" = "0" ]; then
            status="no-data"
            echo "  [!] msprof produced no parsable duration for this target."
            echo "      Inspect $target_dir and, if the CSV layout differs on this"
            echo "      CANN release, extend the parser column lists in bench.sh."
        else
            echo "  mean=${mean} us  min=${vmin} us  max=${vmax} us  over ${samples} sample(s)  [from ${source}]"
        fi
    else
        status="run-failed"
    fi

    if [ "$status" = "ok" ]; then
        ok_count=$((ok_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi

    ROWS+=("$id|$mean|$vmin|$vmax|$samples|$status")
    echo "$id,$script,$mean,$vmin,$vmax,$samples,\"$source\",$status" >> "$SUMMARY_CSV"

    if ! $KEEP_RAW && [ "$status" = "ok" ]; then
        # Keep the log and the summary CSV; drop the bulky raw dump.
        find "$target_dir" -maxdepth 1 -type d -name 'OPPROF*' -exec rm -rf {} + 2>/dev/null || true
    fi
done

# ======================== report ========================

echo ""
echo "================================================================"
echo "Per-kernel results (Task Duration, microseconds, from msprof op)"
echo "================================================================"
printf "%-12s %12s %12s %12s %9s  %s\n" "TARGET" "MEAN(us)" "MIN(us)" "MAX(us)" "SAMPLES" "STATUS"
printf "%-12s %12s %12s %12s %9s  %s\n" "------------" "------------" "------------" "------------" "---------" "----------"
for row in "${ROWS[@]}"; do
    r_id="${row%%|*}"; rest="${row#*|}"
    r_mean="${rest%%|*}"; rest="${rest#*|}"
    r_min="${rest%%|*}"; rest="${rest#*|}"
    r_max="${rest%%|*}"; rest="${rest#*|}"
    r_n="${rest%%|*}"; r_status="${rest#*|}"
    printf "%-12s %12s %12s %12s %9s  %s\n" "$r_id" "$r_mean" "$r_min" "$r_max" "$r_n" "$r_status"
done
echo "================================================================"
echo "Targets measured: $ok_count   without data: $fail_count"
echo "Summary CSV     : $SUMMARY_CSV"
echo "Raw profiles    : $OUTPUT_DIR/<target>/"
echo "================================================================"

if [ "$ok_count" -eq 0 ]; then
    echo "No target yielded profiler data; nothing was measured." >&2
    exit 1
fi
if [ "$fail_count" -ne 0 ]; then
    echo "Some targets yielded no profiler data (see the table above)." >&2
    exit 1
fi

echo "Test Passed!"
