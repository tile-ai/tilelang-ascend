#!/bin/bash

# ================= Usage =================
# bash run_examples.sh [--backend auto|pto] [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]]
#                           [--dirs <dir1 dir2 ...>] [--max-jobs N] [--project-root <path>]
#
# Options:
#   --backend <auto|ascendc|pto>   Set compilation backend (default: auto)
#                          When set to "pto" or "ascendc", TILELANG_JIT_TARGET env var is set accordingly
#                          to override @tilelang.jit target="auto" at runtime (no source modification)
#   --skip-aclgraph[=true|false]  Skip aclgraph examples (default: true, prevents env crash on compute platforms)
#   --skip-pytest[=true|false]    Skip pytest phase (default: true)
#   --dirs <dirs...>       Only run specified directories (incremental mode)
#   --max-jobs N           Max parallel jobs (default: 8)
#   --project-root <path>  Project root directory (auto-detected if omitted)
# ================= ========== =================

BACKEND="auto"
SKIP_ACLGRAPH=true
SKIP_PYTEST=true
TEST_DIRS=""
MAX_JOBS=8
PROJECT_ROOT_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            if [[ "$BACKEND" != "auto" && "$BACKEND" != "ascendc" && "$BACKEND" != "pto" ]]; then
                echo "Error: --backend must be 'auto', 'ascendc' or 'pto'"
                exit 1
            fi
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
        --dirs)
            TEST_DIRS="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "====================================="
echo "Run All Examples Configuration"
echo "  Backend: $BACKEND"
echo "  Skip aclgraph: $SKIP_ACLGRAPH"
echo "  Skip dispatch_combine/shmem: always (require shmem module)"
echo "  Pytest: $([ "$SKIP_PYTEST" = true ] && echo "skipped (--skip-pytest, default)" || echo "enabled (--skip-pytest=false)")"
echo "  Max jobs: $MAX_JOBS"
if [ -n "$TEST_DIRS" ]; then
    echo "  Directories: $TEST_DIRS"
else
    echo "  Directories: all"
fi
echo "====================================="

# ================= Environment Setup =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$PROJECT_ROOT_ARG" ]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT_ARG" && pwd)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi

if [ ! -f "$PROJECT_ROOT/set_env.sh" ]; then
    echo "Error: Cannot find set_env.sh in PROJECT_ROOT=$PROJECT_ROOT"
    echo "  Use --project-root <path> to specify the project root directory"
    exit 1
fi

EXAMPLES_DIR="$PROJECT_ROOT/examples"
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "Error: Cannot find examples/ directory at $EXAMPLES_DIR"
    exit 1
fi

source "$PROJECT_ROOT/set_env.sh"

export TILELANG_AUTO_TUNING_CPU_COUNTS=4
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT=4

# ================= Backend Override (env var) =================
if [[ "$BACKEND" == "pto" || "$BACKEND" == "ascendc" ]]; then
    export TILELANG_JIT_TARGET=$BACKEND
    echo ""
    echo "Set TILELANG_JIT_TARGET=$BACKEND"
    echo "  (Only overrides target='auto' in @tilelang.jit; explicit target='ascendc'/'pto' preserved)"
fi

# ================= Runtime Patch Hook (usercustomize.py) =================
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ================= Script Collection =================
EXTRA_TASKS=(
    "sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_baseline|[bench_sfa] sparse_flash_attn_pa_baseline"
    "sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_developer|[bench_sfa] sparse_flash_attn_pa_developer"
    "sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_no_cv_pipeline|[bench_sfa] sparse_flash_attn_pa_no_cv_pipeline"
    "sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa|[bench_sfa] sparse_flash_attn_pa"
)

collect_test_scripts() {
    local dir="$1"
    local scripts=()

    case "$dir" in
        "dispatch_combine"|"shmem")
            echo "${scripts[@]}"
            return
            ;;
        "gemm_aot"|"torch_tl_ascend")
            if [[ "$dir" == "gemm_aot" ]]; then
                scripts+=("gemm_aot/run_example_gemm_aot.sh")
            elif [[ "$dir" == "torch_tl_ascend" ]]; then
                scripts+=("torch_tl_ascend/test_example.sh")
            fi
            echo "${scripts[@]}"
            return
            ;;
        "flash_attention")
            local py_files=$(find "$EXAMPLES_DIR/$dir" -maxdepth 1 -name "*.py" \
                -not -name "__init__.py" \
                -not -name "*_golden.py" \
                | sort)
            for f in $py_files; do scripts+=("$dir/$(basename "$f")"); done
            echo "${scripts[@]}"
            return
            ;;
        "aclgraph")
            if [[ "$SKIP_ACLGRAPH" == true ]]; then
                echo "${scripts[@]}"
                return
            fi
            ;;
    esac

    local py_files=$(find "$EXAMPLES_DIR/$dir" -maxdepth 2 -name "*.py" \
        -not -name "__init__.py" \
        -not -name "*_golden.py" \
        -not -name "sfa_golden.py" \
        -not -path "*/bench_sfa/*" \
        | sort)
    for f in $py_files; do
        rel_path=$(realpath --relative-to="$EXAMPLES_DIR" "$f")
        scripts+=("$rel_path")
    done

    local sh_files=$(find "$EXAMPLES_DIR/$dir" -maxdepth 2 \( -name "run_*.sh" -o -name "test_*.sh" \) | sort)
    for f in $sh_files; do
        rel_path=$(realpath --relative-to="$EXAMPLES_DIR" "$f")
        scripts+=("$rel_path")
    done

    echo "${scripts[@]}"
}

all_scripts=()

if [ -n "$TEST_DIRS" ]; then
    IFS=' ' read -ra DIR_ARRAY <<< "$TEST_DIRS"
    echo "Incremental test mode - directories: ${DIR_ARRAY[*]}"

    for dir in "${DIR_ARRAY[@]}"; do
        if [ ! -d "$EXAMPLES_DIR/$dir" ]; then
            echo "Warning: directory $EXAMPLES_DIR/$dir not found, skipping"
            continue
        fi

        if [[ "$dir" == "aclgraph" && "$SKIP_ACLGRAPH" == true ]]; then
            echo "  Skipping aclgraph (--skip-aclgraph)"
            continue
        fi

        collected=$(collect_test_scripts "$dir")
        if [ -n "$collected" ]; then
            for script in $collected; do
                all_scripts+=("$script")
            done
            echo "Collected scripts from $dir: $(echo $collected | wc -w) files"
        fi
    done

    if [[ " ${DIR_ARRAY[*]} " =~ " sparse_flash_attention " ]]; then
        for extra_task in "${EXTRA_TASKS[@]}"; do
            all_scripts+=("CUSTOM_TASK::${extra_task}")
        done
    fi

    if [[ " ${DIR_ARRAY[*]} " =~ " flash_attention " ]]; then
        fa_dir="$EXAMPLES_DIR/flash_attention/fa_opt"
        if [ -d "$fa_dir" ]; then
            fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
            if [ -n "$fa_python_files" ]; then
                for file in $fa_python_files; do
                    rel_path=$(realpath --relative-to="$EXAMPLES_DIR" "$file")
                    all_scripts+=("$rel_path")
                done
            fi
        fi
    fi
else
    echo "Full test mode - scanning all directories"

    exclude_dirs_arr=("dispatch_combine" "shmem")
    echo "  Excluding dispatch_combine, shmem (require shmem module)"
    if [[ "$SKIP_ACLGRAPH" == true ]]; then
        exclude_dirs_arr+=("aclgraph")
        echo "  Excluding aclgraph (--skip-aclgraph, prevents env crash on compute platforms)"
    fi

    for dir in $(find "$EXAMPLES_DIR" -maxdepth 1 -type d -not -path "$EXAMPLES_DIR" | sort); do
        dir_name=$(basename "$dir")
        skip=false
        for excl in "${exclude_dirs_arr[@]}"; do
            if [[ "$dir_name" == "$excl" ]]; then
                skip=true
                break
            fi
        done
        if $skip; then continue; fi

        collected=$(collect_test_scripts "$dir_name")
        if [ -n "$collected" ]; then
            for script in $collected; do
                all_scripts+=("$script")
            done
        fi
    done

    for extra_task in "${EXTRA_TASKS[@]}"; do
        all_scripts+=("CUSTOM_TASK::${extra_task}")
    done

    fa_dir="$EXAMPLES_DIR/flash_attention/fa_opt"
    if [ -d "$fa_dir" ]; then
        fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
        if [ -n "$fa_python_files" ]; then
            for file in $fa_python_files; do
                rel_path=$(realpath --relative-to="$EXAMPLES_DIR" "$file")
                all_scripts+=("$rel_path")
            done
        fi
    fi
fi

echo ""
echo "Total scripts to run: ${#all_scripts[@]}"
echo "====================================="

if [ ${#all_scripts[@]} -eq 0 ]; then
    echo "No test scripts found."
    exit 0
fi

# ================= Parallel Execution =================
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
total_scripts=0

for script in "${all_scripts[@]}"; do
    total_scripts=$((total_scripts + 1))

    {
        if [[ "$script" == CUSTOM_TASK::* ]]; then
            task_info=${script#CUSTOM_TASK::}
            task_dir=$(echo "$task_info" | cut -d'|' -f1)
            task_cmd=$(echo "$task_info" | cut -d'|' -f2)
            display_name=$(echo "$task_info" | cut -d'|' -f3)

            output=$(cd "$EXAMPLES_DIR/$task_dir" && eval "$task_cmd" 2>&1)
            exit_code=$?
            current_script_ref="$display_name"
        else
            script_abs="$EXAMPLES_DIR/$script"
            script_dir=$(dirname "$script_abs")
            script_name=$(basename "$script_abs")
            current_script_ref="$script"

            if [[ "$script" == *.py ]]; then
                output=$(cd "$script_dir" && python "$script_name" 2>&1)
                exit_code=$?
            else
                output=$(cd "$script_dir" && bash "$script_name" 2>&1)
                exit_code=$?
            fi
        fi

        last_line=$(echo "$output" | tail -n 1)
        if [[ "$output" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || \
           [[ "$output" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]] || \
           [[ "$script" == CUSTOM_TASK::* && $exit_code -eq 0 ]]; then
            printf "[PASSED] %s\n" "$current_script_ref"
            touch "$temp_dir/pass_$total_scripts"
        else
            local details
            details=$(echo "$output" | tail -n 5 | sed 's/^/  /')
            printf "[FAILED] %s (Exit: %d)\n  Last line: %s\n%s\n" "$current_script_ref" "$exit_code" "$last_line" "$details"
        fi
    } &

    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi
done

wait

# ================= Results Summary =================
passed_scripts=$(ls "$temp_dir" | grep "pass_" | wc -l)
failed_scripts=$((total_scripts - passed_scripts))
rm -rf "$temp_dir"

echo -e "\n====================================="
echo "Execution Summary"
echo "Total: $total_scripts | Passed: $passed_scripts | Failed: $failed_scripts"
if [ $total_scripts -gt 0 ]; then
    echo "Pass rate: $((passed_scripts * 100 / total_scripts))%"
fi
echo "====================================="

# ================= Pytest Phase =================
if [ "$SKIP_PYTEST" = true ]; then
    echo -e "\n====================================="
    echo "Skipping pytest (--skip-pytest, default)"
    echo "====================================="
    if [ $failed_scripts -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
fi

echo -e "\n====================================="
echo "Running pytest tests"
echo "====================================="

pytest --forked "$PROJECT_ROOT/testing/python/" -v -n $MAX_JOBS 2>&1 | tee pytest_output.log
pytest_exit_code=${PIPESTATUS[0]}

pytest_summary=$(grep -E "[0-9]+ (passed|failed|xfailed)" pytest_output.log | tail -1)

pytest_passed=0
pytest_failed=0
pytest_xfailed=0

if [ -n "$pytest_summary" ]; then
    if echo "$pytest_summary" | grep -q "passed"; then
        pytest_passed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ passed" | grep -Eo "[0-9]+" || echo "0")
    fi
    if echo "$pytest_summary" | grep -q "failed"; then
        pytest_failed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ failed" | grep -Eo "[0-9]+" || echo "0")
    fi
    if echo "$pytest_summary" | grep -q "xfailed"; then
        pytest_xfailed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ xfailed" | grep -Eo "[0-9]+" || echo "0")
    fi
fi

if [ $pytest_exit_code -eq 0 ]; then
    echo -e "\n====================================="
    echo "All pytest tests PASSED!"
    echo "====================================="
else
    echo -e "\n====================================="
    echo "Some pytest tests FAILED!"
    echo "====================================="
fi

total_all=$((total_scripts + pytest_passed + pytest_failed + pytest_xfailed))
passed_all=$((passed_scripts + pytest_passed + pytest_xfailed))
failed_all=$((failed_scripts + pytest_failed))

echo -e "\n====================================="
echo "Final Execution Summary (Bench + Pytest)"
echo "  Backend: $BACKEND"
echo "  Aclgraph: $([ "$SKIP_ACLGRAPH" = true ] && echo "skipped (--skip-aclgraph, default)" || echo "included (--skip-aclgraph=false)")"
echo "Bench: Total: $total_scripts | Passed: $passed_scripts | Failed: $failed_scripts"
echo "Pytest: Passed: $pytest_passed | Failed: $pytest_failed | Xfailed: $pytest_xfailed (expected failures, counted as passed)"
echo "Total: $total_all | Passed: $passed_all | Failed: $failed_all"
if [ $total_all -gt 0 ]; then
    echo "Pass rate: $((passed_all * 100 / total_all))%"
fi
echo "====================================="

rm -f pytest_output.log

if [ $failed_scripts -gt 0 ] || [ $pytest_exit_code -ne 0 ]; then
    exit 1
else
    exit 0
fi