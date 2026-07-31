#!/bin/bash

# ================= 参数解析 =================
SKIP_PYTEST=false
ENABLE_COVERAGE=false
ENABLE_CPP_COVERAGE=false
TEST_DIRS=""
EXPERIMENT_DIRS=""
PYTEST_MARKERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pytest)
            SKIP_PYTEST=true
            shift
            ;;
        --coverage)
            ENABLE_COVERAGE=true
            shift
            ;;
        --enable-cpp-coverage)
            ENABLE_CPP_COVERAGE=true
            shift
            ;;
        --dirs)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "Error: --dirs requires an argument" >&2
                exit 1
            fi
            # 吃掉后续所有非 -- 开头的 token（支持 CI 的 unquoted 多目录传参）
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TEST_DIRS="${TEST_DIRS:+$TEST_DIRS }$1"
                shift
            done
            ;;
        --experiment-dirs)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "Error: --experiment-dirs requires an argument" >&2
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPERIMENT_DIRS="${EXPERIMENT_DIRS:+$EXPERIMENT_DIRS }$1"
                shift
            done
            ;;
        --pytest-markers)
            if [[ $# -lt 2 ]]; then
                echo "Error: --pytest-markers requires an argument" >&2
                exit 1
            fi
            PYTEST_MARKERS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ================= 全局环境变量设置 =================
PROJECT_ROOT="$(cd .. && pwd)"

# An operator listed here is covered by its Pytest test, which the workflow
# runs separately; executing it here as well would compile and run the same
# kernel twice.
declare -A MIGRATED_SOURCES=()
# Its test is skipped here for the same reason, and this is how: by the entry
# that claims it, not by its name. A test this repository never registered
# belongs to whoever wrote it and is collected like any other script.
declare -A MIGRATED_TESTS=()
OPERATOR_TEST_RESOLVER="${PROJECT_ROOT}/scripts/ci/resolve_operator_tests.py"
if [ -f "$OPERATOR_TEST_RESOLVER" ]; then
    # This script has no `set -e`, so a failure here would otherwise leave the
    # skip list empty and let every migrated operator run twice unnoticed.
    if ! python "$OPERATOR_TEST_RESOLVER" validate; then
        echo "Operator test manifest is invalid" >&2
        exit 1
    fi
    python "$OPERATOR_TEST_RESOLVER" check-orphans
    while IFS=$'\t' read -r source test; do
        [ -n "$source" ] || continue
        MIGRATED_SOURCES["$source"]=1
        MIGRATED_TESTS["$test"]=1
    done < <(python "$OPERATOR_TEST_RESOLVER" list --format tsv)
fi

# experiment 算子根目录（相对 examples 工作目录）
EXPERIMENT_ROOT="../examples_experiment"

# 解析 TEST_DIRS / EXPERIMENT_DIRS 为数组
DIR_ARRAY=()
EXP_DIR_ARRAY=()
if [ -n "$TEST_DIRS" ]; then
    IFS=' ' read -ra DIR_ARRAY <<< "$TEST_DIRS"
fi
if [ -n "$EXPERIMENT_DIRS" ]; then
    IFS=' ' read -ra EXP_DIR_ARRAY <<< "$EXPERIMENT_DIRS"
fi
if [ -n "$TEST_DIRS" ] || [ -n "$EXPERIMENT_DIRS" ]; then
    echo "Running incremental tests - examples: [${DIR_ARRAY[*]}] experiment: [${EXP_DIR_ARRAY[*]}]"
else
    echo "Running full tests (all directories, examples + experiment)"
fi
# ===========================================

# ================= 配置区 =================
MAX_JOBS=8  # 同时并行执行的任务数，建议根据 NPU 负载调整
export TILELANG_AUTO_TUNING_CPU_COUNTS=4 # for autotuner
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT=4 # for autotuner

# --- 新增：特定目录执行特定指令配置 ---
# 每个任务独立加入测试队列，分别执行、分别显示结果
# 格式: EXTRA_TASKS 数组，每项为 "目录|命令|显示名称"
EXTRA_TASKS=(
    "./sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_baseline|[bench_sfa] sparse_flash_attn_pa_baseline"
    "./sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_developer|[bench_sfa] sparse_flash_attn_pa_developer"
    "./sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa_no_cv_pipeline|[bench_sfa] sparse_flash_attn_pa_no_cv_pipeline"
    "./sparse_flash_attention/bench_sfa|python bench_sfa.py --file sparse_flash_attn_pa|[bench_sfa] sparse_flash_attn_pa"
)
# ==========================================


# ================= Coverage 清除逻辑（确保纯净结果）=================
# 在每次测试开始前清除旧的 coverage 数据，确保结果只包含本次执行的数据
if [ "$ENABLE_COVERAGE" = true ] || [ "$ENABLE_CPP_COVERAGE" = true ]; then
    echo ""
    echo "====================================="
    echo "Cleaning old coverage data..."
    echo "====================================="
      
    # 1. 清除 Python coverage 数据
    echo "Removing Python coverage files..."
    rm -rf "${PROJECT_ROOT}/coverage_data/.coverage*" 2>/dev/null || true
    rm -f "${PROJECT_ROOT}/coverage_data/*.json" 2>/dev/null || true
    
    echo "  ✓ Python coverage files cleaned"
    
    # 2. 清除 C++ coverage 数据（如果启用）
    if [ "$ENABLE_CPP_COVERAGE" = true ]; then
        echo "Removing C++ coverage files..."
        
        # 清除 build 目录的 .gcda 文件（运行时覆盖率数据）
        find "${PROJECT_ROOT}/build" -name "*.gcda" -type f -delete 2>/dev/null || true
        
        # 清除旧的 coverage.info
        rm -f "${PROJECT_ROOT}/coverage_data/coverage.info" 2>/dev/null || true
        
        echo "  ✓ C++ coverage files cleaned"
    fi
    
    # 3. 清除旧的报告
    echo "Removing old coverage reports..."
    rm -f "${PROJECT_ROOT}/core_files_coverage_report.md" 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/coverage_reports" 2>/dev/null || true
    
    echo "  ✓ Old reports cleaned"
    
    # 4. 创建干净的目录
    mkdir -p "${PROJECT_ROOT}/coverage_data"
    mkdir -p "${PROJECT_ROOT}/coverage_reports"
    
    echo ""
    echo "✓ Coverage cleanup completed. Ready for fresh test."
    echo "====================================="
    echo ""
fi
echo "Starting parallel unified test execution (Live Output)..."
echo "====================================="

total_scripts=0
passed_scripts=0
all_scripts=()

should_skip_python_script() {
    local candidate="$1"
    local repo_relative

    repo_relative=$(realpath --relative-to="$PROJECT_ROOT" "$candidate")
    if [[ -n "${MIGRATED_SOURCES[$repo_relative]:-}" ]]; then
        echo "Skip migrated operator in legacy runner: $candidate" >&2
        return 0
    fi
    if [[ -n "${MIGRATED_TESTS[$repo_relative]:-}" ]]; then
        echo "Skip migrated operator test in legacy runner: $candidate" >&2
        return 0
    fi

    return 1
}

# 函数：收集单个目录下的测试脚本
collect_test_scripts() {
    local dir="$1"
    local scripts=()
    
    # 特殊目录：排除整个目录，不收集任何 py 文件
    case "$dir" in
        "./gemm_aot"|"./torch_tl_ascend"|"./dispatch_combine"|"./shmem")
            # 只收集特定 bash 脚本
            if [[ "$dir" == "./gemm_aot" ]]; then
                scripts+=("./gemm_aot/run_example_gemm_aot.sh")
            elif [[ "$dir" == "./torch_tl_ascend" ]]; then
                scripts+=("./torch_tl_ascend/test_example.sh")
            fi
            echo "${scripts[@]}"
            return
            ;;
        "./flash_attention")
            # 收集主目录的 py 文件（排除 fa_opt）
            local py_files=$(find "$dir" -maxdepth 1 -name "*.py" \
                -not -name "__init__.py" \
                -not -name "*_golden.py" \
                | sort)
            for f in $py_files; do
                should_skip_python_script "$f" || scripts+=("$f")
            done
            
            echo "${scripts[@]}"
            return
            ;;
    esac
    
    # 搜索 maxdepth 2 的 py 文件（排除特殊文件和 bench_sfa 子目录）
    local py_files=$(find "$dir" -maxdepth 2 -name "*.py" \
        -not -name "__init__.py" \
        -not -name "*_golden.py" \
        -not -name "sfa_golden.py" \
        -not -name "utils.py" \
        -not -path "*/bench_sfa/*" \
        -not -path "*/generative_recommendation/golden.py" \
        -not -path "*/generative_recommendation/testcase.py" \
        | sort)
    for f in $py_files; do
        should_skip_python_script "$f" || scripts+=("$f")
    done
    
    # 搜索 bash 脚本（特定命名模式）
    local sh_files=$(find "$dir" -maxdepth 2 \( -name "run_*.sh" -o -name "test_*.sh" \) | sort)
    for f in $sh_files; do scripts+=("$f"); done
    
    echo "${scripts[@]}"
}

# 1. 收集脚本逻辑
if [ -n "$TEST_DIRS" ] || [ -n "$EXPERIMENT_DIRS" ]; then
    # 增量测试：只运行指定目录
    echo "Incremental test mode - examples: [${DIR_ARRAY[*]}] experiment: [${EXP_DIR_ARRAY[*]}]"

    for dir in "${DIR_ARRAY[@]}"; do
        test_dir="./$dir"
        if [ ! -d "$test_dir" ]; then
            echo "Warning: directory $test_dir not found, skipping"
            continue
        fi

        collected=$(collect_test_scripts "$test_dir")
        if [ -n "$collected" ]; then
            for script in $collected; do
                all_scripts+=("$script")
            done
            echo "Collected scripts from $dir: $(echo $collected | wc -w) files"
        fi
    done

    # experiment 增量目录（examples_experiment 根下）
    for dir in "${EXP_DIR_ARRAY[@]}"; do
        test_dir="$EXPERIMENT_ROOT/$dir"
        if [ ! -d "$test_dir" ]; then
            echo "Warning: experiment directory $test_dir not found, skipping"
            continue
        fi

        collected=$(collect_test_scripts "$test_dir")
        if [ -n "$collected" ]; then
            for script in $collected; do
                all_scripts+=("$script")
            done
            echo "Collected scripts from experiment/$dir: $(echo $collected | wc -w) files"
        fi
    done

    # sparse_flash_attention 的 EXTRA_TASKS
    if [[ " ${DIR_ARRAY[*]} " =~ " sparse_flash_attention " ]]; then
        for extra_task in "${EXTRA_TASKS[@]}"; do
            et_dir=$(echo "$extra_task" | cut -d'|' -f1)
            [ -d "$et_dir" ] || { echo "Skip EXTRA_TASK (dir missing): $et_dir"; continue; }
            all_scripts+=("CUSTOM_TASK::${extra_task}")
        done
    fi

    # flash_attention/fa_opt 单独处理
    if [[ " ${DIR_ARRAY[*]} " =~ " flash_attention " ]]; then
        fa_dir="./flash_attention/fa_opt"
        if [ -d "$fa_dir" ]; then
            fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
            if [ -n "$fa_python_files" ]; then
                for file in $fa_python_files; do
                    should_skip_python_script "$file" || all_scripts+=("$file")
                done
            fi
        fi
    fi
else
    # 全量测试：使用 collect_test_scripts 遍历所有目录
    echo "Full test mode - scanning all directories"
    
    # 遍历所有一级目录
    for dir in $(find . -maxdepth 1 -type d -not -name "." -not -name "dispatch_combine" -not -name "shmem" | sort); do
        collected=$(collect_test_scripts "$dir")
        if [ -n "$collected" ]; then
            for script in $collected; do
                all_scripts+=("$script")
            done
        fi
    done
    
    # EXTRA_TASKS
    for extra_task in "${EXTRA_TASKS[@]}"; do
        et_dir=$(echo "$extra_task" | cut -d'|' -f1)
        [ -d "$et_dir" ] || { echo "Skip EXTRA_TASK (dir missing): $et_dir"; continue; }
        all_scripts+=("CUSTOM_TASK::${extra_task}")
    done

    # flash_attention/fa_opt 单独处理
    fa_dir="./flash_attention/fa_opt"
    if [ -d "$fa_dir" ]; then
        fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
        if [ -n "$fa_python_files" ]; then
            for file in $fa_python_files; do
                should_skip_python_script "$file" || all_scripts+=("$file")
            done
        fi
    fi

    # experiment 全量：扫描 examples_experiment 所有一级目录
    if [ -d "$EXPERIMENT_ROOT" ]; then
        echo "Full test mode - scanning experiment directories under $EXPERIMENT_ROOT"
        for dir in $(find "$EXPERIMENT_ROOT" -maxdepth 1 -type d -not -path "$EXPERIMENT_ROOT" | sort); do
            collected=$(collect_test_scripts "$dir")
            if [ -n "$collected" ]; then
                for script in $collected; do
                    all_scripts+=("$script")
                done
            fi
        done
    fi
fi

echo "Total scripts to run: ${#all_scripts[@]}"
# =================================================

# Run the migrated operator tests when coverage is being collected.
# The workflow runs them after this script, so there is nothing to do on the
# normal path. Coverage is collected by invoking this script by hand, and that
# path has no workflow behind it: the collection above already skipped these
# operators, their tests are run by nobody, and what they cover of tilelang
# drops out of the report. They run before the examples so a scoped invocation
# that collects no script still reaches them, and so they compile against a
# cold cache.
#
# Narrowed to the directories this invocation was given, as everywhere else
# those two options appear. A report that mixed the examples of one directory
# with the operators of all of them would not say what it was measuring.
operator_exit_code=0
operator_passed=0
operator_failed=0
operator_xfailed=0
if [ "$ENABLE_COVERAGE" = true ] || [ "$ENABLE_CPP_COVERAGE" = true ]; then
    OPERATOR_TESTS=()
    OPERATOR_SCOPE_ARGS=()
    if [ -n "$TEST_DIRS" ]; then
        OPERATOR_SCOPE_ARGS+=(--dirs $TEST_DIRS)
    fi
    if [ -n "$EXPERIMENT_DIRS" ]; then
        OPERATOR_SCOPE_ARGS+=(--experiment-dirs $EXPERIMENT_DIRS)
    fi
    if [ -f "$OPERATOR_TEST_RESOLVER" ]; then
        while IFS= read -r operator_test; do
            [ -n "$operator_test" ] && OPERATOR_TESTS+=("${PROJECT_ROOT}/$operator_test")
        done < <(python "$OPERATOR_TEST_RESOLVER" list-tests "${OPERATOR_SCOPE_ARGS[@]}")
    fi
    if [ ${#OPERATOR_TESTS[@]} -gt 0 ]; then
        echo -e "\n====================================="
        echo "Running migrated operator tests for coverage (${#OPERATOR_TESTS[@]} file(s))"
        echo "====================================="
        OPERATOR_MARKER_ARGS=()
        if [ -n "$PYTEST_MARKERS" ]; then
            OPERATOR_MARKER_ARGS+=(-m "$PYTEST_MARKERS")
        fi
        mkdir -p "${PROJECT_ROOT}/coverage_data"
        # No --forked here, unlike the runs elsewhere in this script: coverage
        # does not follow the fork, so the lines executed inside it are lost.
        # Measured over these tests, 2237 lines of tilelang are recorded with it
        # against 3285 without, and the difference is concentrated in the path
        # that compiles a kernel: jit/kernel.py 45 against 110,
        # cache/kernel_cache.py 48 against 125, engine/phase.py 12 against 58.
        #
        # One file per call, and no xdist. Without the fork the memory a test
        # reserves is only returned when the process holding it ends, and these
        # reserve a lot: of the sparse attention kernels one takes 31 GiB of a
        # 61 GiB device on its own. Anything that puts two of those in flight,
        # whether in sequence within a worker or across workers at once, runs
        # the device out. One at a time is the only arrangement that cannot.
        #
        # Each call writes its own coverage file, since pytest-cov merges into
        # COVERAGE_FILE as it exits and a shared name would leave only the last.
        # The combine step downstream already globs for them.
        : > operator_output.log
        for operator_test_path in "${OPERATOR_TESTS[@]}"; do
            export COVERAGE_FILE="${PROJECT_ROOT}/coverage_data/.coverage_operator_$(echo "$operator_test_path" | sed 's/[\/\.]/_/g')"
            pytest "$operator_test_path" -v \
                --cov=tilelang --cov-config="${PROJECT_ROOT}/.coveragerc" --cov-report= \
                "${OPERATOR_MARKER_ARGS[@]}" 2>&1 | tee -a operator_output.log
            operator_batch_code=${PIPESTATUS[0]}
            if [ "$operator_exit_code" -eq 0 ]; then
                operator_exit_code=$operator_batch_code
            fi
            unset COVERAGE_FILE
        done
        # One summary per batch, so these accumulate rather than reading the last.
        while IFS= read -r operator_summary; do
            [ -n "$operator_summary" ] || continue
            batch_passed=$(echo "$operator_summary" | grep -Eo "[0-9]+ passed" | grep -Eo "[0-9]+")
            batch_failed=$(echo "$operator_summary" | grep -Eo "[0-9]+ failed" | grep -Eo "[0-9]+")
            batch_xfailed=$(echo "$operator_summary" | grep -Eo "[0-9]+ xfailed" | grep -Eo "[0-9]+")
            operator_passed=$((operator_passed + ${batch_passed:-0}))
            operator_failed=$((operator_failed + ${batch_failed:-0}))
            operator_xfailed=$((operator_xfailed + ${batch_xfailed:-0}))
        done < <(grep -E "^=+ .*[0-9]+ (passed|failed|error).* =+$" operator_output.log)
        echo "Operator tests: ${operator_passed} passed, ${operator_failed} failed over ${#OPERATOR_TESTS[@]} file(s)"
        rm -f operator_output.log
    fi
fi

if [ ${#all_scripts[@]} -eq 0 ]; then
    echo "No test scripts found."
    exit $operator_exit_code
fi

# 2. 并行执行逻辑
# 注意：我们通过文件描述符或子进程退出码来统计结果
temp_dir=$(mktemp -d) # 创建临时目录仅用于存放结果标记文件，不存日志

for script in "${all_scripts[@]}"; do
    total_scripts=$((total_scripts + 1))

    # 启动后台子进程
    {
        # 判断是否为自定义任务
        if [[ "$script" == CUSTOM_TASK::* ]]; then
            # 提取任务信息（去掉前缀后按 | 分割: 目录|命令|显示名称）
            task_info=${script#CUSTOM_TASK::}
            task_dir=$(echo "$task_info" | cut -d'|' -f1)
            task_cmd=$(echo "$task_info" | cut -d'|' -f2)
            display_name=$(echo "$task_info" | cut -d'|' -f3)

            # 在指定目录下执行指定命令
            output=$(cd "$task_dir" && eval "$task_cmd" 2>&1)
            exit_code=$?
            current_script_ref="$display_name"
        else
            # 原有普通脚本执行逻辑
            script_dir=$(dirname "$script")
            script_name=$(basename "$script")
            current_script_ref="$script"

            # 执行脚本并捕获输出到变量，不在磁盘生成日志文件
            if [[ "$script" == *.py ]]; then
                if [ "$ENABLE_COVERAGE" = true ]; then
                    # 使用 coverage run 统计 examples 执行的 Python 代码覆盖率
                    # 每个脚本使用独立的 coverage 文件名，避免并行冲突
                    safe_name=$(echo "$script" | sed 's/[\/\.]/_/g')
                    output=$(cd "$script_dir" && COVERAGE_FILE="${PROJECT_ROOT}/coverage_data/.coverage_${safe_name}" coverage run --rcfile="${PROJECT_ROOT}/.coveragerc" "$script_name" 2>&1)
                    exit_code=$?
                else
                    output=$(cd "$script_dir" && python "$script_name" 2>&1)
                    exit_code=$?
                fi
            else
                output=$(cd "$script_dir" && bash "$script_name" 2>&1)
                exit_code=$?
            fi
        fi

        # 结果判定逻辑
        # 判定条件：
        # 1. 原有正则匹配 (KERNEL OUTPUT MATCH 或 TEST PASSED!)
        # 2. OR (是自定义任务 且 退出码为 0)
        last_line=$(echo "$output" | tail -n 1)
        if [[ "$output" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || \
           [[ "$output" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]] || \
           [[ "$script" == CUSTOM_TASK::* && $exit_code -eq 0 ]]; then
            echo "[PASSED] $current_script_ref"
            touch "$temp_dir/pass_$total_scripts"
        else
            echo "[FAILED] $current_script_ref (Exit: $exit_code)"
            echo "  Last line: $last_line"
            # 失败时打印最后5行方便调试
            echo "$output" | tail -n 5 | sed 's/^/  /'
        fi
    } &

    # 并发控制
    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi
done

wait # 等待所有任务完成

# 3. 统计结果
passed_scripts=$(ls "$temp_dir" | grep "pass_" | wc -l)
failed_scripts=$((total_scripts - passed_scripts))
rm -rf "$temp_dir" # 清理计数文件

echo -e "\n====================================="
echo "Execution Summary"
echo "Total: $total_scripts | Passed: $passed_scripts | Failed: $failed_scripts"
if [ $total_scripts -gt 0 ]; then
    echo "Pass rate: $((passed_scripts * 100 / total_scripts))%"
fi
echo "====================================="

# 4. 最后执行 pytest 自动发现并运行所有测试
if [ "$SKIP_PYTEST" = true ]; then
    echo -e "\n====================================="
    echo "Skipping pytest (only examples/ .py/.md/.png files modified)"
    echo "====================================="
    exit $operator_exit_code
fi

echo -e "\n====================================="
echo "Running pytest tests"
echo "====================================="

# 构建 pytest marker 过滤参数
PYTEST_MARKER_ARGS=()
if [ -n "$PYTEST_MARKERS" ]; then
    PYTEST_MARKER_ARGS+=(-m "$PYTEST_MARKERS")
    echo "Applying pytest marker filter: -m \"$PYTEST_MARKERS\""
fi

# 自动发现并运行 testing/python/ 目录下的所有测试文件（包括所有子目录）
# 运行 pytest 并捕获输出（使用 tee 同时显示和保存）
if [ "$ENABLE_COVERAGE" = true ]; then
    export COVERAGE_FILE="${PROJECT_ROOT}/coverage_data/.coverage_pytest"
    # C++ coverage 时不使用 --forked，避免多进程并发写入 .gcda 文件冲突
    COV_ARGS="--cov=tilelang --cov-report=term --cov-report=json:${PROJECT_ROOT}/coverage_data/pytest_coverage.json --cov-config=${PROJECT_ROOT}/.coveragerc"
    if [ "$ENABLE_CPP_COVERAGE" = true ]; then
        pytest "${PROJECT_ROOT}/testing/python/" -v $COV_ARGS "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee pytest_output.log
    else
        pytest --forked "${PROJECT_ROOT}/testing/python/" -v -n $MAX_JOBS $COV_ARGS "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee pytest_output.log
    fi
    unset COVERAGE_FILE
else
    pytest --forked "${PROJECT_ROOT}/testing/python/" -v -n $MAX_JOBS "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee pytest_output.log
fi
pytest_exit_code=${PIPESTATUS[0]}

# 提取 pytest 统计（最后一行包含 passed/failed/xfailed）
pytest_summary=$(grep -E "[0-9]+ (passed|failed|xfailed)" pytest_output.log | tail -1)

# 解析 pytest 结果
pytest_passed=0
pytest_failed=0
pytest_xfailed=0

if [ -n "$pytest_summary" ]; then
    # 提取 passed 数量
    if echo "$pytest_summary" | grep -q "passed"; then
        pytest_passed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ passed" | grep -Eo "[0-9]+" || echo "0")
    fi
    
    # 提取 failed 数量（不含 xfailed）
    if echo "$pytest_summary" | grep -q "failed"; then
        pytest_failed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ failed" | grep -Eo "[0-9]+" || echo "0")
    fi
    
    # 提取 xfailed 数量（预期失败，不计入失败）
    if echo "$pytest_summary" | grep -q "xfailed"; then
        pytest_xfailed=$(echo "$pytest_summary" | grep -Eo "[0-9]+ xfailed" | grep -Eo "[0-9]+" || echo "0")
    fi
fi

# 统计 pytest 结果
if [ $pytest_exit_code -eq 0 ]; then
    echo -e "\n====================================="
    echo "All pytest tests PASSED!"
    echo "====================================="
else
    echo -e "\n====================================="
    echo "Some pytest tests FAILED!"
    echo "====================================="
fi

# ================= Coverage Collection and Report =================
if [ "$ENABLE_COVERAGE" = true ] || [ "$ENABLE_CPP_COVERAGE" = true ]; then
    echo -e "\n====================================="
    echo "Collecting Coverage Data"
    echo "====================================="
    
    mkdir -p "${PROJECT_ROOT}/coverage_data" "${PROJECT_ROOT}/coverage_reports"
    
    # Python coverage
    if [ "$ENABLE_COVERAGE" = true ]; then
        echo "Collecting Python coverage..."
        
        # Combine all coverage files in coverage_data (examples + pytest already write here)
        coverage_files=$(find "${PROJECT_ROOT}/coverage_data" -name ".coverage*" -type f)
        if [ -n "$coverage_files" ]; then
            cd "${PROJECT_ROOT}"
            export COVERAGE_FILE="${PROJECT_ROOT}/coverage_data/.coverage"
            coverage combine --keep $coverage_files 2>&1 || true
            coverage json -o "${PROJECT_ROOT}/coverage_data/coverage.json" --include=tilelang/* 2>&1 || true
            unset COVERAGE_FILE
            cd "${PROJECT_ROOT}/examples"
            echo "✓ Python coverage collected"
        fi
    fi
    
    # C++ coverage
    if [ "$ENABLE_CPP_COVERAGE" = true ]; then
        echo "Collecting C++ coverage..."
        # Only collect from tilelang_objs.dir (contains only tilelang-ascend/src)
        # This is faster and avoids timeout issues with collecting entire build directory
        lcov --capture --directory "${PROJECT_ROOT}/build/CMakeFiles/tilelang_objs.dir" --output-file "${PROJECT_ROOT}/coverage_data/coverage.info" --no-checksum --ignore-errors source,graph 2>&1 || true
        
        if [ -f "${PROJECT_ROOT}/coverage_data/coverage.info" ]; then
            echo "✓ C++ coverage collected (tilelang-ascend/src only)"
        fi
    fi
    
    # Generate report
    if [ -f "${PROJECT_ROOT}/scripts/generate_coverage_stats_report.py" ]; then
        python "${PROJECT_ROOT}/scripts/generate_coverage_stats_report.py"
        echo "✓ Coverage report generated"
    fi
fi
# 输出合并后的结果（用于 CI workflow 解析）
# xfailed 是预期失败的测试，在 pytest 视角下属于"成功"状态（符合预期）
# 应计入 passed_all，而不应计入 failed_all
pytest_passed=$((pytest_passed + operator_passed))
pytest_failed=$((pytest_failed + operator_failed))
pytest_xfailed=$((pytest_xfailed + operator_xfailed))

total_all=$((total_scripts + pytest_passed + pytest_failed + pytest_xfailed))
passed_all=$((passed_scripts + pytest_passed + pytest_xfailed))
failed_all=$((failed_scripts + pytest_failed))

echo -e "\n====================================="
echo "Final Execution Summary (Bench + Pytest)"
echo "Bench: Total: $total_scripts | Passed: $passed_scripts | Failed: $failed_scripts"
echo "Pytest: Passed: $pytest_passed | Failed: $pytest_failed | Xfailed: $pytest_xfailed (expected failures, counted as passed)"
echo "Total: $total_all | Passed: $passed_all | Failed: $failed_all"
if [ $total_all -gt 0 ]; then
    echo "Pass rate: $((passed_all * 100 / total_all))%"
fi
echo "====================================="

# 清理临时文件
rm -f pytest_output.log

if [ "$pytest_exit_code" -eq 0 ]; then
    exit $operator_exit_code
fi
exit $pytest_exit_code
