#!/bin/bash

# ================= 参数解析 =================
SKIP_PYTEST=false
ENABLE_COVERAGE=false
ENABLE_CPP_COVERAGE=false
TEST_DIRS=""
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
            TEST_DIRS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ================= 全局环境变量设置 =================
# TILELANG_LIBRARY_PATH 需要在所有操作前设置，确保子进程继承
if [ "$ENABLE_CPP_COVERAGE" = true ]; then
    PROJECT_ROOT="$(cd .. && pwd)"
    export TILELANG_LIBRARY_PATH="${PROJECT_ROOT}/build"
    echo "TILELANG_LIBRARY_PATH set to: ${TILELANG_LIBRARY_PATH}"
fi

# 解析 TEST_DIRS 为数组
if [ -n "$TEST_DIRS" ]; then
    IFS=' ' read -ra DIR_ARRAY <<< "$TEST_DIRS"
    echo "Running incremental tests for directories: ${DIR_ARRAY[*]}"
else
    echo "Running full tests (all directories)"
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
    
    PROJECT_ROOT="$(cd .. && pwd)"
    
    # 1. 清除 Python coverage 数据
    echo "Removing Python coverage files..."
    rm -rf "${PROJECT_ROOT}/coverage_data/.coverage*" 2>/dev/null || true
    rm -f "${PROJECT_ROOT}/coverage_data/*.json" 2>/dev/null || true
    
    # 清除 examples 子目录的 coverage 文件（可能遗漏）
    find "${PROJECT_ROOT}/examples" -name ".coverage*" -type f -delete 2>/dev/null || true
    
    # 清除 testing/python 目录的 coverage 文件
    find "${PROJECT_ROOT}/testing/python" -name ".coverage*" -type f -delete 2>/dev/null || true
    
    # 清除项目根目录的 coverage 文件
    rm -f "${PROJECT_ROOT}/.coverage" "${PROJECT_ROOT}/.coverage.*" 2>/dev/null || true
    
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
    rm -f "${PROJECT_ROOT}/coverage_report.md" 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/coverage_reports/htmlcov" 2>/dev/null || true
    rm -rf "${PROJECT_ROOT}/coverage_reports/cpp_html" 2>/dev/null || true
    
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

# 定义 PROJECT_ROOT 用于 coverage 运行
PROJECT_ROOT="$(cd .. && pwd)"

total_scripts=0
passed_scripts=0
all_scripts=()

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
            for f in $py_files; do scripts+=("$f"); done
            
            echo "${scripts[@]}"
            return
            ;;
    esac
    
    # 搜索 maxdepth 2 的 py 文件（排除特殊文件和 bench_sfa 子目录）
    local py_files=$(find "$dir" -maxdepth 2 -name "*.py" \
        -not -name "__init__.py" \
        -not -name "*_golden.py" \
        -not -name "sfa_golden.py" \
        -not -path "*/bench_sfa/*" \
        | sort)
    for f in $py_files; do scripts+=("$f"); done
    
    # 搜索 bash 脚本（特定命名模式）
    local sh_files=$(find "$dir" -maxdepth 2 \( -name "run_*.sh" -o -name "test_*.sh" \) | sort)
    for f in $sh_files; do scripts+=("$f"); done
    
    echo "${scripts[@]}"
}

# 1. 收集脚本逻辑
if [ -n "$TEST_DIRS" ]; then
    # 增量测试：只运行指定目录
    echo "Incremental test mode - directories: ${DIR_ARRAY[*]}"
    
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
    
    # sparse_flash_attention 的 EXTRA_TASKS
    if [[ " ${DIR_ARRAY[*]} " =~ " sparse_flash_attention " ]]; then
        for extra_task in "${EXTRA_TASKS[@]}"; do
            all_scripts+=("CUSTOM_TASK::${extra_task}")
        done
    fi
    
    # flash_attention/fa_opt 单独处理
    if [[ " ${DIR_ARRAY[*]} " =~ " flash_attention " ]]; then
        fa_dir="./flash_attention/fa_opt"
        if [ -d "$fa_dir" ]; then
            fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
            if [ -n "$fa_python_files" ]; then
                for file in $fa_python_files; do all_scripts+=("$file"); done
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
        all_scripts+=("CUSTOM_TASK::${extra_task}")
    done
    
    # flash_attention/fa_opt 单独处理
    fa_dir="./flash_attention/fa_opt"
    if [ -d "$fa_dir" ]; then
        fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
        if [ -n "$fa_python_files" ]; then
            for file in $fa_python_files; do all_scripts+=("$file"); done
        fi
    fi
fi

echo "Total scripts to run: ${#all_scripts[@]}"
# =================================================

if [ ${#all_scripts[@]} -eq 0 ]; then
    echo "No test scripts found."
    exit 0
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
    exit 0
fi

echo -e "\n====================================="
echo "Running pytest tests"
echo "====================================="

# 自动发现并运行 testing/python/ 目录下的所有测试文件（包括所有子目录）
PROJECT_ROOT="$(cd .. && pwd)"

if [ "$ENABLE_COVERAGE" = true ]; then
    export COVERAGE_FILE="${PROJECT_ROOT}/coverage_data/.coverage_pytest"
    # C++ coverage 时不使用 --forked，避免环境变量继承问题
    if [ "$ENABLE_CPP_COVERAGE" = true ]; then
        pytest "${PROJECT_ROOT}/testing/python/" -v         --cov=tilelang         --cov=examples         --cov-report=term         --cov-report=json:${PROJECT_ROOT}/coverage_data/pytest_coverage.json         --cov-config=${PROJECT_ROOT}/.coveragerc
    else
        pytest --forked "${PROJECT_ROOT}/testing/python/" -v -n $MAX_JOBS         --cov=tilelang         --cov=examples         --cov-report=term         --cov-report=json:${PROJECT_ROOT}/coverage_data/pytest_coverage.json         --cov-config=${PROJECT_ROOT}/.coveragerc
    fi
    unset COVERAGE_FILE
else
    pytest --forked "${PROJECT_ROOT}/testing/python/" -v -n $MAX_JOBS
fi
pytest_exit_code=$?

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
    
    PROJECT_ROOT="$(cd .. && pwd)"
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
            coverage json -o "${PROJECT_ROOT}/coverage_data/coverage.json" --include=tilelang/*,examples/* 2>&1 || true
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
    if [ -f "${PROJECT_ROOT}/scripts/generate_coverage_report.py" ]; then
        python "${PROJECT_ROOT}/scripts/generate_coverage_report.py"
        echo "✓ Coverage report generated: coverage_report.md"
    fi
fi

exit $pytest_exit_code
