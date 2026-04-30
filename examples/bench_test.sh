#!/bin/bash

# ================= 配置区 =================
MAX_JOBS=8  # 同时并行执行的任务数，建议根据 NPU 负载调整
export TILELANG_AUTO_TUNING_CPU_COUNTS=4 # for autotuner
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT=4 # for autotuner

# 覆盖率配置：设置为 true 启用覆盖率统计（可通过环境变量 ENABLE_COVERAGE=true 覆盖）
ENABLE_COVERAGE=${ENABLE_COVERAGE:-false}
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
export PROJECT_ROOT  # 导出 PROJECT_ROOT 使子进程可访问

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

echo "Starting parallel unified test execution (Live Output)..."
echo "====================================="

total_scripts=0
passed_scripts=0
all_scripts=()

# 1. 收集脚本逻辑 (保持原样)
python_files=$(find . -maxdepth 2 -name "*.py" -not -path "./gemm_aot/*" -not -path "./dispatch_combine/*" -not -path "./shmem/*" -not -path "./torch_tl_ascend/*" -not -name "sfa_golden.py" -not -name "__init__.py" | sort)
if [ -n "$python_files" ]; then
    for file in $python_files; do all_scripts+=("$file"); done
fi

# ===== add examples/flash_attention/fa_opt/flash_*.py =====
fa_dir="./flash_attention/fa_opt"
if [ -d "$fa_dir" ]; then
    fa_python_files=$(find "$fa_dir" -maxdepth 1 -name "flash_*.py" | sort)
    if [ -n "$fa_python_files" ]; then
        for file in $fa_python_files; do all_scripts+=("$file"); done
    fi
fi


if [ -d "./gemm_aot" ]; then
    bash_scripts=$(find ./gemm_aot -maxdepth 1 -name "run_example_gemm_aot.sh" | sort)
    if [ -n "$bash_scripts" ]; then
        for script in $bash_scripts; do all_scripts+=("$script"); done
    fi
fi

# ===== add examples/torch_tl_ascend/test_example.sh =====
if [ -d "./torch_tl_ascend" ]; then
    bash_scripts=$(find ./torch_tl_ascend -maxdepth 1 -name "test_example.sh" | sort)
    if [ -n "$bash_scripts" ]; then
        for script in $bash_scripts; do all_scripts+=("$script"); done
    fi
fi

# ====== 新增：将特定目录任务逐个加入测试队列 ======
for extra_task in "${EXTRA_TASKS[@]}"; do
    # 使用特殊前缀标记，格式: CUSTOM_TASK::目录|命令|显示名称
    all_scripts+=("CUSTOM_TASK::${extra_task}")
done
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
                if [[ "$ENABLE_COVERAGE" == true ]]; then
                    output=$(cd "$script_dir" && python -m coverage run --source=$PROJECT_ROOT/tilelang,$PROJECT_ROOT/examples --rcfile=$PROJECT_ROOT/.coveragerc --data-file=$temp_dir/.coverage.$total_scripts "$script_name" 2>&1)
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

# 覆盖率数据合并（在清理前）
if [[ "$ENABLE_COVERAGE" == true ]]; then
    echo -e "\n====================================="
    echo "Combining coverage data..."
    echo "====================================="
    # 合并所有 coverage 数据文件
    if ls "$temp_dir"/.coverage.* 1> /dev/null 2>&1; then
        python -m coverage combine --rcfile=$PROJECT_ROOT/.coveragerc --data-file=.coverage_pytest_combined "$temp_dir"/.coverage.* .coverage_pytest 2>/dev/null || \
        python -m coverage combine --rcfile=$PROJECT_ROOT/.coveragerc --data-file=.coverage_pytest_combined "$temp_dir"/.coverage.*
        mv .coverage_pytest_combined .coverage_combined
        python -m coverage report --rcfile=$PROJECT_ROOT/.coveragerc --data-file=.coverage_combined --sort=Cover
        echo ""
        echo "Python coverage data combined successfully"
    fi
fi

rm -rf "$temp_dir" # 清理计数文件

echo -e "\n====================================="
echo "Execution Summary"
echo "Total: $total_scripts | Passed: $passed_scripts | Failed: $failed_scripts"
if [ $total_scripts -gt 0 ]; then
    echo "Pass rate: $((passed_scripts * 100 / total_scripts))%"
fi
echo "====================================="

# 4. 最后执行 pytest 自动发现并运行所有测试
echo -e "\n====================================="
echo "Running pytest tests"
echo "====================================="

# 自动发现并运行 testing/python/ 目录下的所有测试文件（包括所有子目录）
if [[ "$ENABLE_COVERAGE" == true ]]; then
    pytest --forked ../testing/python/ -v -n $MAX_JOBS --cov-config=$PROJECT_ROOT/.coveragerc --cov=tilelang --cov=examples --cov-report=term-missing --cov-append
    pytest_exit_code=$?
else
    pytest --forked ../testing/python/ -v -n $MAX_JOBS
    pytest_exit_code=$?
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

# 5. 覆盖率收集（测试完成后自动生成 Markdown 报告）
if [[ "$ENABLE_COVERAGE" == true ]]; then
    echo -e "\n====================================="
    echo "[Coverage] Generating coverage report..."
    echo "====================================="
    
    cd "$PROJECT_ROOT"
    
    REPORT_FILE="coverage_report.md"
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # 创建 Markdown 报告头部
    cat << EOF > $REPORT_FILE
# TileLang Coverage Report

**Generated at**: $TIMESTAMP

---

## Python Coverage (tilelang/ + examples/)

EOF
    
    # Python 覆盖率
    if [ -f "examples/.coverage_combined" ]; then
        cp examples/.coverage_combined . 2>/dev/null || true
    fi
    
    if [ -f ".coverage_combined" ]; then
        echo "### Summary" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        python -m coverage report --rcfile=.coveragerc --data-file=.coverage_combined | tail -1 >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        
        echo "### Module Details" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        echo "| Name | Stmts | Miss | Cover |" >> $REPORT_FILE
        echo "|------|-------|------|-------|" >> $REPORT_FILE
        python -m coverage report --rcfile=.coveragerc --data-file=.coverage_combined --sort=Cover --omit="tilelang/3rdparty/*" | \
            grep -v "^Name" | grep -v "^TOTAL" | grep -v "^---" | \
            awk '{printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4}' >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    else
        echo "No Python coverage data found" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    fi
    
    echo "---" >> $REPORT_FILE
    echo "" >> $REPORT_FILE
    
    # C++ 覆盖率
    cat << EOF >> $REPORT_FILE
## C++ Coverage (src/)

EOF
    
    GCNO_COUNT=$(find build -name "*.gcno" 2>/dev/null | wc -l)
    GCDA_COUNT=$(find build -name "*.gcda" 2>/dev/null | wc -l)
    
    if [ "$GCNO_COUNT" -gt 0 ] && [ "$GCDA_COUNT" -gt 0 ]; then
        geninfo build/CMakeFiles/tilelang_objs.dir -o coverage_cpp.info --quiet 2>/dev/null
        lcov --extract coverage_cpp.info "*/tilelang-ascend/src/*" --output-file coverage_cpp_filtered.info --quiet 2>/dev/null
        
        echo "### Summary" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        lcov --summary coverage_cpp_filtered.info 2>&1 | grep -E "(lines|functions)" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        
        echo "### File Details" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        python3 "$PROJECT_ROOT/scripts/generate_cpp_coverage_md.py" coverage_cpp_filtered.info >> $REPORT_FILE
        
        rm -f coverage_cpp.info coverage_cpp_filtered.info
        echo "" >> $REPORT_FILE
    else
        echo "C++ coverage not available (need to rebuild with ENABLE_COVERAGE=ON)" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    fi
    
    echo "---" >> $REPORT_FILE
    echo "" >> $REPORT_FILE
    echo "**Note**: Coverage based on executed tests. Files with 0% were not executed." >> $REPORT_FILE
    
    # 输出报告
    echo ""
    cat $REPORT_FILE
    echo ""
    echo "====================================="
    echo "Report saved to: $REPORT_FILE"
    echo "====================================="
    
    cd examples
fi

exit $pytest_exit_code
