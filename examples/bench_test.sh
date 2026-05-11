#!/bin/bash

# ================= 参数解析 =================
SKIP_PYTEST=false
TEST_DIRS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pytest)
            SKIP_PYTEST=true
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

echo "Starting parallel unified test execution (Live Output)..."
echo "====================================="

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
        -not -path "*/moe_pytorch_reference/*" \
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
                output=$(cd "$script_dir" && python "$script_name" 2>&1)
                exit_code=$?
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

# 运行 pytest 并捕获输出（使用 tee 同时显示和保存）
pytest --forked ../testing/python/ -v -n $MAX_JOBS 2>&1 | tee pytest_output.log
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

# 输出合并后的结果（用于 CI workflow 解析）
# xfailed 是预期失败的测试，在 pytest 视角下属于"成功"状态（符合预期）
# 应计入 passed_all，而不应计入 failed_all
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

exit $pytest_exit_code
