#!/bin/bash

# ================= 配置区 =================
MAX_JOBS=8  # 同时并行执行的任务数，建议根据 NPU 负载调整
# ==========================================

echo "Starting parallel unified test execution (Live Output)..."
echo "====================================="

total_scripts=0
passed_scripts=0
all_scripts=()

# 1. 收集脚本逻辑 (保持原样)
python_files=$(find . -maxdepth 2 -name "*.py" -not -path "./gemm_aot/*" -not -name "sfa_golden.py" -not -name "__init__.py" | sort)
if [ -n "$python_files" ]; then
    for file in $python_files; do all_scripts+=("$file"); done
fi
if [ -d "./gemm_aot" ]; then
    bash_scripts=$(find ./gemm_aot -maxdepth 1 -name "run_example_gemm_aot.sh" | sort)
    if [ -n "$bash_scripts" ]; then
        for script in $bash_scripts; do all_scripts+=("$script"); done
    fi
fi

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
        script_dir=$(dirname "$script")
        script_name=$(basename "$script")
        
        # 执行脚本并捕获输出到变量，不在磁盘生成日志文件
        if [[ "$script" == *.py ]]; then
            output=$(cd "$script_dir" && python "$script_name" 2>&1)
            exit_code=$?
        else
            output=$(cd "$script_dir" && bash "$script_name" 2>&1)
            exit_code=$?
        fi

        # 结果判定逻辑
        last_line=$(echo "$output" | tail -n 1)
        if [[ "$last_line" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || [[ "$last_line" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]]; then
            echo "[PASSED] $script"
            touch "$temp_dir/pass_$total_scripts"
        else
            echo "[FAILED] $script (Exit: $exit_code)"
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

# 4. 最后执行特定的 pytest (保持串行以确保环境稳定)
python ../testing/python/language/test_tilelang_ascend_language_parallel.py
python ../testing/python/language/test_tilelang_ascend_language_cast_on_copy.py
python ../testing/python/language/test_tilelang_ascend_language_parallel_discrete.py
