#!/bin/bash

echo "Starting unified test script execution..."
echo "====================================="

total_scripts=0
passed_scripts=0
failed_scripts=0
failed_files=()

# 收集所有要执行的脚本文件
all_scripts=()

# 添加普通Python文件（排除gemm_aot目录以及reference，golden代码）
python_files=$(find . -maxdepth 2 -name "*.py" -not -path "./gemm_aot/*" -not -name "sfa_golden.py" | sort)
if [ -n "$python_files" ]; then
    for file in $python_files; do
        all_scripts+=("$file")
    done
fi

# 添加gemm_aot目录中的bash脚本
if [ -d "./gemm_aot" ]; then
    bash_scripts=$(find ./gemm_aot -maxdepth 1 -name "run_example_gemm_aot.sh" | sort)
    if [ -n "$bash_scripts" ]; then
        for script in $bash_scripts; do
            all_scripts+=("$script")
        done
    fi
fi

# 显示找到的所有脚本文件
if [ ${#all_scripts[@]} -eq 0 ]; then
    echo "No test scripts found."
    exit 0
fi

echo "Found test scripts:"
for script in "${all_scripts[@]}"; do
    echo "  - $script"
done
echo

# 统一执行所有脚本
for script in "${all_scripts[@]}"; do
    echo "Executing: $script"
    total_scripts=$((total_scripts + 1))

    # 获取脚本所在目录
    script_dir=$(dirname "$script")
    script_name=$(basename "$script")

    # 保存当前目录
    current_dir=$(pwd)

    # 根据文件类型选择执行方式
    if [[ "$script" == *.py ]]; then
        # 切换到脚本目录执行Python脚本
        cd "$script_dir" || exit
        output=$(python "$script_name" 2>&1)
        exit_code=$?
        script_type="Python"
    elif [[ "$script" == *.sh ]]; then
        # 切换到脚本目录执行Bash脚本
        cd "$script_dir" || exit
        output=$(bash "$script_name" 2>&1)
        exit_code=$?
        script_type="Bash"
    else
        echo "  Status: SKIPPED (unknown file type)"
        failed_scripts=$((failed_scripts + 1))
        failed_files+=("$script")
        echo
        continue
    fi

    # 返回原目录
    cd "$current_dir" || exit

    # 获取最后一行输出
    last_line=$(echo "$output" | tail -n 1)

    # 检查最后一行是否包含通过关键词（不区分大小写）
    if [[ "$last_line" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || [[ "$last_line" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]]; then
        echo "  Status: PASSED ($script_type)"
        echo "  Last line: $last_line"
        passed_scripts=$((passed_scripts + 1))
    else
        echo "  Status: FAILED ($script_type)"
        echo "  Exit code: $exit_code"
        echo "  Last line: $last_line"
        echo "  Full output (last 10 lines):"
        echo "  ----------------------------------------"
        echo "$output" | tail -n 10 | sed 's/^/  /'
        echo "  ----------------------------------------"
        
        if [ $exit_code -eq 0 ]; then
            echo "  Note: $script_type script executed successfully but didn't output expected pass phrase"
        else
            echo "  Error: $script_type script execution failed with exit code $exit_code"
        fi
        failed_scripts=$((failed_scripts + 1))
        failed_files+=("$script")
    fi
    echo
done

echo "====================================="
echo "Unified Execution Summary"
echo "====================================="
echo "Total scripts executed: $total_scripts"
echo "Passed: $passed_scripts"
echo "Failed: $failed_scripts"
echo

if [ $failed_scripts -gt 0 ]; then
    echo "Failed scripts:"
    for file in "${failed_files[@]}"; do
        echo "  - $file"
    done
    echo
fi
# Calculate and display percentage
if [ $total_scripts -gt 0 ]; then
    percentage=$((passed_scripts * 100 / total_scripts))
    echo "Pass rate: $percentage% ($passed_scripts/$total_scripts)"
else
    echo "Pass rate: 0% (0/0)"
fi

echo "====================================="