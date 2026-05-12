#!/bin/bash
# 验证 coverage 清除逻辑是否完整

echo "======================================="
echo "Coverage Cleanup Verification"
echo "======================================="

cd "$(dirname "$0")/.."

echo ""
echo "检查 bench_test.sh 清除逻辑："
echo ""

# 检查 Python coverage 清除
echo "1. Python coverage 清除："
if grep "rm.*coverage_data/.coverage" examples/bench_test.sh; then
    echo "  ✓ 清除 coverage_data/.coverage*"
else
    echo "  ✗ 缺少 coverage_data 清除"
fi

if grep "find.*examples.*coverage.*delete" examples/bench_test.sh; then
    echo "  ✓ 清除 examples 子目录"
else
    echo "  ✗ 缺少 examples 子目录清除"
fi

if grep "find.*testing/python.*coverage.*delete" examples/bench_test.sh; then
    echo "  ✓ 清除 testing/python 子目录"
else
    echo "  ✗ 缺少 testing/python 清除"
fi

# 检查 C++ coverage 清除
echo ""
echo "2. C++ coverage 清除："
if grep "find.*build.*gcda.*delete" examples/bench_test.sh; then
    echo "  ✓ 清除 build .gcda 文件"
else
    echo "  ✗ 缺少 .gcda 清除"
fi

if grep "rm.*coverage_data/coverage.info" examples/bench_test.sh; then
    echo "  ✓ 清除旧的 coverage.info"
else
    echo "  ✗ 缺少 coverage.info 清除"
fi

# 检查报告清除
echo ""
echo "3. 报告清除："
if grep "rm.*coverage_report.md" examples/bench_test.sh; then
    echo "  ✓ 清除旧报告"
else
    echo "  ✗ 缺少报告清除"
fi

# 检查执行条件
echo ""
echo "4. 执行条件："
if grep "ENABLE_COVERAGE.*true" examples/bench_test.sh | head -3; then
    echo "  ✓ 只在 --coverage 时执行清除"
else
    echo "  ✗ 清除条件不明确"
fi

echo ""
echo "======================================="
echo "Verification Complete"
echo "======================================="
