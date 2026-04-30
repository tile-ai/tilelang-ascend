#!/bin/bash
# TileLang Coverage Runner - 统计 tilelang/ examples/ src/ 三目录覆盖率
#
# 使用方法：
#   完整运行：bash scripts/run_coverage.sh
#   分步运行：
#     1. 编译：bash scripts/run_coverage.sh --skip-tests
#     2. 测试：cd examples && ENABLE_COVERAGE=true bash bench_test.sh
#     3. 收集：bash scripts/run_coverage.sh --skip-rebuild --skip-tests

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "======================================"
echo "TileLang Coverage Runner"
echo "======================================"
echo "统计范围："
echo "  Python: tilelang/ examples/"
echo "  C++:    src/"
echo ""

# 参数解析
REBUILD_CPP=true
RUN_TESTS=true
CLEANUP=true
COLLECT_CPP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-rebuild)
            REBUILD_CPP=false
            shift
            ;;
        --skip-tests)
            RUN_TESTS=false
            shift
            ;;
        --skip-collect)
            COLLECT_CPP=false
            shift
            ;;
        --skip-cleanup)
            CLEANUP=false
            shift
            ;;
        --only-tests)
            REBUILD_CPP=false
            COLLECT_CPP=false
            shift
            ;;
        --only-collect)
            REBUILD_CPP=false
            RUN_TESTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "可用选项:"
            echo "  --skip-rebuild    跳过重新编译"
            echo "  --skip-tests      跳过运行测试"
            echo "  --skip-collect    跳过收集C++覆盖率"
            echo "  --skip-cleanup    跳过清理临时文件"
            echo "  --only-tests      只运行测试"
            echo "  --only-collect    只收集覆盖率报告"
            exit 1
            ;;
    esac
done

# 1. 重新编译 C++（带覆盖率选项）
if [[ "$REBUILD_CPP" == true ]]; then
    echo "======================================"
    echo "[Step 1] Rebuilding C++ with coverage flags..."
    echo "======================================"
    
    cd "$PROJECT_ROOT"
    
    # 清理旧的覆盖率数据
    find . -name "*.gcda" -type f -delete 2>/dev/null || true
    find . -name "*.gcno" -type f -delete 2>/dev/null || true
    
    # 重新构建
    if [ -d "build" ]; then
        cd build
        cmake .. -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
        make -j4  # 减少并发数避免内存不足
    else
        mkdir -p build && cd build
        cmake .. -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
        make -j4
    fi
    
    cd "$PROJECT_ROOT"
fi

# 2. 运行测试（带覆盖率）
if [[ "$RUN_TESTS" == true ]]; then
    echo ""
    echo "======================================"
    echo "[Step 2] Running tests with coverage..."
    echo "======================================"
    echo "提示：此步骤可能需要较长时间..."
    echo ""
    
    cd "$PROJECT_ROOT/examples"
    ENABLE_COVERAGE=true bash bench_test.sh
    cd "$PROJECT_ROOT"
fi

# 3. 收集 C++ 覆盖率
if [[ "$COLLECT_CPP" == true ]]; then
    echo ""
    echo "======================================"
    echo "[Step 3] Collecting C++ coverage data..."
    echo "======================================"

    cd "$PROJECT_ROOT"

    # 确保库文件在正确位置
    mkdir -p tilelang/lib
    cp build/libtilelang.so build/libtilelang_module.so tilelang/lib/ 2>/dev/null || true

    # 初始化 lcov
    lcov --zerocounters --directory build --quiet 2>/dev/null || true

    # 捕获覆盖率数据
    if find build -name "*.gcda" | grep -q .; then
        echo "Found .gcda files, generating coverage report..."
        lcov --capture --directory build --output-file coverage_cpp.info --no-external --quiet 2>/dev/null || true
        
        # 只保留 src/ 目录的覆盖率
        lcov --extract coverage_cpp.info "*/src/*" --output-file coverage_cpp_filtered.info --quiet 2>/dev/null || \
        cp coverage_cpp.info coverage_cpp_filtered.info
        
        echo "C++ coverage data collected successfully"
    else
        echo "WARNING: No .gcda files found. Need to run tests first."
        echo "Run: cd examples && ENABLE_COVERAGE=true bash bench_test.sh"
    fi
fi

# 4. 显示 Python 覆盖率汇总
echo ""
echo "======================================"
echo "[Step 4] Coverage Summary"
echo "======================================"

echo ""
echo "--- Python Coverage (tilelang/ + examples/) ---"
if [ -f "$PROJECT_ROOT/examples/.coverage_combined" ]; then
    cd "$PROJECT_ROOT"
    cp examples/.coverage_combined . 2>/dev/null || true
    coverage report --rcfile=.coveragerc --data-file=.coverage_combined --sort=Cover | tail -30
    echo ""
    echo "Python coverage data collected successfully"
elif [ -f "$PROJECT_ROOT/.coverage_combined" ]; then
    coverage report --rcfile=.coveragerc --data-file=.coverage_combined --sort=Cover | tail -30
    echo ""
    echo "Python coverage data collected successfully"
else
    echo "No Python coverage data found"
    echo "Run: cd examples && ENABLE_COVERAGE=true bash bench_test.sh"
fi

echo ""
echo "--- C++ Coverage (src/) ---"
if [ -f "$PROJECT_ROOT/coverage_cpp_filtered.info" ]; then
    lcov --summary coverage_cpp_filtered.info 2>&1 | grep -E "(lines|functions|branches)" || true
    echo ""
    echo "C++ coverage data collected successfully"
else
    echo "No C++ coverage data found"
fi

# 5. 清理
if [[ "$CLEANUP" == true ]]; then
    echo ""
    echo "======================================"
    echo "[Step 5] Cleanup"
    echo "======================================"
    rm -f coverage_cpp.info coverage_cpp_filtered.info
    echo "Coverage reports generated in:"
    echo "  - Markdown report: $PROJECT_ROOT/coverage_report.md"
fi

echo ""
echo "======================================"
echo "Done!"
echo "======================================"