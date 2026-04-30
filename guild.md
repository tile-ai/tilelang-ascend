# TileLang 覆盖率报告生成指南

本文档记录了如何生成完整的代码覆盖率报告，包括 tilelang/、examples/ 和 src/ 三个目录。

---

## 覆盖率报告摘要

| 目录 | 类型 | 覆盖率 | 说明 |
|------|------|-------|------|
| **tilelang/** | Python | 49% | 核心框架代码 |
| **examples/** | Python | 93-100% | 测试用例代码 |
| **src/** | C++ | 41.2% 行覆盖率 | 后端 IR 变换和代码生成 |
| **src/** | C++ | 44.6% 函数覆盖率 | 后端 IR 变换和代码生成 |

---

## 执行流程

### 1. 清理旧数据

清理所有旧的覆盖率数据文件，确保新生成的报告是准确的。

```bash
# 清理 C++ 覆盖率数据
find build -name "*.gcda" -delete 2>/dev/null
find build -name "*.gcno" -delete 2>/dev/null

# 清理 Python 覆盖率数据
find examples -name ".coverage*" -delete 2>/dev/null

# 清理旧报告文件
rm -f coverage_report.md coverage_cpp.info coverage_cpp_filtered.info \
      examples/.coverage_combined .coverage_combined

echo "清理完成"
```

---

### 2. 重新编译 C++（带覆盖率选项）

C++ 覆盖率需要编译时添加 `--coverage` 编译选项，生成 `.gcno` 文件（覆盖率注释文件）。

```bash
cd build

# 清理旧的编译产物
make clean

# 重新配置 CMake，启用覆盖率选项
cmake .. -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug

# 编译 tilelang_objs（生成 .gcno 文件）
make tilelang_objs -j8

# 编译完整的 tilelang 和 tilelang_module（用于运行测试）
make -j8 tilelang tilelang_module

# 验证 .gcno 文件是否生成
find build -name "*.gcno" -type f | wc -l  # 应输出 68
```

**关键说明：**
- `DENABLE_COVERAGE=ON` 会添加 `-fprofile-arcs -ftest-coverage` 编译选项
- `.gcno` 文件记录源代码行号信息，用于覆盖率报告生成
- `.gcda` 文件会在运行测试时自动生成，记录实际执行数据

---

### 3. 运行所有测试用例（89个）

运行 `examples/` 目录下的所有测试，同时收集 Python 和 C++ 覆盖率数据。

```bash
cd examples
ENABLE_COVERAGE=true bash bench_test.sh
```

**执行内容：**
- 运行 **89 个测试用例**（88 通过，1 失败）
- 为每个 Python 测试生成 `.coverage.*` 数据文件
- 运行时生成 C++ 的 `.gcda` 文件（覆盖率运行数据）
- 自动合并覆盖率数据并生成报告

**输出：**
```
=====================================
Execution Summary
Total: 89 | Passed: 88 | Failed: 1
Pass rate: 98%
=====================================
Report saved to: coverage_report.md
=====================================
```

---

## 覆盖率数据收集原理

### Python 覆盖率收集

`bench_test.sh` 为每个测试脚本运行 coverage 工具：

```bash
# 每个测试运行时（第 103 行）
python -m coverage run \
    --source=$PROJECT_ROOT/tilelang,$PROJECT_ROOT/examples \
    --rcfile=$PROJECT_ROOT/.coveragerc \
    --data-file=$temp_dir/.coverage.$N \
    test_script.py
```

**关键参数：**
- `--source`：指定要追踪的源代码目录（tilelang/ 和 examples/）
- `--rcfile`：指定配置文件（`.coveragerc`）
- `--data-file`：指定覆盖率数据输出文件

### Python 覆盖率合并

测试完成后，合并所有 `.coverage.*` 文件：

```bash
# 合并所有覆盖率数据（第 153-154 行）
python -m coverage combine \
    --rcfile=$PROJECT_ROOT/.coveragerc \
    --data-file=.coverage_combined \
    $temp_dir/.coverage.*

# 生成报告（第 156 行）
python -m coverage report \
    --rcfile=$PROJECT_ROOT/.coveragerc \
    --data-file=.coverage_combined \
    --sort=Cover
```

### C++ 覆盖率收集

使用 `lcov` 工具收集 C++ 覆盖率：

```bash
# 收集 C++ 覆盖率数据（bench_test.sh 第 254-255 行）
lcov --capture --directory build --output-file coverage_cpp.info --no-external

# 提取 src/ 目录的覆盖率（排除其他路径）
lcov --extract coverage_cpp.info "*/tilelang-ascend/src/*" \
    --output-file coverage_cpp_filtered.info

# 生成摘要
lcov --summary coverage_cpp_filtered.info
```

---

## 核心代码修改

为了让 `examples/` 目录也能统计覆盖率，修改了 `bench_test.sh`：

### 1. 支持环境变量覆盖配置（第 9-11 行）

```bash
# 修改前
ENABLE_COVERAGE=false
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")

# 修改后
ENABLE_COVERAGE=${ENABLE_COVERAGE:-false}  # 支持环境变量覆盖
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
export PROJECT_ROOT  # 导出 PROJECT_ROOT 使子进程可访问
```

**原因：** 原代码硬编码 `false`，会覆盖环境变量 `ENABLE_COVERAGE=true`。

### 2. 添加 --source 参数（第 103 行）

```bash
# 修改前
output=$(cd "$script_dir" && python -m coverage run \
    --rcfile=$PROJECT_ROOT/.coveragerc \
    --data-file=$temp_dir/.coverage.$total_scripts \
    "$script_name" 2>&1)

# 修改后
output=$(cd "$script_dir" && python -m coverage run \
    --source=$PROJECT_ROOT/tilelang,$PROJECT_ROOT/examples \
    --rcfile=$PROJECT_ROOT/.coveragerc \
    --data-file=$temp_dir/.coverage.$total_scripts \
    "$script_name" 2>&1)
```

**原因：** coverage 工具默认只追踪被导入的模块，`--source` 参数强制追踪直接运行的脚本。

### 3. 修复 coverage 命令路径

将所有 `coverage` 命令改为 `python -m coverage`，确保命令可执行：

```bash
# 修改前
coverage run ...
coverage combine ...
coverage report ...

# 修改后
python -m coverage run ...
python -m coverage combine ...
python -m coverage report ...
```

### 4. 添加表格标题（第 230-235 行）

```bash
# 添加到报告生成部分
echo "| Name | Stmts | Miss | Cover |" >> $REPORT_FILE
echo "|------|-------|------|-------|" >> $REPORT_FILE
```

---

## 配置文件说明

### .coveragerc

覆盖率配置文件，位于项目根目录：

```ini
[run]
source =
    tilelang
    examples
omit =
    tilelang/3rdparty/*
    tilelang/__pycache__/*
    tilelang/lib/*
    examples/gemm_aot/*
    examples/dispatch_combine/*
    examples/shmem/*
    examples/torch_tl_ascend/*
    examples/sparse_flash_attention/bench_sfa.py
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:
    @abstractmethod
    pass

[html]
directory = htmlcov_combined
```

**关键配置：**
- `source`：指定追踪的源代码目录
- `omit`：排除不需要统计的目录
- `exclude_lines`：排除不需要统计的代码行

---

## 使用方法

### 快速生成覆盖率报告

```bash
# 一键生成完整报告（包括编译 + 测试 + 收集）
bash scripts/run_coverage.sh
```

### 仅运行测试收集覆盖率（跳过编译）

```bash
# 适用于已编译过的情况
cd examples && ENABLE_COVERAGE=true bash bench_test.sh
```

### 查看覆盖率报告

```bash
# Markdown 报告
cat coverage_report.md

# HTML 报告（可视化）
python -m coverage html --rcfile=.coveragerc --data-file=examples/.coverage_combined
# 打开 htmlcov_combined/index.html
```

---

## 覆盖率文件说明

### Python 覆盖率文件

| 文件 | 说明 |
|------|------|
| `.coverage` | 单次测试的覆盖率数据 |
| `.coverage.*` | 多次测试的覆盖率数据（临时） |
| `.coverage_combined` | 合并后的覆盖率数据（最终） |
| `coverage_report.md` | Markdown 格式报告 |
| `htmlcov_combined/` | HTML 格式报告 |

### C++ 覆盖率文件

| 文件 | 说明 |
|------|------|
| `*.gcno` | 覆盖率注释文件（编译时生成） |
| `*.gcda` | 覆盖率运行数据（测试时生成） |
| `coverage_cpp.info` | lcov 原始覆盖率数据 |
| `coverage_cpp_filtered.info` | 过滤后的覆盖率数据（仅 src/） |

---

## 高覆盖率模块

### Python（examples/）

多数测试文件达到 **93-100%** 覆盖率：
- `examples/activation/*.py`: 100%
- `examples/gemm/*.py`: 67-100%
- `examples/normalization/*.py`: 100%
- `examples/developer_mode/*.py`: 100%

### C++（src/）

核心模块覆盖率较高：

| 文件 | 行覆盖率 | 函数覆盖率 |
|------|-------|----------|
| op/ascend.cc | 92.6% | 98.1% |
| transform/ascend_collect_buffer_shape.cc | 94.7% | 92.9% |
| transform/cross_core_pipeline.cc | 93.4% | 98.1% |
| transform/ascend_sync_insert.cc | 88.7% | 98.4% |
| transform/ascend_memory_planning.cc | 89.6% | 91.7% |
| transform/ascend_combinecv.cc | 88.1% | 94.3% |
| transform/ascend_infer_buffer_scope.cc | 81.7% | 90.9% |
| target/codegen_ascend.cc | 67.9% | 70.0% |

---

## 常见问题

### Q1: examples/ 目录覆盖率显示为 0%？

**原因：** coverage 工具默认只追踪被导入的模块，examples 下是直接运行的脚本。

**解决：** 添加 `--source=$PROJECT_ROOT/tilelang,$PROJECT_ROOT/examples` 参数。

### Q2: C++ 覆盖率数据显示 "stamp mismatch"？

**原因：** `.gcno` 和 `.gcda` 文件时间戳不匹配，通常是编译后重新编译导致。

**解决：** 清理所有 `.gcno` 和 `.gcda` 文件，重新编译并运行测试。

```bash
find build -name "*.gcno" -delete
find build -name "*.gcda" -delete
cd build && cmake .. -DENABLE_COVERAGE=ON && make tilelang_objs -j8
```

### Q3: coverage 命令找不到？

**原因：** `coverage` 可执行文件不在 PATH 中。

**解决：** 使用 `python -m coverage` 替代 `coverage`。

### Q4: Python 覆盖率数据没有合并？

**原因：** `bench_test.sh` 中 `$temp_dir` 目录在合并前被清理。

**解决：** 确保合并逻辑在清理 `$temp_dir` 之前执行（第 147-160 行）。

---

## 参考链接

- [Python coverage.py 文档](https://coverage.readthedocs.io/)
- [lcov 文档](https://github.com/linux-test-project/lcov)
- [GCC Coverage 选项](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html)