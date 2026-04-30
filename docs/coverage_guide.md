# TileLang Coverage 测试指南

统计 `tilelang/`、`examples/`、`src/` 三目录的代码覆盖率。

## 快速开始

```bash
# 完整运行（编译 + 测试 + 收集）
bash scripts/run_coverage.sh
```

## 分步执行（推荐）

编译和测试耗时较长，建议分步执行：

### Step 1: 编译 C++（带覆盖率选项）

```bash
bash scripts/run_coverage.sh --skip-tests
```

### Step 2: 运行测试

```bash
cd examples
ENABLE_COVERAGE=true bash bench_test.sh
cd ..
```

### Step 3: 收集覆盖率报告

```bash
bash scripts/run_coverage.sh --skip-rebuild --skip-tests
```

## 覆盖率报告位置

| 报告 | 路径 | 说明 |
|-----|------|------|
| Python 综合 | `htmlcov_combined/index.html` | tilelang/ + examples/ |
| C++ | `htmlcov_cpp/index.html` | src/ 目录 |

## 统计范围

| 目录 | 文件数 | 说明 |
|-----|-------|------|
| `tilelang/` | 143 个 Python 文件 | 核心框架代码 |
| `examples/` | 83 个 Python 文件 | 测试脚本（排除三方库） |
| `src/` | 128 个 C++ 文件 | 编译引擎、IR 变换、代码生成 |

**排除的内容**：
- `tilelang/3rdparty/`（三方库）
- `examples/gemm_aot/`、`examples/shmem/` 等（排除目录）
- `3rdparty/`（全局三方库）

## 常用选项

```bash
# 只运行测试（不编译、不收集C++）
bash scripts/run_coverage.sh --only-tests

# 只收集报告（不编译、不测试）
bash scripts/run_coverage.sh --only-collect

# 保留临时文件（用于调试）
bash scripts/run_coverage.sh --skip-cleanup
```

## 注意事项

1. **首次编译**需要重新构建 TVM + TileLang，耗时约 10-30 分钟
2. **测试运行**会执行所有 examples 和 pytest，耗时约 30-60 分钟
3. **内存需求**：编译时并发数已限制为 4，避免内存不足
4. **覆盖率数据**：
   - Python: `.coverage_combined`
   - C++: `coverage_cpp_filtered.info`

## 技术细节

- **Python 覆盖率**：使用 `coverage.py` + `pytest-cov`
- **C++ 覆盖率**：使用 `gcov` + `lcov` + `genhtml`
- **编译选项**：`--coverage -fno-inline`（Debug 模式）

## 配置文件

- `.coveragerc`：Python 覆盖率配置
- `CMakeLists.txt`：C++ 覆盖率编译选项（`ENABLE_COVERAGE=ON`）