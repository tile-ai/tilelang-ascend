# TileLang-Ascend 覆盖率统计使用手册

## 快速开始

### 1. 编译（启用覆盖率）

```bash
bash install_ascend.sh --enable-coverage
```

该选项会在 CMake 配置中写入 `ENABLE_COVERAGE ON`，为 C++ 代码添加 `--coverage -fprofile-arcs -ftest-coverage` 编译选项。

### 2. 设置环境

```bash
source set_env.sh
```

### 3. 运行全量测试

```bash
bash bench_test.sh --coverage --enable-cpp-coverage
```

- `--coverage`：启用 Python 覆盖率统计（`coverage run` + `pytest --cov`）
- `--enable-cpp-coverage`：启用 C++ 覆盖率统计（`lcov` 采集）

## 产物位置

| 产物 | 路径 |
|------|------|
| Python 覆盖率 JSON | `coverage_data/coverage.json` |
| C++ 覆盖率 info | `coverage_data/coverage.info` |
| 核心文件覆盖率报告 | `core_files_coverage_report.md` |

## 查看具体文件覆盖率详情

### Python 文件（逐行可视化）

```bash
python scripts/generate_simple_html_coverage.py <file_path>
```

示例：

```bash
python scripts/generate_simple_html_coverage.py tilelang/language/ascend_tile.py
```

输出：`coverage_reports/simple_html/` 下生成 HTML 文件，绿色=已执行，红色=未执行。

批量生成覆盖率低于 80% 的文件：

```bash
python scripts/generate_simple_html_coverage.py --all --threshold 80
```

批量生成所有文件的覆盖率报告：

```bash
python scripts/generate_simple_html_coverage.py --all
```

### C++ 文件（逐行可视化）

```bash
python scripts/generate_cpp_coverage_html.py <file_name>
```

示例：

```bash
python scripts/generate_cpp_coverage_html.py ascend_lower_parallel_to_vector
```

输出：`coverage_reports/cpp_html/` 下生成 HTML 文件。

列出所有 C++ 文件覆盖率概览：

```bash
python scripts/generate_cpp_coverage_html.py --list
```