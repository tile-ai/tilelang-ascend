---
name: tilelang-run-examples
description: TileLang-Ascend 全量运行 examples 目录下算子脚本的技能。支持选择后端类型（auto/ascendc/pto/ascendc+pto 双后端）、是否运行 aclgraph 脚本、并行控制等。触发关键词："全量运行"、"run all examples"、"跑全部算子"、"运行所有示例"、"批量测试"、"全量测试"、"run examples"、"bench all"、"跑 examples"、"运行样例"、"运行示例"、"跑所有算子"、"双后端"、"两个后端都跑"。
---

# TileLang Run Examples

全量运行 `examples/` 目录下的算子脚本，支持后端选择（单后端或 ascendc+pto 双后端一次运行）和 aclgraph 控制。

## ⚠️ 核心规则：强制交互流程

**Agent 触发此 skill 后，必须严格按"⭐ Agent 强制执行流程"中的 5 个步骤执行，每个需要交互的步骤都必须使用 `question` 工具向用户提问，禁止跳过询问直接执行操作。**

---

## 路径与参数说明

- **`<skill-path>`**：本 SKILL.md 所在目录的绝对路径。Agent 调用脚本时，必须将此占位符替换为该目录的实际绝对路径。
- **`--project-root`**：tilelang-ascend 项目根目录（包含 `set_env.sh` 的目录）。**不传时默认为当前工作目录（cwd）**，即 opencode 打开项目时的根路径。

## 执行步骤

核心脚本：

1. **`<skill-path>/scripts/run_examples.sh`** — 主执行脚本，负责环境准备、脚本收集、并行执行、结果统计
2. **`<skill-path>/scripts/run_examples_multi.sh`** — 多后端 wrapper，循环调用 `run_examples.sh`，每个后端产出独立日志（`run_<backend>.log`）

```bash
# 单后端
bash <skill-path>/scripts/run_examples.sh --backend <auto|ascendc|pto> [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]] [--dirs <dirs>] [--max-jobs N] [--project-root <path>]

# 多后端（ascendc + pto 一次运行，各自产出 run_ascendc.log / run_pto.log）
bash <skill-path>/scripts/run_examples_multi.sh --backend <both|ascendc,pto> [--output-dir <path>] [其他 run_examples.sh 选项...]
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backend <auto\|ascendc\|pto>` | 编译后端类型（`run_examples.sh`）。`pto`/`ascendc` 时通过环境变量 `TILELANG_JIT_TARGET` 覆盖 `target="auto"`，无需修改源文件 | `auto` |
| `--backend <both\|ascendc,pto>` | 多后端（仅 `run_examples_multi.sh`）。`both` 或逗号分隔的 `ascendc,pto`，按顺序依次运行，每个后端产出独立日志 | 无 |
| `--output-dir <path>` | 多后端日志输出目录（仅 `run_examples_multi.sh`，不转发给 `run_examples.sh`） | `./tmp` |
| `--skip-aclgraph[=true\|false]` | 跳过 `aclgraph/` 目录下的脚本（A3 环境，算力平台不支持，运行会导致环境崩溃）。`--skip-aclgraph` 或 `--skip-aclgraph=true` 开启跳过；`--skip-aclgraph=false` 关闭跳过（即运行） | 跳过（默认 true） |
| `--skip-pytest[=true\|false]` | 跳过 pytest 测试阶段。`--skip-pytest` 或 `--skip-pytest=true` 开启跳过；`--skip-pytest=false` 关闭跳过（即运行 pytest） | 跳过（默认 true） |
| `--dirs <dirs...>` | 只运行指定目录（增量模式） | 全量运行 |
| `--max-jobs N` | 最大并行任务数 | 8 |
| `--project-root <path>` | 项目根目录路径 | cwd（当前工作目录） |
| `--task-timeout <秒>` | 单任务超时秒数。超时则 SIGTERM（30s 后 SIGKILL）并标记 `[TIMEOUT]`。`0` 禁用 | 600 |
| `--pytest-timeout <秒>` | pytest 阶段整体超时秒数（与 `--task-timeout` 独立）。pytest 跑数百用例，单任务的 600s 对整阶段过紧，会在打印汇总行前被杀导致统计为 0。`0` 禁用 | 1800 |

## 后端切换机制（环境变量）

`--backend pto` 或 `--backend ascendc` 通过设置 `TILELANG_JIT_TARGET` 环境变量实现后端切换，**无需修改任何源文件**。

### 优先级规则

```
TILELANG_JIT_TARGET 环境变量 > 装饰器 target 参数 > 默认值 "auto"

env=TILELANG_JIT_TARGET=pto       +  target未设/"auto"  →  target="pto"       (覆盖)
env=TILELANG_JIT_TARGET=ascendc   +  target未设/"auto"  →  target="ascendc"   (覆盖)
env=TILELANG_JIT_TARGET=pto       +  target="ascendc"   →  target="ascendc"   (保留显式意图)
env=TILELANG_JIT_TARGET=ascendc   +  target="pto"       →  target="pto"       (保留显式意图)
env=未设置                        +  target未设          →  target="auto"      (默认)
```

**关键行为**：环境变量仅覆盖 `target="auto"`（默认值），不覆盖显式指定的 `target="ascendc"` 或 `target="pto"`，保留用户的显式意图。

### 运行时注入机制（Import Hook）

环境变量覆盖通过 `usercustomize.py`（位于 `scripts/` 目录）实现，**不修改 `tilelang/` 任何源文件**。

工作原理：

1. `run_examples.sh` 设置 `TILELANG_JIT_TARGET` 环境变量后，将 `$SCRIPT_DIR` 注入 `PYTHONPATH`
2. Python 启动时自动导入 `usercustomize.py`（Python `site` 模块的标准行为）
3. `usercustomize.py` 注册一个 `sys.meta_path` Finder，拦截 `tilelang.jit` 模块加载
4. 模块加载完成后，monkey-patch `_JitImplementation.__init__`，将 `target="auto"` 替换为环境变量值
5. 显式指定的 `target="ascendc"` 或 `target="pto"` 不受影响

```python
# usercustomize.py 核心逻辑（简化）
_orig_init = _JitImplementation.__init__
def _patched_init(self, *args, **kwargs):
    if kwargs.get("target", None) == "auto" or (len(args) >= 3 and args[2] == "auto"):
        # replace target="auto" with env var value
        ...
    _orig_init(self, *args, **kwargs)
```

该补丁适用于所有 Python 调用方式：直接 `python script.py`、Shell 脚本内嵌 Python 调用（如 `gemm_aot/run_example_gemm_aot.sh`）、以及 pytest 运行。

### 对比旧方案（源码修改）

| 维度 | Import Hook 方案（当前） | 源码修改方案（旧，已废弃） |
|------|-------------------------|-------------------------|
| 源文件 | **不修改** | 逐文件修改 `@tilelang.jit` |
| 注入方式 | `PYTHONPATH` + `usercustomize.py` | `sed` / 手动修改源码 |
| 风险 | 无风险（env 退出即失效） | 进程中断可能残留 `.jit_orig` |
| git 状态 | 无 dirty state | 源文件被修改 |
| 覆盖范围 | 仅覆盖 `target="auto"` | 修改所有不含 `target="pto"/"ascendc"` 的文件 |
| 恢复 | 自动（env var 生命周期） | 需 `trap cleanup EXIT` 恢复 |

## 使用示例

> 以下示例中 `<skill-path>` 需替换为本 skill 所在目录的绝对路径。`--project-root` 省略时默认使用 cwd。

```bash
# 默认运行：auto 后端，默认跳过 aclgraph，不含 pytest（cwd 为项目根目录时无需 --project-root）
bash <skill-path>/scripts/run_examples.sh

# pto 后端（通过环境变量覆盖，不修改源文件）
bash <skill-path>/scripts/run_examples.sh --backend pto

# 双后端一次运行（ascendc + pto），各自产出 run_ascendc.log / run_pto.log
bash <skill-path>/scripts/run_examples_multi.sh --backend both --output-dir ./tmp

# 双后端指定顺序（先 pto 后 ascendc）
bash <skill-path>/scripts/run_examples_multi.sh --backend pto,ascendc --output-dir ./tmp

# 不跳过 aclgraph（注意：A3 环境，算力平台不支持，运行可能导致环境崩溃）
bash <skill-path>/scripts/run_examples.sh --backend pto --skip-aclgraph=false

# 只运行指定目录
bash <skill-path>/scripts/run_examples.sh --dirs "gemm softmax normalization"

# 控制并行度
bash <skill-path>/scripts/run_examples.sh --max-jobs 4

# 调整单任务超时（默认 600s，设为 0 禁用保护）
bash <skill-path>/scripts/run_examples.sh --task-timeout 900
bash <skill-path>/scripts/run_examples.sh --task-timeout 0

# 启用 pytest（默认跳过，加 --skip-pytest=false 启用）
bash <skill-path>/scripts/run_examples.sh --skip-pytest=false

# 手动设置环境变量（不通过脚本，直接运行单个示例）
# 必须同时设置 PYTHONPATH 以加载 usercustomize.py 补丁
export TILELANG_JIT_TARGET=pto
export PYTHONPATH=<skill-path>/scripts${PYTHONPATH:+:$PYTHONPATH}
python examples/gemm/example_gemm.py

# 显式指定 target 不被环境变量覆盖
export TILELANG_JIT_TARGET=pto
python examples/developer_mode/gemm_developer.py  # target="ascendc" preserved
```

## 特殊目录处理

以下目录有特殊处理：

| 目录 | 处理方式 |
|------|---------|
| `dispatch_combine/` | 完全排除，不收集任何脚本（依赖 shmem 模块） |
| `shmem/` | 完全排除，不收集任何脚本（依赖 shmem 模块） |
| `gemm_aot/` | 只运行 `run_example_gemm_aot.sh` |
| `torch_tl_ascend/` | 只运行 `test_example.sh` |
| `flash_attention/` | 只收集主目录 `.py`，不收集 `fa_opt/`（全量模式单独处理 `fa_opt/`） |
| `aclgraph/` | 默认跳过（`--skip-aclgraph`），`--skip-aclgraph=false` 时运行 |
| `sparse_flash_attention/` | 通过 EXTRA_TASKS 运行 bench_sfa 子目录的特定脚本 |

## 结果判定

脚本通过以下模式判定测试通过：

- 输出包含 `Kernel Output Match`（大小写不敏感）
- 输出包含 `TEST PASSED!`（大小写不敏感）
- 自定义任务（CUSTOM_TASK）退出码为 0

## 执行流程

```
参数解析 → 环境准备（source set_env.sh） → 后端覆盖（若 pto/ascendc，设置 TILELANG_JIT_TARGET） →
脚本收集 → 并行执行 → 结果统计 → pytest（若 `--skip-pytest=false` 启用） → 输出汇总
```

## 输出格式

运行中每个任务完成时输出进度行（`(完成序号/总数)` 前缀，并行下多行输出由 flock 串行化不会交错）：

```
Total scripts to run: 118
=====================================
(1/118) [PASSED] gemm/example_gemm.py
(2/118) [FAILED] some/fail.py (Exit: 139)
  Last line: Segmentation fault
  <details...>
(3/118) [TIMEOUT] some/hung.py (Exit: 124)
  Last line: Timeout after 600s
  <details...>
(4/118) [PASSED] softmax/example_softmax.py
...
```

最终汇总：

```
=====================================
Final Execution Summary (Bench + Pytest)
  Backend: pto
  Aclgraph: skipped (--skip-aclgraph, default)
Bench: Total: 118 | Passed: 115 | Failed: 3
Pytest: Passed: 42 | Failed: 1 | Xfailed: 2 (expected failures, counted as passed)
Total: 163 | Passed: 159 | Failed: 4
Pass rate: 97%
=====================================
```

## ⭐ Agent 强制执行流程（不可跳过任何步骤）

**以下 5 个步骤必须严格按顺序执行，需要交互的步骤必须与用户确认后才进入下一步。禁止跳过任何询问步骤直接执行。**

### 步骤 1：确认运行配置

> 本步骤的提问次数不设上限。自定义路径含 1 次「默认/自定义」总选 + 最多 4 次逐项问（1.2a/1.2b/1.2c/1.2d），共最多 5 次 `question` 调用。不得以"轮次限制"为由合并或裁剪 1.2 的问题。

#### Step 1.1（必须使用 question 工具）

展示默认配置摘要，询问用户使用默认配置还是自定义配置：

```
默认运行配置：
  • 后端类型：auto
  • 跳过 aclgraph：是（A3 环境，算力平台不支持，运行会导致环境崩溃）
  • 运行 pytest：否
  • 并发数：8
```

选项：
- **「使用默认配置」** — 直接以默认参数进入步骤 2
- **「自定义配置」** — 进入 Step 1.2 逐个询问

→ 用户选默认配置 → 直接跳到步骤 2，不再询问其他问题
→ 用户选自定义 → 继续 Step 1.2

#### Step 1.2（仅在用户选择"自定义配置"时执行）

以下四个问题必须**逐个询问**，每个问题**单独一次 `question` 工具调用**（`questions` 数组中只能有一个 question），**等用户回答当前问题后才问下一个**。

**严禁将多个问题合并在同一次 `question` 调用中。**

> ⚠️ **硬约束：无论用户选择自定义的原因是什么（哪怕只想改后端），1.2a / 1.2b / 1.2c / 1.2d 四项必须全部逐个问完，不得以"其余保持默认"为由跳过任何一项。** 1.2a 问完用户的回答不决定是否继续问 1.2b/1.2c/1.2d——四项必问。

- **1.2a** 单独询问后端类型（auto、ascendc、pto 或 ascendc+pto 双后端，默认 auto）
  - 选「ascendc + pto」时，使用 `run_examples_multi.sh --backend both`，两个后端依次运行，各自产出 `run_ascendc.log` / `run_pto.log`
  - 单后端时使用 `run_examples.sh --backend <选定后端>`
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2b** 单独询问是否跳过 aclgraph（默认跳过；可选择 `--skip-aclgraph=false` 运行）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2c** 单独询问是否运行 pytest（默认跳过；可选择 `--skip-pytest=false` 启用）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2d** 单独询问并发数 `--max-jobs`（默认 8；NPU 负载高时可降低，如 4 或 2）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题

> ⛔ **进入步骤 2 前的自检门：Agent 必须自检 1.2a / 1.2b / 1.2c / 1.2d 是否都已得到用户明确回答。若有任一未问，必须回退补问，禁止直接进入步骤 2。**

### 步骤 2：运行测试

运行测试脚本，输出 tee 到 `./tmp/` 目录下的日志文件。**`--project-root` 省略时默认为当前工作目录（cwd）。** 先创建 `./tmp/` 目录再执行。

**单后端**（auto / ascendc / pto）：

```bash
mkdir -p ./tmp && bash <skill-path>/scripts/run_examples.sh --backend <auto|ascendc|pto> [--project-root <cwd>] [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]] --max-jobs <N> 2>&1 | tee ./tmp/run_<backend>.log
```

**双后端**（ascendc + pto）：wrapper 自动将每个后端输出 tee 到 `./tmp/run_<backend>.log`，无需手动 tee：

```bash
mkdir -p ./tmp && bash <skill-path>/scripts/run_examples_multi.sh --backend both --output-dir ./tmp [--project-root <cwd>] [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]] --max-jobs <N>
```

示例（pto 后端，cwd 默认为项目根目录）：

```bash
mkdir -p ./tmp && bash <skill-path>/scripts/run_examples.sh --backend pto --skip-aclgraph --skip-pytest=false 2>&1 | tee ./tmp/run_pto.log
```

### 步骤 3：询问是否导出 Excel（必须使用 question 工具）

测试完成后，**必须询问用户是否导出 Excel**，不能自动跳过此步直接导出。

**必须使用 `question` 工具提问，选项包括"导出（默认）"和"跳过"。**

### 步骤 4：用户确认后导出 Excel

仅当用户确认导出后，才运行导出脚本（输出目录默认 `./tmp/`，脚本会自动创建）：

**单后端**：对本次日志导出一次，生成 `Round 1 (后端)` Sheet：

```bash
python <skill-path>/scripts/export_to_excel.py --log ./tmp/run_<backend>.log --backend <auto|ascendc|pto>
```

**双后端**：对每个后端日志各导出一次（同一 xlsx），生成 `Round 1 (ascendc)` + `Round 2 (pto)` + 自动跨后端对比 Sheet：

```bash
python <skill-path>/scripts/export_to_excel.py --log ./tmp/run_ascendc.log --backend ascendc
python <skill-path>/scripts/export_to_excel.py --log ./tmp/run_pto.log --backend pto
```

> 多后端时按 ascendc 在前、pto 在后的顺序导出，使 Round1=ascendc、Round2=pto，对比 Sheet 自动呈现「哪些脚本 ascendc 通过但 pto 失败」等跨后端差异。

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--log` | run_examples.sh 输出日志文件路径（必填） | 无 |
| `--backend` | 本次测试使用的后端类型（必填，auto、ascendc 或 pto） | 无 |
| `--output-dir` | Excel 输出目录 | `./tmp` |

Excel 输出到 `./tmp/` 目录，文件名 `run_examples_results.xlsx`。

### 步骤 5：向用户汇报导出结果和对比摘要

### 流程检查清单

Agent 在执行过程中必须自检每一步是否完成：

| 步骤 | 检查项 | 是否与用户交互 |
|------|--------|--------------|
| 1.1 | 是否用 question 工具询问了默认/自定义配置？ | ✅ 必须 |
| 1.2a | 自定义模式下，是否单独一次 question 调用问了后端类型？ | ✅ 必须 |
| 1.2b | 自定义模式下，是否单独一次 question 调用问了是否跳过 aclgraph？ | ✅ 必须 |
| 1.2c | 自定义模式下，是否单独一次 question 调用问了是否运行 pytest？ | ✅ 必须 |
| 1.2d | 自定义模式下，是否单独一次 question 调用问了并发数（--max-jobs）？ | ✅ 必须 |
| 1.2 自检 | 进入步骤 2 前，1.2a/1.2b/1.2c/1.2d 是否都已得到用户明确回答？ | ✅ 必须 |
| 2 | 是否将输出 tee 到日志文件？ | ❌ 不需要 |
| 3 | 是否用 question 工具询问了是否导出 Excel？ | ✅ 必须 |
| 4 | 是否仅在用户确认后才执行导出？ | ✅ 必须确认 |
| 5 | 是否汇报了最终结果？ | ❌ 不需要 |

**如果任何"必须"交互的步骤被跳过，视为流程违规，必须回退补执行。**

### 多轮测试对比机制

多次运行测试后，Excel 文件会自动追加新轮次结果并维护对比分析：

- **第 1 次运行**：创建 Excel，生成 Sheet "Round 1 (auto/ascendc/pto)"，含逐项详情 + 失败分类汇总
- **第 2 次运行**：追加 Sheet "Round 2 (auto/ascendc/pto)"，自动生成 "对比分析" Sheet（逐项对比 R1 vs R2，标记 FIXED/NEW FAIL/无变化）
- **第 N 次运行**：追加 Sheet "Round N"，更新 "对比分析" Sheet（多轮对比，最新轮次 vs 上一轮次变化）

### 双后端对比机制

使用 `--backend both` 一次运行 ascendc + pto 后，导出 Excel 时对两个日志依次导出（先 ascendc 后 pto），同一 xlsx 会生成：

- `Round 1 (ascendc)`：ascendc 后端逐项结果
- `Round 2 (pto)`：pto 后端逐项结果
- `对比分析`：自动呈现 ascendc→pto 的逐项变化（标记 FIXED/NEW FAIL/无变化）

> 注意：对比 Sheet 的 "FIXED/NEW FAIL" 在双后端场景下含义为「ascendc 失败但 pto 通过 / ascendc 通过但 pto 失败」，反映后端能力差异而非代码回归。

### Excel 文件结构

| Sheet | 内容 | 何时生成 |
|-------|------|---------|
| Round 1 (auto/ascendc/pto) | 逐项测试详情（序号、脚本、结果、失败类型、失败详情）+ 汇总 | 第 1 次运行 |
| Round N (auto/ascendc/pto) | 同上，追加第 N 轮结果 | 第 N 次运行 |
| 失败分类汇总 | 最新一轮的失败类型分类统计 | 每次运行时更新 |
| 对比分析 | 多轮逐项对比 + 汇总统计（修复数/新增失败数/通过率变化） | 第 2 次及以后运行 |

### 对比分析 Sheet 字段说明

- **变化(最新vs上一轮)** 列标记：
  - `FIXED`（绿色）：上一轮失败 → 最新轮通过
  - `NEW FAIL`（红色）：上一轮通过 → 最新轮失败
  - `无变化`：两轮结果相同
  - `变化`（黄色）：其他状态变化

### 失败分类规则

导出脚本自动根据错误信息分类失败类型：

| 失败类型 | 匹配关键词 |
|---------|-----------|
| 编译失败 | `Compilation Failed!` |
| pto不支持 | `Unsupport SyncAll` / `Unresolved call Op(tl.ascend_reinterpretcast)` |
| 精度不匹配 | `Mismatched elements` / `accuracy:` / `The precision is not correct` |
| NPU设备错误 | `vector::reserve` / `aicore exception` / `open device failed` |
| 超时(TIMEOUT) | `Exit: 124` / `Timeout after Ns` / `[TIMEOUT]` 标记 |
| 段错误(Segfault) | `Exit: 139` |
| 内部错误 | `Downcast ... failed` |
| 运行时错误 | 其他 Exit code |

## 注意事项

- **超时保护**：默认每个任务最多运行 600s，超时则 SIGTERM（30s grace 后 SIGKILL）并标记 `[TIMEOUT]`，避免 NPU 死锁/同步原语 hang 导致整个脚本永久阻塞。`--task-timeout 0` 可禁用此保护
- **pytest 阶段独立超时**：pytest 阶段默认 1800s（`--pytest-timeout`），与单任务的 `--task-timeout` 解耦。pytest 跑数百用例，若复用 600s 的单任务超时会在打印汇总行前被整体杀掉，导致 `Pytest: Passed: 0 | Failed: 0` 的错误统计（实际有大量用例已跑完，只是未完成收尾）。`--pytest-timeout 0` 可禁用
- 环境变量 `TILELANG_JIT_TARGET` 仅在当前 shell 生命周期内生效，退出后自动失效，无需手动恢复
- 并行执行时，8 个任务同时运行可能对 NPU 造成较大负载，可通过 `--max-jobs` 降低
- pytest 默认不执行，需使用 `--skip-pytest=false` 启用
- 退出码综合反映 bench 和 pytest 两个阶段的结果：任一阶段有失败则退出码为 1
- Excel 导出依赖 `openpyxl`，若未安装需先 `pip install openpyxl`
- 启用 pytest（`--skip-pytest=false`）时依赖 `pytest-forked` 和 `pytest-xdist`（已在 requirements.txt 声明，正常安装环境无需额外操作）
- 多轮测试的 Excel 文件会持续追加，不会覆盖已有轮次数据
- 日志与 Excel 默认输出到 `./tmp/` 目录（已被 `.gitignore` 忽略），避免污染项目根目录
