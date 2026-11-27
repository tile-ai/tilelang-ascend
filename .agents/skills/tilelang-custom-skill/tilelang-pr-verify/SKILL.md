---
name: tilelang-pr-verify
description: >-
  根据 PR 链接生成验证报告：checkout PR 修改前（merge-base）和修改后（head）两个版本，
  各编译运行一次全量 examples，对比标注 FIXED/NEW FAIL。复用 run-examples 的执行与 Excel 导出。
  触发：验证 PR、PR 验证报告、PR 对比、before after、merge-base、pr verify、
  验证 pull request、PR 回归测试、PR 影响分析。
---

# TileLang PR Verify

根据 PR 链接，自动 checkout PR **修改前**（merge-base）和**修改后**（head）两个版本，各编译运行一次全量 examples，生成 before/after 对比验证报告（Excel + Markdown），精准标注 PR 修复了哪些（FIXED）、新增了哪些回归（NEW FAIL）。

## ⚠️ 核心规则：强制交互流程

**Agent 触发此 skill 后，必须严格按"⭐ Agent 强制执行流程"中的 3 个步骤执行，需要交互的步骤必须使用 `question` 工具向用户提问，禁止跳过询问直接执行操作。**

---

## 路径与参数说明

- **`<skill-path>`**：本 SKILL.md 所在目录的绝对路径。Agent 调用脚本时，必须将此占位符替换为该目录的实际绝对路径。
- **`<run-examples-path>`**：tilelang-run-examples skill 的 scripts 目录绝对路径，即 `<skill-path>/../tilelang-run-examples/scripts`。本 skill 直接复用其下的 `run_examples.sh`、`export_to_excel.py`，不复制。
- **`--project-root`**：tilelang-ascend 项目根目录（包含 `set_env.sh` 的目录）。**不传时默认为当前工作目录（cwd）**。

## 执行步骤

核心脚本：

1. **`<skill-path>/scripts/verify_pr.sh`** — 主执行脚本，负责 PR 解析、两次 checkout/build/run、状态恢复、Excel 导出
2. **`<skill-path>/scripts/generate_report.py`** — 解析 before/after 日志，生成 Markdown 验证报告

```bash
bash <skill-path>/scripts/verify_pr.sh --pr <url|number> [--backend <auto|ascendc|pto|both|ascendc,pto>] [--project-root <path>] [--skip-aclgraph[=true|false]] [--max-jobs N] [--output-dir <path>]
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pr <url\|number>` | PR 链接或编号（必填）。支持完整 URL（`https://github.com/owner/repo/pull/N`）或纯数字编号（需配合 `--repo`） | 无（必填） |
| `--repo <owner/repo>` | PR 所在仓库（仅当 `--pr` 为纯数字时需要）。如 `tile-ai/tilelang-ascend` | 无 |
| `--backend <...>` | 编译后端类型 | `auto` |
| | 单后端：`auto` / `ascendc` / `pto` | |
| | 多后端：`both`（=ascendc,pto）或逗号分隔 `ascendc,pto`（按顺序运行）。多后端时每个后端产出独立子目录（日志/Excel/报告） | |
| `--project-root <path>` | 项目根目录路径 | cwd |
| `--skip-aclgraph[=true\|false]` | 跳过 aclgraph 脚本（透传给 run_examples.sh） | 跳过（默认 true） |
| `--skip-pytest[=true\|false]` | 跳过 pytest 测试阶段（透传给 run_examples.sh） | 跳过（默认 true） |
| `--max-jobs N` | 最大并行任务数（透传给 run_examples.sh） | 8 |
| `--task-timeout <秒>` | 单任务超时秒数（透传给 run_examples.sh）。`0` 禁用 | 600 |
| `--pytest-timeout <秒>` | pytest 阶段整体超时秒数（透传给 run_examples.sh，与 `--task-timeout` 独立）。pytest 跑数百用例需更大预算。`0` 禁用 | 1800 |
| `--build-timeout <秒>` | 单次编译超时秒数（make/install_ascend.sh）。SIGTERM 后 60s grace 再 SIGKILL。`0` 禁用 | 1800 |
| `--output-dir <path>` | 输出目录路径 | `<cwd>/tmp/pr_verify_<timestamp>_<pr_number>` |

## 工作原理

### 1. PR 解析与 commit 获取

```
PR URL → (owner/repo/number)
       → gh pr view --json baseRefOid,headRefOid,title,url
       → git fetch <remote> pull/<N>/head
       → BEFORE_SHA = git merge-base <baseRefOid> <headRefOid>  (PR 修改前)
       → AFTER_SHA  = <headRefOid>                               (PR 修改后)
```

- **merge-base**：PR 分支与目标分支的共同祖先，只隔离 PR 自身改动的影响
- **remote 映射**：根据 `git remote -v` 自动将 `owner/repo` 映射到本地 remote 名称
- **网络重试**：`gh pr view` / `git fetch` / `git submodule update` 均带超时+重试保护（详见注意事项），避免网络抖动导致脚本 hang 死

### 2. 重编译检测

脚本自动分析 PR diff（`git diff --name-only $BEFORE_SHA $AFTER_SHA`），决定是否需要重新编译：

| PR diff 路径 | 动作 |
|-------------|------|
| `src/**/*.{cc,cpp,cxx,c,h,hpp}` | 重编译：`cd build && make -j$(nproc)` |
| `CMakeLists.txt` / `cmake/**/*.cmake` | 重编译（make 自动触发 cmake reconfigure） |
| `build/config.cmake` | 重编译（先 `cd build && cmake ..`） |
| `3rdparty/**` | `git submodule update --init --recursive` + 重编译 |
| `src/tl_templates/**` | 不重编 .so；清 `~/.tilelang/cache` |
| `tilelang/**/*.py` | 不重编；重新 import 生效 |
| `examples/**` / `testing/**` / `docs/**` | 不重编 |

每次 run_examples.sh 执行前都会清 `~/.tilelang/cache`，确保 kernel 重新生成。

### 3. 状态保护

- 运行前：保存当前分支名 + stash 未提交改动
- 运行中：任何中断（Ctrl-C、kill、错误退出）都通过 `trap` 恢复原分支 + pop stash
- `build/` 目录在 `.gitignore` 中，checkout 不影响它，增量 make 可正常工作

### 4. 报告生成

- **Excel**：before → Round 1，after → Round 2，自动生成"对比分析"Sheet（FIXED / NEW FAIL / 无变化）
- **Markdown**：突出显示新增回归（NEW FAIL）和 PR 修复（FIXED），附 pass rate 变化

## 使用示例

> 以下示例中 `<skill-path>` 需替换为本 skill 所在目录的绝对路径。`--project-root` 省略时默认使用 cwd。

```bash
# 验证 tile-ai 仓库的 PR #123（auto 后端）
bash <skill-path>/scripts/verify_pr.sh --pr https://github.com/tile-ai/tilelang-ascend/pull/123

# 验证 fork 仓库的 PR，使用 pto 后端
bash <skill-path>/scripts/verify_pr.sh --pr https://github.com/erhsh/tilelang-ascend/pull/45 --backend pto

# 双后端验证（ascendc + pto 都跑，各产出独立报告）
bash <skill-path>/scripts/verify_pr.sh --pr https://github.com/tile-ai/tilelang-ascend/pull/123 --backend both

# 仅传 PR 编号（需指定 repo）
bash <skill-path>/scripts/verify_pr.sh --pr 123 --repo tile-ai/tilelang-ascend

# 控制并行度
bash <skill-path>/scripts/verify_pr.sh --pr https://github.com/.../pull/123 --max-jobs 4

# 调整超时（单任务 900s，编译 3600s，pytest 2400s；设为 0 禁用）
bash <skill-path>/scripts/verify_pr.sh --pr https://github.com/.../pull/123 --task-timeout 900 --build-timeout 3600 --pytest-timeout 2400
```

## 输出结构

### 单后端

```
tmp/pr_verify_<timestamp>_<pr_number>/
├── before.log                          # merge-base 版本的完整运行日志
├── after.log                           # head 版本的完整运行日志
├── run_examples_results.xlsx           # Excel 对比（Round 1=before, Round 2=after + 对比分析）
└── pr_verify_report.md                 # Markdown 验证报告摘要
```

### 多后端（`--backend both`）

```
tmp/pr_verify_<timestamp>_<pr_number>/
├── pr_verify_report.md                 # 汇总报告（聚合所有后端对比摘要、总体结论）
├── ascendc/                            # ascendc 后端独立子目录
│   ├── before.log
│   ├── after.log
│   ├── run_examples_results.xlsx       # Round 1=before, Round 2=after（ascendc）
│   └── pr_verify_report.md             # ascendc 验证报告
└── pto/                                # pto 后端独立子目录
    ├── before.log
    ├── after.log
    ├── run_examples_results.xlsx       # Round 1=before, Round 2=after（pto）
    └── pr_verify_report.md             # pto 验证报告
```

> 多后端时根目录额外生成汇总报告 `pr_verify_report.md`，聚合各后端的通过率/FIXED/NEW FAIL/持续失败，附总体结论。各后端子目录仍保持独立 xlsx 和报告。

## ⭐ Agent 强制执行流程（不可跳过任何步骤）

**以下 3 个步骤必须严格按顺序执行，需要交互的步骤必须与用户确认后才进入下一步。禁止跳过任何询问步骤直接执行。**

### 步骤 1：确认 PR 链接与运行配置

> 本步骤的提问次数不设上限。自定义路径含 1 次「默认/自定义」总选 + 最多 4 次逐项问（1.2a/1.2b/1.2c/1.2d），共最多 5 次 `question` 调用。不得以"轮次限制"为由合并或裁剪 1.2 的问题。

若用户消息中**没有** PR 链接，必须先用 `question` 工具询问用户提供 PR 链接，**禁止猜测或编造 URL**。获取 PR 链接后进入 Step 1.1。

#### Step 1.1（必须使用 question 工具）

展示默认配置摘要，询问用户使用默认配置还是自定义配置：

```
PR 验证配置：
  • PR 链接：<从消息中提取的 URL>
  • 后端类型：auto
  • 跳过 aclgraph：是
  • 运行 pytest：否
  • 并发数：8
```

选项：
- **「使用默认配置」** — 以默认参数直接进入步骤 2
- **「自定义配置」** — 进入 Step 1.2 逐个询问

→ 用户选默认配置 → 直接跳到步骤 2，不再询问其他问题
→ 用户选自定义 → 继续 Step 1.2

> 「ascendc+pto 双后端」会依次运行两个后端，各产出独立子目录（日志/Excel/报告），步骤 3 展示两份报告摘要。

#### Step 1.2（仅在用户选择"自定义配置"时执行）

以下四个问题必须**逐个询问**，每个问题**单独一次 `question` 工具调用**（`questions` 数组中只能有一个 question），**等用户回答当前问题后才问下一个**。

**严禁将多个问题合并在同一次 `question` 调用中。**

> ⚠️ **硬约束：无论用户选择自定义的原因是什么（哪怕只想改后端），1.2a / 1.2b / 1.2c / 1.2d 四项必须全部逐个问完，不得以"其余保持默认"为由跳过任何一项。** 1.2a 问完用户的回答不决定是否继续问 1.2b/1.2c/1.2d——四项必问。

- **1.2a** 单独询问后端类型（auto、ascendc、pto 或 ascendc+pto 双后端，默认 auto）
  - 选「ascendc + pto」时，两个后端依次运行，各自产出独立子目录（日志/Excel/报告）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2b** 单独询问是否跳过 aclgraph（默认跳过；可选择 `--skip-aclgraph=false` 运行）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2c** 单独询问是否运行 pytest（默认跳过；可选择 `--skip-pytest=false` 启用）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题
- **1.2d** 单独询问并发数 `--max-jobs`（默认 8；NPU 负载高时可降低，如 4 或 2）
  ⚠️ 必须单独一次 `question` 调用，只包含这一个问题

> ⛔ **进入步骤 2 前的自检门：Agent 必须自检 1.2a / 1.2b / 1.2c / 1.2d 是否都已得到用户明确回答。若有任一未问，必须回退补问，禁止直接进入步骤 2。**

### 步骤 2：运行验证

运行 `<skill-path>/scripts/verify_pr.sh`，输出 tee 到 `./tmp/` 目录下的日志文件。此步骤耗时较长（两次全量编译运行），Agent 需耐心等待完成。先创建 `./tmp/` 目录再执行。

```bash
mkdir -p ./tmp && bash <skill-path>/scripts/verify_pr.sh --pr <url> --backend <auto|ascendc|pto|both|ascendc,pto> [--project-root <cwd>] [--skip-aclgraph[=true|false]] [--skip-pytest[=true|false]] --max-jobs <N> 2>&1 | tee ./tmp/pr_verify_console.log
```

### 步骤 3：展示报告摘要

脚本完成后，读取生成的 `./tmp/pr_verify_<timestamp>_<pr_number>/pr_verify_report.md`，向用户展示验证摘要：

- **FIXED**：PR 修复的测试数量
- **NEW FAIL**：PR 新增的回归数量
- **Pass rate 变化**：before → after 的通过率变化
- **报告路径**：Excel 和 Markdown 文件的完整路径

> 多后端（`--backend both`）时，根目录会生成汇总报告 `pr_verify_report.md`（聚合各后端对比摘要和总体结论）。Agent 优先读取汇总报告展示整体结论，再附上各后端子目录的独立报告路径（`ascendc/pr_verify_report.md`、`pto/pr_verify_report.md`）。

### 流程检查清单

| 步骤 | 检查项 | 是否与用户交互 |
|------|--------|--------------|
| 1（PR链接） | PR 链接是否已确认（缺失时是否用 question 问了用户）？ | ✅ 必须 |
| 1.1 | 是否用 question 工具询问了默认/自定义配置？ | ✅ 必须 |
| 1.2a | 自定义模式下，是否单独一次 question 调用问了后端类型？ | ✅ 必须 |
| 1.2b | 自定义模式下，是否单独一次 question 调用问了是否跳过 aclgraph？ | ✅ 必须 |
| 1.2c | 自定义模式下，是否单独一次 question 调用问了是否运行 pytest？ | ✅ 必须 |
| 1.2d | 自定义模式下，是否单独一次 question 调用问了并发数（--max-jobs）？ | ✅ 必须 |
| 1.2 自检 | 进入步骤 2 前，1.2a/1.2b/1.2c/1.2d 是否都已得到用户明确回答？ | ✅ 必须 |
| 2 | 是否将输出 tee 到日志文件？ | ❌ 不需要 |
| 3 | 是否展示了 FIXED/NEW FAIL 摘要和报告路径？ | ❌ 不需要 |

**如果任何"必须"交互的步骤被跳过，视为流程违规，必须回退补执行。**

## 注意事项

- **超时保护**：单任务默认 600s（`--task-timeout`，透传 run_examples.sh）、单次编译默认 1800s（`--build-timeout`）、pytest 阶段默认 1800s（`--pytest-timeout`，与单任务超时解耦），超时则 SIGTERM（task 30s / build 60s grace 后 SIGKILL）并标记为失败，避免 hang 导致整轮卡死。三者设为 `0` 可禁用。pytest 用独立超时是因为它跑数百用例，复用 600s 单任务超时会在打印汇总行前被整体杀掉，导致 `Pytest: Passed: 0 | Failed: 0` 的错误统计
- 本 skill 会切换 git 分支/commit，运行结束后自动恢复。**请确保运行前无重要未提交改动**（脚本会 stash 保护，但建议先提交或保存）
- 两次全量运行耗时较长（每次约 10-30 分钟，取决于 PR 是否需要重编译），请耐心等待
- 环境变量 `TILELANG_JIT_TARGET` 由 `run_examples.sh` 内部处理，本 skill 透传 `--backend` 参数即可
- 若 PR 来自 fork 仓库，脚本通过 `git fetch <base-remote> pull/<N>/head` 获取 head commit（GitHub 在 base repo 暴露此 ref）
- Excel 导出依赖 `openpyxl`，若未安装需先 `pip install openpyxl`
- 若增量编译失败，脚本自动回退到 `install_ascend.sh --enable-incremental`；若仍失败则报错并恢复状态
- **网络重试机制**：所有涉及网络的 git/gh 操作均带超时+重试保护，避免网络抖动或 GitHub 连接慢导致脚本 hang 死。重试策略为指数退避（2→4→8→16→30s，cap 30s），具体参数如下：

  | 操作 | stall 检测 | 硬超时兜底 | 重试次数 | 说明 |
  |------|-----------|-----------|---------|------|
  | `gh pr view` | — | 30s | 5 | GitHub API 取 PR 元数据 |
  | `git fetch`（PR head / base 分支） | lowSpeedTime=10s | 120s | 5 | 传输速度 <1KB/s 持续 10s 即中止重试 |
  | `git fetch`（兜底遍历所有 remote） | lowSpeedTime=10s | 60s | 1 | 每个 remote 只试一次，不重试 |
  | `git submodule update` | lowSpeedTime=15s | 600s | 3 | 子模块数据量大，stall 阈值更宽松 |

  - stall 检测通过 `git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=<N>` 临时注入，不修改全局/仓库 config
  - 硬超时通过 `timeout` 命令实现，SIGTERM 后 5s grace 再 SIGKILL
- 所有输出（日志、Excel、Markdown 报告）默认落到 `./tmp/pr_verify_<timestamp>_<pr_number>/` 目录（已被 `.gitignore` 忽略），避免污染项目根目录
