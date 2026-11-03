# 一键复测 Ascend examples

在仓库根目录运行：

```bash
bash scripts/run_ascend_examples.sh
```

默认复跑 [固定的原始 215 项清单](../../../../ci/ascend_sync_examples.json)，
包括原有的 54 项显式跳过及其原因，保持与之前复测相同的统计口径。
清单中的入口、参数、pytest marker 和单项超时原样保留；新增 example 不会自动纳入此清单。
脚本不修改 examples。

运行前需已配置 CANN、驱动、torch_npu 和本仓库的 CMake `build/`。
沿用当前终端中的 CANN 环境（例如已执行安装目录中的 `set_env.sh`）。
入口自动加载仓库的 `set_env.sh`，优先使用 `.venv/bin/python`，否则使用 `python3`；
也可通过 `TILELANG_TEST_PYTHON=/absolute/path/to/python` 指定解释器。
脚本不会安装依赖或更新子模块。

默认先增量构建当前源码，再检查实际加载的同步引擎 ABI、Z3 和 NPU。
使用设备 0、1：每张卡串行，两张卡之间并行，避免同卡样例互相干扰。
强制使用 upstream 同步、普通同步和向量标量同步，关闭 Bisheng 自动同步及 TileLang 缓存。
只有一张卡时运行：

```bash
bash scripts/run_ascend_examples.sh --devices 0
```

常用选项：

```bash
# 仅查看清单和设备映射，不构建、不导入 TileLang、不使用 NPU
bash scripts/run_ascend_examples.sh --dry-run

# 已确认当前源码编译完成时，跳过增量构建
bash scripts/run_ascend_examples.sh --skip-build

# 只复跑指定的原始入口
bash scripts/run_ascend_examples.sh --only examples/deepseek_v4/test_hc_split_sinkhorn.py

# 仅复跑某轮失败、超时、未执行或中断未完成的入口
bash scripts/run_ascend_examples.sh --failed-from build/ascend-examples/某轮目录/results.json
```

`--timeout 120` 设置没有单项超时配置的入口的默认超时秒数；
清单里的 `timeout_seconds` 优先。`--jobs 8` 设置增量构建并行数。
`--devices 2,3` 可指定其他物理卡；清单中的原设备编号映射到所选设备，并记录实际编号。
`--help` 查看全部选项。

每轮创建独立目录 `build/ascend-examples/<时间戳>-<进程号>/`，最后打印报告路径：

- `summary.md`：状态计数、逐项诊断、编译/调用/断言数和日志链接。
- `results.json`：本轮完整机器可读结果，可作为 `--failed-from` 输入。
- `logs/`、`kernels/`：各入口日志、元数据与生成的内核源码。
- `build.log`、`preflight.log`：构建日志（执行构建时）和环境检查日志。
- `entries.json`、`run_context.json`、`source_snapshot.json`：本轮清单、环境和源码快照。

可用 `--output /path/to/new-directory` 指定新目录；已存在的目录会被拒绝，防止覆盖历史证据。
失败项复跑仅统计本轮所选入口，不混入旧的通过结果。

PASS 要求原入口成功完成，并具有本轮编译、实际调用、配置和 ABI 的匹配证据。
捕获到的断言失败、编译失败、调用失败或内嵌 pytest 失败即使被 example 捕获也会判失败。
没有记录到 TileLang 调用会单列为 `NO_EXECUTION`；没有捕获到数值断言的 PASS
会保留断言数 0，不额外宣称精度验证。

运行期间检查 `src/`、`tilelang/`、`examples/`、清单、运行脚本、构建配置、
子模块状态和实际加载的 TileLang 原生库是否变化；变化后本轮结果标记无效。
需要修改源码时先按 Ctrl+C 停止，脚本会清理子进程组并保留已完成项及未完成项。
被中断或环境检查失败的运行不能作为完整复测结论。

退出码：`0` 表示本轮有效且没有失败/超时/未执行项；`1` 表示存在这些项目；
`2` 表示构建、环境或完整性检查错误；`130` 表示用户中断。
