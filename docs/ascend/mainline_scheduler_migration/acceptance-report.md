# S1–S8 A3 整体集成验收

## 1. 范围与状态

用户在 S8 交付后选择整体集成验收。本轮对象为 S8 提交
`4c94fa120920a56e3a3718a7f642da0077b2680e`，工作分支仍为
`codex/ascend-a2a3-scheduler-migration`，remote 仍为 `erhsh`。
固定迁移基线 `43b0b69720f479894a13c02746599eccee5c5769` 和主仓参考
`1c0f402e166a7d0a055c991d2f82d2175a7917b7` 不变。

最终状态：**S1–S8 已交付合同的 A3 集成正确性验收 PASS**。三项补充主机入口、
七个旧阶段设备入口及原始证据末审均已完整通过，性能回归如第 5 节保留。只验收
A3，不新增 S9 功能、改变默认策略、扩大 auto 支持范围或开展性能优化。

本报告区分本轮新增执行、同源码/库的已有执行证据及历史阶段证据。S1–S7 原始
设备结果的 native 构建各不相同，不能据此宣称最终 S8 构建全量回归通过。
同源的 S8 原生、CLI 和设备结果在核验原始报告、输入和库身份后复用，不计为本轮
新执行，也不重写原 JSON 中的 commit/dirty 字段。

## 2. 冻结身份

本轮复用 `/tmp/ascend-migration-s8/native-build.json`，不重新构建或修改生产源码。

| 项目 | 身份 |
| --- | --- |
| 174 个 native 输入 SHA256 | `257659998dc6656fe5456e2c4207e4eb4da00465284f7879fa723d49032a2dec` |
| libtilelang_module.so SHA256 | `4771eeefbf6ff07edc1cd17d94f6ea834e3e553f94d6162455f56a4d1de84d98` |
| libtvm.so SHA256 | `79912ad742112ced14bbb8e64e623e108a70c3a93436cf3ab410e138426c3910` |
| libtvm_runtime.so SHA256 | `a98401d89eca389dc9ee8b45f372c93bd5a99ac7b52d5d580fbe032d8d9d11cd` |
| TVM 固定版本 | `c2921fdaf795b1103d21abc962e83a209c7258d7`，原两项仓库补丁 |
| S8 最终 237 个 Python 输入 | `8657121883e9f25bf2398df5e41a31e821a6a2fefadf9adab949a9fcfe603f76` |
| 本轮 537 个 Python/helper/配置输入 | `dbe0336dd9b451204784790cb8e425d4e8fd9e34d3e56258832b84a3c392aa87` |

本轮额外冻结全部项目 Python、验证脚本、`testing/`、`examples/`、根目录 Python
及 `set_env.sh`、`pyproject.toml` 等存在的配置，覆盖旧设备入口实际导入的 language
测试 helper。TVM 的 23940 个输入、补丁、CMake、构建日志及实际三库仍由原 manifest
独立核验。每个新增设备入口执行前后均核验这两类身份。
537 个输入中，533 个跟踪文件与验收提交一致；另 4 个 ignored/local examples
文件只按本机实际字节冻结，不把它们表述为 Git 提交内容，也不纳入本轮提交。

S8 部分原始结果使用此前 Python 冻结 `9c5ca67d…`。其后仅
`test_s8_sync_contract.py` 排版变化，AST 相同，其余 236 个文件字节未变；
`python-format-provenance-bridge.json` 保留完整证据，S8 原生 74 项已在最终冻结后
重跑。本轮不将这一输入桥改写为“原报告使用最终冻结”。
旧 237 清单未包含 language 测试 helper；本轮 537 清单覆盖它们，S1 已在该冻结下
实际重验原 Expert GEMM C128、L1 尾块等 helper，不倒推旧清单包含这些文件。

设备为逻辑 0、实际 `Ascend910_9362`；Python 3.12.9、torch 2.7.1+cpu、
torch_npu 2.7.1.post4、CANN 9.1.0、Z3 4.15.4.0。实际 Bisheng SHA256 为
`7ffc97205a0582ce00a9ec163530696bf3971b24669c43c1c02cc9c7b56f541f`。

## 3. 同源已有执行证据

以下为 S8 最终 native 构建的已有结果，原始文件均位于
`/tmp/ascend-migration-s8/`；本轮另核验其原始报告、计数、产物和输入关系。

| 范围 | 已有实际结果 | 原始报告 |
| --- | --- | --- |
| S1 legacy native | 104 PASS | `compat/s1_check_legacy_lowering.json` |
| S2 Task IR/FFI | 170 PASS | `compat/s2_check_task_ir.json` |
| S3 loop/domain | 129/29 PASS | `compat/s3_check_loop_ir.json`、`compat/s3_check_persistent_domain.json` |
| S4 pipeline/sync | 45/50 PASS | `compat/s4_check_pipeline_ir.json`、`compat/s4_check_sync_coverage.json` |
| S5 versions/成本/校准等合同 | 235 PASS | `compat/s5_check_versions_ir.json` |
| S6 owner/lifecycle | 61/49 PASS | `compat/s6_check_owner_ir.json`、`compat/s6_check_lifecycle_guards.json` |
| S7 event/handoff | 70/60 PASS | `compat/s7_check_event_budget.json`、`compat/s7_check_handoff_ir.json` |
| S8 integration/cache/fault | 74 PASS | `integration-ir.json` |
| S1–S8 CLI/phase/计时政策 | 283 PASS | `host-cli.json` |
| S8 A3 correctness | 14 worker、1400 调用及逐轮 golden、600 轮严格配对 | `workloads-npu.json` |
| S8 A3 benchmark | 6 worker、480 调用及 golden、120 samples、240 条配对 | `benchmark-npu.json` |

这些入口均为 0 fail/skip，测试节点可能重叠，不相加当作唯一测试总数。
S5 的 235 项包含成本、校准、导出、缓存和 runner 合同；CLI 283 自身不代表全部
纯主机验证。本轮补齐的独立纯主机入口列于下一节。

独立审计 `reused-evidence-audit.json` 已实际 PASS、零缺口，SHA256 为
`5060c2224332cafa35079b34386110a9597dc1b11df45f502dd12c34ef702d55`。
其核对原索引 10 份报告、11 个 compat 原始 JSON/JUnit/必跑节点、实际冻结源码、
三库、格式桥及 201 个原始证据/产物文件的哈希，包含 20 组真实 cpp/kernel.so。
S8 设备原始配对与 benchmark 中位数/区间亦独立复算一致；本次审计没有重新执行
pytest、构建或设备调用。该报告位于本轮验收临时证据目录。

## 4. 本轮新增执行

证据写入 `/tmp/ascend-migration-acceptance/`，生产代码和旧原始报告保持不变。
本轮设备报告、完整命令、日志和前后身份/设备快照位于其子目录
`device-run-20261029T093557Z/`，队列状态记录为 `queue-state.json`；
`active-device-run.txt` 保存该次独立目录的绝对路径。

| 纯主机入口 | 实际结果 | 报告 |
| --- | --- | --- |
| s1_check_baseline | 18 PASS，0 fail/skip | `s1-baseline.json` |
| s2_check_solver | solver/oracle 105 PASS，0 fail/skip | `s2-solver.json` |
| s2_check_capabilities | 42 PASS，0 fail/skip | `s2-capabilities.json` |

设备矩阵使用原公开入口和原断言，A3/device0/seed0/repeat100/timeout300。所有设备
worker 独立进程及缓存，入口严格串行；首失败或超时即停止队列。本轮七个入口
均实际完整 PASS、0 fail/skip/未运行，原始结果如下。

| 设备入口 | 实际数值 worker/编译 | 实际逐轮调用/golden | 状态 |
| --- | --- | --- | --- |
| S1 baseline | 14 | 1400 | PASS，14 编译/1400 call/golden |
| S3 loop | 9 | 1000，符号 kernel 两个绑定 | PASS，9 编译/1000 call/golden |
| S4 vector | 8 | 800 | PASS，8 编译/800 call/golden |
| S5 pipeline | 18 | 1800 | PASS；另 4 个 msprof worker 通过 |
| S5 persistent | 8 | 800 | PASS；另 4 个 msprof worker 通过 |
| S6 reuse | 20 | 2000 | PASS，20 编译/2000 call/golden |
| S7 sync stress | 20 | 3200，四个双 stream worker 各 400 | PASS，20 编译/3200 call/golden |

S2 无新增设备执行路径。上述矩阵覆盖 legacy、constraint、A3 新同步，结合第 3 节
同源 auto 结果构成三路集成验证。S5 两父入口另已实际完成四组 msprof 性能配对，
8 个性能 worker、600 次调用、480 个 measured samples、24 次组后 golden；
它们与数值矩阵逐轮 golden 分列，不能宣称 600 次调用均有独立 golden。
本轮新增数值执行共 97 个 worker、97 次真实 Bisheng 编译、11000 次
attempted/completed/golden；另有上述 8 次性能编译/600 次调用/24 次轮末检查。
各入口间有同类回归，数字表示真实执行次数，不是去重 kernel 数。精度判定沿用
各用例原 dtype 容差，SHA 是额外复核证据，不代表统一要求逐位相同。
队列最终 `not_run=[]`、退出码 0；运行后 537 输入和 native manifest 核验通过，
全部测试父/worker 退出，最终 NPU 快照明确为 `No running processes found`。

S6 独立原始审计通过：2000 条逐轮记录、1000 轮跨策略记录一致，输入逐轮刷新；
1800 条输出 SHA 等于 golden，`readonly_f16` 两策略的 200 条按原
`rtol/atol=1e-2` 通过，跨策略输出 SHA 仍严格一致，不宣称全部结果逐位等于
golden。实际 allocation/remap、10 份 lifetime JSON 和混合 dtype owner 亦已核验：
128 字节源容量、实际 3×128=384 字节分配，f16/f32 指针共享源 owner。
313600 个源定义 canary 元素由原整 tensor 含 NaN golden 的比较覆盖，不补造
原报告未保存的 raw canary 值。审计报告为 `raw-device-audit-s6_run_reuse_npu.json`，
SHA256 为 `d05220e07698d2c56634dda3e8072cd67b07b6b204ec6113910ae48f36f5fc01`。

S7 独立原始审计通过：3200 条逐轮记录、1600 轮跨策略配对、400 个双 stream
窗口均核验 wait→enqueue→done/query pending→release→sync 的实际顺序。
3200 个输出 SHA、1200 个 workspace SHA 均等于 golden；源整 tensor/NaN 断言
覆盖 44902400 个 canary 元素，未补造 raw 值。并发结论只限启动门控制的同时
pending，不宣称自然窗口或硬件执行时间重叠。112 个原始报告、日志、实际编译
输入/库及新同步 contract 文件哈希均通过，实际 N1/N2/N3、输出域和固定 GM
握手参数亦复核通过。`raw-device-audit-s7_run_sync_stress_npu.json` 的 SHA256 为
`5ea5ef812b70f0839e2628aa2a045a865c9647f2e003fdef6edfac717098a5c0`。

独立的 `all-new-entry-identity-gates-audit.json` 另复核当前 537 个输入，以及七个
入口 28 份 Python/native 前后 gate 日志，全部通过；SHA256 为
`70463e9cae7dcd5b21493b2c9483c2487a862f1396434075f3baa5c7d947a4c1`。
设备执行 owner 的第二份只读审计 `device-validation-audit.json` 位于设备 run 子目录，
SHA256 为 `f9d5765e1678013c717eea9c657fb83d5c1fb2ba93427e64991ab534de159f97`。
这些审计不增加编译或调用次数。

最终机器可读索引为验收目录下的 `acceptance-summary.json`，SHA256 为
`3e25f7c1e854f5a8237796c2d370262503147ea9fee9c8b95c61832627ca8c4c`。
该索引分别关联本轮三主机/七设备入口、逐入口独立审计、S5 原始性能、S8 同源复用
及实际 native/输入冻结身份；原始 commit/dirty 和全部报告字节保持不变。

## 5. 模型、性能和交付边界

S5/6/7 Vector constraint 使用原完整
`/tmp/ascend-migration-s5/a3-effective-cost-model.json`；本轮已只读核验 schema、
内容 digest 和显式预期身份。匹配项为 A3/实际 SoC/CANN/Bisheng SHA/有序 flags
`[-O2]`，不包含 native module SHA。模型校准来源仍为历史构建，不称为当前库
重新校准；每个设备 worker 继续执行原严格 identity 门禁。

S7 GM 保持 legacy planner 与完整双向 READY/FREE 协议，不应用 Vector 成本模型。
S8 auto 拒绝 S5 模型，使用 `abstract_issue_step_v1`，实际 flags 为
`[-O2, --cce-auto-sync=off, -O3]`。不同策略只在各自原合同内配对，不强行统一模型。

S5 的 msprof BasicInfo Task Duration 与 S8 的 NPU Event span 分列；后者包含提交
间隙，不当作纯 kernel 指令延迟。S8 已有观测中，mixed_reduce 的 Event 中位数
增加 2.48%，auto 冷编译 wall 为 legacy 的约 1.17–10.68 倍，详情见
[迁移计划第 14.3 节](migration-plan.md#143-a3-矩阵与前置-pilot)。本轮不预设提速
门槛，不将历史性能值冒充本轮结果。

本轮 S5 实测如下，单位为微秒，括号为中位数的 95% bootstrap 区间；每 worker
60 个 measured samples，之前另有 15 次 warmup。正数表示 constraint 耗时增加。

| 用例 | legacy 中位数（95%区间） | constraint 中位数（95%区间） | 变化 |
| --- | --- | --- | --- |
| steady2 | 5.451 (5.181–5.881) | 5.941 (5.171–6.461) | +8.99% |
| tail_f16 | 4.411 (4.121–4.781) | 4.651 (4.161–5.481) | +5.44% |
| group_tail | 6.281 (6.161–6.461) | 7.251 (7.101–7.381) | +15.44% |
| partial_f16 | 4.721 (4.661–4.821) | 4.751 (4.681–4.991) | +0.64% |

四组本次均观察到耗时增加，作为已知性能限制保留。独立审计已逐一核验 600 个
BasicInfo CSV、8×75 条 journal、实际时钟/flags、编译输入/kernel 库哈希，复算
median/CI 与原始报告一致；每 worker 只有 3 次轮末 golden，不补造逐调用检查。
独立汇总为 `s5-current-performance-independent-summary.json`，SHA256 为
`c99115c17bafbe298b118248b6ce3fd83e6cc41bebc31f344f17f5f25b60b02f`。

只提交本报告和迁移计划进展；日志、JSON、profile、临时驱动和二进制保留在临时
证据目录。最终 PASS 须同时满足必跑入口完整、零失败/跳过/未运行、真实编译与
调用/golden 计数、策略配对、输入/库身份和已知限制均可复核。本轮这些条件均已
实际满足，未发现须修改生产源码的正确性问题；已知性能限制如上。推送后停止
等待用户审核。

## 6. 复核方式

本轮沿用各阶段公开入口；完整环境和独立入口命令见
[迁移计划](migration-plan.md)对应阶段。统一原生 manifest 可通过环境变量传入：

```bash
source set_env.sh
(
set -e
export TL_ASCEND_MIGRATION_BUILD_MANIFEST=/tmp/ascend-migration-s8/native-build.json
migration_python=/tmp/ascend-migration-s2/venv/bin/python
acceptance_dir=/tmp/ascend-migration-acceptance/review
mkdir -p "$acceptance_dir"

"$migration_python" scripts/ascend_migration/s1_check_baseline.py --output "$acceptance_dir/s1-baseline.json"
"$migration_python" scripts/ascend_migration/s2_check_solver.py --timeout 300 --output "$acceptance_dir/s2-solver.json"
"$migration_python" scripts/ascend_migration/s2_check_capabilities.py --platform a3 --timeout 300 --output "$acceptance_dir/s2-capabilities.json"

for entry in s1_run_baseline_npu s3_run_loop_npu s4_run_vector_npu s5_run_pipeline_npu s5_run_persistent_pipeline_npu s6_run_reuse_npu s7_run_sync_stress_npu; do
    cost_args=()
    case "$entry" in
        s5_*|s6_*|s7_*) cost_args=(--cost-profile /tmp/ascend-migration-s5/a3-effective-cost-model.json) ;;
    esac
    "$migration_python" "scripts/ascend_migration/$entry.py" --platform a3 --device 0 --seed 0 --repeat 100 --timeout 300 "${cost_args[@]}" --output "$acceptance_dir/$entry.json"
done
)
```

上述路径为本机已有环境和证据，不是仓库自带依赖；manifest 须与被验收源码及实际
库匹配，重建方法沿用技术参考第 11.5 节。局部子 shell 保留首个失败的非零退出码，
不改变调用者全局 shell 选项；任一主机或设备入口失败均停止。先诊断并确认上下文
清理，再决定后续执行，不能绕过失败继续并宣称整体 PASS。
