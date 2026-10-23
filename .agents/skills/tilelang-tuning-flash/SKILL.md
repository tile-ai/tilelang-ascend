---
name: tilelang-tuning-flash
description: 执行 Ascend A2/A3 TileLang 算子的完整 Flash 调优会话，管理隔离候选、性能测量、搜索覆盖和最终 pytest 验收。用于已有正确实现的性能优化或 Stage 3 续跑；Flash 指调优组织方式，适用于不同算子。
---

# A2/A3 Flash 调优工作流

由当前执行者完成整次调优会话。Stage 3 中执行者是 `tilelang-op-perf-tuner`；直接调用
本 Skill 时是当前 Agent。按名称加载下列 Skill，不再分发性能分析或实施子 Agent：

- `tilelang-op-profiling`：以 `execution_mode: collect_only` 采集、映射、摘要和归档。
- `tilelang-perf-optimization`：消费证据，提出候选及逐 case 搜索义务。
- `tilelang-performance-best-practices`：查询 A2/A3 API、lowering、实现约束和知识成熟度。
- `npu-arch`：结合现有设备证据确认硬件规格；复用
  `../tilelang-custom-skill/tilelang-env-check/scripts/get_npu_arch.py --json`。

本 Skill 唯一管理本次候选循环、源码修改、精度节奏、晋升、回滚与归档；不读写
`.orchestrator_state.json`。主控拥有全局阶段状态。Flash 不承诺更短耗时，也不要求
必须产生收益；无默认轮数、时长或连续无收益次数上限。用户显式预算可以停止执行，
但预算耗尽不能替代搜索收敛证据。

## 1. 冻结输入与建立隔离目录

开始前需要：工程根目录、源码和实际入口、完整正确性测试命令与工作目录、编程模式、
实际 target/设备，以及用户选定的全部性能 case（稳定 ID、shape、dtype、属性、参数）。
可以使用用户已确认的 DESIGN.md 或 case 文件，不重复询问已有信息；缺失时返回具体
缺项，不能从 pytest 静默挑选或补充性能 case。数值性能目标可选；没有目标时记录
`performance_target: null`，不得自造阈值。沿用已确定的 Developer / Expert / 混合模式。

target 沿用调用方或真实入口；未显式指定时遵循 profiling Skill 的 `auto → AscendC`
规则，新 wrapper 显式使用 `ascendc`。记录实际解析的 target，不引入 A5 的固定
`TILELANG_DEFAULT_TARGET=pto/ascend` 协议。不同 target、设备或计时方式的数据不能混比。

新会话在 `examples/{op}/perf_tuning/flash_<run_id>/` 创建不重名的 `run_dir`；已有算子
位于其他目录时使用该实际算子目录下的 `perf_tuning/`，或调用方指定的绝对路径。
输入未齐全前不创建目录、编译、采集或修改代码。

```text
<run_dir>/
├── manifest.json                         # 冻结输入、版本、命令、文件清单和 SHA256
├── baseline/                             # B0：原始实现，创建后只读
├── working/                              # B*：当前已验证的最好版本
├── candidates/Ci/                        # 每个候选从当前 B* 隔离创建
├── profiling/<measurement_id>/<case_id>/  # raw/ 与 summary/，每次采集使用新 ID
├── optimization-search-coverage.json      # 唯一搜索账本，schema 3
├── timing.json                           # 基线/各候选/验收的墙钟起止与耗时
├── final_optimized/                      # 仅放最终验收通过的交付文件
└── flash-report.md                       # 双状态、证据、未决项、交付位置
```

复制最小可运行文件集：源码、现有测试、直接/间接仓内依赖、必要 `__init__.py`、
`conftest.py` 和测试配置。保留相对路径；排除输出目录，避免递归复制。manifest 列出
每个文件的原位置、隔离位置、SHA256，以及源码、测试和 case 清单身份。后续候选与
交付复用该清单；只在运行证明缺少依赖时补充并记录原因。不得通过修改 golden、
容差、测试 case 或公开接口制造通过。原工程保持不变，交付指向 `final_optimized/`。

从隔离目录运行时校验实际导入的 `module.__file__`、源码 SHA 与调用链；配置
PYTHONPATH/工作目录时不能让候选入口意外导入原工程或已安装的旧算子。JIT 缓存或
AOT 二进制要重新核对构建输入和当前版本。仓库 pytest 默认搜索路径不能代替显式
指定目标测试：使用已确认的完整测试路径、配置和参数，不能空跑整个 pytest 默认入口。

manifest 同时记录编译器/TileLang/CANN 版本、硬件探测、target、设备、精度基线证据、
warmup/repeat/launch、单 kernel 或整链计时口径，以及用户预算和退化容忍约定。
物理容量与编译器有效容量分别记录；使用 A3 实际 lowering 依据，不抄 A5 容量常数。

## 2. 基线与实现约束

`B0` 不可变；`B*` 初始为 `B0`；`Ci` 只从当时的 `B*` 创建。既有完整精度证据只有在
源码、测试、case、模式和运行环境身份匹配时才能复用；不匹配或没有可信证据时先补跑
完整测试。仍须确认本地基线能编译、正确导入并实际运行，不能把“默认已正确”当证据。

保留既有实现的公开接口和 kernel 链。记录 primfunc/dispatch/实际 runtime kernel 名称
及 launch 数；既有多 kernel 算子按 profiling 的
[整链约定](../tilelang-op-profiling/references/kernel-chain-measurement.md)计时。
不把 A5 临时“单 kernel”限制施加到既有 A3 算子；新增拆分或特化必须有接口与当前模式
依据，并比较完整算子链。禁止只对精确性能 case 增加投机分支，或只计加速的局部 kernel。

先阅读与当前瓶颈相关的最佳实践、现有同类实现和历史候选。历史证据仅在接口、case、
环境与物理结构兼容时作为候选依据，不能直接晋升。记录复用来源和不兼容项；名称相同
不代表 buffer、搬运、同步、tile/core 或 lowering 相同。

对 B0 全部目标 case 做 profiling，重复采集估计噪声，建立逐 case latency、资源账本、
任务粒度/并行度、语义重复工作量和数据流模型。下界缺证据时填 null 并列出补采项。
pipe ratio 只用于定位线索，瓶颈归因需要源码工作量、生成代码或 timeline 的交叉证据。

## 3. profiling 的调用契约

每次调用显式传 `execution_mode: collect_only`、`variant_output_dir: <run_dir>`、
唯一 `measurement_id`，以及当前 B0/B*/Ci 的源码位置、源码 SHA256、测试 SHA256、
case 清单 SHA256、case ID/完整参数、target、设备、命令/工作目录、warmup/launch、
计时统计口径和已通过的本次精度证据。

显式传 `raw_output_dir: <run_dir>/profiling/<measurement_id>/<case_id>/raw` 和
`archive_output_dir: <run_dir>/profiling/<measurement_id>/<case_id>/summary`。
复测创建新 measurement_id，不覆盖已有目录；返回原始数据、metadata 和 summary 路径。
先复用可靠的已有入口/AOT/AclGraph，必要时才创建最小 wrapper。每个版本重新核对实际
导入与 kernel/case/launch 映射，不能复用仍指向旧源码的入口。

合并读取返回的 `metadata.json` 与 `collection_context.json`，核对当前 manifest/候选；
测试 SHA 和整链映射可位于后者，不能因合法 sidecar 而误报缺字段。任何必需身份为空、不匹配或映射
不唯一时，该测量无效，不能晋升或用于目标判定。`perf_summary.py` 的可选参数不代表
本工作流可省略身份。多 kernel/fallback 的整链摘要仍由调用方核验。

首选 msprof op；仅在最新入口直调正常、故障确属插桩/导出时按 profiling Skill fallback。
一方 fallback 后对另一方也按相同方式重采，保留真实来源。pytest benchmark 仅用于
筛选，host/pytest 总耗时不能替代设备 kernel/整链时间。设备选择与可见设备映射遵循
profiling Skill，完整 pytest 仍保留已有设备参数。

## 4. 搜索记录与单假设迭代

首次改源码前阅读[搜索覆盖门禁](references/optimization_search_coverage_gate.md)，按
[起始样例](assets/optimization_search_coverage.example.json)建立 schema 3 账本。
样例是结构示意，必须替换为当前算子的事实。令 `FLASH_SKILL_DIR` 为本 SKILL.md
所在目录，执行：

```bash
python3 "$FLASH_SKILL_DIR/scripts/validate_optimization_search_coverage.py" \
  "$RUN_DIR/optimization-search-coverage.json"
```

只有普通校验通过才开始新候选。它检查记录结构和门禁一致性，不验证文件引用内容的
真实性；执行者仍须打开测试日志、metadata、源码和生成代码核实证据。

将基线/当前最佳证据及实现约束交给 optimization Skill，取得最多三个已准入方案和
全部结构义务；队列大小不限制义务数。分析 Skill 返回缺失证据时先补采，不凭经验猜
候选。每项记录可证伪假设、适用 case、收益上界、当前后端依据、风险和验证成本。
按瓶颈、目标差距、证据成熟度与成本排序；没有生成代码差异的等价写法合并处理。

每轮：

1. 从 B* 按文件清单创建 Ci，追加 CREATED，冻结六轴身份和 applicable_cases。
   记录父版本、源码 SHA、假设和回退点。主要假设变化使用新 ID；必要耦合改动须解释。
2. 在 Ci 中实施，先验证变更影响的相关 shape/dtype/尾块和分支。测试命令与实际收集到的
   node ID/用例数量落盘；不能用空选择器、跳过项或原目录导入冒充验证。影响范围不明或
   涉及跨 case 状态时扩大到完整测试。此时通过只代表迭代精度，不代表最终验收。
3. 对全部目标性能 case 采集，核对 profiling 身份，比较 Ci/B*/B0；不只采收益最大的 case。
   记录增量加速比 `time(B*)/time(Ci)` 和累计加速比 `time(B0)/time(Ci)`。
4. 追加与 CREATED 身份/case 一致的 RESULT，逐 case 标明修改路径是否实际执行及其性能
   证据。编译/精度失败只有健康基线对照能归因到候选时才是 rejected；环境/设备/profiler
   故障记 BLOCKED，关联义务保持未决。结果样式见[终局样例](assets/optimization_search_coverage.final.example.json)。
5. 更新义务、最优指标与证据，通过普通校验后再创建下一候选。瓶颈、粒度、物理数据流或
   buffer 生命周期变化后重开相关组合义务；旧结构上的失败不否决新组合。

流水候选按 A3 最佳实践检查所有存活 buffer、自动/手动版本、满波/尾部和生成代码；
单个自动组合失败不直接否决手动实现。直接寄存器路线仅在当前 A3 API/lowering 有依据
时实施；未知能力保持待核验，不能照搬 A5 的 vld/vselr/VF 路线，也不能直接写“不支持”。

## 5. 晋升与恢复

候选只有相关精度通过、全部目标 case 同口径测量有效、整体收益超过噪声与安全余量、
逐 case 无不可接受退化且结果可复现时才能晋升。默认报告几何平均加速比、最差 case
和全部退化项；用户权重/目标优先。不明确的 case 取舍先保留候选，不能擅自覆盖 B*。
差异接近噪声时交替复测 B* → Ci → B* → Ci。流水/多版本/持久化候选还须同进程遍历
全部 case，并对不同通用路径做逆序或交错复测；跨 case 漂移视为未验证。

晋升时才以 Ci 更新 working，账本 best_metrics 与对应 measurement 绑定；被拒绝的
候选保留自身目录，不污染 B0/B*。无收益后重新评估瓶颈和高收益结构方向；连续无收益
仅触发重新分析，不能自动关闭搜索义务。

恢复时读取 manifest、账本、文件 SHA 和最新报告，校验输入、环境和 B* 来源。仍可恢复
的 CREATED 无 RESULT 候选继续原假设并补齐 RESULT，不重复追加 CREATED；身份变化则
保留旧证据、先如实关闭或阻塞原候选，再分配新 ID。历史 BLOCKED RESULT 不覆盖；同
假设恢复用新 ID、相同六轴/cases，并在 CREATED/RESULT 写 `resumes_candidate_id` 指向
旧候选，实际恢复通过后才解除对应义务。环境漂移则保留旧测量、重新采集对照；源码/
测试/case 合约变化需新会话，旧记录不能冒充新输入的完整搜索历史。

旧单轮日志可作为带来源的候选线索；由主控保留历史状态并启动新 Flash run。不能将
`baseline_iterN` 或旧轮数转换成虚构的 CREATED/RESULT、完整精度或收敛证据。

## 6. 停止与最终验收

正常完成需要运行 `--final`。目标全部满足返回 TARGET_MET；没有数值目标时，只有结构
覆盖及逐 case 下界审计完成才返回 CONVERGED_WITHOUT_NUMERIC_TARGET。有目标但未达到
时，普通 `--final` 的 SEARCH_INCOMPLETE 要继续搜索；只有真实的逐 case 覆盖与下界
审计闭环后，才可用 `--final --allow-unmet-convergence` 取得 UNMET_BUT_CONVERGED。
不能为了过门禁自造下界、删除慢 case 或改目标。待核验能力、未恢复环境阻塞和未决
高收益方向须保留。用户预算耗尽、主动停止或环境故障可中断，但返回未完成状态。

搜索拟结束后：

1. 对 B* 的隔离副本执行现有完整 pytest 套件，保留原 golden/容差、配置、设备参数；
   记录完整命令、收集用例、退出码和日志。最终失败不交付 B*，不标记 Stage 3 完成。
   此时最终 `search_gate` 记 NOT_RUN、`performance_status` 记 NOT_MET_UNFINISHED；此前
   目标命中只能作为历史测量记录，不能报告为当前可交付 MET。
2. 对 B0/B* 全部目标 case 同口径复测；漂移时重采基线。检查测试/源码 SHA、真实导入、
   kernel 链和测量身份。以最终结果更新账本 best_metrics/证据，并重新运行终局校验，
   不能沿用最终复测之前的 TARGET_MET。
3. 全部门禁通过且相对 B0 有可靠收益时归档 B*；没有可靠收益时对 B0 完成同样的完整
   验收后归档 B0。交付版本发生改变时必须重算账本指标并重验目标，不能拿候选指标宣称
   基线达标。未完成验证的源码只能留在 candidates，不能放入 final_optimized。
4. 以文件清单核对归档 SHA 与实际验收版本一致，写报告和 timing 总耗时。报告列出所有
   case 对比、最差退化、未完成候选、停止原因、来源复用和绝对产物路径。

返回契约与 perf-tuner 一致：`stage: 3`、`run_dir`、`report_path`、`deliverable_dir`
（未交付为 null）、`stop_reason`、`implementation_status`、`performance_status`，以及
`final_validation`：

- `full_precision`、`same_method_performance`、`source_identity`：PASS / FAIL / NOT_RUN；
- `search_gate`：TARGET_MET / CONVERGED_WITHOUT_NUMERIC_TARGET / UNMET_BUT_CONVERGED /
  SEARCH_INCOMPLETE / FAIL / NOT_RUN。

| 字段 | 状态及含义 |
|---|---|
| implementation_status | VERIFIED：完整验收通过且有可靠收益；NO_CHANGE：完整验收后的基线；CANDIDATE：候选未完成最终验收；BLOCKED：环境、编译器或测试阻塞 |
| performance_status | MET：对应 TARGET_MET；CONVERGED_WITHOUT_NUMERIC_TARGET：同名门禁；NOT_MET_CONVERGED：对应 UNMET_BUT_CONVERGED；NOT_MET_UNFINISHED：其余情况 |

仅 VERIFIED/NO_CHANGE、前三项 final_validation 全 PASS、search_gate 为三个完成终态且
双状态一致时，主控才可完成 Stage 3。VERIFIED 不等于 MET；改善但未完成搜索仍是未完成。
返回明确指向 final_optimized 的交付路径，不能声称原工程已自动替换。

## 来源与离线检查

迁移自 CANNBot `9cd449120897368cde5a45c12ca45a133be157d7` 的
`plugins-official/tilelang-op-orchestrator/workflows/Ascend950/` Flash 工作流与搜索门禁，
按本仓 A3 的 pytest、profiling、分析 Skill 和三阶段主控适配。多 CLI 注册与 CANNBot
回同步另行处理。

```bash
python3 -m unittest discover -s "$FLASH_SKILL_DIR/tests" -v
```

离线测试验证记录与停止判定；真实编译、精度、NPU profiling 和端到端 Agent 调用需在
对应 A3 环境另行验证，不能由样例校验通过推导出来。
