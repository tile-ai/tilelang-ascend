---
name: tilelang-perf-optimization
description: 基于已有精度与性能证据，为 Ascend A2/A3 TileLang 算子分析逐 case 瓶颈、设计 Tiling 和流水候选并输出性能调优方案。适用于精度通过后的性能方案分析；性能采集、代码实施和迭代管理由调用方负责。
---

> 平台：Ascend A2/A3（Ascend910B / Ascend910_93，DAV_2201）。本文件及其同目录资源仅用于此平台；相对路径以本文件所在目录为基准。

# TileLang A2/A3 算子性能方案分析

本 Skill 只面向当前仓库的 TileLang A2/A3 前端与已选后端。目标是把实测瓶颈转化为可由 TileLang 代码实现和验证的方案，不维护其他编程模型的 API、代码模板或迁移规则。

## 职责边界

- 本 Skill 负责：基线证据整理、逐 case 瓶颈分析、case 分组、Tiling 与流水候选设计、方案排序和验证设计。
- 名为 `tilelang-performance-best-practices` 的 Skill 负责：TileLang API、执行域、buffer、模板成熟度、当前后端 lowering 和实现门禁。制定方案时必须同时加载；未安装或无法按名称加载时停止并报告。
- 当前目标算子、公开接口与测试是产品事实源；实际导入的 TileLang 源码及其 `examples/`、`testing/` 是前端能力事实源。
- 当前后端接收 TileLang lowering 的结果。底层编译器能表达某项能力，不等于 TileLang 前端已有对应 API。
- 本 Skill 不直接修改 kernel。用户要求实施时，把通过门禁的方案交给实现阶段。
- 性能采集、迭代采纳/回滚与最终归档由调用方负责；本技能消费已有证据，缺少数据时返回具体补采需求。保持用户或 DESIGN.md 已确定的 Developer / Expert / 混合模式。

由 [Flash 工作流](../tilelang-tuning-flash/SKILL.md) 调用时，额外接收当前 B0/B*/Ci
身份、已冻结 case、实现/整链约束、用户预算和已有搜索义务。返回方案时关联 case ID、
已有 obligation ID（或明确建议新建的义务）、六轴候选身份和验证需求；已有失败仅在
相同结构与上下文下复用。最多三个已准入方案限制当前执行队列，不能删掉未入队义务。
本 Skill 不写 Flash ledger、不标记晋升或终局；由执行者核验并落盘。

## 必需输入

开始分析前确认以下信息齐全：

- 待优化的完整 TileLang kernel、dispatch/build 入口和正确性测试。
- 全部目标 case：case ID、shape、dtype、属性和公开接口约束。
- 当前代码的精度基线与 当前后端编译状态。
- 同一设备、环境、warmup、repeat 和并发设置下的逐 case kernel latency。
- 可获得的 profiling 指标或 trace；没有指标时不得虚构 bound 类型。

缺少逐 case latency、代码与 case 映射或基线不能正确运行时，先返回数据采集阶段，不输出确定性优化结论。

## 工作流

### 1. 固定事实与比较口径

多 launch 的公开算子先核对核名、聚合规则和 case→launch 映射，避免在漏计的基线上制定方案。调用方已接入 `tilelang-op-profiling` 时，按名称加载并传入 A2/A3 平台和设备证据，读取其整链计时说明；该技能尚未提供时消费调用方给出的同等证据，缺失项返回数据采集阶段，不假定该技能已经安装。

1. 记录当前 commit、实际导入的 TileLang 路径、CANN 与实际后端版本、设备和测试命令。
2. 从测试与 dispatch 还原每个 case 实际进入的 kernel 分支，不能按算子名推测。
3. 校验 profiling 中的 kernel 来自目标源码，整理全部 case 的基线 latency 与可用指标。
4. 按 [证据与瓶颈归因](references/evidence-and-diagnosis.md) 区分计算、搬运、标量/调度、并行度和 launch 开销。

### 2. 按执行路径分组

先按实际计算路径分组，再考虑 shape 大小。推荐顺序：

1. 语义路径：elementwise、broadcast、reduction、transpose/gather、GEMM、attention 或 irregular。
2. dtype 与累加精度。
3. 当前 dispatch/kernel 分支。
4. 对齐、尾块、空任务和特殊属性。
5. 实测瓶颈特征与 shape 规模。

每个 case 必须出现在且只出现在一个基线瓶颈组中。分组方法见 [Tiling 与 case 建模](references/tiling-and-cases.md)。

### 3. 建立当前实现的资源模型

对每个组写出可检查的当前值，而不是先套固定模板：

- 独立任务数、实际核数、每核任务量及负载尾差。
- tile shape、循环次数、有效元素数和 padded footprint。
- GM/UB/L1/L0 buffer 大小、buffer versions、常驻数据与安全余量。
- 每 tile 的读写字节、有效计算量、重复搬运和中间结果生命周期。
- 输出表达式依赖的逻辑维度，以及不依赖 table、channel、head、output column 等维度的循环不变量和公共子表达式；量化外提或复用前后的逻辑工作量。
- 对 irregular/标量索引路径分别统计整数乘除/取模、位运算、cast、分支、索引计算和每任务 live state，不能只用 FLOP 代表实际计算成本。
- `T.Persistent`、`T.Pipelined`、`T.Parallel`、`T.tile.*`、`T.gemm_v0`、`T.mma` 或归约结构的实际使用方式。
- 按名称加载 [npu-arch](../npu-arch/SKILL.md) 并传入 A2/A3（DAV_2201）和已确认的完整 SoC/NpuArch，复用入口的完整硬件探测证据，记录目标架构、核数和硬件容量，并按 `tilelang-performance-best-practices` 从当前 lowering 核对平台用户可用量、编译器预算和完整 footprint；执行域或 buffer 变化时重算，不能沿用任意安全 cap。
- 对 Elementwise、gather/scatter 或布局转换，按对应指南建立嵌套任务树、tile 边界和固定有效 payload 的 Vector 物理路线成本；可实施且可能改善瓶颈的路线必须有代表候选或确定的淘汰证据。
- 多级流水按对应指南记录每个跨迭代 buffer 的生命周期、版本、稳态/尾部范围和总 footprint。

对全部 case 记录 `outer_items`、独立 inner chunks、核数、outer/flattened task waves、单任务 payload、kernel time、搬运/计算下界和主要 pipe，并逐 case 分析粒度边界、完整物理路线和交互组合；planar/scratch 物化不能记为直接 register-resident。未进入最终三个方案的结构方向仍须给出裁决依据，具体状态和落盘格式由调用它的调优工作流管理。

只有公式所依赖的硬件容量、对齐和 API 行为已从当前源码或已验证实现确认后，才能给出具体数值参数。

### 4. 生成并筛选候选

内部混合精度、HF32 和近似指令可列为实验候选，但不得修改参考实现、测试、原容差或公开约定；只有正确性通过且同口径性能提升时才保留。

本 Skill 中的优化点、经验和模板是经过总结或验证的候选来源与示例，不构成 TileLang 算子优化空间的穷举或封闭方案集。先依据当前算子源码、全部目标 case、资源模型和 profiling 证据生成候选；可以迁移、组合或扩展本 Skill 中的相似优化原理，也可以提出本 Skill 未收录的候选。命中现有模板不是停止探索的条件；仍须考察与实测瓶颈直接相关、且有 TileLang API 与当前后端 lowering 依据的非模板候选。所有候选都必须有可证伪假设，并经过后续的精度与同口径性能验证；证据不足时按本节门禁标为 `DESIGN_ONLY`。

每个候选必须对应一个可证伪假设，例如：

- 将不依赖某输出维度的中间结果外提并在 UB 中复用，消除跨 table/channel/head 的重复计算。
- 增大或减小 tile，以降低固定开销、提高并行度或控制 buffer 压力。
- 调整任务到核的映射，减少负载不均或空核。
- 使用 `T.Persistent`/`T.Pipelined` 与完整的 buffer versions 改善搬算重叠；自动分析不适用时按当前 API 设计手动版本。
- 在同一执行域消费中间值，减少 GM 往返或重复读取。
- 为 reduction/GEMM 保留必要的 fp32 状态，并调整分层归约或 K 维流水。
- 为对齐边界和尾块拆分 dispatch 分支，避免所有 case 承担通用路径成本。

候选实施前先按可消除工作量、与实测瓶颈匹配度、覆盖目标差距的可能性、实现风险和验证成本排序。逻辑工作量下降只能作为收益上界；只有同条件实测才能报告实际加速。多个语义等价算术写法若没有 lowering/生成 IR 差异，只保留一个候选。

随后先加载名为 `tilelang-performance-best-practices` 的 Skill，并传入平台 A2/A3（DAV_2201），逐项完成以下门禁：

1. 在目标仓库同类算子 `.py` / `.tl` 或实际 TileLang 源码中找到 API 和调用结构依据。
2. 复用 `tilelang-performance-best-practices` bundled reference 时，标注真实 Python 路径及成熟度；只有 `PRODUCTION_REFERENCE` 或重新验证后的 `VERIFIED` 可作为直接复用候选。当前仓库或实际 TileLang 源码中的真实实现不要求先登记到该状态表，但仍须核验 API、lowering、精度与适用范围。
3. bundled reference 中的 `EXECUTABLE_BASELINE` 只作为正确性起点；`PARTIAL`、`DESIGN_ONLY` 或无可执行 kernel 的资料不得以模板身份进入直接实施方案。若候选另有当前仓库或实际 TileLang 源码证据，应按该直接证据独立判断，不因模板库未收录而自动降级。
4. 无法确认 lowering、布局或 API 参数时，将候选标为 `DESIGN_ONLY`，不编造 TileLang 写法。
5. 流水候选必须阅读 `tilelang-performance-best-practices` 的 `references/elementwise/double_buffer_design.md`，写明自动/手动版本选择、所有被版本化的 buffer、调度器可见控制流、生成代码检查点、上板生效判据和顺序稳定性测试。只写 `num_stages=N` 不构成可实施方案。
6. Elementwise、gather/scatter 或布局转换候选必须阅读 `tilelang-performance-best-practices` 的 A2/A3 资源 `references/elementwise/tiling_task_vector_search.md`，输出 tiling 边界、嵌套任务并行度和 Vector 指令成本；未选的高收益物理路线必须有淘汰证据。
7. 容量影响 tiling、驻留或流水时必须使用 `npu-arch` 确认硬件资源，并从当前 TileLang lowering 核对编译器预算与预留；附上硬件规格、平台用户可用量、编译器预算与预留、显式/常驻/多版本 footprint 和剩余容量账本。增加一个仅用于初始化或尾部的执行侧操作、cast、scratch 或临时 buffer 也要按最终 kernel IR 重算。

最终最多保留三个已准入方案，状态只能是 `IMPLEMENTABLE` 或 `EXPERIMENT`。尽量让每个方案只验证一个主要假设；互斥 dispatch 分支可合并到同一方案，但必须列出分支条件。`DESIGN_ONLY` 单列为“未准入候选”，说明缺失的 API、lowering 或验证证据；仅在候选依赖 bundled reference 时说明缺失的模板证据。它不占三个方案名额，也不交给实现阶段。方案名额不能删除结构义务；若多个正交轴会改变彼此的连续 payload、固定成本摊销或瓶颈，必须保留组合方案或明确的不可实施证据。

### 5. 输出方案与验证顺序

按 [方案与验证](references/plan-and-validation.md) 输出：

- 基线事实表和逐 case 瓶颈表。
- 每个方案的证据、假设、具体参数、TileLang 代码改动位置、参考实现、成熟度和适用 case。
- 精度、当前后端 lowering、逐 case 性能比较及回退条件。
- 全部 case 覆盖清单；没有证据支持优化时明确写“暂不修改”。

## 硬性规则

- 不把经验阈值当成结论；bound 必须由本轮数据支持。
- 不遗漏慢 case、尾块 case 或当前失败 case，不用平均值掩盖单 case 退化。
- 不通过降低 reference 精度、放宽容差、减少测试或 host 侧短路获得通过。
- 不把底层 IR、编译器内部能力或设计文档写成 TileLang 前端 API。
- 不把“理论上更快”写成性能收益；只有同条件实测才报告加速比。
- 候选失败只否决记录中的准确组合；自动版本化失败、某个 tile 的 stage 2 变慢、某种 gather 索引精度失败、planar 物化回退或小 group 收益不足，不得扩大为手动版本化、其他 tile、直接寄存器重排或完整连续单元均不可行。
- 结构覆盖审计不依赖用户是否给出数值目标：高收益义务必须进入准入方案或有淘汰/不适用证据；存在能减少主瓶颈物理访存、额外物化、提高有效 lane、补足核并行度或完善已准入流水的未裁决路线时，不得声称证据已收敛。
- 流水一旦由每核迭代数、容量和 MTE/Compute 可重叠性准入，自动失败或部分生效后必须继续裁决手动显式多版本代表；只在自动完整生效或有 API/lowering/容量/依赖的确定证据时关闭。
- “严格受支配”须附同一有效 payload 的具体指令、访存、转换、物化和同步成本；笼统推测不能关闭候选。
- 不在方案分析阶段改源码；实现、编译和实测由后续阶段完成。

## 场景

- 已有全部 case 的编译、精度和 profiling 数据：完成分支映射、瓶颈分组并输出最多三个已准入方案；`DESIGN_ONLY` 另列为未准入候选。
- 只有源码或少量 latency：输出缺失证据和采集计划，不猜测 bound、具体 tile 或加速比。
- 只有底层编译能力说明、没有当前 TileLang 表达与 lowering 依据：保留为 `DESIGN_ONLY`，不交给直接实施。

## A2/A3 实现与调用域补充

- 使用 [最佳实践入口](../tilelang-performance-best-practices/SKILL.md) 和 [性能关注项](../tilelang-performance-best-practices/references/performance-antipatterns.md) 定位当前平台资料。
- 当前 `T.Persistent` 负责任务映射，签名不包含 `num_stages`；流水由 `T.Pipelined` 或已验证的手动多版本表达，不把其他版本的注解参数直接移植到本仓库。
- 支持单文件内嵌 golden/test 与 kernel/test 分离布局；从实际入口确定测试和 dispatch，不为分析改造工程结构。
- 候选扩大 shape、batch、dtype、尾块或 stage 时，先在方案中记录 kernel 兼容性维度和未知项；存在 unknown/false 时不得直接扩大 dispatch，需先修复或新建 kernel 并设计独立验证。
- 按布局依赖、buffer 数量/生命周期和配置依赖安排顺序；必要耦合改动作为同一方案。实施、测试和采纳由调用方完成。
