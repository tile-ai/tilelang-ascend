# 主仓 Ascend 调度实践迁移到 A2/A3 的技术参考

目标是在 A2/A3（910B/910C）上逐步引入主仓的任务依赖分析、自动调度、
缓冲区生命周期管理和同步验证，并增强现有 Parallel、Pipelined、Persistent。
推荐先为现有规划器增加约束求解策略，输出现有软件流水注解，复用专仓的流水展开与同步生成；
随后再迁移 owner/epoch、版本管理及新的同步协议。每个阶段独立验收。

分阶段交付、验证脚本与人工审核见 [migration-plan.md](migration-plan.md)。
未注明已实现的接口和 Pass 仍为设计提案；S1 审计见第 11 节，S2 合同见第 12 节，
S3 的循环与标记实现边界见第 13 节。
S4 的约束桥接与最终同步验证合同见第 14 节，实际验证状态以迁移计划为准。
S5 的多 stage/版本与成本模型合同见第 15 节。
S6 的 storage/view owner 与完整生命周期 bridge 合同见第 16 节。

用户已明确当前及后续只要求 A3 测试和验收。本文的 A2/A3 目标范围保留既有后端
兼容性，不要求新增 A2 专属测试，也不把 A2 历史未运行当作步骤验收缺口。

## 1. 基线与范围

| 项目 | 源码基线 | 用途 |
| --- | --- | --- |
| 主仓 tilelang | 1c0f402e166a7d0a055c991d2f82d2175a7917b7 | Ascend 任务模型、调度器、生命周期与测试合同的来源 |
| 专仓 tilelang-ascend | 43b0b69720f479894a13c02746599eccee5c5769 | A2/A3 迁移和回归比较的固定源码基线 |

tilelang-ascend 的源码基线不随工作分支推进或提交合并而改变；如需调整，
按迁移计划的审核规则单独确认。

首轮以 AscendC + Cython 为执行基线。保留 A2/A3 的向量指令、GM workspace
跨 C/V 交接及现有代码生成器；新调度采用显式启用方式。现有 Expert 与
Developer 内核继续使用 legacy 路径，默认行为不变。

本迁移不要求升级 TVM 到 TIRX，不迁入 950 的 SIMD/SIMT VF 执行、直接 C/V
搬运、专属指令或 flag spill 协议，也不以切换 TVM-FFI 为前置条件。
PTO 的新调度支持另设验收，不从 AscendC 结果推导其可用性。

### 1.1 已有能力与真实缺口

| 能力 | 专仓已有实现 | 需要迁移的合同 |
| --- | --- | --- |
| Parallel | 1D/2D Vector lowering；部分表达式串行回退；3D 拒绝 | 可选 annotations、显式任务边界；保留现有向量执行方式 |
| Pipelined | num_stages/order/stage/sync/group/cross_interval | 兼容追加 annotations、新规划策略；保留已有跨核参数 |
| Persistent | 固定 core 按步长遍历 tile 域，含 partial-wave 保护 | num_stages/annotations、非整除 group 的 padding 和有效 tile 谓词 |
| 自动同步 | AscendSyncInsert/VS；CombineCV 的跨核 workspace 同步 | 统一读写/别名/生命周期描述、guard 与 token 验证、资源预算诊断 |
| 自动调度 | PipelinePlanning + CrossCorePipeline | 可验证的任务图、资源约束和成本模型、约束调度器 |
| Task/Stage | 缺少对应公开接口和消费链 | 标记、结构验证、消费器和未支持情形诊断 |
| 缓冲区版本 | 旧 InjectSoftwarePipeline 的多版本展开 | 后续加入显式 owner/epoch、iteration/counter 版本时钟 |

Persistent 在此表示固定数量的 core 持续遍历多个 tile，不是编译缓存，
首轮也不引入全局原子领取任务的动态队列。

### 1.2 现有编译合同

| 项目 | 源码设施 | 实施要求 |
| --- | --- | --- |
| Phase 2 末端 | ResourceScopeVerify，A2/A3 AscendC 的 InstructionSelection/MaskLegalize | 保留调用顺序和末端边界 |
| C/V 自动划分 | AUTO_CV_COMBINE 默认 true，其余自动同步/内存策略默认 false | 新策略校验所需配置，保持 legacy 默认值 |
| 操作分类 | 共用 ResourceForCall/资源验证及扩展 OperationConfig | 复用分类接口，审计实际访问足迹和复合副作用 |
| Phase 1/2 顺序 | scope 推断先于临时缓冲区；CombineCV 后重算 allocation placement | 以 phase.py 为准 |
| Persistent | 参数检查和 partial-single-wave 保护 | 回归已有合同，再增强注解/非整除 group |
| 软件流水 | stage/order ABI 和 L1 分形多版本修复 | 复用一次 injector，验证最终版本和物理容量 |
| 测试 | 已有保护性回归 | 核对覆盖及实际运行结果 |

## 2. 为什么分阶段适配

主仓在 TileOp lowering 之前提取任务并运行调度链；专仓在 Phase 1 已完成
Parallel、TileOp 和虚拟 workspace lowering，Phase 2 才做流水和同步。
主仓的 task/IR 结构还依赖 TIRX、较新的 FFI、VF、TileOp 访问区间和 950 成本模型。
因此首轮在专仓已降低的硬件语义调用上构造适配层，不直接追加主仓整套 Pass。

最小接入点是现有 PipelinePlanning 内部的策略选择：CombineCV 之后、
InjectSoftwarePipeline 之前。抽取共享输入，保留旧启发式并增加约束求解策略；
不复制一份大型规划 Pass。新策略输出同一 software-pipeline 注解 ABI，
每个循环只经过一次规划和一次流水展开。

### 2.1 推荐的桥接流水线

```text
专仓 Phase 1：scope/Vector/Layout/TileOp/workspace lowering
  → CrossCorePipeline（首片不接受其处理过的跨核循环）
  → CombineCV
  → allocation placement
  → PipelinePlanning（legacy 或 constraint 策略，二选一）
  → 原 InjectSoftwarePipeline（执行一次）
  → 展平、存储重写等现有变换
  → 原 AscendMemoryPlanning
  → 原 AscendSyncInsert / AscendSyncInsertVS
  → constraint 模式的旧同步协议只读验证
  → ResourceScopeVerify
  → A2/A3 InstructionSelection / MaskLegalize
  → 现有 AscendC codegen 与执行
```

这是首轮桥接模式。新规划器只决定允许的重排和流水注解，旧同步器仍是实际
flag/barrier 生成者。后续启用新同步生成器时，必须切换同一区域的同步所有权，
不叠加两套自动同步；同理，主仓 MaterializeMultiBuffer 不能与旧 injector
同时管理同一个缓冲区。

### 2.2 配置与集成提案

拟新增 `TL_ASCEND_PIPELINE_PLANNER = "tl.ascend_pipeline_planner"`，
取值为 legacy 或 constraint，默认 legacy。
它选择首轮桥接规划策略，不宣称具有主仓完整 AutoSchedule 的全部语义。
新增配置必须进入编译缓存身份；现有 pass_configs 缓存路径需有回归验证。

Phase 2 的 Python 接入点保持原有调用，不增加第二个规划 Pass：

```python
mod = tilelang.transform.CrossCorePipeline()(mod)
mod = tilelang.transform.CombineCV()(mod)
mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
```

PipelinePlanning 内部拟议分发逻辑如下，方法名为设计接口：

```text
strategy = PassContext[tl.ascend_pipeline_planner, default=legacy]
validate strategy
if strategy == legacy:
    return ExistingPipelinePlanning(loop)
inputs = CollectPipelineInputs(loop)
ValidateConstraintMode(inputs, target, platform)
plan = PlanWithConstraintSolver(inputs, capabilities)
VerifyPipelinePlan(inputs, plan)
return EmitLegacyPipelineAnnotations(loop, plan)
```

新增严格验证只作用于 constraint 分支，不给 legacy 增加新的拒绝条件。
共享 helper 的抽取需用回归证明 legacy 行为一致；可以先保留完整旧入口，
再逐步合并经过验证的抽取逻辑。

选择 constraint 时，入口统一校验 A2/A3、自动同步及内存规划的所需配置。
首轮要求使用已有自动同步配置；显式关闭与新模式冲突时给出诊断，不能静默跳过
同步，也不能擅自清除用户手写 flag。legacy 路径不增加新默认值。

首轮只接受完全受支持的内核/循环；遇到不支持的调用或控制流明确报错。
后续按区域回退时，必须证明区域边界的依赖和同步完整，不能静默局部回退。

bridge 模式另增一个只读验证 Pass，位置在现有同步生成和最后一次同步相关简化
之后、AscendResourceScopeVerify 之前；仅 constraint 策略调用它。它不产生事件，
因此不会与旧同步器形成双重同步插入。后端 InstructionSelection/MaskLegalize
之后仍不增加 TIR 变换。

## 3. 任务与依赖合同

### 3.1 拟议实现组件

| 组件 | 职责 | 主要方法 |
| --- | --- | --- |
| A2A3ScheduleCapabilities | 提供存储预算、合法搬运/PIPE/core、事件域和成本输入 | ValidateTarget、MemoryBudget、EventBudget |
| LegacyOperationAdapter | 将审计过的 lowered call 转换为显式操作效果 | ClassifyCall、CollectEffects |
| ScheduleTaskGraphBuilder | 构造任务、读写、控制依赖及保守别名关系 | Build、VisitStmt、AddDependence |
| TaskScheduleVerifier | 独立检查求解结果的合法性 | VerifyCoverage、VerifyDependencies、VerifyResources |
| LegacySyncProtocolVerifier | 在 bridge 的最终同步 IR 上检查事件复用合同 | VerifyEventLifetimes、VerifySetWaitPairs |
| LegacyPipelineAnnotationEmitter | 按旧 injector ABI 生成 stage/order | Emit、ValidateAnnotationShape |
| PipelinePlanning 的策略分发 | 编排共享抽取、旧启发式/新求解器、验证和注解写回 | CollectPipelineInputs、PlanWithConstraintSolver |

拟议 C++ 类采用专仓 `tvm.tir`、旧 FFI 和 C++17。收集器使用 StmtExprVisitor；
注解写回器使用 StmtExprMutator。保持原 PipelinePlanning 的 TVM_REGISTER_GLOBAL
与 Python wrapper；solver 通过旧 tvm.register_func 接入，不引入主仓 BackendContext。

新增 solver 的纯数字约束逻辑使用独立 Python 模块，避免依赖设备运行时。
Z3 作为明确的可选依赖声明，legacy 初始化不应导入它。constraint 模式缺少依赖时
给出安装提示；保留超时、无解和异常的可区分结果，并独立验证整数调度输出。

任务至少记录：稳定 ID、原始语句顺序、core 归属、PIPE/复合资源、读写 storage
及区间、标量/配置状态副作用、控制 guard、循环域、latency/II 和用户约束。
storage identity 以 Buffer/data Var 与显式 alias 关系为准，不以名字为准。

OperationConfig 可复用操作分类与 buffer 参数读写信息，但不能独自证明完整
DMA 物理足迹、复合 helper 的多 PIPE 使用或隐式状态副作用。首片只支持经过审计
的调用 allow-list；无法精确恢复访问区间时保守使用整个 buffer。

未知 extern/raw code、手写 mask/flag、未经审计的标量配置必须拒绝或作为已证明
安全的顺序边界。gemm_v0 等内部包含 MTE1/M 的 helper 不得仅因 default_pipeline
为 M 就作为单 PIPE 指令自由重排；可后续作为不可拆的复合任务建模。

### 3.2 依赖不变量

1. 保留 RAW、WAR、WAW 和影响执行语义的控制/配置状态依赖。
2. 读写区间按实际 DMA、padding、layout 和 workspace 足迹计算。
3. 同一 data Var 的不同视图、偏移和 dtype 视图必须进入同一别名分析。
4. 尚未建立完整 epoch 模型时，不假定一次区域写入覆盖所有后续读取。
5. 不跨未知副作用重排，不把手工 flag 当成普通无副作用调用。
6. 求解失败、预算不足或缺少 solver 时明确诊断；不返回看似成功的未验证调度。

区间分析须先将 min/extent 内部索引算术提升到 int64，再构造半开区间端点，
不能只对已溢出的 int32 结果做外层转换。DMA 和连续切片必须累加原 elem_offset，
把非零基址、视图和最终物理别名纳入同一访问足迹。

HF32、MMAD 方向、padding、atomic mode 等适用配置寄存器只有一份物理状态，
不能按 buffer 版本复制。开放手动 Stage 且 enable_offset 时，读者应消费逻辑阶段
顺序中的 reaching setter，并先于下一次物理覆盖；启动时的入口旧值也须受保护。
主仓 solver 的可选 register_phase_deps 使用 (u, v, physical_wrap, gap)，
wrap 只取 0/1，要求 r[v] + wrap * II >= r[u] + gap。移植旧 FFI 时需同步协议与
独立 verifier；条件 setter、opaque effects 和尚未支持的组合保持保守依赖或拒绝。

### 3.3 输入输出伪 IR

以下是合同示意，不是可执行 TIR。首轮静态循环的输入为已审计硬件语义调用：

```text
for k in serial(K):
  load_gm_to_ub(A[k], x)       # MTE2, write x
  vector_add(x, b, y)         # V, read x/b, write y
  store_ub_to_gm(y, C[k])     # MTE3, read y
```

规划器保持这些语句的计算语义，产生与直接子语句一一对应的注解：

```text
for k in serial(K), annotations={
    software_pipeline_stage: [s_load, s_compute, s_store],
    software_pipeline_order: [o_load, o_compute, o_store]}:
  load_gm_to_ub(A[k], x)
  vector_add(x, b, y)
  store_ub_to_gm(y, C[k])
```

stage 为非负且满足依赖距离；order 必须是 0..N-1 的唯一排列，N 为 injector
实际恢复的直接子 Block 数，注解数组按原语句索引对应。
CollectPipelineInputs 必须镜像旧 injector 对嵌套 BlockRealize/SeqStmt 的展开规则，
不能只按 planner 当前表层 SeqStmt 长度计算 N；需专门测试这种嵌套结构。
示例符号表示求解结果，不预先规定具体阶段。M1 限定单阶段/单版本；M2 才允许
跨迭代展开。PIPE 异步不等同 TVM async queue：首片不凭名称生成 async-stage 注解，
只保留当前 A2/A3 路径实际使用的合同，不能因引入自动调度而随意标记异步队列。
重写 SeqStmt 后必须重新校验注解长度和语句索引，不能沿用失效的旧数组。

旧 injector 负责 prologue/body/epilogue 和物理多版本；规划器不能另外添加一层
stage 维度。M2 必须核对实际展开后的版本数、对齐和存储占用，不能仅验证 solver
预测的逻辑大小。

## 4. A2/A3 硬件适配边界

主仓算法可以复用，容量、资源与延迟参数必须由 A2/A3 适配层提供。
当前两仓的编译器预算已经不同，例如专仓 codegen 中 UB=196352、
L0C=131072、L1=524032 字节；主仓 scheduler 使用 UB=248 KiB、L0C=256 KiB
等 950 预算。这些是源码预算，不能替代工具链/设备的最终可用资源合同。

能力输入至少包括：

- 每 core 的 UB/L1/L0A/L0B/L0C 可用预算、保留空间和对齐。
- 操作到 C/V、PIPE 的映射，复合操作资源占用及合法 copy 路由。
- 核内事件域、跨核 flag bank/模式/范围、用户已占用 ID 和可用槽位。
- GM workspace 分配、访问区间、可见性及生产/消费两向通知协议。
- core/vid 分工、launch 与 persistent tile 分工的对应关系。
- A2/A3 的 latency/II 来源和校准参数。

旧 SyncInsert/VS 存在按模 8 分配 ID 的实现，不能据此证明新调度下事件复用安全。
M1 只支持审计过的即时 set/wait 协议，并在旧同步器输出之后增加 constraint 专属的
只读协议验证；遇到同一事件槽仍存活就重新 set、不可配对 wait 或无法证明的控制流，
拒绝该调度。依赖边数不是事件峰值，验证必须按实际事件域、set/wait 和活跃区间计算。
验证还须将受支持任务的 RAW/WAR/WAW 和最终内存地址复用边映射到最终代码，
确认每条依赖由合法 PIPE 隐式顺序、barrier 或事件完成关系覆盖；事件配对本身
不能证明消费者等待生产完成、下一次写入等待前次消费结束。
该检查位于最后的同步相关简化之后、ResourceScopeVerify 之前，不改变 legacy 路径。
M2 跨迭代流水还需扩展验证器覆盖版本和 epoch，不沿用单迭代的证明。

完整事件分配器和按域报告所需/可用槽位的预算诊断在 M4 交付；不迁用未经 A2/A3
验证的 950 get_buf/rls_buf spill。成本估计先采用明确、保守的模型或经验证的用户提示，
不把 950 延迟表当成 910B/910C 的实测数据。

专仓原 annotate_address 与主仓 annotate_buffer_addresses 的合同须分别审计。
若新策略引入精确 placement，需显式正版本数，容量包含全部物理版本与 layout padding；
同一 placed pool 中不同 storage 的物理区间不能重叠，整个 pool 不做生命周期存储复用；
未指定地址的分配使用对齐的空闲区间。placement 不代替同步，不改 legacy 地址语义。
地址、alias/view、raw storage 指针和分配元数据应随最终物理改写保持一致。
A2/A3 的对齐、容量及可用 pool 由能力表决定，不复制 950/SS/SIMT 预留常量。
此能力独立于首片 bridge，不提前开放新地址 API。

## 5. 分阶段交付与验收

| 阶段 | 可独立评审的交付 | 完成标准 |
| --- | --- | --- |
| M0 | 配置/能力合同、审计过的操作效果、只读任务图、solver 协议与独立 verifier | legacy 无行为变化；依赖、别名、预算和未支持情况能被构造测试验证 |
| M1 | 新规划器替换单核静态循环的旧规划策略，单阶段/单版本，复用旧 injector/同步器并验证最终 set/wait 协议 | 单 Vector 路径正确；无双规划/双展开；无活跃事件槽覆盖；默认行为不变；A3 数值与重复运行通过 |
| M2 | 接入 A2/A3 latency/II、跨迭代阶段与旧 injector 多版本，扩展最终协议验证 | 短循环、尾块、0/1/N 次循环、存储预算、事件复用和最终展开结果验证通过 |
| M3 | 迁入 owner/epoch、alias-aware 版本计划和 readiness/reuse 验证 | 初始化、消费、复用和退出路径完整；可空/条件循环不会生成无对应 set 的 wait |
| M4 | 在固定、保守调度下接入新同步生成，再逐步开放重排 | 新旧同步按区域单一所有权；A3 事件预算/跨核 workspace 协议经实机验证 |
| M5 | 完整任务调度与新版本管理接管选定区域，扩展混合 C/V | 退出旧 injector/同步路径的区域有完整新后端合同；混合算子与旧 Expert 回归通过 |

Persistent/循环接口增强可与 M0 并行，但单独验收；Task/Stage 的执行语义应与
其消费器一起交付。首个可运行调度切片建议为 GM→UB→Vector→GM，不以复杂
FlashAttention 或隐藏多 PIPE 的 GEMM helper 作为唯一验收样例。

### 5.1 旧新组件的所有权

| 模式 | 规划 | 版本/流水展开 | 实际同步生成 |
| --- | --- | --- | --- |
| legacy | 旧 PipelinePlanning | 旧 InjectSoftwarePipeline | 旧 SyncInsert/VS 与既有跨核协议 |
| M1/M2 bridge | 新规划器 | 旧 injector，一次 | 旧同步器，一套 |
| M3 分析阶段 | bridge 规划并校验 epoch | 旧 injector，验证其结果 | 旧同步器，验证其 token 合同 |
| M4 同步试验 | 固定保守调度 | 已验证版本路径 | 选定区域仅新同步生成器；其余仍 legacy |
| M5 完整路径 | 新任务调度 | 新版本管理 | 新同步生成器 |

M3/M4 的 bridge 不自动取得任意混合区域的支持。区域混用需定义明确边界，
在覆盖不足时采用内核级拒绝，不能让两个生成器分别保护同一依赖。
VS/标量相关同步只有在新模型覆盖对应副作用后才能退出旧生成器。

M4 的新同步器必须消费最终 AscendMemoryPlanning 的 address_map/size_map，
或从等价的最终分配重建物理别名与复用依赖。StorageRewrite/MemoryPlanning
可能让不同逻辑 buffer 复用同一地址，因此不能直接沿用 M0/M1 的逻辑 storage 图。
物理地址 WAR/WAW 复用边必须在最终 MemoryPlanning 之后、ResourceScopeVerify
之前补齐并验证；固定保守调度也不能替代这一分析。

## 6. Parallel、Pipelined、Persistent 与 Task/Stage

### 6.1 独立循环接口增强

Pipelined 以兼容方式追加 annotations，保留 sync/group/cross_interval。
旧 order/stage 与新 Stage 约束混用必须明确拒绝或验证一致，不静默覆盖。
新 annotations 中只有已实现的语义可以启用，未消费的调度字段必须诊断。

Persistent 增加 num_stages/annotations 并传到生成的 wave loop；对末维 group
非整除情形采用 padded tile 域和有效性 guard。保留现有参数正数、index 检查及
partial-single-wave 保护。静态空 domain/零 extent 继续按已有合同报错；
符号域先覆盖合法正值，不把运行时零 extent 自动解释为受支持的空循环。
少于 core 数、跨 wave 和符号域分别测试。

传递 num_stages 注解只代表接口贯通，不等于 Persistent wave 流水已经可用。
必须另测带有效 tile guard 的 wave body 是否被旧 planner/injector 正确接受，
并验证尾 wave 的依赖和同步；接口 PR 不提前宣称完成其流水执行语义。

Parallel 保持 A2/A3 当前 Vector lowering。annotations 是额外的调度元信息，
不将主仓 SimtVF 线程域引入 A2/A3；现有不支持的维数仍需明确诊断。
新任务/布局注解只挂在最外层 Parallel loop，避免专仓当前逐维复制注解时重复
引入调度作用域；已有 coalesced_width 的行为单独保持并回归。

### 6.2 Task/Stage 标记

先实现 Task() 与 Stage(n) 的 AttrStmt 合同：Task 只包含同一 PIPE、同一 C/V
归属的操作，不隐式 launch、不提供任务体内跨 PIPE 同步；Stage 必须包住完整任务，
不可嵌在 Task/Parallel 内部。首版拒绝 Stage 与旧 order/stage 同时使用。
Stage 约束完整任务和每 PIPE 的顺序。未经审计的复合 helper 不假装
满足单 PIPE Task 合同。非零 Stage、enable_offset 与 num_stages 的语义需与消费器
同时实现；初期可拒绝未支持组合，而不是让标记静默失效。

PerCoreTask、条件 exact-once 分派和完整 stateful/grouped tile scheduler 后置。
后续支持 Task 内部条件时，应保留整个任务的同步边界并汇总所有分支的资源/effects，
区分内部条件与外层候选选择；单 PIPE/core 约束和 exact-once 合同不因此取消。
主仓 AscendTileScheduler 依赖 meta_class/inline/BaseTileScheduler，专仓没有该前端
体系；先增强现有 Persistent，避免为一个调度类迁入整个 eager/meta 前端。

## 7. 文件和依赖清单

下表是分阶段的路径提案；其中已落地的 S2 部分以第 12 节实际接口为准，
其余不代表已经实现。

| 位置 | 计划改动 |
| --- | --- |
| src/transform/pipeline_planning.cc | 抽取共享输入，保留 legacy 并增加 constraint 策略分发 |
| src/transform/common/ascend_schedule_analysis.{h,cc} | 拟新增 post-lower 任务/效果适配与结果验证 |
| src/transform/common/ascend_schedule_contract.h | 拟新增任务/依赖/能力与 solver 数据合同 |
| src/transform/ascend_schedule_sync_verify.cc | 拟新增 bridge 最终 set/wait 协议只读验证 Pass |
| src/transform/common/operation_config.h | 复用已有操作分类；补审计过的显式效果，不另建矛盾分类表 |
| tilelang/transform/pass_config.py | 拟议策略配置，默认 legacy |
| tilelang/engine/phase.py | 保留原规划/展开调用；constraint 在最终同步后增加只读协议验证，后续接管时显式分支 |
| tilelang/transform/ascend_z3_scheduler.py | 拟新增独立数据协议、solver 和结果验证；避免导入设备运行时 |
| tilelang/transform/__init__.py、可选依赖声明 | 以旧 FFI 注册 solver；legacy 不强制加载 Z3 |
| tilelang/language/pipeline.py、parallel.py、persistent.py | 兼容追加注解；Persistent 参数与尾块语义 |
| src/ir.cc 及对应 FFI 声明 | 同步修改 native builder 与 Python 签名 |
| tilelang/language/schedule_hint.py | 拟新增 Task/Stage，随验证/消费器交付 |
| testing/python/language/ 与新调度测试目录 | 复用旧回归，添加依赖、注解、容量和循环覆盖测试 |

主仓可复用来源按依赖分组：

| 来源 | 复用方式 |
| --- | --- |
| Ascend auto_schedule.cc、Python loop scheduler | 迁移求解思想和数据协议；替换 TIRX/新 FFI 与 A2/A3 成本输入 |
| dependency_analysis.cc、src/transform/common/int64_promoter.h | 区间算术提升后再比较端点；补大索引与动态尾界回归 |
| buffer_addresses.h、merge_ub_allocations.cc、remove_no_op.cc | 参考固定地址/版本校验、指针改写与分配元数据维护；不照搬 950 pool 参数 |
| memory_detector/task_analysis/ir_structure/scheduled_tir | 参考任务/依赖/控制流合同；首片使用专仓 post-lower 适配，后续逐项迁移 |
| annotate_multi_buffer_eligible、prepare/materialize_multi_buffer | 后续迁移 owner/epoch 和版本时钟，不与旧 injector 同时改同一存储 |
| insert_sync、rewrite_flag_to_buf | 迁移依赖/token 验证实践；同步发射和事件资源必须适配 A2/A3 |
| language/loop.py、Ascend schedule_hint.py、src/ir.cc | 兼容增强循环/标记 API；保留专仓扩展参数 |
| testing/ascend/auto_schedule | 迁移行为与边界测试，替换 950 VF/直连等硬件夹具 |

## 8. 验证方案

| 层次 | 关键覆盖 | 验证目标 |
| --- | --- | --- |
| 数据协议/solver | DAG、RAW/WAR/WAW、资源冲突、配置寄存器 phase、不可满足约束 | 结果由独立 verifier 校验，不以 solver 成功等同合法 |
| 操作与别名 | 多视图、int32 边界、elem_offset、连续切片、workspace、配置状态 | 正确物理足迹和内部位宽；不按名字判断独立 |
| 流水 ABI | stage/order 长度、唯一顺序、async 注解、单次展开 | 规划器输出与 injector 真实消费合同一致 |
| 控制流 | 0/1/N trip、trip 小于 stage 数、条件 producer、嵌套/连续循环 | prologue/epilogue 和等待路径完整，复用不越过生命周期 |
| 资源 | UB/L1/L0C 边界、对齐、多版本、事件域上限与超限 | 核算实际物理分配与事件活跃区间，超限可诊断 |
| 显式地址 | 正版本数、全版本大小、非零基址、raw pointer、dead/unknown storage | placement/版本/别名一致，重叠拒绝，未知项不被静默丢弃 |
| 循环域 | Persistent 静态空域拒绝、partial wave、非整除 group、合法正值 symbolic | 非法域保持诊断；每个有效 tile 恰好一次，无无效 tile 的副作用 |
| native lowering/codegen | A2、A3、默认/bridge、AscendC；PTO 另列 | 无 950 专属调用；C/V scope、mask 末端合同保持 |
| 实机 | copy-vector-copy、GEMM、归约、混合 C/V、尾块与随机 scratch | 数值正确，无未初始化读和跨 tile 污染 |
| 压力/性能 | 重复 launch、多 core、并发 stream、长流水 | 无死锁/串扰；分开记录编译、host 和设备耗时 |

测试验证依赖和协议不变量，不固定具体 flag ID 或整份生成源码。可能挂起的
硬件测试用独立进程隔离超时，不能仅用线程超时后继续复用同一设备任务。
性能只在正确性和稳定性验收之后判断，不预设新规划器一定更快。

优先复用专仓 test_ascend_sync_insert.py、zero_trip/one_trip、Persistent、
gemm_v0_pipeline 和现有 Developer/Expert 示例。迁移主仓 sync_boundaries、
cross_core_flags、flag_allocation 等测试的行为合同，改成 A2/A3 可执行夹具。
同时参考 test_special_registers、test_ascend_buffer_addresses、DMA offset 与
strided-slice 回归，替换 950 专属指令和资源参数。

## 9. 验证环境

纯数据与 solver 测试使用独立主机环境；TIR、FFI、lowering 和 codegen 测试
要求与源码匹配的 TVM/专仓原生构建。设备验收要求 Linux/CANN、Bisheng、
兼容的 torch_npu 及实际 A3 设备；后续不要求 A2 验收，历史结果分别保留。
仅编译生成代码不能替代数值、同步稳定性和性能验证。

## 10. 源码核对入口

- 专仓：tilelang/engine/phase.py；src/transform/pipeline_planning.cc；
  src/transform/inject_pipeline.cc；src/transform/ascend_sync_insert.cc；
  src/transform/ascend_combinecv.cc；src/transform/ascend_workspace_reduction.cc。
- 专仓：tilelang/language/{parallel,pipeline,persistent}.py；src/ir.cc；
  src/target/codegen_ascend.cc；testing/python/language/。
- 主仓：tilelang/ascend/pipeline.py；src/ascend/transform/auto_schedule.cc；
  annotate_multi_buffer_eligible.cc；prepare_multi_buffer.cc；insert_sync.cc。
- 主仓：tilelang/ascend/language/schedule_hint.py；tilelang/language/loop.py；
  src/ir.cc；testing/ascend/auto_schedule/。

设计采用当前源码确定的顺序；遇到旧 reference 与 phase.py 不一致时，以实际
输入/输出合同和源码为准。新增变换不得跨越最终 MemoryPlanning、ResourceScopeVerify、
InstructionSelection 与 MaskLegalize 的依赖边界。

## 11. S1 基线审计映射

本节以专仓固定基线 `43b0b69720f479894a13c02746599eccee5c5769` 为比较对象，
区分已有合同、S1 回归和 S2 之后的设计。运行结果、实际源码身份和 A2/A3 验收状态
统一记录在迁移计划中；下表列出源码与测试映射，不表示对应测试已经运行。
目前的 `legacy` 指既有编译路径；第 2.2 节的 planner 策略配置仍是后续提案。

### 11.1 编译顺序、配置与 FFI

| 合同 | 已核对源码 | S1 固定内容 |
| --- | --- | --- |
| Phase 1 前置顺序 | `tilelang/engine/phase.py::LowerAndLegalize` | `AscendInferBufferScope` 先于带 `plan_vid_reduction=True` 的 `InjectTmpBuffer`，再运行 `AscendVidReduction`；工作区大小与最终 UB scope 一致 |
| 调度桥接入口 | `tilelang/engine/phase.py::OptimizeForTarget` | allocation placement → CrossCorePipeline → CombineCV → 再次 placement → PipelinePlanning → 一次 InjectSoftwarePipeline；S1 不添加新规划器 |
| 最终地址与同步 | 同上；`src/transform/ascend_memory_planning.cc` | MemoryPlanning 发布最终 `address_map/size_map` 后运行两类同步；不能在其后改变 buffer/access/alias 合同 |
| 资源与后端末端 | 同上；`ascend_combinecv.cc::AscendResourceScopeVerifier` | 同步及最终相关 Simplify 后验证资源归属；仅 A2/A3 AscendC/auto 接 Selection → MaskLegalize；其后无 TIR 变换 |
| CV 与跨核同步默认 | `src/transform/ascend_combinecv.cc::CombineCV::Substitute` | `tl.ascend_auto_cv_combine=true`；`tl.ascend_auto_cross_core_sync=false`；尊重有效显式 C/V scope |
| 普通同步默认 | `src/transform/ascend_sync_insert.cc::Substitute` | `tl.ascend_auto_sync=false`；存在 Pass 调用不等于启用自动同步 |
| 内存规划默认 | `ascend_memory_planning.cc::Substitute` | `tl.ascend_memory_planning=false` 选择既有线性策略，Pass 仍运行并发布映射 |
| VS 默认与入口差异 | `ascend_sync_insert_vs.cc::Substitute`；`tilelang/transform/pass_config.py::_TARGET_PASS_DEFAULTS` | C++ 默认 false；Python JIT 对 PTO 补默认 true，AscendC/auto 不补；用户显式设置优先，不将裸 PassContext 与 JIT 目标默认混淆 |
| mask 状态复用 | `ascend_vector_mask_legalize.cc`；`pass_config.py` | `tl.ascend_vector_mask_reuse=true`；未知 extern/raw code 会使现有 mask facts 失效，但这不构成完整调度副作用模型 |
| 原生接口 | `tilelang/transform/_ffi_api.py`、`__init__.py`；对应 `TVM_REGISTER_GLOBAL` | 继续使用 `tvm.tir`、`tvm._ffi._init_api("tl.transform", ...)` 与 `.set_body_typed(...)`；Pass wrapper 参数与 native 注册配对；CMake 要求 C++17 |

现有 workflow reference 的 Phase 1 顺序仍写成 InjectTmpBuffer → InferBufferScope，
与本基线的实际调用不同。本次以源码为准记录，不把 reference 的步骤号当作执行顺序。

### 11.2 资源分类与操作效果边界

`ascend_combinecv.cc` 中的 `ResourceForCall` 同时服务于 CV 拆分和最终 verifier，
会校验 copy 的局部存储 scope，并处理手写 flag/barrier、mask setter、未知 extern、
`ascend_src_code` 与 `CustomizedCode`。未知代码没有可推断 owner 时要求显式 scope；
在显式 scope 中保留 opaque 调用，只证明归属，不证明它可重排或没有隐藏副作用。
GM scalar store 也必须有显式 owner，不能按附近操作推断后复制到两侧。

`src/transform/common/operation_config.h` 已提供 `GetWorkspaceOpConfigs` 和
`ResolveOperationConfig`。可选 workspace 在 InjectTmpBuffer 后可能插入或移除，
后者按最终 access pointer 的读写 mask 恢复参数位置；现有 planner 和两类同步器均已消费。
S1 复用这项保护，不重新维护一套 workspace 位置表。
其中 operation 配置表和 workspace resolver 已位于可复用的 common header；
资源分类器目前位于 CombineCV 文件内部，尚不是新的公开分析接口。后续若抽取为
共享 helper，必须保持现有 CV/最终 verifier 的共同规则，并用既有回归证明行为一致。

现有 `OperationConfig` 只记录参数读写、默认 PIPE 及 copy scope/owner，不能表达完整的
物理 DMA 足迹、复合 helper 多 PIPE 占用、配置寄存器 effects 或 owner/epoch。
例如 `gemm_v0` 的 `PIPE_M` 分类不证明 helper 内部只使用 M PIPE。现有同步器对未列入
配置表的 extern 不恢复完整访问效果；显式资源 scope 不能填补这个缺口。
这些是 S2 的 effect adapter/allow-list 与后续 constraint 入口需要补充的合同，
不据此给 legacy 增加全局拒绝条件。未知调用、手写状态和复合 helper 在新规划器内
应保守建模或明确拒绝，不能默认无副作用。

### 11.3 旧地址注解与新 placement 的区别

`tilelang/language/__init__.py::annotate_address` 将 Buffer 转为 data Var，并把 byte
offset 写入 `address_map`；`AscendMemoryPlanning::SetPreAllocBuffer` 内部再以
`name_hint` 索引预分配地址。这个旧合同没有新 placement 的版本数参数，也不等价于
alias-aware storage identity。不同逻辑 Buffer 可以显式指向同一物理地址，现有同步
回归正是用它验证复用依赖，S1 不把这种行为改为一律拒绝重叠。

线性策略保留显式地址的原值，auto allocator 则把显式地址向上对齐到 32 bytes，
两条路径不能假定对未对齐输入有相同结果。其他分配沿用各自的现有分配策略；
现有容量检查也不是新约束规划器的完整
资源预算证明。因此第 4 节要求的正版本数、全版本容量、alignment、placed pool 内
不重叠及未放置项空闲区间分配，属于将来新 placement 的独立合同，不能当成旧 API
已提供的能力。最终 `address_map/size_map` 消费和物理地址复用同步仍须保留。

### 11.4 精确回归入口

以下文件均位于 `testing/python/language/`。S1 native 入口按具体 node 选择既有断言，
只做 TIR、FFI、lowering 和生成源码检查；不调用 Bisheng，也不创建 NPU tensor。

| 合同 | 精确测试节点 |
| --- | --- |
| 默认 CV、opt-out 与显式 owner | `test_tilelang_ascend_resource_scope.py::{test_combine_default_and_explicit_opt_out,test_default_combine_lowers_vector_kernel_with_optional_scope,test_default_combine_preserves_explicit_owners_and_rejects_conflicts,test_unscoped_gm_scalar_store_is_not_inferred_from_context}` |
| CV 后依赖与循环结构 | 同文件 `::{test_default_combine_keeps_dependencies_across_explicit_vector_blocks,test_default_combine_does_not_use_local_events_for_cross_core_dependencies,test_pipeline_with_nested_serial_loop_lowers_after_cv_split}` |
| 同步后的资源验证与末端顺序 | `test_tilelang_ascend_vector_mask.py::{test_selection_runs_after_phase_two_and_legalizer_runs_last,test_resource_scope_is_explicit_nested_and_fail_closed,test_resource_scope_checks_automatic_barriers}` |
| opaque 与手写状态 | 同文件 `::{test_opaque_symbol_changes_invalidate_mask_facts,test_resource_scope_normalizes_pipe_case_but_rejects_ambiguous_sync,test_target_scope_keeps_a5_and_pto_unchanged}` |
| 可选 workspace 参数 | `test_tilelang_ascend_language_explicit_tmp.py::{test_every_workspace_api_has_the_target_specific_lowered_layout,test_optional_workspace_metadata_uses_the_lowered_call_layout,test_pto_select_metadata_tracks_shifted_src_and_destination}` |
| Persistent 已有安全性 | `test_tilelang_ascend_language_persistent.py::{test_persistent_guard_for_static_domains,test_persistent_guard_for_dynamic_domain,test_persistent_rejects_invalid_static_arguments,test_partial_single_wave_guard_reaches_codegen,test_multi_wave_break_guard_reaches_codegen}` |
| 物理 alias 和零次循环 | `test_ascend_sync_insert.py::{test_outstanding_reader_survives_another_pipe_read,test_new_reader_is_not_protected_by_old_alias_handoff,test_loop_exit_preserves_the_zero_trip_reader,test_optional_loops_compose_without_losing_the_entry_reader}` |
| 一次循环退出 | `test_ascend_sync_insert_one_trip.py::{test_loop_exit_does_not_borrow_a_second_iteration_handoff,test_nested_loop_exit_keeps_a_writer_created_in_the_outer_iteration}` |
| S1 Buffer view 与 offset | `test_ascend_migration_baseline.py::{test_copy_elem_offset,test_atomic_elem_offset,test_contiguous_view_offsets,test_tile_region_elem_offset,test_tile_fill_view_elem_offset}` |
| S1 原生内存与 L1 物理版本 | 同文件 `::{test_l1_pipeline_physical_versions,test_legacy_memory_maps}` |

`test_ascend_memory_planning.py` 的多项名称虽是 offset/IR 检查，其 helper 实际调用
`tilelang.compile`；`test_ascend_sync_insert_zero_trip.py` 和 GEMM K-tail 流水回归也包含
真实设备编译或 launch。因此它们不能仅凭名称放进无需 Bisheng/NPU 的 native 入口。
设备回归由独立 NPU 入口选择，S1 补充的内存映射和 L1 版本检查直接运行原生 Pass。

native 测试仍依赖 pytest、可导入的 TileLang/TVM 原生构建及 PyTorch：语言测试的
`conftest.py` 与 TileLang adapter/profiler 会导入 torch；若该 PyTorch 环境加载了
torch_npu 扩展，还需其 CANN runtime 动态库可加载。无需实际 NPU 不等于只安装 TVM
即可，也不等于 `--help` 需要这些依赖。

native 入口要求显式重建后记录的源码/库身份 manifest，验证当前项目原生源码、
实际 TVM 构建源码、CMakeCache 与三个本地 `.so` 的哈希，并核对实际加载库路径。
脚本不自动重建；`--help` 仅依赖标准库。已有 fixtures 内部的 A2/A3/PTO 参数保持原样，
报告列出它们的配置范围；`--platform a2|a3` 不把这些 host 检查转换为相应平台的设备
验收。必跑测试未执行、skip/xfail、零收集或 JUnit 缺失不能记录为 PASS。

### 11.5 显式准备 native 构建身份

以下 Bash 命令在仓库根目录运行，要求 Linux/CANN 环境、CMake/Ninja/C++ 编译器和
原仓已完整初始化的 TVM 子模块。它使用独立的本地 TVM checkout，应用仓库内两项
patch，嵌套依赖只读复用本地子模块；不联网、不改原 TVM checkout，也不安装依赖。
请为 `S1_EVIDENCE` 选择新的输出目录；构建日志和 manifest 不提交到仓库。

```bash
set -euo pipefail
export S1_EVIDENCE=/tmp/ascend-migration-s1
export S1_TVM_SOURCE="$S1_EVIDENCE/build/tvm-source"
mkdir -p "$S1_EVIDENCE/build"
git clone --shared --no-checkout 3rdparty/tvm "$S1_TVM_SOURCE"
git -C "$S1_TVM_SOURCE" checkout --detach c2921fdaf795b1103d21abc962e83a209c7258d7
for patch in "$PWD"/3rdparty/patches/tvm_*.patch; do
    git -C "$S1_TVM_SOURCE" apply --check "$patch"
    git -C "$S1_TVM_SOURCE" apply "$patch"
done
python - <<'PY'
import os
from pathlib import Path
import subprocess

original = Path.cwd() / "3rdparty/tvm"
source = Path(os.environ["S1_TVM_SOURCE"])
entries = subprocess.check_output(["git", "-C", str(source), "ls-files", "--stage", "-z"])
for entry in filter(None, entries.split(b"\0")):
    metadata, name = entry.split(b"\t", 1)
    if metadata.split()[0] != b"160000":
        continue
    relative = name.decode()
    dependency = original / relative
    if not dependency.is_dir() or not any(dependency.iterdir()):
        raise SystemExit(f"Missing initialized local dependency: {dependency}")
    destination = source / relative
    if destination.exists():
        destination.rmdir()  # Only an empty gitlink placeholder may be removed.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(dependency, target_is_directory=True)
PY
cmake -S . -B build -G Ninja -DTVM_SOURCE_DIR="$S1_TVM_SOURCE" \
    -DUSE_ASCEND=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_LLVM=OFF \
    -DUSE_GTEST=OFF -DUSE_LIBBACKTRACE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --target tilelang_module tvm_runtime --parallel 8 \
    2>&1 | tee "$S1_EVIDENCE/build/native-build.log"
export TL_ASCEND_MIGRATION_BUILD_MANIFEST="$S1_EVIDENCE/build/native-build.json"
PYTHONPATH=scripts/ascend_migration python - <<'PY'
import os
from _native_build import write_manifest

write_manifest(
    os.environ["TL_ASCEND_MIGRATION_BUILD_MANIFEST"],
    tvm_source=os.environ["S1_TVM_SOURCE"],
    build_command=["cmake", "--build", "build", "--target", "tilelang_module", "tvm_runtime",
                   "--parallel", "8"],
    build_log=os.path.join(os.environ["S1_EVIDENCE"], "build/native-build.log"),
)
PY
python scripts/ascend_migration/s1_check_legacy_lowering.py --platform a3 \
    --timeout 120 --output "$S1_EVIDENCE/native-a3.json"
```

只有构建成功后才生成 manifest；构建失败则停止，不继续运行后面的记录命令。
manifest 记录 TVM 固定提交、两项 patch 的 SHA 和已应用状态，源码哈希包含 patched
TVM 的 Python 文件。native/NPU 入口依据 manifest 同时设置 `PYTHONPATH` 和
`TVM_IMPORT_PYTHON_PATH`，并验证实际 `tvm.__file__` 来自该 checkout；不能把新的
`.so` 与原仓未应用 slice-step patch 的 TVM Python 混用。入口还要求实际加载的
TileLang/TVM 库来自 manifest 记录的本地构建。S1 历史 A2 host 记录保留；
后续不要求 A2 专属检查或设备验收。

### 11.6 S1 的 elem_offset 重复累加修复

修复前的最小诊断用例设置 GM view 的 `elem_offset=32`、切片起点 3，以及输出
view 的 `elem_offset=16`、切片起点 5，原生 lowering 生成的地址是 `a[67]`、`o[37]`，
而所需地址为 `a[35]`、`o[21]`。Python tile region helper 也存在同一合同问题：
`elem_offset=16` 加 region 起点 3 得到 pointer offset 35，public `T.tile.fill`
生成 `v[35]`，都应为 19。诊断库的复现用于定位，不能替代重建后当前源码的验收。

共同原因是 `Buffer::OffsetOf` 已包含 view 的基址，而 `access_ptr` 还会累加
`elem_offset`。S1 只在交接 offset 给 `access_ptr` 前扣除一次基址：C++ 涉及
`src/op/ascend.cc` 的 `AscendCopy::Lower`（src/dst/tmp）与 `AscendAtomicAdd::Lower`
（src/dst）；Python 涉及 `ascend_tile.py` 的 `_get_buffer_info`、
`_handle_buffer_region`、`_handle_buffer_region_2d`。不更改 copy route、layout、
同步策略或公开 API。零基址、双向非零基址、非零切片起点、连续 view 及 public
tile 调用由 `test_ascend_migration_baseline.py` 的精确节点覆盖；正式结果以重建后
native 和相应平台设备记录为准。

## 12. S2：显式分析、能力与数值协议

本节描述 S2 的实现合同；实际构建、检查命令和结果统一记录在迁移计划第 8 节。
S2 不接入默认编译流程，不启用 constraint 策略，不发射 stage/order、版本或同步。
原有 A2 编译路径保留；新增分析和本轮验证明确面向 A3。

### 12.1 原生分析与任务边界

Python 接口为 `AnalyzeAscendSchedule(func: PrimFunc, platform="A3") -> Map`，
旧注册名为 `tl.transform.AnalyzeAscendSchedule`。实现位于
`src/transform/ascend_schedule_analysis.cc`，而非不会被当前 CMake glob 收集的
`common/*.cc`。输入 IR 和 attrs 不变，测试比较分析前后的结构相等。

图保留调用级 task、不可拆 statement group、循环和 guard。所有 ID 按源码遍历
生成；group 对应旧 injector 直接子语句的展开规则，后续 annotation 应按 group
发射，不能把硬件调用数量误当成 injector 的 statement 数量。共享抽取只移动
既有分类和分组规则；legacy pipeline 的行为由已有编译回归验证。

操作归属与访问模式复用现有 `ResourceForCall`、`OperationConfig` 和
`ResolveOperationConfig`，新适配器只补已经审计的物理 footprint 和副作用条件。
首片允许 float16/float32、单行完整连续且合法对齐的 GM↔UB copy，以及受审计的
managed fill/add。部分 copy helper 会执行 Duplicate、padding 和内部事件，
因此动态 copy 尾块不能仅凭 `default_pipeline=MTE2` 被当成单 PIPE 可调度任务。
已知多调用 group 保留组内顺序并标明不可调度；未知调用、raw code、未建模的
配置状态或手写 mask/flag 明确诊断，不返回可被误用的部分成功图。

访问使用 int64 字节半开区间，先递归提升内部索引运算，再构造端点。
`tvm_access_ptr` offset 已包含 view 基址，不再次加 `elem_offset`。
相同 data Var 的不同 view 和 dtype 进入同一 storage；不同名字不证明独立。
未声明 noalias 的 GM 参数、显式地址重叠和无法证明不相交的动态范围保守保留依赖。
这不是最终 MemoryPlanning 后的物理复用验证，后者仍留给后续同步接管。
当前图明确声明 single-iteration 范围；含循环的 group 要求顺序迭代并不可调度，
不能用同迭代的区间不相交证明跨迭代独立。无循环根 owner region 可以暴露子 group，
但保留 parent group 和原 wrapper 映射，不能把它直接当作任意 For 的 injector ABI。

### 12.2 A3 能力合同

`ascend_schedule_capabilities.py` 可以直接按文件加载，不导入 TileLang/TVM/Z3。
`get_capabilities("A3")` 提供 pool、allocation、route 和 local event 验证；
未知平台或未审计路由明确拒绝，不改 legacy 对应路径。

| pool | owner | 编译器预算（字节） | allocation alignment |
| --- | --- | ---: | ---: |
| shared.ub | V | 196352 | 32B |
| shared.l1 | C | 524032 | 32B |
| wmma.matrix_a | C | 65536 | 32B |
| wmma.matrix_b | C | 65536 | 32B |
| wmma.accumulator | C | 131072 | 32B |

每版本 size 必须包含 layout padding 和 scratch，验证器另计算 allocation rounding
及物理版本总占用。表中数值来自现有 codegen/allocator，不把与 nominal 容量的
差额解释为已经证明的硬件保留区。native 夹具只独立核对 allocator 的有效边界、
地址对齐和 codegen `InitBuffer`，不表示 Duplicate 等测试占位操作在所有 pool 都可执行。

当前 CANN 9.1.0 对 `dav-2201` 声明 `QUE_MAX_EVENT=8`，核内 ID 为 0..7；domain
包含 core 实例与有向 PIPE pair，reserved ID 按域扣除。首片仅开放 Vector 侧
MTE2/V/MTE3 的有向不同 PIPE 组合，同 PIPE 顺序另行处理。声明可编码不等于全部空闲，
有隐藏内部事件或未知状态的 helper 不因此获准调度。
跨核四位 ID 编码不证明存在 16 个可自由分配槽，故其预算仍未知且不支持。
`timing_model=None`，不迁入 950 latency、容量、直连或 spill 参数。

新增 verifier 对超预算给出明确拒绝。默认 legacy linear 策略仍按已有
`check_overflow=false` 行为运行，不添加新的拒绝条件。审计时旧 auto allocator 的
UB 超量探针曾退出 139，根因未定位；不能把它记成容量错误诊断通过，正式负例先验证
新 capability 接口的拒绝行为。

### 12.3 Solver 与独立结果验证

`ascend_z3_scheduler.py` 是只使用标准库的单文件模块，首次显式求解才导入 Z3。
host 通过 `spec_from_file_location` 加载并登记 `sys.modules`，避免普通 package
导入先初始化 TileLang/TVM。可选 extra 为
`ascend-scheduler: z3-solver>=4.13.0,<4.15.5`，范围依据固定主仓的依赖声明；
不加入基础 requirements，不修改全局环境。

请求为普通整数/数组组成的 schema version 1：

- 公共字段包括 mode、latencies、iis、resource_flags、data/resource/owner/PIPE
  依赖，以及 timeout_ms 和 seed；时序输入都是 synthetic 提示，不是硬件性能预测。
- `data_deps=(u,v,distance,gap)` 表示
  `start[v]+distance*II >= start[u]+gap`；负 distance `-b-1` 引用 buffer b 的版本数。
  gap 是请求已经明确给出的最小间隔，不能由 solver 猜测未建模的真实 effects。
- modulo 提供有限的 inclusive II 域、buffer 的物理字节数、容量组、版本上限和
  可选 manual stages。寄存器 phase 使用独立的 `(u,v,wrap,gap)`，wrap 仅 0/1，
  不能把寄存器状态当作普通 buffer 多版本化。

结果区分 SAT、UNSAT、TIMEOUT、UNKNOWN、MISSING_DEPENDENCY 和 ERROR。
非法请求单独抛参数错误；缺少 Z3 与安装损坏分开处理，不返回未验证的顺序调度。
UNSAT 仅表示给定有限域内所有候选均不可满足。搜索遵守全局 deadline；不能在带
int64 完成时间上界的问题上假定 II 可行性单调并用二分跳过候选。

SAT 返回 raw starts、II、order 和 buffer_versions。modulo 顺序按 phase 升序、
stage 降序、源码顺序恢复；不隐式归零而破坏 manual stage 的意义。
独立 verifier 不调用 Z3，不复用 solver 的约束构造函数，重新检查覆盖、整数域、
依赖、从 mask 重建的资源互斥、容量、PIPE/stage 和寄存器 phase/tie 规则。
只有验证成功的候选才能返回 SAT；这些数字结果不代表物理多版本或 wave 执行已验收。

`ascend_schedule_solver_ffi.register_solver()` 通过旧 `tvm.register_func` 显式注册
`tl.ascend.solve_schedule`，严格转换 TVM 数值 containers，拒绝符号表达式和非字符串
字段名；不自动从 Task IR 求解，不覆盖已有用户回调。legacy 初始化不探测或导入 Z3。

### 12.4 验证分层

三个独立入口分别验证纯 solver、原生 Task IR 和纯 host capabilities；共同使用
S1 runner 的退出码、JUnit/JSON、缺依赖与超时约定，S2 报告显式记录 step。
原生入口在新 C++ 重建完成后核对源码身份和实际库加载路径。缺依赖、未知结果、
受控超时、损坏候选解和未执行必跑用例均不能冒充 PASS。
本步没有新的 NPU 执行路径，S2 结果不得被描述为新调度器的 A3 设备验收。

## 13. S3：兼容循环接口与受限 Task/Stage

### 13.1 参数与注解合同

`Parallel(*extents, coalesced_width=None, annotations=None)` 保留既有逐维
coalesced_width，只把新增 metadata 放在最外层。map 与参数给出相同 width 可用，
冲突时拒绝。`Pipelined` 的八个既有参数不变，在末尾追加 keyword-only annotations；
native 注册接受旧 8 参数与新 9 参数。原有 stage/order/sync/group/cross_interval
的生成方式、num_stages=0 和原有负值的禁用行为不变。

`Persistent(domain, wave_size, index, group_size=8, num_stages=0, *, annotations=None)`
保持第四位置参数的意义，native 接受旧 4 参数和新 6 参数。所有 annotations 均复制后
合并，不修改调用者 map。Python 和 native 都拒绝保留字段注入：canonical pipeline
控制（tl_pipeline_order/stage/sync/group、tl_cross_interval）、software_pipeline_*、
Task/Stage、enable_offset，以及当前
不支持的 parallel_loop_layout；普通 metadata 不获得额外调度语义。

### 13.2 Persistent 的实际调度域

末维非整除 group_size 时先 padding 调度域，再同时检查还原后的有效坐标与末尾
partial wave。guard 不只保护最后一个 wave，还排除中间 group 的 padding 空洞。
保留静态正参数、index 范围和 partial-single-wave 的已有保护；负 num_stages 拒绝。

domain 乘积、group/worker/wave 运算先提升至 int64，ceildiv 使用商加非零余数，
避免先做加法溢出；对外坐标显式转回原 domain dtype。静态超出 int64 的乘积、
最终索引及 uint64 域明确拒绝。符号值的正数、运行时 index 范围及 int64 可表示性
仍由调用者保证，不声称支持动态零域或任意大整数。

正 num_stages 仅接受可证明恰好一个完整 wave、末维 group 整除、无需 guard 的
情况。partial、非整除、多 wave 或无法证明上述条件的符号输入明确要求先使用 0；带 guard 的
wave 流水及跨迭代多版本继续归 S5。单个完整 wave 的注解和设备测试不证明这些能力。

域测试解释 Simplify 后实际生成的 For/If/Let/BufferStore，而非重算一遍调度公式；
逐个 core 检查全部有效 tile 恰好一次、无越界。静态大域 `[65536, 65536]` 单独检查
生成 IR 的位宽、extent 和首尾坐标，避免枚举数十亿个 tile。

### 13.3 标记的前端、验证和消费

`Task(latency=None, ii=None)` 与 `Stage(0)` 使用当前 TVM attr builder。非空 cost
override、非零 Stage、bool stage 和 PerCoreTask 尚未开放；公开循环接口拒绝
enable_offset，Stage(0) 也不得与手造 IR 的该注解混用。
Task 首片只包含一个 S2 白名单内的 f16/f32 Vector call：单行完整对齐 GM↔UB、
managed fill 或 add。复合 helper、多个 call、内部循环/条件、嵌套 Task、Cube 和
手写同步明确拒绝；祖先 guard 可保留。Stage 包住完整 Task，不能位于 Task/Parallel
内部，也不能与旧 stage/order 数组或外层正 num_stages 混用。

`AscendValidateScheduleHints()` 放在 LowerAndLegalize 首部，在简化和 Parallel
lowering 前检查结构。它只对有标记的函数记录 pending 事实；后续改写即使删掉
dead marker，仍必须经过后端门禁。`AscendLowerScheduleHints(target, platform)`
放在 OptimizeForTarget 首部，仅在 A3 AscendC/auto 消费标记，然后清除 pending 事实。
原本无标记且无 pending 的 PrimFunc 原样返回，不扩大 legacy 后端拒绝范围，
不加载 solver。

late consumer 为每个 Task 保留祖先 allocation、scope、loop、guard 和 Let 环境，
排除未标记兄弟语句，再调用共用的 `AnalyzeAscendSchedule`；不复制第二套资源分类。
未标记的合法 GEMM/raw 区域不会因此被审计拒绝。通过审计后擦除标记，继续使用旧
planner、injector 与同步器。受支持 Vector fixture 的标记版和原版在 ascendc/auto
生成完全相同的 kernel_source。显式 S2 分析则保留 explicit_task 和 stage=0 元数据，
不含标记时保持原有记录 schema。

### 13.4 验证入口

交付 `s3_check_loop_ir.py`、`s3_check_persistent_domain.py` 和 `s3_run_loop_npu.py`。
native 入口先校验源码 manifest 与实际库/Python 加载身份；设备入口复用 S1 的隔离
worker、真实 Bisheng 编译计数、输出 sentinel 和逐次 golden。S1 用例集和验收控制
保留，共享 runner 的抽取不改变其设备用例。实际结果与复现命令见迁移计划第 9 节。

## 14. S4：单 stage 约束桥接与最终同步证明

### 14.1 显式配置与接入

新增 `TL_ASCEND_PIPELINE_PLANNER = "tl.ascend_pipeline_planner"`，仅允许
`"legacy"` / `"constraint"`，默认 legacy，不自动改用户的同步或内存配置。
constraint 首片限制为 A3 AscendC/auto；A2 保留 legacy，未增加 A2 验收。

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_PIPELINE_PLANNER: "constraint",
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

`PipelinePlanning()` 保持无参接口，C++ pass_func 分发到原 legacy 类或受限 constraint
实现；unknown strategy 明确报错。Python wrapper 在当前 PassContext 内构造时，只对
constraint 懒注册 S2 数字 solver callback。legacy 不导入或探测 Z3。缺 solver、
UNSAT/TIMEOUT/UNKNOWN 等状态不能回退成一个未经验证的顺序结果。
正常编译链仍是一次 PipelinePlanning、一次旧 InjectSoftwarePipeline、一套旧同步。
策略作为 pass_configs 的一部分进入已有缓存身份，不另造一套设备缓存。

### 14.2 候选与旧 injector ABI

只选择显式 `num_stages=1` 的静态 serial 循环，至少一个候选；原 loop var 限 signed
int32/int64，半开终点和更新必须在实际 dtype 中可表示。普通/零 stage 的
静态循环作为顺序容器。全内核只接受经 S2 审计的完整对齐 GM↔UB、Vector fill/add，
拒绝混合 C/V、opaque/raw、标量 buffer effects、手写 flag/barrier 和用户提供的
stage/order/async/sync/group 等冲突控制。多 stage/version 留给 S5。

UB allocation 必须为静态 f16/f32。规划前按原生可用预算 196352 字节做保守总量检查，
分析之前先对真实 UB 声明逐维做除法前置检查，再形成乘积，防止 TVM 常量乘法回绕。
总量检查计入 Flatten2D 的行对齐；显式地址 placement 在此首片拒绝，避免把超量或碎片化
请求送入旧 allocator。最终验证再次检查每个 UB 地址/大小及访问区间均落在该预算内。
这沿用旧 allocator 的实际预算常量，不新增一份与原生实现分离的容量定义。

候选 unit 采用共用的 `CollectAscendPipelineGroups` 顺序，且每个 unit 恰好一个
受审计调用。循环变量保留为符号索引，构造单迭代保守 RAW/WAR/WAW；不把不精确
区间当作无依赖，也不将循环变量替成零后猜测独立性。S2 的 single-iteration 图
不等于已经证明循环回边，回边由最终验证另行检查。

调用 S2 straight solver，以单位 latency/II 表达排序成本，不把它当作 A3 硬件时延。
native 消费结果前检查 SAT、完整整数排列和候选的全部单迭代依赖；不消费 start_times，
不另声称验证了硬件时延。标准 S2 callback 的数字结果另有其独立 verifier。发出
`software_pipeline_stage=[0,...,0]`；order 数组为“原 unit → 执行 rank”，
所以 solver 执行序 `[2,0,1]` 对应旧 ABI `[1,2,0]`。不发 async stage，不增加
version 轴，不把任务移动到其他迭代；硬件既有的异步完成关系仍由旧同步和最终验证证明。

### 14.3 跨改写的任务身份

内部 `tl.ascend_constraint_task_id` Attr 的 node 为 int64 unit ID，value 为 int32 1；
`tl.ascend_constraint_iteration` 的 node 为 int64 loop ID，初始 value 为 int64 迭代表达式；
允许后续合法整数缩位，最终以静态数值匹配原迭代身份。
迭代表达式必须放在 value，当前 TVM Substitute 不改写普通 Attr 的 node。
因此 injector、unroll 和索引重写后仍能识别实际任务实例。

最终映射同时核对操作/PIPE、pointer dtype/mask、逻辑读写区间及非 pointer 的
标量/字符串参数，包括 count 与 fill 值；允许合法整数缩位。首片拒绝跨 dataVar 的
逻辑 effect remap，不从相同字节 footprint 推导 payload 相同。

函数级 `tl.ascend_constraint_plan` 保存 schema、原始 source_graph、含相同内部标记
但不含 manifest 的 source_function 快照和候选的 unit/source_path 映射。快照不
构成循环引用。这些是编译器内部合同，不是新公开调度注解。

### 14.4 最终协议与依赖覆盖

`AscendVerifyConstraintSync(target, platform)` 仅由 constraint 分支调用，位置为
最后一次 Simplify 之后、AscendResourceScopeVerify 之前；它只读验证，既不插入
事件，也不修改最终地址。Selection/MaskLegalize 的末端顺序继续保持。

依据稳定任务身份和 source_function 识别原 logical RAW/WAR/WAW 方向；依据最终
address_map/size_map 重建物理区间，物理 reuse 边按实际发射方向检查，两者不能
互相替代。GM 参数未知别名保持保守，不能要求普通 kernel 凭空具有 noalias。
有限静态展开检查相邻迭代与退出到尾部；每条 trace 最多 4096 个硬件任务实例、
65536 次控制访问，单 loop extent 不超过 65536。超过上限或无法证明的控制流明确拒绝。

具体而言，copy/add/copy 静态循环在不同 GM 参数可能别名时，上一轮输出可能覆盖
下一轮输入，旧同步未必提供 MTE3→MTE2 的完成关系；此时 constraint 必须拒绝。
设备 fixture 显式声明 tir.noalias=True，并由三个独立 torch 分配实际满足该合同；
native 另保留 noalias=False 的拒绝测试，不把设备分配事实推广给普通未知调用者。

按时间顺序维护完成集合：set(src,dst,id) 捕获当时 src 的任务和已继承完成关系，
wait 精确消费对应活跃 token 并传递快照；barrier 只覆盖它之前的相关任务。
检查重复 set、无来源 wait、活跃槽冲突和退出 drain。模 8 的 flag 分配不能代替
liveness 证明，同 PIPE 也不能笼统当作依赖完成，相关 V→V 必须有显式完成证明；
旧生成器通常使用 PIPE_V barrier，合法且按时间顺序的事件传递也可构成证明。
旧 barrier 不会自动保护其后才出现的 writer。

设备尾块采用完整、对齐的短 tile：160=64×2+32，动态 padding DMA 不在本片白名单。
独立尾 allocation 不一定发生物理复用，因此实际复用另用 native 定址正负例验证。
三个入口及真实结果记录在迁移计划；不从 TIR 验证或一次 smoke 推导设备验收通过。

## 15. S5：校准成本、多 stage 与物理版本

### 15.1 成本模型和缓存

`TL_ASCEND_PIPELINE_COST_MODEL = "tl.ascend_pipeline_cost_model"` 接收完整 canonical
Map；旧 TVM 配置系统内部以 DictAttrs 承载，用户无需构造 Attrs。纯主机工具
`load_ascend_pipeline_cost_model(path, expected_identity=...)` 返回已校验内容，
不是把路径作为缓存身份。内容包括 schema/model/unit、实际平台/SoC/CANN/Bisheng/
有序 flags、逐 op/dtype/elements/bytes/rows 的有效成本、原始样本及方法证据。
排除 profile_sha256 本身后对 sorted JSON 求 SHA256；所有内容参与已有 cache key。
相同路径的新内容不能命中旧模型；未知几何、未通过校准或单位不匹配时不退回单位成本。

模型为 `a3-effective-op-v1`，单位 ns，带显式正整数 quantum_ns。latency/II 是 msprof
测得时间随操作数 N 的斜率上界，经量子向上取整得到的整数成本，包含 issue、同步和
有界 arena drain 开销；不是纯指令延迟，不使用 SYS_CNT 名称猜测 core-cycle 频率。
独立最终依赖/事件/地址证明不依赖成本估计准确，性能结果不预设收益。

校准先做逐 N 准确性，再以统一 BasicInfo-only、预热、采样和缓存政策测量 dependent
fenced chain 与 independent batch。多 N/多 round 及 hierarchical bootstrap 纳入
round 漂移，要求 CI 下界为正、相对半宽≤20%、相对残差≤10%；噪声失败不 clamp 成 1。
ns/整数成本与原始 CSV/journal 对应，PMU 内部 replay 不当作独立应用 launch。

校准可使用显式搬移快照：native manifest schema 2 标记 relocated_verified_build，
source snapshot 相对内容与原 schema 1 完全相同，三库、原 cache/log/manifest 均按原字节
复制核验，保留真实原 CMAKE_HOME_DIRECTORY 和 build command，不冒称新构建。
这样运行中的校准不会加载随后被 S5 链接覆盖的库；普通 schema 1 的严格检查保持。

### 15.2 规划与真实版本 ABI

num_stages 1/2/3 为版本搜索上限，不等于实际 N，也不是 manual Stage 的上限。
S4 单 stage、无 profile 的 straight 路径保留；多 stage 使用匹配模型和 modulo solver。
独立纯 verifier 通过显式 FFI 桥检查消费的 starts/II/资源/依赖/版本和容量，不导入 Z3；
legacy 不自动注册 solver/verifier。实际版本修订后再次验证，不仅验证原数字候选。

只有固定区间、整 buffer、单行连续且每版字节 32B 对齐的 loop-private UB 可重命名，
每次迭代须 full-write 后读；该首片不接受候选使用外部只读 UB 或共享写 UB。
read-before-write 和跨迭代 carry 不擅自多版本化；本迭代 full-write 后的 in-place
full-read/write 仍须通过同一 producer/slot 证明。这不扩展为 S6 的通用 owner/epoch 分析。

planner 与旧 injector 共用 required-N/shape/stride 推导。constraint-only override
不得小于旧 required N、不得超过搜索上限，真实 N/物理 shape/stride/elem_offset/
selector 及分配字节写入 remap 合同。真实 padding、对齐和所有版本的总量先过原生预算。
Pointer 与 BufferLoad/Store 的 selector 原点统一为 logical_iteration-min。
不从 solver.buffer_versions 或请求数单独推导实际物化结果。

已实证旧 injector 在 trip1、stage[0,1,2] 重复发射中间 unit。分段修复保持 prologue、
steady、epilogue 连续不重叠，steady 不产生负 extent。实际生成谓词的解释测试核对
0/1/N、trip 小于 stage 跨度和非零 min；规划前核验 staging horizon 与整数域，防止
显式 unroll 后才发现资源爆量。独立 staging horizon 为 64，窗口×unit 控制预算为
65536、实际任务展开预算为 4096。既有 L1 分形 tail/stride 继续走 legacy 回归。

### 15.3 Stage、offset 与短循环

Python Stage 接收非负 int32 literal，bool 拒绝。非零 Stage 仅在受支持 constraint
候选和 enable_offset=True 下消费，legacy 编译仍拒绝；Task latency/ii override
继续拒绝，由显式校准模型提供成本。Stage 约束源程序的排程，不重解释 read 的语义；
与原 imperative 数据依赖矛盾时必须 UNSAT，不能按 stage 重新排序源值绑定。
出现任一显式 Stage 后，未标注 unit 按既有完整 manual_stages 协议视为 Stage0；
单独 Task 而无 Stage 不冻结自动 stage 选择。候选外非零 Stage 明确拒绝，Stage0 可按
普通顺序区域等价消费，不能静默擦除未实现的排程要求。

仅 Pipelined annotations 解禁 bool enable_offset，由 planner/solver 消费；旧
injector 不读取它。多 stage 自动模式默认允许 offset；显式 False 固定 stage 0。
Parallel/Persistent 仍拒绝注入该字段和内部 loop/version/guard 标记。
early prepare 在 Simplify 前记录候选意图，检查 0/1-trip 事实没有静默丢失。
具体设备案例以实际展开 IR 为准，折叠循环不声称运行不存在的 prologue/epilogue。

### 15.4 最终数据与配置状态证明

对真实 N>1 的 UB，不再照搬跨不同 slot 的源 WAR/WAW：源 read 必须有唯一 reaching
full writer，最终同 slot 的最近 writer 必须匹配该源 producer 的 task/iteration 身份。
去除已验证的版本基址后再比较逻辑区间，不能仅放宽 S4 的 SameTask 范围检查。
GM、非重命名 buffer 及 N1 仍保留原依赖方向；实际同 slot 物理 hazard 继续要求完成关系。
source/final 按静态 core/vid 分别解释，事件域隔离；不把每核证明称为跨核握手证明。

managed Vector mask 有真实 mode/lo/hi effects，不能随 buffer version 复制。
最后 Selection/MaskLegalize 按实际调度 IR 修复，Legalize 内部末端增加 constraint-only
只读 MaskContractVerifier，复用 ContractOf/requires/ensures/入口未知和 must-facts。
每 terminal 检查 reaching mode/payload，并按发射次序应用 ensures，不能让后一 setter
提前覆盖前一 reader。检查器不 Repair、不插额外 barrier，不在末端增加 TIR rewrite。
未审计的 deq/atomic/HF32/MMAD、opaque 和条件 setter/reader 继续拒绝。

### 15.5 Guarded Persistent 与验收

新正 stages 的 guarded Persistent 由内部 marker 区分。有效谓词同时排除 padded
group 空洞与末 wave overshoot；正流水改用纯 If-noelse，不把 loop_break 移入某 stage。
每 unit 的 guard 随同一 logical iteration 替换，实际 post-injector IR 核对每 worker 的
有效 tile 恰好一次、无 invalid 读写。num_stages=0 的 legacy 路径保留已有保护。

对旧 TVM 无法直接证明的 guarded 仿射范围，可由 Analyzer 的精确 ModularSet 将
`start+span<=capacity` 归一化为整数商的等价上界，再要求 CanProve。此路径保留
非负 start、正静态 span/capacity、规范 residue 与算术可表示性检查；未知仍拒绝。
例如 `4*wave+core<9` 下的 768B 步长，不以各轴独立区间的过宽上界误拒绝，
也不放行移除 guard 或真实越界的访问。

交付 s5_check_versions_ir.py、s5_run_pipeline_npu.py 和
s5_run_persistent_pipeline_npu.py。分别保存实际版本/容量/事件/配置故障注入、
0/1/N/短 trip 与 1/2/3 物理版本、静态短尾、guarded 单/多 wave、重复调用和同条件
msprof 性能记录。最终新构建 manifest、实际结果与剩余边界记录在迁移计划，未验证不记 PASS。

## 16. S6：alias-aware owner 与完整 epoch 证明

### 16.1 职责与接入位置

本步继续使用一次旧 injector 和一套旧同步生成器。增加 storage/view 规范化与完整
readiness/reuse 证明，不引入 S7 事件生成器，legacy 默认行为保持。

新增 `AscendCanonicalizeStorageAliases()` 是 constraint-only 的幂等变换，设置两个
检查点：LowerAndLegalize 最首、hint syntax/TileOp 降低之前；OptimizeForTarget 策略
校验后、AscendLowerScheduleHints 之前。前者保留并统一输入 alias，后者收拢 lowering
产生的形式并维护 shape 元数据；校验通过且无需任何规范化变化时返回原 PrimFunc
identity。所有地址、buffer 和 pointer 改写位于 MemoryPlanning 前，不增加规划/注入次数。

PipelinePlanning 内先构造 canonical effects/owner catalog，再调用
`BuildAscendLifetimeContract(marked_source, canonical_graph, canonical_catalog)`，
在求解和注入前证明源程序的完整定义关系。现有最终 AscendVerifyConstraintSync 位置
保持：最后同步相关 Simplify 后、ResourceScopeVerify 前；新增 schema3 物理 owner/
epoch 检查并复用事件 completion closure。Selection/MaskLegalize 尾部边界保持。

两个显式只读接口采用旧 FFI，不调用 solver、不修改输入 IR：

- `AnalyzeAscendStorageOwners(PrimFunc, platform="A3") -> Map`：storage、allocation site、
  view、pointer alias、字节偏移与物理元数据。
- `AnalyzeAscendLifetimes(PrimFunc, platform="A3") -> Map`：实际控制路径的 activation、
  producer 区间、readiness 和 reuse 期望。

constraint 缓存身份加入内部固定的 schema3 编译合同 ABI，不能由 pass_configs 覆盖。
现有 key 的 git HEAD 不区分同 HEAD 下的 dirty C++ 升级，必须隔离旧 schema2 的
memory/disk 产物，避免绕过新 readiness；legacy key 不添加该字段，字节身份保持。

### 16.2 Canonical storage 与 view

本节 owner 指 allocation/版本的生命周期归属；操作的 C/V/PIPE 归属仍复用既有资源分类。
storage 以 allocation/data Var 身份及已证明别名关系归并；名称不证明相同或独立。
catalog 的 storage id 对齐 canonical graph.storage_id，保存 canonical data、
source_data、physical capacity、allocation_site_id 与 allocation 的 parent_loop_ids。
alias 项另记 source aliasVar、storage_id、int64 byte-offset Expr 及 binding scope。
每个 view 保留 dtype、shape、stride、elem_offset 和相对 storage 的字节范围。

已知纯 Let 的 dataVar/access_ptr 别名规范化为 root dataVar 加一次字节偏移；消费
位置按 pointer dtype 换算元素偏移。unknown/opaque escape、不整除offset、越界或
无法证明的范围拒绝。Block、BufferMap、alloc_buffers、match_buffer 及元数据 keys
随实际改写更新。同名异 Var 仍为不同 owner；必要时为旧 name-based 消费器生成唯一
内部 root 名称，保留 source_data 身份，不以字符串替代别名证明。

多版本按 canonical dataVar 确定统一 N 和 byte pitch。所有 dtype/offset view、raw
pointer、BufferLoad/Store、Allocate 和 remap 使用同一物理步长；容量包含 owner 的
完整 footprint、对齐和 padding。最终核对 actual N、selector、pitch 和真实分配。
本步不新增精确 placement 的公开 API；既有 annotate_address 保留字节 hint 与向上
32B 对齐的 legacy 语义，允许已证明生命周期不重叠的存储复用。最终 address_map/
size_map 的物理重叠必须进入 producer/reuse 校验，actual N×pitch 完整进入地址预算。

### 16.3 源程序的完整定义证明

不能直接迁入上游 storage-granular write-first/AssumeForExecutes 启发式：部分区域
写入不代表完整 read 已定义，zero/skipped 循环不产生 producer。

- Task instance：launch/core/vid、布尔 case、unit ID、全部祖先 loop ID/iteration。
- Storage activation：同一 case、allocation site、该 allocation 的词法祖先迭代。

source walker 只解释已审计控制和分配，effects 复用 canonical S2 adapter。
`MarkAscendScheduleSource(PrimFunc, graph)` 复用原 StableTaskMarker 的稳定ID算法，
供 planner 与显式分析接口使用。每个 activation 维护 byte segment→source producer。
task 先检查所有 reads，再应用 writes；in-place 不能用本次写入补齐自己的输入。
read 全范围必须有定义且无 hole；N1 可由多次部分写的 union 完整覆盖。
原 AST 的整数运算在 folding/substitution 前逐步检查 int64 可表示性，原 int32 内部
索引算术仍按既有 int64 提升；显式缩位必须可表示，不能用回绕后的常量证明范围。
同一 task 的实际 R/W 区间只接受完全同址或不相交；非同址的部分重叠不按单次调用
就假定 overlap-safe，多 repeat helper 的内部覆盖顺序尚无该保证。此项同时检查 source
alias 与最终物理 alias，不限制不同 task 的已证明 partial-write union。

外层 allocation 的 N1 状态跨 inner/连续循环保留；体内 allocation 每次 entry
重新为 unknown，不能借上一 activation。支持完整初始化后的 readonly UB 和 N1
carry。private N>1 仍要求 owner 内完整定义且无 carry，不自动多版本化 carry。

有界解释保持每 trace 4096 任务、65536 control visits、4096 launch/case 上限。
支持静态 guard、zero/one/N、nested/连续 serial loop；nested positive pipeline拒绝。
有限 bool scalar 仅用于 predicate 时穷举所有组合并计入 case 预算；不开放未界定
索引、数据相关条件、一般动态整数循环或 while，不推导其设备 ABI 已验收。

### 16.4 最终物理 epoch 与 completion

schema3 manifest 增加 lifetime_contract，保留 source、canonical owner/view 与完整
实例身份。最终重新构造 source producer，不信任单独的 PASS 标记或旧缓存结论。
历史 schema1/2 直接协议验证保留原合同；legacy 不自动新增初始化拒绝。

最终实际访问映射到 MemoryPlanning 后的 physical storage、byte range 和 slot。
每个 read segment 的最近物理 writer 必须对应期望 source producer；不同逻辑
storage 不能因错误物理重叠覆盖仍需读取的值，即使同步配对正确也须拒绝。
producer 完成、所有 consumer/readers 完成、下一次写入和退出 drain，分别由实际
PIPE 顺序、barrier 或 set/wait completion closure 证明。scope 退出不是 barrier，
不能借后次迭代 handoff 补前次缺失；guard/zero 路径不得制造无来源 wait。
mask 合同继续保持每个 Vector 域一份状态。
退出 drain 的静态检查指未消费事件 token 必须为空；不把它扩称为额外证明每个无后继
末尾调用均已硬件完成。kernel 返回与 host 可见性沿既有 AscendC/runtime 合同验证。

### 16.5 验证与交付

先实现最小初始化正负冒烟，再覆盖 alias、guard、activation 与真实物理故障注入。
统一重建并核验 source/库 manifest 后才运行正式 native/device：

- s6_check_owner_ir.py：同 data 多 view、同名异 Var、offset/dtype/raw pointer、
  actual version/pitch/Allocate/address/size/remap；错误 owner 与陈旧元数据拒绝。
- s6_check_lifecycle_guards.py：完整初始化、partial union/hole、read-before-write、
  readonly/N1 carry、guard/zero、nested/连续 activation、缺 completion/退出 token。
- s6_run_reuse_npu.py：沿已有 Vector fixture，合法新路径与 legacy 尽量配对；同 kernel
  重复运行，每轮确定性改变 scratch/input/tile 标签，检查 golden、未触及 canary、
  实际 owner/epoch/remap/Allocate。非法状态仅 native 注入，不发设备。

仅 A3 验收，不重校准 S5 成本或新增性能门槛。结果与边界记入迁移计划，提交推送后
等待审核，不启动 S7。

## 17. S7：A3 事件生成与固定 C/V workspace 握手

### 17.1 选择与生成所有权

新增 `tl.ascend_sync_strategy="legacy"|"a3"`，默认 legacy。它与已有 planner
选择正交，但首片只开放两种明确合同：constraint+a3 沿 S6 已审计的 Vector 路径
替换核内生成器；legacy planner+a3 沿固定、显式 C/V、静态 serial、N1 workspace
路径替换跨核握手。后者保留旧核内同步和 VS 保护，不开放混合流水重排。
未知 strategy、平台/后端不匹配、不完整的同步/内存规划配置和超出合同的输入明确拒绝。

Vector 路径只运行新核内生成器，保留最终独立 constraint verifier。固定 C/V 路径
只由新 handoff 生成器负责 workspace ready/release，CombineCV 不再为这些依赖生成
旧 cross 通知；旧核内/VS 只负责其原来的域。未建模的 VS/标量或 opaque effects
不得因新 strategy 被静默移除。默认 legacy 的 Pass 顺序、缓存身份与生成行为保持。

### 17.2 A3 能力来源与事件分配

本机 CANN9.1 的 `__NPU_ARCH__=2201` 定义 QUE_MAX_EVENT=8，核内 ID 为0–7，
按 core kind、物理 core instance 和有向 PIPE pair 独立分池；先验证该 core 的 HardEvent 合法性，再按
同时活跃区间分配，显式 ID 先预留。有请求但可用池为空、未知/符号 ID、越界、重复活跃占用和
预算耗尽均给出诊断，不能模8复用仍活跃的槽。分配器是生成器实际使用的共享合同，
只读分配入口可检验7/8/9个重叠区间，不能把总 set 数称为活跃峰值。
首片只读请求的 core=1 表示 Vector kind，instance 为非负物理实例（默认0）；
src/dst 为不同的 MTE2/V/MTE3，begin/end 为非负半开活跃区间，可选 fixed_id 和
reserved_ids。其它 core kind/PIPE 尚未审计时拒绝，不能把 C 上无效的 Vector event
当成已生成硬件同步。
空请求列表合法并返回空 allocations/domains，不制造 zero-trip 事件。

跨核采用 A3 mode2，用户池保守取0–10；11–14由 SyncAll 使用，15按当前工具链
保留。硬件16计数器、用户可分配11个、单计数器深度15是三个不同预算。首片只接管
一个 GM channel，固定 READY0/RELEASE1；已有手写 cross 协议全部拒绝，不混合管理或
声称已支持为任意显式协议预留后合并。未知隐式库占用拒绝。不同 mode 不构成独立 ID bank，
wait 在该平台不能按 mode/consumer PIPE 再分域，不假定不同 ID 通知完成顺序。

主仓参考提交采用的 mode4/16-bank 不适用于 DAV_C220：本机编码截断 mode 的高位。
能力依据包括本机 kernel_macros/kernel_event/block_sync 实现和
[稳定 CANN900 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html)、
[CrossCoreSetFlag](https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核间同步/CrossCoreSetFlag%28ISASI%29.html) 与
[CrossCoreWaitFlag](https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核间同步/CrossCoreWaitFlag%28ISASI%29.html)。
不迁入950的VF、spill或容量假设。

### 17.3 最终物理依赖与核内生成

新核内生成器位于 AscendMemoryPlanning 后，读取最终 address_map/size_map、实际
allocation/version/pitch 和调用区间，重建 RAW/WAR/WAW 及物理地址复用边。
早期 canonical graph 继续用于值/epoch 身份证明，不能代替实际物理 hazard。
生成器保持已给定的任务发射顺序，以保守 set/wait/barrier 证明 completion，不承担
新的排程优化。只读最终 verifier 再解释实际事件和任务，检查合法域、配对、活跃
占用、覆盖和退出未消费 token，拒绝元数据与实际地址/事件不符。
多 Vector 实例之间只允许共享只读 GM，或可证明不相交的写入区间；实际跨实例 GM
hazard 明确拒绝。核内事件不能充当跨 Vector 实例同步，该范围不由 S6 的逐核证明推导。

核内 flag 是 binary 状态，Set 不阻塞 source PIPE，Wait 仅在 destination PIPE 执行时
清除状态。IR 中相邻 Set/Wait 不能证明 source 下次 Set 前 destination 已清零。
因此首版每个跨 PIPE Set→Wait 后增加真实 `PipeBarrier<PIPE_ALL>()`，其官方语义是
全部 PIPE 的先前接口完成后才允许后续执行；事件租约到这个 fence 才归还。
最终 S7 独立 lease 检查将 Wait 记为已排队，只有 ALL fence 才 retire 已匹配 Wait 的
token；未消费 Set 不能借 barrier 清除，同 PIPE barrier 不能代替 ALL，未归还即再
Set 或退出必须拒绝。生成计划的 issue interval 不是硬件完成时间。
此保守方案会串行化同步边界，不承诺提高吞吐。依据为官方
[SetFlag/WaitFlag](https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核内同步/SetFlag-WaitFlag%28ISASI%29.html) 和
[PipeBarrier](https://asc.gitcode.com/api/SIMD-API/基础API/同步控制/核内同步/PipeBarrier%28ISASI%29.html)，
原文和本机SDK摘录保留在S7硬件证据目录。

### 17.4 固定 workspace 协议与联合验证

`AscendPrepareSyncHandoff(target, platform)` 在 OptimizeForTarget 第一次 allocation
location 更新后、CrossCorePipeline/CombineCV 前保存受审计的源 workspace 与参与者
身份，拒绝 positive mixed pipeline。`AscendInsertSyncHandoff(target, platform)` 位于
最终 MemoryPlanning、旧核内/VS 后，依据实际 GM Var、字节区间和控制生成协议。
`AscendVerifySyncHandoff(target, platform)` 在最后 Simplify 后、ResourceScopeVerify
前联合解释 C 与两个 V 的实际事件，不把每核独立的 S6 检查称为跨核证明。
三个接口均为旧 FFI PrimFuncPass，legacy strategy 返回原函数。

首片使用真实 cv_1_2 映射、group-local mode2、单 GM channel 和单物理 slot。每次生产后在真实 writer
PIPE 上通知 READY；所有 consumer 在读取前等 READY，并在最后一次 GM 读取完成后
通知 RELEASE；producer 在通知 READY 后立即等待 RELEASE，再执行下一条源任务。
这个保守形式不跨 round 重叠，确保下次覆写前已经消费完成。两个 V 的通知须聚合后 C 才能继续，
C 的通知须覆盖两个 V。正向 C→V 与反向 V→C 均不能由通知名字推导完成关系。
首轮空 slot 不制造虚假 READY/free token；首次写入前不等 RELEASE，末次消费后
producer 自然 drain RELEASE；零次路径不发事件。guard 必须使完整参与者一致地进入/退出协议，
部分 sid 参与、未知动态控制或无法证明的 workspace footprint 明确拒绝。
非空 handoff 必须有 `tir.noalias=True`，由 caller 保证不同 GM 参数的实际存储不相交；
不同 Var/名字本身不是参数地址独立性证明。
Vector GM↔UB rectangle保守限制rows<=4095，以满足本机DataCopyPad blockCount硬件域；
某些UBGM flat分支可转单block也不在首片额外放宽。Cube GM2L1/FIX仍用各自审计条件，
此数不是全部DMA引擎的统一行上限。template、dtype与runtime几何在建effect前严格核对。
全部 GM 副作用还须核对 group/actor 之间的区间：跨 group 的 RAW/WAR/WAW 均拒绝；
同 group 非 channel 的跨 actor 写冲突也拒绝，只读共享与可证明不相交的输出允许。
区间集合只合并真实相交或相邻部分，不用含空洞的 hull 误判交错但不相交的输出。

显式 C/V scope 是同时执行的两类硬件流，可以各自含独立的静态 serial loop。
首片声明固定 FIFO 合同：同一 GM channel 按各 actor 的 occurrence ordinal 配对
round，要求 C 与两个 V 的次数、完整字节覆盖、guard 和 source/final 流内顺序一致。
祖先 loop/activation 身份仍完整保留；不能把整个 AST 的先后遍历顺序当成跨核完成关系，
也不能把独立流的不同 loop Var 名称当成轮次匹配证据。
首片还要求配对端全部祖先迭代的数值坐标序列一致（忽略 loop ID/Var 名称）。
同次数但 C 仅执行偶数轮、V 仅执行奇数轮，或不同 min/嵌套 epoch，不做 FIFO 跨 epoch
重配，必须拒绝；一般 offset FIFO 关系不在此步开放。

最终验证独立重建生产→读取、全部读取→覆写，核对 cid/vid、round/activation、字节
范围、通知的 PIPE 完成快照、计数器有界与退出 drain。存储的 PASS、channel 列表和
早期 logical edge 不是证明。已白名单审计的复合 C helper 仍靠既有核内保护与保守边界，不能
伪装为单 PIPE 操作。并发 stream 测试只用 mode2，不引入 mode0 全局 barrier。

### 17.5 构建、验收与停止点

S7 实现、测试和 runner 分工；唯一 native build owner 在所有运行进程退出后重建，
新 source/SO manifest 冻结并核验后才进行正式 native 与 A3 执行。
三个入口为 s7_check_event_budget.py、s7_check_handoff_ir.py、s7_run_sync_stress_npu.py。
覆盖事件上限与显式冲突、最终物理 alias、ready/release 故障注入、初始化/drain、
guard/zero-trip，以及 A3 长流水、多 core、重复 launch 与两个真实 stream 的独立
输入/output/workspace。设备进程设置整体超时；非法协议只做 native 负例。
双stream验证使用本机公开ExternalEvent，每stream一独立闭门，先wait并提交原目标
batch与done event，确认两侧调用返回且done/stream均未完成后由第三stream释放。
两门在任何wait前构造，done资源提前materialize；前轮目标完成后才复用自动复位的门。
异常路径finally释放并保存live metadata，worker-only关闭PER_STREAM_QUEUE与launch
blocking，不改变目标kernel或任务规模。此验收证明start-gated同时pending提交，
不推导自然提交窗口或硬件执行时间重叠。精确安装版源码与真实单轮验证见计划S7记录。
S5 成本模型复用，不新增性能调优或重校准，不执行 A2 测试。
最终记录实际支持边界、构建身份与结果，提交推送后等待审核，不启动 S8。

## 18. S8：选定有限域的完整自动调度接管

### 18.1 配置和所有权

新增 `tl.ascend_pipeline_planner="auto"`，要求A3 AscendC/auto及
`tl.ascend_sync_strategy="a3"`。legacy默认和constraint bridge保持原合同。
auto整核接管，不静默混合新旧区域，不运行旧CrossCorePipeline、CombineCV、
PipelinePlanning、InjectSoftwarePipeline、AscendSyncInsert或AscendSyncInsertVS。
新materializer负责执行侧分派；新同步器同时覆盖准入任务的核内与跨核依赖。
自动同步和内存规划必须开启；旧自动跨核/VS配置不得另开。未知调用、手写事件、
未经审计的scalar/config状态或超出本节边界的输入明确拒绝，不回退后报告成功。

“完整”表示选定受审计域的调度、版本和同步所有权完整，不表示任意动态TIR或全部
Ascend API均已支持。首片为有界静态控制域、UB/L1的N1/N2、固定N1的单GM channel；
多channel、多slot GM、动态while、950/VF/spill及PTO不在本片范围。
有限展开每函数至多256个任务、65536次控制访问；straight数字协议使用至多62个
实际参与的`(group, actor, pipe)`资源域，紧凑编号后以显式Int64位图表示。这是数字
表示的边界，不是硬件事件池大小或实测资源吞吐；超限明确拒绝。

### 18.2 Pass位置和结构合同

```text
Phase1最早：AscendPrepareAutoSchedule(target)，保留短循环/原始控制事实
  → 既有scope/layout/TileOp/workspace lowering
Phase2 allocation placement
  → AscendAutoSchedule(target, platform)：联合源任务/producer图、版本候选、数字求解
  → AscendMaterializeSchedule(target, platform)：自有有限展开、版本布局、执行侧分派
  → 既有flatten/storage rewrite与AscendMemoryPlanning
  → AscendInsertScheduledSync(target, platform)：唯一实际同步生成
  → final Simplify
  → AscendVerifyAutoSchedule(target, platform)：独立只读最终证明
  → ResourceScopeVerify → InstructionSelection → MaskLegalize → codegen
```

新状态统一使用`tl.ascend_auto_schedule`前缀，主合同`tl.ascend_auto_schedule`含原函数、
storage catalog、逐read源producer、控制域、数字问题/结果和发射实例身份。
`tl.ascend_auto_schedule_buffer_remap`记录实际dataVar、版本数、byte pitch及总容量。
不得发出旧`software_pipeline_*` ABI或让旧injector再次处理同一buffer。
Python/native均保留新前缀；auto产物增加独立compiler cache ABI，旧键格式不变。

### 18.3 新任务图与版本物化

在C/V分流前保留联合源函数，以canonical storage、view偏移和原activation重建
每个read的reaching producer。native有界枚举合法N1/N2候选，先校验实际对齐、
容量及slot复用边，再调用既有straight数字solver和独立整数verifier。
straight输入全为有限distance=0 DAG，不塞入其不支持的modulo/manual-stage字段。
本步使用统一`abstract_issue_step_v1`模型；它只决定合法任务次序，不是硬件ns，
不把S5四类Vector成本与未测Cube/reduce成本混算，也不声称求解结果预测设备吞吐。

新物化器直接消费execution permutation，生成有限实例、UB/L1版本布局及typed
访问重写。只对已证明每activation完整定义、无跨轮carry的owner使用多版本；
carry和GEMM内部scratch固定N1。只删除被实际不同slot消除的WAR/WAW，source read
仍须读取原producer的值。独立跨PIPE任务允许合法预取重排，不给所有原actor任务
追加一条人为源码总序。channel通知端点和共享配置的必需顺序仍须守恒。
必须验证至少一个真实N2及非源码顺序的正例，不能把N1串行旧路径改名作为交付。

caller GM参数不扩容、不静默增加版本，也不改变workspace最终可见值。跨核保持
S7已审计的cv_1_2、完整两AIV参与、单channel/N1 FIFO合同；新backend可复用纯协议
helper，但不调用旧pass。source producer按独立actor/FIFO epoch绑定，不能把跨C/V
AST遍历顺序当作happens-before。

### 18.4 复合调用、物理同步和末端证明

GEMM按不可拆复合任务建模，审计真实MTE1/M及其借用资源、内部事件0/1和SPR行为，
不以OperationConfig单个PIPE_M标签替代。首片为f16输入、f32L0C、完整对齐、
clear=true、n_actual=N、unitFlag=0；HF32/deq/atomic及未审计配置拒绝。
L0A/B内部ping-pong scratch各按真实64KB预算、固定N1，不与外部显式L0任务重叠。
归约首片为已审计的clear=true f32行归约；精确区分dst的must_define、保守
may-access和内部scratch，不以scratch整区写足迹冒充入口初始化证明。

新核内生成器可在原子任务边界使用真实ALL完成fence，隔离复合helper内部flag租约，
但ALL不代替mask/SPR值证明。auto专属末端mask验证扩展归约合同，保持旧constraint
fill/add白名单及历史schema语义。最终验证从原函数和实际IR独立重建任务、
producer/epoch、remap、Allocate和物理hazard；local物理域区分core kind/cid/vid/pool，
GM另作全局别名检查。错误地址、pitch、selector、初始化、释放/drain或mask均拒绝。

### 18.5 交付和实际验收

提交独立入口`s8_check_integration_ir.py`、`s8_run_workloads_npu.py`和
`s8_benchmark_npu.py`，沿用既有manifest、隔离worker、退出码和机器可读报告合同。
native检查新版本与实际重排、source producer/物理复用故障、compound scratch、
mask、GM双向握手、0/1/N与guard/尾输出exact-once，以及旧路径调用次数为零。
默认legacy/constraint及旧Expert另做兼容回归，不用legacy算子通过冒称auto支持。

A3实际运行新路径GEMM、归约和GEMM→GM→Vector/归约，并逐次刷新输入、golden、
canary和workspace终值。benchmark在正确性前置通过后，固定配对输入、flags、
warmup、样本数和计时方法，分别记录compile、host dispatch和NPU Event span。
证据序列化、输入重置及golden在计时区外；不把Event span冒称纯指令延迟，不预设
提速或隐藏回归。最终范围、构建身份、真实结果及限制随S8记录更新。
