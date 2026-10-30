# S2 实现骨架：显式分析与数字协议

确认依据：已批准的 S2 实施方案及本轮“Implement the plan”。本步包括实现、测试、
原生构建与验证，不额外设置骨架确认停点。

## 原生分析

- `AnalyzeAscendSchedule(PrimFunc, platform="A3") -> Map`，注册名
  `tl.transform.AnalyzeAscendSchedule`；不构造 Pass，不接入 `phase.py`。
- `src/transform/ascend_schedule_analysis.cc` 使用只读 visitor 与 analyzer，收集
  task/storage/group/loop/dependency，内部算术先提升 int64，再计算字节半开区间。
- task 对应硬件调用，group 保留 injector 的不可拆直接子语句，loop/guard/原语句引用
  不丢失。ID 按源码遍历生成，storage 依据 data Var 身份，而非名字。
- 共用 `common/ascend_resource.h` 的既有 owner 分类，以及
  `common/ascend_pipeline_structure.h` 的既有 injector 分组规则；workspace 参数继续
  通过 `ResolveOperationConfig` 解析。共享抽取保持 legacy 语义。
- 首片只允许受审计的整块、连续、合法对齐 GM↔UB 与 managed fill/add。未知 effects
  明确报错；可完整分析但尚不能调度的复杂 group 显式标记不可调度。
- 返回版本、平台、tasks、storages、groups、loops、dependencies；保留 TVM 的
  Var/Stmt/PrimExpr，不能返回伪装成成功的部分图，也不生成 JSON 版 TIR。

## 能力与求解

- `ascend_schedule_capabilities.py` 提供纯标准库 A3 profile；区分编译器预算、工具链
  声明和设备验证，跨核预算未知。对齐、容量、路由与核内事件域验证仅用于显式新接口。
- `ascend_z3_scheduler.py` 为独立单文件协议、solver 与整数 verifier。mode 为 straight
  或 modulo；数据是 latency/II、资源、依赖、容量和 phase 等普通整数，不接受符号 TIR。
- Z3 仅首次求解时加载；结果区分 SAT、UNSAT、TIMEOUT、UNKNOWN、缺依赖和错误。
  SAT 保留原始 starts，必须通过不依赖 Z3 的整数 verifier；无未经验证的兜底调度。
- `ascend_schedule_solver_ffi.py` 显式注册 `tl.ascend.solve_schedule`；旧 containers
  严格转成 plain data 后调用 solver，不自动从分析接口求解。
- `setup.py` 只新增可选 extra `ascend-scheduler`，不改变基础安装依赖。

## 消费与生成边界

消费 lowered calls、buffer/view、guard 和循环域；不更改输入 IR、现有 attrs、地址、
pipeline 注解、版本或同步。原有默认配置不变；A2 兼容代码保留，验收仅记录 A3。
S3 循环接口、S4 planner 接管及后续同步生成仍不在本步实施。

## 验证和交付

先做注册/直线图冒烟，再完成作用域、别名、动态范围和错误路径；修改 C++ 后重建
本地库并更新源码身份 manifest。三个独立入口为 `s2_check_solver.py`、
`s2_check_task_ir.py`、`s2_check_capabilities.py`，复用既有薄 runner。
本轮同时交付永久测试，覆盖独立 verifier 拒绝损坏解、受控错误状态、分析只读性、
能力边界和 legacy 隔离。完成后更新迁移记录，单独提交推送 S2 并等待审核。

## S3 实施骨架：循环接口与安全标记

确认依据：用户已明确要求开始 S3，沿用迁移计划的单步交付、A3 验收和审核规则。

- Parallel 保留 coalesced_width 逐维行为，只在外层增加新的非保留 metadata。
  Pipelined 在既有参数后追加 annotations，保持 native 8/9 参数调用兼容。
- Persistent 保留第 4 位置 group_size，追加 num_stages=0 与 annotations，native
  4/6 参数兼容。非整除最后一维采用 padding 域与完整有效 tile guard；内部 wave/
  index 算术提升 int64。positive stages 仅无 guard 的单个完整 wave 可透传，其余明确
  要求使用 0，避免在 S5 前开启未验证的 wave 流水。
- `schedule_hint.py` 使用旧 TVM attr builder，Task 只接受当前单调用 Vector 白名单，
  Stage 只支持 0；未支持 cost、offset、多调用/控制流和旧 stage/order 混用明确拒绝。
- `AscendValidateScheduleHints()` 位于 LowerAndLegalize 首部，防止非法嵌套先被
  Parallel lowering 或简化抹掉；`AscendLowerScheduleHints(target, platform)` 位于
  OptimizeForTarget 首部，在 planner 和 CombineCV 前消费已验证的单调用标记。
  无 hint 时返回原 PrimFunc，不改变原有默认配置和后端行为。
- late consumer 为 Task 保留祖先的 allocation/scope/loop/guard 环境，并复用
  `AnalyzeAscendSchedule` 的审计规则，不因未标记区域的合法 GEMM/raw code 增加拒绝。
  S2 分析实际消费 Task/Stage，保留显式任务 group 与 stage 0，而非忽略未知 Attr。
- 验证覆盖旧调用/FFI、真实生成 IR 的域解释与 exact-once、绕过 builder 的非法标记，
  以及已有 Persistent GEMM/Vector 控制方式下的 A3 数值与重复调用。统一重建匹配源码的
  native 库后才正式运行；三个独立入口交付后更新计划、提交推送并等待审核，不进 S4。

## S4 实施骨架：单 stage 约束规划与最终同步验证

确认依据：用户已明确要求开始 S4；沿用既定单步实现、A3 验收、提交推送后等待审核。

- `PipelinePlanning()` 保持现有 Python/FFI 签名与 phase.py 调用位置。C++ pass_func
  读取 `tl.ascend_pipeline_planner`，只接受 legacy/constraint，默认 legacy；legacy
  继续原有类，constraint 调用独立的受限 planner。Python wrapper 只在 constraint
  下懒注册 S2 solver，不给 legacy 新增 Z3 导入或严格检查。
- 首片要求 A3 AscendC/auto、自动核内同步与内存规划开启、跨核同步和 VS 同步关闭。
  全内核只允许 S2 已审计的完整对齐 GM↔UB、fill/add 单 Vector 操作及静态顺序容器。
  至少一个显式 num_stages=1 的静态 serial 候选循环；手写同步、opaque/raw、混合 C/V、
  多 stage/version、动态 padding DMA 和用户 stage/order/async 等控制拒绝。
- 候选 body 依照共用 `CollectAscendPipelineGroups` 的 injector 顺序编号，确保每个
  unit 恰好一个受审计调用。S2 单迭代的保守 effects 构建 RAW/WAR/WAW，不把其
  `schedulable=false` 或 `exact=false` 直接当成不安全，也不把循环变量替成 0。
  使用 straight solver 与单位排序成本；native 独立复核 SAT、排列和候选单迭代依赖。
  输出 stage 全 0，order 为 solver 执行次序的逆排列；只经过原 injector 一次。
- `tl.ascend_constraint_task_id` Attr 携带稳定调用 ID，`tl.ascend_constraint_iteration` 的 value 携带迭代
  表达式，使 Substitute/unroll 能同步改写。`tl.ascend_constraint_plan` 保存原始 effects、
  storage 与 logical 依赖身份；覆盖候选外的受支持调用和尾部。内部标记不作为新 API。
- 新 `AscendVerifyConstraintSync(target, platform)` 只在 constraint 下接入，位于
  最后 Simplify 之后、ResourceScopeVerify 之前。它不生成或删除硬件同步，读取最终
  address_map/size_map，核对稳定身份及展开 ABI，并验证原 logical 依赖和实际物理
  复用边；逻辑方向来自原 source，物理 reuse 方向来自计划后的实际执行次序。
- 事件检查按时间顺序维护 completion 集：set 捕获当时 producer 快照，wait 精确
  消费对应 PIPE/event ID，barrier 只覆盖它之前的相关任务。检查重复 set、无来源 wait、
  活跃槽冲突和 drain；同 PIPE 的依赖不能无条件视为已完成，V→V 依赖需显式完成证明。
  静态循环展开有明确上限，检查相邻迭代与循环退出到尾部，超限或无法证明时明确拒绝。
- 交付三个独立入口及永久回归。A3 设备使用已有 add 骨架：full192=64×3、
  tail160=64×2+独立32尾 tile；f16/f32 × 两种形状 × 两策略分别冷编译并各运行100轮。
  对比同输入的实际输出，不将独立尾 allocation 自动解释为物理 reuse。物理全/部分
  重叠另做 native 正负例；策略缓存身份调用实际 cache key 验证。
- planner、最终 verifier 与 runner 分工实现，统一重建/冻结 manifest 后串行加载库和
  设备测试。默认值、原 planner/injector/同步器数量和尾部 Selection/MaskLegalize
  顺序保持；S5 的跨迭代流水、多版本与性能优化不纳入本步。

## S5 实施骨架：受校准成本约束的多 stage 与真实多版本

确认依据：用户明确授权 S5；仅 A3 验收，完成后提交推送并等待审核，不进入 S6。

- 保留 legacy 默认、一次 PipelinePlanning/InjectSoftwarePipeline 和旧同步器。扩展
  constraint 的静态 Vector full/aligned f16/f32 copy/fill/add 候选到 num_stages 1/2/3，
  num_stages 是版本搜索上限，不能直接当作每个 buffer 的物理 N 或 manual Stage 上限。
- `tl.ascend_pipeline_cost_model` 接收完整 canonical Map。schema/model/unit、实际 A3
  SoC/CANN/Bisheng/flags、op/dtype/count/geometry、ns 量子、effective latency/II、
  原始样本和方法证据参与 profile hash 与真实缓存身份。没有匹配校准模型时拒绝多 stage，
  不把 950 数据或 S4 单位成本当作 A3 校准值。
- 校准复用 legacy 明确同步的 microbench：dependent fenced chain 和 independent batch，
  多个动态 N、固定 setup/drain、逐次准确性检查后用 msprof op 采样。耗时随 N 的斜率
  包含 issue/同步开销，仅是 effective 成本；正确性仍由最终任务/地址/事件证明保障。
- planner 使用 modulo solver 并独立核验消费的 starts/II/stage/order/versions/依赖与容量。
  Pipelined 的 annotations 仅新解禁 bool enable_offset，由 planner 消费；旧 injector
  没有该功能。多 stage 自动模式默认允许 offset，显式 False 则限定 stage 0。
  非零 Stage 只在 constraint 的完整受审计 Task/unit 上开放，不能先被 late hint 擦除。
  legacy 对新非零 Stage/offset 仍诊断；Task latency/ii override 暂不开放。
- 新 early prepare 在 Simplify 前保留 0/1-trip 的候选意图，避免请求静默丢失；退化循环
  不声称执行了不存在的流水。原生测试另直接解释 injector 生成的 0/1/N/短 trip 谓词。
- 抽取旧 injector 的 required-N/版本布局逻辑，planner 与 materializer 共用。每个候选
  发布真实 dataVar、N、物理 shape/stride/elem_offset/selector remap；constraint-only
  override 不得小于旧 required N，最终实际计划重新核验容量与约束。候选外只读 UB 或
  共享写 UB、read-before-write/跨迭代 recurrence 拒绝；本迭代 fullwrite 后的 inplace 需证明。
- 已实证旧 trip1、stage[0,1,2] 中间 unit 会重复。局部修正 prologue/body/epilogue 为
  连续不重叠区间，body 不产生负 extent；pointer 与 BufferLoad/Store 的版本原点统一，
  回归非零 min、实际展开 exact-once 和已有 L1 分形 tail/stride。
- 最终 verifier 依据原 source 为每个 UB read 绑定 reaching writer，核对去除真实版本
  基址后的逻辑区间，不能照搬 S4 跨不同 slot 的 WAR/WAW。GM/非重命名逻辑依赖保留，
  final 同 slot 的物理 RAW/WAR/WAW 负责复用；按稳定 task/iter 身份解释真实展开。
- Persistent 正 stages 的 guarded wave 只在新受支持 constraint 路径接入。由内部标记
  区分生成的有效 tile 谓词，不能把 loop_break 移进不同 stage；每 unit 保持同一逻辑
  迭代的有效性条件。按实际 worker/展开 IR 证明所有有效 tile 恰好一次、无 padding 副作用。
- 配置寄存器不随 buffer version 复制。deq/atomic/HF32/MMAD/opaque/条件 setter 未审计
  时拒绝；managed Vector mask 有真实 effects，继续由末端 Selection/Legalize 按实际
  调度 IR 修复。在 Legalize 内部末端增加 constraint-only 只读合同检查，复用 requires/
  ensures/入口未知/分支 must facts，逐 terminal 校验 reaching mode/payload 与覆盖顺序，
  不在 MaskLegalize 后增加 TIR rewrite，不为校验虚构额外 barrier。
- 三个独立入口为 s5_check_versions_ir.py、s5_run_pipeline_npu.py、
  s5_run_persistent_pipeline_npu.py。覆盖真实 1/2/3 版本与容量、0/1/N/短 trip、
  静态短尾、guarded 单/多 wave、重复调用，以及相同输入/flags/缓存热度的 msprof 性能。
- 校准在已验证的 S4 source/libs 搬移快照中运行，保留原构建 cache/log 字节与身份；
  workspace 的 S5 C++ 可独立应用和构建。正式设备入口仍等成本模型及新 native manifest
  都验证后执行，所有未执行项保持未验证状态。

## S6 实施骨架：canonical storage 与完整生命周期 bridge

确认依据：用户明确授权S6，沿用实现、独立测试、A3验收、计划更新和提交推送后审核。

- 新 common/ascend_storage_alias.h 与 ascend_storage_alias.cc 以 dataVar/allocation
  身份构造catalog，保存view的dtype/shape/stride/elem_offset/byte range。
  AscendCanonicalizeStorageAliases() 使用既有StmtExprMutator模式规范化已知Let
  dataVar/access_ptr别名；unknown/escape/不整除offset/越界拒绝。无需任何规范化变化且
  校验通过时返回原func。
  同名异Var在必要时唯一内部命名，同时维护body/BufferMap/Block.alloc_buffers/
  match_buffer及metadata keys；readonly AnalyzeAscendStorageOwners(f,"A3")注册旧FFI。
- phase.py仅constraint在LowerAndLegalize最首、HintSyntax/TileOp之前，以及
  OptimizeForTarget策略校验后、LowerScheduleHints之前设置幂等规范化检查点。
  wrapper/FFI签名一致，不增加planner/injector/同步生成次数；所有地址改写在MP前。
- ascend_schedule_analysis.cc复用canonical catalog；旧materializer在inject_pipeline.cc
  与common/ascend_pipeline_versions.h按dataVar统一N和byte pitch，对所有dtype/offset
  view、raw pointer及真实Allocate/remap同步改写。
- ascend_infer_buffer_scope.cc的ScopeCorrector仅在constraint分支按Buffer对象缓存view，
  保持各dtype/offset/stride，并用RemapAscendStorageMetadata同步scope替换；Block的
  reads/writes/match/init、address/layout keys使用同一Buffer映射，legacy沿原分支。
- 新common/ascend_lifecycle.h、ascend_lifecycle_analysis.cc的source walker复用canonical
  S2 effects，只解释控制和词法分配。公共只读FFI为AnalyzeAscendLifetimes(f,"A3")->Map；
  内部BuildAscendLifetimeContract(marked_source,canonical_graph,canonical_catalog)->Map。
  MarkAscendScheduleSource(f,graph)抽取原StableTaskMarker算法，planner/分析接口共用。
  planner在Solve/Inject前消费source完整定义证明，task所有reads先于本task writes核验。
- storage activation含allocation site及全部词法祖先迭代；task instance含unit、
  launch/core/vid、全部祖先迭代及有限bool case。byte segment追踪producer；zero/skipped
  不制造epoch，N1支持已证明的partial-write union、readonly与carry；privateN>1要求
  完整write-first且无carry。nested positive pipeline仍拒绝，有限case/trace预算保持。
- schema3在原manifest加lifetime_contract。ascend_constraint_sync.cc在原finalSimplify后
  位置重建source期望，匹配actual slot/range最近writer并复用completion闭包，分别核验
  依赖中的producer、所有readers及下一写完成关系，以及未消费event token的退出drain。
  不把drain扩称额外证明所有无后继末尾call硬件完成。历史schema1/2 fixtures保留原合同。
- 原AST在folding/substitution前逐步checked int64；同Task的source/physical R/W仅
  exact-or-disjoint，保留exact inplace和不同Task partial-write union。
  原式范围与byte乘加由PromoteAscendStorageIndex、CheckedAscendStorageByteOffset/
  AddAscendStorageByteOffsets守护；source/final共用EvaluateAscendLifetimeInteger和
  ValidateAscendLifetimeCallIndices，不拿已回绕常量作范围证书。
- 不新增精确placement API；旧annotate_address语义保持，最终address_map/size_map和
  actual版本占用进入别名与reuse验证，不用名称证明storage独立。
- kernel_cache.py的实际_generate_key仅对constraint加入内部schema3 ABI盐，不能由
  用户覆盖；覆盖effective策略与memory/disk路由，legacy key同输入字节保持不变。
  存储分析私有function事实的tl.ascend_storage_前缀在Python/native loop builder均保留。
- 实现与独立测试分工，先最小source初始化正负冒烟，再扩至alias/guard/activation。
  构建由唯一owner负责，链接期间不加载workspace库。新manifest冻结后运行owner、
  lifecycle两个native入口及reuse A3入口，记录逐轮随机scratch/golden和真实IR证据；
  更新计划并提交推送后停在S6，不进入S7。

## S7 实施骨架：A3新同步生成器与固定workspace协议

确认依据：用户明确授权S7；本节落实pass-design第17节，完成后提交推送并等待审核。

- `tl.ascend_sync_strategy` 注册String，值legacy(default)/a3。root维护Python
  pass_config、phase、wrapper和缓存ABI；`a3`路径加入固定内部同步ABI盐，legacy
  默认key字节保持。constraint+a3只开放已有受审计Vector域；legacy planner+a3
  只开放固定explicit-C/V serial handoff，未覆盖输入明确拒绝。
- 新核内生成器由`ascend_sync_a3.cc`及必要common header承载，复用
  ascend_constraint_sync的最终任务/物理区间解释；构建入口
  `AscendInsertSyncA3(Target,String)`，旧FFI注册`tl.transform.AscendInsertSyncA3`。
  `AllocateAscendSyncEvents(Array<Map<String,ObjectRef>>,String)->Map`为共享分配器的
  显式只读入口。记录有向core/PIPE事件域、活跃起止、显式预留和分配结果；不改输入IR。
  生成保守完成关系，不改变现有planner/injector次数或任务排序。
- allocator按A3合法core/有向pair验证、预留显式槽后分配0–7，禁止活跃冲突与模复用。
  每跨PIPE Set→Wait 后增加真实PIPE_ALL fence，租约到ALL才归还；独立S7检查器不能
  把IR中的Wait当成已执行的清零。未wait token不借ALL清除，同PIPE barrier不能代替ALL。
  新local只在constraint+a3负责已审计域，phase跳过旧SyncInsert；VS未覆盖效果必须
  拒绝或保留原保护，不能无条件关闭。末端仍执行独立AscendVerifyConstraintSync。
- 新`ascend_sync_handoff.cc`和`common/ascend_sync_handoff.h`负责固定跨核路径。
  三个旧FFI/同名Python wrapper均接收(Target,String)：
  `AscendPrepareSyncHandoff`、`AscendInsertSyncHandoff`、`AscendVerifySyncHandoff`。
  Prepare使用StmtExprVisitor审计source/control/参与者，再以StmtExprMutator保留
  身份；Insert根据最终MP数据生成ready/release；Verifier只读重建joint trace。
  helper按source ownership、physical span、event operation分工，不信任单独PASS字段。
- Prepare位于OptimizeForTarget第一次PlanAndUpdateBufferAllocationLocation之后、
  CrossCorePipeline之前；在CombineCV中仅对新cross ownership抑制旧通知。
  Insert位于MemoryPlanning→旧local→旧VS后；Verify位于末Simplify后、
  ResourceScopeVerify前。所有TIR改写仍在Selection/MaskLegalize前；legacy为identity。
- handoff私有function attrs保存稳定source/GM owner/participant provenance，
  经现有lowering传递，最终实际Var/range/guard/round重新核验；缺少/陈旧facts拒绝。
  mode2只支持cv_1_2、N1单slot和完整uniform参与；ready绑定写PIPE，release绑定最后
  GM读PIPE。首写不等待free，末次producer drain，zero无事件；两个V聚合不可省略。
  事件用户池0–10，首片单channel固定READY0/RELEASE1，手写cross协议保守拒绝；
  不混管显式ID，未知隐式占用拒绝，11–15保守不分配。
  不按mode或wait端PIPE拆成虚假bank，不从跨ID发射次序推completion。
- 构建由唯一owner负责；C++修改期间不声称native已通过，链接期间所有runtime退出。
  新manifest核验后先导入/最小lowering冒烟，再由独立测试责任人执行native及三个入口。
  测试包含独立allocator边界、实际IR故障注入和A3长循环/多core/重复/双stream，
  不将legacy CV兼容执行计入新cross握手。A2测试跳过，S8混合规划不纳入。

## S8 实施骨架：有限域的完整调度、版本与同步后端

用户已明确授权S8，设计见技术参考第18节；先完成本骨架，再在本步内实施并验证。

- 配置新增`planner="auto"`、强制`sync="a3"`；legacy/constraint原路径保留。
- `AscendPrepareAutoSchedule(target)`在Phase1早期保留控制/短循环请求。
- `AscendAutoSchedule(target, platform)`位于Phase2 allocation placement后、旧C/V分流前。
  收集联合source/effects/producer，枚举合法N1/N2布局，以straight solver及独立数字
  verifier求解有限DAG，统一abstract_issue_step_v1；不混用S5 ns成本。
- `AscendMaterializeSchedule(target, platform)`直接按execution permutation展开有限
  实例、typed版本访问与core分派；auto不调用旧CrossCorePipeline/CombineCV/planner/
  injector。caller GM固定N1不扩容，UB/L1新版本须有真实N2正例。
- 复用既有flatten/storage rewrite/MP；新`AscendInsertScheduledSync`在MP后独占
  local/cross生成。复合GEMM/归约task在审计完整effects、scratch和状态后，以ALL边界
  隔离；GM保持单channel两AIV READY/RELEASE。
- `AscendVerifyAutoSchedule`在最后Simplify后、ResourceScopeVerify前独立重建source
  producer、实际实例/版本/Allocate/地址及协议，不能信plan中的PASS字段。
- 末端MaskLegalize增加auto专属归约验真，旧constraint schema和白名单不变。
- 主合同`tl.ascend_auto_schedule`与remap `tl.ascend_auto_schedule_buffer_remap`记录真实所有权；
  Python/native预留内部调度元数据命名空间，auto使用独立compiler cache ABI，旧路径不加新盐。

实现文件职责：

| 文件 | 职责 |
| --- | --- |
| src/transform/ascend_auto_schedule.cc | 准入、原任务/producer图、版本候选及solver协议 |
| src/transform/ascend_materialize_schedule.cc | 新finite实例、版本布局/访问重写和core分派 |
| src/transform/common/ascend_auto_schedule.h | 共享typed合同、审计helper和跨文件接口 |
| src/transform/ascend_scheduled_sync.cc | 最终MP物理依赖、ALL边界及GM双向协议生成 |
| src/transform/ascend_auto_schedule_verify.cc | 独立source/final producer/物理/协议证明 |
| src/transform/ascend_vector_mask_legalize.cc | auto专属末端mask证明，不放宽旧路径 |
| tilelang/engine/phase.py | 三种planner与新后端互斥路由 |
| tilelang/transform/__init__.py、pass_config.py | FFI封装、懒solver注册和公开配置说明 |
| tilelang/cache/kernel_cache.py | auto产物ABI隔离 |
| src/ir.cc、tilelang/language/_loop_annotations.py | S8保留字段保护 |

测试与三入口由独立验证任务实现：真实N2/重排、producer和slot fault injection、
GEMM/reduce/mixed、owner互斥spy、默认兼容、A3逐次golden，以及三类真实计时。
不把abstract cost、未执行项、部分通过或旧路径执行当作本步验收完成。
  双stream由两个独立ExternalEvent闭门对齐提交，done/stream均pending后第三stream
  释放并同步；同一live参数dict保留异常现场。该控制只在验证worker内，不改kernel，
  不用mode0或sleep估时，不将pending证据称为硬件时间重叠。
