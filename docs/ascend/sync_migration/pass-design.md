# 主仓 Ascend InsertSync 后向移植设计

## 1. 目标与基线

用户目标：用 tilelang 主仓的自动插同步实现，替换本仓 `AscendSyncInsert` 与
`AscendSyncInsertVS` 的自动同步算法。本文是完整目标设计；当前可选后端已覆盖核内
ordinary/VS、If/else 与运行时快照、受限 UB 流水、typed GEMM、现有 C→V 发布、
Cython fresh workspace 契约，以及受限 Persistent、Persistent+SIP 与尾块 helper 适配。
精确支持范围、实际执行记录及尚未验收项见[实现进展](implementation-status.md)。
S3/S4 尚未整体退出，S5 尚未执行；默认仍为 legacy，没有提交完整替换的结论。

Persistent 适配仅将已验证的规范领先 break 化为等价的每 worker 实际迭代数；保留原
payload、数组词法作用域和所有实际执行的计算，不把 break 当作无副作用操作删除。
它不引入任意 break/While 语义。Persistent 与软件流水组合目前只验收了 4.5 所列
的单 owner 窄范围，不能推广到任意嵌套流水。

技术方向已采纳：移植主仓同步引擎及 Z3 证明核心，通过 TileLang 侧服务适配旧 TIR。
历史调研、路线比较与成本讨论见
[RFC #1971](https://github.com/tile-ai/tilelang-ascend/issues/1971)。
本文保留实施契约和实际组件边界；已采用的来源及适配点见[来源记录](upstream-sources.md)。

分析仅使用当前本地工作树，不参考其他分支或 PR。

| 项目 | 本仓 | 主仓 |
| --- | --- | --- |
| 路径 | `~/workspace/git/github/erhsh/tilelang-ascend` | `~/workspace/git/github/tile-ai/tilelang` |
| HEAD | `43b0b69720f479894a13c02746599eccee5c5769` | `1c0f402e166a7d0a055c991d2f82d2175a7917b7` |
| TVM HEAD | `c2921fdaf795b1103d21abc962e83a209c7258d7` | `4211874e9e4b8770fd20ad0d98ab80afd1df5028` |
| IR/FFI | `tvm.tir`、`TVM_REGISTER_GLOBAL` | `tirx/s_tir`、新 FFI |

本仓 TVM 存在用户本地改动：`python/tvm/tir/buffer.py`、`src/runtime/logging.cc`。
迁移不覆盖这些改动，不升级整个 TVM，不引入主仓 AutoSchedule 调度求解器或运行时；
引入用于依赖证明的 Z3 服务。Z3 的调度求解与依赖证明属于不同用途，不能混为一谈。

成功标准是运行主仓移植后的 `AnalyzeDependencies → SyncAnalyzer → ValidateSyncPoints →
FlagAllocator → SyncMaterializer`，通过适配层对接本仓。仅改名、包装旧算法或保留旧 VS
算法补洞，都不属于替换完成。

## 2. 范围和位置

### 2.1 替换边界

- 替换 `AscendSyncInsert` 和 `AscendSyncInsertVS` 的依赖分析、同步点规划和事件分配。
- 保留已有 `tl.ascend_auto_sync`、`tl.ascend_auto_sync_vs` 的配置含义与有效默认值。
- 汇总普通同步和 V/S 同步需求，由同一个引擎分配事件，避免两次独立分配冲突。
- 保留现有内存规划、C/V 划分、流水展开、codegen 和运行时。
- 保留 `CrossCorePipeline`、`CombineCV` 已产生的跨核同步；将其作为输入约束和资源预留。
  本轮不顺带启用主仓 950 特有的跨 AIV 策略或 `get_buf/rls_buf` 溢出机制。
- 不将“非流水用例通过”表述成完整替换；流水、多版本及 VS 是退出旧实现的必要条件。

### 2.2 Phase 2 位置

```text
InjectSoftwarePipeline（发布流水 provenance）
  → Flatten/StorageRewrite 等现有变换（传播 provenance）
  → AscendMemoryPlanning
  → AscendInsertSync（新增统一入口，替换两个旧调用）
  → final Simplify（现有条件分支）
  → AscendResourceScopeVerify
  → InstructionSelection → MaskLegalize（现有条件分支）
```

当前权威顺序：`tilelang/engine/phase.py:97-138`。
新引擎不修改 buffer 分配、访问、版本数或地址映射，不在同步后增加内存复用。
全部同步调用必须在 `AscendResourceScopeVerify` 前物化；MaskLegalize 后没有 TIR 改写。

当前过渡接入如下：

```python
mod = tilelang.transform.AscendMemoryPlanning()(mod)
if pass_ctx.config.get("tl.ascend_sync_backend", "legacy") == "upstream":
    mod = tilelang.transform.AscendInsertSync(target, platform)(mod)
elif pass_ctx.config.get("tl.ascend_sync_backend", "legacy") == "legacy":
    mod = tilelang.transform.AscendSyncInsert(target, platform)(mod)
    mod = tilelang.transform.AscendSyncInsertVS(target, platform)(mod)
else:
    raise ValueError("Unknown Ascend synchronization backend")
```

过渡期只允许整条同步链二选一。`upstream` 模式不静默回退，也不混用旧 VS。
完整验收后，将默认值改为 `upstream`，删除 legacy 算法及过渡 backend 配置，pipeline
只保留一个 `AscendInsertSync(target, platform)` 调用。

## 3. 主仓来源与最小闭包

以下路径相对主仓，来自上述固定 HEAD。移植文件保留原许可证头；另建来源清单，记录
源路径、commit、blob hash、目标路径及语义适配点。没有直接来源的 adapter 单独标注。

| 来源 | 迁移处理 |
| --- | --- |
| `src/ascend/transform/insert_sync.cc` | 迁移 site registry、依赖闭包、SyncAnalyzer、ValidateSyncPoints、FlagAllocator、SyncMaterializer；替换输入输出和 target 常量边界 |
| `auto_schedule/dependency_analysis.{cc,h}` | 迁移 RAW/WAR/WAW、循环依赖及冲突判断；适配物理冲突查询与 Z3ProofService |
| `auto_schedule/ir_structure.{cc,h}` | 移植 Task/Control 树、guards、parent/index、上下文等必要模型；替换新 IR API |
| `auto_schedule/multi_buffer.h`、`buffer_version.h` | 移植 owner/version/epoch 契约；由本仓流水 provenance 提供事实 |
| `auto_schedule/buffer_lifetime_analysis.h` | 作为来源审计；其 post-sync alias 生命周期工具不纳入当前实现，事件完成与复用证明由同步核心承担 |
| `transform/common/constr_visitor.h` | 适配 loop/Let/guard 约束、mutable read snapshot；不得省略上下文语义 |
| `core_mask.h` | 保留必要表示，将 core 数、bank 和支持关系交给 target policy |
| `auto_schedule/task_analysis.h`、`memory_detector.h`、`ascend_pipe.h` | 用本仓 lowered-call adapter 替换，不引入其高层 op/layout 闭包 |
| `auto_schedule/scheduled_tir.{cc,h}`、`task_annotations.h`、`kernel_rewriter.h` | 不移植 codec；由本仓直接构建树、保存原语句并回写 |
| `auto_schedule/buffer_alias_analysis.h`、`buffer_alias.h` | 不移植同步后 storage-alias 规划及仅供其消费的 context，本轮不改变已规划地址 |
| TVM `src/target/z3/z3_prover_on.cc`、`z3_prover_off.cc`、`include/tvm/arith/analyzer.h` 中的 Z3Prover 声明 | 移植 SMT 翻译、solver 管理、约束 journal 与恢复；适配旧 TIR 和旧容器/hash，不挂接全局 Analyzer ABI |
| `cmake/pypi-z3/FindZ3.cmake`、主仓及 TVM CMake 的 Z3 集成 | 适配头文件/库发现、编译链接和安装后运行库定位；源码清单单独记录构建逻辑来源 |

`auto_schedule/` 表格项均相对 `src/ascend/transform/`。实际构建源文件必须显式列入 CMake；
当前 `src/transform/*.cc` glob 不递归收集新子目录。

切断高层 op 检测和 alias 输出消费者依赖后，仍保留同步核心算法及其覆盖验证。
当前实现不保留没有消费者的 alias-planner 抽象接口和 lifetime 记录。关闭 storage-alias
规划不关闭同步，也不等同于关闭 flag 安全复用；核心依赖闭包、同步覆盖、事件完成与
复用校验仍必须完整保留。
裁剪不得顺手移除 flag 复用安全条件、循环首尾握手或 mutable-read 上下文。

## 4. 输入契约与适配

### 4.1 LegacySyncImporter

入口接受传统 `tir::PrimFunc`、target/platform、最终 `address_map/size_map` 和两类同步
生效配置。直接构建移植的 Task/Control 树，不伪造主仓前端 `tl.schedule_unit`。

- typed 调用由 `ResolveOperationConfig` 提供 pipe、参数读写与 copy resource。
- extern 调用规范化名称后查相同目录；extern 参数索引多出函数名一项。
- 使用实际 `tvm_access_ptr` read/write mask；同一次访问可以同时读写。
- 处理 BufferLoad/Store、Let、For、If、resource_scope 及其约束。
- 每个任务保留原始 Stmt、控制条件和资源归属；适配层不重排计算。
- 保留配置寄存器、scalar 状态和控制表达式读取的依赖。
- 未识别的有副作用操作不能当成纯函数或无访问；在 upstream 模式给出调用名、位置和
  缺失契约的诊断。纯操作必须有明确分类。

### 4.2 PhysicalStorageResolver

主仓 `dependency_analysis.cc:151` 按 data Var 身份排除不同 storage；本仓不同 Var
可能已经复用同一物理地址。直接照搬这一判断会漏同步。

Resolver 根据 Var 身份、scope、target 物理内存域、最终地址和大小建立分析视图。
逻辑 storage/owner/epoch 与物理冲突身份分开保存：两个逻辑 buffer 可以共用物理地址，
但不能因此合并它们的版本数、owner 或时钟。

```text
访问的物理字节区间 = address_map[var] + 元素偏移 × dtype 字节数 + 实际 DMA footprint
```

- 可能相交的存储进入共同的物理 alias group；不以 name_hint 建立身份。依赖分析在
  Var 不同的情况下先查询 PhysicalConflictOracle，不能直接返回不冲突。
- 每条依赖保留 producer/consumer 各自的逻辑 storage、owner 和 epoch；跨 storage
  依赖不得随意采用其中一方的 ring 宽。不能证明两端时钟投影关系时明确拒绝。
- 区分 GM、UB、L1、L0A/B/C；AIV 私有 UB 的身份按 target 物理归属处理。
- 分析视图不改变原 buffer、地址或真实计算语句。
- 区间覆盖对齐、padding、fractal pitch、workspace、版本轴和 view 的实际访问。
- 缺少必需 maps 或无法解释地址单位时明确报错；maps 中不含 GM 参数本身不算缺失。
- 对动态表达式，只有证明不相交才消除依赖；否则扩大为保守区间。
- 不凭参数 Var 不同断言 GM 不别名；需要已有 noalias 契约或保守处理。

### 4.3 流水 provenance

非流水路径可从 source order、stage=0 和物理单版本开始验证。不能对一般流水简单将
`buffer_versions` 清空、stage 全置零就宣称兼容：主仓 distance=-1 的解析依赖最终
ring/owner；旧流水已生成 skew、分段和 guards，逻辑时钟可能丢失。

最终流水支持要求在旧 injector 中发布而不改变物化行为的 `tl.ascend_sync_provenance`：

| 字段 | 含义 |
| --- | --- |
| owner_id / storage_id | 稳定流水 owner 与逻辑 storage 身份 |
| logical_iteration | 原循环逻辑迭代表达式，与物理循环变量的关系 |
| stage / order | 原流水计划以及对应物化任务 |
| physical_versions / slot_expr | injector 实际版本数和索引，而非 solver 预测值 |
| execution_guard | 本任务实际执行条件 |
| segment / original_task_id | prologue、steady、epilogue 与原任务的关联 |

元数据结构使用旧 TVM Map/Array/ObjectRef；随语句与 buffer 改写传播。对会丢失身份的
flatten、storage rewrite、unroll 等位置增加显式传播和验证，不能仅在 PrimFunc 上留下
一份过期表。Importer 将其转换成主仓 owner/version/epoch 输入；已经物化的版本不能
再次扩展。provenance 缺失或失效的流水在 upstream 模式拒绝编译，并给出所属变换。
stage/order、物理版本数和逻辑时钟是不同事实，不能仅凭两仓同名的 `num_stages` 参数
认定流水语义等价。当前只消费旧 injector 的实际版本与时钟，不生成自动 owner 计划。

### 4.4 单次 C→V 发布前的 C 循环（v5 窄范围已实现）

目标仅为已有静态 C-side ordinary/K-loop GEMM 完成后，在循环外执行唯一 mode-2 FIX
Set，再由 flat V Wait/epilogue 消费；不放宽任意 mixed For/If 或循环跨核 channel。

- importer 保留 C Control 树、header S task、子节点及原 before/after 锚点；根据全部
  后代确定单一 C owner，再按 root 投影到各核内图。不能将 C 循环压成单个 summary
  Task 后交给核内引擎，否则会丢失回边和末迭代完成关系。
- 跨核 preflight 另持有精确 real leaf records 与 control 的原语句/occurrence span。
  只接受经内部审计的 static、positive、serial C prefix，无 If、内部 cross 调用和
  未覆盖的 GM 写；typed GEMM 的真实 helper/clear 审计继续生效。
- Set 必须位于整个 C control span 之后且不属于该 control；Wait 和 V 后缀仍保持
  已验证的 flat/B1、C1V2、单发布契约。仅比较 leaf preorder 小于 Set 不足以证明
  循环全部迭代已发布，Set 藏在循环内必须拒绝。
- 真实 leaf 使用各自原上下文做完整 C/V 冲突检查；For/Let 私有变量参与跨 stream
  freshening。只在构造 release/acquire shadow 时使用完整 GM 参数区域，避免将 K
  变量泄露到循环外。
- noalias-free 路径保留 fresh workspace、单工作组、V 单输出及真实 vid footprint
  门禁。CSet.write(C GM 输入读) 与 VWait.read(V GM 输出) 的 WAR shadow 不得删去；
  C 核内原引擎负责证明最后一迭代读完成后才到达 Set。

验收包含真实 Developer K-loop GEMM→exp、输出 alias A/B、Set 在循环内、zero-trip
未初始化发布、C 后续覆盖、V 控制流及未知非 GM 冲突等正反例。已完成的 CPU 和设备
范围见实现进展；它不放宽其他 C/V 循环或多次发布。

### 4.5 Persistent 与已认证内层 SIP 的组合（v6 窄范围已实现）

将已独立认证的 abs SIP 建成保留原内部计算和同步的 compound task，由同一引擎
分析外层 Persistent。实际实现采用 S acquire / S completion：每个 wave 入口
S→V/MTE2/MTE3、出口 V/MTE2/MTE3→S 共六对事件，内层 seed/drain 也在封口内。
早期 MTE2 acquire / MTE3 completion 候选没有作为完成证明使用；不能仅因
seed/drain token 配对就假定全部异步操作已完成。

外层 summary 必须覆盖真实 GM 及全部 UB scratch 的完整物理 span，并保留原逻辑
owner/epoch 和真实版本事实。特别是内层最后一个 MTE3 对 B1 result 的读，必须通过
外层真实 backedge 在下一 wave 的 V 写之前完成。仅把同步过的 inner body 塞回 outer
For 会遗漏这条 WAR。summary 不得冒充某个 slot clock，也不能清空实际 ring 版本表。

内外层显式互相预留封口事件资源；不能让两次独立 allocator 在异步存活区间复用
同一 token。实际输入严格限制 A3 AscendC/auto、FP32 完整 64 元素块、显式 noalias、
单个规范静态多 wave Persistent、内部 UB B2/lead2 与静态 trip≥3，且 ordinary/VS
同时启用。容量、实际 loop exit、原计算计数、真实物理字节完成关系及设备均已验收。
每 wave 额外 12 次 flag 调用串行化外层，内层 SIP 保留，尚无性能收益结论。
它不推广到任意嵌套 pipeline、动态 Persistent 或多 owner。

### 4.6 实际组件与分析状态

| 组件 | 输入与职责 |
| --- | --- |
| `legacy_adapter.{cc,h}` | 消费 lowered call、OperationConfig、资源 scope、指针 mask 和控制树；保存原语句及插入锚点，导入与导出不另做依赖决策 |
| `physical_storage.{cc,h}` | 持有最终 maps 与物理 storage views，提供冲突 oracle；分析 view 不逃逸成真实 buffer，也不修改分配 |
| `legacy_snapshot.{cc,h}`、`upstream/constraints.{cc,h}` | 在原求值位置捕获可变读取，维护 Let、循环范围和 guard 的词法上下文，并送入 ProofAnalyzer/Z3ProofService |
| `common/pipeline_ring_provenance.h` 及对应适配模块 | 校验实际 owner、B、slot、clock 与三段流水事实，保留已物化版本；复合执行模块补充审计后的 acquire/completion 和隐藏事件预留 |
| `upstream/insert_sync.{cc,h}`、`target_policy.{cc,h}` | 运行统一依赖、覆盖验证、资源分配及物化；硬件差异集中在 policy，不依赖旧 SyncInsert/VS 补洞 |

Importer 的 scope、约束栈、父控制节点和原节点映射随一次分析拥有；Exporter 使用
同一次分析保存的 AST 锚点回写。任何事实采集后若对应 IR 改变，须重建分析；失败不返回
部分物化结果。物理冲突身份、逻辑 owner/epoch 和资源归属始终分别保存。

## 5. 引擎与硬件策略

### 5.1 真正执行的算法

```text
导入任务、控制树、物理访问及版本事实
 → 收集同步插入 site
 → 主仓 AnalyzeDependencies / SyncAnalyzer
 → 主仓 ValidateSyncPoints
 → 预留已有显式同步资源
 → 主仓 FlagAllocator
 → 主仓 SyncMaterializer + TargetSyncPolicy
 → 导出旧 TIR 同步调用、验证事件与作用域
```

主仓出处：`insert_sync.cc:2531-2580`。新引擎禁止调用旧 SyncInsert/VS 的依赖决策方法。

### 5.2 TileLang 侧 Z3 证明服务

主仓使用 `Analyzer::z3_prover`，本仓 TVM 没有该接口。采用局部回移方案：在 TileLang
中建立 `Z3ProofService`，复用主仓 TVM 证明器原有的 SMT 翻译、solver 管理、约束
journal、作用域恢复和副作用表达式缓存规则；不修改全局 Analyzer 的布局或 CanProve
调用路径。`ProofAnalyzer` 作为同步分析的调用门面，现有 Analyzer 仅辅助简化和
范围分析，不能替代所需 Z3 证明。

服务适配 `tirx` 到旧 `tir`、新 FFI 到旧 ObjectRef/容器/结构 hash。主仓核心实现
`z3_prover_on.cc` 约 967 行，包含范围不等于移植 diff 或工期。虽然保存了 Analyzer
指针，但未发现调用该指针的方法；这是局部回移的源码可行性依据。
该局部服务现已通过本仓构建与聚焦测试，不能据此认为完整同步迁移已完成。

`ConstrSet` 经 `ProofAnalyzer` 显式传入 Let binding、循环变量范围、分支条件和可变内存读取快照。
每个分析实例拥有独立 solver/context；约束进入和退出必须异常安全，不能把历史假设
遗留给下一个查询。同步所需接口为 Bind、EnterConstraint、CanProve、SetRLimit 和
SetTimeoutMs；保留必要调试查询用于复现证明差异。

构建适配包括 Z3 headers/library 发现、链接、安装后的运行库路径和开关诊断。Linux
依赖先沿用主仓约束 `z3-solver>=4.13.0,<4.15.5`，在独立构建与安装测试中验证。
启用新同步后端必须具备 Z3 服务；未编入或缺失运行库时明确报错，不能无提示退化成
旧 Analyzer。已启用服务在资源限制、超时或 solver unknown 下返回“未证明”，不据此
消除冲突。未证明、谓词不成立和解析失败不能混成“已证明安全”。

保守保留冲突的规则仅用于访问冲突判断。guard 匹配、epoch 对齐或同步覆盖无法证明时，
必须选择有证明的合法同步位置，否则拒绝输入，不能靠“增加冲突”掩盖协议不成立。
区间运算进行必要的 int64 提升；验证负数 floor-div/mod、cast、窄化、位运算和溢出边界。
通过差分测试对齐主仓需要的证明查询；引入相同 Z3 核心不等于已经证明优化效果相同。

### 5.3 TargetSyncPolicy

主仓面向 950 的 materializer 会跳过部分同 S/V/M pipe 依赖；不能把这个分支直接用于
本仓所有平台。Policy 显式决定：

- 同 pipe 是否天然有序，或需要 barrier/event。
- A2/A3 的 V→V barrier、A5 的 V_V event 和 S↔V 支持关系。
- core/subcore 数量、可通信关系、每类 hard event 的槽数、bank 和保留资源。
- 现有手动/自动 flag 的识别、合法范围和配对约束。
- 无法表达的同步给出诊断；不默认为 `PIPE_ALL` 就称问题已解决。

Policy 的初始值逐项核对当前 VS 实现、helper 和 codegen，并用现有编译/设备测试验证；
不能直接把主仓的 8/16 槽或两个 AIV 写成全平台常量。

### 5.4 输出 ABI 和动态 ID

基本核内输出继续使用已有：

```text
tl.ascend_auto_barrier(pipe)
tl.ascend_auto_set_flag(event, id)
tl.ascend_auto_wait_flag(event, id)
```

PTO 的 auto event 打印与 AscendC 的 auto cross flag 当前部分只支持 literal ID。
Exporter 对已证明属于有限合法集合的动态 ID，生成条件分派，各分支使用 literal ID，
保持 set/wait 的原条件和编号表达式一致。无法证明范围则拒绝，禁止截断或取模补救。
本轮不通过修改 codegen ABI 隐藏这一差异。

已有显式事件必须计入资源预留；seed/drain、循环回边和条件事件参与完整生命周期验证。
事件资源不足时编译失败并报告资源域，不启用 950 特有 spill。

## 6. API、配置与重复执行

统一入口：`AscendInsertSync(target: Target, platform: str)`，注册为
`tl.transform.AscendInsertSync`。它汇总现有 ordinary/VS 配置，只运行一遍统一引擎；
两者均关闭时返回原函数。

迁移期旧两个 API 保留在显式 legacy 分支，用于对照。最终删除旧算法后，这两个名字
作为弃用 alias 指向统一入口；文档说明：alias 按当前两项配置执行统一同步，而不再
分别拥有一个算法。现有 VS 的 target-dependent 默认逻辑仍在公共配置层解析。

统一入口记录 `tl.ascend_sync_proof`：引擎版本、policy/配置签名和同步后整个 PrimFunc
的 structural hash（排除 proof attr 本身）。这同时覆盖 body、buffer_map、地址/大小
maps、流水 provenance 及其他同步输入 attrs。相同输入重复调用不再次分配事件。任何
同步输入、配置或 target 变化时拒绝重复处理，要求从插同步之前的 IR 重新运行；不能
凭标记存在就跳过。已有 proof 的核对先于“配置全关闭则返回”分支。

## 7. IR 变换示例

下述为伪 IR，不是可直接执行的 DSL。假设 a、b 是 UB 中不同 Var，但最终物理区间相同。

输入：

```text
address_map[a] = address_map[b] = 0
resource_scope V:
    copy_ub_to_gm(a, output)       # MTE3 读取旧值
    vector_fill(b, 3)             # V 覆盖相同物理地址
```

输出：

```text
resource_scope V:
    copy_ub_to_gm(a, output)
    auto_set_flag(MTE3_V, id)     # 等待前一读取完成
    auto_wait_flag(MTE3_V, id)
    vector_fill(b, 3)
```

输出不移动原计算、不改变地址。具体编号由资源分配决定；测试验证完成关系和配对，
不固定某个任意 ID。空循环、可选分支不能导致无 producer 的 wait 或逃离 guard 的 set。

## 8. 实施顺序及验收门槛

| 阶段 | 交付 | 退出条件 |
| --- | --- | --- |
| S0 | 设计、组件边界、来源清单、基线 | 契约确认；来源可追溯，明确新旧边界 |
| S1 | 核心移植、旧 TIR importer/exporter、Z3ProofService/物理 alias 适配 | 新引擎独立构建；证明语义和状态恢复、非流水核内语义及负例通过；仍显式 opt-in |
| S2 | 统一 ordinary/VS、target policy、显式资源预留 | 旧 VS 的语义覆盖完整；无旧 VS 补洞；动态 ID 和容量负例通过 |
| S3 | 流水 provenance、owner/epoch 适配 | prologue/steady/epilogue、多版本、嵌套/动态循环和 Persistent 组合验证通过 |
| S4 | 完整编译、设备正确性与性能对照 | 各启用 target/codegen 分别验收；关闭 Bisheng 自动补同步时仍正确 |
| S5 | 默认切换、删除旧算法和过渡配置 | pipeline 仅新入口；旧 API alias 有兼容测试；完整聚焦回归通过 |

S1 不是最终替换。S3/S4 未通过的 target 不得切换默认。upstream 模式拒绝未覆盖输入，
不静默返回未同步的函数、不部分调用旧实现。

### 8.1 测试分层

1. CPU Pass：手工构造事实与任务，覆盖 RAW/WAR/WAW、不同 Var 同地址、不同 dtype/view、
   read-write、动态/空/单次循环、分支、配置寄存器、手动事件、耗尽、重复执行。
2. 流水契约：独立构造 owner、slot、guard；故意破坏 provenance 必须失败；不让主仓
   scheduler 或新 importer 自己输出全部 oracle。
3. 编译：AscendC/PTO 各自验证同步 ABI、动态 ID 分派、资源 scope、尾块与物理 footprint。
4. NPU：先最小无流水与 zero-trip，再流水、VS、Persistent 组合；进程超时检测死锁。
5. 性能：固定设备、输入、编译参数和 warmup；报告延迟分布、同步数量、编译时间、事件
   和内存占用。迁移安全性与性能分别下结论，不预先承诺减少事件或提速。
6. Z3 服务：主仓表达式与等价旧 TIR fixture 的差分证明、嵌套约束恢复、mutable-read
   缓存、负数 div/mod、cast/位运算、资源限制、并行独立实例，以及安装后动态库加载。

主仓测试来源包括 `testing/ascend/auto_schedule/test_sync_dependencies.py`、
`test_sync_boundaries.py`、`test_sync_access_footprints.py`、`test_buffer_epochs.py`、
`test_epoch_guards.py`、`test_flag_allocation.py`、`test_cross_core_flags.py`。

已验证的现有基线（上一轮执行，新实现尚不存在）：51 个 CPU IR/源码生成回归及
3 个 NPU zero-trip 同步回归通过；未测性能。CPU 基线命令：

```bash
python -m pytest -q \
  testing/python/language/test_ascend_sync_insert.py \
  testing/python/language/test_ascend_sync_insert_one_trip.py \
  testing/python/language/test_tilelang_ascend_language_persistent.py \
  -p no:cacheprovider
```

独立设备测试为 `testing/python/language/test_ascend_sync_insert_zero_trip.py`，3 例
使用 AscendC/A3 配置且显式 `--cce-auto-sync=off`。VS 测试会调用编译器，不计为纯 Pass 测试。

## 9. 文档自检与当前状态

- 阶段：Phase 2，MemoryPlanning 后、ResourceScopeVerify 前，已明确。
- 依赖：任务/访问/约束/物理地址/owner/epoch，来源和缺失策略已明确。
- 输入/输出：保留计算及地址，只新增同步，提供伪 IR 示例。
- 实现：移植核心、Z3ProofService 与 adapter 分工见 4.6，具体接口以对应源码头文件为准。
- 集成：单入口、新旧互斥、默认切换和删除条件明确。
- 验证：Pass、编译、设备、性能分层和独立证明服务测试明确；未将现有测试结果当作新实现结果。
- 已采纳并落地可选后端及静态 physical-ring 首片；完整 owner/epoch、多资源与复合
  helper 支持仍未完成。
