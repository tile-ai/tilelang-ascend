# 自动同步移植来源与适配记录

本文件仅记录实际移植或直接参考的来源。同步引擎完整闭包仍以
[迁移设计](pass-design.md)为准；当前可选后端的实际覆盖范围见[实现进展](implementation-status.md)。
历史调研与路线讨论统一指向
[RFC #1971](https://github.com/tile-ai/tilelang-ascend/issues/1971)，不再另行维护实现前的拟议接口骨架。

## 固定比较与迁移边界

旧后端基线为 `43b0b69720f479894a13c02746599eccee5c5769`，使用 TVM
`c2921fdaf795b1103d21abc962e83a209c7258d7`；主仓参考为
`1c0f402e166a7d0a055c991d2f82d2175a7917b7`，使用 TVM
`4211874e9e4b8770fd20ad0d98ab80afd1df5028`。源码 blob 记录不包含原工作树两份 TVM 修改；
仅有 commit 不能重建那些受测快照，验收证据边界见[摘要](validation/summary.json)。

迁移决策保留以下源码比较事实：

- 本仓 `tilelang/engine/phase.py` 在旧 injector、flatten/storage rewrite 和最终
  MemoryPlanning 后插同步；主仓 `tilelang/ascend/pipeline.py` 在
  `LowerScheduledTIR` 前对任务/core/stage/version 分析，后续还可消费 alias proof。
  因此本仓必须按最终物理地址分析，不能在同步后重新分配地址。
- 主仓 `testing/ascend/_ir.py` 有 source-order/stage-0 的独立同步测试输入，说明同步核心
  可以独立于 AutoSchedule 调度求解器接入；这不允许丢弃一般流水的 owner/epoch。
- 本次保留旧 TIR、AscendC/PTO codegen 和现有 launch ABI，不迁移主仓 TVM-FFI runtime、
  task scheduler 或 scheduled-TIR codec。源码闭包按实际调用裁剪，文件行数和直接 include
  统计不能代替编译、协议或设备验收。

## Z3 证明服务

证明算法来自主仓 TVM commit `4211874e9e4b8770fd20ad0d98ab80afd1df5028`。
原许可证头随文件保留。blob hash 针对 Git 中固定内容，不包含本地工作树差异。

| 上游路径 | 源 blob hash | 本仓目标 |
| --- | --- | --- |
| `src/target/z3/z3_prover_on.cc` | `66e514bb3b66aba52a0a4e84e5dd3d2842d127d0` | `src/transform/ascend_sync/z3_proof_service.cc` |
| `src/target/z3/z3_prover_off.cc` | `c9925a51653c2859bec9cee566a6ff4d34a27712` | `src/transform/ascend_sync/z3_proof_service_off.cc` |
| `include/tvm/arith/analyzer.h` 中的 Z3Prover 声明 | `6195575e7e89dd4276588dd9f09b57dc0b0f7e34` | `src/transform/ascend_sync/z3_proof_service.h` |

构建发现参考主仓 commit `1c0f402e166a7d0a055c991d2f82d2175a7917b7` 的
`cmake/pypi-z3/FindZ3.cmake`，blob 为
`5ee8ea632305b97e3cf6c189464e3b684c8407c6`。本仓查找模块为适配实现，非原文件复制。

随本机验证的 Z3 4.15.4 分发的完整许可证来自官方固定 tag：
[z3-4.15.4/LICENSE.txt](https://raw.githubusercontent.com/Z3Prover/z3/z3-4.15.4/LICENSE.txt)。
原文保存在 `src/transform/ascend_sync/licenses/Z3-LICENSE.txt`，SHA256 为
`e617cad2ab9347e3129c2b171e87909332174e17961c5c3412d0799469111337`。
它保留原始版权声明；不是用 TileLang 的 MIT 文本代替第三方许可证。
原文第三行包含行尾空格；`.gitattributes` 仅对此许可证关闭该项空白诊断，保持上述
官方原文哈希，其他源码和文档继续执行空白检查。

主要适配边界：

- `tirx`、新 FFI 容器/hash 转为本仓旧 TIR、ObjectRef 和结构操作。
- 独立 `tvm::tl::ascend_sync::Z3ProofService`，不修改全局 Analyzer ABI 或 CanProve。
- 保留 SMT 翻译、lazy journal、独立 context、资源限制及副作用缓存的核心逻辑。
- 禁用实现明确报告服务不可用，不将空实现返回 false 误当作正常启用服务。
- 不支持的窄化等表达式必须诊断，不得用数学整数模型输出不安全的位宽证明。
- 内部 PackedFunc 调用门面及 ON/OFF 构建接入为本仓新增，不属于上游公开 API。

语义测试同时参考主仓 `testing/python/arith/test_arith_hard.py`。
不能仅依据源码复制声称行为等价；实际验证范围见[实现进展](implementation-status.md)。

### 证明服务的语义修正

移植测试发现以下需要明确记录的差异，均有主机回归覆盖：

1. scope 内绑定的 memo/表达式槽在退出时恢复，lazy journal 回放与 eager 查询一致。
2. `allow_override` 在即时绑定和回放中都生效，失效的复合表达式缓存被清理。
3. unsigned 位运算按结果符号解释，unsigned 右移使用逻辑右移，避免高位被证明为负。
4. shift 量必须先在已有上下文下证明合法；不把未经证明的合法性条件永久加入 solver，
   避免将状态变成 UNSAT 后证明任意命题。
5. 非法 Bind/EnterConstraint 输入在提交 journal 前拒绝；异常时恢复 scope 和临时 memo。
6. 非保值 cast 明确拒绝。保留上游数学整数索引模型，不宣称完整机器整数回绕语义。

## 同步核心、模型和约束

以下来源均位于主仓 commit `1c0f402e166a7d0a055c991d2f82d2175a7917b7`。

| 主仓源路径 | 源 blob hash |
| --- | --- |
| `src/ascend/transform/insert_sync.cc` | `1ded68e6062bf1c6bc285479c72a6c9e225db4e2` |
| `src/ascend/transform/auto_schedule/ir_structure.h` | `36ac7ebfb1a88ace6c7ff10565b9b7d124d3e1db` |
| `src/ascend/transform/auto_schedule/ir_structure.cc` | `e91280062adb0f8c956ad4b4bfcdc2d1e5ef6802` |
| `src/ascend/transform/auto_schedule/dependency_analysis.h` | `6eb96f486d6422b82c6799ab2ffe5e35367eb8f1` |
| `src/ascend/transform/auto_schedule/dependency_analysis.cc` | `732eda27c13b205662d95cf5a8da183cbfe3d382` |
| `src/ascend/transform/auto_schedule/multi_buffer.h` | `1cf02494b49687db79831107d32b6cef94c8d590` |
| `src/ascend/transform/auto_schedule/buffer_lifetime_analysis.h` | `b708810a2f6f5bde5909ecedbb9a62a9c7f265b1` |
| `src/ascend/transform/auto_schedule/buffer_alias_analysis.h` | `df0a2e65ad20a516ce47d651b4930006742003bf` |
| `src/ascend/transform/buffer_version.h` | `641bc0fcf76c2fbb0d5744ffff9db28a141f99d6` |
| `src/ascend/transform/core_mask.h` | `ee137c43ec5eeeb17d4cdbcca0e9e4aeed0908b6` |
| `src/ascend/transform/ascend_pipe.h` | `52bffd8c8f65278cc10f339aa1acb2150d082ff6` |
| `src/transform/common/constr_visitor.h` | `3230b9967d380cef4f0207c67822aadcc5530a81` |
| `src/ascend/transform/normalize_control_flow_for_schedule.cc` | `7ba1a0193e76b4f3760ad9074ca6a88cd26d2534` |

活跃的任务、依赖、约束和版本模型移至 `src/transform/ascend_sync/upstream/`，并做旧 TIR/API 适配。
表中的 alias-analysis 与 generation-lifetime 来源保留作审计记录；后续地址复用 planner
不移入当前链，其无消费者的 context、记录和辅助头文件已裁剪。事件复用仍由同步核心的
依赖闭包、完成关系和复用校验保护，不随这些 alias-only 工具删除。
`ascend_pipe.h` 仅用于 ResourcePipe 和事件辅助类型的来源，不引入高层 op 检测目录。
`constraints.{h,cc}` 通过独立 ProofAnalyzer 桥接旧 Analyzer 与 Z3 服务，不改变 TVM ABI。
`legacy_snapshot.{h,cc}` 参考上游 normalization 的运行时求值位置；旧 TIR 使用真实
LetStmt 替代 flat Bind，并为 Let value 单独记录 S task 与求值前后插入锚点。

本仓新增的 `legacy_adapter`、`physical_storage`、`target_policy`、统一 Pass 入口及 FFI
不是上游直接副本。主要边界如下：

- 输入/输出直接处理旧 TIR，保留原始语句与精确 before/after 锚点，省去新 scheduled codec。
- 物理冲突 oracle 使用最终 maps、实际地址域和保守 DMA allocation footprint；逻辑身份不合并。
- 事件容量超限明确报错，不依赖 950 特有的后续 flag spill。
- 旧平台同 pipe 完成规则与 VS 开关语义由 policy 决定；不照搬 950 的隐式完成假设。
- 仅接受能证明顺向配对的已有手动事件；拒绝可能与新依赖构成循环的反向等待协议。
- injector/storage rewrite 发布并传播物理 ring 的 B、root/base、实际 padded stride 和访问
  clock；同步使用主仓已有 manual physical-ring 模式，不引入自动调度/owner 计划。
- 生成的动态核内 ID 分派为有界 literal 调用，满足旧 PTO 的 `EVENT_ID<n>` ABI。
- 闭包分析前检查条件域；不同条件域的潜在写冲突（含共同循环的跨迭代冲突）明确拒绝，
  防止可选桥接任务错误覆盖必需的直接同步。
- 未覆盖的版本/owner、跨资源协议及复合 helper 明确拒绝，不调用旧算法补洞。
  下述新增适配只放行各自审计过的真实执行范围，不推导其他 helper 的完成语义。

### 后续本仓适配模块

下列模块是本仓新增的旧 ABI 适配，并非从主仓同名算法直接复制；上游核心仍负责依赖
闭包、覆盖验证、同步点和事件分配。实际验证及平台限制见实现进展。

| 模块 | 来源和语义边界 |
| --- | --- |
| `legacy_composite.{h,cc}` | 审计本仓 `src/tl_templates/ascend/common.h` 的 typed `gemm_v0`；提供实际 MTE1 acquire、Cube completion、A/B/C 访问及隐藏 L0 事件预留。受限外部 K-loop 保留真实 clear 条件，不把 helper 简化成单 PIPE_M。 |
| `legacy_cross_core.{h,cc}` | 审计本仓 `CombineCV`/codegen 已生成的单次 mode-2 FIX C→V 发布；保留原跨核调用，验证全部真实跨资源冲突后增加分析用 release/acquire effects，C/V 分别运行原核心并合并回写。没有迁移 950 新跨核策略。 |
| `legacy_runtime_contract.h`、`tilelang/jit/ascend_runtime_contract.py` | 审计本仓 Cython wrapper 的每次调用自动 workspace 分配；版本化 attr 经真实参数位置解析为 fresh GM 事实。缓存 manifest 核对 lowered 参数数目、真实索引表及原函数源码 hash，不能把缺失表当作合法空表，也不伪造整函数 `tir.noalias`。 |
| `legacy_tail_copy.{h,cc}` | 审计本仓 AscendCopy→CopyCodegen→`copy_gm_to_ub` 实际模板、mask、stride 和内部 event-0 协议；保留一次原 helper 调用，完整 full/tail envelope 使用 MTE2 acquire/completion，覆盖整块 Duplicate footprint 并预留 Vector 隐藏事件。 |
| `legacy_persistent_normalize.{h,cc}` | 识别本仓 `src/ir.cc::PersistentFor` 的规范领先 guard，以及 Simplify 的 `cid//d` / 有证据的 EQ 形式；验证原 For 和真实 worker 范围后收紧实际 trip count。拒绝其他 break，不引入新的调度器。 |
| `legacy_pipeline_partition.{h,cc}` | 消费本仓 injector 发布的真实 owner/extent/segment/stage 与 ring provenance；将已验证的短 trip/steady 路径放入各自完整协议分支，保留 true B、clock 和分配壳，不以清空版本或放宽用户 If guard 规避契约。 |
| `legacy_persistent_pipeline.{h,cc}` | 消费已验证的规范 Persistent 证书和内部单 owner UB B2/lead2 事实；保留原计算与分配，以六对事件串行封口外层 wave，内部和外部均使用真实核心图。该受限组合是本仓适配，并非上游 automatic owner scheduler 的移植。 |

为对接复合执行边界及原 If，核心增加显式 acquire/completion 和 optional-control 适配点；
这些属于移植后的本仓语义修订，不应描述为主仓文件的逐字副本。Persistent 归一化是
进入同步图之前的等价控制流处理，不移动原计算到其他 worker，也不改变地址规划。

### A3 PTO GEMM helper 协议修正

后续 `legacy_composite` 的 PTO profile 单独审计本仓 `src/tl_templates/pto/common.h`。
该 helper 反复使用物理地址零的一个隐藏 L0A/L0B slot，六种隐藏事件方向均使用 ID1；
不能套用 AscendC 的双 slot prime/drain 协议。输入仅放行 A3、flat/B1、M=N16、
K128/256、kL0=128、fp16→fp32、clear=True、无 transpose 且 n_actual=N。

独立四 FIFO 模型发现原 helper 每非末 chunk 的 `MTE1_MTE2#1` 可能重复置位，
≥3 chunk 时 `FIX_M#1` 也有同类问题。修正是本仓原 helper 的协议修复，并非又一套
自动同步算法：`MTE1_MTE2` 每 helper 末尾一次，`FIX_M` 仅首个非末 chunk 一次，
每 chunk 的 `M_MTE1`/`MTE1_M` 反馈、计算和搬运均保留。C++ 中相邻的 set/wait
不能证明独立硬件 pipe 已消费 token。

实际 TIR 的 K128/256 检查使用真实物理字节；1/2/3/4/8 chunk 的附加检查仅是协议
抽象，不扩大编译器或设备支持范围。模型枚举 24 种 FIFO 优先级，不声称穷尽任意
interleaving。旧协议反例保留。原生 engine ABI 升至 v8，保证显式 upstream 编译缓存
重建；这不自动使旧 legacy 磁盘缓存失效。共享 helper 修复也作用于重新编译的 legacy
PTO，非整除尾块、transpose、更大 K 和 A2/A5 GEMM 尚未认证。
