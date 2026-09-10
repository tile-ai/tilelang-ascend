# Pass Registry - Ascend Platform

本文件记录 Ascend 平台专用 Pass 的名称、路径和配置信息。

---

## Pass 注册表

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| AscendSyncInsert | tl.transform.AscendSyncInsert | `AscendSyncInsert(target, platform)` | `ascend_sync_insert.cc` | `tl.ascend_auto_sync` |
| AscendSyncInsertVS | tl.transform.AscendSyncInsertVS | `AscendSyncInsertVS(target, platform)` | `ascend_sync_insert_vs.cc` | `tl.ascend_auto_sync_vs` |
| AscendMemoryPlanning | tl.transform.AscendMemoryPlanning | `AscendMemoryPlanning()` | `ascend_memory_planning.cc` | `tl.ascend_memory_planning` controls automatic strategy; pass always publishes maps |
| CombineCV | tl.transform.CombineCV | `CombineCV()` | `ascend_combinecv.cc` | `tl.ascend_auto_cv_combine`; `tl.ascend_auto_cross_core_sync` controls its optional sync insertion |
| AscendResourceScopeVerify | tl.transform.AscendResourceScopeVerify | `AscendResourceScopeVerify()` | `ascend_combinecv.cc` | - |
| AscendVectorInstructionSelection | tl.transform.AscendVectorInstructionSelection | `AscendVectorInstructionSelection(target, platform)` | `ascend_vector_instruction_selection.cc` | A2/A3 AscendC/auto pipeline gate |
| AscendVectorMaskLegalize | tl.transform.AscendVectorMaskLegalize | `AscendVectorMaskLegalize(target, platform)` | `ascend_vector_mask_legalize.cc` | `tl.ascend_vector_mask_reuse` (default `true`) |
| CrossCorePipeline | tl.transform.CrossCorePipeline | `CrossCorePipeline()` | `cross_core_pipeline.cc` | - |
| AscendLowerParallelToVector | tl.transform.AscendLowerParallelToVector | `AscendLowerParallelToVector()` | `ascend_lower_parallel_to_vector.cc` | - |
| AscendStorageRewrite | tl.transform.AscendStorageRewrite | `AscendStorageRewrite(is_npu)` | `ascend_storage_rewrite.cc` | - |
| InferAllocScope | tl.transform.InferAllocScope | `AscendInferBufferScope()` | `ascend_infer_buffer_scope.cc` | - |
| AscendLowerOpaqueBlock | tl.transform.AscendLowerOpaqueBlock | `AscendLowerOpaqueBlock()` | `ascend_lower_opaque_block.cc` | - |
| Flatten2DBuffer | tl.transform.Flatten2DBuffer | `Flatten2DBuffer()` | `ascend_collect_buffer_shape.cc` | - |
| CollectBufferShapes | tl.transform.CollectBufferShapes | `CollectBufferShapes()` | `ascend_pto_save_buffer_shape.cc` | - |
| BufferShapeCollector | tl.transform.BufferShapeCollector | `BufferShapeCollector()` | `ascend_collect_buffer_shape.cc` | - |
| HostLegalize | tl.transform.HostLegalize | `HostProcesser()` | `ascend_host.cc` | - |

---

## Pass 详细信息

### AscendSyncInsert

**核心类：** `AscendSyncInsert` (继承 `IRMutatorWithAnalyzer`) + `ForLoopUnroller` + `LoopRebuilder`

**核心方法：**
- `VisitStmt_(EvaluateNode)` - ⭐ 核心处理函数，包含完整的依赖分析和同步插入流程：
  - `AnalyzeStmtAccesses()` - 分析语句的内存访问（buffer、pipeline、读写类型）
  - `FindRelatedBuffers()` - 查找地址重叠的 buffer（基于 address_map）
  - `GetRequiredSyncType()` - 根据依赖类型选择同步指令
  - `InsertSynchronization()` - 插入 PipeBarrier 或 EventPair
- `PreprocessUnrollForLoops()` - 循环展开预处理（每个 For → iter1 + iter2）
- `MergeAndRebuildForLoops()` - 合并 iter1/iter2 同步，重建循环

**同步类型：**
- `PipeBarrier_ALL` - 全局同步（切片操作、if 分支）
- `PipeBarrier_MTE2/MTE1/MTE3/M/FIX/V/S` - 同 pipeline 内同步
- `EventPair_<src>_<dst>` - 跨 pipeline 同步（以 `GetEventMapping()` 当前表为准）

**功能简述：** 通过循环展开分析内存依赖，在 VisitStmt_(EvaluateNode) 中完成依赖检测、同步选择和插入，确保多 pipeline 异步执行的正确性。

---

### AscendMemoryPlanning

**核心类：** `AscendMemoryPlanner` (继承 `StmtExprVisitor`)

**核心方法：**
- `Substitute()` - Pass 入口
- `GetAddressMap()` - 获取 buffer 地址映射
- `GetBufferSizes()` - 获取 buffer 尺寸

**功能简述：** 为 Ascend NPU 规划内存，分配 buffer 地址，优化内存复用。

---

### CombineCV

**核心类：** `CombineCV` + `CVCombineEmitter` + 共用 `AscendResource` classifier

**核心方法：**
- `ResourceForCall()` - 按 operation contract、pipe 参数和 access pointer 分类
- `MergeResources()` - 交叉校验 operation resource 与 local buffer resource
- `ContextualSyncResolver` - 只给 pure region / 同 owner 双侧证据中的已知歧义 sync 补 scope
- `VisitStmt_(BlockRealizeNode)` - 找到 tilelang_root，创建 C/V 两个 Emitter
- `CVCombineEmitter` - common 两侧保留、C/V 单侧保留、opaque fail closed

**工作流程：**
```
outer input → 保守解析已知上下文型 sync
            → 用共用 classifier 拒绝不可分类 opaque call / 冲突 scope
            → 创建两个 Emitter(is_aiv=true/false)
            → common 两侧保留，resource-specific operation 只保留在 owner 分支
            → 包装为 AttrStmt[resource_scope=0/1]
```

**功能简述：** 把未显式分域的 Developer/Hybrid IR 分为 C/V 两支。copy 归属由实际
src/dst storage scope 决定；MMA、Vector、pipe/event 等由 operation contract 或参数决定。
无法分类的 outer Ascend hardware / opaque call fail closed；普通 resource-neutral statement
仍保留既有分支上下文。

详细分类与 verifier 设计见
`pass-designs/design_ascend_combinecv.md`。

---

### AscendResourceScopeVerify

**核心类：** `AscendResourceScopeVerifier`

**功能简述：** 在 `AscendSyncInsert` / `AscendSyncInsertVS` 生成最终 hardware calls 后，验证
每个 resource-specific operation、local BufferLoad/Store 和 vectorized loop 都位于正确的
`resource_scope`。outer 只允许 common / resource-independent control；同类嵌套允许，C/V
冲突嵌套拒绝。

该 pass 与 CombineCV 调用同一个 `ResourceForCall()`，避免自动分离和最终验证使用两套规则。

---

### AscendVectorInstructionSelection

**功能简述：** 对 A2/A3 `ascendc` / `auto`，在所有已有 TIR rewrite 和同步 pass 结束后，把
managed `T.tile.*` semantic operation 改写为带类型化 repeat/mask 参数的内部 terminal。它同时
校验 exact dtype、operand ABI、count/mask/repeat/stride bounds，并重验已有 selected call 的
variant 与 payload。

完整 Selection 规则由 `src/op/ascend_vector_mask_ops.inc` catalog 驱动；公共编译契约见
`docs/ascend/compiler_managed_vector_mask.md`。

---

### AscendVectorMaskLegalize

**功能简述：** 跟踪 NORMAL/COUNTER mode 及两个 mask words 的 must-facts，根据每条 selected
terminal 的 `requires` / `ensures` 插入最少量 setter。`tl.ascend_vector_mask_reuse=false` 时，
每条 terminal 前后清空 facts，从而完整重建 required mask，但仍保留 Selection 和严格 scope
验证。

它是 AscendC managed path 的最后一个 TIR-transforming pass；后面不能再添加会移动或改写
selected terminal 的 pass。

---

### CrossCorePipeline

**核心类：** `CrossCorePipeline` (继承 `IRMutatorWithAnalyzer`) + `CrossCoreDetector` + `LoopAnalyzer` + `LoopRewriter`

**核心方法：**
- `CrossCoreDetector.VisitStmt_(ForNode)` - 检测 num_stages 注解的循环
- `CrossCoreDetector.VisitStmt_(EvaluateNode)` - 判断 Cube/Vector 操作混合
- `LoopAnalyzer.Analyze()` - 分析 Cube/Vector 操作分布
- `LoopRewriter.Rewrite()` - 重写为多 stage 流水线

**功能简述：** 检测跨核流水线并重写多-stage loop；具体 C/V handoff 由 cross-core
notification 协议表达。

---

### AscendLowerParallelToVector

**核心类：** `AscendLowerParallelToVector` (继承 `IRMutatorWithAnalyzer`)

**核心方法：**
- `VisitStmt_(ForNode)` - 检测 Parallel 循环
- `VisitStmt_(EvaluateNode)` - 将元素级操作转为 Vector 指令

**功能简述：** 将 Parallel 循环 lowering 为 Ascend Vector 指令。

---

### AscendStorageRewrite

**核心类：** `LinearAccessPatternFinder` (继承 `StmtExprVisitor`) + `StoragePlanRewriter` (继承 `StmtExprMutator`)

**核心方法：**
- `LinearAccessPatternFinder.VisitStmt_(BufferStoreNode/BufferLoadNode)` - 记录 buffer 访问
- `LinearAccessPatternFinder.VisitStmt_(AllocateNode)` - 记录分配信息

**功能简述：** 分析内存访问模式，构建线性访问序列，优化存储共享。

---

### InferAllocScope

**核心类：** `ScopeCorrector` (继承 `StmtExprMutator`) + `BufferUseCollector` (继承 `StmtExprVisitor`)

**核心方法：**
- `BufferUseCollector.VisitExpr_(CallNode)` - 分析 buffer 在 GEMM 中的位置
- `InferCorrectScopes()` - 根据 gemm_position 推断 L0A/L0B/L0C
- `ScopeCorrector.VisitStmt_(BlockNode)` - 应用 scope 修正
- `InjectDefaultLayoutMap()` - 注入默认 zN Layout

**功能简述：** 根据 buffer 在 GEMM 中的位置推断 scope，为 L1 buffer 注入默认 Layout。

---

### AscendLowerOpaqueBlock

**核心类：** `OpaqueBlockLower` (继承 `StmtExprMutator`)

**核心方法：**
- `VisitStmt_(BlockRealizeNode)` - 将 Block 转换为 Allocate 嵌套
- `VisitStmt_(ForNode)` - 处理 unit loop 和 ThreadBinding
- `VisitExpr_(VarNode)` - 替换 unit loop 变量

**功能简述：** 将 Block IR lowering 为可执行底层 IR，移除调度抽象。

---

### Flatten2DBuffer

**功能简述：** 将 buffer 形状扁平化为 2D，适配 Ascend 硬件要求。

**变换规则：**
- 1D [M] → 2D [1, M]
- 2D [N, M] → 2D [N, M] (不变)
- ND [D1, D2, ..., Dn] → 2D [D1*D2*...*Dn-1, Dn]

---

### CollectBufferShapes / BufferShapeCollector

**功能简述：** 收集 buffer 形状信息，供后续 pass 使用。

---

### HostLegalize

**功能简述：** Host 端代码合法化处理。

---

## 配置键说明

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `tl.ascend_auto_sync` | `false` | 启用 AscendSyncInsert |
| `tl.ascend_memory_planning` | `false` | 启用自动规划策略；Pass 仍运行并发布 address/size maps |
| `tl.ascend_auto_cv_combine` | `false` | 启用 CombineCV |
| `tl.ascend_auto_cross_core_sync` | `false` | 启用 CombineCV 中的 workspace cross-core sync insertion |
| `tl.ascend_auto_sync_vs` | target-dependent | 启用 AscendSyncInsertVS |
| `tl.ascend_vector_mask_reuse` | `true` | 跨 selected Vector terminal 复用兼容 mask facts；`false` 为保守 repair |

---

## 文件路径汇总

```
src/transform/
├── ascend_sync_insert.cc
├── ascend_memory_planning.cc
├── ascend_combinecv.cc
├── ascend_vector_instruction_selection.cc
├── ascend_vector_mask_legalize.cc
├── common/ascend_vector_mask.{h,cc}
├── cross_core_pipeline.cc
├── ascend_lower_parallel_to_vector.cc
├── ascend_storage_rewrite.cc
├── ascend_infer_buffer_scope.cc
├── ascend_lower_opaque_block.cc
├── ascend_collect_buffer_shape.cc
├── ascend_pto_save_buffer_shape.cc
└── ascend_host.cc
```
