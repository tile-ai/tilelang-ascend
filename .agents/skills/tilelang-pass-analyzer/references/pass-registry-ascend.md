# Pass Registry - Ascend Platform

本文件记录 Ascend 平台专用 Pass 的名称、路径和配置信息。

---

## Pass 注册表

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| AscendSyncInsert | tl.transform.AscendSyncInsert | `AscendSyncInsert(target, platform)` | `ascend_sync_insert.cc` | `tl.ascend_auto_sync` |
| AscendMemoryPlanning | tl.transform.AscendMemoryPlanning | `AscendMemoryPlanning()` | `ascend_memory_planning.cc` | `tl.ascend_memory_planning` |
| CombineCV | tl.transform.CombineCV | `CombineCV()` | `ascend_combinecv.cc` | `tl.ascend_auto_cv_combine` |
| CrossCorePipeline | tl.transform.CrossCorePipeline | `CrossCorePipeline()` | `cross_core_pipeline.cc` | `tl.ascend_auto_cross_core_sync` |
| AscendLowerParallelToVector | tl.transform.AscendLowerParallelToVector | `AscendLowerParallelToVector()` | `ascend_lower_parallel_to_vector.cc` | - |
| AscendStorageRewrite | tl.transform.AscendStorageRewrite | `AscendStorageRewrite(is_npu)` | `ascend_storage_rewrite.cc` | - |
| InferAllocScope | tl.transform.InferAllocScope | `AscendInferBufferScope()` | `ascend_infer_buffer_scope.cc` | - |
| AscendLowerOpaqueBlock | tl.transform.AscendLowerOpaqueBlock | `AscendLowerOpaqueBlock()` | `ascend_lower_opaque_block.cc` | - |
| Flatten2DBuffer | tl.transform.Flatten2DBuffer | `Flatten2DBuffer()` | `ascend_collect_buffer_shape.cc` | - |
| CollectBufferShapes | tl.transform.CollectBufferShapes | `CollectBufferShapes()` | `ascend_collect_buffer_shape.cc` | - |
| BufferShapeCollector | tl.transform.BufferShapeCollector | `BufferShapeCollector()` | `ascend_pto_save_buffer_shape.cc` | - |
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
- `EventPair_<src>_<dst>` - 跨 pipeline 同步（共26种组合，见 operation_config.h:264-300）

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

**核心类：** `CombineCV` (继承 `IRMutatorWithAnalyzer`) + `CVCombineEmitter` (继承 `StmtMutator`)

**核心方法：**
- `VisitStmt_(BlockRealizeNode)` - 找到 tilelang_root，创建两个 Emitter
- `CVCombineEmitter.VisitStmt_(EvaluateNode)` - 根据 API 名称和 buffer scope 过滤
- `CVCombineEmitter.VisitStmt_(BufferStoreNode)` - 根据 buffer scope 过滤写入

**工作流程：**
```
tilelang_root → 创建两个 Emitter(is_aiv=true/false)
             → 分别过滤 Cube/Vector 操作
             → 包装为 AttrStmt[resource_scope=0/1]
```

**功能简述：** 分离 Cube 和 Vector 操作，将混合代码拆分为两块独立代码块（resource_scope=0/1），分别发送给 Cube 核和 Vector 核执行。

---

### CrossCorePipeline

**核心类：** `CrossCorePipeline` (继承 `IRMutatorWithAnalyzer`) + `CrossCoreDetector` + `LoopAnalyzer` + `LoopRewriter`

**核心方法：**
- `CrossCoreDetector.VisitStmt_(ForNode)` - 检测 num_stages 注解的循环
- `CrossCoreDetector.VisitStmt_(EvaluateNode)` - 判断 Cube/Vector 操作混合
- `LoopAnalyzer.Analyze()` - 分析 Cube/Vector 操作分布
- `LoopRewriter.Rewrite()` - 重写为多 stage 流水线

**功能简述：** 检测跨核流水线，将单循环拆分为多 stage，使用 set_flag/wait_flag 实现异步流水线。

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
| `tl.ascend_memory_planning` | `false` | 启用 AscendMemoryPlanning |
| `tl.ascend_auto_cv_combine` | `false` | 启用 CombineCV |
| `tl.ascend_auto_cross_core_sync` | `false` | 启用 CrossCorePipeline |

---

## 文件路径汇总

```
src/transform/
├── ascend_sync_insert.cc          (1559 行)
├── ascend_memory_planning.cc      (884 行)
├── ascend_combinecv.cc            (~700 行)
├── cross_core_pipeline.cc         (~1200 行)
├── ascend_lower_parallel_to_vector.cc (~2000 行)
├── ascend_storage_rewrite.cc      (~2200 行)
├── ascend_infer_buffer_scope.cc   (~900 行)
├── ascend_lower_opaque_block.cc   (~400 行)
├── ascend_collect_buffer_shape.cc (~300 行)
├── ascend_pto_save_buffer_shape.cc (~100 行)
└── ascend_host.cc                 (~100 行)
```