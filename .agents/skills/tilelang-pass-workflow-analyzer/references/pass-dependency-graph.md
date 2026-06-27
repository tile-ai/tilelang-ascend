# Pass 依赖关系图

本文档详细说明 TileLang-Ascend Pass 之间的依赖关系和数据流传递。

## 数据传递方式

Pass 之间通过 `PrimFunc` attrs 传递数据：

```cpp
// 示例：设置 attr
f = f.WithAttr("address_map", address_map);

// 示例：读取 attr
auto address_map = f->GetAttr<Map<Var, PrimExpr>>("address_map").value();
```

---

## 关键数据流概览

```
[Phase 1: LowerAndLegalize]
    ↓
AscendInferBufferScope → buffer scope annotations
    ↓
BufferShapeCollector → buffer shapes (初步)
    ↓
CollectBufferShapes → buffer_shapess (Map<Var, Array<PrimExpr>>)
    ↓
[Phase 2: OptimizeForTarget]
    ↓
CrossCorePipeline (使用 buffer scope)
    ↓
CombineCV (使用 buffer scope)
    ↓
AscendMemoryPlanning (使用 buffer_shapess) → address_map, size_map
    ↓
AscendSyncInsert (使用 address_map, size_map)
    ↓
Final IR
```

---

## 数据依赖关系表

### 主要数据流

| 数据名称 | Attr 键 | 类型 | 产生 Pass | 消费 Pass | 用途 |
|---------|---------|------|-----------|-----------|------|
| **Buffer Scope** | buffer annotations | Map<Var, String> | `AscendInferBufferScope` | `CrossCorePipeline`, `CombineCV` | 标注 buffer 应分配在哪个内存层级 |
| **Buffer Shapes** | `buffer_shapess` | Map<Var, Array<PrimExpr>> | `CollectBufferShapes` | `AscendMemoryPlanning` | 记录每个 buffer 的形状信息 |
| **Address Map** | `address_map` | Map<Var, PrimExpr> | `AscendMemoryPlanning` | `AscendSyncInsert` | 记录每个 buffer 的起始地址 |
| **Size Map** | `size_map` | Map<Var, PrimExpr> | `AscendMemoryPlanning` | `AscendSyncInsert` | 记录每个 buffer 的大小 |
| **Pipeline Layout** | pipeline layout annotations | Map | `PipelinePlanning` | `InjectSoftwarePipeline` | 记录流水线的 layout 信息 |
| **Cross-Core Pipeline** | cross-core annotations | Map | `CrossCorePipeline` | `CombineCV` | 记录 Cube-Vector 核间流水线信息 |

### 辅助数据流

| 数据名称 | 产生 Pass | 消费 Pass | 用途 |
|---------|-----------|-----------|------|
| **Target Attr** | `tir.transform.BindTarget` | 多个 Pass | 记录编译目标（Ascend NPU） |
| **Buffer Allocation Plan** | `tir.transform.PlanAndUpdateBufferAllocationLocation` | `Flatten2DBuffer`, `FlattenBuffer` | 确定 buffer 分配位置 |
| **Vectorized IR** | `AscendLowerParallelToVector` | `LegalizeVectorizedLoop` | 提供 Vector 指令 IR |

---

## Phase 1 内部依赖链

### 数据流图（Phase 1）

```
DSL IR
  ↓
AscendInferBufferScope
  ↓ (output: buffer scope annotations)
BufferShapeCollector
  ↓ (output: buffer shapes (初步))
tir.transform.BindTarget
  ↓ (output: target attr)
HostProcesser
  ↓
tir.transform.Simplify
  ↓ (output: simplified IR)
AscendLowerParallelToVector
  ↓ (output: vectorized IR)
LayoutInference
  ↓ (output: layout annotations)
CollectBufferShapes
  ↓ (output: buffer_shapess) ← **关键输出**
LowerTileOp
  ↓ (output: lowered tile ops)
LegalizeVectorizedLoop
  ↓
LegalizeSafeMemoryAccess
  ↓
tir.transform.Simplify
  ↓
Phase 1 Final IR
```

### Phase 1 依赖说明

| Pass | 输入依赖 | 输出供给 | 重要性 |
|------|---------|---------|--------|
| `AscendInferBufferScope` | DSL IR | buffer scope annotations | **基础 Pass**：为 Phase 2 多个 Pass 提供依赖 |
| `BufferShapeCollector` | buffer scope | buffer shapes (初步) | 辅助 Pass：收集初步形状信息 |
| `AscendLowerParallelToVector` | simplified IR | vectorized IR | **核心 Lowering Pass**：将 Parallel → Vector |
| `CollectBufferShapes` | layout annotations | **`buffer_shapess`** | **关键 Pass**：Phase 2 的 `AscendMemoryPlanning` 依赖此输出 |
| `LowerTileOp` | buffer shapes | lowered tile ops | **核心 Lowering Pass**：将 Tile DSL → 底层 IR |

---

## Phase 2 内部依赖链

### 数据流图（Phase 2）

```
Phase 1 Final IR (含 buffer_shapess)
  ↓
tir.transform.PlanAndUpdateBufferAllocationLocation
  ↓ (output: buffer allocation plan)
CrossCorePipeline
  ↓ (output: cross-core annotations, 使用 buffer scope)
CombineCV
  ↓ (output: separated CV ops, 使用 buffer scope)
PipelinePlanning
  ↓ (output: pipeline layout)
InjectSoftwarePipeline
  ↓ (output: software pipeline, 使用 pipeline layout)
AscendLowerOpaqueBlock
  ↓
tir.transform.NarrowDataType(32)
  ↓
ConfigIndexBitwidth
  ↓
Flatten2DBuffer
  ↓
FlattenBuffer
  ↓
tir.transform.Simplify
  ↓
VectorizeLoop
  ↓
AscendStorageRewrite
  ↓ (output: optimized storage)
tir.transform.UnrollLoop
  ↓
tir.transform.RenormalizeSplitPattern
  ↓
tir.transform.Simplify
  ↓
tir.transform.RemoveNoOp
  ↓
tir.transform.RewriteUnsafeSelect
  ↓
tir.transform.HoistIfThenElse
  ↓
AscendMemoryPlanning
  ↓ (output: **address_map**, **size_map**, 使用 buffer_shapess)
AscendSyncInsert
  ↓ (使用 address_map, size_map)
AscendSyncInsertVS
  ↓ (使用 address_map, size_map; 可选，由 tl.ascend_auto_sync_vs 控制)
Phase 2 Final IR
```

### Phase 2 依赖说明

| Pass | 输入依赖 | 输出供给 | 重要性 |
|------|---------|---------|--------|
| `CrossCorePipeline` | buffer scope (来自 Phase 1) | cross-core annotations | **核心优化 Pass**：依赖 Phase 1 的 buffer scope 信息 |
| `CombineCV` | buffer scope, cross-core annotations | separated CV ops | **核心优化 Pass**：依赖 buffer scope 判断 Cube/Vector 分离 |
| `InjectSoftwarePipeline` | pipeline layout | software pipeline | **性能优化 Pass**：依赖 layout 信息 |
| `AscendStorageRewrite` | vectorized loops | optimized storage | **内存优化 Pass**：为 memory planning 提供优化的 IR |
| `AscendMemoryPlanning` | **`buffer_shapess`** (来自 Phase 1) | **`address_map`**, **`size_map`** | **关键 Pass**：跨阶段依赖，使用 Phase 1 输出 |
| `AscendSyncInsert` | **`address_map`**, **`size_map`** | final IR with syncs | **末环之一**：依赖 memory planning 的输出，处理部分跨流水线依赖及 `PIPE_ALL` barrier |
| `AscendSyncInsertVS` | **`address_map`**, **`size_map`** | final IR with syncs | **可选末环**：补位 V→V 与 S↔其他 场景，与 `AscendSyncInsert` 互补协同（可同开），需配套 `TL_CCE_AUTO_SYNC=off` |

---

## 跨阶段依赖链

### Phase 1 → Phase 2 的数据传递

```
[Phase 1]
CollectBufferShapes → buffer_shapess
[Phase 2]
AscendMemoryPlanning ← buffer_shapess (使用)

[Phase 1]
AscendInferBufferScope → buffer scope annotations
[Phase 2]
CrossCorePipeline ← buffer scope (使用)
CombineCV ← buffer scope (使用)
```

### 跨阶段依赖的重要性

1. **数据一致性**：Phase 1 必须先执行，为 Phase 2 提供必要数据
2. **顺序约束**：Phase 2 的 Pass 不能提前执行，否则缺少输入数据
3. **调试边界**：Phase 1 和 Phase 2 的输出可以独立验证

---

## 关键 Pass 的依赖关系详解

### 依赖链 1: Buffer Scope 相关

```
AscendInferBufferScope (Phase 1, 步骤 1)
  ↓ (output: buffer scope annotations)
  
[跨阶段传递]
  
CrossCorePipeline (Phase 2, 步骤 2)
  ↓ (使用 buffer scope, output: cross-core annotations)
CombineCV (Phase 2, 步骤 3)
  ↓ (使用 buffer scope, output: separated CV ops)
```

**依赖说明**：
- `AscendInferBufferScope` 推断每个 buffer 应分配在哪个内存层级（L1/UB/L0A/L0B/L0C）
- `CrossCorePipeline` 需要知道 buffer 的 scope 才能规划核间流水线（Cube buffer 在 L1，Vector buffer 在 UB）
- `CombineCV` 需要知道 buffer 的 scope 才能分离 Cube/Vector 操作

### 依赖链 2: Buffer Shapes 相关

```
CollectBufferShapes (Phase 1, 步骤 8)
  ↓ (output: buffer_shapess)
  
[跨阶段传递]
  
AscendMemoryPlanning (Phase 2, 步骤 20)
  ↓ (使用 buffer_shapess, output: address_map, size_map)
AscendSyncInsert (Phase 2, 步骤 21)
  ↓ (使用 address_map, size_map)
```

**依赖说明**：
- `CollectBufferShapes` 收集所有 buffer 的形状信息（维度、大小）
- `AscendMemoryPlanning` 使用形状信息计算每个 buffer 的地址和大小
- `AscendSyncInsert` 使用地址和大小信息插入同步指令（知道哪些 buffer 需要同步）

### 依赖链 3: Lowering 相关

```
AscendLowerParallelToVector (Phase 1, 步骤 6)
  ↓ (output: vectorized IR)
LegalizeVectorizedLoop (Phase 1, 步骤 10)
  ↓ (使用 vectorized IR, output: legalized loops)

LowerTileOp (Phase 1, 步骤 9)
  ↓ (output: lowered tile ops)
LegalizeSafeMemoryAccess (Phase 1, 步骤 11)
  ↓ (使用 lowered tile ops, output: safe memory IR)
```

**依赖说明**：
- `AscendLowerParallelToVector` 将 Parallel 循环 lowering 到 Vector 指令
- `LegalizeVectorizedLoop` 确保向量化循环符合硬件约束
- `LowerTileOp` 将 Tile DSL 原语 lowering 到底层 IR
- `LegalizeSafeMemoryAccess` 检查内存访问是否安全

### 依赖链 4: 流水线相关

```
CrossCorePipeline (Phase 2, 步骤 2)
  ↓ (output: cross-core annotations)
CombineCV (Phase 2, 步骤 3)
  ↓ (使用 cross-core annotations, output: separated CV ops)
PipelinePlanning (Phase 2, 步骤 4)
  ↓ (output: pipeline layout)
InjectSoftwarePipeline (Phase 2, 步骤 5)
  ↓ (使用 pipeline layout, output: software pipeline)
```

**依赖说明**：
- `CrossCorePipeline` 规划 Cube-Vector 核间流水线
- `CombineCV` 分离 Cube/Vector 操作，依赖跨核流水线信息
- `PipelinePlanning` 推断流水线的 layout
- `InjectSoftwarePipeline` 注入软件流水线，依赖 layout 信息

---

## 依赖关系的代码证据

### AscendMemoryPlanning 的依赖读取

```cpp
// 文件: src/transform/ascend_memory_planning.cc
auto buffer_shapess = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess").value();
// 使用 buffer_shapess 计算地址和大小
```

### AscendSyncInsert 的依赖读取

```cpp
// 文件: src/transform/ascend_sync_insert.cc
auto address_map = f->GetAttr<Map<Var, PrimExpr>>("address_map").value();
auto size_map = f->GetAttr<Map<Var, PrimExpr>>("size_map").value();
// 使用 address_map 和 size_map 插入同步
```

### CrossCorePipeline 的依赖读取

```cpp
// 文件: src/transform/cross_core_pipeline.cc
// 需要知道 buffer 的 scope 才能规划核间流水线
// buffer scope 来自 AscendInferBufferScope (Phase 1)
```

---

## 依赖关系的可视化图

### 整体依赖图

```
DSL IR
  ↓
[Phase 1]
  ├─ AscendInferBufferScope → buffer scope
  ├─ BufferShapeCollector → buffer shapes (初步)
  ├─ AscendLowerParallelToVector → vectorized IR
  ├─ CollectBufferShapes → buffer_shapess ★
  └─ LowerTileOp → lowered tile ops
  ↓
[跨阶段数据传递]
  ├─ buffer scope → Phase 2 多个 Pass
  └─ buffer_shapess → AscendMemoryPlanning
  ↓
[Phase 2]
  ├─ CrossCorePipeline ← buffer scope
  ├─ CombineCV ← buffer scope
  ├─ InjectSoftwarePipeline ← pipeline layout
  ├─ AscendStorageRewrite → optimized storage
  ├─ AscendMemoryPlanning ← buffer_shapess → address_map, size_map ★
  ├─ AscendSyncInsert ← address_map, size_map
  └─ AscendSyncInsertVS ← address_map, size_map (可选)
  ↓
Final IR
```

### 关键数据流路径

```
[Phase 1]
AscendInferBufferScope → buffer scope
  ↓ [跨阶段]
CrossCorePipeline ← buffer scope
CombineCV ← buffer scope

[Phase 1]
CollectBufferShapes → buffer_shapess
  ↓ [跨阶段]
AscendMemoryPlanning ← buffer_shapess → address_map, size_map
  ↓ [Phase 2 内部]
AscendSyncInsert ← address_map, size_map
AscendSyncInsertVS ← address_map, size_map (可选，与 AscendSyncInsert 互补协同，可同开)
```

---

## 依赖关系的影响

### 顺序约束

1. **Phase 1 必须先执行**
   - Phase 2 的 Pass 依赖 Phase 1 的输出（buffer scope, buffer_shapess）
   - Phase 2 不能独立运行

2. **Phase 2 内部顺序严格**
   - `AscendMemoryPlanning` 必须在 `AscendSyncInsert` / `AscendSyncInsertVS` 前执行
   - `CrossCorePipeline` 必须在 `CombineCV` 前执行
   - `AscendSyncInsert` 与 `AscendSyncInsertVS` 互补协同，可同时开启；前者先执行处理部分跨流水线依赖，后者补位 V→V 与 S↔其他 场景

3. **跨阶段依赖不可打破**
   - 不能在 Phase 1 之前添加需要 Phase 2 输入的 Pass
   - 不能在 Phase 2 之后添加需要 Phase 1 输出的 Pass

### 添加新 Pass 的约束

1. **如果新 Pass 需要 buffer scope**
   - 必须在 `AscendInferBufferScope` 后执行
   - 可以在 Phase 1 或 Phase 2

2. **如果新 Pass 需要 buffer_shapess**
   - 必须在 `CollectBufferShapes` 后执行
   - 必须在 `AscendMemoryPlanning` 前执行（如果 `AscendMemoryPlanning` 需要新 Pass 的输出）

3. **如果新 Pass 需要 address_map/size_map**
   - 必须在 `AscendMemoryPlanning` 后执行
   - 通常在 Phase 2 末尾

---

## 依赖关系的调试建议

### 如何验证依赖关系？

1. **检查 Phase 1 输出**
   - 使用 `T.dump_tensor` 或 IR 打印工具查看 Phase 1 输出的 attrs
   - 确认 buffer scope 和 buffer_shapess 已正确输出

2. **检查 Phase 2 输入**
   - 在 Phase 2 Pass 中打印 attrs，确认输入数据可用
   - 如果缺少输入，说明上游 Pass 未正确输出或顺序错误

3. **跨阶段调试**
   - 在 Phase 1 和 Phase 2 之间保存中间 IR
   - 独立验证 Phase 1 输出的正确性

### 常见依赖问题

1. **Phase 2 Pass 缺少输入**
   - 原因：Phase 1 的 Pass 未正确输出 attrs
   - 解决：检查 Phase 1 Pass 的实现，确保正确设置 attrs

2. **顺序错误导致数据缺失**
   - 原因：Pass 顺序不符合依赖关系
   - 解决：调整 `tilelang/engine/phase.py` 中的 Pass 顺序

3. **跨阶段数据传递失败**
   - 原因：Phase 1 和 Phase 2 之间的数据未正确传递
   - 解决：检查 Phase 1 输出的 attrs 是否在 Phase 2 可访问

---

## 参考资料

- **Pipeline 架构详解**：`pass-pipeline-overview.md`
- **新 Pass 定位指南**：`new-pass-placement-guide.md`
- **Pass 注册表**：`.agents/skills/tilelang-pass-analyzer/references/pass-registry-ascend.md`