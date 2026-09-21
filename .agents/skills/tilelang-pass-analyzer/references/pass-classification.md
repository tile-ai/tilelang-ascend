# Pass Classification (按硬件平台)

本文件将 TileLang 的 Pass 按硬件平台分类，便于快速查找和理解。

---

## 一、Ascend 平台专用 Pass

### 1. 内存管理

| Pass | 功能简述 | 配置项 | 核心类 |
|------|---------|--------|--------|
| AscendMemoryPlanning | 内存规划与地址分配，优化 buffer 复用 | `tl.ascend_memory_planning` | AscendMemoryPlanner |
| AscendStorageRewrite | 存储重写优化，适配 Ascend 内存层级 | - | LinearAccessPatternFinder |
| InferAllocScope | 推断 buffer scope (L1/UB/L0A/L0B/L0C) | - | ScopeCorrector |
| Flatten2DBuffer | Buffer 扁平化到 2D，适配硬件要求 | - | - |

### 2. 同步与流水线

| Pass | 功能简述 | 配置项 | 核心类 |
|------|---------|--------|--------|
| AscendSyncInsert | 自动插入同步指令，确保数据依赖正确 | `tl.ascend_auto_sync` | AscendSyncInsert |
| CrossCorePipeline | 跨核 (Cube-Vector) 流水线同步调度 | `tl.ascend_auto_cross_core_sync` | CrossCorePipeline |
| CombineCV | 分离 Cube/Vector 操作，拆分为两块独立代码 | `tl.ascend_auto_cv_combine` | CVCombineEmitter |

### 3. 向量化与 Lowering

| Pass | 功能简述 | 配置项 | 核心类 |
|------|---------|--------|--------|
| AscendLowerParallelToVector | Parallel 循环 lowering 为 Vector 指令 | - | AscendLowerParallelToVector |
| AscendLowerOpaqueBlock | Opaque Block 结构 lowering | - | OpaqueBlockLower |

### 4. 数据收集与分析

| Pass | 功能简述 | 配置项 | 核心类 |
|------|---------|--------|--------|
| CollectBufferShapes | 收集 buffer 形状信息 | - | - |
| BufferShapeCollector | Buffer 形状收集器 | - | - |

### 5. Host 处理

| Pass | 功能简述 | 配置项 | 核心类 |
|------|---------|--------|--------|
| HostLegalize | Host 端代码合法化 | - | - |

---

## 二、通用 Pass（跨平台）

### 1. 循环优化

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| VectorizeLoop | 标量循环向量化 | `tir.disable_vectorize` |
| LoopVectorizeDynamic | 动态形状循环向量化 | - |
| LoopPartition | 循环分区优化 | - |

### 2. 内存优化

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| FlattenBuffer | Buffer 扁平化到 1D | - |
| LegalizeSafeMemoryAccess | 安全内存访问合法化 | `tl.disable_safe_memory_legalize` |
| LegalizeVectorizedLoop | 向量化循环合法化 | - |

### 3. 流水线

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| PipelinePlanning | 流水线规划，推断 layout | - |
| InjectSoftwarePipeline | 软件流水线注入 | `tl.disable_dynamic_tail_split` |

### 4. Layout 与 Lowering

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| LayoutInference | Tensor layout 推断 | - |
| LowerTileOp | Tile 操作 lowering，应用 Layout 变换 | - |
| LowerDeviceStorageAccessInfo | 存储访问信息 lowering | - |

### 5. 前端与后端

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| FrontendLegalize | 前端合法化 | - |
| MakePackedAPI | API 打包 | - |
| AnnotateDeviceRegions | 设备区域标注 | - |
| Simplify | IR 简化优化 | `tl.Simplify` |

### 6. 其他优化

| Pass | 功能简述 | 配置项 |
|------|---------|--------|
| IfStmtBinding | If 语句绑定 | - |
| MergeIfStmt | If 语句合并 | - |
| MultiVersionBuffer | 多版本 buffer | - |
| ConfigIndexBitwidth | 索引位宽配置 | `tl.config_index_bitwidth` |

---

## 三、典型使用场景

### 场景 1: Ascend NPU GEMM 算子编译

**推荐 Pass 组合：**
```
FrontendLegalize
  → InferAllocScope (推断 buffer scope)
  → CollectBufferShapes (收集形状)
  → Flatten2DBuffer (扁平化)
  → AscendMemoryPlanning (内存规划)
  → AscendSyncInsert (插入同步)
  → AscendLowerParallelToVector (lowering)
  → MakePackedAPI
```

**配置示例：**
```python
PassContext.current().config = {
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_sync": True,
}
```

---

### 场景 2: Ascend NPU Vector 算子编译

**推荐 Pass 组合：**
```
FrontendLegalize
  → AscendLowerParallelToVector (Parallel → Vector)
  → Simplify (简化)
  → MakePackedAPI
```

---

### 场景 3: Ascend NPU 跨核流水线

**推荐 Pass 组合：**
```
FrontendLegalize
  → CrossCorePipeline (跨核流水线规划)
  → CombineCV (分离 Cube/Vector)
  → AscendMemoryPlanning
  → AscendSyncInsert
  → MakePackedAPI
```

---

## 四、Pass 间协作关系图

### Ascend 平台典型协作链

```
[数据收集阶段]
  CollectBufferShapes → Flatten2DBuffer
         ↓
[内存规划阶段]
  InferAllocScope → AscendMemoryPlanning
         ↓ (输出 address_map)
[同步插入阶段]
  AscendSyncInsert (使用 address_map)
         ↓
[Lowering 阶段]
  AscendLowerParallelToVector / AscendLowerOpaqueBlock
         ↓
[后端处理]
  MakePackedAPI
```

---

## 五、快速查找指南

### 按功能关键词查找

| 关键词 | 相关 Pass |
|--------|----------|
| 内存 / memory | AscendMemoryPlanning, AscendStorageRewrite, FlattenBuffer |
| 同步 / sync | AscendSyncInsert, CrossCorePipeline |
| 流水线 / pipeline | CrossCorePipeline, InjectSoftwarePipeline |
| 向量化 / vector | AscendLowerParallelToVector, VectorizeLoop, LoopVectorizeDynamic |
| Lowering | AscendLowerOpaqueBlock, LowerTileOp |
| Layout | LayoutInference, InferAllocScope |
| 简化 / simplify | Simplify |

### 按平台查找

- **Ascend 专用：** 见本文件第一部分
- **通用 Pass：** 见本文件第二部分