# Pass Registry - General (跨平台)

本文件记录通用 Pass（非 Ascend 专用，非 CUDA 专用）的名称、路径和配置信息。

---

## Pass 注册表

### 循环优化类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| VectorizeLoop | tl.transform.VectorizeLoop | `VectorizeLoop(enable_vectorize)` | `vectorize_loop.cc` | `tir.disable_vectorize` |
| LoopVectorizeDynamic | tl.transform.LoopVectorizeDynamic | `LoopVectorizeDynamic()` | `loop_vectorize_dynamic.cc` | - |
| LoopPartition | tl.transform.LoopPartition | `LoopPartition()` | `loop_partition.cc` | - |

### 内存优化类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| FlattenBuffer | tl.transform.FlattenBuffer | `FlattenBuffer()` | `flatten_buffer.cc` | - |
| LegalizeSafeMemoryAccess | tl.transform.LegalizeSafeMemoryAccess | `LegalizeSafeMemoryAccess()` | `legalize_safe_memory_access.cc` | `tl.disable_safe_memory_legalize` |
| LegalizeVectorizedLoop | tl.transform.LegalizeVectorizedLoop | `LegalizeVectorizedLoop()` | `legalize_vectorized_loop.cc` | - |

### 流水线类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| PipelinePlanning | tl.transform.PipelinePlanning | `PipelinePlanning()` | `pipeline_planning.cc` | - |
| InjectSoftwarePipeline | tl.transform.InjectSoftwarePipeline | `InjectSoftwarePipeline()` | `inject_pipeline.cc` | `tl.disable_dynamic_tail_split` |

### Layout 与 Lowering 类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| LayoutInference | tl.transform.LayoutInference | `LayoutInference()` | `layout_inference.cc` | - |
| LowerTileOp | tl.transform.LowerTileOp | `LowerTileOp()` | `lower_tile_op.cc` | - |
| LowerDeviceStorageAccessInfo | tl.transform.LowerDeviceStorageAccessInfo | `LowerDeviceStorageAccessInfo()` | `lower_device_storage_access_info.cc` | - |

### 前端与后端类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| FrontendLegalize | tl.transform.FrontendLegalize | `FrontendLegalize()` | `frontend_legalize.cc` | - |
| MakePackedAPI | tl.transform.MakePackedAPI | `MakePackedAPI()` | `make_packed_api.cc` | - |
| AnnotateDeviceRegions | tl.transform.AnnotateDeviceRegions | `AnnotateDeviceRegions()` | `annotate_device_regions.cc` | - |
| Simplify | tl.Simplify | `Simplify()` | `simplify.cc` | `tl.Simplify` |

### 其他优化类

| Pass 名称 | 注册名 | Python 函数 | C++ 文件 | 配置键 |
|-----------|--------|-------------|---------|--------|
| IfStmtBinding | tl.transform.IfStmtBinding | `IfStmtBinding()` | `if_stmt_binding.cc` | - |
| MergeIfStmt | tl.transform.MergeIfStmt | `MergeIfStmt()` | `merge_if_stmt.cc` | - |
| MultiVersionBuffer | tl.transform.MultiVersionBuffer | `MultiVersionBuffer()` | `multi_version_buffer_rewriter.cc` | - |
| ConfigIndexBitwidth | tl.transform.ConfigIndexBitwidth | `ConfigIndexBitwidth()` | `config_index_bitwidth.cc` | `tl.config_index_bitwidth` |

---

## Pass 详细信息

### VectorizeLoop

**核心类：** `VectorizeLoop`

**功能简述：** 将标量循环向量化，提高并行度。

---

### LoopVectorizeDynamic

**功能简述：** 处理动态形状循环的向量化。

---

### LoopPartition

**功能简述：** 循环分区优化。

---

### FlattenBuffer

**功能简述：** 将多维 buffer 扁平化为 1D。

---

### LegalizeSafeMemoryAccess

**功能简述：** 合法化安全内存访问，处理边界情况。

---

### PipelinePlanning

**功能简述：** 推断 fragment/shared memory 的 layout。

---

### InjectSoftwarePipeline

**功能简述：** 注入软件流水线优化。

---

### LayoutInference

**功能简述：** 推断 tensor layout。

---

### LowerTileOp

**核心类：** `LowerTileOpPass` (继承 `IRMutatorWithAnalyzer`)

**核心方法：**
- `VisitStmt_(BlockNode)` - 处理 layout_map，重映射 buffer
- `VisitStmt_(EvaluateNode)` - 解析 Tile 操作并调用 Lower
- `VisitExpr_(BufferLoadNode/BufferStoreNode)` - 应用 layout 变换

**Tile 操作 Lower 实现：**
- 文件：`src/op/ascend.cc`
- `AscendCopy::Lower()` - 将 T.copy 转换为具体 API（如 copy_gm_to_l1, copy_l1_to_l0a 等）
- 其他 Tile 操作的 Lower 方法也在此文件中

**功能简述：** 将高级 Tile 操作 lowering 为底层 IR，应用 Layout 变换。

---

### Simplify

**功能简述：** IR 简化优化。

---

## 配置键说明

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `tir.disable_vectorize` | `false` | 禁用向量化 |
| `tl.disable_safe_memory_legalize` | `false` | 禁用安全内存访问合法化 |
| `tl.disable_dynamic_tail_split` | `false` | 禁用动态尾部分割 |
| `tl.Simplify` | `true` | 启用简化 |
| `tl.config_index_bitwidth` | `32` | 索引位宽配置 |

---

## 文件路径汇总

```
src/transform/
├── vectorize_loop.cc
├── loop_vectorize_dynamic.cc
├── loop_partition.cc
├── flatten_buffer.cc
├── legalize_safe_memory_access.cc
├── legalize_vectorized_loop.cc
├── pipeline_planning.cc
├── inject_pipeline.cc
├── layout_inference.cc
├── lower_tile_op.cc
├── lower_device_storage_access_info.cc
├── frontend_legalize.cc
├── make_packed_api.cc
├── annotate_device_regions.cc
├── simplify.cc
├── if_stmt_binding.cc
├── merge_if_stmt.cc
├── multi_version_buffer_rewriter.cc
└── config_index_bitwidth.cc
```