# Pass Pipeline 架构详解

本文档详细说明 TileLang-Ascend 的两阶段编译 Pipeline 架构。

## 架构总览

TileLang-Ascend 编译流程采用**两阶段分离**设计：

```
Python DSL (@tilelang.jit)
    ↓
[Phase 1: LowerAndLegalize] ← 前端标准化、Lowering
    - 目标：将高级 DSL 转换为标准化 TIR
    - 特点：语义保持、平台无关优化
    - Pass 数量：12 个
    ↓
[Phase 2: OptimizeForTarget] ← 后端优化、平台特化
    - 目标：针对 Ascend 硬件特性的优化
    - 特点：硬件相关、性能导向
    - Pass 数量：21 个
    ↓
CANN 工具链 → NPU 执行
```

## 设计原理

### 为什么采用两阶段分离？

1. **职责清晰**
   - Phase 1：确保语义正确性，将 DSL lowering 到标准化 IR
   - Phase 2：针对硬件特性进行性能优化

2. **维护性强**
   - 前端修改（新增 DSL 原语）只影响 Phase 1
   - 后端修改（硬件特性优化）只影响 Phase 2

3. **可扩展性好**
   - 新增平台无关优化 → Phase 1
   - 新增 Ascend 特定优化 → Phase 2

4. **调试友好**
   - Phase 1 输出的标准化 IR 可以独立验证
   - Phase 2 可以基于 Phase 1 的输出进行优化

### Pass 的编排原则

| 原则 | 说明 | 示例 |
|-----|------|-----|
| **依赖优先** | 上游 Pass 必须先执行 | `AscendInferBufferScope` → `BufferShapeCollector` |
| **数据就绪** | Pass 需要的数据必须可用 | `AscendMemoryPlanning` 需要 `buffer_shapess` |
| **逻辑连贯** | 相关 Pass 集中编排 | Lowering 相关 Pass 在 Phase 1 连续编排 |
| **优化分层** | 基础优化先执行，高级优化后执行 | `Simplify` 在多处执行，逐步优化 IR |

---

## Phase 1: LowerAndLegalize（前端标准化）

### 总体目标

将 Python DSL 编写的 kernel 转换为标准化、可分析的 TIR。

### Pass 列表（按执行顺序）

| 步骤 | Pass | 功能 | 输入依赖 | 输出供给 | 关键逻辑 |
|------|------|------|---------|---------|---------|
| 1 | `AscendInferBufferScope()` | 推断 buffer scope (L1/UB/L0A/L0B/L0C) | DSL IR | buffer scope annotations | 分析 buffer 访问模式，推断内存层级 |
| 2 | `BufferShapeCollector()` | 收集 buffer 形状信息 | buffer scope | buffer shapes (初步) | 为后续 Pass 提供形状信息 |
| 3 | `tir.transform.BindTarget(target)` | 绑定 Target 信息 | IR | target attr | 记录编译目标（Ascend NPU） |
| 4 | `HostProcesser()` | Host 端数据处理 | IR | processed host data | 处理 CPU 端数据准备 |
| 5 | `tir.transform.Simplify()` | 简化 IR 表达式 | IR | simplified IR | 算术简化、常量折叠 |
| 6 | `AscendLowerParallelToVector()` | Parallel 循环 → Vector 指令 | simplified IR | vectorized IR | **核心 Pass**：将高级 Parallel 原语 lowering 到 Vector 指令 |
| 7 | `LayoutInference()` | 推断 fragment/shared memory layout | vectorized IR | layout annotations | 分析数据布局，推断最优 layout |
| 8 | `CollectBufferShapes()` | 再次收集 buffer 形状 | layout annotations | `buffer_shapess` | **关键输出**：为 Phase 2 提供 buffer 形状 |
| 9 | `LowerTileOp()` | Tile 操作 → 底层 IR | buffer shapes | lowered tile ops | **核心 Pass**：将 `T.copy`、`T.matmul` 等 lowering 到具体硬件操作 |
| 10 | `LegalizeVectorizedLoop()` | 合法化向量化循环 | lowered tile ops | legalized loops | 确保向量化循环符合硬件约束 |
| 11 | `LegalizeSafeMemoryAccess()` | 安全内存访问检查 | legalized loops | safe memory IR | 检查内存访问是否越界、是否符合硬件规范 |
| 12 | `tir.transform.Simplify()` | 再次简化 | safe memory IR | final Phase 1 IR | 清理冗余 IR，为 Phase 2 准备 |

### Phase 1 关键 Pass 说明

#### `AscendInferBufferScope`

- **功能**：推断 buffer 应该分配在哪个内存层级
- **核心逻辑**：
  1. 分析 buffer 的访问模式（读/写、频率、大小）
  2. 根据 Ascend 硬件内存层级（GM/L1/UB/L0A/L0B/L0C）推断最优 scope
  3. 标注 buffer 的 scope 属性
- **重要性**：Phase 2 的 `CrossCorePipeline`、`CombineCV` 依赖此 Pass 的输出

#### `AscendLowerParallelToVector`

- **功能**：将 Developer 模式的 `T.Parallel` lowering 到 Vector 指令
- **核心逻辑**：
  1. 分析 Parallel 循环的迭代模式
  2. 推断 Vector 指令的执行方式（广播、向量运算、归约）
  3. 生成 Vector 指令序列
- **重要性**：Developer 模式的关键 Pass，决定了如何执行 Vector 计算

#### `LowerTileOp`

- **功能**：将 Tile DSL 原语 lowering 到底层 IR
- **核心逻辑**：
  1. 识别 Tile 原语：`T.copy`、`T.matmul`、`T.add`、`T.max` 等
  2. 根据硬件特性生成对应的底层操作
  3. 插入必要的同步和约束
- **重要性**：核心 Lowering Pass，直接对应 DSL API

#### `CollectBufferShapes`

- **功能**：收集所有 buffer 的形状信息
- **核心逻辑**：
  1. 遍历所有 buffer 定义
  2. 记录每个 buffer 的形状（维度、大小）
  3. 输出 `buffer_shapess` attr（Map<Var, Array<PrimExpr>>）
- **重要性**：Phase 2 的 `AscendMemoryPlanning` 依赖此输出进行内存规划

---

## Phase 2: OptimizeForTarget（后端优化）

### 总体目标

针对 Ascend 硬件特性进行性能优化，生成高效的机器码。

### Pass 列表（按执行顺序）

| 步骤 | Pass | 功能 | 输入依赖 | 输出供给 | 关键逻辑 |
|------|------|------|---------|---------|---------|
| 1 | `tir.transform.PlanAndUpdateBufferAllocationLocation()` | Buffer 分配位置规划 | Phase 1 IR | buffer allocation plan | 确定每个 buffer 在代码中的分配位置 |
| 2 | `CrossCorePipeline()` | 跨核流水线规划 | buffer scope, allocation plan | cross-core pipeline | **核心 Pass**：规划 Cube-Vector 核间流水线 |
| 3 | `CombineCV()` | 分离 Cube/Vector 操作 | cross-core pipeline | separated CV ops | 将操作分离为 Cube 和 Vector 两部分 |
| 4 | `PipelinePlanning()` | 流水线 layout 推断 | separated CV ops | pipeline layout | 推断流水线中每个阶段的 layout |
| 5 | `InjectSoftwarePipeline()` | 软件流水线注入 | pipeline layout | software pipeline | 注入软件流水线，提升吞吐量 |
| 6 | `AscendLowerOpaqueBlock()` | Block IR → 可执行 IR | software pipeline | executable IR | 将 Block IR lowering 到可执行形式 |
| 7 | `tir.transform.NarrowDataType(32)` | 数据类型缩窄 | executable IR | narrowed data types | 缩窄数据类型以减少内存占用 |
| 8 | `ConfigIndexBitwidth()` | 索引位宽配置 | narrowed data types | configured indices | 配置索引变量的位宽 |
| 9 | `Flatten2DBuffer()` | Buffer 扁平化到 2D | configured indices | 2D buffers | 将多维 buffer 扁平化为 2D |
| 10 | `FlattenBuffer()` | Buffer 扁平化到 1D | 2D buffers | 1D buffers | 将 2D buffer 扁平化为 1D |
| 11 | `tir.transform.Simplify()` | 简化 | 1D buffers | simplified IR | 清理扁平化后的冗余 IR |
| 12 | `VectorizeLoop()` | 循环向量化（可配置） | simplified IR | vectorized loops | 将循环转换为向量指令 |
| 13 | `AscendStorageRewrite(is_npu)` | 存储重写优化 | vectorized loops | optimized storage | **核心 Pass**：优化内存访问模式，共享存储 |
| 14 | `tir.transform.UnrollLoop()` | 循环展开 | optimized storage | unrolled loops | 展开小循环以提升性能 |
| 15 | `tir.transform.RenormalizeSplitPattern()` | 重规范化分割模式 | unrolled loops | renormalized patterns | 规范化循环分割模式 |
| 16 | `tir.transform.Simplify()` | 简化 | renormalized patterns | simplified IR | 清理展开后的冗余 IR |
| 17 | `tir.transform.RemoveNoOp()` | 移除空操作 | simplified IR | no-op removed | 删除无实际作用的 IR |
| 18 | `tir.transform.RewriteUnsafeSelect()` | 重写不安全 select | no-op removed | safe select | 重写可能导致硬件异常的 select |
| 19 | `tir.transform.HoistIfThenElse()` | 提升 if-then-else | safe select | hoisted conditionals | 提升 if-then-else 以减少分支开销 |
| 20 | `AscendMemoryPlanning()` | 内存规划 | `buffer_shapess` | `address_map`, `size_map` | **关键 Pass**：规划 buffer 地址，输出地址映射 |
| 21 | `AscendSyncInsert()` | 同步插入 | `address_map`, `size_map` | final IR with syncs | **最后一环**：插入同步指令 |

### Phase 2 关键 Pass 说明

#### `CrossCorePipeline`

- **功能**：规划 Cube-Vector 核间流水线
- **核心逻辑**：
  1. 分析操作的 Cube/Vector 属性
  2. 规划 Cube 和 Vector 核的协作方式
  3. 确定流水线的阶段划分
- **重要性**：Ascend 多核架构的核心优化，决定性能上限

#### `CombineCV`

- **功能**：将操作分离为 Cube 和 Vector 两部分
- **核心逻辑**：
  1. 识别哪些操作应该在 Cube 核执行（如 matmul）
  2. 识别哪些操作应该在 Vector 核执行（如 element-wise）
  3. 分离操作并插入跨核数据传递
- **重要性**：配合 `CrossCorePipeline` 实现核间协作

#### `InjectSoftwarePipeline`

- **功能**：注入软件流水线
- **核心逻辑**：
  1. 分析循环的迭代依赖
  2. 识别可以流水化的阶段
  3. 重构循环为流水线形式
- **重要性**：提升吞吐量的关键优化

#### `AscendStorageRewrite`

- **功能**：优化内存访问模式，实现存储共享
- **核心逻辑**：
  1. 分析 buffer 的生命周期和访问模式
  2. 识别可以共享的 buffer
  3. 重写存储分配，减少内存占用
- **重要性**：内存优化核心 Pass，直接影响内存占用和性能

#### `AscendMemoryPlanning`

- **功能**：规划 buffer 的地址分配
- **核心逻辑**：
  1. 使用 `buffer_shapess`（来自 Phase 1）
  2. 计算每个 buffer 的地址和大小
  3. 输出 `address_map` 和 `size_map`
- **重要性**：Phase 2 的最后关键 Pass，为 `AscendSyncInsert` 提供地址信息

#### `AscendSyncInsert`

- **功能**：插入同步指令
- **核心逻辑**：
  1. 使用 `address_map` 和 `size_map`（来自 `AscendMemoryPlanning`）
  2. 分析操作的依赖关系
  3. 在必要位置插入同步（`T.barrier_all`、`T.set_flag`、`T.wait_flag`）
- **重要性**：Pipeline 的最后一环，确保执行正确性

---

## Pass 注册和调用机制

### C++ 注册

```cpp
// 文件: src/transform/<pass_name>.cc
TVM_REGISTER_GLOBAL("tl.transform.AscendSyncInsert")
    .set_body_typed(AscendSyncInsert);
```

### Python 加载

```python
# 文件: tilelang/transform/_ffi_api.py
import tvm._ffi
tvm._ffi._init_api("tl.transform", __name__)
```

### Python Wrapper

```python
# 文件: tilelang/transform/__init__.py
def AscendSyncInsert(target: Target, platform: str):
    """Auto insert sync for Ascend."""
    return _ffi_api.AscendSyncInsert(target, platform)
```

### Pipeline 调用

```python
# 文件: tilelang/engine/phase.py:79-105
def OptimizeForTarget(mod, target, platform):
    """Optimize the TIR for the target platform."""
    # ... (调用所有 Phase 2 Pass)
    mod = AscendMemoryPlanning()(mod)
    mod = AscendSyncInsert(target, platform)(mod)
    return mod
```

---

## 配置机制

### PassContext 配置

```python
# 文件: tilelang/jit/kernel.py:223
with tvm.transform.PassContext(opt_level=3, config=pass_configs):
    # Pass 配置通过 PassContext 传递
```

### 配置键定义

```python
# 文件: tilelang/transform/pass_config.py
class PassConfigKey(str, Enum):
    TL_ASCEND_AUTO_SYNC = "tl.ascend_auto_sync"
    """Enable/disable AscendSyncInsert. Default: False"""
    
    TL_ASCEND_MEMORY_PLANNING = "tl.ascend_memory_planning"
    """Enable/disable AscendMemoryPlanning. Default: False"""
    
    TL_ASCEND_AUTO_CV_COMBINE = "tl.ascend_auto_cv_combine"
    """Enable/disable CombineCV. Default: False"""
    
    TL_ASCEND_AUTO_CV_SYNC = "tl.ascend_auto_cross_core_sync"
    """Enable/disable CrossCorePipeline. Default: False"""
```

### Pass 内部读取配置

```cpp
// 文件: src/transform/ascend_sync_insert.cc:65-69
bool ascend_auto_sync = 
    ctx->GetConfig<Bool>(kAscendAutoSync, Bool(false)).value();
if (!ascend_auto_sync) {
  return f;  // 配置为 false 时跳过该 Pass
}
```

---

## 关键文件路径

| 类别 | 文件路径 | 说明 |
|-----|---------|------|
| **Pipeline 定义** | `tilelang/engine/phase.py` | LowerAndLegalize 和 OptimizeForTarget |
| **编译入口** | `tilelang/engine/lower.py` | `lower()` 函数，调用两阶段 Pipeline |
| **Pass 注册（C++）** | `src/transform/*.cc` | 44 个 Pass 的 C++ 实现 |
| **Pass Wrapper（Python）** | `tilelang/transform/__init__.py` | Python API 封装 |
| **配置键定义** | `tilelang/transform/pass_config.py` | PassConfigKey 枚举 |
| **公共工具** | `src/transform/common/*.h` | collector, operation_config, attr 等 |

---

## 扩展指南

### 如何添加新 Pass 到 Pipeline？

1. **确定阶段归属**：判断 Pass 属于 Phase 1 还是 Phase 2
2. **分析依赖关系**：识别上游 Pass 的输出和下游 Pass 的输入
3. **定位插入位置**：根据依赖关系确定在哪个位置插入
4. **实现 Pass**：C++ 实现 + Python Wrapper + Pipeline 集成

**详细定位指南请查阅：** `new-pass-placement-guide.md`

### 如何修改现有 Pipeline？

1. **调整 Pass 顺序**：修改 `tilelang/engine/phase.py`
2. **添加配置选项**：修改 `tilelang/transform/pass_config.py`
3. **修改 Pass 实现**：修改 `src/transform/<pass_name>.cc`

---

## 参考资料

- **Pass 详细注册表**：`.agents/skills/tilelang-pass-analyzer/references/pass-registry-ascend.md`
- **依赖关系图**：`pass-dependency-graph.md`
- **新 Pass 定位指南**：`new-pass-placement-guide.md`
- **架构总览**：`.agents/skills/tilelang-custom-skill/architecture.md`