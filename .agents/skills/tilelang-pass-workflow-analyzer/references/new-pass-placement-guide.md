# 新 Pass 定位指南

本文档提供添加新 Pass 到 TileLang-Ascend Pipeline 的完整指南，帮助确定 Pass 应该在哪个位置添加。

---

## 定位原则

添加新 Pass 时，需要遵循以下四大原则：

### 1. 功能归属原则

判断 Pass 的功能属于哪个阶段：

| Pass 功能特征 | 阶段归属 | 理由 |
|-------------|---------|------|
| **DSL Lowering** | Phase 1 | 将高级 DSL 转换为底层 IR，属于前端标准化 |
| **IR 合法化** | Phase 1 | 确保 lowered IR 符合规范，属于前端标准化 |
| **信息收集** | Phase 1 | 收集 IR 信息供 Phase 2 使用，属于前端准备 |
| **硬件优化** | Phase 2 | 针对 Ascend 硬件特性优化，属于后端特化 |
| **内存优化** | Phase 2 | 利用硬件内存层级优化，属于后端特化 |
| **流水线优化** | Phase 2 | 多核流水线规划，属于后端特化 |

### 2. 数据依赖原则

识别 Pass 需要哪些输入数据，确保上游 Pass 已经执行：

| 输入数据 | 上游 Pass | 约束 |
|---------|-----------|------|
| **buffer scope** | `AscendInferBufferScope` (Phase 1, 步骤 1) | 必须在此 Pass 后执行 |
| **buffer_shapess** | `CollectBufferShapes` (Phase 1, 步骤 8) | 必须在此 Pass 后执行 |
| **address_map** | `AscendMemoryPlanning` (Phase 2, 步骤 20) | 必须在此 Pass 后执行 |
| **size_map** | `AscendMemoryPlanning` (Phase 2, 步骤 20) | 必须在此 Pass 后执行 |
| **pipeline layout** | `PipelinePlanning` (Phase 2, 步骤 4) | 必须在此 Pass 后执行 |
| **cross-core annotations** | `CrossCorePipeline` (Phase 2, 步骤 2) | 必须在此 Pass 后执行 |

### 3. 输出供给原则

识别 Pass 产生哪些输出数据，确保下游 Pass 可以访问：

| 输出数据 | 下游 Pass | 约束 |
|---------|-----------|------|
| **buffer scope** | `CrossCorePipeline`, `CombineCV` | 必须在这两个 Pass 前执行 |
| **buffer_shapess** | `AscendMemoryPlanning` | 必须在此 Pass 前执行 |
| **address_map** | `AscendSyncInsert` | 必须在此 Pass 前执行 |
| **size_map** | `AscendSyncInsert` | 必须在此 Pass 前执行 |

### 4. 语义范围原则

判断 Pass 是平台无关还是平台特定：

| Pass 语义范围 | 阶段归属 | 理由 |
|-------------|---------|------|
| **平台无关** | Phase 1 | 不依赖特定硬件特性，属于通用优化 |
| **Ascend 特定** | Phase 2 | 针对 Ascend NPU 的硬件特性优化 |
| **GPU/CPU 兼容** | Phase 1 | 如果 Pass 需要支持多种平台，放在 Phase 1 |

---

## 典型定位场景

根据 Pass 的功能特征，推荐以下位置：

### 场景 1: IR 合法化检查

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：IR 合法化检查 | Phase 1 末尾 | 确保 lowered IR 正确性，在进入 Phase 2 前验证 |
| **依赖**：lowered IR | 在 `LowerTileOp` 后 | 需要 lowered tile ops |
| **输出**：合法化报告 | 供调试使用 | 不影响后续 Pass |
| **语义**：平台无关 | Phase 1 | 不依赖 Ascend 特性 |

**具体位置**：Phase 1，在 `tir.transform.Simplify`（步骤 12）后，或替换为新的合法化 Pass。

**示例**：
```python
# tilelang/engine/phase.py
def LowerAndLegalize(mod, target):
    # ... 现有 Pass (步骤 1-11)
    mod = tir.transform.Simplify()(mod)  # 步骤 12
    # 新增 Pass
    mod = MyIRLegalizationPass()(mod)
    return mod
```

### 场景 2: 新 Tile 操作 Lowering

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：新 Tile 操作 lowering | Phase 1，`LowerTileOp` 后 | 紧跟核心 lowering pass，处理新的 DSL 原语 |
| **依赖**：buffer shapes, layout | 在 `CollectBufferShapes` 后 | 需要形状和 layout 信息 |
| **输出**：lowered tile ops | 供后续合法化 Pass 使用 | 输出 lowered IR |
| **语义**：平台无关（如通用操作）或 Ascend 特定（如硬件原语） | Phase 1 或 Phase 2 | 根据操作的语义范围决定 |

**具体位置**：
- 如果是**通用 Tile 操作**：Phase 1，在 `LowerTileOp`（步骤 9）后，`LegalizeVectorizedLoop`（步骤 10）前。
- 如果是**Ascend 特定 Tile 操作**：可以考虑在 Phase 2，`AscendLowerOpaqueBlock`（步骤 6）后。

**示例**：
```python
# tilelang/engine/phase.py
def LowerAndLegalize(mod, target):
    # ... 现有 Pass (步骤 1-8)
    mod = LowerTileOp()(mod)  # 步骤 9
    # 新增 Pass
    mod = MyNewTileOpLoweringPass()(mod)
    mod = LegalizeVectorizedLoop()(mod)  # 步骤 10
    # ...
```

### 场景 3: 内存优化

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：内存优化（如 buffer 共享、地址重排） | Phase 2，`AscendStorageRewrite` 后 | 利用 memory planning 信息，属于后端内存优化 |
| **依赖**：buffer scope, buffer_shapess | Phase 1 已提供 | 跨阶段依赖已满足 |
| **输出**：优化后的内存分配 | 供 `AscendMemoryPlanning` 使用 | 影响 memory planning |
| **语义**：Ascend 特定 | Phase 2 | 针对 Ascend 内存层级优化 |

**具体位置**：Phase 2，在 `AscendStorageRewrite`（步骤 13）后，`AscendMemoryPlanning`（步骤 20）前。

**注意**：
- 如果新 Pass 影响 buffer 的数量或大小，需要在 `AscendMemoryPlanning` 前执行。
- 如果新 Pass 不影响 buffer 数量和大小，可以在 `AscendMemoryPlanning` 后执行（如地址重排）。

**示例**：
```python
# tilelang/engine/phase.py
def OptimizeForTarget(mod, target, platform):
    # ... 现有 Pass (步骤 1-13)
    mod = AscendStorageRewrite(is_npu)(mod)  # 步骤 13
    # 新增 Pass（影响 buffer 数量/大小）
    mod = MyMemoryOptimizationPass()(mod)
    mod = tir.transform.UnrollLoop()(mod)  # 步骤 14
    # ... (步骤 15-19)
    mod = AscendMemoryPlanning()(mod)  # 步骤 20
    # 新增 Pass（不影响 buffer 数量/大小）
    mod = MyAddressReorderPass()(mod)
    mod = AscendSyncInsert(target, platform)(mod)  # 步骤 21
    return mod
```

### 场景 4: 同步优化

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：同步优化（如减少同步次数、优化同步位置） | Phase 2，`AscendSyncInsert` 前 | 为 sync insert 提供信息，或替代 sync insert |
| **依赖**：address_map, size_map | 在 `AscendMemoryPlanning` 后 | 需要地址和大小信息 |
| **输出**：同步建议或替代同步策略 | 供 `AscendSyncInsert` 使用或替代 | 影响同步插入 |
| **语义**：Ascend 特定 | Phase 2 | 针对 Ascend 多核同步优化 |

**具体位置**：Phase 2，在 `AscendMemoryPlanning`（步骤 20）后，`AscendSyncInsert`（步骤 21）前，或替代 `AscendSyncInsert`。

**示例**：
```python
# tilelang/engine/phase.py
def OptimizeForTarget(mod, target, platform):
    # ... 现有 Pass (步骤 1-20)
    mod = AscendMemoryPlanning()(mod)  # 步骤 20
    # 新增 Pass
    mod = MySyncOptimizationPass()(mod)
    mod = AscendSyncInsert(target, platform)(mod)  # 步骤 21
    return mod
```

### 场景 5: 流水线优化

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：流水线优化（如调整流水线阶段、优化核间协作） | Phase 2，流水线相关 Pass 附近 | 与现有流水线 Pass 协同 |
| **依赖**：buffer scope, cross-core annotations | 在 `AscendInferBufferScope` 和 `CrossCorePipeline` 后 | 需要内存层级和跨核信息 |
| **输出**：优化的流水线配置 | 供 `InjectSoftwarePipeline` 使用 | 影响软件流水线 |
| **语义**：Ascend 特定 | Phase 2 | 针对 Ascend 多核流水线 |

**具体位置**：
- 如果是**调整流水线阶段**：Phase 2，在 `PipelinePlanning`（步骤 4）后，`InjectSoftwarePipeline`（步骤 5）前。
- 如果是**优化核间协作**：Phase 2，在 `CombineCV`（步骤 3）后，`PipelinePlanning`（步骤 4）前。

**示例**：
```python
# tilelang/engine/phase.py
def OptimizeForTarget(mod, target, platform):
    # ... 现有 Pass (步骤 1-3)
    mod = CombineCV()(mod)  # 步骤 3
    # 新增 Pass
    mod = MyCrossCoreOptimizationPass()(mod)
    mod = PipelinePlanning()(mod)  # 步骤 4
    # ...
```

### 场景 6: 信息收集 Pass

| Pass 功能特征 | 推荐位置 | 理由 |
|-------------|---------|------|
| **功能**：信息收集（如收集新的 buffer 属性、操作统计） | Phase 1 或 Phase 2，根据收集时机 | 为下游 Pass 提供信息 |
| **依赖**：根据收集内容决定 | 确保需要的信息已产生 | 依赖上游 Pass |
| **输出**：收集的信息 attrs | 供下游 Pass 使用 | 下游 Pass 必须在后面 |
| **语义**：通常平台无关 | Phase 1 或 Phase 2 | 根据收集的信息类型决定 |

**具体位置**：根据收集的信息类型：
- **收集 buffer 信息**：Phase 1，在 `AscendInferBufferScope` 或 `CollectBufferShapes` 后。
- **收集操作统计**：Phase 1，在 `LowerTileOp` 后。
- **收集流水线信息**：Phase 2，在 `CrossCorePipeline` 或 `CombineCV` 后。

**示例**：
```python
# tilelang/engine/phase.py
def LowerAndLegalize(mod, target):
    # ... 现有 Pass (步骤 1-7)
    mod = LayoutInference()(mod)  # 步骤 7
    # 新增 Pass
    mod = MyBufferInfoCollectorPass()(mod)
    mod = CollectBufferShapes()(mod)  # 步骤 8
    # ...
```

---

## 定位决策流程

### Step 1: 分析 Pass 功能

填写以下表格：

| 问题 | 答案 |
|-----|------|
| **Pass 的主要功能是什么？** | [描述] |
| **Pass 修改 IR 还是收集信息？** | [修改/收集] |
| **Pass 是平台无关还是 Ascend 特定？** | [平台无关/Ascend 特定] |
| **Pass 属于哪类优化？** | [Lowering/合法化/内存/流水线/同步/其他] |

### Step 2: 分析依赖关系

填写以下表格：

| 问题 | 答案 |
|-----|------|
| **Pass 需要哪些输入数据？** | [buffer scope/buffer_shapess/address_map/size_map/其他] |
| **这些数据由哪个 Pass 产生？** | [Pass 名称] |
| **Pass 产生哪些输出数据？** | [输出数据名称] |
| **这些数据由哪个 Pass 使用？** | [Pass 名称] |

### Step 3: 确定阶段归属

根据 Step 1 和 Step 2 的答案：

| 如果... | 则... |
|--------|-------|
| Pass 功能是 DSL Lowering | Phase 1 |
| Pass 功能是 IR 合法化 | Phase 1 |
| Pass 功能是硬件优化 | Phase 2 |
| Pass 需要的数据来自 Phase 1 | 可在 Phase 1 或 Phase 2 |
| Pass 需要的数据来自 Phase 2 | 必须在 Phase 2 |
| Pass 输出的数据被 Phase 2 使用 | Phase 1 或 Phase 2（取决于数据类型） |
| Pass 输出的数据被 Phase 1 使用 | Phase 1 |

### Step 4: 确定具体位置

根据阶段归属和依赖关系，确定在 Pipeline 的哪个步骤：

| 方法 | 说明 |
|-----|------|
| **依赖优先** | Pass 必须在产生其输入数据的 Pass 后执行 |
| **供给优先** | Pass 必须在使用其输出数据的 Pass 前执行 |
| **功能相邻** | 将 Pass 放在功能相似的 Pass 附近 |

### Step 5: 验证定位

检查以下约束：

| 约束 | 检查方法 |
|-----|---------|
| **依赖数据可用** | 确认上游 Pass 在此 Pass 前执行 |
| **输出数据可访问** | 确认下游 Pass 在此 Pass 后执行 |
| **顺序正确** | 确认不影响其他 Pass 的依赖关系 |
| **逻辑连贯** | 确认 Pass 功能与周围 Pass 功能相关 |

---

## 案例分析

### 案例 1: 添加一个 Buffer 大小优化 Pass

**场景**：用户想添加一个 Pass，优化 buffer 的大小（减少内存占用）。

**Step 1: 分析功能**
- 主要功能：优化 buffer 大小
- 修改 IR：修改 buffer 定义
- 平台特定：Ascend 特定（利用 Ascend 内存层级）
- 优化类型：内存优化

**Step 2: 分析依赖**
- 输入数据：`buffer_shapess`（buffer 形状信息）
- 输入数据来源：`CollectBufferShapes` (Phase 1, 步骤 8)
- 输出数据：优化后的 buffer 定义
- 输出数据使用：`AscendMemoryPlanning` (Phase 2, 步骤 20) 会使用 buffer 信息

**Step 3: 确定阶段**
- 功能是硬件优化 → Phase 2
- 输入数据来自 Phase 1 → 可在 Phase 2
- 输出数据被 Phase 2 使用 → Phase 2

**结论**：Phase 2

**Step 4: 确定位置**
- 输入依赖：必须在 `CollectBufferShapes` 后 → Phase 1 已完成
- 输出供给：必须在 `AscendMemoryPlanning` 前 → Phase 2 步骤 20 前
- 功能相邻：内存优化 Pass → 在 `AscendStorageRewrite`（步骤 13）附近

**推荐位置**：Phase 2，在 `AscendStorageRewrite`（步骤 13）后，`AscendMemoryPlanning`（步骤 20）前。

**Step 5: 验证**
- 依赖数据可用：`buffer_shapess` 来自 Phase 1，已满足 ✓
- 输出数据可访问：`AscendMemoryPlanning` 在后面 ✓
- 顺序正确：不影响其他 Pass ✓
- 逻辑连贯：内存优化 Pass 连续编排 ✓

**最终建议**：
```python
# tilelang/engine/phase.py
def OptimizeForTarget(mod, target, platform):
    # ... (步骤 1-13)
    mod = AscendStorageRewrite(is_npu)(mod)  # 步骤 13
    # 新增 Pass
    mod = BufferSizeOptimizationPass()(mod)
    mod = tir.transform.UnrollLoop()(mod)  # 步骤 14
    # ... (步骤 15-19)
    mod = AscendMemoryPlanning()(mod)  # 步骤 20
    # ...
```

### 案例 2: 添加一个 L0C Buffer 数据布局优化 Pass

**场景**：用户想添加一个 Pass，优化 L0C buffer 的数据布局（提升矩阵计算性能）。

**Step 1: 分析功能**
- 主要功能：优化 L0C buffer 数据布局
- 修改 IR：修改 L0C buffer 的 layout
- 平台特定：Ascend 特定（L0C 是 Ascend 硬件特有）
- 优化类型：内存优化 + 流水线优化

**Step 2: 分析依赖**
- 输入数据：buffer scope（知道哪个 buffer 是 L0C）
- 输入数据来源：`AscendInferBufferScope` (Phase 1, 步骤 1)
- 输出数据：优化后的 L0C layout
- 输出数据使用：`CrossCorePipeline` (Phase 2, 步骤 2) 会考虑 L0C layout

**Step 3: 确定阶段**
- 功能是硬件优化 → Phase 2
- 输入数据来自 Phase 1 → 可在 Phase 2
- 输出数据被 Phase 2 使用 → Phase 2

**结论**：Phase 2

**Step 4: 确定位置**
- 输入依赖：必须在 `AscendInferBufferScope` 后 → Phase 1 已完成
- 输出供给：应该在 `CrossCorePipeline` 前或后，取决于是否影响流水线规划
- 功能相邻：L0C 相关优化 → 在 Cube/Vector 分离附近

**考虑两种方案**：
1. **方案 A**：在 `CrossCorePipeline` 前（影响流水线规划）
   - 位置：Phase 2，步骤 2 前（替换或修改步骤 1）
2. **方案 B**：在 `CrossCorePipeline` 后（不影响流水线规划）
   - 位置：Phase 2，步骤 2 后，`CombineCV` 前

**推荐方案**：方案 A（影响流水线规划更合理）

**Step 5: 验证**
- 依赖数据可用：buffer scope 来自 Phase 1 ✓
- 输出数据可访问：`CrossCorePipeline` 在后面 ✓
- 顺序正确：在流水线规划前 ✓
- 逻辑连贯：L0C layout 影响流水线规划 ✓

**最终建议**：
```python
# tilelang/engine/phase.py
def OptimizeForTarget(mod, target, platform):
    # ... (步骤 1)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)  # 步骤 1
    # 新增 Pass
    mod = L0CLayoutOptimizationPass()(mod)
    mod = CrossCorePipeline()(mod)  # 步骤 2
    # ...
```

### 案例 3: 添加一个 IR 正确性验证 Pass

**场景**：用户想添加一个 Pass，验证 Phase 1 输出的 IR 是否正确（调试辅助）。

**Step 1: 分析功能**
- 主要功能：验证 IR 正确性
- 修改 IR：不修改，只检查
- 平台无关：通用验证
- 优化类型：合法化/调试

**Step 2: 分析依赖**
- 输入数据：Phase 1 输出的 IR
- 输入数据来源：Phase 1 所有 Pass
- 输出数据：验证报告（供调试）
- 输出数据使用：调试使用，不影响后续 Pass

**Step 3: 确定阶段**
- 功能是 IR 合法化 → Phase 1
- 输入数据来自 Phase 1 → Phase 1
- 输出数据不影响后续 Pass → Phase 1

**结论**：Phase 1

**Step 4: 确定位置**
- 输入依赖：必须在 Phase 1 最后几个 Pass 后
- 输出供给：不影响后续 Pass，位置灵活
- 功能相邻：合法化 Pass → 在 Phase 1 末尾

**推荐位置**：Phase 1，在 `tir.transform.Simplify`（步骤 12）后，Phase 1 末尾。

**Step 5: 验证**
- 依赖数据可用：Phase 1 IR 已完成 ✓
- 输出数据可访问：调试报告，不需要后续 Pass ✓
- 顺序正确：不影响其他 Pass ✓
- 逻辑连贯：在 Phase 1 末尾验证 ✓

**最终建议**：
```python
# tilelang/engine/phase.py
def LowerAndLegalize(mod, target):
    # ... (步骤 1-11)
    mod = tir.transform.Simplify()(mod)  # 步骤 12
    # 新增 Pass
    mod = IRCorrectnessValidationPass()(mod)
    return mod
```

---

## 实现指南

### Step 1: C++ 实现

**文件位置**：`src/transform/<pass_name>.cc`

**基本结构**：
```cpp
namespace tvm {
namespace tl {

class MyNewPass : public arith::IRMutatorWithAnalyzer {  // 或 StmtExprVisitor
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 核心逻辑
    return f;
  }
  
private:
  // 内部实现方法
};

tvm::transform::Pass MyNewPass() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return MyNewPass::Substitute(std::move(f), ctx);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MyNewPass", {});
}

TVM_REGISTER_GLOBAL("tl.transform.MyNewPass")
    .set_body_typed(MyNewPass);

// 如果需要配置选项
static constexpr const char *kMyNewPassConfig = "tl.my_new_pass";
TVM_REGISTER_PASS_CONFIG_OPTION(kMyNewPassConfig, Bool);

} // namespace tl
} // namespace tvm
```

### Step 2: Python Wrapper

**文件位置**：`tilelang/transform/__init__.py`

**基本结构**：
```python
def MyNewPass():
    """MyNewPass description."""
    return _ffi_api.MyNewPass()
```

### Step 3: 配置键（可选）

**文件位置**：`tilelang/transform/pass_config.py`

**基本结构**：
```python
class PassConfigKey(str, Enum):
    TL_MY_NEW_PASS = "tl.my_new_pass"
    """Enable/disable MyNewPass. Default: False"""
```

### Step 4: Pipeline 集成

**文件位置**：`tilelang/engine/phase.py`

**基本结构**：
```python
# 根据定位指南确定的位置
def LowerAndLegalize(mod, target):
    # ... 现有 Pass
    mod = MyNewPass()(mod)  # 添加新 Pass
    # ... 现有 Pass
    return mod

# 或
def OptimizeForTarget(mod, target, platform):
    # ... 现有 Pass
    mod = MyNewPass()(mod)  # 添加新 Pass
    # ... 现有 Pass
    return mod
```

---

## 参考实现示例

### 示例 1: Buffer 大小优化 Pass

**功能**：优化 buffer 大小，减少内存占用。

**定位**：Phase 2，在 `AscendStorageRewrite` 后，`AscendMemoryPlanning` 前。

**C++ 实现要点**：
```cpp
class BufferSizeOptimizationPass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 读取 buffer_shapess
    auto buffer_shapess = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess").value();
    
    // 分析 buffer 使用情况
    // 优化 buffer 大小（如合并、共享）
    
    // 更新 buffer 定义
    return f;
  }
};
```

### 示例 2: L0C Layout 优化 Pass

**功能**：优化 L0C buffer 的数据布局，提升矩阵计算性能。

**定位**：Phase 2，在 `CrossCorePipeline` 前。

**C++ 实现要点**：
```cpp
class L0CLayoutOptimizationPass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 读取 buffer scope
    // 找到 L0C buffer
    
    // 分析 L0C 的访问模式
    // 优化 layout（如行主序 vs 列主序）
    
    // 更新 L0C buffer 的 layout 属性
    return f;
  }
};
```

### 示例 3: IR 正确性验证 Pass

**功能**：验证 Phase 1 输出的 IR 是否正确。

**定位**：Phase 1 末尾。

**C++ 实现要点**：
```cpp
class IRCorrectnessValidationPass : public StmtExprVisitor {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 验证 buffer scope 是否正确
    // 验证 buffer shapes 是否合理
    // 验证 lowered tile ops 是否符合规范
    
    // 输出验证报告（可选：打印或设置 attr）
    return f;
  }
};
```

---

## 注意事项

### 1. 避免破坏依赖链

添加新 Pass 时，确保不破坏现有的依赖关系：
- 如果新 Pass 需要某个数据，确保产生该数据的 Pass 在前面
- 如果新 Pass 产生某个数据，确保使用该数据的 Pass 在后面

### 2. 注意跨阶段依赖

跨阶段依赖（Phase 1 → Phase 2）尤其重要：
- Phase 1 的输出 attrs 必须正确传递到 Phase 2
- Phase 2 的 Pass 不能依赖 Phase 1 尚未产生的数据

### 3. Pass 顺序的影响

Pass 顺序直接影响编译结果：
- 顺序错误可能导致 Pass 缺少输入数据
- 顺序错误可能导致输出数据无法被下游 Pass 使用

### 4. 测试验证

添加新 Pass 后，务必测试：
- 测试 Pass 的正确性（功能测试）
- 测试 Pass 的依赖关系（顺序测试）
- 测试 Pass 的性能影响（性能测试）

---

## 参考资料

- **Pipeline 架构详解**：`pass-pipeline-overview.md`
- **依赖关系图**：`pass-dependency-graph.md`
- **Pass 注册表**：`.agents/skills/tilelang-pass-analyzer/references/pass-registry-ascend.md`
- **架构总览**：`.agents/skills/tilelang-custom-skill/architecture.md`