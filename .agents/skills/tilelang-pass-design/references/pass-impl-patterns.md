# Pass 实现模式参考

本文档提供 TileLang-Ascend Pass 的典型实现模式，供 Pass 设计时参考。

---

## 1. 父类选择指南

### 1.1 父类对比

| 父类 | 用途 | 适用 Pass 类型 | 特点 |
|------|------|----------------|------|
| `IRMutatorWithAnalyzer` | 修改 IR 结构 | IR 变换类 Pass | 提供 `MutateFunc()` 和表达式分析器 |
| `StmtExprVisitor` | 遍历 IR、收集信息 | 信息收集类 Pass | 不修改 IR，只访问节点 |
| `StmtExprMutator` | 简单 IR 变换 | 简单变换类 Pass | 提供 `VisitStmt_()` 和 `VisitExpr_()` |

### 1.2 选择决策树

```
Pass 是否修改 IR？
├─ 是 → 是否需要表达式分析器？
│   ├─ 是 → IRMutatorWithAnalyzer
│   └─ 否 → StmtExprMutator
└─ 否 → StmtExprVisitor（仅遍历和收集）
```

---

## 2. IR 变换类 Pass 模板

### 2.1 基本模板

```cpp
// 文件: src/transform/<pass_name>.cc

#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/arith/analyzer.h>
#include "common/attr.h"  // 常用 attr 定义

namespace tvm {
namespace tl {

class MyIRTransformPass : public arith::IRMutatorWithAnalyzer {
public:
  // Pass 入口方法（静态方法，供外部调用）
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 1. 读取配置（可选）
    bool config_enabled = ctx->GetConfig<Bool>(kMyPassConfig, Bool(false)).value();
    if (!config_enabled) {
      return f;  // 配置为 false 时跳过
    }
    
    // 2. 读取输入 attrs（如有依赖）
    auto input_attr = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess");
    if (!input_attr.defined()) {
      LOG(WARNING) << "buffer_shapess not found, skipping MyPass";
      return f;
    }
    
    // 3. 创建变换器实例并执行
    MyIRTransformPass mutator(f, ctx, input_attr.value());
    PrimFunc new_f = mutator.MutateFunc(f);
    
    // 4. 设置输出 attrs（如有输出）
    if (mutator.HasOutput()) {
      new_f = new_f.WithAttrs({{"output_attr", mutator.GetOutput()}});
    }
    
    return new_f;
  }
  
private:
  // 构造函数
  MyIRTransformPass(PrimFunc f, PassContext ctx, Map<Var, Array<PrimExpr>> input_attr)
      : IRMutatorWithAnalyzer(f->body), input_attr_(input_attr) {}
  
  // 成员变量
  Map<Var, Array<PrimExpr>> input_attr_;
  Map<Buffer, PrimExpr> output_data_;  // 输出数据（可选）
  
  // 核心 Visit 方法（重写需要处理的节点类型）
  Stmt VisitStmt_(const ForNode* op) final {
    // 处理 For 循环
    // ...
    Stmt body = VisitStmt(op->body);  // 递归处理子节点
    // ...
    return For(op->loop_var, op->min, op->extent, op->kind, body, op->annotations);
  }
  
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    // 处理 BufferStore（写 buffer）
    // ...
    return BufferStore(op->buffer, VisitExpr(op->value), op->indices);
  }
  
  Expr VisitExpr_(const BufferLoadNode* op) final {
    // 处理 BufferLoad（读 buffer）
    // ...
    return BufferLoad(op->buffer, op->indices);
  }
  
  Stmt VisitStmt_(const EvaluateNode* op) final {
    // 处理 Evaluate（执行表达式）
    // ...
    return Evaluate(op->value);
  }
  
  // 辅助方法
  bool IsTargetBuffer(Buffer buffer) {
    // 检查 buffer 是否为目标 buffer
    // ...
  }
  
  bool HasOutput() const { return output_data_.size() > 0; }
  Map<Buffer, PrimExpr> GetOutput() const { return output_data_; }
};

// Pass 注册函数
tvm::transform::Pass MyIRTransformPass() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return MyIRTransformPass::Substitute(std::move(f), ctx);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MyIRTransformPass", {});
}

// TVM 全局注册（供 Python 调用）
TVM_REGISTER_GLOBAL("tl.transform.MyIRTransformPass")
    .set_body_typed(MyIRTransformPass);

// 配置键注册（可选）
static constexpr const char *kMyPassConfig = "tl.my_pass";
TVM_REGISTER_PASS_CONFIG_OPTION(kMyPassConfig, Bool);

} // namespace tl
} // namespace tvm
```

### 2.2 典型 Visit 方法实现示例

#### 处理 For 循环

```cpp
Stmt VisitStmt_(const ForNode* op) final {
  // 检查是否为目标循环类型（如 Parallel）
  if (op->kind == ForKind::Parallel) {
    // 执行变换逻辑
    Stmt transformed_body = TransformParallelLoop(op);
    return For(op->loop_var, op->min, op->extent, 
               ForKind::Serial, transformed_body, op->annotations);
  }
  
  // 非目标循环，递归处理子节点
  Stmt body = VisitStmt(op->body);
  return For(op->loop_var, op->min, op->extent, op->kind, body, op->annotations);
}
```

#### 处理 BufferStore

```cpp
Stmt VisitStmt_(const BufferStoreNode* op) final {
  // 检查 buffer scope
  std::string scope = op->buffer.scope();
  if (scope == "local.L0C") {
    // 对 L0C buffer 执行特殊处理
    // ...
  }
  
  // 递归处理 value 和 indices
  Expr value = VisitExpr(op->value);
  Array<PrimExpr> indices = MutateArray(op->indices);
  return BufferStore(op->buffer, value, indices);
}
```

---

## 3. 信息收集类 Pass 模板

### 3.1 基本模板

```cpp
// 文件: src/transform/<pass_name>.cc

namespace tvm {
namespace tl {

class MyInfoCollectorPass : public StmtExprVisitor {
public:
  static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
    // 创建收集器实例
    MyInfoCollectorPass collector(f);
    
    // 执行遍历（不修改 IR）
    collector.VisitStmt(f->body);
    
    // 获取收集的信息
    Map<Buffer, Array<PrimExpr>> collected_info = collector.GetCollectedInfo();
    
    // 设置输出 attrs（信息收集类 Pass 通常有输出）
    return f.WithAttrs({{"collected_attr", collected_info}});
  }
  
private:
  MyInfoCollectorPass(PrimFunc f) : StmtExprVisitor() {}
  
  // 收集的数据
  Map<Buffer, Array<PrimExpr>> buffer_info_;
  std::unordered_map<Buffer, std::string> buffer_scopes_;
  
  // Visit 方法（只访问，不修改）
  void VisitStmt_(const AllocateNode* op) final {
    // 记录 buffer 分配信息
    buffer_info_.Set(op->buffer, op->extents);
    buffer_scopes_[op->buffer] = op->scope;
    
    // 继续遍历子节点
    VisitStmt(op->body);
  }
  
  void VisitExpr_(const BufferLoadNode* op) final {
    // 记录 buffer 访问信息
    // ...
    VisitExpr(op->buffer);
    for (auto idx : op->indices) {
      VisitExpr(idx);
    }
  }
  
  // 辅助方法
  Map<Buffer, Array<PrimExpr>> GetCollectedInfo() const { return buffer_info_; }
};

// Pass 注册（同上）
tvm::transform::Pass MyInfoCollectorPass() { ... }
TVM_REGISTER_GLOBAL("tl.transform.MyInfoCollectorPass").set_body_typed(MyInfoCollectorPass);

} // namespace tl
} // namespace tvm
```

---

## 4. 典型 Pass 实现模式

### 4.1 AscendSyncInsert 模式（复杂 IR 变换）

**特点**：
- 多类协作（`AscendSyncInsert` + `ForLoopUnroller` + `LoopRebuilder`）
- 循环展开 + 分析 + 重建

**核心流程**：
```cpp
class AscendSyncInsert : public IRMutatorWithAnalyzer {
  // 1. PreprocessUnrollForLoops() - 循环展开预处理
  // 2. VisitStmt_(EvaluateNode) - 分析内存依赖，插入同步
  // 3. MergeAndRebuildForLoops() - 合并同步，重建循环
};
```

**适用场景**：需要复杂分析和多步处理的 IR 变换。

### 4.2 AscendMemoryPlanning 模式（信息收集 + 计算）

**特点**：
- 继承 `StmtExprVisitor`（不修改 IR）
- 收集 buffer 信息 + 计算地址分配
- 输出 attrs 供后续 Pass 使用

**核心流程**：
```cpp
class AscendMemoryPlanner : public StmtExprVisitor {
  // 1. Substitute() - Pass 入口
  // 2. VisitStmt_() - 收集 buffer 和 shape 信息
  // 3. GetAddressMap() - 计算地址分配
  // 4. 输出 address_map 和 size_map attrs
};
```

**适用场景**：信息收集类 Pass，为后续 Pass 提供数据。

### 4.3 CrossCorePipeline 模式（多类协作）

**特点**：
- 主类 + 辅助类（`CrossCoreDetector` + `LoopAnalyzer` + `LoopRewriter`）
- 分离检测、分析、重写职责

**核心流程**：
```cpp
class CrossCorePipeline : public IRMutatorWithAnalyzer {
  // 1. CrossCoreDetector - 检测跨核流水线特征
  // 2. LoopAnalyzer - 分析 Cube/Vector 操作分布
  // 3. LoopRewriter - 重写为多 stage 流水线
};
```

**适用场景**：职责分离、多阶段处理的复杂 Pass。

---

## 5. Attr 读写模式

### 5.1 读取 Attr

```cpp
// 必需 attrs（缺失时报错或跳过）
auto required_attr = f->GetAttr<Map<Var, Array<PrimExpr>>>("buffer_shapess");
if (!required_attr.defined()) {
  LOG(WARNING) << "buffer_shapess not found";
  return f;
}

// 可选 attrs（缺失时使用默认值）
auto optional_attr = f->GetAttr<Bool>("some_bool", Bool(false));
```

### 5.2 设置 Attr

```cpp
// 设置单个 attr
new_f = f.WithAttr("output_attr", output_data);

// 设置多个 attrs
new_f = f.WithAttrs({
  {"address_map", address_map},
  {"size_map", size_map}
});
```

### 5.3 常用 Attr 名称

| Attr 名称 | 类型 | 产生 Pass | 使用 Pass |
|-----------|------|-----------|-----------|
| `buffer_shapess` | `Map<Var, Array<PrimExpr>>` | `CollectBufferShapes` | `AscendMemoryPlanning` |
| `address_map` | `Map<Buffer, PrimExpr>` | `AscendMemoryPlanning` | `AscendSyncInsert` |
| `size_map` | `Map<Buffer, PrimExpr>` | `AscendMemoryPlanning` | `AscendSyncInsert` |
| `buffer scope annotations` | `Map<Buffer, String>` | `AscendInferBufferScope` | `CrossCorePipeline`, `CombineCV` |

---

## 6. 配置键注册模式

### 6.1 C++ 配置键

```cpp
// 定义配置键
static constexpr const char *kMyPassConfig = "tl.my_pass";

// 注册配置选项
TVM_REGISTER_PASS_CONFIG_OPTION(kMyPassConfig, Bool);
// 或其他类型
TVM_REGISTER_PASS_CONFIG_OPTION(kMyPassConfig, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kMyPassConfig, String);
```

### 6.2 Pass 内读取配置

```cpp
static PrimFunc Substitute(PrimFunc f, PassContext ctx) {
  // 读取配置（带默认值）
  bool enabled = ctx->GetConfig<Bool>(kMyPassConfig, Bool(false)).value();
  int threshold = ctx->GetConfig<Integer>(kMyPassThreshold, Integer(100)).value();
  
  if (!enabled) {
    return f;  // 配置为 false 时跳过 Pass
  }
  
  // 使用配置值执行逻辑
  // ...
}
```

### 6.3 Python 配置键定义

```python
# tilelang/transform/pass_config.py
class PassConfigKey(str, Enum):
    TL_MY_PASS = "tl.my_pass"
    """Enable/disable MyPass. Default: False"""
    
    TL_MY_PASS_THRESHOLD = "tl.my_pass_threshold"
    """Threshold value for MyPass. Default: 100"""
```

---

## 7. Python Wrapper 模式

### 7.1 基本模式

```python
# tilelang/transform/__init__.py

import tvm._ffi
tvm._ffi._init_api("tl.transform", __name__)  # 加载 C++ 注册的函数

def MyPass():
    """MyPass description.
    
    This pass does X, Y, Z.
    
    Returns:
        Pass: The registered pass.
    """
    return _ffi_api.MyPass()  # 调用 C++ 注册的函数
```

### 7.2 带参数的模式

```python
def MyPassWithParams(target: Target, platform: str):
    """MyPass with parameters.
    
    Args:
        target: The compilation target.
        platform: The platform name ("npu" or other).
    
    Returns:
        Pass: The registered pass.
    """
    return _ffi_api.MyPassWithParams(target, platform)
```

---

## 8. 文件组织建议

### 8.1 新 Pass 文件结构

```
src/transform/
├── my_pass.cc              # 主实现文件
└── common/                 # 公共工具（如需）
    ├── attr.h              # Attr 定义
    ├── collector.h         # 信息收集器
    └── operation_config.h  # 操作配置
```

### 8.2 代码行数参考

| Pass 类型 | 典型行数 | 示例 |
|-----------|----------|------|
| 简单信息收集 | 100-300 行 | `BufferShapeCollector` |
| 简单 IR 变换 | 300-500 行 | `Flatten2DBuffer` |
| 复杂 IR 变换 | 500-1500 行 | `AscendSyncInsert` (1559行) |
| 多类协作 | 1000-2000 行 | `CrossCorePipeline` (~1200行) |

---

## 9. 注意事项

### 9.1 IR 变换原则

- **语义保持**：变换后的 IR 必须保持原有计算语义
- **类型一致**：变换后的类型必须与预期一致
- **依赖完整**：确保上游依赖数据可用

### 9.2 性能考虑

- **避免深度递归**：对于大型 IR，考虑使用迭代或限制递归深度
- **缓存中间结果**：重复计算时使用缓存
- **最小化 IR 修改**：只修改必要部分，避免全量重建

### 9.3 错误处理

- **配置缺失**：配置键缺失时使用默认值，并记录警告
- **依赖缺失**：上游 attrs 缺失时根据情况报错或跳过
- **IR 异常**：遇到异常 IR 结构时记录警告并跳过处理

---

## 10. 参考 Pass 列表

| Pass | 文件 | 行数 | 特点 |
|------|------|------|------|
| `AscendSyncInsert` | `ascend_sync_insert.cc` | 1559 | 多类协作、循环展开 |
| `AscendMemoryPlanning` | `ascend_memory_planning.cc` | 884 | 信息收集、地址计算 |
| `CrossCorePipeline` | `cross_core_pipeline.cc` | ~1200 | 多类协作、流水线规划 |
| `CombineCV` | `ascend_combinecv.cc` | ~700 | Cube/Vector 分离 |
| `AscendLowerParallelToVector` | `ascend_lower_parallel_to_vector.cc` | ~2000 | Parallel lowering |
| `AscendStorageRewrite` | `ascend_storage_rewrite.cc` | ~2200 | 存储优化 |
| `InferAllocScope` | `ascend_infer_buffer_scope.cc` | ~900 | Scope 推断 |

**建议**：设计新 Pass 时，参考功能相似的现有 Pass 实现结构。