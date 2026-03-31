---
name: tilelang-mode-conversion
description: TileLang Ascend 编程模式选择与转换指南。当用户询问 Developer/Expert 模式的区别、如何选择编程模式、需要将 Expert 模式算子转换为 Developer 模式、或需要确定算子开发应采用哪种模式时触发此 skill。
---

# TileLang Ascend Developer 与 Expert 编程模式

## 模式概览

TileLang Ascend 提供两种编程模式，位于编译降级流程的不同层级：

| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **目标用户** | 对 AI 芯片内存层次有基本了解的开发者 | 对 Ascend 底层硬件有深入理解的专家 |
| **核心理念** | 使用抽象化 Tile Library，编译器自动处理 | 显式控制存储层级、执行作用域、同步时机 |
| **代码简洁度** | 简洁，易于维护 | 复杂度高，平台绑定 |
| **跨平台兼容** | 理论上可跨架构兼容 | 特定于 Ascend 平台 |
| **性能控制** | 编译器自动优化 | 可进行极致性能调优 |

---

## 关键 API 差异

### 内存分配

| 场景 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| Cube 输入缓存 | `T.alloc_shared(shape, dtype)` | `T.alloc_L1(shape, dtype)` |
| Vector 工作空间 | `T.alloc_shared(shape, dtype)` | `T.alloc_ub(shape, dtype)` |
| Cube 累加器 | `T.alloc_fragment(shape, dtype)` | `T.alloc_L0C(shape, dtype)` |

> Developer 模式中，编译器根据 buffer 使用上下文自动映射到正确的存储层级。

### 计算表达

| 场景 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| 元素级加法 | `for i,j in T.Parallel(...): c[i,j] = a[i,j] + b[i,j]` | `T.tile.add(c, a, b)` |
| 元素级乘法 | `for i,j in T.Parallel(...): c[i,j] = a[i,j] * b[i,j]` | `T.tile.mul(c, a, b)` |
| 指数运算 | `for i,j in T.Parallel(...): c[i,j] = T.exp(a[i,j])` | `T.tile.exp(c, a)` |
| 行广播 | `for i,j in T.Parallel(...): c[i,j] = a[i,j] - m[i]` | `for h in range(...): T.tile.sub(c[h,:], a[h,:], m[h])` |

> Developer 模式的 `T.Parallel` 原生支持广播，Expert 模式需手动逐行处理。

### 同步与作用域

| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| Cube/Vector 分离 | 编译器自动（需 `AUTO_CV_COMBINE`） | 显式 `with T.Scope("C")` / `T.Scope("V")` |
| 核内同步 | 自动插入 `barrier_all`（需 `AUTO_SYNC`） | 手动 `T.barrier_all()` |
| 核间同步 | 自动插入 `set_cross_flag/wait_cross_flag`（需 `AUTO_CV_SYNC`） | 手动 `T.set_cross_flag()` / `T.wait_cross_flag()` |

### pass_configs 配置

**Developer 模式标准配置：**
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

**Expert 模式：** 通常不设 pass_configs，或只保留 `MEMORY_PLANNING`。

---

## 模式选择指南

### 推荐：Developer 模式

**适用场景：**
- 快速原型开发和算法验证
- 追求代码可读性和可维护性
- 不熟悉 Ascend 硬件细节
- 需要跨平台兼容性
- 大多数算子开发场景

**优势：**
- 代码量少，逻辑清晰
- 编译器自动处理同步和内存映射
- 更容易调试和维护
- 支持混合编程（可调用 Expert 扩展 API）

### 选择 Expert 模式

**仅在以下场景选择 Expert 模式：**
- 性能关键路径需要极致优化
- 需要精确控制流水线同步时机
- 需要使用 `T.annotate_address` 手动规划内存布局
- 需要使用 `T.use_swizzle` 等高级特性
- Developer 模式无法满足性能要求

### 混合编程

实践中最常见的是混合编程：Developer 模式处理主体逻辑，遇到 Developer 模式暂不支持的操作时调用 Expert 扩展接口（如 `T.tile.fill`、`T.tile.cast` 等）。

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def my_kernel(...):
    @T.prim_func
    def main(...):
        # Developer 模式主体
        a_ub = T.alloc_shared((block_M, block_N), dtype)
        
        # 调用 Expert 扩展 API（混合编程）
        T.tile.fill(a_ub, 0.0)
        
        # 继续使用 Developer 模式
        for i, j in T.Parallel(block_M, block_N):
            a_ub[i, j] = T.exp(a_ub[i, j])
```

---

## Expert → Developer 转换指南

当需要将 Expert 模式算子转换为 Developer 模式时，按以下步骤操作：

### 步骤 1：添加 pass_configs

在 `@tilelang.jit` 装饰器中添加完整 pass_configs：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

### 步骤 2：替换内存分配 API

| Expert API | Developer API |
|------------|---------------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` |

### 步骤 3：移除 T.Scope

删除所有 `with T.Scope("C"):` 和 `with T.Scope("V"):` 包裹，编译器会自动分离。

### 步骤 4：移除手动同步

删除所有：
- `T.barrier_all()`
- `T.set_cross_flag(...)` / `T.wait_cross_flag(...)`
- `T.set_flag(...)` / `T.wait_flag(...)`

### 步骤 5：转换计算原语

将 `T.tile.xxx` 函数调用转换为 `T.Parallel` + 符号 API：

```python
# Expert 模式
T.tile.add(c_ub, a_ub, b_ub)
T.tile.mul(c_ub, c_ub, sm_scale)
T.tile.exp(c_ub, a_ub)

# Developer 模式
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = c_ub[i, j] * sm_scale
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = T.exp(a_ub[i, j])
```

### 步骤 6：处理行广播

Expert 模式的逐行循环可简化为 `T.Parallel` 自动广播：

```python
# Expert 模式
for h_i in range(block_M):
    T.tile.sub(c_ub[h_i, :], a_ub[h_i, :], m_i[h_i])

# Developer 模式
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] - m_i[i]
```

### 步骤 7：移除 T.annotate_address

Developer 模式开启 `MEMORY_PLANNING` 后，编译器自动处理内存复用，无需手动指定地址。

### 步骤 8：处理 workspace

Expert 模式中 Cube → Vector 数据交换通过 GM 中转，Developer 模式同样需要 workspace tensor，但编译器自动处理同步。

---

## 转换检查清单

### Expert → Developer

- [ ] 添加完整 `pass_configs`（4 个开关全部开启）
- [ ] `T.alloc_L1` / `T.alloc_ub` → `T.alloc_shared`
- [ ] `T.alloc_L0C` → `T.alloc_fragment`
- [ ] 删除所有 `T.Scope("C")` / `T.Scope("V")`
- [ ] 删除所有 `T.barrier_all()`
- [ ] 删除所有 `T.set_cross_flag` / `T.wait_cross_flag`
- [ ] 删除所有 `T.set_flag` / `T.wait_flag`
- [ ] `T.tile.add/sub/mul/div` → `T.Parallel` + 符号运算
- [ ] `T.tile.exp/log/abs/sqrt` → `T.Parallel` + `T.exp/log/abs/sqrt`
- [ ] 逐行循环 → `T.Parallel` 自动广播
- [ ] 删除 `T.annotate_address`

---

## 详细参考

转换过程中如需更详细的 API 映射规则，请查阅：

- [内存分配转换详解](references/convert-memory.md)
- [计算原语转换详解](references/convert-compute.md)
- [同步与作用域转换详解](references/convert-sync-scope.md)
- [完整转换示例](references/convert-examples.md)

## 示例代码位置

- **Developer 模式示例**：`examples/developer_mode/`
- **Expert 模式示例**：`examples/gemm/example_gemm.py`、`examples/flash_attention/flash_attn_bhsd.py`