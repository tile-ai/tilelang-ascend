# 模式对比与选择策略

## 两种模式定位

| 维度 | Developer 模式（自动化） | Expert 模式（手动控制） |
|------|-------------------------|------------------------|
| **定位** | Hardware-Aware with Tile Library | Hardware-Aware with Thread Primitives |
| **目标用户** | 对 AI 芯片内存层次有基本了解的开发者 | 对底层硬件特性有深入理解的专家 |
| **内存分配** | `T.alloc_shared` / `T.alloc_fragment`（编译器自动映射） | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C`（显式指定） |
| **计算表达** | `T.Parallel` + 符号运算（`+`, `-`, `*`, `/`, `T.exp` 等） | `T.tile.add` / `T.tile.exp` / `T.tile.max` 等 |
| **执行作用域** | 编译器自动分离 Cube/Vector（无需指定） | 显式 `with T.Scope("C"):` 和 `with T.Scope("V"):` |
| **同步控制** | 自动（通过 pass_configs 开关） | 手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag` / `T.set_cross_flag` |
| **pass_configs** | 需要开启多项自动化开关 | 通常全部关闭或不设置 |
| **代码复杂度** | 低 | 高 |
| **性能上限** | 良好 | 极致 |
| **跨平台兼容** | 理论上可跨架构 | 特定于 Ascend 平台 |

---

## 选择决策树

```
用户需求
  ├─ 快速原型 / 算法验证 / 不熟悉硬件 → Developer 模式
  ├─ 纯 Vector 算子（elementwise, softmax, layernorm 等） → Developer 模式
  ├─ 简单 Cube 算子（基础 GEMM） → Developer 模式
  ├─ Cube + Vector 融合算子（matmul_add, flash_attention 等） → Developer 模式（优先）或混合模式
  ├─ 性能关键路径的极致优化 → Expert 模式
  ├─ 需要精确控制流水线同步 → Expert 模式
  ├─ 需要手动双缓冲 / 多级流水 → Expert 模式
  └─ 用户明确指定模式 → 按用户要求
```

---

## 推荐：Developer 模式

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

---

## 选择 Expert 模式

**仅在以下场景选择 Expert 模式：**
- 性能关键路径需要极致优化
- 需要精确控制流水线同步时机
- 需要手动双缓冲和多级流水线
- 需要使用 `T.annotate_address` 手动规划内存布局
- 需要使用 `T.use_swizzle` 等高级特性
- Developer 模式无法满足性能要求

---

## 混合编程

实践中最常见的方式：**Developer 模式处理主体逻辑**，Expert 模式的扩展接口补充 Developer 模式暂不支持的操作。

可在 Developer 模式中直接使用的 Expert 扩展 API：
- `T.tile.fill(buffer, value)` — 初始化填充
- `T.tile.cast(dst, src, mode, count)` — 精度转换
- `T.tile.broadcast(dst, src, tmp)` — 广播
- `T.tile.axpy(dst, src, scalar)` — 向量乘加
- `T.reduce_max` / `T.reduce_sum` / `T.reduce_min` — 归约操作
- `T.tile.compare` / `T.tile.select` — 比较与条件选择

详见 [混合模式指南](mixed-mode-guide.md)

---

## pass_configs 速览

### Developer 模式（4 个开关全开）

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,    # 自动 Cube/Vector 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
}
```

> 纯 Vector 算子（无 GEMM）可只开 `AUTO_SYNC` + `MEMORY_PLANNING`

### Expert 模式

```python
@tilelang.jit(out_idx=[-1])  # 无 pass_configs，或全部设为 False
```

详见 [pass_configs 配置说明](convert-passconfigs.md)
