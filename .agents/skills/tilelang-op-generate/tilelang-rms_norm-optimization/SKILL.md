---
name: tilelang-rms_norm-optimization
description: TileLang Ascend RMSNorm 算子性能优化与问题修复经验总结。涵盖 gather-mask 优化、FP32 累加、多核调度、host 端预计算、sin/cos 广播、interleave/half 旋转模式等。当用户开发 RMSNorm 算子、优化 TileLang vector kernel 性能、或遇到类似的编译/精度/性能问题时触发。
---

# TileLang RMSNorm 算子优化与问题修复指南

## 适用场景

- 开发 RMSNorm (Root Mean Square Layer Normalization) 算子
- 优化 TileLang Ascend vector kernel 性能
- 遇到精度、UB内存规划等相关问题

## 核心架构决策

### 1. Ping-Pong 流水线（推荐）

为了防止 AI Core 在等待数据从 Global Memory (GM) 搬运到 UB 时处于闲置状态，代码巧妙地实现了手动软流水：

- 在 `T.serial(n_num // 2)` 循环中，偶数块加载到 `a_ub_0`，奇数块加载到 `a_ub_1`。
- 配合顶部的宏定义 `TL_ASCEND_AUTO_CV_SYNC: True`，TileLang 的底层编译器会自动将 "数据搬运 (Copy)" 和 "向量计算 (Compute)" 并行化。当计算缓冲区 0 时，缓冲区 1 正在后台进行异步加载。
- 末尾通过单独的 `if n_num % 2 != 0:` 分支来处理奇数尾块。

### 2. 精度与防溢出控制（必须）

当输入数据类型为 `float16` 或 `bfloat16` 时，直接计算平方和 `(x^2)` 极易超出数据类型的表示范围从而导致溢出。**必须在计算平方前将其 Cast 到** `float32`，计算完毕后再 Cast 回去：

```python
need_cast = x.dtype in (TL.float16, TL.bfloat16)

# 1. 从 GM 搬运到 UB
T.copy(x[row_idx, :], x_ub)

# 2. 精度提升以防平方和溢出
if need_cast:
    T.tile.cast(x_f32_ub, x_ub)
else:
    x_f32_ub = x_ub  # float32 直接引用

# 3. 计算平方
T.tile.mul(sq_ub, x_f32_ub, x_f32_ub)
```


## 性能优化方法

详见 [optimization-patterns.md](references/optimization-patterns.md)

## 问题修复方法

详见 [bugfix-patterns.md](references/bugfix-patterns.md)