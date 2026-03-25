# 计算原语转换规则

## 总体原则

- **Developer 模式**：使用 `T.Parallel` 循环 + 符号 API（`+`, `-`, `*`, `/`, `T.exp` 等）
- **Expert 模式**：使用 `T.tile.xxx` 函数式调用（`T.tile.add`, `T.tile.exp` 等）
- **GEMM 和 Reduce 操作**：两种模式共用相同的 API（`T.gemm_v0`, `T.reduce_max` 等）

## 双目运算转换

### Expert → Developer

```python
# Expert 模式
T.tile.add(c_ub, a_ub, b_ub)

# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

| Expert | Developer (T.Parallel 内) | 功能 |
|--------|---------------------------|------|
| `T.tile.add(c, a, b)` | `c[i,j] = a[i,j] + b[i,j]` | 加法 |
| `T.tile.sub(c, a, b)` | `c[i,j] = a[i,j] - b[i,j]` | 减法 |
| `T.tile.mul(c, a, b)` | `c[i,j] = a[i,j] * b[i,j]` | 乘法 |
| `T.tile.div(c, a, b)` | `c[i,j] = a[i,j] / b[i,j]` | 除法 |
| `T.tile.max(c, a, b)` | `c[i] = T.max(a[i], b[i])` | 最大值 |
| `T.tile.min(c, a, b)` | `c[i] = T.min(a[i], b[i])` | 最小值 |

### 标量运算

```python
# Expert 模式
T.tile.mul(c_ub, a_ub, sm_scale)

# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * sm_scale
```

## 单目运算转换

```python
# Expert 模式
T.tile.exp(c_ub, a_ub)

# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = T.exp(a_ub[i, j])
```

| Expert | Developer (T.Parallel 内) | 功能 |
|--------|---------------------------|------|
| `T.tile.exp(dst, src)` | `T.exp(x)` | 指数 |
| `T.tile.ln(dst, src)` | `T.log(x)` | 对数 |
| `T.tile.abs(dst, src)` | `T.abs(x)` | 绝对值 |
| `T.tile.sqrt(dst, src)` | `T.sqrt(x)` | 平方根 |
| `T.tile.rsqrt(dst, src)` | `T.rsqrt(x)` | 平方根倒数 |
| `T.tile.relu(dst, src)` | `T.max(x, 0)` | ReLU |

## 行广播运算转换

**重要区别**：Developer 模式的 `T.Parallel` 原生支持行广播（2D buffer 与 1D buffer 运算），Expert 模式需要手动编写逐行循环。

```python
# Expert 模式 — 需要手动逐行循环
for h_i in range(block_M // VEC_NUM):
    T.tile.sub(c_ub[h_i, :], a_ub[h_i, :], m_i[h_i])

# Developer 模式 — 自动广播
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] - m_i[i]
```

## 数据填充

```python
# Expert 模式
T.tile.fill(c_ub, 0.0)

# Developer 模式
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = 0.0
```

> `T.tile.fill` 在两种模式中都可以使用，Developer 模式也常用它来初始化 buffer。

## GEMM — 两种模式共用

```python
# 两种模式完全相同
T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)
```

## Reduce — 两种模式共用

```python
# 两种模式完全相同
T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
```

## Developer 模式可调用的 Expert 扩展 API

以下 Expert API 可在 Developer 模式中直接调用（混合编程）：

| Expert API | 功能 | Developer 模式中是否可用 |
|-----------|------|------------------------|
| `T.tile.fill` | 填充 | 可用 |
| `T.tile.cast` | 精度转换 | 可用 |
| `T.tile.compare` | 逐元素比较 | 可用 |
| `T.tile.select` | 条件选择 | 可用 |
| `T.tile.sin/cos` | 三角函数 | 可用 |

> 这些操作在 Developer 模式中无等价的 `T.Parallel` 表达，建议直接调用。

## 转换注意事项

1. **T.Parallel 会自动分配临时 buffer**：复杂表达式会被拆解，而 `T.tile.xxx` 需要手动管理中间结果
2. **T.Parallel 支持 break/continue**：Expert 模式的 `T.tile.xxx` 不支持
3. **T.Parallel 中条件语句**：支持 `if/else`
4. **性能差异**：两种方式最终生成的硬件指令相同，性能理论上一致