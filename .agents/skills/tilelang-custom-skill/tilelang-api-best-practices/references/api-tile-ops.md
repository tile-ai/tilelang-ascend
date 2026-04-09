# Tile 扩展原语（Expert 模式）

## 概述

`T.tile.xxx` 系列接口直接触发 Tile 级的 Vector 操作指令，是 Expert 模式下的编程方式。与 `T.Parallel` + 符号 API 相比，提供更底层的硬件控制。

## 数学计算

### 基础算术

| API | 功能 | 示例 |
|-----|------|------|
| `T.tile.add(dst, src0, src1)` | dst = src0 + src1 | `T.tile.add(c_ub, a_ub, b_ub)` 或 `T.tile.add(c_ub, a_ub, 2)` |
| `T.tile.sub(dst, src0, src1)` | dst = src0 - src1 | `T.tile.sub(c_ub, a_ub, b_ub)` |
| `T.tile.mul(dst, src0, src1)` | dst = src0 * src1 | `T.tile.mul(c_ub, a_ub, 2)` |
| `T.tile.div(dst, src0, src1)` | dst = src0 / src1 | `T.tile.div(c_ub, a_ub, b_ub)` |
| `T.tile.max(dst, src0, src1)` | dst = max(src0, src1) | `T.tile.max(c_ub, a_ub, 2)` |
| `T.tile.min(dst, src0, src1)` | dst = min(src0, src1) | `T.tile.min(c_ub, a_ub, b_ub)` |

> src1 可以是 buffer 类型或 scalar 标量。

### 单目数学运算

| API | 功能 | 示例 |
|-----|------|------|
| `T.tile.exp(dst, src0)` | dst = exp(src0) | `T.tile.exp(c_ub, a_ub)` |
| `T.tile.ln(dst, src0)` | dst = ln(src0) | `T.tile.ln(c_ub, a_ub)` |
| `T.tile.abs(dst, src0)` | dst = abs(src0) | `T.tile.abs(c_ub, a_ub)` |
| `T.tile.reciprocal(dst, src0)` | dst = 1/src0 | `T.tile.reciprocal(c_ub, a_ub)` |
| `T.tile.sqrt(dst, src0)` | dst = √src0 | `T.tile.sqrt(c_ub, a_ub)` |
| `T.tile.rsqrt(dst, src0)` | dst = 1/√src0 | `T.tile.rsqrt(c_ub, a_ub)` |
| `T.tile.relu(dst, src0)` | dst = max(0, src0) | `T.tile.relu(c_ub, a_ub)` |
| `T.tile.leaky_relu(dst, src0, scalar)` | Leaky ReLU | `T.tile.leaky_relu(c_ub, a_ub, 0.01)` |
| `T.tile.axpy(dst, src0, scalar)` | dst = scalar * src0 + dst | `T.tile.axpy(c_ub, a_ub, scalar)` |
| `T.tile.sin(dst, src0)` | dst = sin(src0) | `T.tile.sin(c_ub, a_ub)` |
| `T.tile.cos(dst, src0)` | dst = cos(src0) | `T.tile.cos(c_ub, a_ub)` |

### 逻辑运算

| API | 功能 |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | 左移操作 |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | 右移操作 |

## 比较操作

### T.tile.compare(dst, src0, src1, mode)

逐元素比较，结果为 bit mask（1=true，0=false）。

**mode 取值**：`"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")    # tensor vs tensor
T.tile.compare(c_ub, a_ub, 1.0, "GT")      # tensor vs scalar
```

## 选择操作

### T.tile.select(dst, selMask, src0, src1, selMode)

根据 mask 从两个源中选择元素。mask bit=1 选 src0，bit=0 选 src1。

**selMode 取值**：
- `"VSEL_CMPMASK_SPR"`：根据 compare mask 选择
- `"VSEL_TENSOR_SCALAR_MODE"`：tensor 和 scalar 之间选择
- `"VSEL_TENSOR_TENSOR_MODE"`：两个 tensor 之间选择

```python
T.tile.select(c_ub, selmask_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")
T.tile.select(c_ub, selmask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
```

## 精度转换

### T.tile.cast(dst, src, mode, count)

**mode 取值**：`"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`

```python
T.tile.cast(b_ub, a_ub, "CAST_RINT", 4096)
```

## 数据操作

| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 用 value 填充 buffer |
| `T.tile.createvecindex(dst, first_value)` | 创建从 first_value 开始的向量索引 |
| `T.tile.transpose(dst, src)` | 16×16 矩阵转置 |
| `T.tile.gather(dst, src, offset, base_addr)` | 按偏移收集数据 |
| `T.tile.arith_progression(buf, first, diff, count)` | 生成等差数列 |

## 排序操作

| API | 功能 |
|-----|------|
| `T.tile.sort(dst, src, indices, repeat_time)` | 降序排序 |
| `T.tile.merge_sort(dst, src, block_size, block_num, is_copy)` | 合并排序 |
| `T.tile.topk(dst, src, block_size)` | 获取 Top-K |

## 两种编程范式对比

```python
# 方式一：T.Parallel + 符号 API（推荐，跨平台兼容）
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    b_ub[i, j] = T.exp(a_ub[i, j])

# 方式二：T.tile 扩展原语（Expert 模式，直接触发硬件指令）
T.tile.exp(b_ub, a_ub)
```
