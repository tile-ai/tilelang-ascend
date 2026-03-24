# Element-wise 运算

## 概述

TileLang 通过 `T.Parallel` 调度原语结合符号化 API 实现逐元素运算。这是 Developer 模式的推荐编程方式，具有良好的可读性和跨平台移植性。

## T.Parallel 语法

### 基础语法

```python
# 1D
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + b_ub[j]

# 2D
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

每次 `(i, j)` 迭代独立执行，表示该区域可并行。

### 复杂表达式

编译器会自动分配临时缓冲区来分解复杂表达式：

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i, j] + a_ub[i, j] / b_ub[i, j]
```

> 建议开启自动缓冲区复用功能以避免空间浪费。

## 支持的运算

### 浮点单目运算

| 运算 | 表达 | 说明 |
|------|------|------|
| 绝对值 | `T.abs(x)` | \|x\| |
| 指数 | `T.exp(x)` | e^x |
| 对数 | `T.log(x)` | log(x) |
| 开平方 | `T.sqrt(x)` | √x |
| 平方根倒数 | `T.rsqrt(x)` | 1/√x |
| ReLU | `T.max(a, 0)` | max(x, 0) |

### 浮点双目运算

| 运算 | 表达 | 说明 |
|------|------|------|
| 加法 | `a + b` | c = a + b |
| 减法 | `a - b` | c = a - b |
| 乘法 | `a * b` | c = a * b |
| 除法 | `a / b` | c = a / b |
| 最小值 | `T.min(a, b)` | min(a, b) |
| 最大值 | `T.max(a, b)` | max(a, b) |

### 整形运算

| 运算 | 表达 | 说明 |
|------|------|------|
| 位非 | `~x` | 按位取非 |
| 左移 | `x << s` | x 左移 s 位 |
| 右移 | `x >> s` | x 右移 s 位 |
| 与 | `a & b` | 按位与 |
| 或 | `a \| b` | 按位或 |

## 使用场景

### 向量-标量运算

```python
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + 1
```

### 行广播

```python
# a_ub.shape = (block_M // VEC_NUM, block_N)
# b_ub.shape = (block_M // VEC_NUM,)
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]
```

### 列切分模式

```python
for i in range(block_M // VEC_NUM):      # 行顺序
    for j in T.Parallel(block_N):         # 列并行
        c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

### GM → UB 拷贝 + 计算

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    C[bx * block_M + vid * block_M // VEC_NUM + i, by * block_N + j] = T.exp(a_ub[i, j])
```

## 最佳实践

1. **优先使用 T.Parallel + 符号 API**，而非直接调用 `T.tile.xxx`，以保持代码可移植性
2. **开启自动缓冲区复用**：复杂表达式会自动分配临时 buffer
3. **支持 break/continue**：在 T.Parallel 循环中可使用
4. **条件语句**：支持 if/elif/else，条件应为 TIR expression
