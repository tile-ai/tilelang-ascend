# 归约操作

## 概述

TileLang 提供按指定维度进行归约的操作，包括求和、求最大值和求最小值。所有归约 API 都需要提供临时 buffer。

## API

### T.reduce_sum(buffer, out, tmp, dim, real_shape=[0, 0])

对输入 buffer 按指定维度求和。

**参数**：
- `buffer`：输入 buffer（2D）
- `out`：目的输出 buffer
- `tmp`：临时申请 buffer（用于内部中间计算）
- `dim`：reduce 轴（-1 表示最后一维）
- `real_shape`：实际 shape（可选，[0,0] 表示使用 buffer 原始 shape）

**示例**：
```python
tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N], "uint8")
T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
```

### T.reduce_max(buffer, out, tmp, dim, real_shape=[0, 0])

对输入 buffer 按指定维度求最大值。

**示例**：
```python
tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N], "uint8")
T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
```

### T.reduce_min(buffer, out, tmp, dim, real_shape=[0, 0])

对输入 buffer 按指定维度求最小值。

**示例**：
```python
tmp = T.alloc_ub((2 * sub_block_M, N), "uint8")
T.reduce_min(a_ub, b_ub, tmp, dim=-1)
```

## 归约轴说明

对 shape 为 (M, N) 的 2D 矩阵：
- `dim=0`：沿第一维归约，输出 shape 为 (N,)
- `dim=-1`：沿最后一维归约，输出 shape 为 (M,)

## 临时 buffer 大小

临时 buffer 需要额外空间存储中间变量，推荐分配方式：
```python
tmp_size = 3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N
tmp_ub = T.alloc_ub([tmp_size], "uint8")
```

## 最佳实践

1. **临时 buffer 必须提供**：所有 reduce 操作都需要 tmp buffer，忘记分配会导致错误
2. **开启自动内存规划**以复用临时 buffer 空间
3. **在 Softmax 中的典型用法**：
   ```python
   # 求行最大值
   T.reduce_max(scores_ub, row_max_ub, tmp_ub, dim=-1)
   # 计算 exp
   for i, j in T.Parallel(block_M, block_N):
       scores_ub[i, j] = T.exp(scores_ub[i, j] - row_max_ub[i])
   # 求行和
   T.reduce_sum(scores_ub, row_sum_ub, tmp_ub, dim=-1)
   ```
