# 性能优化模式

## 1. Block 大小选择

block_M 影响 UB 内存占用和并行度：

| dtype | 推荐 block_M | 原因 |
|-------|-------------|------|
| float16/bfloat16 | 64 | UB 空间充足，并行度好 |
| float32 | 32 | 元素占 4 字节，UB 空间紧张 |

选择逻辑：
在固定的内存预算下，通过启发式规则，主动增大 N 方向的分块尺寸（block_N），并确保它对齐到矩阵维度，以充分发挥硬件向量化访问的潜力。同时，用预算反推 M 方向的分块尺寸。
```python
def _get_optimized_tiling(M, N, block_M_in, block_N_in, vec_num):
    budget = block_M_in * block_N_in
    ideal_n = budget // 16

    block_N = min(N // 2, ideal_n)
    if block_N < 128:
        block_N = 128 if N >= 256 else N

    while N % block_N != 0:
        block_N -= 1
        if block_N <= 0:
            block_N = 1
            break

    block_M = budget // block_N
    if M % block_M != 0:
        block_M = block_M_in if M % block_M_in == 0 else vec_num
    return block_M, block_N
```

## 2. 广播优化

归一化阶段需要对 [ROWS, block_N] 范围内的每一个元素都乘以该行对应的倒数 RMS 值。broadcast 一次性将每行的标量值复制成连续的一整行，生成一个完整的 [ROWS, block_N] 瓦片 inv_rms_tile。后续的乘法运算只需从该瓦片中连续读取数据，与输入数据 a_ub_cast_0/1 的访存模式完全一致。这消除了循环内的条件广播逻辑：

```python
for by in T.serial(n_num // 2):
    # Buffer 0
    col_off_0 = (by * 2) * block_N
    if need_cast:
        T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_0)
        T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
    else:
        T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_cast_
    T.tile.mul(a_ub_cast_0, a_ub_cast_0, inv_rms_tile)
```

## 3. 数学指令替换

在计算公式 $x / \sqrt{\text{mean}(x^2) + \epsilon}$ 时，避免使用 sqrt 然后再做除法（div），这不仅慢而且消耗更多指令周期。应直接使用 NPU 的硬件级平方根倒数指令 rsqrt：

```python
# 错误做法：求 sqrt 后再求倒数
# T.tile.sqrt(sqrt_ub, mean_sq_eps_ub)
# T.tile.div(inv_ub, 1.0, sqrt_ub)

# 性能优化做法：一步到位使用 rsqrt
T.tile.rsqrt(rsqrt_ub, mean_sq_eps_ub)
```

## 4. Host 端预处理与常量折叠

将以下操作移到 host 端：
- **eps 标量注入**：在 Host 端配置好 float32 的 eps，直接通过 kernel 参数作为标量传入，不要在 kernel 里从 GM 读或者现场构造。
- **归一化因子**：不要在 kernel 里面做浮点除法（`sum / hidden_size`），在 Host 端提前算好 `inv_hidden_size = 1.0 / hidden_size`，传进 kernel 用乘法替代除法（`sum * inv_hidden_size`）。


## 5. 避免的反模式

| 反模式 | 后果 | 正确做法 |
|--------|------|---------|
| Kernel 内部做除法求平均 | 除法指令耗时高 | Host 传入 `1.0/hidden_size`，使用向量乘法 |
| 不使用 `rsqrt` 指令 | 增加指令开销与访存压力 | 统一使用 `T.tile.rsqrt` |