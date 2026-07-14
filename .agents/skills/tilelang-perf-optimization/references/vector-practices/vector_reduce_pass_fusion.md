# Vector 算子归约遍数融合（Reduce Pass Fusion）

本文档介绍 Vector 型算子中一种减少 GM 数据扫描遍数的优化方法。核心思想：将原本需要多次完整扫描归约维度的操作，通过数学等价变换合并为更少的扫描遍数，从而降低 GM↔UB 数据搬运量。

典型代表：Online Softmax（3-pass → 2-pass）。

参考实现：`examples/softmax/example_online_softmax.py`

---

## 适用场景

### 适用条件（全部满足）

| 编号 | 条件 | 说明 |
|------|------|------|
| C1 | 算子类型为 Vector（`IS_ASCEND_AIV`） | 归约维度沿列方向，数据需分块从 GM 搬入 UB |
| C2 | 归约维度远大于 UB 单次容量 | 列维度 N 无法一次全部搬入 UB，必须分块扫描 |
| C3 | 存在 ≥2 次完整归约维度扫描 | 如 softmax 的 max → sum → output 三遍扫描 |
| C4 | 相邻扫描之间存在数学可融合关系 | 后续扫描的结果可以用修正公式从前一轮推导，无需重算 |

### 不适用条件

| 编号 | 条件 | 原因 |
|------|------|------|
| X1 | 归约维度可一次装入 UB | 无需分块，不存在多遍扫描问题，直接计算即可 |
| X2 | 各遍扫描之间无数学关联 | 无法通过修正公式融合，强行合并会破坏正确性 |
| X3 | 算子本身只有 1 遍扫描 | 无融合空间 |

### 可泛化的算子模式

此优化不限于 Softmax，任何符合"先归约求统计量 → 再逐元素使用统计量"模式且归约维度需分块扫描的算子均可考虑：

| 算子 | 原始遍数 | 融合后遍数 | 融合方式 |
|------|---------|-----------|---------|
| Softmax | 3（max → sum → output） | 2（max+sum → output） | Online 修正 |
| LogSumExp | 2（max → sum） | 1（max+sum 同时计算） | 同 Online Softmax Pass 1 |
| Stable L2 Norm | 2（max → sum_sq） | 1（max+sum_sq 同时计算） | 修正公式类似 |
| 带数值稳定的归一化 | 3（max → sum → normalize） | 2（max+sum → normalize） | 同 Softmax |

---

## 约束条件

### 硬件约束

| 约束 | 说明 |
|------|------|
| UB 容量 | A2/A3 的 UB 预算为 192 KB，融合后需要额外存储 `prev_max`、`prev_sum` 等状态 buffer，需确保总 UB 占用不超限 |
| 对齐要求 | 所有 UB buffer 需 32B 对齐（fp32 为 8 元素，fp16/bf16 为 16 元素） |
| VEC_NUM | 核内有 2 个向量处理单元，`sub_block_M = block_M // 2` |

### 算法约束

| 约束 | 说明 |
|------|------|
| 修正公式精度 | `exp(m_old - m_new)` 在 fp16 下可能精度不足，中间计算必须提升到 fp32 |
| 初始值 | `prev_max` 初始化为 `-inf`，`prev_sum` 初始化为 `0`，不可用其他值 |
| pad_value | 非整除场景下，`T.copy` 的 `pad_value` 必须设为 `-inf`（不影响 max 运算） |
| 广播形状 | `prev_max`/`prev_sum` 为 `[rows, 1]`，参与逐元素运算前必须 broadcast 到 `[rows, block_N]` |

---

## 优化思路

### 问题本质：数据搬运是瓶颈

在 NPU Vector 核上，GM（全局内存）→ UB（核内缓冲）的搬运带宽是主要瓶颈。传统 softmax 对每行数据扫描 3 遍：

```
Pass 1: 读全行 → 求 max            → 搬运量 = M × N
Pass 2: 读全行 → 求 sum(exp(x-max)) → 搬运量 = M × N
Pass 3: 读全行 → 算 exp(x-max)/sum  → 搬运量 = M × N
                                    总计 = 3 × M × N
```

### 核心思路：合并 Pass 1 和 Pass 2

不是一整行求一个最大值再求和，而是**分块逐步处理**：每读入一小块数据，同时更新"当前最大值"和"当前指数和"。当最大值发生变化时，用修正系数调整之前累积的和。

```
初始化: m = -inf, s = 0
对每一块 tile_j:
    m_new = max(m, max(tile_j))                          ← 更新全局最大值
    s = s * exp(m - m_new) + sum(exp(x_i - m_new))       ← 修正旧和 + 新块贡献
    m = m_new
```

**直觉**：如果新块的最大值更大，之前累积的 exp 值都偏大了，乘以 `exp(m_old - m_new) < 1` 来"缩小"。

### 融合后

```
Pass 1: 读全行 → 同时求 max 和 sum    → 搬运量 = M × N
Pass 2: 读全行 → 算 exp(x-max)/sum    → 搬运量 = M × N
                                       总计 = 2 × M × N
```

搬运量减少 **33%**。

---

## 实现要点

### Pass 1 核心代码结构

```python
prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
T.tile.fill(prev_max, -T.infinity(cal_dtype))
T.tile.fill(prev_sum, 0.0)

for by in T.serial(n_num):
    T.copy(A[row_slice, by*block_N:(by+1)*block_N], a, pad_value=-T.infinity(cal_dtype))
    # cast 到 fp32（若输入为低精度）
    T.reduce_max(a_cal, tile_max, dim=-1)           # 当前块每行最大值
    T.tile.max(tile_max, prev_max, tile_max)         # 更新全局最大值
    T.tile.sub(tmp_exp, prev_max, tile_max)          # m_old - m_new
    T.tile.exp(tmp_exp, tmp_exp)                     # 修正系数
    T.tile.mul(tmp_exp, prev_sum, tmp_exp)           # 修正旧和
    T.tile.broadcast(tile_max_2d, tile_max)
    T.tile.sub(a_cal, a_cal, tile_max_2d)            # x_i - m_new
    T.tile.exp(a_cal, a_cal)                         # exp(x_i - m_new)
    T.reduce_sum(a_cal, tile_sum, dim=-1)            # 当前块指数和
    T.tile.add(prev_sum, tile_sum, tmp_exp)          # s = 修正旧和 + 新块贡献
    T.copy(tile_max, prev_max)
```

### Pass 2 核心代码结构

```python
T.tile.broadcast(prev_max_2d, prev_max)
T.tile.broadcast(prev_sum_2d, prev_sum)

for by in T.serial(n_num):
    T.copy(A[row_slice, by*block_N:(by+1)*block_N], a)
    T.tile.sub(a_cal, a_cal, prev_max_2d)            # x_i - m_N
    T.tile.exp(a_cal, a_cal)                         # exp(x_i - m_N)
    T.tile.div(a_cal, a_cal, prev_sum_2d)            # / s_N
    T.copy(a, B[row_slice, by*block_N:(by+1)*block_N])
```

### UB 预算估算

融合后 Pass 1 同时活跃的 buffer：

```
a           : sub_block_M × block_N × input_bytes
a_cal       : sub_block_M × block_N × cal_bytes
tile_max    : sub_block_M × 1 × cal_bytes
tile_max_2d : sub_block_M × block_N × cal_bytes
prev_max    : sub_block_M × 1 × cal_bytes
prev_sum    : sub_block_M × 1 × cal_bytes
tile_sum    : sub_block_M × 1 × cal_bytes
tmp_exp     : sub_block_M × 1 × cal_bytes
```

Pass 2 额外需要 `prev_max_2d` 和 `prev_sum_2d`（各 `sub_block_M × block_N × cal_bytes`），但开启 `MEMORY_PLANNING` 后 Pass 1 的部分临时 buffer 可被复用。

block_N 的选择应通过 UB 预算反推（参考 optimization-guide.md §2.11）。

---

## 验证方法

### 精度验证

```python
ref = torch.nn.functional.softmax(a, dim=1)
torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)
```

精度容差建议：

| dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-4 | 1e-4 |
| float16 | 1e-2 | 1e-3 |
| bfloat16 | 1e-2 | 1e-3 |

### 关键测试用例

| 场景 | 目的 |
|------|------|
| N 远大于 block_N（如 N=51200, block_N=128） | 验证多块融合的修正公式正确性 |
| N 不能被 block_N 整除 | 验证 pad_value=-inf 和余数块处理 |
| 全相同值输入 | 边界情况：max 始终不变，修正系数恒为 1 |
| 含极大/极小值 | 验证数值稳定性（不溢出、不 underflow） |
| M=1（单行） | 最小工作量，验证基础正确性 |

### 性能验证

```bash
msprof op --kernel-name="main_kernel" --output=./msprof_output python ./examples/softmax/example_online_softmax.py
```

对比基线（传统 3-pass softmax）关注：
- GM 读字节数应减少约 33%
- 总耗时应有对应下降（memory bound 场景下接近 33%）
- Vector 计算时间可能略增（多了修正系数的 exp/mul），但应远小于搬运节省

---

## 与其他优化的组合

Online Softmax（归约遍数融合）与以下优化正交，可叠加使用：

| 优化项 | 组合方式 | 参考 |
|--------|---------|------|
| Double Buffer | Pass 1/2 的列块循环内可做 MTE2↔V 流水 | optimization-guide.md §2.2 |
| 指令向量化 | broadcast + tile 指令替代逐行循环 | optimization-guide.md §2.6 |
| 指令融合 | `T.tile.axpy` 替代 mul+add | optimization-guide.md §2.7 |
| UB 预算反推 | 动态计算最优 block_N | optimization-guide.md §2.11 |
| Fixed Core | 任务数远大于物理核数时启用 | optimization-guide.md §2.9 |

---

## 检查清单

- [ ] 确认归约维度无法一次装入 UB（C2 满足）
- [ ] 确认原始实现存在 ≥2 遍完整扫描（C3 满足）
- [ ] 中间计算使用 fp32（低精度输入场景）
- [ ] `prev_max` 初始化为 `-inf`，`prev_sum` 初始化为 `0`
- [ ] 非整除场景 `pad_value=-T.infinity(cal_dtype)`
- [ ] UB 预算未超限（通过 `find_max_tile` 反推 block_N）
- [ ] 精度测试覆盖多种 dtype 和边界用例
- [ ] `msprof op` 确认 GM 读字节数下降
