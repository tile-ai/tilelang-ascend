# Vector 融合算子性能优化最佳实践

本文档总结了多 pass Vector 融合算子（含归约 + 归一化 + 量化等多步操作）在 TileLang-Ascend 上的性能优化手段，以 AddRmsNormDynamicQuant（Add + RMSNorm + Dynamic Quantization）为参考案例，对比基线实现与优化版本的关键差异。

> **核心发现**：算法层优化（pass 数量缩减）贡献了 47% 的总性能提升，远超任何核内优化。多 pass 算子应首先审视算法结构，而非直接进入 Double Buffer / Tiling 调优。

## 目录

- [优化概览](#优化概览)
- [优化手段详解](#优化手段详解)
  - [1. Pass 数量缩减（算法层，最高优先级）](#1-pass-数量缩减算法层最高优先级)
  - [2. Readback 模式（避免 Pass 2 重算）](#2-readback-模式避免-pass-2-重算)
  - [3. Tiling 参数优化（block_M + 自适应 block_N）](#3-tiling-参数优化block_m--自适应-block_n)
  - [4. Double Buffer 三阶段流水](#4-double-buffer-三阶段流水)
  - [5. 向量化广播](#5-向量化广播)
  - [6. 混合 kernel 策略](#6-混合-kernel-策略)
- [关键代码模式](#关键代码模式)
- [常见陷阱](#常见陷阱)
- [最佳实践建议](#最佳实践建议)
- [适用场景](#适用场景)
- [检查点](#检查点)
- [参考资料](#参考资料)

---

## 优化概览

| 优化项 | 基线实现（3-pass 串行） | 优化实现（2-pass + 混合策略） | 性能收益 |
|--------|------------------------|-------------------------------|---------|
| Pass 数量 | 3-pass（GM 读取 6x） | 2-pass（GM 读取 4x） | GM 读取 -33%，性能 +117% |
| Tiling | block_M=4, block_N=128 | block_M=16/32, 自适应 block_N | UB 利用率 10%→83% |
| 流水策略 | 串行（无 Double Buffer） | Double Buffer 三阶段流水 | MTE2/V/MTE3 重叠 |
| 广播方式 | scalar 循环 | `T.tile.broadcast` 向量化 | 小 shape +14~27% |
| Kernel 架构 | 单 kernel | 混合策略（小 shape 单核 / 大 shape 双核） | 解决小 shape 退化 |
| 同步模式 | AUTO_SYNC=True | AUTO_SYNC=True + 交替 buffer | 消除 V pipe RAW hazard |

### 各轮次贡献占比

| 轮次 | 优化类型 | 核心变化 | 占总提升比 |
|------|---------|---------|-----------|
| R1 | **算法层** | 3-pass → 2-pass | **47%** |
| R3 | **Tiling 层** | block_M=16 + 自适应 block_N | **34%** |
| R2 | **内存带宽层** | Double Buffer + 向量化广播 | **12%** |
| R7 | **架构层** | 混合 kernel 策略 | **6%** |
| R5 | 指令层 | mul_add_dst 融合 | ~0% |
| R6 | 架构层 | AUTO_SYNC=False + Fixed Core | 失败 |

---

## 优化手段详解

### 1. Pass 数量缩减（算法层，最高优先级）

多 pass 算子的 pass 数量直接决定 GM 带宽消耗。每减少一个 pass，GM 读取次数降低 33%~50%。这是收益最大的优化方向。

#### 基线实现（3-pass，GM 读取 6x）

```
Pass 1: 读 x1,x2 → h=x1+x2 → 写 x_out + 累加 sum_sq
Pass 2: 读 x1,x2 → h=x1+x2 → normed=h*inv_rms*gamma → 累加 abs_max  ← 需要 inv_rms
Pass 3: 读 x1,x2 → h=x1+x2 → normed=h*inv_rms*gamma → 量化 → 写 output  ← 需要 scale
```

**问题**：Pass 2 需要 inv_rms（Pass 1 产出），Pass 3 需要 scale（Pass 2 产出），三个 pass 必须串行。每个 pass 都重复读 x1、x2 并重算 h = x1 + x2。

#### 优化实现（2-pass，GM 读取 4x）

**关键数学变换**：

```
max(|h * inv_rms * gamma|) = inv_rms * max(|h * gamma|)
```

inv_rms 是行级标量，可以提取到 max 外面。因此 Pass 1 可以预计算 `abs_max(|h*gamma|)`（不需要 inv_rms），从而消除整个 Pass 2。

```
Pass 1: 读 x1,x2 → h=x1+x2 → 写 x_out + 累加 sum_sq + 预计算 abs_max(|h*gamma|)
Pass 间: inv_rms = rsqrt(sum_sq/H + eps); scale = abs_max * inv_rms / 127
Pass 2: 读 x1,x2 → h=x1+x2 → normed=h*inv_rms*gamma → 量化 → 写 output
```

**Pass 1 中 abs_max 预计算的关键代码**：

```python
# Pass 1 循环体内：在累加 sum_sq 的同时预计算 abs_max(|h*gamma|)
T.tile.add(h_fp32, x1_fp32, x2_fp32)                    # h = x1 + x2
T.tile.mul_add_dst(sum_sq_acc, h_fp32, h_fp32)           # sum_sq += h*h

# 预计算 abs_max(|h*gamma|) ← 关键：不需要 inv_rms
T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
T.tile.broadcast(gamma_tile, gamma_fp32)
T.tile.mul(hw_a, h_fp32, gamma_tile)                     # hw_a = h * gamma
T.tile.abs(hw_b, hw_a)                                   # hw_b = |h * gamma|
T.reduce_max(hw_b, tile_max, dim=-1)                     # 行内 max
T.tile.max(abs_max, abs_max, tile_max)                   # 累加全局 max
```

**Pass 间计算**（在两个 pass 之间执行，不在循环内）：

```python
# 归约 + Newton-Raphson rsqrt
T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
T.tile.mul(sum_sq_row, sum_sq_row, inv_H)                # sum_sq / H
T.tile.add(sum_sq_row, sum_sq_row, eps_val)              # + eps
T.tile.rsqrt(inv_rms_ub, sum_sq_row)                     # 1/sqrt(...)
# Newton-Raphson 精化（2 轮，见"关键代码模式"章节）
...
# scale = abs_max * inv_rms / 127
T.tile.fill(min_val, 1e-12)
T.tile.max(abs_max, abs_max, min_val)                    # 防止除零
T.tile.mul(scale_ub, abs_max, inv_rms_ub)
T.tile.div(scale_ub, scale_ub, 127.0)
```

**收益**：GM 读取从 6x 降至 4x（-33%），性能得分 +117%，是七轮中收益最大的一步。

**通用排查方法**（适用于其他多 pass 算子）：
1. 列出每个 pass 的输入/输出和 GM 读取次数
2. 分析 pass 间的数据依赖：哪些 pass 必须串行？哪些可以合并？
3. 寻找行级标量（如 inv_rms、mean、std）——这些标量可以在 pass 间计算，不需要额外 pass
4. 检查 max/min/sum 操作的对象是否包含可提取的标量因子

---

### 2. Readback 模式（避免 Pass 2 重算）

当 Pass 1 已经计算了中间结果 h 并写出到 GM（如 x_out），Pass 2 可以直接读回 x_out 而非重新从 x1+x2 计算 h。

#### 基线实现（recompute，GM 读取 3x）

```python
# Pass 2: 重新从 x1+x2 计算 h
T.copy(x1[row_start:row_start+ROWS, col_off:col_off+block_N], x1_ub[cur, :, :])
T.copy(x2[row_start:row_start+ROWS, col_off:col_off+block_N], x2_ub[cur, :, :])
T.tile.cast(x1_fp32, x1_ub[cur, :, :], "CAST_NONE", tile_elements)
T.tile.cast(x2_fp32, x2_ub[cur, :, :], "CAST_NONE", tile_elements)
T.tile.add(h_fp32, x1_fp32, x2_fp32)                    # 重算 h
```

#### 优化实现（readback，GM 读取 2x）

```python
# Pass 2: 直接读回 Pass 1 写出的 x_out（fp16），省去 x1+x2 两次读取
T.copy(x_out[row_start:row_start+ROWS, col_off:col_off+block_N], xout_ub[cur, :, :])
T.tile.cast(h_fp32, xout_ub[cur, :, :], "CAST_NONE", tile_elements)  # 从 x_out 恢复 h
```

**收益**：大 shape GM 读取从 3x 降至 2x（-33%），Case 1 (8192×8192) +14%，Case 6 (524288×128) +65%。

**精度约束**：readback 引入 fp16 往返误差（fp32 → fp16 → fp32）。当 |h| ≥ 100 时 fp16 精度不足，需回退到 recompute。推荐运行时动态选择：

```python
max_val = max(x1.abs().max().item(), x2.abs().max().item())
use_readback = max_val < 100.0  # 精度安全阈值
```

> **注意**：`.item()` host 同步约 5~15μs，对小 shape（kernel 耗时 < 50μs）分发开销占比 > 10%。需配合混合 kernel 策略（见 §6）使用。

---

### 3. Tiling 参数优化（block_M + 自适应 block_N）

#### 3.1 block_M 递增

block_M 决定每个 block 处理的行数，直接影响 UB 利用率和 block 调度开销。

| block_M | ROWS (=block_M/2) | UB 利用率 | block 数 (M=8192) | 效果 |
|---------|-------------------|----------|-------------------|------|
| 4 | 2 | ~10% | 2048 | 基线 |
| 8 | 4 | ~21% | 1024 | +162% |
| 16 | 8 | ~42% | 512 | +53% |
| 32 | 16 | ~83% | 256 | 大 shape +8~65%，小 shape -15~47% |

**关键发现**：block_M=32 在大 shape 下优于 block_M=16，但在小 shape 下因 block 数过少导致并行度不足。最优值取决于 kernel 架构（单 kernel vs 双 kernel），必须做交叉实验。

#### 3.2 自适应 block_N

当 H 远小于默认 block_N=256 时，使用实际 H 值可消除尾块处理。

#### 基线实现（固定 block_N=256）

```python
block_N = 256
n_num = H // block_N  # H=128 → n_num=0.5，尾块问题
# 分配 [2, 8, 256] buffer，但只用 [2, 8, 128]
# 50% 的 UB 被浪费，50% 的计算是无效的（处理填充的 0）
```

#### 优化实现（自适应 block_N）

```python
block_N = 256
if H < block_N and H % 16 == 0:  # 16 元素对齐要求
    block_N = H  # H=128 → block_N=128，消除尾块
n_num = (H + block_N - 1) // block_N  # H=128 → n_num=1，无尾块
```

**收益**：Case 6 (524288×128) 2.06x 加速，Case 19 (1024×128) 2.02x 加速。消除尾块后有效计算比例从 50% → 100%。

---

### 4. Double Buffer 三阶段流水

将每个 Pass 内的 H 维度循环改造为 prefetch → main body → epilogue 结构，MTE2/V/MTE3 三条流水线并行重叠。

#### 基线实现（串行）

```python
for by in T.serial(0, n_num):
    T.copy(x1[...], x1_ub)       # MTE2
    T.copy(x2[...], x2_ub)       # MTE2
    h_fp32 = x1_fp32 + x2_fp32   # V pipe
    # ... 计算 ...
    T.copy(out_ub, x_out[...])    # MTE3
```

```
时间线:
Block0: [MTE2][V][MTE3]
Block1:                  [MTE2][V][MTE3]
Block2:                                     [MTE2][V][MTE3]
```

#### 优化实现（Double Buffer 三阶段）

```python
stages = 2
x1_ub = T.alloc_ub([stages, ROWS, block_N], dtype)  # 双份
x2_ub = T.alloc_ub([stages, ROWS, block_N], dtype)  # 双份
gamma_ub = T.alloc_ub([stages, block_N], dtype)     # 双份

# Prefetch: 加载第 0 个 tile
T.wait_flag("mte3", "mte2", 0)
T.copy(x1[..., 0:block_N], x1_ub[0, :, :])
T.copy(x2[..., 0:block_N], x2_ub[0, :, :])
T.copy(gamma[0:block_N], gamma_ub[0, :])
T.set_flag("mte2", "v", 0)

# Main body: 流水执行
for by in T.serial(0, n_num - 1):
    cur = by % stages
    nxt = (by + 1) % stages

    # MTE2: 预取下一个 tile（与 V pipe 并行）
    T.wait_flag("mte3", "mte2", nxt)
    T.copy(x1[..., col_off_nxt:col_off_nxt+block_N], x1_ub[nxt, :, :])
    T.copy(x2[..., col_off_nxt:col_off_nxt+block_N], x2_ub[nxt, :, :])
    T.copy(gamma[col_off_nxt:col_off_nxt+block_N], gamma_ub[nxt, :])
    T.set_flag("mte2", "v", nxt)

    # V pipe: 计算当前 tile（与 MTE2/MTE3 并行）
    T.wait_flag("mte2", "v", cur)
    # ... 计算 ...
    T.set_flag("v", "mte3", cur)

    # MTE3: 写回当前 tile（与 V pipe 并行）
    T.wait_flag("v", "mte3", cur)
    T.copy(out_dtype_ub, x_out[..., col_off_cur:col_off_cur+block_N])
    T.set_flag("mte3", "mte2", cur)

# Epilogue: 处理最后一个 tile
last = (n_num - 1) % stages
T.wait_flag("mte2", "v", last)
# ... 计算 ...
T.set_flag("v", "mte3", last)
T.wait_flag("v", "mte3", last)
T.copy(out_dtype_ub, x_out[..., col_off_last:col_off_last+block_N])
T.set_flag("mte3", "mte2", last)
```

```
时间线:
Prefetch: [MTE2→s0]
Main[0]:    [MTE2→s1] [V→s0] [MTE3→s0]
Main[1]:      [MTE2→s0] [V→s1] [MTE3→s1]
Epilogue:                 [V→last] [MTE3→last]
```

**收益**：MTE2/V/MTE3 三条流水线并行重叠，小 shape +14~27%。

> **同步模式**：使用 `AUTO_SYNC=True` + 手动 `set_flag`/`wait_flag`。`AUTO_SYNC=False` 存在 V pipe 队列排空问题（详见 [compiler-limitations.md](../compiler-limitations.md) §1）。

---

### 5. 向量化广播

#### 基线实现（scalar 循环）

```python
# gamma 在 V pipe 中间加载（MTE2 操作混入 V pipe）
T.copy(gamma[...], gamma_ub)

# scalar 循环广播
for i in range(ROWS):
    for j in range(block_N):
        gamma_tile[i, j] = gamma_fp32[j]
```

#### 优化实现（tile 指令 + prefetch 阶段加载）

```python
# gamma 在 prefetch 阶段与 x1/x2 一起加载（MTE2 操作集中）
T.copy(gamma[col_off:col_off+block_N], gamma_ub[cur, :])

# 向量化广播（1 条指令替代 ROWS×block_N 次 scalar）
T.tile.cast(gamma_fp32, gamma_ub[cur, :], "CAST_NONE", block_N)
T.tile.broadcast(gamma_tile, gamma_fp32)
```

**收益**：gamma 的 DMA 传输与 x1/x2 合并为一批；1 条 tile 指令替代 1024 次 scalar 操作（ROWS=4, block_N=256）。

---

### 6. 混合 kernel 策略

当不同 shape 范围的最优配置矛盾时（如 block_M=16 小 shape 优、block_M=32 大 shape 优），按维度动态分发到不同 kernel。

#### 基线实现（单一 kernel）

```python
# 所有 shape 使用同一 kernel（block_M=32 + 双 kernel）
kernel = _kernel_readback(M, H, block_M=32, ...)
# 小 shape (M=256): 8.9μs ← 分发开销占比 > 10%
```

#### 优化实现（按 M 维度分发）

```python
HYBRID_THRESHOLD = 1024

def _get_kernel(M, H, tl_dtype, eps, use_readback):
    if M < HYBRID_THRESHOLD:
        # 小 shape: 单 kernel (block_M=16, 无分发开销)
        block_M, block_N = _get_tiling_small(M, H)  # block_M=16
        return _kernel_single(M, H, block_M, block_N, eps, dtype=tl_dtype)
    else:
        # 大 shape: 双 kernel (block_M=32, readback/recompute)
        block_M, block_N = _get_tiling_large(M, H)  # block_M=32
        if use_readback:
            return _kernel_readback(M, H, block_M, block_N, eps, dtype=tl_dtype)
        else:
            return _kernel_recompute(M, H, block_M, block_N, eps, dtype=tl_dtype)
```

**分发开销评估**（引入双 kernel 前必做）：

```python
dispatch_overhead_us = 10  # .item() host 同步约 5~15μs
kernel_us = estimated_current_us
overhead_ratio = dispatch_overhead_us / kernel_us

if overhead_ratio > 0.10:
    use_single_kernel = True   # 分发开销 > 10%，不适合双 kernel
elif overhead_ratio < 0.05:
    use_dual_kernel = True     # 分发开销 < 5%，适合双 kernel
else:
    use_hybrid = True          # 5~10% 灰色地带，需要混合策略
```

**收益**：小 shape +15~46%（Case 17: 8.9μs → 4.8μs），大 shape 保持不变。

---

## 关键代码模式

### Newton-Raphson rsqrt 精化

硬件 `rsqrt` 精度有限，通过 2 轮 Newton-Raphson 迭代提升精度（`x = x * (1.5 - 0.5 * a * x²)`）：

```python
T.tile.rsqrt(inv_rms_ub, sum_sq_row)
# 第 1 轮迭代
T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
T.tile.mul(nr_temp, nr_temp, sum_sq_row)
T.tile.mul(nr_temp, nr_temp, -0.5)
T.tile.add(nr_temp, nr_temp, 1.5)
T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
# 第 2 轮迭代（同上）
T.tile.mul(nr_temp, inv_rms_ub, inv_rms_ub)
T.tile.mul(nr_temp, nr_temp, sum_sq_row)
T.tile.mul(nr_temp, nr_temp, -0.5)
T.tile.add(nr_temp, nr_temp, 1.5)
T.tile.mul(inv_rms_ub, inv_rms_ub, nr_temp)
T.tile.broadcast(inv_rms_tile, inv_rms_ub)
```

### 交替 buffer 消除 V pipe RAW hazard

`AUTO_SYNC=True` 下连续 tile 指令存在 RAW 依赖时（dst → src），使用两个独立 buffer 交替：

```python
hw_a = T.alloc_ub([ROWS, block_N], "float32")
hw_b = T.alloc_ub([ROWS, block_N], "float32")

# Pass 1: abs_max 计算链
T.tile.mul(hw_a, h_fp32, gamma_tile)   # hw_a = h * gamma
T.tile.abs(hw_b, hw_a)                 # hw_b = |hw_a|（避免 hw_a → hw_a RAW）

# Pass 2: 量化链（更长的交替链）
T.tile.mul(hw_a, h_fp32, inv_rms_tile)
T.tile.mul(hw_b, hw_a, gamma_tile)
T.tile.div(hw_a, hw_b, scale_tile)
T.tile.round(hw_b, hw_a, tile_elements)
T.tile.clamp(hw_a, hw_b, -128.0, 127.0, tile_elements)
```

> **UB 开销**：额外 1 个 `[ROWS, block_N]` fp32 buffer（ROWS=8, block_N=256 → 8KB），通常可接受。
> 详细说明见 [optimization-guide.md](../optimization-guide.md) §2.2.2。

---

## 常见陷阱

### 陷阱 1：在 bandwidth-bound 算子上尝试指令融合

```python
# 错误：算子当前耗时 1558μs / 理论最优 299μs = 5.2x > 3x → bandwidth-bound
# 在 bandwidth-bound 上做 mul_add_dst 融合 → 性能变化在噪声范围内（±1%）

# 正确：先做瓶颈预判（Step 0），bandwidth-bound 禁止指令融合
# 优先优化 GM 带宽（减少 pass 数、readback）
```

### 陷阱 2：AUTO_SYNC=False 导致精度失败

```python
# 错误：AUTO_SYNC=False → barrier_all() 无法排空 V pipe 指令队列
# 当 n_num > 1 时，后续 scalar 操作（reduce_sum、rsqrt）读到旧值
# 结果：17/20 精度失败

# 正确：保持 AUTO_SYNC=True，使用交替 buffer（hw_a/hw_b）消除 RAW 依赖
```

### 陷阱 3：Fixed Core 在 m_num < CORE_NUM 时编译失败

```python
# 错误：m_num=8 < 24 cores → TVM StmtSimplifier InternalError
# 空循环 range 约束冲突

# 正确：使用标准 m_num launch，不做 Fixed Core
```

### 陷阱 4：单维度递增搜索 block_M（未做交叉实验）

```python
# 错误：逐轮递增 block_M (4→8→16→32)，未与 kernel 架构做交叉实验
# 结果：block_M=16 在双 kernel 下退化 15~28%（R5a 失败）

# 正确：构建 2×2 交叉矩阵（block_M × kernel 架构），一轮实验覆盖所有组合
# 发现：单 kernel + block_M=16 和 双 kernel + block_M=32 的组合最优
```

### 陷阱 5：双 kernel 未评估分发开销

```python
# 错误：引入双 kernel（readback/recompute）但未评估 .item() 分发开销
# 小 shape (M=256, kernel 4.7μs)：分发开销 10μs 占比 > 100% → 退化 -89%

# 正确：引入双 kernel 前评估分发开销占比
# overhead_ratio = 10μs / kernel_us > 10% → 使用混合策略
```

---

## 最佳实践建议

### ✅ 推荐做法

1. **多 pass 算子首先审视 pass 数量**
   - 列出每个 pass 的 GM 读取次数，寻找可合并的 pass
   - 寻找行级标量（inv_rms、mean、std），提取到 pass 间计算
   - 检查 max/min/sum 操作是否包含可提取的标量因子

2. **优化前先做瓶颈预判**
   - 计算理论最小耗时（基于 GM 带宽），判断 bandwidth/compute/launch-bound
   - bandwidth-bound → 减少 pass 数、readback；禁止指令融合
   - compute-bound → 指令融合、流水优化

3. **使用 Double Buffer + 交替 buffer**
   - `AUTO_SYNC=True` + 手动 `set_flag`/`wait_flag` 控制三路流水
   - 连续 tile 指令的 dst/src 使用 hw_a/hw_b 交替，消除 RAW hazard

4. **存在多个优化维度时做交叉实验**
   - 构建 2×2 矩阵（如 block_M × kernel 架构），快速验证所有组合
   - 避免单维度递增搜索导致的紧耦合问题

5. **引入双 kernel 前评估分发开销**
   - `.item()` host 同步约 5~15μs
   - 分发开销占比 > 10% → 使用混合策略（按 shape 分发）

### ❌ 避免做法

1. **避免在 bandwidth-bound 算子上尝试 compute-bound 优化**
   - 指令融合（mul_add_dst）、AUTO_SYNC=False 等在 bandwidth-bound 上无收益

2. **避免 AUTO_SYNC=False（当前编译器）**
   - V pipe 指令队列排空问题未解决，保持 AUTO_SYNC=True

3. **避免 Fixed Core（m_num < CORE_NUM 时）**
   - TVM StmtSimplifier bug，使用标准 m_num launch

4. **避免跳过交叉实验直接单维度递增**
   - Tiling 参数与 kernel 架构紧耦合，必须做交叉验证

---

## 适用场景

本优化方案适用于以下算子：

- ✅ 多 pass Vector 融合算子（含归约 + 归一化 + 量化/激活等多步操作）
- ✅ 需要多次遍历 GM 数据的算子（pass 数量 > 1）
- ✅ 包含行级标量计算的算子（如 RMSNorm 的 inv_rms、LayerNorm 的 mean/std）
- ✅ 多 shape 范围性能差异大的算子（需要混合策略）
- ✅ bandwidth-bound 的 Vector 算子（当前耗时 / 理论最优 > 3x）

**核心思想**：算法层优化（pass 缩减）> Tiling 优化 > 内存带宽优化 > 架构层优化。先减少 GM 读取次数，再优化每次读取的效率。

---

## 检查点

- [ ] 瓶颈预判完成？（bandwidth/compute/launch-bound）
- [ ] pass 数量是否已最小化？（行级标量是否提取到 pass 间？）
- [ ] Double Buffer 三阶段流水是否实现？（prefetch/main/epilogue）
- [ ] 交替 buffer（hw_a/hw_b）是否用于连续 tile 指令链？
- [ ] block_M 是否通过交叉实验确定？（与 kernel 架构做 2×2 矩阵）
- [ ] 自适应 block_N 是否处理？（H < 256 且 H % 16 == 0 时 block_N = H）
- [ ] 双 kernel 分发开销是否评估？（`.item()` 占比 < 10%？）
- [ ] 混合策略是否实现？（小 shape 单核 / 大 shape 双核）
- [ ] 生成的 Ascend C 中能看到 `MTE3_MTE2`、`MTE2_V`、`V_MTE3` 三类事件？
- [ ] AUTO_SYNC=True 保持？（当前编译器限制）

---

## 参考资料

- **完整代码**：`examples/add_rms_norm_dynamic_quant/example_add_rms_norm_dynamic_quant.py`
- **优化日志**：`.agents/skills/tilelang-op-test-design/cann-bench/add_rms_norm_dynamic_quant/optimization_log.md`
- **详细分析**：`.agents/skills/tilelang-op-test-design/cann-bench/add_rms_norm_dynamic_quant/All_profiling.md`
- **Double Buffer 流水模板**：[vector_add_pipeline.md](vector_add_pipeline.md)
- **编译器限制**：[compiler-limitations.md](../compiler-limitations.md)
- **优化指南**：[optimization-guide.md](../optimization-guide.md)（§一.五 算法层优化、§2.2.2 交替 buffer）
