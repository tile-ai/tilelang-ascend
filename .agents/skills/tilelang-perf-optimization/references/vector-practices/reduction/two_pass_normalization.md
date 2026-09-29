# 两遍扫描归一化算子优化（Two-Pass Normalization）

本文档介绍 Vector 型两遍扫描归一化算子的完整优化模式。此类算子的核心结构为"Pass 1 归约求统计量（mean/rms 等）+ Pass 2 逐元素应用统计量并写回"，属于纯 Vector 型（无 Cube 操作）。

参考实现：`examples/cann-bench/rms_norm/example_rms_norm.py`

---

## 适用场景

### 适用条件（全部满足）

| 编号 | 条件 | 说明 |
|------|------|------|
| C1 | 算子类型为 Vector（`IS_ASCEND_AIV`） | 归约维度沿列方向，数据需分块从 GM 搬入 UB |
| C2 | 统计量需分块累加 | 归约维度 D 大于 UB 单次容量，Pass 1 必须分块扫描累加 |
| C3 | Pass 2 需写回 GM | 存在 MTE3 写回，为 Double Buffer 闭环提供条件 |
| C4 | 两遍之间存在数据依赖 | Pass 2 依赖 Pass 1 的统计量，无法合并为单遍 |

### 不适用条件

| 编号 | 条件 | 原因 |
|------|------|------|
| X1 | `n_num == 1`（block_N ≥ D） | 无流水空间，Hybrid 单 kernel 自动同步更快，见"双 kernel 策略" |
| X2 | D 极小（≤8）+ S 极大 | 每块计算量过小，软件循环开销占比高，见 optimization-guide.md §2.9 失效条件 |

### 可泛化的算子模式

| 算子 | Pass 1 统计量 | Pass 2 应用 |
|------|--------------|------------|
| RMSNorm | sum of squares → rsqrt(mean) | x × inv_rms × gamma |
| LayerNorm | mean + variance → rsqrt(var) | (x − mean) × inv_std × gamma + beta |
| GroupNorm | 组内 mean + variance | 同 LayerNorm（按组归约） |
| 任何"归约统计量 → 逐元素应用"算子 | 分块累加统计量 | 广播应用 |

---

## 约束条件

### 硬件约束

| 约束 | 说明 |
|------|------|
| UB 容量 | A2/A3 的 UB 预算为 192 KB，按预算公式反推 block_M / block_N（见优化 1） |
| 对齐要求 | 所有 UB buffer 需 32B 对齐（fp32 为 8 元素，fp16/bf16 为 16 元素） |
| AIV flag 通道 | `mte3→mte2` ✅、`mte2→v` ✅、`v→mte3` ✅、`v→mte2` ❌（不可用，死锁）——决定 Pass 1 不可双缓冲 |
| `T.copy(UB→UB)` | 走 MTE2 引擎（DMA），不能与 GM→UB 重叠（见优化 6） |

### 算法约束

| 约束 | 说明 |
|------|------|
| 中间计算精度 | fp16/bf16 输入必须 cast 到 fp32 计算，否则无法满足 MERE/MARE 阈值 |
| rsqrt 精度 | `T.tile.rsqrt` 为查表近似，精度约 1e-3；fp32 rtol=1e-4 需加 Newton 迭代（见优化 4） |
| pass_configs 编译期固定 | `pass_configs` 是 decorator 参数，jit 闭包内不能动态切换（催生双 kernel 策略） |

---

## 优化模式清单（按实施顺序）

### 优化 1：自适应 tiling（host 侧 `_select_tiling`）

**适用条件**：所有 case（无限制）

**实现**：host 侧根据 S、D、dtype 动态选择 block_M / block_N，按 UB 预算反推。候选 block_M: (1024, 512, 256, 128, 64, 32, 16)。评分公式: `n_num * 100000 + m_penalty`（n_num 优先减少 GM 读取，m_num ∈ [24, 48] 避免空闲核和过多 launch）。

**收益**：RMSNorm 实测 0.43x → 0.58x（+35%）

### 优化 2：单遍扫描（n_num=1 时 Pass 2 跳过 GM 读取）

**适用条件**：`n_num == 1`（block_N ≥ D）

**实现**：Pass 1 只读 a_ub（cast 到 a_cal），不修改 a_ub。Pass 2 可跳过 `T.copy(A, a_ub)`，复用 Pass 1 的 a_ub。jit 闭包加 `single_pass = (n_num_int == 1)` 编译期常量。

**收益**：n_num=1 的 case 减少 1 次 GM→UB 读取，+2-3%

### 优化 3：output_ub 分离 a_ub 双重写入

**适用条件**：fp16/bf16 dtype（需 cast back 到低精度）

**问题**：Pass 2 中 `T.tile.cast(a_ub, a_cal, CAST_RINT)` 让 a_ub 被 V 写入，与 MTE2 写入冲突，阻止 Double Buffer。

**实现**：引入独立 output_ub 做 cast back，使 a_ub 只被 MTE2 写入。

**收益**：为 Pass 2 Double Buffer 铺路

### 优化 4：rsqrt Newton 迭代（精度修复）

**适用条件**：使用 `T.tile.rsqrt` 且精度要求 rtol < 1e-3

**问题**：`T.tile.rsqrt` 精度约 1e-3（查表近似），对 fp32 rtol=1e-4 不足。

**实现**：`y1 = y0 * (1.5 - 0.5 * x * y0²)`，精度提升到 ~1e-6。

> **UB buffer 命名约束**：不能命名为 `tmp_ub`（与 `AscendMemoryPlanning` pass 内部临时 buffer 冲突），用 `newton_ub` 等语义化命名。

### 优化 5：gamma 1D→2D 广播乘

用 `T.tile.broadcast + T.tile.mul`，不要用 `T.Parallel` 赋值与 `T.tile.*` 混用。

### 优化 6：T.copy(UB→UB) 走 MTE2 的限制

`T.copy(UB→UB)` 生成 `copy_ub_to_ub` 走 MTE2 引擎，不能与 GM→UB 重叠。gamma 预加载类优化**无效**。

### 优化 7：mul_add_dst 硬件性能反转

`mul_add_dst` 在 910B3/910C 上可能比 `mul+add` **更慢**。必须实测验证，不更快则回退。

### 优化 8：Pass 2 Double Buffer（Expert 模式）

**适用条件**：n_num > 1 + Expert 模式（AUTO_SYNC=False）

**前置条件**：已实施优化 3（output_ub 分离）；Pass 2 无跨迭代累加器；inv_rms_tile 在循环外计算。

**实现**：参考 optimization-guide.md §2.2 Vector 核三阶段流水 + §2.2.0 同步决策表。Pass 1 不可双缓冲（见 optimization-guide.md §2.2.2）。

**收益**：大 D case 耗时下降 10-17%

---

## 双 kernel 策略（n_num 自适应模式选择）

**适用场景**：同一算子在不同 shape 下有截然不同的最优 pass_configs，且 pass_configs 不能在 jit 闭包内动态切换。

**实现**：定义两个 jit 函数（Hybrid + Expert），host 侧按 n_num 选择：

- `n_num=1`：Hybrid（AUTO_SYNC=True）——无流水空间，自动同步避免 barrier 开销，且可实施优化 2 单遍扫描；
- `n_num>1`：Expert（AUTO_SYNC=False + Pass 2 Double Buffer）——MTE2/V/MTE3 三路重叠收益大于手动同步成本。

### cann-bench profiler linecache 适配

profiler 子进程清除 `linecache.cache`，导致 `OSError: could not get source code`。在 `_get_kernel` 中调 `_ensure_source_cached()` 预缓存源代码。

---

## 完整优化检查清单

- [ ] 自适应 tiling 已实施？
- [ ] 单遍扫描已实施（n_num=1）？
- [ ] output_ub 已分离（fp16/bf16）？
- [ ] rsqrt Newton 迭代已加？
- [ ] gamma 广播用 `broadcast + mul`？
- [ ] UB buffer 无 `tmp_ub` 命名？
- [ ] mul_add_dst 已实测验证？
- [ ] Pass 2 Double Buffer 已实施（n_num>1）？
- [ ] Pass 1 保持串行（barrier_all）？

---

## 关联文档

- 反模式自查：[performance-antipatterns.md](../../performance-antipatterns.md)（T.copy(UB→UB) 走 MTE2、mul_add_dst 性能反转、tmp_ub 命名冲突、Pass 1 双缓冲不可行、Fixed Core 小 D 大 S 失效）
- 同步机制：[optimization-guide.md](../../optimization-guide.md) §2.2（三阶段流水）、§2.2.0（同步决策表）、§2.2.2（Pass 1 vs Pass 2 双缓冲可行性）、§2.9（Fixed Core 失效条件）
- 同族优化：[vector_reduce_pass_fusion.md](../vector_reduce_pass_fusion.md)（归约遍数融合，Online Softmax 3-pass → 2-pass）
