# TileLang 性能优化指南

## 目录

- [一、优化优先级与算子类型对应](#一优化优先级与算子类型对应)
- [二、核内优化](#二核内优化)
  - [2.13 多行 Tile 粒度扩展（Multi-row Tile Granularity）](#213-多行-tile-粒度扩展multi-row-tile-granularity)
    - [2.13.9 动态 block_S 计算（UB 预算反推）](#2139-动态-block_s-计算多行-tile-强制不可固定值)
    - [2.13.10 多行 Tile 强制配套 pass_configs](#21310-多行-tile-强制配套-pass_configs)
- [三、核间优化](#三核间优化)
- [四、常见问题速查](#四常见问题速查)

### 相关 Skill 参考（优化前必读）

- **API 用法**：查阅 [tilelang-api-best-practices SKILL.md](../../tilelang-custom-skill/tilelang-api-best-practices/SKILL.md) 及其 references 目录
- **编程模式和 pass_configs**：查阅 [tilelang-expert-to-developer SKILL.md](../../tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md) 及其 references 目录

---

## 一、优化优先级与算子类型对应

根据算子类型选择优化范围（算子类型通过 `get_kernel_source()` 中的 `IS_ASCEND_AIC` / `IS_ASCEND_AIV` 判断）：

| 算子类型 | 判断依据 | 优化范围 |
|---------|---------|---------|
| **Cube 型** | 代码含 `IS_ASCEND_AIC` | Cube 核内优化 + Fixed Core |
| **Vector 型** | 代码含 `IS_ASCEND_AIV` | Vector 核内优化 + Fixed Core |
| **CV 融合型** | 代码两者均有 | 先核内优化（Cube + Vector）→ 再核间优化 + Fixed Core |

> **Fixed Core 模式**适用于所有算子类型（核内/核间均可使能），见 2.9 节。

优先使用 Developer 特性（自动同步、自动内存规划），按以下顺序尝试优化：

```
核内优化 → 核间优化
```
> `pass_configs`（`AUTO_SYNC`、`MEMORY_PLANNING` 等）是其他优化的**伴随修改**，不是独立步骤。需要改动时在对应优化点内部一并处理（如 Double Buffer 时同步设 `AUTO_SYNC=False`）。

---

## 二、核内优化

> **优化顺序**：
> - **Cube 型算子**：执行 Cube 核内优化（2.1 Split-K 切分策略、2.2 Double Buffer、2.3 MTE2 预取、2.4 Full-Load、2.5 小数据块合并载入）+ 2.9 Fixed Core
> - **Vector 型算子**：执行 Vector 核内优化（2.2 Double Buffer Vector 侧、2.6 指令向量化、2.7 指令融合、2.8 稀疏访存优化）+ 2.9 Fixed Core
> - **CV 融合型算子**：先执行 Cube 核内优化 → 再执行 Vector 核内优化 → 最后执行核间优化（见第三章）+ 2.9 Fixed Core

### 2.1 Split-K 切分策略（Cube 核）

**适用场景**：
- 矩阵乘 K 维度较大，单次 L1 → L0 搬运无法容纳全部数据
- GEMM 的 K 维度远大于 L0 buffer 容量
- 代码中存在 K 维度循环，但每次循环都等待前一次搬运完成

**原理**：将 K 维度切分为多个小块，配合 Ping-Pong 双缓冲实现 MTE1 搬运与 Cube 计算的流水重叠。这是后续 Double Buffer 优化的前置切分策略。

**优化前**（串行搬运和计算）：
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[:, :])
    T.mma(l0a[:, :], l0b[:, :], l0c[:, :])
```

**优化后**（K 轴切块 + Ping-Pong 双缓冲）：
```python
for k in T.serial(loop_k):
    side = k % 2
    T.wait_flag("M", "MTE1", SIG_L0AB + side)
    T.copy(k_l1, l0b[side, :, :])
    T.set_flag("MTE1", "M", SIG_L0AB + side)

    T.wait_flag("MTE1", "M", SIG_L0AB + side)
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
    T.set_flag("M", "MTE1", SIG_L0AB + side)
```

### 2.2 Double Buffer（Cube / Vector 核通用）

**适用场景**：
- 循环内包含多个串行操作（搬运 → 计算 → 写回）
- 数据块可以切分为多份，支持流水线并行
- 使用 `T.serial` 的循环

**注意**：
- 切分后的数据块不能太小，否则无法发挥流水掩盖效果：
  - Vector 核：切分后每个数据块元素数应 ≥ 128
  - Cube 核：切分后每个数据块元素数应 ≥ 256
- 实现方式：手写 Double Buffer（手动分配双份 buffer，通过 `side = k % 2` 交替使用）
- 同步方式：Vector/Cube 核内流水当前需要手动控制不同流之间的同步。同步模式选择见下方 §2.2.0 决策表。**`set_flag` / `wait_flag` / `barrier_all` 的底层机理、双缓冲 Flag 完整模式详见 [sync-primitives-guide.md](../sync-primitives-guide.md)**

#### 2.2.0 同步模式决策表（Double Buffer 实施前必须先判断）

实施 Double Buffer 前，必须先根据算子类型选择同步模式。选错会导致流水完全失效或性能退化。

| 算子类型 | 判断条件 | AUTO_SYNC | 需要同步的 flag 对 |
|---------|---------|-----------|------------------|
| **纯 AIV Vector 算子** | MSProf 报告 aicore compute < 20%，无 MMA/Conv | **False** | `mte3→mte2`、`mte2→v`、`v→mte3` |
| **纯 AIC Cube 算子** | 只有 MMA/Cast 等 Cube 指令 | **True** 起步即可 | MTE1/M/Cube 层 flag 复杂，自动同步更安全 |
| **CV 融合算子** | 同时有 AIC+AIV（`KERNEL_TYPE_MIX`） | **False** | `mte3→mte2`、`mte2→v`、`v→mte3`（AIV 侧）+ cross flag（核间） |

> **铁律**：纯 AIV Vector 算子实施 Double Buffer 时，`AUTO_SYNC=True` 的编译器会在每个 stage 之间自动插入冗余 `barrier_all`，使 MTE2/V/MTE3 三条流水线无法并行重叠，导致双缓冲形同虚设（性能与基线持平）。**Vector 核 Double Buffer 必须从 `AUTO_SYNC=False` 开始。**
>
> **自检**：实施 Vector Double Buffer 后性能无明显提升（相对基线 < 10%），第一时间检查 `AUTO_SYNC` 是否为 True。

**原理**：
```
串行模式:
  Block0: [MTE2][VEC][MTE3]
  Block1:        ----------[MTE2][VEC][MTE3]

Double Buffer:
  Block0: [MTE2][VEC][MTE3]
  Block1:   [MTE2][VEC][MTE3]
```

**Cube 核示例**：

**优化前**（串行执行）：
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[:, :])
    T.mma(l0a[:, :], l0b[:, :], l0c[:, :])
```

**优化后**（手写 Ping-Pong 双缓冲，开启自动同步）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# 分配双缓冲
l0a = T.alloc_L0A([2, block_M, dim], dtype)
l0b = T.alloc_L0B([2, dim, block_N], dtype)
l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

for k in T.serial(loop_k):
    side = k % 2
    T.copy(k_l1, l0b[side, :, :])
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
```

**优化后**（手写 Ping-Pong 双缓冲 + 手动同步，自动同步不符合预期时使用）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# 分配双缓冲
l0a = T.alloc_L0A([2, block_M, dim], dtype)
l0b = T.alloc_L0B([2, dim, block_N], dtype)
l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

# 初始化信号
T.set_flag("M", "MTE1", SIG_L0AB)
T.set_flag("M", "MTE1", SIG_L0AB + 1)
T.set_flag("FIX", "M", SIG_L0C)
T.set_flag("FIX", "M", SIG_L0C + 1)

for k in T.serial(loop_k):
    side = k % 2
    # MTE1 搬运与 Cube 计算流水重叠
    T.wait_flag("M", "MTE1", SIG_L0AB + side)
    T.copy(k_l1, l0b[side, :, :])
    T.set_flag("MTE1", "M", SIG_L0AB + side)

    T.wait_flag("MTE1", "M", SIG_L0AB + side)
    T.wait_flag("FIX", "M", SIG_L0C + side)
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
    T.set_flag("M", "MTE1", SIG_L0AB + side)
    T.set_flag("M", "FIX", SIG_L0C + side)
```

**Vector 核示例**：

**优化前**（串行执行）：
```python
for k in T.serial(loop_k):
    T.copy(GM_data[k], ub_buf)
    T.tile.exp(result_buf, ub_buf)
    T.copy(result_buf, GM_out[k])
```

**优化后**（手写 Ping-Pong 双缓冲 + 手动三路 flag，AUTO_SYNC=False）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # Vector 核必须 False
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# 分配双缓冲（GM↔UB 的 buffer 双份，V pipe 中间 buffer 单份）
input_buf = T.alloc_ub([2, block_size], dtype)     # MTE2 写入目标 → 双份
compute_buf = T.alloc_ub([block_size], "float32")   # 仅 V pipe → 单份
output_buf = T.alloc_ub([2, block_size], dtype)     # MTE3 读出源 → 双份

# 初始化：允许 MTE2 读第 0/1 块
T.set_flag("mte3", "mte2", 0)
T.set_flag("mte3", "mte2", 1)
# prefetch 第 0 块
T.wait_flag("mte3", "mte2", 0)
T.copy(GM_data[0], input_buf[0, :])
T.set_flag("mte2", "v", 0)

for si in T.serial(1, loop_k):
    cur = si % 2
    nxt = (si + 1) % 2
    # 预取第 si 块到 nxt
    if si < loop_k - 1:
        T.wait_flag("mte3", "mte2", nxt)
        T.copy(GM_data[si], input_buf[nxt, :])
        T.set_flag("mte2", "v", nxt)
    # 消费第 si-1 块（cur）
    T.wait_flag("mte2", "v", cur)
    T.tile.exp(compute_buf, input_buf[cur, :])   # V pipe 单缓冲安全
    T.copy(compute_buf, output_buf[cur, :])
    T.set_flag("v", "mte3", cur)
    T.wait_flag("v", "mte3", cur)
    s_off = (si - 1) * block_size
    T.copy(output_buf[cur, :], GM_out[s_off:])
    T.set_flag("mte3", "mte2", cur)

# epilogue：消费并写回最后一块
last_cur = (loop_k - 1) % 2
T.wait_flag("mte2", "v", last_cur)
T.tile.exp(compute_buf, input_buf[last_cur, :])
T.copy(compute_buf, output_buf[last_cur, :])
T.set_flag("v", "mte3", last_cur)
T.wait_flag("v", "mte3", last_cur)
T.copy(output_buf[last_cur, :], GM_out_last)
T.set_flag("mte3", "mte2", last_cur)

T.wait_flag("mte3", "mte2", 0)
T.wait_flag("mte3", "mte2", 1)
```

**Vector 核三路 flag 时序说明**（每个 flag 对的含义）：

| flag 对 | 触发时机 | 含义 |
|---------|---------|------|
| `set_flag("mte3", "mte2", sid)` | GM 写回完成后 | 通知 MTE2：buffer `sid` 已释放，可搬入下一块 |
| `set_flag("mte2", "v", sid)` | GM→UB 搬运完成后 | 通知 V pipe：buffer `sid` 输入数据就绪，可开始计算 |
| `set_flag("v", "mte3", sid)` | V 计算完成后 | 通知 MTE3：buffer `sid` 计算结果就绪，可写回 GM |

> **易错点**：`wait_flag("v", "mte3", sid)` 必须在 cast/copy 到 output_buf 之后、GM write 之前，否则 MTE3 会读到未就绪的旧值。

**Vector 核内三阶段流水 reference**：

完整说明见 [vector_add_pipeline](best-practices/vector_add_pipeline.md)。该模板使用两个 UB stage 和 `mte3 -> mte2`、`mte2 -> v`、`v -> mte3` 事件，把 Vector 算子组织为：

- prefetch：预取第 0 个 tile 到 stage 0
- main body：预取 `tile + 1`，同时消费并写回 `tile`
- epilogue：消费并写回最后一个已预取 tile

编写 Vector pipeline 时优先保持这三个阶段的边界清晰，再替换 main body 中的 `T.tile.add` 为目标算子的 Vector 计算。

#### 2.2.1 Double Buffer 分配规则（哪些 buffer 需要双缓冲）

> **核心判断标准**：该 buffer 是否被 MTE2（GM→UB 读）或 MTE3（UB→GM 写）**异步**访问？

| 访问源 | 示例 | 分配方式 | 原因 |
|--------|------|---------|------|
| MTE2 写入（`T.copy(GM, buf)` 的目标） | `T.copy(X[...], data_buf)` | `[2, ...]` **双缓冲** | MTE2 与 V pipe 并行时，buffer 共享会读到未就绪数据 |
| MTE3 读出（`T.copy(buf, GM)` 的源） | `T.copy(out_buf, Y[...])` | `[2, ...]` **双缓冲** | MTE3 与 V pipe 并行时，buffer 共享会写到未完成的 buffer |
| 仅 V pipe（cast/sub/mul/add 等指令的中间 buffer） | `data_cal_p2` | `[1, ...]` **单缓冲** | V pipe 单条指令链天然串行，同 buffer 读写安全 |

> **常见错误**：把 V pipe 的中间计算 buffer（如 `data_cal`、`tmp`）也分配为 `[2, ...]` 双缓冲。这会浪费 `cpg × block_S × dtype_bytes` 的 UB 空间，在 cpg 较大时（如 GroupNorm cpg=257）直接导致 UB 溢出（`Memory allocation failed`），且不带来任何性能收益。
>
> **自检方法**：每个 `[2, ...]` buffer 必须能找到对应的 `T.copy(GM_buf, ...)` 或 `T.copy(..., GM_buf)` 语句。找不到 → 该 buffer 改为单份。

### 2.3 MTE2 预取优化（Cube 核）

**适用场景**：
- 已开启 Double Buffer 但各流水线 busy ≤ 70%（准无 bound）
- K 方向切分次数 `kL1Iter ≥ 2`

**原理**：将主循环改造为「首轮预取 → 正式循环」三段结构，让 MTE2 提前搬入下一轮数据，消除流水起停开销。

**优化前**（每轮搬运 + 计算串行）：
```python
for k in T.serial(loop_k):
    T.copy(k_l1, l0b[side, :, :])  # MTE2 搬入
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])  # Cube 计算
```

**优化后**（首轮预取 + 流水掩盖）：
```python
# 首轮预取 PING
T.copy(k_l1_iter0, l0b[0, :, :])

for k in T.serial(1, loop_k):
    side = k % 2
    next_side = (k + 1) % 2
    # 预取下一轮数据到 PONG
    if k < loop_k - 1:
        T.copy(k_l1_next, l0b[next_side, :, :])
    # 消费当前轮
    T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :])
```

### 2.4 减少重复载入 / Full-Load（Cube 核）

**适用场景**：
- 一侧矩阵较小（如 `baseM × K × dtype ≤ L1/2`）
- 对侧循环次数 `T ≥ 2`（如 N 方向有多轮迭代）
- 小侧矩阵在每轮循环中重复从 GM 搬运到 L1

**原理**：将小侧矩阵一次性驻留 L1，消除对侧循环中的重复 GM→L1 搬运，等效把 MTE2 总字节数压缩 `(T-1)/T`。

**优化前**（每轮都搬运小侧矩阵 A）：
```python
for n_iter in T.serial(T):
    for k in T.serial(loop_k):
        T.copy(A[bz, by, :, :], a_l1)  # 每轮重复搬运
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
        T.gemm_v0(a_l1, k_l1, acc_l0c, transpose_B=True)
```

**优化后**（A 一次性驻留 L1）：
```python
# 初始化阶段：A 一次性驻留 L1
T.copy(A[bz, by, :, :], a_l1)

for n_iter in T.serial(T):
    for k in T.serial(loop_k):
        # A 已驻留，跳过搬运
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
        T.gemm_v0(a_l1, k_l1, acc_l0c, transpose_B=True)
```

### 2.5 小数据块合并载入（Cube 核）

**适用场景**：
- 存在小块随路数据（如 Scale、Bias、LUT 等），单次搬运量 < 20 KB
- K 方向循环次数较多，小块数据被反复搬运
- MTE2 带宽利用率低（< 70%）

**原理**：将 K 方向上被切碎的小块数据合并成一次大搬运（≥ 20 KB），摊薄 MTE2 发射头开销，使带宽利用率从 50%–70% 拉回到 80%+。

**优化前**（每轮都搬小块 scale）：
```python
for k in T.serial(loop_k):
    T.copy(scale[k * base_scale:(k + 1) * base_scale], scale_l1)  # 每次 2 KB
    T.copy(data[k * block_N:(k + 1) * block_N, :], data_l1)
    T.gemm_v0(data_l1, scale_l1, acc_l0c)
```

**优化后**（合并多轮 scale 一次搬运）：
```python
# 合并 8 轮 scale 一次搬运（2 KB × 8 = 16 KB）
for k in T.serial(loop_k):
    if k % 8 == 0:
        T.copy(scale[k * base_scale:(k + 8) * base_scale], scale_l1_merged)
    # 从合并 buffer 中按偏移取对应片
    T.copy(data[k * block_N:(k + 1) * block_N, :], data_l1)
    T.gemm_v0(data_l1, scale_l1_merged[k % 8], acc_l0c)
```

### 2.6 指令向量化（Vector 核）

**适用场景**：
- 代码中存在 for 循环下的多次 scalar 运算（逐行/逐元素操作）
- 使用 `range()` 循环对 tensor 的多个切片分别执行相同操作
- 算子包含大量逐元素数学运算（如 Softmax 中的逐行归一化）

**注意**：向量化改造必须保证运算逻辑不变，特别是存在数据依赖或累加操作的场景，需仔细验证等价性。

**优化前**（循环中多次 scalar 运算）：
```python
for h_i in range(block_M // 2):
    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
```

**优化后**（单次 tile 操作）：
```python
T.tile.broadcast(m_i_2d, m_i, tmp_ub)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

### 2.7 指令融合（Vector 核）

**适用场景**：
- 符合特定模式的连续运算（如 `y = a * x + y`）
- 需要减少指令下发次数

**AXPY 融合**：`dst = scalar * src0 + dst`

**优化前**（两条指令）：
```python
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

**优化后**（使用 AXPY 融合）：
```python
T.tile.axpy(acc_s_ub, m_i_2d, sm_scale)
```

**其他融合指令**：
- `T.tile.leaky_relu(dst, src0, scalar)`：ReLU + 乘法融合（`dst = max(0, src0) if src0 >= 0 else src0 * scalar`）

**提示**：除上述融合指令外，应主动搜索代码中可融合的计算模式，尝试使用 `T.tile` 提供的其他复合运算指令（如 `T.tile.select`、`T.tile.clamp`、`T.tile.compare` 等）替代多步基础运算。

> **注意**：实施指令融合前必须询问用户确认，说明融合方案和预期收益，经用户同意后再修改代码。

### 2.8 稀疏访存优化（Vector 核）

**适用场景**：
- KV 数据在 Global Memory 中呈离散分布（如 Paged Attention、Sparse Attention）
- 使用索引表/页表访问 KV 数据
- 需要先将离散数据 Gather 为连续块再进行计算

**优化前**（逐元素 Gather + 频繁同步）：
```python
# 单 buffer，每次循环搬运后立即写出，且包含大量 barrier
kv_ub = T.alloc_ub([D], dtype)
kv_tail_ub = T.alloc_ub([D_tail], dtype)

for bi_i in range(BI // 2):
    index_i = indices_ub_[bi_i + vid * BI // 2]
    T.barrier_all()
    if index_i > -1:
        block_idx = index_i // block_size
        block_i = block_table[b_i, block_idx]
        block_inter = index_i % block_size
        T.barrier_all()
        # 逐元素离散拷贝
        T.copy(KV[block_i, block_inter, 0, :D], kv_ub)
        T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub)
    else:
        T.tile.fill(kv_ub, 0.0)
        T.tile.fill(kv_tail_ub, 0.0)
    T.barrier_all()
    # 逐元素写出到 Workspace
    T.copy(kv_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
    T.copy(kv_tail_ub, workspace_2[cid, bi_i + vid * BI // 2, :])
    T.barrier_all()
```

**优化后**（双 Buffer Gather + 批量写出）：
```python
# 分配双 Buffer 用于 Gather
kv_ub_gather = T.alloc_ub([BI // 2, D], dtype)
kv_tail_ub_gather = T.alloc_ub([BI // 2, D_tail], dtype)

for bi_i in range(BI // 2):
    index_i = indices_ub_[bi_i + vid * BI // 2]
    block_idx = index_i // block_size
    block_i = block_table[b_i, block_idx]
    block_inter = index_i % block_size
    # 离散数据 Gather 到双 Buffer（减少 barrier）
    T.copy(KV[block_i, block_inter, 0, :D], kv_ub_gather[bi_i, :])
    T.copy(KV[block_i, block_inter, 0, D:], kv_tail_ub_gather[bi_i, :])

# Gather 完成后，一次性批量写出到 Workspace
T.copy(kv_ub_gather, workspace_1[cid, vid * BI // 2 : (vid + 1) * BI // 2, :])
T.copy(kv_tail_ub_gather, workspace_2[cid, vid * BI // 2 : (vid + 1) * BI // 2, :])
```

**关键优化点**：
- **离散 KV Gather**：先将离散 KV 从 GM 收集到 UB 的连续区域，再一次性搬出
- **双 Buffer 机制**：使用 `[BI // 2, D]` 的双 buffer 替代单 buffer，支持 Gather 与后续计算的流水掩盖
- **减少同步**：移除循环内的 `T.barrier_all()` 和条件分支，提升指令下发效率

### 2.9 Fixed Core 模式（所有算子类型通用）

**适用场景**：
- 逻辑任务数远大于物理核数（如 block_num >> 24）
- Workspace 显存分配随 block_num 线性增长
- 算子包含大量 `alloc_buffer`、`annotate_address` 等初始化操作

**优化前**（按逻辑任务数 launch）：
```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    workspace = T.alloc_L1([block_M, block_N], dtype)
    T.copy(result, workspace[cid, :, :])
```

**优化后**（按物理核数 launch，手动分配任务）：
```python
with T.Kernel(core_num, is_npu=True) as (cid, vid):
    workspace = T.alloc_L1([block_M, block_N], dtype)
    single_core_load = T.ceildiv(block_num, core_num)
    for block_idx in T.serial(cid * single_core_load, (cid + 1) * single_core_load):
        ...
        T.copy(result, workspace[cid, :, :])  # workspace[cid] 被复用
```

### 2.10 pass_configs 调优（最后手段）

> **注意**：更改 pass_configs 设置相当于少用 Developer 特性，应在其他优化手段尝试后再使用。此优化适用于所有算子类型（核内/核间）。

#### 关闭自动同步

**适用场景**：
- 以上优化手段均已尝试，性能仍不达标
- 使用 Expert 模式且需要精确控制同步时机
- 自动插入的同步指令导致不必要的等待（可通过查看生成的 Ascend C 代码确认）

**优化前**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

**优化后**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
# 手动插入 T.barrier_all() / T.set_flag / T.wait_flag
```

---

### 2.11 UB 预算优化（Vector 核 tile size 首选方法）

**适用场景**：Vector 型算子的 tiling 参数（`block_S` / `block_C`）需要确定，且 Double Buffer / T.Pipelined 已确定 buffer 份数。

> **唯一规则**：tiling 参数**必须通过 UB 预算反推**（而非硬编码），不设硬编码上限（如 512/1024），UB 预算公式通过 `N_cal`/`N_input` 自然约束 tile 大小。

**硬件常量（A2/A3）**：UB 预算 `192 * 1024` 字节；tile 尺寸必须 16 元素对齐；fp16/bf16 存储 2 字节、计算提升到 fp32 为 4 字节。

#### 预算公式

```python
def find_max_tile(total_dim, fixed_dim, N_cal, N_input, dtype_str):
    """根据 UB 容量反推最大 tile 尺寸。"""
    UB_BUDGET = 192 * 1024
    cal_bytes   = 4 if dtype_str in ("float16", "bfloat16") else int(dtype_str[-2:]) // 8
    input_bytes = cal_bytes if dtype_str == "float32" else 2
    c = max(fixed_dim, 1)
    per_unit = c * (N_cal * cal_bytes + N_input * input_bytes)
    max_tile = (UB_BUDGET // per_unit // 16) * 16
    max_tile = min(max_tile, ((total_dim + 15) // 16) * 16)
    for t in range(max_tile, 0, -16):
        if total_dim % t == 0:
            return t
    best = max_tile
    min_pad = float("inf")
    for t in range(max_tile, 0, -16):
        pad = (t - total_dim % t) % t
        if pad < min_pad:
            min_pad = pad
            best = t
    return max(16, best)
```

**枚举 buffer 要点**（决定 `N_cal` / `N_input`）：
- 只计算**同时活跃**的 buffer，双缓冲 input/output 算双份
- 开启 `TL_ASCEND_MEMORY_PLANNING: True` 后编译器自动复用已死亡 buffer

#### Vector 核参数广播模式（5 步）

参数在 kernel 签名中**必须声明为 2D** `(G, rows)`，kernel 内 copy **必须用 1D 切片** `param[g, 0:rows]`（禁止 2D 切片，否则 DMA stride 导致数据错位）。

```python
rows_padded = ((rows + 15) // 16) * 16
# Step 1: 16 对齐 staging buffer
gamma_raw = T.alloc_ub([rows_padded, 1], dtype)
# Step 2: fill + pad_value 拷贝（必须 1D 切片）
T.tile.fill(gamma_raw, 1.0)
T.copy(gamma[g, 0:rows], gamma_raw, pad_value=0.0)
# Step 3: cast
T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, rows_padded)
# Step 4: broadcast 到对齐维度
gamma_bc_full = T.alloc_ub([rows_padded, block_S], cal_dtype)
T.tile.broadcast(gamma_bc_full, gamma_cal)
# Step 5: 切片到实际维度
gamma_bc = T.alloc_ub([rows, block_S], cal_dtype)
T.copy(gamma_bc_full[0:rows, 0:block_S], gamma_bc)
```

> `gamma_bc` 放循环外，生命周期覆盖所有迭代。`gamma_bc_full` 切片后即可释放。

#### 实施顺序

UB 预算**必须在所有 kernel 级优化完成后最后实施**（多行 Tile 改变 tile 形状、Double Buffer 让 buffer 翻倍、指令融合减少临时 buffer——任何一项在 UB 预算之后实施都会导致返工重算）。

#### 检查清单

- [ ] tile size 通过 `find_max_tile` 反推（非硬编码），无硬编码上限？
- [ ] 枚举了所有同时活跃的 buffer（双缓冲算双份）？
- [ ] 16 元素对齐？优先找整除值？
- [ ] 开启 `TL_ASCEND_MEMORY_PLANNING: True`？
- [ ] UB 预算在所有 kernel 级优化完成后最后实施？

---

### 2.12 Host 侧预处理内化（消除 pad / contiguous）

**适用场景**：Host 中有 `F.pad` / `.contiguous()` / padded tensor 传入 kernel。

**核心思路**：Host 直接传原始 tensor（不 pad、不 contiguous），Kernel 用「整块循环 + 余数块」覆盖所有数据，Host 对输出做 view-slice 裁剪。

#### 关键约束

| 编号 | 约束 | 违反后果 |
|------|------|---------|
| **K1** | `n_full`/`partial` 必须是 JIT 编译期 Python 整数（`@tilelang.jit` 函数体内用 `//` 和 `%` 计算） | TIR 变量导致条件无法求值，编译失败 |
| **K2** | 余数块 buffer 必须用 `T.alloc_ub`（`pad_value` 仅支持 GM→UB） | L1/shared 路径无此参数 |
| **K3** | 余数块 buffer 按完整 `[row_chunk, block_N]` 分配（非 partial 大小） | `partial==0` 时未定义行为 |
| **K4** | 输出 tensor 归约维度声明为 `padded_group_size` | 否则写穿 GM 边界 |
| **K5** | `T.tile.fill(buf, 0)` 在 `T.copy(src, buf, pad_value=0)` **之前** | 否则 fill 覆盖已拷贝数据 |
| **K6** | 整块循环用 `n_full = total_dim // block_N`（非 `padded_dim`） | 否则越界读取 |

#### Host 端改造

```python
# 优化前
x_2d = torch.nn.functional.pad(x_2d, (0, pad_size), value=0)
x_2d = x_2d.contiguous()
output_2d = func(x_2d)

# 优化后
x_2d = x.reshape(total_rows, group_size)
output_2d = func(x_2d)
output = output_2d[:, :group_size].reshape(x.shape)  # view-slice，零拷贝
```

Kernel 签名：input 用原始 `group_size`，output 保持 `padded_group_size`（K4）。

#### 整块循环

```python
n_full = total_dim // block_N
for by in T.serial(n_full):
    T.copy(data[row:row+row_chunk, by*block_N:(by+1)*block_N], a_ub)
    # ... 与原始代码相同的计算 ...
```

#### 余数块（`partial > 0` 时）

```python
partial = total_dim % block_N
if partial > 0:
    a_p = T.alloc_ub([row_chunk, block_N], dtype)   # K3: 完整块大小
    T.tile.fill(a_p, 0.0)                            # K5: 必须先清零
    T.copy(data[row:row+row_chunk, n_full*block_N:total_dim], a_p, pad_value=0)
    # ... 计算（Pass 1: 累加 / Pass 2: 输出）...
    T.copy(a_p, output[row:row+row_chunk, n_full*block_N:n_full*block_N+block_N])
```

#### 检查清单

- [ ] `F.pad` + `.contiguous()` 已删除？
- [ ] `n_full`/`partial` 为 Python 整数（K1）？buffer 用 `T.alloc_ub`（K2）？
- [ ] 余数块 buffer 为完整块大小（K3）？output 按 padded 声明（K4）？
- [ ] fill 在 copy 之前（K5）？循环用 `n_full`（K6）？
- [ ] 优化后精度与优化前一致？

---

### 2.13 多行 Tile 粒度扩展（Multi-row Tile Granularity）

**适用场景**：Vector 算子存在"外层维度循环 + 内层 tile 处理"的双重循环结构（如 `for c in serial(cpg): for s in serial(s_num):`），每次 tile 只处理 `[1, block_S]`。将多行合并为 `[rows, block_S]` tile，消除外层循环。

**原理**：将 `(N, C, S)` reshape 为 `(N, G, cpg, S)` 4D layout，使组内 cpg 行在 GM 中连续，kernel 内用 `[cpg, block_S]` tile 一次处理所有通道。

---

#### 决策依赖链

多行 Tile 的决策分三层，上层约束下层：

```
Layer 1（Layout 约束）：数据在 GM 中是否连续？→ 决定能不能做多行 Tile、用什么 layout
    ↓ 约束了 rows 的取值和内存排列
Layer 2（维度设计）：rows / rows_padded / block_S / S_padded → 决定所有 buffer 的 shape
    ↓ 约束了 buffer 大小和归约范围
Layer 3（计算模式）：归约 / broadcast / V pipe buffer → 决定代码怎么写
```

查阅时按需定位：需要知道"能不能做" → Layer 1；"维度怎么算" → Layer 2；"代码怎么写" → Layer 3；报错了 → 错误速查表。

---

#### Layer 1：Layout 约束

**规则**：多行 slice `T[..., a:a+R, b:b+BS]` 要求 R 维和 BS 维在内存中**相邻**（R 的 stride = BS 的长度）。不满足时 `T.copy` 输出全错且**无编译报错**。

| 当前 layout | 切片维度 | 是否连续 | 修复方式 |
|------------|---------|---------|---------|
| `(N, C, S)` | C 维 | ❌ stride=S，各行断开 | reshape 为 `(N, G, cpg, S)`，切 cpg 维 |
| `(N, G, cpg, S)` | cpg 维 | ✅ stride=S | 无需修改 |
| `(N, H, W)` | H 维 | ❌ stride=W | reshape 暴露连续维 |

**验证方法**：用 `torch.arange(N*C*S).reshape(shape)` 创建测试 tensor，`T.copy` 后读出比对值是否连续递增。

**MIX 类型 guard**：`get_kernel_source()` 含 `KERNEL_TYPE_MIX` 时，所有 buffer 分配和计算必须包在 `if vid == 0:` 内。否则多个 AIV 重复执行，结果翻倍或 UB 越界。

---

#### Layer 2：维度设计

**Host 侧计算**（每步依赖前一步）：

```python
rows = C // G                                    # 要合并的行数
rows_padded = max(((rows + 15) // 16) * 16, 16)  # cast 对齐，至少 16
block_S = find_block_S(S, rows, dtype_str)       # 动态计算（不可硬编码）
s_num = (S + block_S - 1) // block_S
S_padded = s_num * block_S                       # 输出末维必须对齐

x_4d = x.reshape(N, G, rows, S)                  # 4D 暴露连续行维
gamma_2d = gamma.reshape(G, rows)                 # 2D，kernel 内用 1D 切片
```

> **为什么 block_S 不能硬编码**：多行 Tile 使 UB 占用从 `O(block_S)` 变为 `O(rows × block_S)`。rows=257 时 block_S=256 需要 257×256×4×6 ≈ 1.5MB，远超 UB 的 192KB。

**`find_block_S` 函数**（直接复用）：

```python
def find_block_S(S, rows, dtype_str):
    UB_BUDGET = 192 * 1024
    cal_bytes = 4 if dtype_str in ("float16", "bfloat16") else int(dtype_str[-2:]) // 8
    dtype_bytes = cal_bytes if dtype_str == "float32" else 2
    per_block = max(rows, 1) * (6 * cal_bytes + 6 * dtype_bytes)
    max_bs = (UB_BUDGET // per_block // 16) * 16
    max_bs = min(max_bs, 512)
    for bs in range(max_bs, 0, -16):
        if S % bs == 0: return bs
    best, min_pad = max_bs, float("inf")
    for bs in range(max_bs, 0, -16):
        pad = (bs - S % bs) % bs
        if pad < min_pad: min_pad, best = pad, bs
    return max(16, best)
```

典型值：rows=4→512, rows=16→256, rows=127→32, rows=257→16。

**Kernel 签名**（分离"对齐/实际"和"输入/输出"维度）：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 必须开启
}

@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def kernel(N, G, rows_padded, S_padded, block_S, s_num, eps, rows, S_orig, dtype):
    @T.prim_func
    def main(
        x: T.Tensor((N, G, rows, S_orig), dtype),       # 输入：实际 rows + 原始 S
        gamma: T.Tensor((G, rows), dtype),               # per-row 参数：2D
        beta: T.Tensor((G, rows), dtype),
        y: T.Tensor((N, G, rows, S_padded), dtype),      # 输出：实际 rows + padded S
    ):
```

签名规则：
- 输入 tensor 用 `S_orig`，输出 tensor 用 `S_padded`（防止尾块越界写，无编译报错）
- gamma/beta 声明为 `(G, rows)` 2D，kernel 内用 `gamma[g, 0:rows]` 1D 切片
- `rows_padded` 和 `rows` 都作为 kernel 参数传入（分别用于对齐 buffer 和实际计算）

---

#### Layer 3：计算模式

##### 归约模式

**规则**：循环内**只用 `T.tile.add` 累积**，禁止 `reduce_sum`；循环外用 `T.reduce_sum(dim=-1)` **两步归约**。`reduce_sum` 输出必须是 **1D buffer**（`[rows]` 不是 `[rows, 1]`），否则报 `Invalid reduce output shape`。**只用 `dim=-1`**（`dim=0` 实测结果随机）。

```python
# kernel 函数体内首先定义这些辅助变量
tile_elem = rows * block_S
use_fp32 = dtype in ("float16", "bfloat16")
cal_dtype = "float32" if use_fp32 else dtype

with T.Kernel(N * G, is_npu=True) as (cid, vid):
    n = cid // G; g = cid % G
    if vid == 0:                                    # MIX 类型必须加 guard
        sum_a = T.alloc_ub([rows, block_S], cal_dtype)
        data_buf = T.alloc_ub([rows, block_S], dtype)
        data_cal = T.alloc_ub([rows, block_S], cal_dtype)

        with T.Scope("V"):
            T.tile.fill(sum_a, 0.0)
            for si in T.serial(s_num):
                s_off = si * block_S
                T.copy(x[n, g, 0:rows, s_off:s_off+block_S], data_buf)
                if use_fp32:
                    T.tile.cast(data_cal, data_buf, CAST_LOW2HIGH, rows * block_S)
                else:
                    T.copy(data_buf, data_cal)
                T.tile.add(sum_a, sum_a, data_cal)   # 循环内只累积

            # 循环外两步归约
            row_buf = T.alloc_ub([rows], cal_dtype)   # 1D！不是 [rows, 1]
            total = T.alloc_ub([1], cal_dtype)
            T.reduce_sum(sum_a, row_buf, dim=-1)       # [rows, BS] → [rows]
            T.reduce_sum(row_buf, total, dim=-1)        # [rows] → [1]

            # 计算统计量（用 T.cast 做标量类型转换）
            cnt = T.cast(rows * S_orig, cal_dtype)
            T.tile.div(mean, total, cnt)
```

**归约链路模板**（按算子类型选择）：

```
跨行共享统计量（GroupNorm / BatchNorm）：
  [rows, BS] → reduce(dim=-1) → [rows] → reduce(dim=-1) → [1]
  [1] → fill([rows, 1]) → broadcast → [rows, BS]

逐行独立统计量（RMSNorm / LayerNorm / Softmax）：
  [rows, BS] → reduce(dim=-1) → [rows]
  [rows] → fill([rows, 1]) → broadcast → [rows, BS]
```

> 共享统计量需要额外一步 `[rows] → [1]` 归约（所有行共享同一个 mean/std）；逐行统计量不需要。

##### broadcast 模式

两种场景，维度选择不同：

| 场景 | 数据来源 | 中间 buffer 维度 | 示例 |
|------|---------|-----------------|------|
| **A：标量扩展** | UB 内的统计量 | `[rows, 1]`（实际 rows） | mean, std |
| **B：GM 参数加载** | GM 中的 per-row 参数 | `[rows_padded, 1]`（对齐值） | gamma, beta |

**场景 A**（`[1] → [rows, 1] → [rows, block_S]`）：

```python
mean_col = T.alloc_ub([rows, 1], cal_dtype)
mean_bc = T.alloc_ub([rows, block_S], cal_dtype)
T.tile.fill(mean_col, mean_val)                   # fill 支持 buffer 值源
T.tile.broadcast(mean_bc, mean_col)               # 只沿 dim=-1 扩展
```

**场景 B**（`GM → [rows_padded, 1] → [rows_padded, BS] → 切片 → [rows, BS]`）：

```python
gamma_raw = T.alloc_ub([rows_padded, 1], dtype)
gamma_cal = T.alloc_ub([rows_padded, 1], cal_dtype)
gamma_bc_full = T.alloc_ub([rows_padded, block_S], cal_dtype)
gamma_bc = T.alloc_ub([rows, block_S], cal_dtype)

T.copy(gamma[g, 0:rows], gamma_raw, pad_value=0.0)  # 1D 切片 + pad_value
T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, rows_padded)  # cast 用 rows_padded
T.tile.broadcast(gamma_bc_full, gamma_cal)
T.copy(gamma_bc_full[0:rows, 0:block_S], gamma_bc)  # 切片回实际 rows
```

broadcast 规则：
- **只沿 dim=-1 扩展**：`[1]→[rows, BS]` 必须两步（fill → broadcast），一步到位会编译报错
- `T.tile.fill` 支持 buffer 值源（`T.tile.fill(col, scalar_buf)`），不需要 Python 字面量
- `T.copy` 不能做 1D↔2D 转换（报 `StructuralEqual check failed`），用 `T.tile.fill` 替代
- cast 元素数必须用 `rows_padded`（AscendC cast 要求 16 对齐）

##### V pipe buffer 分离

`AUTO_SYNC=True` 下编译器自动插入 `PipeBarrier<PIPE_V>`，V pipe 串行化，可以安全地 in-place 复用 buffer（dst 和 src 相同）：

```python
# 归一化计算链（AUTO_SYNC=True 下 in-place 安全）
T.tile.sub(data_cal_p2, data_cal_p2, mean_bc)
T.tile.div(data_cal_p2, data_cal_p2, std_bc)
T.tile.mul(data_cal_p2, data_cal_p2, gamma_bc)
T.tile.add(data_cal_p2, data_cal_p2, beta_bc)
```

> 直接 `div(std_bc)` 除 std，不要先算 `rstd = 1/std` 再 `mul(rstd)`（多一步除法引入浮点误差）。
>
> 若后续改为 `AUTO_SYNC=False`（如做 Double Buffer），连续 tile 指令的 dst 必须交替使用不同 buffer（V pipe 异步发射，读到旧值）。

---

#### 错误速查表

| 报错 | 所属层 | 修复 |
|------|--------|------|
| `T.copy` 输出全错（无编译报错） | Layer 1 | 改 4D layout 使行维连续 |
| 结果翻倍 / UB 越界 | Layer 1 | MIX 类型加 `if vid == 0:` guard |
| cpg 大时 aicore exception | Layer 2 | `find_block_S()` 动态计算 + `MEMORY_PLANNING` |
| 尾块数据错误 | Layer 2 | 输出末维用 `S_padded` |
| gamma/beta cast 报错 | Layer 2 | cast 元素数用 `rows_padded` |
| `Invalid reduce output shape` | Layer 3 | reduce 输出改为 1D `[rows]` |
| 归约结果 NaN / 随机 | Layer 3 | 循环外归约 + `dim=-1` |
| `Broadcast dimension mismatch` | Layer 3 | broadcast 分两步（fill → broadcast） |
| `StructuralEqual check failed` | Layer 3 | 用 `T.tile.fill` 替代 `T.copy` 做跨维度传递 |
| 连续 tile 指令结果异常 | Layer 3 | `AUTO_SYNC=False` 时 dst 需交替使用独立 buffer；`AUTO_SYNC=True` 下 in-place 安全 |

---

## 三、核间优化

> **适用对象**：仅 **CV 融合型算子** 需要执行核间优化。纯 Cube 或纯 Vector 算子跳过本章。

### 3.1 num_stages 调优

**适用场景**：
- 使用 `T.Pipelined` 进行核间流水优化
- 循环次数较多（如 `loop_range ≥ 4`）
- Cube 核和 Vector 核的耗时差异较大，存在明显核间等待气泡

**调优建议**：
- **约束**：`num_stages ≥ 2` 且 `num_stages ≤ loop_range`（最大不超过循环次数）
- 循环次数较多或 CV 耗时差距大时，需要较大的 `num_stages` 值
- 从 `num_stages=2` 开始，逐步增加，观察性能变化选择最优值
- 注意 `num_stages` 过大会增加内存占用。开启 `TL_ASCEND_MEMORY_PLANNING` 后，如果内存超限会报错，此时应调小 `num_stages` 数量

### 3.2 核间同步优化

**适用场景**：
- CV 交互次数多，循环次数多
- 注释掉所有计算和搬运代码后，仅保留核间同步的耗时占比 > 50%

**调优建议**：
- 此操作会降低 CV 间并行度，需谨慎使用
- 同步间隔等参数最大调节到 2
- 实施后必须验证性能收益，如果没有收益则立即回退

**优化前**（每次任务都同步）：
```python
for i in range(n):
    process()
    T.set_cross_flag("FIX", SEM_ID)
    T.wait_cross_flag(SEM_ID)
```

**优化后**（多次任务后同步）：
```python
for i in range(n):
    process()
    if (i + 1) % cross_interval == 0 or i == n - 1:
        T.set_cross_flag("FIX", SEM_ID)
        T.wait_cross_flag(SEM_ID)
```

> **核间 Pipeline**：使用 `T.Pipelined` 实现核间流水掩盖，详见 [T.Pipelined 使用教程](../../../../docs/tutorials/t_pipelined.md)。

---

## 四、常见问题速查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| C 核大量气泡 | V 核耗时长，`num_stages` 太小 | 增大 `num_stages` |
| 内存溢出 | `num_stages` 过大或 buffer 过大 | 减小分块参数或 `num_stages` |
| 指令下发慢 | scalar 操作过多 | 改用 `T.tile` 向量化操作 |
| GM 带宽未打满 | 数据搬运效率低 | 开启 L1 常驻、Double Buffer |
| scalar bound 高 | 同步次数过多 | 减少 sync 频率，使用 `cross_interval` |
| 多行 tile 归约结果 NaN | 循环内逐块 `reduce_sum` + `dim=0` | 改为循环外 `reduce_sum(dim=-1)` 两步归约，见 §2.13 P1 |
| fill + broadcast segfault | fill 源 buffer 用了 `rows_padded` | 标量扩展用实际 `rows`，见 §2.13 P2 |
| 连续 tile 指令结果异常 | V pipe 异步发射导致读到旧值 | 下游指令 `dst` 用独立 buffer，见 §2.13 P3 |
| broadcast 编译报错 dim mismatch | 试图 `[1,1] → [rows, block_S]` 一步到位 | 使用 `[rows, 1]` 中间态，见 §2.13 P4 |
| 多行 tile 性能退化到单行水平 | Double Buffer 用 `[2, block_S]`（2D） | 改为 `[2, rows, block_S]`（3D），见 §2.2 |
| 多行 tile 2D copy 输出错误 | 数据 layout 不连续（多行在内存中 stride 断开） | 改为 4D layout `(N, G, cpg, S)`，见 §2.13 前置条件 |
| `reduce_sum` 结果全错 | 使用了 `dim=0` 或 dim≠-1 | 优先尝试 `dim=-1` 分两步归约，见 §2.13 P1 |
| 第二遍 copy X 数据导致输出异常 | gamma/beta broadcast 后缺 `T.barrier_all()` | 在 gamma/beta 广播后插入 `T.barrier_all()` |
| 输出末位元素为随机值 | 输出 tensor 末维未对齐，尾块越界写 | 末维 pad 到 `s_num * block_S`，host 切片取回，见 §2.13 P5 |
| Mix 类型算子结果加倍 | 未加 `if vid == 0:` guard，多个 AIV 重复执行 | 加 `if vid == 0:` guard，见 §2.13 P6 |
| reduce_sum 编译报错 `Invalid reduce output shape` | 输出用了 `[rows, 1]` 2D buffer 而非 1D `[rows]` | output 改为 `alloc_ub([rows], dtype)` 纯 1D，见 §2.13 P1 |
| cpg 较大时 aicore exception（UB 越界） | 固定 block_S=256 导致多行 tile buffer 溢出 UB | 使用 `find_block_S()` 动态计算 + 开启 `MEMORY_PLANNING`，见 §2.13 |
| `T.copy` 编译报 `StructuralEqual check failed` | 试图用 copy 做 1D↔2D shape 转换 | 改用 `T.tile.fill` 跨维度值传递，见 §2.13 P4 |
