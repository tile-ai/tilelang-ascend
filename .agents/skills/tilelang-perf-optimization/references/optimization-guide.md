# TileLang 性能优化指南

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
核内优化 → 核间优化 → pass_configs 调优（最后手段）
```

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
- 同步方式：Vector/Cube 核内流水当前需要手动控制不同流之间的同步。手写双缓冲时可先开启 `TL_ASCEND_AUTO_SYNC: True` 让编译器自动插入同步，若翻译结果不符合预期再改为手动 `set_flag` / `wait_flag`

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

**优化后**（手写 Ping-Pong 双缓冲，开启自动同步）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# 分配双缓冲
ub_buf = T.alloc_ub([2, block_size], dtype)
result_buf = T.alloc_ub([2, block_size], dtype)

for k in T.serial(loop_k):
    side = k % 2
    T.copy(GM_data[k], ub_buf[side, :])
    T.tile.exp(result_buf[side, :], ub_buf[side, :])
    T.copy(result_buf[side, :], GM_out[k])
```

**Vector 核内三阶段流水 reference**：

完整说明见 [vector_add_pipeline](best-practices/vector_add_pipeline.md)。该模板使用两个 UB stage 和 `mte3 -> mte2`、`mte2 -> v`、`v -> mte3` 事件，把 Vector 算子组织为：

- prefetch：预取第 0 个 tile 到 stage 0
- main body：预取 `tile + 1`，同时消费并写回 `tile`
- epilogue：消费并写回最后一个已预取 tile

编写 Vector pipeline 时优先保持这三个阶段的边界清晰，再替换 main body 中的 `T.tile.add` 为目标算子的 Vector 计算。

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
