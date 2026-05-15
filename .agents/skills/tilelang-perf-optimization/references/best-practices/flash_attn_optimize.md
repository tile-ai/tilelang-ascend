# Flash Attention 算子性能优化最佳实践

本文档总结了 Flash Attention 算子在 TileLang-Ascend 上的高级性能优化手段，对比基础实现与 Expert 优化版本的关键差异。

---

## 优化概览

| 优化项 | 基础实现 (`flash_attn_bhsd.py`) | Expert 优化 (`fa_opt/flash_attn_bhsd_expert_h16_d128.py`) | 性能收益 |
|--------|--------------------------------|----------------------------------------------------------|---------|
| 流水线深度 | 单缓冲，逐块处理 | 多缓冲流水线（num_stages=14） | 隐藏搬运延迟 |
| 同步机制 | `barrier_all` + 简单 cross_flag | 细粒度 Flag + cross_interval 批量同步 | 减少同步开销 |
| 计算指令 | `T.gemm_v0` 高层抽象 | `T.mma` intrinsic + L0 双缓冲 | 硬件最优 |
| 数据布局 | 默认布局 | `make_zn_layout/nz_layout` 优化 | 提高带宽利用率 |
| 任务分配 | 简单 block_num 映射 | 静态任务分配 + NUM_CORES | 负载均衡 |
| Softmax 计算 | 逐块计算 | 批量 Softmax（num_stages 批次） | 减少同步 |
| 内存复用 | 独立缓冲 | io_buf, buf_2d, work_ub 复用 | 减少内存占用 |
| Pass 配置 | 默认配置 | 关闭所有自动化 Pass | 完全手动控制 |
| Workspace 结构 | `[block_num, block_M, block_N]` | `[NUM_CORES, num_stages, block_M, block_N]` | 优化数据布局 |

---

## 核心优化技术详解

### 1. 多级流水线与多缓冲（num_stages）

#### 基础实现

```python
# 单缓冲：逐块处理，每次只处理一个 K/V 分块
for k in T.serial(T.ceildiv(seq_len, block_N)):
    # 1. 搬运 K/V 分块
    T.copy(K[...], k_l1)
    T.copy(V[...], v_l1)
    
    # 2. 计算 QK^T
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, ...)
    
    # 3. Softmax（等待 Cube 核完成）
    T.wait_cross_flag(0)
    
    # 4. 计算 Attention*V
    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, ...)
```

**问题**：
- 数据搬运和计算串行执行
- 无法隐藏内存访问延迟
- 每次都要等待跨核同步

#### Expert 优化

```python
# 多缓冲流水线：批量处理多个 K/V 分块
num_stages = 14  # 一次处理 14 个 K/V 分块
num_outer = T.ceildiv(num_iters, num_stages)  # 外层循环次数

for k in T.serial(num_outer):
    # 计算 batch_iters = min(num_stages, remaining)
    batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)
    
    # GEMM1: 批量计算 QK^T（14 个分块）
    for i in T.serial(batch_iters):
        side = i % 2  # 双缓冲轮转
        # 使用 L0 双缓冲 + MMA intrinsic
        T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
        T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
        # 每隔 cross_interval 次同步一次
        if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
            T.set_cross_flag("FIX", SEM_WS1_C2V)
    
    # GEMM2: 批量计算 Attention*V（14 个分块）
    for i in T.serial(batch_iters):
        T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
        T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
        if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
            T.set_cross_flag("FIX", SEM_WS3_C2V)
```

**收益**：
- 批量处理减少同步次数（从 `seq_len/block_N` 次减少到 `num_outer` 次）
- 流水线并行：数据搬运和计算重叠
- 典型性能提升：50-100%

**关键参数**：
- `num_stages`：流水线深度（推荐 8-16）
- `cross_interval`：跨核同步间隔（推荐 2-4）
- `num_outer`：外层循环次数 = `ceil(seq_len/block_N / num_stages)`

---

### 2. 细粒度 Flag 同步机制

#### 基础实现

```python
# 所有流水同步：开销大
T.barrier_all()  # Cube 核内部同步
T.set_cross_flag("FIX", 0)  # 通知 Vector 核
T.wait_cross_flag(1)  # 等待 Vector 核
T.barrier_all()  # 再次所有流水同步
```

**问题**：
- `barrier_all` 是所有流水同步，开销大
- 每次 K/V 分块都要跨核同步

#### Expert 优化

```python
# Intra-core Flag：Cube 核内部流水线同步
SIG_K_L1 = 0    # K 搬运到 L1 的信号
SIG_P_L1 = 1    # P 搬运到 L1 的信号
SIG_V_L1 = 2    # V 搬运到 L1 的信号
SIG_L0AB = 3    # L0A/L0B 双缓冲基址
SIG_L0C = 5     # L0C 双缓冲基址

# Cross-core Semaphore：Cube ↔ Vector 同步
SEM_WS1_C2V = 0  # workspace_1 (QK^T) 就绪
SEM_WS1_V2C = 1  # workspace_1 被消费
SEM_WS2_V2C = 2  # workspace_2 (softmax) 就绪
SEM_WS2_C2V = 3  # workspace_2 被消费
SEM_WS3_C2V = 4  # workspace_3 (Attention*V) 就绪
SEM_WS3_V2C = 5  # workspace_3 被消费

# 初始化 Flag（模拟消费者已释放）
T.set_cross_flag("MTE2", SEM_WS2_C2V)
T.set_flag("MTE1", "MTE2", SIG_K_L1)
T.set_flag("MTE1", "MTE2", SIG_P_L1)
T.set_flag("MTE1", "MTE2", SIG_V_L1)
T.set_flag("M", "MTE1", SIG_L0AB)
T.set_flag("M", "MTE1", SIG_L0AB + 1)
T.set_flag("FIX", "M", SIG_L0C)
T.set_flag("FIX", "M", SIG_L0C + 1)

# 流水线同步示例
T.wait_flag("MTE1", "MTE2", SIG_K_L1)  # 等待 K 缓冲可用
T.copy(K[...], k_l1)                   # 搬运 K
T.set_flag("MTE2", "MTE1", SIG_K_L1)   # 通知 K 已就绪

T.wait_flag("MTE2", "MTE1", SIG_K_L1)  # 等待 K 就绪
T.copy(k_l1, l0b[side, :, :], transpose=True)  # 搬运到 L0B
T.set_flag("MTE1", "MTE2", SIG_K_L1)   # 通知 L1 缓冲可重用
T.set_flag("MTE1", "M", SIG_L0AB + side)  # 通知 L0 数据就绪

T.wait_flag("MTE1", "M", SIG_L0AB + side)  # 等待 L0 数据就绪
T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
T.set_flag("M", "MTE1", SIG_L0AB + side)  # 通知 L0 缓冲可重用
```

**收益**：
- 精确控制数据流，避免不必要的等待
- 实现 GM → MTE2 → L1 → MTE1 → L0 → M → FIX 的完整流水线
- 批量跨核同步（每 `cross_interval` 次），减少同步开销
- 典型性能提升：30-50%

**关键概念**：
- **Intra-core Flag**：核内部流水线同步（MTE2 ↔ MTE1 ↔ M ↔ FIX）
- **Cross-core Semaphore**：Cube ↔ Vector 核同步
- **双缓冲基址**：`SIG_L0AB` 和 `SIG_L0C` 使用连续编号（side=0/1）

---

### 3. L0 分块与 MMA Intrinsic

#### 基础实现

```python
# 直接从 L1 进行矩阵乘
acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
```

**问题**：
- `gemm_v0` 是高层抽象，无法精细控制 L0 数据流
- 无法实现 L0 双缓冲

#### Expert 优化

```python
# L0 双缓冲分配
l0a = T.alloc_L0A([2, block_M, dim], dtype)     # 2 个 L0A 缓冲
l0b = T.alloc_L0B([2, dim, block_N], dtype)     # 2 个 L0B 缓冲
l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)  # 2 个 L0C 缓冲

# L1 → L0 分块搬运 + MMA 计算
side = i % 2  # 双缓冲轮转

T.wait_flag("M", "MTE1", SIG_L0AB + side)
T.copy(q_l1, l0a[side, :, :])  # Q 搬运到 L0A（只需搬运一次）

T.copy(k_l1, l0b[side, :, :], transpose=True)  # K 搬运到 L0B（需要 transpose）
T.set_flag("MTE1", "M", SIG_L0AB + side)

T.wait_flag("MTE1", "M", SIG_L0AB + side)
T.wait_flag("FIX", "M", SIG_L0C + side)
T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)  # MMA intrinsic
T.set_flag("M", "MTE1", SIG_L0AB + side)
T.set_flag("M", "FIX", SIG_L0C + side)

T.wait_flag("M", "FIX", SIG_L0C + side)
T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])  # 结果搬运到 workspace
T.set_flag("FIX", "M", SIG_L0C + side)
```

**收益**：
- `T.mma` 更贴近 Ascend NPU 硬件的 MMA 指令
- L0 双缓冲实现流水线并行
- 精细控制 L0A/L0B/L0C 的数据流
- 典型性能提升：20-30%

**关键 API**：

- `T.alloc_L0A([2, ...])`：L0A 双缓冲
- `T.alloc_L0B([2, ...])`：L0B 双缓冲
- `T.alloc_L0C([2, ...])`：L0C 双缓冲
- `T.mma`：矩阵乘累加 intrinsic 指令
- `transpose=True`：在搬运时进行 transpose（K 需要转置）

---

### 4. 数据布局优化（Layout Annotation）

#### 基础实现

```python
# 默认布局
q_l1 = T.alloc_L1([block_M, dim], dtype)
k_l1 = T.alloc_L1([block_N, dim], dtype)
```

**问题**：
- 默认布局可能不是最优，影响矩阵乘效率
- K 需要 transpose，布局不匹配会导致性能损失

#### Expert 优化

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout

q_l1 = T.alloc_L1([block_M, dim], dtype)
k_l1 = T.alloc_L1([block_N, dim], dtype)
p_l1 = T.alloc_L1([block_M, block_N], dtype)
v_l1 = T.alloc_L1([block_N, dim], dtype)

T.annotate_layout({
    q_l1: make_zn_layout(q_l1),  # ZN layout for Q
    k_l1: make_nz_layout(k_l1),  # NZ layout for K（适配 transpose）
    p_l1: make_zn_layout(p_l1),  # ZN layout for P
    v_l1: make_zn_layout(v_l1),  # ZN layout for V
})
```

**收益**：
- `make_zn_layout`：采用ZN layout，适配矩阵乘输入
- `make_nz_layout`：采用NZ layout，适配 transpose 操作
- 提高数据搬运和矩阵乘效率
- 典型性能提升：10-20%

**关键概念**：
- **ZN layout**：分形矩阵内部是列主序，适配矩阵乘的输入布局
- **NZ layout**：分形矩阵内部是行主序，适配 transpose 操作
- **矩阵乘布局匹配**：
  - Q @ K^T：Q 使用 ZN，K 使用 NZ（transpose 后适配）
  - P @ V：P 使用 ZN，V 使用 ZN

---

### 5. 静态任务分配与负载均衡

#### 基础实现

```python
# 简单映射：每个 kernel block 处理一个输出块
block_num = seq_len // block_M * heads * batch
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    bx = cid % (seq_len // block_M)
    by = cid // (seq_len // block_M) % heads
    bz = cid // (seq_len // block_M) // heads % batch
```

**问题**：
- 当 `block_num` 大于物理核数时，调度效率低
- 核间负载可能不均衡

#### Expert 优化

```python
NUM_CORES = 24  # 910B has 24 AI Cores

# 静态任务分配：均匀分配任务到 NUM_CORES
q_tasks = block_num // NUM_CORES
r_tasks = block_num % NUM_CORES

def task_range(cid_val):
    """Return (start, count) for core cid_val."""
    start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
    count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
    return start, count

with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
    my_start, my_count = task_range(cid)
    
    for t in T.serial(my_count):
        task_id = my_start + t
        bx = task_id % num_seq_blocks
        by = (task_id // num_seq_blocks) % heads_q
        bz = task_id // (num_seq_blocks * heads_q)
        kv_by = by // (heads_q // heads_kv)  # MQA/GQA 支持
        
        # 处理任务 task_id
```

**收益**：
- 固定物理核数（NUM_CORES=24），避免核调度开销
- 均匀分配任务，前 `r_tasks` 个核多处理一个任务
- 支持循环多任务（每个核处理 `my_count` 个任务）
- 典型性能提升：10-30%（当 `block_num` 远大于核数时）

**关键概念**：
- `q_tasks`：每个核的基础任务数
- `r_tasks`：余数，前 `r_tasks` 个核多处理一个任务
- `task_range(cid)`：返回核 `cid` 的任务范围 `[start, start+count)`

---

### 6. 批量 Softmax 计算

#### 基础实现

```python
# 逐块 Softmax
for _k in T.serial(T.ceildiv(seq_len, block_N)):
    # 1. 等待 QK^T 结果
    T.wait_cross_flag(0)
    T.copy(workspace_1[cid, ...], acc_s_ub_)
    
    # 2. Softmax 计算
    T.reduce_max(acc_s_ub, m_i, dim=-1)
    T.tile.max(m_i, m_i, m_i_prev)
    T.tile.sub(m_i_prev, m_i_prev, m_i)
    T.tile.exp(m_i_prev, m_i_prev)
    T.tile.sub(acc_s_ub, acc_s_ub, m_i)
    T.tile.exp(acc_s_ub, acc_s_ub)
    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
    T.tile.mul(sumexp, sumexp, m_i_prev)
    T.tile.add(sumexp, sumexp, sumexp_i_ub)
    
    # 3. 等待 Attention*V 结果并累加
    T.wait_cross_flag(2)
    T.copy(workspace_3[cid, ...], acc_o_ub)
    T.tile.add(acc_o, acc_o, acc_o_ub)
```

**问题**：
- 每次都要进行完整的 Softmax 计算
- 无法批量处理，同步开销大

#### Expert 优化

```python
# 批量 Softmax：一次处理 num_stages 个分块
r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

for k in T.serial(num_outer):
    # Softmax 批次：计算 num_stages 个分块的 r_factors 和 sumexp_is
    for i in T.serial(batch_iters):
        T.wait_flag("V", "MTE2", SIG_IO_UB)
        if i % cross_interval == 0:
            T.wait_cross_flag(SEM_WS1_C2V)  # 批量同步
        T.copy(workspace_1[cid, i, ...], io_buf)
        
        T.copy(io_buf, work_ub)
        T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
        T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
        T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
        
        # 计算 r_factors[i] = exp(m_prv - m_cur)
        T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
        
        # 计算 sumexp_is[i] = rowsum(exp(S - m_cur))
        T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
        
        # 存储 softmax 后的注意力分数
        T.copy(work_ub, acc_s_half)
        T.copy(acc_s_half, workspace_2[cid, i, ...])
        if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
            T.set_cross_flag("MTE3", SEM_WS2_V2C)
    
    # O 累加批次：批量累加 num_stages 个分块
    for i in T.serial(batch_iters):
        # 计算 exp(r_factors[i])
        T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
        
        # 修正 sumexp
        T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
        T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
        
        # 修正 acc_o
        T.tile.broadcast(buf_2d, r_factors[i, :, :])
        T.tile.mul(acc_o, acc_o, buf_2d)
        
        # 等待 Attention*V 结果并累加
        if i % cross_interval == 0:
            T.wait_cross_flag(SEM_WS3_C2V)
        T.copy(workspace_3[cid, i, ...], io_buf)
        T.copy(io_buf, work_ub)
        T.tile.add(acc_o, acc_o, work_ub)
```

**收益**：
- 批量计算减少同步次数
- 预计算 `r_factors` 和 `sumexp_is`，避免重复计算
- 使用 `neg_sm` 双缓冲存储最大值历史
- 典型性能提升：20-40%

**关键变量**：
- `r_factors[i]`：第 i 个分块的修正因子 `exp(m_prev - m_cur)`
- `sumexp_is[i]`：第 i 个分块的 `rowsum(exp(S - m_cur))`
- `neg_sm[cur/prv]`：当前/上一轮的最大值（双缓冲）

---

### 7. Workspace 结构优化

#### 基础实现

```python
# Workspace 按 block_num 分配
workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype)
workspace_2: T.Tensor([block_num, block_M, block_N], dtype)
workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype)
```

**问题**：
- `block_num` 可能很大，内存占用高
- 不支持多缓冲流水线

#### Expert 优化

```python
# Workspace 按 NUM_CORES 和 num_stages 分配
workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype)
workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype)
workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype)

# 访问方式
T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])  # cid: 核 ID，i: 分块索引
```

**收益**：
- 内存占用：`NUM_CORES * num_stages` vs `block_num`
- 支持多缓冲流水线（num_stages 批次）
- 每个核独立管理自己的 workspace
- 典型内存节省：50-80%

---

### 8. 内存复用优化

#### 基础实现

```python
# 独立缓冲，不复用
acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
```

#### Expert 优化

```python
# 复用缓冲，减少内存占用
io_buf = T.alloc_ub([block_M // 2, block_N], dtype)      # 输入/输出缓冲
acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)  # float16 注意力分数
work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # 工作缓冲
buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)   # 广播缓冲

# 使用方式
T.copy(workspace_1[cid, i, ...], io_buf)  # 从 workspace 搬运
T.copy(io_buf, work_ub)                   # 类型转换/计算
T.tile.mul(work_ub, work_ub, sm_scale)    # 计算
T.copy(work_ub, acc_s_half)               # 结果存储
```

**收益**：
- 减少内存占用
- 简化数据流管理
- 典型内存节省：30-50%

---

### 10. Pass 配置优化

#### 基础实现

```python
# 默认 Pass 配置
@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6])
```

#### Expert 优化

```python
# 关闭所有自动化 Pass，完全手动控制
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6], pass_configs=pass_configs)
```

**收益**：
- 完全手动控制流水线和同步
- 避免编译器优化干扰手动优化
- 适用于 Expert 模式的高性能 kernel

---

## 性能对比总结

| 维度 | 基础实现 | Expert 优化 | 提升 |
|------|---------|-----------|------|
| **流水线效率** | 单缓冲，串行处理 | 多缓冲流水线（num_stages=14） | ↑ 50-100% |
| **同步开销** | 每次 K/V 分块同步 | 批量同步（cross_interval=2） | ↓ 70-80% |
| **计算指令** | `gemm_v0` 高层抽象 | `mma` intrinsic + L0 双缓冲 | ↑ 20-30% |
| **数据布局** | 默认布局 | ZN/NZ layout 优化 | ↑ 10-20% |
| **任务分配** | 简单映射 | 静态任务分配 + NUM_CORES | ↑ 10-30% |
| **Softmax 效率** | 逐块计算 | 批量 Softmax | ↑ 20-40% |
| **内存占用** | block_num workspace | NUM_CORES*num_stages workspace | ↓ 50-80% |

---

## 流水线可视化

### 基础实现（单缓冲串行）

```
时间轴（每个 K/V 分块）：
─────────────────────────────────────────────────────
Cube 核: [搬运 K]──等待──[QK^T]──等待──[搬运 V]──等待──[Attn*V]
Vector 核:          [等待]──[Softmax]──[等待]──[累加]
```

### Expert 优化（多缓冲流水线）

```
时间轴（num_stages=14 个 K/V 分块）：
─────────────────────────────────────────────────────
Cube 核:
  GEMM1 批次: [搬运 K0][搬运 K1][搬运 K2]...[搬运 K13]
               ↓      ↓      ↓           ↓
             [mma0] [mma1] [mma2]...  [mma13]
               ↓      ↓      ↓           ↓
             [ws1_0][ws1_1][ws1_2]...[ws1_13]
                                             ↓
                                      批量同步（cross_interval）

  GEMM2 批次: [搬运 V0][搬运 V1][搬运 V2]...[搬运 V13]
               ↓      ↓      ↓           ↓
             [mma0] [mma1] [mma2]...  [mma13]
               ↓      ↓      ↓           ↓
             [ws3_0][ws3_1][ws3_2]...[ws3_13]

Vector 核:
  Softmax 批次: [softmax0][softmax1][softmax2]...[softmax13]
                 ↓        ↓        ↓            ↓
               [ws2_0]  [ws2_1]  [ws2_2]...   [ws2_13]
                                                 ↓
                                          批量同步（cross_interval）

  O 累加批次: [修正0][累加0][修正1][累加1]...[修正13][累加13]
```

---

## 最佳实践建议

### ✅ 推荐做法

1. **使用多缓冲流水线**
   - 参数：`num_stages=8-16`，`cross_interval=2-4`
   - 批量处理 K/V 分块，减少同步开销
2. **实现细粒度 Flag 同步**
   - Intra-core Flag：控制 MTE2 → MTE1 → M → FIX 流水线
   - Cross-core Semaphore：Cube ↔ Vector 批量同步
   - 初始化 Flag：模拟消费者已释放，避免初始等待
3. **使用 MMA Intrinsic**
   - `T.mma` 替代 `T.gemm_v0`
   - L0 双缓冲：`l0a/l0b/l0c` 各分配 2 个缓冲
   - Flag 双缓冲基址：`SIG_L0AB` 和 `SIG_L0C` 使用连续编号
4. **优化数据布局**
   - `make_zn_layout`：适配矩阵乘输入（Q, P, V）
   - `make_nz_layout`：适配 transpose 操作（K）
   - 使用 `T.annotate_layout` 标注布局
5. **静态任务分配**
   - 固定物理核数（NUM_CORES=24）
   - 均匀分配任务，避免核调度开销
   - 循环多任务，提高核利用率
6. **批量 Softmax 计算**
   - 预计算 `r_factors` 和 `sumexp_is`
   - 使用双缓冲存储最大值历史（`neg_sm[cur/prv]`）
   - 批量累加，减少同步开销
7. **优化 Workspace 结构**
   - 按 `NUM_CORES * num_stages` 分配
   - 支持多缓冲流水线
   - 减少内存占用
8. **内存复用**
   - 使用 `io_buf`、`work_ub`、`buf_2d` 复用缓冲
   - 减少内存占用，简化数据流

### ❌ 避免做法

1. **避免单缓冲**
   - 单缓冲无法隐藏搬运延迟
   - 数据搬运和计算串行执行

2. **避免频繁同步**
   - 每次 K/V 分块同步开销大
   - 应使用批量同步（cross_interval）

3. **避免高层抽象指令**
   - `gemm_v0` 无法精细控制 L0 数据流
   - 应使用 `mma` intrinsic + L0 双缓冲

4. **避免默认数据布局**
   - 默认布局可能不是最优
   - 应使用 ZN/NZ layout 优化

5. **避免简单任务映射**
   - 简单映射可能导致核调度开销
   - 应使用静态任务分配

---

## 参数配置指南

### 流水线参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `num_stages` | 8-16 | 流水线深度，一次处理的 K/V 分块数 |
| `cross_interval` | 2-4 | 跨核同步间隔，每 cross_interval 次同步一次 |
| `num_outer` | `ceil(seq_len/block_N / num_stages)` | 外层循环次数 |

### 分块参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `block_M` | 128 | Q 的分块大小（行数） |
| `block_N` | 128 | K/V 的分块大小 |
| `dim` | 128 | 隐藏维度（需要匹配） |

### 核数配置

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `NUM_CORES` | 24 | 910B 物理核数 |

---

## 适用场景

本优化方案适用于以下算子：

- ✅ Flash Attention（Self-Attention）
- ✅ Multi-Query Attention（MQA）
- ✅ Grouped-Query Attention（GQA）
- ✅ Cross-Attention
- ✅ 需要 Cube + Vector 双核协同的算子

**核心思想**：通过多缓冲流水线、细粒度同步、批量 Softmax 计算，最大化 Cube + Vector 双核的并行效率，隐藏内存访问延迟。

---

## Flag 与 Semaphore ID 分配指南

### Intra-core Flag（Cube 核内部）

| ID | 名称 | 说明 |
|----|------|------|
| 0 | SIG_K_L1 | K 搬运到 L1 的信号 |
| 1 | SIG_P_L1 | P 搬运到 L1 的信号 |
| 2 | SIG_V_L1 | V 搬运到 L1 的信号 |
| 3-4 | SIG_L0AB | L0A/L0B 双缓冲（side=0/1） |
| 5-6 | SIG_L0C | L0C 双缓冲（side=0/1） |

### Intra-core Flag（Vector 核内部）

| ID | 名称 | 说明 |
|----|------|------|
| 0 | SIG_IO_UB | IO 缓冲信号 |
| 1 | SIG_S_HALF | Softmax 结果信号 |

### Cross-core Semaphore（Cube ↔ Vector）

| ID | 名称 | 方向 | 说明 |
|----|------|------|------|
| 0 | SEM_WS1_C2V | Cube → Vector | workspace_1 (QK^T) 就绪 |
| 1 | SEM_WS1_V2C | Vector → Cube | workspace_1 被消费 |
| 2 | SEM_WS2_V2C | Vector → Cube | workspace_2 (softmax) 就绪 |
| 3 | SEM_WS2_C2V | Cube → Vector | workspace_2 被消费 |
| 4 | SEM_WS3_C2V | Cube → Vector | workspace_3 (Attention*V) 就绪 |
| 5 | SEM_WS3_V2C | Vector → Cube | workspace_3 被消费 |

---

## 参考资料

- 基础实现：`examples/flash_attention/flash_attn_bhsd.py`
- Expert 优化：`examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`
- API 参考：`.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md`
- 流水线同步：`.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md`
- GEMM 优化：`examples/gemm/gemm_intrinsic_optimize.md`

---

## 附录：Flash Attention 算法详解

### 标准 Attention（O(N²) 内存）

```python
S = QK^T / √d              # [block_M, seq_len]
P = softmax(S)             # [block_M, seq_len]
O = PV                      # [block_M, dim]
```

### Flash Attention（O(N) 内存）

```python
# 分块计算
for each block of Q:
    O = 0, sumexp = 0, m = -∞
    
    for each block of K, V:
        # 1. 计算 QK^T 分块
        S_block = Q_block @ K_block^T / √d
        
        # 2. 在线 Softmax
        m_new = max(m_old, rowmax(S_block))
        
        # 修正因子
        r_factor = exp(m_old - m_new)
        P_block = exp(S_block - m_new)
        sumexp_new = sumexp_old * r_factor + rowsum(P_block)
        
        # 3. 累加输出
        O_new = O_old * r_factor + P_block @ V_block
    
    # 4. 最终归一化
    O = O_new / sumexp_new
```

### Flash Attention 的数学推导

关键公式：`softmax([x1, x2]) = [exp(x1-m)/Σ, exp(x2-m)/Σ]`，其中 `m=max(x1, x2)`

对于分块计算：

```
设 m_old 为之前的最大值，m_new 为当前最大值

softmax_old = exp(x_old - m_old) / Σ_old
softmax_new = exp(x_new - m_new) / Σ_new

合并后的 softmax：
  m_combined = max(m_old, m_new)
  
  exp(x_old - m_combined) = exp(x_old - m_old) * exp(m_old - m_combined)
                           = softmax_old * Σ_old * exp(m_old - m_combined)
  
  Σ_combined = Σ_old * exp(m_old - m_combined) + Σ_new
  
  因此：
    r_factor = exp(m_old - m_combined)
    O_combined = O_old * r_factor + P_new @ V_block
```

---

## 总结

Expert 优化版本的 Flash Attention 实现展示了 TileLang-Ascend 的高级优化技术：

1. **多缓冲流水线**：num_stages=14，批量处理 K/V 分块
2. **细粒度同步**：Intra-core Flag + Cross-core Semaphore
3. **MMA Intrinsic**：L0 双缓冲 + mma 指令
4. **数据布局优化**：ZN/NZ layout 适配矩阵乘
5. **静态任务分配**：NUM_CORES 固定核数 + 均匀分配
6. **批量 Softmax**：预计算 r_factors 和 sumexp_is
7. **Workspace 优化**：NUM_CORES*num_stages 结构
8. **内存复用**：io_buf, work_ub, buf_2d
10. **Pass 配置**：关闭自动化 Pass，完全手动控制

这些技术组合使用，可实现接近硬件峰值性能的 Flash Attention 实现，适用于 LLM 推理优化。