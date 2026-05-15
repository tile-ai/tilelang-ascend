# GEMM 算子性能优化最佳实践

本文档总结了 GEMM（矩阵乘法）算子在 TileLang-Ascend 上的高级性能优化手段，对比基础实现与 intrinsic 优化版本的关键差异。

---

## 优化概览

| 优化项 | 基础实现 (`example_gemm.py`) | Intrinsic 优化 (`example_gemm_intrinsic.py`) | 性能收益 |
|--------|------------------------------|---------------------------------------------|---------|
| 数据搬运同步 | `T.barrier_all()` 阻塞所有流水 | `set_flag/wait_flag` 细粒度同步 | 流水线重叠 |
| 缓冲策略 | 单缓冲 | 多缓冲（S1=2, S2=2） | 隐藏搬运延迟 |
| 计算指令 | `T.gemm_v0` | `T.mma` intrinsic | 更贴近硬件 |
| L0 分块 | 无 L0 分块 | L0A/L0B + block_K | 提高计算密度 |
| 负载均衡 | 简单映射 | `T.use_swizzle` 优化 | 减少热点块 |
| 任务调度 | 单任务/核 | 循环多任务/核 | 提高核利用率 |
| 分块参数 | block_K = K_L1 | block_K = 64 | 更细粒度控制 |

---

## 核心优化技术详解

### 1. 多缓冲流水线（Double/Multi Buffering）

#### 基础实现

```python
# 单缓冲：数据搬运与计算串行执行
for k in T.serial(loop_k):
    T.copy(A[bx * block_M, k * K_L1], A_L1)  # 搬运
    T.copy(B[k * K_L1, by * block_N], B_L1)  # 搬运
    T.barrier_all()                          # 等待搬运完成
    T.gemm_v0(A_L1, B_L1, C_L0, ...)         # 计算
    T.barrier_all()                          # 等待计算完成
```

**问题**：数据搬运和计算无法重叠，每次都要等待前一步完成。

#### Intrinsic 优化

```python
# 多缓冲：L1 和 L0 均使用双缓冲
A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)  # S1=2，双缓冲
B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)  # S1=2，双缓冲
A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)  # S2=2，双缓冲
B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)  # S2=2，双缓冲

# 流水线策略：当前计算使用缓冲 0，下一轮数据搬运到缓冲 1
for k in T.serial(loop_k):
    if k < loop_k - 1:
        # 预取下一轮数据到另一个缓冲
        T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
        T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
    
    # 使用当前缓冲进行计算
    T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, ...)
```

**收益**：
- 数据搬运和计算并行执行
- 隐藏内存访问延迟
- 典型性能提升：20-40%

**关键概念**：
- `S1`：L1 缓冲深度（通常 2-4）
- `S2`：L0 缓冲深度（通常 2）
- 通过 `% S1`、`% S2` 实现缓冲轮转

---

### 2. 细粒度流水线同步（Flag-based Synchronization）

#### 基础实现

```python
# 流水同步：所有流水都要等待，开销大
T.barrier_all()  # 所有流水等待搬运完成
T.gemm_v0(...)
T.barrier_all()  # 所有流水等待计算完成
```

**问题**：
- `barrier_all` 是所有流水同步，开销大
- 无法精确控制数据依赖关系

#### Intrinsic 优化

```python
# Flag 机制：细粒度流水线同步
# 数据流：GM → MTE2 → L1 → MTE1 → L0 → M → L0C → FIX → GM
@T.macro
def init_flag():
    T.set_flag("mte1", "mte2", 0)  # MTE1 通知 MTE2：L1 缓冲 0 可用（供搬运）
    T.set_flag("mte1", "mte2", 1)  # MTE1 通知 MTE2：L1 缓冲 1 可用（供搬运）
    T.set_flag("m", "mte1", 0)     # M 通知 MTE1：L0 缓冲 0 可用（供搬运）
    T.set_flag("m", "mte1", 1)     # M 通知 MTE1：L0 缓冲 1 可用（供搬运）
    T.set_flag("fix", "m", 0)      # FIX 通知 M：初始状态就绪

@T.macro
def clear_flag():
    # 等待所有 Flag 完成，确保流水线正确结束
    T.wait_flag("mte1", "mte2", 0)  # 等待 MTE1 通知 MTE2 的 flag 0
    T.wait_flag("mte1", "mte2", 1)  # 等待 MTE1 通知 MTE2 的 flag 1
    T.wait_flag("m", "mte1", 0)     # 等待 M 通知 MTE1 的 flag 0
    T.wait_flag("m", "mte1", 1)     # 等待 M 通知 MTE1 的 flag 1
    T.wait_flag("fix", "m", 0)      # 等待 FIX 通知 M 的 flag 0
```

**流水线阶段说明**：

| 阶段 | 功能单元 | 操作 | 说明 |
|------|---------|------|------|
| **MTE2** | 内存搬运引擎 2 | GM → L1 | 数据从全局内存搬运到 L1 |
| **MTE1** | 内存搬运引擎 1 | L1 → L0 | 数据从 L1 搬运到 L0A/L0B |
| **M** | 矩阵计算单元 | MMA | 执行矩阵乘累加 |
| **FIX** | 固定功能单元 | L0C → GM | 结果写回全局内存 |

**流水线数据流**：

```
GM → MTE2 → L1 → MTE1 → L0A/L0B → M → L0C → FIX → GM
```

**Flag 双向通信机制**：

Flag 支持双向通信，用于协调缓冲的生产者和消费者：

```
数据就绪通知（生产者 → 消费者）：
- MTE2 完成 GM → L1 搬运后，设置 "mte2→mte1" flag，通知 MTE1 数据就绪
- MTE1 完成 L1 → L0 搬运后，设置 "mte1→m" flag，通知 M 数据就绪

缓冲可用通知（消费者 → 生产者）：
- MTE1 完成 L1 缓冲使用后，设置 "mte1→mte2" flag，通知 MTE2 缓冲可重用
- M 完成 L0 缓冲使用后，设置 "m→mte1" flag，通知 MTE1 缓冲可重用
```

**Flag 同步模式示例**：

```python
# 1. MTE2 搬运数据到 L1（生产者）
T.wait_flag("mte1", "mte2", k % S1)      # MTE2 等待 L1 缓冲可用
T.copy(A[...], A_L1[k % S1, :, :])      # MTE2: GM → L1 搬运
T.set_flag("mte2", "mte1", k % S1)       # MTE2 通知 MTE1 数据已就绪

# 2. MTE1 搬运数据到 L0（消费者 + 生产者）
T.wait_flag("mte2", "mte1", k % S1)      # MTE1 等待 L1 数据就绪
T.copy(A_L1[k % S1, ...], A_L0[kk % S2]) # MTE1: L1 → L0 搬运
T.set_flag("mte1", "mte2", k % S1)       # MTE1 通知 MTE2 L1 缓冲可重用
T.set_flag("mte1", "m", kk % S2)         # MTE1 通知 M 数据已就绪

# 3. M 执行矩阵乘（消费者）
T.wait_flag("mte1", "m", kk % S2)        # M 等待 L0 数据就绪
T.mma(A_L0[...], B_L0[...], C_L0)        # M: 执行矩阵乘
T.set_flag("m", "mte1", kk % S2)         # M 通知 MTE1 L0 缓冲可重用
```

**收益**：

- 精确控制数据依赖，避免不必要的等待
- 实现真正的流水线并行
- 典型性能提升：30-50%

---

### 3. L0 分块与 MMA 指令

#### 基础实现

```python
# 直接从 L1 进行矩阵乘
T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
```

**问题**：
- `gemm_v0` 是高层抽象，可能无法充分利用硬件
- L1 → L0C 的数据流不够精细

#### Intrinsic 优化

```python
# L1 进一步分块到 L0A/L0B
A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)  # L0A 寄存器
B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)  # L0B 寄存器

# L1 分块循环
loop_kk = T.ceildiv(K_L1, block_K)
for kk in T.serial(loop_kk):
    T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
    T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
    
    # 使用 MMA intrinsic 进行矩阵乘累加
    T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, 
          init=T.And(k == 0, kk == 0))
```

**收益**：
- `T.mma` 更贴近 Ascend NPU 硬件的 MMA 指令
- L0 分块提高计算密度和寄存器利用率
- 典型性能提升：10-20%

**关键 API**：
- `T.alloc_L0A`：分配矩阵乘输入 A 的寄存器
- `T.alloc_L0B`：分配矩阵乘输入 B 的寄存器
- `T.mma`：矩阵乘累加 intrinsic 指令

---

### 4. 负载均衡优化（Swizzle）

#### 基础实现

```python
# 简单的一维到二维映射
bx = cid // n_num
by = cid % n_num
```

**问题**：
- 简单映射可能导致某些核负载不均衡
- 内存访问热点可能导致带宽瓶颈

#### Intrinsic 优化

```python
# 使用 swizzle 优化负载均衡
cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
bx = cid // n_num
by = cid % n_num
```

**收益**：
- 优化内存访问模式，减少热点块
- 提高核间负载均衡
- 典型性能提升：5-15%

**关键 API**：
- `T.use_swizzle(idx, M, N, K, block_M, block_N, off=3)`
- `off` 参数控制 swizzle 偏移量

---

### 5. 多任务循环处理

#### 基础实现

```python
# 每个核处理一个输出块
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
    # 处理一个 [block_M, block_N] 输出块
```

**问题**：
- 当 `m_num * n_num` 大于物理核数时，部分核空闲
- 核利用率低

#### Intrinsic 优化

```python
# 固定物理核数，每个核循环处理多个输出块
core_num = 20  # 物理核数（可配置）
with T.Kernel(core_num, is_npu=True) as (cid, _):
    # 循环处理多个输出块
    for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
        cid = T.use_swizzle(i * core_num + cid, ...)
        if cid < m_num * n_num:
            bx = cid // n_num
            by = cid % n_num
            # 处理输出块 [bx * block_M : (bx+1) * block_M, 
            #            by * block_N : (by+1) * block_N]
```

**收益**：
- 提高核利用率，避免核空闲
- 更灵活的任务调度
- 典型性能提升：10-30%（当 `m_num * n_num` 远大于核数时）

---

### 6. 分块参数优化

#### 基础实现

```python
# 分块参数
block_M = 128
block_N = 256
K_L1 = 64  # K 维度的 L1 分块大小，同时也是 gemm_v0 的 block_K
```

**问题**：
- `K_L1` 过小可能导致计算密度不足
- 无法精细控制 L0 分块

#### Intrinsic 优化

```python
# 更精细的分块参数
block_M = 128     # 输出分块 M 维度
block_N = 256     # 输出分块 N 维度
block_K = 64      # L0 分块 K 维度（细粒度）
K_L1 = 256        # L1 分块 K 维度（粗粒度）
S1 = 2            # L1 双缓冲深度
S2 = 2            # L0 双缓冲深度

# 分块层级
GM (全局内存) → L1 [block_M, K_L1] / [K_L1, block_N] 
             → L0 [block_M, block_K] / [block_K, block_N]
```

**收益**：
- 更大的 `K_L1` 提高数据搬运效率
- 更细的 `block_K` 提高计算密度和寄存器利用率
- 双缓冲参数灵活配置

---

## 性能对比总结

| 维度 | 基础实现 | Intrinsic 优化 | 提升 |
|------|---------|---------------|------|
| **流水线效率** | 数据搬运与计算串行 | 多缓冲 + 流水线重叠 | ↑ 50-80% |
| **同步开销** | `barrier_all` 所有流水同步 | Flag 细粒度同步 | ↓ 30-50% |
| **计算指令** | `gemm_v0` 高层抽象 | `mma` intrinsic | ↑ 10-20% |
| **分块粒度** | 粗粒度（仅 L1 分块） | 细粒度（L1 + L0 分块） | ↑ 计算密度 |
| **核利用率** | 单任务/核 | 循环多任务/核 | ↑ 10-30% |
| **负载均衡** | 简单映射 | Swizzle 优化 | ↑ 5-15% |
| **内存带宽** | 单缓冲，串行访问 | 多缓冲，流水线访问 | ↑ 带宽利用率 |

---

## 流水线可视化

### 基础实现（串行）

```
时间轴：
─────────────────────────────────────────────────────
核 0: [搬运 A/B] ──等待── [计算] ──等待── [写回]
核 1: [搬运 A/B] ──等待── [计算] ──等待── [写回]
核 2: [搬运 A/B] ──等待── [计算] ──等待── [写回]
...
```

### Intrinsic 优化（流水线并行）

```
时间轴：
─────────────────────────────────────────────────────
核 0: [搬运 k=0] [搬运 k=1] [搬运 k=2] ...
          ↓       ↓       ↓
      [计算 k=0] [计算 k=1] [计算 k=2] ...
                        ↓
                    [写回]

流水线阶段：
MTE2: ────搬运────搬运────搬运────
MTE1:      ───搬运────搬运────搬运
M:            ──计算──计算──计算──
FIX:              ──写回──写回──
```

---

## 最佳实践建议

### ✅ 推荐做法

1. **使用多缓冲策略**
   - L1 和 L0 均采用双缓冲或三缓冲
   - 参数建议：`S1=2-4`，`S2=2`

2. **实现细粒度流水线同步**
   - 使用 `set_flag/wait_flag` 替代 `barrier_all`
   - 精确控制 MTE2 → MTE1 → M → FIX 的数据流

3. **使用 intrinsic 指令**
   - `T.mma` 替代 `T.gemm_v0`
   - 更贴近硬件，性能更优

4. **精细分块策略**
   - L1 粗粒度分块：`K_L1=128-512`
   - L0 细粒度分块：`block_K=32-128`
   - 平衡计算密度和内存带宽

5. **负载均衡优化**
   - 使用 `T.use_swizzle` 优化任务分配
   - 减少内存热点和带宽瓶颈

6. **多任务循环处理**
   - 每个核循环处理多个输出块
   - 提高核利用率

### ❌ 避免做法

1. **避免单缓冲**
   - 单缓冲无法隐藏搬运延迟
   - 数据搬运和计算串行执行

2. **避免过度同步**
   - `barrier_all` 开销大，影响流水线效率
   - 应使用细粒度 Flag 同步

3. **避免粗粒度分块**
   - 过大的分块导致寄存器利用率低
   - 过小的分块导致计算密度不足

4. **避免简单负载映射**
   - 简单映射可能导致热点块和带宽瓶颈
   - 应使用 swizzle 优化

---

## 适用场景

本优化方案适用于以下算子：

- ✅ GEMM（矩阵乘法）
- ✅ GEMV（矩阵向量乘）
- ✅ Convolution（卷积，可转化为 GEMM）
- ✅ Batch Matmul（批量矩阵乘）
- ✅ 需要细粒度流水线的高性能算子

**核心思想**：通过多缓冲、细粒度同步、intrinsic 指令实现流水线并行，隐藏内存访问延迟，最大化硬件利用率。

---

## 参数配置指南

### 分块参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `block_M` | 128-256 | 输出分块 M 维度，适配 L0C 寄存器容量 |
| `block_N` | 128-256 | 输出分块 N 维度，适配 L0C 寄存器容量 |
| `block_K` | 32-128 | L0 分块 K 维度，适配 L0A/L0B 寄存器容量 |
| `K_L1` | 128-512 | L1 分块 K 维度，适配 L1 缓存容量 |

### 缓冲深度

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `S1` | 2-4 | L1 双缓冲或三缓冲深度 |
| `S2` | 2 | L0 双缓冲深度（通常固定为 2） |

### 核数配置

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `core_num` | 20-40 | 物理核数，根据硬件型号配置 |

## 参考资料

- 基础实现：`examples/gemm/example_gemm.py`
- Intrinsic 优化：`examples/gemm/example_gemm_intrinsic.py`
- API 参考：`.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md`
- 流水线同步：`.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-schedule-sync.md`

---

## 附录：Flag 机制详解

### Flag 操作

| API | 功能 | 使用场景 |
|-----|------|---------|
| `T.set_flag(src, dst, idx)` | 设置 Flag | 生产者通知消费者数据已就绪 |
| `T.wait_flag(src, dst, idx)` | 等待 Flag | 消费者等待数据就绪 |

### Flag 状态机

```
初始状态：Flag = 0

生产者：
  set_flag(src, dst, idx) → Flag = 1

消费者：
  wait_flag(src, dst, idx) → 等待 Flag = 1
  执行操作
  （消费者完成后可重新设置 Flag）
```

### Flag 使用模式

```python
# 模式 1：单缓冲同步
T.copy(A, A_L1)          # MTE2: 生产者（GM → L1）
T.set_flag("mte2", "mte1", 0)  # 通知 MTE1

T.wait_flag("mte2", "mte1", 0) # MTE1: 消费者等待
T.copy(A_L1, A_L0)       # MTE1: 使用数据（L1 → L0）

# 模式 2：多缓冲轮转
for k in T.serial(loop_k):
    # 使用 flag k % S1
    T.set_flag("mte2", "mte1", k % S1)
    T.wait_flag("mte2", "mte1", k % S1)
```

---

## 总结

Intrinsic 优化版本的 GEMM 实现展示了 TileLang-Ascend 的高级优化技术：

1. **多缓冲流水线**：隐藏内存访问延迟
2. **细粒度同步**：精确控制数据依赖
3. **Intrinsic 指令**：贴近硬件优化
4. **精细分块**：提高计算密度
5. **负载均衡**：优化内存访问模式
6. **多任务处理**：提高核利用率

这些技术组合使用，可实现接近硬件峰值性能的 GEMM 实现。