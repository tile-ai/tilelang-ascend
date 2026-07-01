# Cube 核算子优化路径

- 原则1：首先是模式选择，不同模式对应不同的优化路径，要选择对应的优化路径进行参考。
- 原则2：专门的调优实验要比总结的规律重要，总结的规律只有当按照实验做完没有收益时再考虑。

## 1. 模式识别与路径选择

### 两种编程模式

| 模式 | 计算指令 | 标识特征 | 适用场景 |
|------|---------|---------|---------|
| **Expert 模式** | `T.mma` | 显式分配 L0A/L0B/L1 + 手动 flag 同步 + `auto_sync=False` | 追求极致性能 |
| **Developer 模式** | `T.gemm_v0` | L1 分配 + `auto_sync=True` + 无 L0 层显式控制 | 快速开发、已有实现适配 |

### 识别方法

检查 kernel 代码中的关键特征：

```python
# Expert 模式标识：
T.alloc_L0A(...)          # 显式分配 L0A
T.alloc_L0B(...)          # 显式分配 L0B
T.mma(A_L0, B_L0, C_L0)   # 使用 T.mma intrinsic
T.set_flag(...)            # 手动 flag 同步
T.wait_flag(...)           # 手动 flag 同步

# Developer 模式标识：
T.gemm_v0(A_L1, B_L1, C_L0)  # 使用 T.gemm_v0 高层抽象
# 无 L0A/L0B 显式分配
# 无手动 flag 同步（依赖 auto_sync）
```

### 路径选择

```
识别到 T.mma → Expert 路线（§2）
识别到 T.gemm_v0 → Developer 路线（§3）
其他 cube 算子（FA、SparseFA 等）同理识别
```

---

## 2. Expert 模式优化路径（T.mma）

### 2.1 优化路径（4 步）

```
Step 1: 最简 T.mma（单缓冲）
  → 选当前最优 tiling：block_K=K_L1=128（打满 L0，详见 §5 硬件约束速查 / §6 Tiling 参数说明）
  → 性能：0.64x

Step 2: +整体双缓冲 + 三级K分块（不可分割）
  → 重新评估 tiling：block_K=128 双缓冲时 L0B 溢出
  → 调整为 block_K=64, K_L1=256（= block_K × 4）
  → 性能：0.95x（+48.8%）
  
  变体：预取双缓冲（L0 层预取重叠）❌ 不推荐
  → 在整体双缓冲基础上，循环外预取 kk=0，循环内预取 kk+1
  → copy[kk+1] 和 mma[kk] 重叠执行
  → 无 FC：0.93x（比整体双缓冲还差，flag 同步开销暴露）
  → 有 FC：1.01x（与整体双缓冲+FC 持平，无净收益）
  → 结论：预取重叠收益被 flag 同步开销抵消，不推荐采用

Step 3: +Fixed Core
  → tiling 不变，只改 launch 方式（按物理核数 launch，循环处理 tile）
  → 性能：1.01x（+5.7%）
  → 注意：小矩阵（tile 数 ≤ 20）可能劣化

Step 4: +Swizzle（实验性，劣化不采用）
  → 加 T.use_swizzle 重映射 tile 到 core 的分配顺序
  → 性能：0.99x（-1.8%）
  → 结论：整体劣化，不默认采用，但后续算子仍需实验验证
```

### 2.2 双缓冲策略

#### 双缓冲与三级K分块的耦合关系

**双缓冲和三级K分块是不可分割的优化单元，必须同时实现：**

- **没有双缓冲，三级K是纯串行**：K_L1 切成 block_K 的小块，每次 L1→L0 搬运和 MMA 计算串行执行，多切只会多搬运，没有重叠。实测：单独加三级K（单缓冲）性能反而下降 19.6%。
- **没有三级K，双缓冲的 L0 ping-pong 不生效**：block_K=K_L1 时，kL0split=1，buffer 1 从未使用，白白付出双缓冲同步开销。
- **两者同时存在才有意义**：三级K 提供 L1→L0 的内层循环（loop_kk ≥ 2），双缓冲让内层循环的搬运和计算重叠。

#### 整体双缓冲 vs 预取双缓冲

**整体双缓冲**（gemm_intrinsic.py）：
- L0 buffer 数 S2=2，但 copy 和 MMA **串行执行**
- 每个 kk 迭代：copy[kk] → wait → mma[kk]
- 无 FC：0.95x / 有 FC：1.01x

**预取双缓冲**（example_gemm_hiperef.py）：
- L0 buffer 数 S2=2，copy 和 MMA **重叠执行**
- 循环外预取 kk=0，循环内预取 kk+1
- 每个 kk 迭代：copy[kk+1] ∥ mma[kk]

**逐步对比实验**（20 case 全量，3 次取平均）：

| 版本 | 无 FC | 有 FC | 有 FC + Swizzle |
|------|-------|-------|----------------|
| 整体双缓冲 | 0.95x | **1.01x** | 0.99x |
| 预取双缓冲 | 0.93x | 1.01x | 0.97x |

**Fixed Core 在预取双缓冲上的收益**：+9.2%（大部分 case 正向，Case 8 +38.8%）
**Swizzle 在预取双缓冲上的收益**：-4.1%（整体劣化，仅 2 case 正向）

**结论：预取双缓冲无净收益，不推荐采用。**

原因分析：
1. **纯预取双缓冲（无 FC）反而更差**：0.93x < 0.95x。预取多出的 flag 同步开销（循环外预取 kk=0 + 循环内预取 kk+1 的 wait/set flag）在按 tile 数 launch 时暴露，抵消了 copy/mma 重叠的理论收益。
2. **有 FC 时两者持平**：1.01x = 1.01x。FC 省掉的 ~200us 固定开销淹没了预取的同步开销，但预取重叠的收益仍被同步开销抵消，没有净增益。
3. **Swizzle 在预取双缓冲上劣化更严重**：-4.1% vs -1.8%。预取双缓冲的调度更复杂，Swizzle 打乱 tile 顺序对 cache 局部性的负面影响更大。

**最优方案**：整体双缓冲 + Fixed Core（无 Swizzle），即 Step 3，性能 1.01x。

**预取双缓冲代码模式**（仅供参考，不推荐使用）：
```python
# 循环外预取 kk=0 -> L0[0]
T.wait_flag("mte2", "mte1", k % S1)
T.wait_flag("m", "mte1", 0)
T.copy(A_L1[k % S1, 0, 0], A_L0[0, :, :])
T.copy(B_L1[k % S1, 0, 0], B_L0[0, :, :])
T.set_flag("mte1", "m", 0)

for kk in T.serial(loop_kk):
    # 预取 kk+1 -> L0[(kk+1)%S2]，与 mma[kk] 重叠
    if kk < loop_kk - 1:
        T.wait_flag("m", "mte1", (kk + 1) % S2)
        T.copy(A_L1[k % S1, 0, (kk + 1) * block_K], A_L0[(kk + 1) % S2, :, :])
        T.copy(B_L1[k % S1, (kk + 1) * block_K, 0], B_L0[(kk + 1) % S2, :, :])
        T.set_flag("mte1", "m", (kk + 1) % S2)
        if kk == loop_kk - 2:
            T.set_flag("mte1", "mte2", k % S1)

    # 计算 kk：数据已在上一轮预取就绪
    T.wait_flag("mte1", "m", kk % S2)
    T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=T.And(k == 0, kk == 0))
    T.set_flag("m", "mte1", kk % S2)
```

### 2.3 Tiling 推导方法

> **前提**：本节涉及的 L0C/L0A/L0B 容量约束和参数定义详见 [§5 硬件约束速查](#5-硬件约束速查ascend-910b) 和 [§6 Tiling 参数说明](#6-tiling-参数说明)。

#### Step 1: 选 block_M, block_N, block_K

对于 fp16 + fp32 累加：
```
L0C: block_M × block_N ≤ 32768  (128KB / 4B)
L0A: block_M × block_K ≤ 32768  (64KB / 2B)
L0B: block_K × block_N ≤ 32768  (64KB / 2B)
```

**目标**：最大化 L0C 利用率（block_M × block_N 接近 32768）

**单缓冲下的最优选择**：
- (128, 256, 128): L0C = 32768 (100%), L0A = 16384 (50%), L0B = 32768 (100%)

**双缓冲下的最优选择**：
- (128, 256, 64): L0C = 32768 (100%), L0A = 8192 (25%), L0B = 16384 (50%)
- 双缓冲后 L0B = 64 × 256 × 2 × 2 = 64KB (100%)，刚好用满

#### Step 2: 加双缓冲+三级K时重新评估

```
L0A: block_M × block_K × sizeof(dtype) × 2 ≤ 64KB
L0B: block_K × block_N × sizeof(dtype) × 2 ≤ 64KB
```

**调整逻辑**：单缓冲下 block_K=128 打满 L0B，双缓冲后 L0B 翻倍溢出，需将 block_K 减半到 64。

#### Step 3: 选 K_L1

K_L1 应该是 block_K 的倍数，越大搬运效率越高。

对于 (128, 256, 64)：
```
K_L1 = 256: L1 = (128×256 + 256×256) × 2 = 192KB ✅
```

**推荐**：K_L1 = 256（= block_K × 4），每次 L1 加载做 4 轮 MMA

#### 什么时候需要重新评估 tiling

| 触发条件 | 需要检查 | 可能的调整 |
|---------|---------|-----------|
| 加双缓冲+三级K（S1=2, S2=2） | L0A/L0B 是否溢出 | block_K 减半 |
| 加 L1 双缓冲（S1=2） | L1 是否溢出 | K_L1 减小 |
| 改 dtype（如 fp16→int8） | 分形限制、内存占用 | 所有参数重新推导 |
| 改 accum_dtype | L0C 占用变化 | block_M × block_N 上限变化 |

### 2.4 优化案例

#### 逐步优化过程

**Step 1: 最简 T.mma（单缓冲）**
```
block_M = 128, block_N = 256, block_K = 128, K_L1 = 128
性能：0.64x
```

**Step 2: +整体双缓冲 + 三级K分块**
```
block_K: 128 → 64, K_L1: 128 → 256
性能：0.95x（+48.8%）
```

**Step 2 变体: 预取双缓冲** ❌ 不推荐
```
性能：无 FC 0.93x / 有 FC 1.01x（与整体双缓冲持平）
```

**Step 3: +Fixed Core**
```
性能：1.01x（+5.7%）
```

**Step 4: +Swizzle** ❌ 劣化不采用
```
性能：0.99x（-1.8%）
```

#### 最终 tiling

```
block_M = 128, block_N = 256, block_K = 64, K_L1 = 256, S1 = 2, S2 = 2
```

#### 性能验证

| Step | 优化内容 | tiling 变化 | 平均加速比 |
|------|---------|------------|-----------|
| Step 1 | 最简 T.mma（单缓冲） | block_K=128, K_L1=128 | 0.64x |
| Step 2 | +整体双缓冲 + 三级K | block_K=64, K_L1=256 | 0.95x |
| Step 2 变体 | 预取双缓冲 ❌ 不推荐 | 不变 | 无 FC 0.93x / 有 FC 1.01x |
| Step 3 | +Fixed Core | 不变 | **1.01x（最优）** |
| Step 4 | +Swizzle（劣化，不采用） | 不变 | 0.99x |

### 2.5 常见陷阱（Expert 专属）

#### 陷阱 1: L0C 溢出

> L0C 容量限制详见 [§5 硬件约束速查](#5-硬件约束速查ascend-910b)。

```python
# 错误：block_M × block_N × 4 > 128KB
block_M = 256; block_N = 256  # L0C = 256KB > 128KB ❌

# 正确
block_M = 128; block_N = 256  # L0C = 128KB ✅
```

#### 陷阱 2: 加双缓冲后未重新评估 tiling

> L0B 容量限制详见 [§5 硬件约束速查](#5-硬件约束速查ascend-910b)。

```python
# 单缓冲选 block_K=128，加双缓冲后 L0B 溢出
# L0B (双缓冲) = 128×256×2×2 = 128KB > 64KB ❌

# 正确：block_K 减半到 64
# L0B (双缓冲) = 64×256×2×2 = 64KB ✅
```

#### 陷阱 3: 双缓冲与三级K分开实现

```python
# 错误：先单独加三级K（单缓冲）→ 性能下降 19.6%
# 错误：先单独加双缓冲（block_K=K_L1）→ buffer 1 从未使用

# 正确：双缓冲 + 三级K同时实现
# block_K=64, K_L1=256, S1=2, S2=2 → 性能提升 48.8%
```

#### 陷阱 4: 预取双缓冲看起来更优但实测无净收益

```python
# 整体双缓冲：copy 和 MMA 串行
for kk in T.serial(loop_kk):
    T.copy(...)
    T.set_flag("mte1", "m", kk % S2)
    T.wait_flag("mte1", "m", kk % S2)  # 阻塞等待
    T.mma(...)
    T.set_flag("m", "mte1", kk % S2)
# 性能：无 FC 0.95x / 有 FC 1.01x  ← 推荐

# 预取双缓冲：copy 和 MMA 重叠（删除了 wait_flag）
# 但多出的 flag 同步开销抵消了 copy/mma 重叠收益
# 性能：无 FC 0.93x / 有 FC 1.01x  ← 不推荐
```

#### 陷阱 5: Swizzle 不是万能的

```python
# 错误：默认使用 Swizzle → 整体劣化 -1.8%
# 正确：实验验证，劣化就不采用
# 正向 case 特征：tile 数 18~32，计算时间 < 20us
# 劣化 case 特征：tile 数 > 500 或 tile 数 < 18
```

---

## 3. Developer 模式优化路径（T.gemm_v0）

### 3.1 gemm_v0 特性与限制

| 特性 | 说明 | 影响 |
|------|------|------|
| `auto_sync=True` | 编译器自动插入 `barrier_all()` | 同步开销大于手动 flag |
| L0 层双缓冲 | 以kL0Size为单位进行切分  | 内部实现L1到L0的双缓冲优化 |
| 内嵌同步 | 内部实现双缓冲的同时也引入了内部隐式同步 | 算子外部无法实现l1层的双缓冲 |

### 3.2 可调参数

> 参数定义和容量约束详见 [§6 Tiling 参数说明](#6-tiling-参数说明) 和 [§5 硬件约束速查](#5-硬件约束速查ascend-910b)。

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|---------|
| `block_M` | M 维度 tile 大小 | 128 | 固定，受 L0C 约束 |
| `block_N` | N 维度 tile 大小 | 256 | 固定，受 L0C 约束 |
| `K_L1` | GM→L1 搬运粒度 | 128 | 需 > kL0Size 才能触发 ping-pong |
| `kL0Size` | L0 层切片大小（内部参数） | 128 | 需小于K_L1；非前端参数，可去往 `src/tl_templates/ascend/common.h` 的 `constexpr uint32_t kL0Size`进行修改|


### 3.3 优化路径以及实测数据

优化路径可参考如下调优实验

#### 三版逐步优化实验

由于gemmv0的特性与限制，developer模式的乘法算子调优空间较小，集中再tiling策略与核调度策略，优化路径为：基础实现->tiling策略调优->核调度调优，实验数据如下。

| 版本 | K_L1 | kL0Size | Launch | 平均加速比 | 说明 |
|------|------|---------|--------|-----------|------|
| 原始版 | 64 | 128 | tile 数 | 0.44x | ping-pong 失效 |
| 实验A | 128 | 64 | tile 数 | 0.51x (+15.9%) | K_L1/kL0Size 调优，ping-pong 生效 |
| **实验B** | **128** | **64** | **20核 grid-stride** | **0.52x (+2.0%)** | +Fixed Core，有效收益 |

- 实验A：既然无法实现l1阶同步，那就要充分利用l0阶同步，kL0Size一定要小于K_L1，又由于硬件限制，kL0Size改为64，K_L1是其两倍，使双缓冲ping-pong生效。
- 实验B：由于大部分case都是大shape，分核后tile数远超核数，故选择FixedCore进行优化。

**结论**：
1. kL0Size 必须小于 K_L1，否则 L0 层 ping-pong 失效
2. 而K_L1以128为最优，所以将kL0Size改为64
3. 使用FC优化榨干最后的性能收益
### 3.4 常见陷阱（Developer 专属）

#### 陷阱 1: kL0Size ≥ K_L1 导致 ping-pong 失效


```python
# gemm_v0 内部：common.h 默认 kL0Size=128；若前端 K_L1=128
# kL0split = ceil(K_L1 / kL0Size) = ceil(128/128) = 1 → ping-pong 不生效

# 正确方案 A：编辑 common.h 改 kL0Size=64，前端 K_L1=128
#            → kL0split=2，ping-pong 生效（实验A，0.51x）
# 正确方案 B：升级到 Expert 模式，用 T.mma 替代 gemm_v0
```


---

## 4. 共享优化技术

### 4.1 Fixed Core

#### 原理

按物理核数（20）launch，每个核循环处理多个 tile，L1/L0 buffer 只分配一次被复用。相比按 tile 数 launch，节省了重复的 buffer 分配/释放开销。

#### 收益规律

Fixed Core 的收益本质上是**省掉一笔固定开销（约 200us）**，这笔开销在总时间中的占比决定了收益百分比。

**控制变量实验**（tile 数=2048，K 变化，Expert 模式）：

| K | 每 tile 计算量 | Step 2 (us) | Step 3 (us) | 节省 (us) | 收益 |
|---|-------------|-------------|-------------|----------|------|
| 1024 | 33M（小） | 764.50 | 552.22 | 212.28 | **+27.8%** |
| 2048 | 67M（中） | 1254.42 | 1052.06 | 202.36 | **+16.1%** |
| 4096 | 134M（大） | 2262.42 | 2086.24 | 176.18 | **+7.8%** |
| 8192 | 268M（很大） | 4388.34 | 4276.58 | 111.76 | **+2.5%** |

#### 各模式适用性

| 模式 | Fixed Core 效果 | 说明 |
|------|----------------|------|
| Expert (T.mma) | ✅ 有效（+5.7%） | 手动 flag 同步开销小，FC 收益能体现 |
| Developer (T.gemm_v0) | ✅ 有效收益（+2.0%） | auto_sync 的 barrier_all 部分抵消，但仍有净增益 |


#### 代码规范模板

**Fixed Core 模板**：

```python
core_num = 20  # 910B 物理核数
    @T.prim_func
    def main(A, B, C):
        # 1. 固定核数core_num
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            
            # ... buffer申请 ...

            with T.Scope("C"):
                # 2. grid-stride 循环处理所有 tile
                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    # 3. 计算当前 tile ID
                    cid_task = i * core_num + cid
                    # 4. 尾块守卫（tile 数不整除 core_num 时）
                    if cid_task < m_num * n_num:
                        bx = cid_task // n_num
                        by = cid_task % n_num
                        # ... tile 计算 ...
```


**关键规范点**：
1. `T.Kernel(core_num)` 固定核数 launch，`core_num = 20`（910B 物理核数）
2. `if cid_task < m_num * n_num` 尾块守卫（tile 数不整除 core_num 时避免越界）

### 4.2 Swizzle

#### 原理

`T.use_swizzle` 重映射 tile 到 core 的分配顺序，让相邻核访问不同的 A/B 区域，减少 L2 cache 冲突。

```python
# 简单映射：相邻核访问同一块 A，L2 cache 冲突
核0 → tile(0,0)  核1 → tile(0,1)  → 都访问 A[0:128]

# Swizzle 映射（off=3）：相邻核访问不同 A 区域
核0 → tile(0,0)  核1 → tile(1,1)  → 访问 A[0:128] 和 A[128:256]
```

#### 实验结果（Expert 模式，劣化不采用）

**整体表现**：Step 4 (Fixed Core + Swizzle) 平均 0.99x，比 Step 3 (Fixed Core) 的 1.01x 还差 **-1.8%**。

**逐 case 分析**：

| Case | Shape | tiles | Step 3 (us) | Step 4 (us) | Swizzle 收益 |
|------|-------|-------|-------------|-------------|-------------|
| 1 | 1024³ | 32 | 16.18 | 13.90 | **+14.1%** |
| 5 | 1024³ bf16 | 32 | 16.08 | 13.92 | **+13.4%** |
| 12 | 1×3584×4608 | 18 | 31.30 | 28.58 | **+8.7%** |
| 4 | 8192³ | 2048 | 4202.53 | 4366.12 | **-3.9%** |
| 15 | 512×4096² | 512 | 96.76 | 114.55 | **-18.4%** |
| 17 | 1009×1024² | 32 | 13.14 | 15.30 | **-16.4%** |
| 19 | 64³ | 1 | 3.36 | 4.98 | **-48.2%** |

**劣化原因分析**：

1. **小矩阵（tile 数 ≤ 32）**：Swizzle 打乱了原本紧凑的 tile 调度，增加 cache miss
2. **大矩阵（tile 数 > 500）**：计算时间主导，Swizzle 的 cache 优化收益被淹没
3. **中等矩阵（tile 数 100~500）**：收益不稳定，依赖具体 shape

**正向 case 特征**：tile 数适中（18~32）、计算时间较短（< 20us）、L2 cache 冲突是主要瓶颈

#### 各模式适用性

| 模式 | Swizzle 效果 | 说明 |
|------|-------------|------|
| Expert (T.mma) | ❌ 劣化（-1.8%） | 整体劣化，仅 3 case 正向 |
| Developer (T.gemm_v0) | ❌ 劣化 | 与 FC 一起测试，整体劣化 |

#### 使用建议

| 条件 | 建议 | 原因 |
|------|------|------|
| 默认 | **不采用** | 整体劣化 |
| 后续算子调优 | 实验验证 | 不同算子可能有不同效果 |
| tile 数 18~32 且计算时间短 | 可尝试 | 正向 case 的特征 |
| tile 数 > 500 | 不采用 | 计算主导，cache 优化无感 |

**结论**：Swizzle 不是万能的，需要根据具体算子和 shape 实验验证。如果实验发现劣化，就不采用。

---

## 5. 硬件约束速查（Ascend 910B）

### 内存层级容量

| 层级 | 容量 | 用途 |
|------|------|------|
| L0C | 128 KB | 输出累加器（C 矩阵） |
| L0A | 64 KB | A 矩阵寄存器 |
| L0B | 64 KB | B 矩阵寄存器 |
| L1 | ~512 KB | 共享缓存（A_L1, B_L1） |

### 分形限制（最小 tile 维度）

| dtype | L0A (M×K) | L0B (K×N) | L0C (M×N) |
|-------|-----------|-----------|-----------|
| fp16/bf16 | M≥16, K≥16 | K≥16, N≥16 | M≥16, N≥16 |
| int8 | M≥16, K≥32 | K≥32, N≥16 | M≥16, N≥16 |
| fp32 | M≥16, K≥8 | K≥8, N≥16 | M≥16, N≥16 |

### 对齐要求

| 存储单元 | 对齐 |
|---------|------|
| UB / L1 | 32 Byte |
| L0A / L0B | 512 Byte |
| L0C | 64 Byte |

---

## 6. Tiling 参数说明

### L0 层参数（MMA 计算单元）

- **block_M**: 输出 tile 的 M 维度大小
- **block_N**: 输出 tile 的 N 维度大小
- **block_K**: L1→L0 一次搬运的 K 维度大小（MMA 粒度）

### L1 层参数（共享缓存）

- **K_L1**: GM→L1 一次搬运的 K 维度大小（搬运粒度）

### 参数关系

> 容量数值详见 [§5 硬件约束速查](#5-硬件约束速查ascend-910b)。

```
L0 层约束（block_M, block_N, block_K 互相制约）：
  L0C: block_M × block_N × sizeof(accum) ≤ 128KB
  L0A: block_M × block_K × sizeof(dtype) ≤ 64KB
  L0B: block_K × block_N × sizeof(dtype) ≤ 64KB

L1 层约束（K_L1 独立选择）：
  L1: (block_M × K_L1 + K_L1 × block_N) × sizeof(dtype) ≤ 512KB
  K_L1 应该是 block_K 的倍数
```

---

## 7. 通用公式

> 本节使用的 L0C/L0A/L0B 容量参数详见 [§5 硬件约束速查](#5-硬件约束速查ascend-910b)。

### 给定 dtype 和 accum_dtype，快速计算最优 tiling

```python
def compute_optimal_tiling(dtype_size, accum_size, L0C_capacity=128*1024, L0AB_capacity=64*1024,
                           use_double_buffer=False):
    """
    计算最优 tiling 参数

    Args:
        dtype_size: 输入数据类型大小（字节），如 fp16=2, int8=1
        accum_size: 累加器数据类型大小（字节），如 fp32=4
        L0C_capacity: L0C 容量（字节），默认 128KB
        L0AB_capacity: L0A/L0B 容量（字节），默认 64KB
        use_double_buffer: 是否使用双缓冲

    Returns:
        (block_M, block_N, block_K, K_L1)
    """
    # Step 1: 最大化 L0C 利用率
    block_M = 128
    block_N = 256

    # Step 2: 从 L0A/L0B 约束推导 block_K
    buffer_multiplier = 2 if use_double_buffer else 1
    max_block_K_from_L0A = L0AB_capacity // (block_M * dtype_size * buffer_multiplier)
    max_block_K_from_L0B = L0AB_capacity // (block_N * dtype_size * buffer_multiplier)
    block_K = min(max_block_K_from_L0A, max_block_K_from_L0B)

    # Step 3: 选 K_L1（block_K 的倍数）
    K_L1 = block_K * 4

    return block_M, block_N, block_K, K_L1

# 示例：fp16 + fp32 累加
# 单缓冲
bm, bn, bk, kl1 = compute_optimal_tiling(2, 4, use_double_buffer=False)
print(f"单缓冲: block_M={bm}, block_N={bn}, block_K={bk}, K_L1={kl1}")
# 输出：单缓冲: block_M=128, block_N=256, block_K=128, K_L1=512

# 双缓冲
bm, bn, bk, kl1 = compute_optimal_tiling(2, 4, use_double_buffer=True)
print(f"双缓冲: block_M={bm}, block_N={bn}, block_K={bk}, K_L1={kl1}")
# 输出：双缓冲: block_M=128, block_N=256, block_K=64, K_L1=256
```
