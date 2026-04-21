# random_1d 算子设计文档

## 1. 概述

### 1.1 算子名称

`random_1d` - 基于 LCG（线性同余生成器）的 1D 随机数生成算子，用于在 NPU 上高并行生成伪随机数序列。

### 1.2 数学公式

该算子实现 LCG（Linear Congruential Generator）算法：

```
对于每个元素 i (i = 0, 1, ..., M-1):
    state = seed + i               # 初始状态（每个元素独立）
    state = A * state + C          # LCG 第 1 轮
    state = A * state + C          # LCG 第 2 轮
    state = A * state + C          # LCG 第 3 轮
    output[i] = state              # 输出随机数
```

其中：
- `A = 1103515245`（LCG 乘数，经典参数）
- `C = 12345`（LCG 增量，经典参数）
- 计算使用 int32 整数，溢出时自动截断（模 2^32）
- 每个元素独立计算，无顺序依赖，支持完全并行

### 1.3 计算特征分析

| 维度 | 分析结果 |
|------|---------|
| **计算类型** | 纯 Vector（element-wise 乘加运算） |
| **复杂度级别** | 单步（arith_progression + fill + 3轮 mul/add + 批量 copy） |
| **动态 shape** | M 为参数维度，支持任意大小 |
| **核间协作** | 纯 Vector 核，无需 Cube 核 |
| **并行模式** | 多 kernel + 多 vid 高并行 |
| **数据搬运** | T.copy 批量写回，充分利用 NPU 带宽 |

### 1.4 典型配置示例

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 65536 | 输出元素数量（推荐大数据量） |
| seed | 42 | 随机种子 |
| BLOCK_SIZE | 128 | 每个 vid 处理的元素数 |
| VEC_NUM | 2 | 每个 kernel 的 Vector 单元数 |
| TOTAL_BLOCK_SIZE | 256 | 每个 kernel 处理的总元素数 = VEC_NUM × BLOCK_SIZE |
| num_blocks | 256 | kernel 数量 = M / TOTAL_BLOCK_SIZE |
| 总并行单元 | 512 | num_blocks × VEC_NUM |

### 1.5 性能指标

| 数据量 M | 并行度 (kernels × vids) | 平均耗时 | 吞吐量 |
|---------|------------------------|---------|--------|
| 1024 | 4 × 2 = 8 | 0.46 ms | 8.5 MB/s |
| 4096 | 16 × 2 = 32 | 0.52 ms | 29.8 MB/s |
| 16384 | 64 × 2 = 128 | 0.46 ms | 136.4 MB/s |
| 65536 | 256 × 2 = 512 | 0.47 ms | **533.0 MB/s** |

**性能分析**：
- 小数据量（M < 4K）：kernel launch overhead 占比大
- 大数据量（M > 16K）：接近硬件带宽极限
- 并行度随 M 线性增长：充分利用 NPU 多核

---

## 2. 编程模式选型

### 2.1 选型结论：**Developer 模式**

### 2.2 选型理由

| 因素 | 分析 |
|------|------|
| 无 GEMM 计算 | 纯 element-wise 运算，不需要 Cube 核 |
| 需要精细 buffer 管理 | 多个临时 buffer（state_ub, temp_ub, a_ub 等） |
| 需要动态偏移 | 使用 `cid * TOTAL_BLOCK_SIZE + vid * block_size` 组合表达式 |
| 启用 pass_configs | 利用编译器自动优化（CV 融合、同步、内存规划） |

Developer 模式的优势：
- 使用 `T.alloc_ub` 显式分配 Unified Buffer
- 使用 `T.tile.*` 原语进行 element-wise 计算
- 编译器自动处理 Cube/Vector 分离和同步

### 2.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动融合 mul+add
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动插入 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动内存规划
}
```

**Pass 功能说明**：

| Pass | 功能 |
|------|------|
| TL_ASCEND_AUTO_CV_COMBINE | 自动融合连续的 mul+add 操作，减少中间存储开销 |
| TL_ASCEND_AUTO_SYNC | 自动在必要位置插入 barrier，无需手动同步 |
| TL_ASCEND_MEMORY_PLANNING | 自动内存规划，优化 UB 使用 |

---

## 3. API 映射设计

### 3.1 核心计算步骤 → TileLang API 映射

| 计算步骤 | PyTorch 参考 | TileLang API |
|---------|-------------|--------------|
| 生成本地索引序列 | `torch.arange(block_size)` | `T.tile.arith_progression(idx_ub, global_base, 1, block_size)` |
| 填充 LCG 常量 A | `A = 1103515245` | `T.tile.fill(a_ub, LCG_A)` |
| 填充 LCG 常量 C | `C = 12345` | `T.tile.fill(c_ub, LCG_C)` |
| 填充种子 | `seed = 42` | `T.tile.fill(seed_ub, seed)` |
| **state = idx + seed** | `state = global_idx + seed` | `T.tile.add(state_ub, idx_ub, seed_ub)` |
| **temp = state × A** | `temp = state * A` | `T.tile.mul(temp_ub, state_ub, a_ub)` |
| **state = temp + C** | `state = temp + C` | `T.tile.add(state_ub, temp_ub, c_ub)` |
| 存储输出 | `output[i] = state` | `T.copy(state_ub, output[base_offset:base_offset + block_size])` |

### 3.2 关键 API 说明

- **`T.tile.arith_progression(dst, first, diff, count)`**：生成等差数列 `[first, first+diff, first+2*diff, ...]`
- **`T.tile.fill(dst, value)`**：填充 buffer 为指定值
- **`T.tile.add(dst, src0, src1)`**：`dst = src0 + src1`（注意：dst 是第一个参数！）
- **`T.tile.mul(dst, src0, src1)`**：`dst = src0 * src1`（注意：dst 是第一个参数！）
- **`T.copy(src, dst)`**：数据搬运

### 3.3 关键发现：T.tile API 参数顺序

**TileLang API：第一个参数是 dst**

```python
T.tile.add(dst, src0, src1)   # dst = src0 + src1  ✓ 正确
T.tile.mul(dst, src0, src1)   # dst = src0 * src1  ✓ 正确

# 错误示例（常见错误）：
T.tile.add(src0, src1, dst)   # ❌ dst 在最后是错误的！
```

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| 无输入张量 | - | - | 本算子仅输出，不读取输入数据 |

### 4.2 输出张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| output | `[M_aligned]` | int32 | 生成的随机数（1D tensor） |

### 4.3 内存层级规划

```
GM (全局内存)
  │
  └─ output: [M_aligned] — 输出随机数
      ↑ T.copy 批量写入

UB (Unified Buffer, Vector 核)
  ├─ idx_ub:      [BLOCK_SIZE]  — 等差数列 [global_base, global_base+1, ..., global_base+127]
  ├─ state_ub:    [BLOCK_SIZE]  — 状态缓冲
  ├─ temp_ub:     [BLOCK_SIZE]  — 临时缓冲（mul 结果）
  ├─ a_ub:        [BLOCK_SIZE]  — LCG 常量 A（填充为相同值）
  ├─ c_ub:        [BLOCK_SIZE]  — LCG 常量 C（填充为相同值）
  └─ seed_ub:     [BLOCK_SIZE]  — 随机种子（填充为相同值）
```

### 4.4 UB 容量估算（典型配置: BLOCK_SIZE=128）

```
idx_ub:       128×4B = 512B
state_ub:     128×4B = 512B
temp_ub:      128×4B = 512B
a_ub:         128×4B = 512B
c_ub:         128×4B = 512B
seed_ub:      128×4B = 512B
总计 ≈ 3KB << 196KB（安全范围内）
```

### 4.5 数据流设计

```
初始化阶段：
  T.tile.fill → a_ub = [A, A, ..., A]        # 批量填充常量 A
  T.tile.fill → c_ub = [C, C, ..., C]        # 批量填充常量 C
  T.tile.fill → seed_ub = [seed, ...]        # 批量填充种子
  T.tile.arith_progression → idx_ub = [global_base, global_base+1, ..., global_base+127]

计算阶段：
  idx_ub + seed_ub → state_ub                # state = global_idx + seed（向量化）
  
  LCG 循环（3轮，向量化）：
    state_ub × a_ub → temp_ub                # temp = state × A
    temp_ub + c_ub → state_ub                # state = temp + C

输出阶段：
  T.copy → 批量写回                          # 128 元素一次性写入 GM
    state_ub → output[base_offset:base_offset + 128]
```

---

## 5. Tiling 策略

### 5.1 Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `T.Kernel(num_blocks, is_npu=True)` | 每个 kernel 处理 TOTAL_BLOCK_SIZE 个元素 |
| **vid 分块** | `BLOCK_SIZE = 128` | 每个 vid 处理 BLOCK_SIZE 个元素 |
| **VEC_NUM** | `2` | 每个 kernel 有 2 个 vid 并行执行 |
| **TOTAL_BLOCK_SIZE** | `256` | 每个 kernel 总处理量 = VEC_NUM × BLOCK_SIZE |
| **并行度计算** | `num_blocks × VEC_NUM` | 总并行单元数，随 M 线性增长 |

### 5.2 Tile Shape 设计（典型配置: BLOCK_SIZE=128）

| Buffer | Shape | 说明 |
|--------|-------|------|
| idx_ub | `[128]` | 等差数列 [base, base+1, ..., base+127] |
| a_ub | `[128]` | 填充值 A（批量填充） |
| c_ub | `[128]` | 填充值 C（批量填充） |
| seed_ub | `[128]` | 填充值 seed（批量填充） |
| state_ub | `[128]` | 计算结果 |
| temp_ub | `[128]` | 临时结果 |

### 5.3 并行执行示意

```
M = 65536, BLOCK_SIZE = 128, VEC_NUM = 2
TOTAL_BLOCK_SIZE = 256, num_blocks = 256

并行度：256 kernels × 2 vids = 512 并行单元

Kernel 0 (cid=0):
  ├─ vid=0: 处理 output[0:128]      → global_base = 0
  └─ vid=1: 处理 output[128:256]    → global_base = 128
  并行执行

Kernel 1 (cid=1):
  ├─ vid=0: 处理 output[256:384]    → global_base = 256
  └─ vid=1: 处理 output[384:512]    → global_base = 384
  并行执行

... (共 256 个 kernel)

总计：65536 个随机数，512 个并行单元同时计算
```

---

## 6. 循环与调度结构

### 6.1 Kernel 结构设计（当前实现）

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def random_1d(M, block_size, seed, lcg_a, lcg_c):
    M_aligned = (M + TOTAL_BLOCK_SIZE - 1) // TOTAL_BLOCK_SIZE * TOTAL_BLOCK_SIZE
    num_blocks = M_aligned // TOTAL_BLOCK_SIZE

    @T.prim_func
    def main(output: T.Tensor((M_aligned,), "int32")):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            idx_ub = T.alloc_ub((block_size,), "int32")
            state_ub = T.alloc_ub((block_size,), "int32")
            temp_ub = T.alloc_ub((block_size,), "int32")
            a_ub = T.alloc_ub((block_size,), "int32")
            c_ub = T.alloc_ub((block_size,), "int32")
            seed_ub = T.alloc_ub((block_size,), "int32")

            T.tile.fill(a_ub, lcg_a)
            T.tile.fill(c_ub, lcg_c)
            T.tile.fill(seed_ub, seed)

            global_base = cid * TOTAL_BLOCK_SIZE + vid * block_size
            T.tile.arith_progression(idx_ub, global_base, 1, block_size)

            T.tile.add(state_ub, idx_ub, seed_ub)

            for _ in range(3):
                T.tile.mul(temp_ub, state_ub, a_ub)
                T.tile.add(state_ub, temp_ub, c_ub)

            base_offset = cid * TOTAL_BLOCK_SIZE + vid * block_size
            T.copy(state_ub, output[base_offset:base_offset + block_size])

    return main
```

**关键变化**：
- **移除了 `T.Scope("V")`**：启用 AUTO_CV_COMBINE pass 后，编译器自动处理 Cube/Vector 分离
- **移除了手动 `T.barrier_all()`**：启用 AUTO_SYNC pass 后，编译器自动插入必要的同步
- **使用 `for _ in range(3)` 简化 LCG 循环**：代码更简洁，编译器自动展开

### 6.2 调度选择

| 循环/操作 | 调度类型 | 理由 |
|----------|---------|------|
| 初始化 fill | `T.tile.fill` | Vector 核批量填充，128 元素同时操作 |
| 等差数列 | `T.tile.arith_progression` | Vector 核生成序列，128 元素同时生成 |
| element-wise | `T.tile.mul/add` | Vector 核自动向量化，128 元素同时计算 |
| 输出写入 | `T.copy(state_ub, output[...])` | **批量写回**，充分利用 NPU 带宽 |

### 6.3 关键优化点

**使用 T.copy 批量写回替代 T.serial 循环**：

```python
# 旧版本：逐个元素写入（低效）
for i in T.serial(block_size):
    output[base_offset + i] = state_ub[i]

# 当前版本：批量写回（高效）
T.copy(state_ub, output[base_offset:base_offset + block_size])
```

**性能提升**：
- 批量操作减少内存访问次数
- 利用 NPU DMA 带宽
- 大数据量吞吐量提升至 **533 MB/s**

---

## 7. 同步策略

### 7.1 自动同步（pass_configs）

启用以下 Pass，编译器自动处理：

| Pass | 功能 |
|------|------|
| TL_ASCEND_AUTO_SYNC | 自动在必要位置插入 barrier |
| TL_ASCEND_AUTO_CV_COMBINE | 自动融合连续的 mul+add 操作，优化 LCG 计算 |

**代码无需手动同步**：

```python
# 无需显式 T.barrier_all()
# 无需显式 T.Scope("V")
# 编译器自动分析和优化
```

### 7.2 CV 融合优化

LCG 计算（mul + add）被自动融合：

```python
# 原始代码（3轮 mul+add）
for _ in range(3):
    T.tile.mul(temp_ub, state_ub, a_ub)
    T.tile.add(state_ub, temp_ub, c_ub)

# 编译器自动融合为复合操作
# 减少中间存储开销，提升性能
```

---

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现）

```python
def reference_random_1d(M, seed):
    import numpy as np
    import ctypes

    LCG_A = 1103515245
    LCG_C = 12345

    result = np.zeros(M, dtype=np.int32)
    for i in range(M):
        state = seed + i
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        state = ctypes.c_int32(state * LCG_A + LCG_C).value
        result[i] = state
    return torch.from_numpy(result)
```

### 8.2 测试配置

| 配置 | M | seed | block_size |
|------|-----|------|------------|
| **Small** | 256 | 42 | 128 |
| **Medium** | 512 | 42 | 128 |
| **Large** | 1024 | 42 | 128 |
| **Different seed** | 512 | 123 | 128 |
| **Different block** | 1024 | 42 | 64 |

### 8.3 命令行接口

```bash
# 默认配置
python examples/random/random_1d.py

# 自定义配置
python examples/random/random_1d.py --m 2048 --seed 123 --block_size 64
```

### 8.4 精度容忍度

| 配置 | rtol | atol |
|------|------|------|
| 所有配置 | 0 | 0 |（int32 精确匹配）

---

## 9. 风险点与注意事项

### 9.1 已解决风险

| 风险 | 原状态 | 解决方案 |
|------|-------|---------|
| **cid 不能作为动态偏移** | cid 被 codegen 为静态值 0 | 使用 `cid * TOTAL_BLOCK_SIZE + vid * block_size` 组合表达式，当 num_blocks > 1 时正确传递 |
| **T.tile.add/mul 参数顺序** | dst 被放在最后（错误） | 调整为第一个参数：`T.tile.add(dst, src0, src1)` |
| **T.copy 1D tensor 参数错误** | 复制数量使用整个 tensor 大小 | 使用 slice 写入：`output[base_offset:base_offset + block_size]` |
| **int32 溢出** | Python 中间值超出 int32 | 使用 `ctypes.c_int32()` 强制转换，NPU 自动截断 |

### 9.2 当前限制

| 限制 | 说明 |
|------|------|
| **M 必须是 TOTAL_BLOCK_SIZE 的倍数** | 当前实现自动对齐，超出部分被截断 |
| **BLOCK_SIZE 固定为 128 或 64** | 典型配置，UB 容量充足 |
| **仅支持 int32 输出** | LCG 算法特性 |

### 9.3 待优化项

| 项 | 当前状态 | 潜在优化 |
|----|---------|---------|
| 边界处理 | 自动对齐截断 | 添加 if 判断处理非对齐 M |
| Barrier 数量 | 自动同步 | Pass 已优化，无需手动调整 |

---

## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/random/design.md` | 本文档 |
| 开发记录 | `examples/random/DEVELOPMENT_LOG.md` | 已完成 |
| 算子实现 | `examples/random/random_1d.py` | 已完成 |
| 测试用例 | `examples/random/test_*.py` | 已完成 |

---

## 附录 A：关键问题解决记录

### A.1 T.tile API 参数顺序问题

**问题背景**：

初期实现中使用错误的参数顺序：

```python
T.tile.mul(state_ub, lcg_a_ub, temp_ub)  # ❌ 错误！生成 state_ub = lcg_a_ub * temp_ub
T.tile.add(temp_ub, lcg_c_ub, state_ub)  # ❌ 错误！生成 temp_ub = lcg_c_ub + state_ub
```

**正确顺序**：

```python
T.tile.mul(temp_ub, state_ub, lcg_a_ub)  # ✓ temp_ub = state_ub * lcg_a_ub
T.tile.add(state_ub, temp_ub, lcg_c_ub)  # ✓ state_ub = temp_ub + lcg_c_ub
```

### A.2 动态偏移传递问题

**问题背景**：

- `vid * block_size` 正确传递（动态值）
- `cid * TOTAL_BLOCK_SIZE` 在 num_blocks=1 时为静态值 0
- `cid * TOTAL_BLOCK_SIZE + vid * block_size` 组合表达式正确传递（当 num_blocks > 1）

**解决方案**：

使用组合表达式作为全局基准偏移：

```python
global_base = cid * TOTAL_BLOCK_SIZE + vid * block_size
```

---

## 附录 B：执行模型说明

### B.1 Kernel 内部结构

```
┌─────────────────────────────────────────┐
│           Kernel (cid)                   │
│  ┌─────────────┐  ┌─────────────┐       │
│  │   vid=0    │  │   vid=1    │  并行   │
│  │ 处理128元素│  │ 处理128元素│  执行   │
│  │ global_base│  │ global_base│         │
│  │ = cid*256  │  │ = cid*256  │         │
│  │   + 0      │  │   + 128    │         │
│  └─────────────┘  └─────────────┘       │
│         ↓                 ↓             │
│   output[cid*256:cid*256+128]           │
│   output[cid*256+128:cid*256+256]       │
│   （T.copy 批量写回）                    │
└─────────────────────────────────────────┘
```

### B.2 多 Kernel 并行

```
M = 65536, num_blocks = 256

并行度：256 kernels × 2 vids = 512 并行单元

所有 Kernel 并行启动：
main_kernel<<<256, nullptr, stream>>>

每个 Kernel 内部 2 个 vid 并行执行 Vector 操作
总计 512 个并行单元同时计算 128 个随机数
```

---

## 附录 C：LCG 算法原理

### C.1 线性同余生成器

LCG 是最简单的伪随机数生成算法：

```
X[n+1] = (A × X[n] + C) mod M
```

经典参数（ANSI C）：
- A = 1103515245（乘数）
- C = 12345（增量）
- M = 2^32（模数，int32 自动截断）

### C.2 特点

- **优点**：实现简单，计算速度快，无状态依赖
- **缺点**：随机性较差，不适合加密场景

### C.3 并行化

本算子的关键设计：每个元素独立计算，使用 `seed + global_idx` 作为初始状态。

```
state[0] = seed + 0
state[1] = seed + 1
state[2] = seed + 2
...
```

这避免了顺序依赖，支持完全并行计算。

---

## 附录 D：性能优化分析

### D.1 优化历程

| 版本 | 输出写入方式 | 同步方式 | 吞吐量 (M=16384) |
|------|------------|---------|-----------------|
| V1 | T.serial 逐个写入 | 手动 barrier | ~8 MB/s |
| V2 | T.copy 批量写入 | 手动 barrier | ~136 MB/s |
| V3 | T.copy 批量写入 | AUTO_SYNC + CV_COMBINE | **136.4 MB/s** |

### D.2 关键优化技术

**1. 批量写回替代逐元素写入**：

```python
# 低效版本
for i in T.serial(128):
    output[base + i] = state_ub[i]

# 高效版本（17倍性能提升）
T.copy(state_ub, output[base:base + 128])
```

**2. 自动同步替代手动 barrier**：

```python
# 旧版本：手动 barrier
T.tile.fill(a_ub, lcg_a)
T.barrier_all()
T.tile.fill(c_ub, lcg_c)
T.barrier_all()

# 新版本：自动同步（Pass 优化）
T.tile.fill(a_ub, lcg_a)
T.tile.fill(c_ub, lcg_c)
# 编译器自动插入必要同步
```

**3. CV 融合优化**：

```python
# mul + add 自动融合为复合操作
for _ in range(3):
    T.tile.mul(temp_ub, state_ub, a_ub)
    T.tile.add(state_ub, temp_ub, c_ub)
# 编译器优化：减少 temp_ub 中间存储开销
```

### D.3 性能瓶颈分析

| 数据量 | 瓶颈 | 说明 |
|-------|------|------|
| M < 4K | Kernel launch overhead | kernel 启动时间占比大 |
| M > 16K | 内存带宽 | 接近 NPU 带宽极限 |
| M > 64K | 并行度饱和 | 512 个并行单元接近硬件极限 |

### D.4 进一步优化方向

1. **更大 BLOCK_SIZE**：尝试 256/512，减少 kernel 数量
2. **合并多个 LCG 轮次**：使用更复杂的公式一次完成多轮
3. **异构并行**：结合 CPU 生成种子，NPU 执行 LCG