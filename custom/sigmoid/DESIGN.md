# Sigmoid 算子设计文档

## 1. 概述

### 1.1 算子名称

sigmoid

### 1.2 功能描述

Sigmoid 激活函数，对输入张量逐元素计算 `1 / (1 + exp(-x))`，将任意实数映射到 (0, 1) 区间。常用于二分类 logits 输出、门控机制（GRU/LSTM gate）、注意力权重的概率化等场景。

### 1.3 数学公式

$$
\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}
$$

### 1.4 算法描述

Sigmoid 是逐元素（element-wise）激活算子，计算步骤分解为 5 步（与本项目 `examples/activation/sigmoid.py` 已验证实现一致）：

1. **取负**：`t = 0 - x`（等价 `t = -x`）
2. **指数**：`t = exp(t)`（即 `exp(-x)`）
3. **加一**：`t = t + 1`（即 `1 + exp(-x)`）
4. **倒数**：`y = 1 / t`（即 `1 / (1 + exp(-x))`）
5. **写回**：输出到 GM

数值稳定性说明：直接公式 `1/(1+exp(-x))` 在 float16 下边界自动正确——
- `x` 大正数（> 11.09）：`exp(-x) → 0`，`1/(1+0) = 1`，与 `sigmoid(x→+∞) = 1` 一致 ✓
- `x` 大负数（< -11.09）：`-x > 11.09`，`exp(-x) → +inf`，`1/(1+inf) = 0`，与 `sigmoid(x→-∞) = 0` 一致 ✓

float16 的 `exp` 上溢阈值约 11.09，溢出产生 `+inf` 后参与除法自然得到正确极限值，无需 if-else 分支（Ascend SIMD 架构不支持元素级条件分支，且本公式天然免分支）。该行为与 `torch.sigmoid(float16)` 在大 |x| 处的输出一致。

### 1.5 数据流图

```
输入 GM[A] → T.copy → UB[a_ub] → tile.fill(zero_ub,0) → tile.sub(a_ub,zero_ub,a_ub) → tile.exp(a_ub,a_ub)
            → tile.add(a_ub,a_ub,1.0) → tile.reciprocal(b_ub,a_ub) → T.copy → 输出 GM[B]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|---------|------|------|
| 计算类型 | 纯 element-wise，无 matmul、无归约 | 纯 Vector，仅需 UB |
| 复杂度 | 5 步分解（fill/sub/exp/add/reciprocal），无核间协作 | 单核内多步，无 CV 融合需求 |
| 内存层级 | 仅 GM ↔ UB，不涉及 L1/L0A/L0B/L0C | 编译器自动映射 shared→UB 即可 |
| 同步 | 单 block 内 V 核 vid 切分，无跨 block 依赖 | 自动同步足够 |
| 参考实现 | `examples/activation/sigmoid.py` 用 Developer 模式（`T.alloc_shared` + 全 pass_configs 开启）已验证通过 | 同模式可复用 |

用户明确指定 Developer 模式，且算子特征与 `sigmoid.py` 完全契合，无需 Expert/混合模式的手动内存层级控制。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared(shape, dtype)` — 编译器自动映射到 UB（Vector 核缓冲） |
| 计算方式 | `T.tile.xxx` Buffer 级 SIMD 原语（fill/sub/exp/add/reciprocal） |
| 作用域 | 编译器自动分离 Cube/Vector（本算子无 Cube 计算，纯 V 核执行） |
| 同步方式 | 自动同步（`TL_ASCEND_AUTO_SYNC=True`），无需手动 `T.barrier_all` / `T.Scope` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t1 = 0 - x` | 取负（用 sub 实现，避免单独的 negate 原语） |
| 2 | `t2 = exp(t1)` | 指数，`t2 = exp(-x)` |
| 3 | `t3 = t2 + 1` | 加一，`t3 = 1 + exp(-x)` |
| 4 | `y = 1 / t3` | 倒数，`y = 1 / (1 + exp(-x))` |
| 5 | 写回 GM | 输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 来源确认 |
|------|----------|-------------|------|------|----------|
| 搬入 | `a_ub = A[...]` | `T.copy(A[...], a_ub)` | src=GM slice, dst=UB | Developer | `sigmoid.py:31` ✓ / api-kernel-memory.md §3 |
| 填零 | `zero_ub = 0` | `T.tile.fill(zero_ub, 0.0)` | dst=UB, value=0.0 scalar | Developer | `sigmoid.py:32` ✓ / api-compute.md §4.10 |
| 取负 | `a_ub = 0 - a_ub` | `T.tile.sub(a_ub, zero_ub, a_ub)` | dst=UB, src0=zero_ub, src1=a_ub | Developer | `sigmoid.py:33` ✓ / api-compute.md §4.1 |
| 指数 | `a_ub = exp(a_ub)` | `T.tile.exp(a_ub, a_ub)` | dst=UB, src0=UB（原地） | Developer | `sigmoid.py:34` ✓ / api-compute.md §4.2 |
| 加一 | `a_ub = a_ub + 1` | `T.tile.add(a_ub, a_ub, 1.0)` | dst=UB, src0=UB, src1=1.0 scalar | Developer | `sigmoid.py:35` ✓ / api-compute.md §4.1 |
| 倒数 | `b_ub = 1 / a_ub` | `T.tile.reciprocal(b_ub, a_ub)` | dst=UB, src0=UB | Developer | `sigmoid.py:36` ✓ / api-compute.md §4.2 |
| 搬出 | `B[...] = b_ub` | `T.copy(b_ub, B[...])` | src=UB, dst=GM slice | Developer | `sigmoid.py:37` ✓ / api-kernel-memory.md §3 |

**备选简化路径**（Stage 2 可验证）：`T.tile.sigmoid(output_ub, input_ub)` 一步完成（参考 `examples/activation/sigmoidv2.py:29`）。该原语在 `sigmoidv2.py` 中已验证可用，但 `sigmoidv2.py` 用 `T.alloc_ub`（Expert 风格显式层级）。本设计主方案采用 `sigmoid.py` 的分解写法（与 Developer 模式 + `T.alloc_shared` 完全契合，且 5 步分解对各 dtype 行为可预测）；Stage 2 实现时可先尝试 `T.tile.sigmoid` 简化路径，若与 Developer pass_configs 兼容则采用，否则回退到分解方案。

### 3.3 计算伪代码

```python
import tilelang
import tilelang.language as T

VEC_NUM = 2

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float16"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. 分配 buffer（Developer: alloc_shared 自动映射 UB；vid 切分行）
            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # 2. 数据搬入 GM → UB
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            # 3. 计算：y = 1 / (1 + exp(-x))
            T.tile.fill(zero_ub, 0.0)                       # zero = 0
            T.tile.sub(a_ub, zero_ub, a_ub)                 # a = 0 - x = -x
            T.tile.exp(a_ub, a_ub)                          # a = exp(-x)
            T.tile.add(a_ub, a_ub, 1.0)                     # a = 1 + exp(-x)
            T.tile.reciprocal(b_ub, a_ub)                   # b = 1 / (1 + exp(-x))

            # 4. 数据搬出 UB → GM
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.alloc_shared` | api-kernel-memory.md §2 / `sigmoid.py:27` | ✅ 已验证（sigmoid.py 测试通过） |
| `T.copy` (GM↔UB) | api-kernel-memory.md §3 / `sigmoid.py:31,37` | ✅ 已验证 |
| `T.tile.fill` | api-compute.md §4.10 / `sigmoid.py:32` | ✅ 已验证 |
| `T.tile.sub` | api-compute.md §4.1 / `sigmoid.py:33` | ✅ 已验证 |
| `T.tile.exp` | api-compute.md §4.2 / `sigmoid.py:34` | ✅ 已验证 |
| `T.tile.add` (scalar src1) | api-compute.md §4.1 / `sigmoid.py:35` | ✅ 已验证 |
| `T.tile.reciprocal` | api-compute.md §4.2 / `sigmoid.py:36` | ✅ 已验证 |
| `T.ceildiv` | `sigmoid.py:16-17` | ✅ 已验证（处理非整除） |
| `T.tile.sigmoid`（备选） | api-compute.md 未列但 `sigmoidv2.py:29` 使用 | ⚠️ 备选，Stage 2 验证与 Developer pass_configs 兼容性 |

**所有主方案 API 均来自 `examples/activation/sigmoid.py` 已验证实现，无凭记忆猜测。**

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(m_num * n_num)`，无三维需求 |
| threads 参数限制（仅 1 或 2） | No | 不设 threads 参数（用默认；vid 切分由 VEC_NUM=2 隐式处理） |
| 动态循环边界不支持 | No | 无循环迭代（单次搬入+计算+搬出），不涉及循环边界 |
| 流水线不支持动态边界 | No | 不使用 `T.Pipelined`（element-wise 单步足够） |
| GPU 专用 API | No | 全部使用 `T.tile.xxx` Ascend 原语，无 CUDA API |
| GEMM 非整除风险 | No | 无 GEMM 计算 |
| L0C 容量上限 | No | 无 Cube 计算，不分配 L0C |

### 3.5.2 参考实现差异说明

本算子无外部 GPU 参考实现迁移，直接基于本项目 `examples/activation/sigmoid.py` 实现。用户提供的 `torch.sigmoid` 仅作 golden 参考实现，不涉及 kernel 迁移。

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/activation/sigmoid.py` | **极高（同源）** | 完整 sigmoid kernel：Developer 模式 + `T.alloc_shared` + `T.tile.fill/sub/exp/add/reciprocal` 分解 + VEC_NUM=2 vid 切分 + `T.ceildiv` 非整除处理。测试 shape (256,256,64,64)/(300,300,64,64)/(1100,50000,128,128) 全通过 |
| `examples/activation/sigmoidv2.py` | 高 | `T.tile.sigmoid` 一步原语写法（备选简化路径），`T.alloc_ub` 显式层级 |
| `examples/activation/sigmoidv2_slice.py` | 中 | 按 row slice 调用 `T.tile.sigmoid`，处理 2D 切片 |
| `examples/activation/silu.py` | 高 | `silu = x * sigmoid(x)`，sigmoid 部分计算分解与 `sigmoid.py` 完全一致（fill/sub/exp/add/div），可交叉验证 API 用法 |
| `examples/elementwise/elementwise_add.py` | 中 | element-wise 基础结构（`T.Kernel` + vid 切分 + `T.copy` GM↔UB），Expert 风格（`T.alloc_ub` + `T.Scope("V")` + `T.barrier_all`）作对照 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A（x） | (M, N) | float16 / float32 | 输入张量，M/B 和 N 均为运行时维度 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B（y） | (M, N) | 同输入 | 输出张量，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB（alloc_shared 自动映射） | 输入 tile 缓冲 + 中间计算原地复用 |
| b_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 输出缓冲（reciprocal 结果） |
| zero_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 全 0 缓冲，用于 sub 取负 |

### 4.4 内存搬运路径

```
纯 Vector 路径（element-wise）：

GM[A] --T.copy--> UB[a_ub]
                    |
            tile.fill(zero_ub, 0.0)
            tile.sub(a_ub, zero_ub, a_ub)   # a = -x
            tile.exp(a_ub, a_ub)             # a = exp(-x)
            tile.add(a_ub, a_ub, 1.0)        # a = 1 + exp(-x)
            tile.reciprocal(b_ub, a_ub)      # b = 1/(1+exp(-x))
                    |
UB[b_ub] --T.copy--> GM[B]
```

**层级说明**：纯 Vector 算子，数据全程在 UB 上操作，不涉及 L1（Cube 缓存）/ L0A / L0B / L0C。`T.alloc_shared` 在无 Cube 计算时被编译器自动映射到 UB（Vector 核缓冲）。

### 4.5 UB 内存预算

以主配置 `block_M=128, block_N=128, VEC_NUM=2` 为例（每个 V 核处理 64 行）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a_ub | (64, 128) | float16 | 64 × 128 × 2 = 16384 (16 KB) |
| b_ub | (64, 128) | float16 | 16384 (16 KB) |
| zero_ub | (64, 128) | float16 | 16384 (16 KB) |
| **总计** | | | **49152 (48 KB)** |

- 目标平台 UB 容量：196608 Byte（192 KB，Ascend910B3，见 api-kernel-memory.md §2）
- 占用比：48 KB / 192 KB = 25% ✓（充裕）
- float32 时单 buffer 翻倍至 32 KB，总计 96 KB / 192 KB = 50% ✓（仍可接受）

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| M（对应需求的 B） | 作为 `@tilelang.jit` 函数参数传入 | 1 ~ 64K | 行数，每次 shape 编译一个 kernel 版本（参考 `sigmoid.py` 模式） |
| N | 作为 `@tilelang.jit` 函数参数传入 | 1 ~ 64K | 列数，每次 shape 编译一个 kernel 版本 |

**动态 shape 策略说明**：
- **主方案（采用）**：M, N 作为 `jit` 函数参数，每次调用 `sigmoid(M, N, block_M, block_N, dtype)` 编译一个针对该 shape 的 kernel。这是 `examples/activation/sigmoid.py` 的已验证做法，简单可靠，编译器可充分优化。L0 测试计划中每个 shape 各自编译。
- **可选优化（Stage 2 探索）**：用 `T.dyn['M']` / `T.dyn['N']` 声明真动态 shape，一次编译支持任意 shape。本设计不强制采用，留作 Stage 2 性能调优阶段的可选优化项。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],                # B（第 2 个参数）为输出
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    },
)
def sigmoid(M, N, block_M, block_N, dtype="float16"):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：算子仅包含 element-wise 运算（fill/sub/exp/add/reciprocal），无 matmul、无归约。数据全程在 UB 上操作，不涉及 Cube 核（L1/L0）。

### 5.2 Block 划分

```python
block_M = 128   # M 维分块：与 sigmoid.py 大规模用例一致，平衡 UB 占用与并行度
block_N = 128   # N 维分块：128 × fp16(2B) = 256B，满足 UB 32B 对齐（256/32=8）
VEC_NUM = 2     # V 核数：每个 V 核处理 block_M // 2 = 64 行

m_num = T.ceildiv(M, block_M)   # M 方向 block 数（ceildiv 处理非整除）
n_num = T.ceildiv(N, block_N)   # N 方向 block 数
block_num = m_num * n_num       # 一维 block 总数
```

**block size 选择理由**：
- `block_M=128`：与 `sigmoid.py` 的 (1100, 50000, 128, 128) 大规模用例一致，已验证可行
- `block_N=128`：满足 UB 32B 对齐（128 × 2B = 256B），且单 block 数据量适中
- 小 shape 场景（如 L0 的 (256, 256)）可配 `block_M=64, block_N=64`（参考 sigmoid.py 的 (256,256,64,64)）

### 5.3 约束分析

- **UB 对齐约束**：`block_N=128` × fp16(2B) = 256B，32B 整除 ✓（float32 时 128×4B=512B，同样 32B 整除 ✓）
- **UB 容量**：3 buffer × 16KB = 48KB < 192KB（192KB/UB，Ascend910B3）✓
- **L0 容量**：无 Cube 计算，不适用
- **V 核切分**：`block_M // VEC_NUM = 64`，每 V 核处理 64 行，读写索引一致（`bx*block_M + vid*block_M//VEC_NUM`）✓

### 5.4 注意事项（非整除处理）

**非整除场景**：当 `M % block_M ≠ 0` 或 `N % block_N ≠ 0` 时：
- 使用 `T.ceildiv(M, block_M)` / `T.ceildiv(N, block_N)` 计算 block 数（向上取整，保证覆盖所有元素）
- `T.copy` 已支持动态 shape 切片自动处理尾块（参考 api-kernel-memory.md §3 "T.copy 动态 shape 切片"），**不需要 host 侧 zero-padding**
- 尾块 block 中超出有效范围的部分，`T.copy` 会自动处理（参考 `examples/activation/sigmoid.py` 的 (300, 300, 64, 64) 测试用例，300 不被 64 整除但测试通过）
- 本算子为 element-wise 无归约，尾块计算结果独立，无跨 block 竞态风险

**L0 测试**只用 block 整除的规则 shape；非整除/尾块/质数 shape 留给 L1（Stage 2 由 `tilelang-op-test-design` 场景 B 扩展）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级（M×N 分块） | 隐式并行 | `T.Kernel(m_num * n_num)` | 每个 block 处理一个 (block_M, block_N) tile，由硬件调度到不同核 |
| V 核切分 | 隐式 | `vid ∈ {0, 1}` | VEC_NUM=2，每 V 核处理 block_M//2 行（`bx*block_M + vid*block_M//VEC_NUM`） |
| 元素级计算 | 无显式循环 | `T.tile.xxx` 原语 | Buffer 级 SIMD 操作，整个 tile 一次性计算，无需 `T.Parallel` 循环 |

**说明**：本算子用 `T.tile.xxx` 原语（Buffer 级 SIMD）而非 `T.Parallel` + 符号 API。原因：`T.tile.xxx` 直接触发 Ascend Vector 指令，性能更优且与 `sigmoid.py` 已验证实现一致。`T.Parallel` 内的符号 API 会被编译器 lowering 为 `T.tile.xxx`，但显式调用 `T.tile.xxx` 更直接。

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）；无 K 维迭代循环
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    # 单次搬入 → 5 步 tile 计算 → 单次搬出（无循环）
    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
    T.tile.fill(zero_ub, 0.0)
    T.tile.sub(a_ub, zero_ub, a_ub)
    T.tile.exp(a_ub, a_ub)
    T.tile.add(a_ub, a_ub, 1.0)
    T.tile.reciprocal(b_ub, a_ub)
    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- element-wise 单步计算（搬入→计算→搬出），无 K 维迭代累加
- 单 block 内计算量小（64×128=8192 元素），流水线开销大于收益
- `sigmoid.py` 未使用流水线且性能已验证

若 Stage 3 性能调优发现搬入/计算/搬出存在等待，可探索双 buffer 流水线（`T.Pipelined(num_stages=2)`），但本设计不预设。

### 6.4 尾块处理

非整除时尾块由 `T.ceildiv` + `T.copy` 动态切片自动处理（见 §5.4）。无显式尾块循环逻辑。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下由 `TL_ASCEND_AUTO_SYNC=True` 自动插入同步点，无需手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy` 搬入后 | 自动同步 | 确保数据写入 UB 完成后再计算 |
| `T.tile.xxx` 之间 | 自动同步 | 原地复用 a_ub（sub→exp→add），需保证前一步写完成 |
| `T.copy` 搬出前 | 自动同步 | 确保 reciprocal 结果写完成后再搬出 |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内，含上述同步点）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划（UB 分配优化）
}
```

**未开启的配置**：
- `TL_ASCEND_AUTO_CV_SYNC`：核间同步，本算子无 CV 融合（纯 Vector），不需要

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**：否

**判定依据**：sigmoid 是纯 element-wise 激活算子，无 GEMM（matmul）计算，不存在 Cube↔Vector 核间协作需求。Developer 模式下 `TL_ASCEND_AUTO_CV_COMBINE=True` 对纯 Vector 算子退化为无操作（不产生 workspace/vid 开销）。

本章节不适用，无 workspace 规格、无 CV 交互设计。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch

def golden_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid 参考实现（PyTorch）。
    
    Args:
        x: 输入张量，float16 或 float32
    Returns:
        y: sigmoid(x) = 1 / (1 + exp(-x))，shape/dtype 与输入一致
    """
    return torch.sigmoid(x)
```

**Golden 选择说明**：`torch.sigmoid` 内部采用数值稳定实现（对大 |x| 走快速路径），输出为合法 sigmoid 值。对 float16 输入，`torch.sigmoid` 在 NPU 上直接计算 float16 结果；本设计 kernel 直接公式 `1/(1+exp(-x))` 在 float16 边界处产生相同极限值（0 或 1），二者一致。

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**（规则 shape，block 整除），供 Stage 2 快速精度收敛。
> L1（功能，含不规则/尾块/质数 shape）/ L2（异常输入）/ Boundary（INF/NAN/极值）的**完整分层套件由 `tilelang-op-test-design` 场景 B 在 Stage 2 L0 通过后扩展**——不在此枚举。

**算子类别判断**（由 `tilelang-op-test-design` 场景 A 生成）：
- 计算类型：纯 Vector（element-wise，无 matmul）
- 复杂度：Multi（5 步分解：fill/sub/exp/add/reciprocal）
- 数学特征：sigmoid → Activation 类
- 综合类别：Activation（多步激活）
- 测试策略：dtype 组合（float16 + float32）+ 规则 shape 组合，逐元素验证

**L0 用例集**（规则 shape，block 整除；≤50 用例）：

| 用例名 | 级别 | Shape (M, N) | dtype | block (block_M, block_N) | 说明 |
|--------|------|--------------|-------|--------------------------|------|
| l0_min | L0 | (128, 128) | float16 | (128, 128) | 最小规则 shape，单 block，基础功能验证 |
| l0_small | L0 | (256, 256) | float16 | (64, 64) | 小规模规则（参考 sigmoid.py 同 shape），多 block |
| l0_mid | L0 | (512, 512) | float16 | (128, 128) | 中等规则，4×4 block 网格 |
| l0_large | L0 | (1024, 1024) | float16 | (128, 128) | 中大规则，8×8 block 网格 |
| l0_wide | L0 | (1024, 8192) | float16 | (128, 128) | 大规模长行（参考 sigmoid.py 的 50000 列量级，取 8192 保证 128 整除） |
| l0_fp32_small | L0 | (256, 256) | float32 | (64, 64) | float32 基础验证（参考 sigmoid.py 同 shape） |
| l0_fp32_mid | L0 | (512, 512) | float32 | (128, 128) | float32 中等规则 |

**L0 输入数据生成**：`torch.randn(shape, dtype=..., device='npu')`，标准正态分布（|x| 主要落在 [-4, 4]，少量达 ±6），覆盖 sigmoid 的敏感区间（斜率最大处在 x=0）。极值/特殊值输入留给 Boundary 测试（Stage 2 扩展）。

**L0 验证流程**（供 Stage 2 落地参考）：
```python
def test_sigmoid_l0():
    """L0 门槛测试：规则 shape，block 整除。返回是否全过。"""
    test_configs = [
        # (dtype, shape, block)
        ("float16", (128, 128), (128, 128)),
        ("float16", (256, 256), (64, 64)),
        ("float16", (512, 512), (128, 128)),
        ("float16", (1024, 1024), (128, 128)),
        ("float16", (1024, 8192), (128, 128)),
        ("float32", (256, 256), (64, 64)),
        ("float32", (512, 512), (128, 128)),
    ]
    ok = True
    for dtype, shape, block in test_configs:
        M, N = shape
        block_M, block_N = block
        kernel = sigmoid(M, N, block_M, block_N, dtype=dtype)
        x = torch.randn(M, N, dtype=getattr(torch, dtype), device="npu")
        y = kernel(x)
        ref = torch.sigmoid(x)
        passed, ratio, max_abs = check_precision(y, ref, dtype)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0 shape={shape} dtype={dtype} "
              f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
    return ok
```

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关），L0/L1/Boundary 套用精度比对（L2 为非法输入负向测试，不比精度）；整型按 0 误差精确匹配。完整定义见 `tilelang-op-test-design/references/precision-standard.md`。

本算子支持 float16 + float32 两个 dtype（与 §4 数据规格一致）：

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 阈值取自 `precision-standard.md §二`，与算子类别无关。`required_matched_ratio` 浮点统一 0.99；本算子无整型 dtype，不列整型行。
> **与 sigmoid.py 示例的差异**：`sigmoid.py` 测试用 `rtol=1e-2, atol=1e-2`（较宽松），本设计采用 `precision-standard.md` 标准值（更严格）。对 float16 标准正态输入（|x| < 6），`1/(1+exp(-x))` 直接公式的误差主要来自 float16 exp/reciprocal 的舍入，应满足 `atol=6.10e-5, rtol=1.95e-3`；L0 验证时若个别用例不达标，Stage 2 可评估是否对 float16 中间计算升精度（float32 计算 + 末尾 cast 回 float16）。

---

## 10. 风险点与注意事项

### 10.1 已知约束

| 约束 | 本算子状态 | 说明 |
|------|-----------|------|
| 三维 Kernel | 不涉及 | 一维 `T.Kernel(m_num * n_num)` |
| threads 参数 | 不涉及 | 不设 threads，用默认 + VEC_NUM=2 隐式 vid 切分 |
| 动态循环边界 | 不涉及 | 无循环迭代 |
| L0C 容量 | 不涉及 | 无 Cube 计算 |
| GEMM 非整除 | 不涉及 | 无 GEMM |
| UB 容量 | 已预算 | 48KB（fp16）/ 96KB（fp32）< 192KB ✓ |
| UB 对齐 | 已满足 | block_N=128 × fp16(2B) = 256B，32B 整除 ✓ |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| float16 exp 溢出 | `\|x\| > 11.09` 时 `exp(-x)` 上溢为 inf | 边界处产生 0/1 极限值（正确） | 无需处理，极限值与 torch.sigmoid 一致；若 Boundary 测试发现不一致，评估 float32 中间计算 |
| `T.tile.sigmoid` 与 Developer pass_configs 不兼容 | Stage 2 采用备选简化路径时 | 编译错误 | 回退到 5 步分解方案（主方案，已验证） |
| 读写索引不一致 | vid 切分时读写行偏移不匹配 | 结果错乱 | 读写均用 `bx*block_M + vid*block_M//VEC_NUM`（参考 sigmoid.py） |
| block_N 不满足 32B 对齐 | block_N × sizeof(dtype) 不是 32 倍数 | DMA 搬运异常 | block_N=128（fp16: 256B, fp32: 512B）均满足 |
| 尾块越界写 | 非整除时尾块 block 写入超出有效范围 | 越界写邻近数据 | `T.copy` 动态切片自动处理（参考 sigmoid.py 的 300×300 用例）；本算子 element-wise 无跨 block 竞态 |

### 10.3 特殊场景处理

| 场景 | 处理 | 归属层级 |
|------|------|---------|
| 非整除 shape（M%block_M≠0 等） | `T.ceildiv` + `T.copy` 动态切片自动处理尾块 | L1（Stage 2 扩展） |
| 极小 shape（如 (1, 128)） | `T.ceildiv` 保证至少 1 个 block；block_M 可调小至 1 | L1（Stage 2 扩展） |
| 大 \|x\| 极值（\|x\| > 11） | 直接公式极限值正确（0 或 1） | Boundary（Stage 2 扩展） |
| INF/NAN 输入 | `exp(-inf)=0→sigmoid=1`；`exp(-nan)=nan→sigmoid=nan` | Boundary（Stage 2 扩展） |
| 空 tensor（0 元素） | `T.ceildiv(0, block)=0`，无 block 启动 | L2/Boundary（Stage 2 扩展） |
| float32 dtype | 直接用 float32 计算，精度更高，UB 占用翻倍（96KB < 192KB ✓） | L0 已覆盖 |

---

## 11. 交付清单

### 11.1 目录结构

```
custom/sigmoid/
├── DESIGN.md            # 本设计文档
├── proto.yaml           # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── sigmoid.py           # 纯 kernel（@tilelang.jit，可 import，无 golden/测试/__main__）— Stage 2 产出
├── test_sigmoid.py      # from sigmoid import sigmoid + golden + 分层测试 + main — Stage 2 产出
└── README.md            # 使用说明（可选）— Stage 2 产出
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档（11 章 + L0 门槛测试计划） |
| `proto.yaml` | 已完成 | 算子接口规格（dtype 全集 float16+float32，attrs=[]），覆盖门禁 `coverage_check.py --proto` 用 |
| `sigmoid.py` | 待实现 | 纯 kernel（@tilelang.jit），Stage 2 产出 |
| `test_sigmoid.py` | 待实现 | `from sigmoid import sigmoid` + golden + L0 用例 + L1/L2/Boundary 桩 + main（`--level` 分发），Stage 2 产出 |

### 11.3 命名规范

- 目录名：`sigmoid`（snake_case）
- kernel 文件：`sigmoid.py`
- 测试文件：`test_sigmoid.py`（顶部 `from sigmoid import sigmoid`）
- kernel 函数名：`sigmoid`（与 `@tilelang.jit` 装饰的函数一致，可 import）

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ proto.yaml + L0 门槛测试计划（本文件 §9.2）
2. ⬜ kernel 实现（`sigmoid.py`，纯 @tilelang.jit，参考 §3.3 伪代码 + `examples/activation/sigmoid.py`）
3. ⬜ 测试文件（`test_sigmoid.py`）：`from sigmoid import sigmoid` + golden 函数 + L0 用例 + L1/L2/Boundary 桩 + main（`--level` 分发）
4. ⬜ L0 门槛测试通过（精度收敛，按 §9.3 精度标准）
5. ⬜ 扩展分层套件（L1 功能含不规则 shape / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）+ 覆盖门禁 `coverage_check.py` 全 PASS/N/A
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表**（float16 + float32）+ **§4/§1** 的 attr/shape 机械派生，是覆盖门禁 `coverage_check.py --proto` 的**权威 dtype/attr 来源**。checker 只读 `operator.inputs[].dtype` 与 `operator.attrs[].name`。

```yaml
operator:
  name: Sigmoid
  category: Activation
  formula: |
    sigmoid(x) = 1 / (1 + exp(-x))
  attrs: []                              # sigmoid 无影响计算路径的属性
  inputs:
    - name: x
      dtype: [float16, float32]          # 与 §9.3 精度表 dtype 行一致（全集）
  outputs:
    - name: y
      dtype: [float16, float32]          # 输出 dtype 与输入一致
  schema: sigmoid(Tensor x) -> Tensor y
```

> **一致性约束**：`inputs[].dtype` = `[float16, float32]` 与 §9.3 精度表的 dtype 行一致（全集）；`attrs` = `[]`（sigmoid 无 dim/axis/epsilon 等参数，无 D-PARAM-* 派生维度）。

## 12. 性能目标（Stage 3 追加）

> 本章节由 Orchestrator 在 Stage 3 启动前追加，不覆盖既有内容。

| 字段 | 值 |
|------|-----|
| 性能目标类型 | `baseline_compare` |
| Baseline | `torch.sigmoid`（PyTorch 在 NPU 上的实现） |
| 测试 shape | DESIGN.md §9.2 已有代表性 shape，默认 `(1024, 8192) float16` 作为主基准 + `(512, 512) float32` 辅助 |
| 噪声阈值 | 3%（perf-tuner 默认采纳门槛） |
| 最大迭代数 | 10 |
| 中止条件 | ① 迭代达 10 次；② 连续三次无性能提升；③ 相对 baseline 提升达标（建议 ≥ 1.0× 即不退化，> 1.0× 为优化） |
| 精度约束 | 性能调优过程中不得退化精度，每轮迭代后须回归 §9.3 精度标准（float16 atol=2⁻¹⁴ / float32 atol=2⁻¹⁶） |
| 当前 kernel | `custom/sigmoid/sigmoid.py`（采用 `T.tile.sigmoid` 一步原语，Developer 模式 + AUTO_CV_COMBINE/AUTO_SYNC/MEMORY_PLANNING） |

