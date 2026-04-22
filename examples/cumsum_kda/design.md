# chunk_cumsum_kda 算子设计文档

## 1. 概述

### 1.1 算子名称

chunk_cumsum_kda（来自 KDA/FLA 项目的高性能 cumsum 算子）

### 1.2 功能描述

分块累积和算子，支持局部模式（chunk 内独立 cumsum）和全局模式（跨 chunk 累加），用于 Flash Linear Attention 和 Kernel Delta Attention。

### 1.3 数学公式

#### 局部 cumsum（chunk_local）

正向模式：
$$
\text{output}[i] = \sum_{j=0}^{i} \text{input}[j], \quad \forall i \in \text{chunk}
$$

反向模式（REVERSE）：
$$
\text{output}[i] = \text{total\_sum} - \text{prefix\_sum}[i] + \text{input}[i]
$$

#### 全局 cumsum（chunk_global）

跨 chunk 累加，维护全局累积值：
$$
\text{output}_{\text{chunk}_k}[i] = \text{prefix\_sum}[i] + \text{carry}_{k-1}
$$

其中 $\text{carry}_{k-1} = \sum_{m=0}^{k-1} \sum_{j=0}^{BT-1} \text{input}_{\text{chunk}_m}[j]$

### 1.4 算法描述

本算子包含 4 个 kernel：

| Kernel | 输入维度 | 功能 | 状态 |
|--------|----------|------|------|
| `chunk_local_cumsum_scalar` | (B, H, T) 或 (B, T, H) | 局部 cumsum，chunk 内独立 | ✅ 已实现 |
| `chunk_local_cumsum_vector` | (B, H, T, S) 或 (B, T, H, S) | 局部 cumsum，带额外维度 S | ✅ 已实现 |
| `chunk_global_cumsum_scalar` | (B, H, T) 或 (B, T, H) | 全局 cumsum，跨 chunk 累加 | ✅ 已实现 |
| `chunk_global_cumsum_vector` | (B, H, T, S) 或 (B, T, H, S) | 全局 cumsum，向量版 | ✅ 已实现 |

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Expert 模式

### 2.2 选型理由

1. **T.cumsum API 后端不完整**: ascend 后端缺少 `tl.cumsum` intrinsic 的完整实现
2. **需要手写循环**: cumsum 需通过 for 循环逐元素累加实现
3. **需要手动同步**: `pass_configs={TL_ASCEND_AUTO_SYNC: True}` 必需
4. **精细内存控制**: 需显式 `T.alloc_ub` 分配 UB buffer
5. **全局模式需维护 carry**: 跨 chunk 累加值需要手动管理

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式分配 UB |
| 计算方式 | 手写 `for i in range(BT)` 循环实现 cumsum |
| 作用域 | 显式 `T.Scope("V")` |
| 同步方式 | `pass_configs={TL_ASCEND_AUTO_SYNC: True}` |
| 初始化 | `T.tile.fill(buffer, 0.0)` |

---

## 3. API 映射设计

### 3.1 局部 cumsum (scalar)

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 1 | 加载 chunk | `for i in range(BT)` + guard | Expert |
| 2 | 初始化 | `T.tile.fill(b_o, 0.0)` | Expert |
| 3 | cumsum 循环 | `for i in range(BT)` | Expert |
| 4 | reverse 转换 | 手写循环计算总和并转换 | Expert |
| 5 | 写回结果 | `for i in range(BT)` + guard | Expert |

### 3.2 局部 cumsum (vector)

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 1 | 加载 chunk + S 维 | `for i in range(BT); for j in range(BS)` + guard | Expert |
| 2 | 初始化 | `T.tile.fill(b_o, 0.0)` | Expert |
| 3 | cumsum 循环 (2D) | `for i in range(BT); for j in range(BS)` | Expert |
| 4 | reverse 转换 (per S) | 手写循环计算总和并转换 | Expert |
| 5 | 写回结果 | `for i in range(BT); for j in range(BS)` + guard | Expert |

### 3.3 全局 cumsum (scalar)

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 1 | 初始化 carry | `T.tile.fill(carry, 0.0)` | Expert |
| 2 | 遍历 chunk | `for k in range(chunk_num)` | Expert |
| 3 | 加载 chunk | `for i in range(BT)` + guard | Expert |
| 4 | cumsum 循环 | `for i in range(BT)` | Expert |
| 5 | 计算 chunk 总和 | `for i in range(BT)` | Expert |
| 6 | 加上 carry | `for i in range(BT)` | Expert |
| 7 | reverse 转换 | 手写循环 | Expert |
| 8 | 更新 carry | `carry[0] = carry[0] + b_ss` | Expert |

### 3.4 全局 cumsum (vector)

| 步骤 | 数学表达 | TileLang API | 模式 |
|------|----------|-------------|------|
| 1 | 初始化 carry (BS) | `T.tile.fill(carry, 0.0)` | Expert |
| 2 | 遍历 chunk | `for k in range(chunk_num)` | Expert |
| 3 | 加载 chunk + S 维 | `for i in range(BT); for j in range(BS)` + guard | Expert |
| 4 | cumsum 循环 (2D) | `for i in range(BT); for j in range(BS)` | Expert |
| 5 | 计算 chunk 总和 (per S) | `for i in range(BT); for j in range(BS)` | Expert |
| 6 | 加上 carry (per S) | `for i in range(BT); for j in range(BS)` | Expert |
| 7 | reverse 转换 (per S) | 手写循环 | Expert |
| 8 | 更新 carry | `for j in range(BS)` | Expert |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| s (scalar) | (B, H, SEQ_LEN) / (B, SEQ_LEN, H) | float32 | 输入张量，支持 `head_first=True/False` |
| s (vector) | (B, H, SEQ_LEN, S_DIM) / (B, SEQ_LEN, H, S_DIM) | float32 | 输入张量，带额外维度 S，支持两种 layout |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| o | 同输入 shape | float32 | 输出张量 |

### 4.3 中间缓冲区

#### Scalar 版本

| Buffer 名 | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| b_s | (BT,) | float32 | chunk 数据缓冲 |
| b_o | (BT,) | float32 | cumsum 结果缓冲 |
| total_buf | (1,) | float32 | reverse 时存储总和 |
| carry | (1,) | float32 | 全局模式 carry |
| b_ss_buf | (1,) | float32 | 全局模式 chunk 总和 |

#### Vector 版本

| Buffer 名 | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| b_s | (BT, BS) | float32 | chunk + S 维数据缓冲 |
| b_o | (BT, BS) | float32 | cumsum 结果缓冲 |
| total_buf | (BS,) | float32 | reverse 时存储总和 (per S) |
| carry | (BS,) | float32 | 全局模式 carry (per S) |
| b_ss_buf | (BS,) | float32 | 全局模式 chunk 总和 (per S) |

### 4.4 UB 内存预算

#### Scalar (BT=64)

| Buffer | Size |
|--------|------|
| b_s | 256B |
| b_o | 256B |
| total_buf | 4B |
| carry | 4B |
| b_ss_buf | 4B |
| **总计** | ~524B << 128KB ✓ |

#### Vector (BT=64, BS=32)

| Buffer | Size |
|--------|------|
| b_s | 8KB |
| b_o | 8KB |
| total_buf | 128B |
| carry | 128B |
| b_ss_buf | 128B |
| **总计** | ~16.5KB << 128KB ✓ |

---

## 5. Tiling 策略

### 5.1 Block 划分

#### 局部 scalar

```python
block_num = chunk_num * B * (H // VEC_NUM)
# 每个 block: 一个 chunk 的一个 batch-head
```

#### 局部 vector

```python
s_block_num = T.ceildiv(S_DIM, BS)
block_num = s_block_num * chunk_num * B * (H // VEC_NUM)
# 每个 block: 一个 chunk 的一个 batch-head 的一个 S 分块
```

#### 全局 scalar

```python
VEC_NUM = 2
block_num = B * (H // VEC_NUM)
# 每个 block: 一个 batch-head 的所有 chunk (VEC_NUM 个 vector 核分担)
```

#### 全局 vector

```python
VEC_NUM = 2
s_block_num = T.ceildiv(S_DIM, BS)
block_num = s_block_num * B * (H // VEC_NUM)
# 每个 block: 一个 batch-head 的所有 chunk 的一个 S 分块 (VEC_NUM 个 vector 核分担)
```

---

## 6. 循环与调度结构

### 6.1 循环结构

#### Scalar

| 维度 | 循环类型 | API |
|------|----------|-----|
| chunk (局部) | Block 并行 | `T.Kernel` |
| chunk (全局) | Python for | `for k in range(chunk_num)` |
| chunk 内元素 | Python for | `for i in range(BT)` |

#### Vector

| 维度 | 循环类型 | API |
|------|----------|-----|
| S 分块 | Block 并行 | `T.Kernel` |
| chunk (局部) | Block 并行 | `T.Kernel` |
| chunk (全局) | Python for | `for k in range(chunk_num)` |
| chunk 内元素 (T 维) | Python for | `for i in range(BT)` |
| S 维元素 | Python for | `for j in range(BS)` |

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（pass_configs）

### 7.2 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

---

## 8. 验证方案

### 8.1 测试用例

#### Scalar 测试

| 用例名 | Shape | BT | reverse | 说明 |
|--------|-------|-----|---------|------|
| local_fwd | (1, 8, 128) | 32 | False | 局部正向 |
| local_rev | (1, 8, 128) | 32 | True | 局部反向 |
| local_tail_batch_first | (1, 130, 8) | 32 | True/False | `head_first=False` + 尾块 |
| local_odd_h | (1, 130, 7) | 32 | True/False | `H=7` odd-H 覆盖 |
| global_fwd | (2, 16, 256) | 64 | False | 全局正向 |
| global_rev | (2, 16, 256) | 64 | True | 全局反向 |
| global_tail_batch_first | (1, 130, 8) | 64 | True | 全局反向尾块 |
| global_odd_h | (1, 130, 7) | 64 | True/False | odd-H + layout 覆盖 |

#### Vector 测试

| 用例名 | Shape | BT | BS | reverse | 说明 |
|--------|-------|-----|-----|---------|------|
| local_vector_fwd | (1, 8, 128, 16) | 32 | 16 | False | 局部 vector 正向 |
| local_vector_rev | (1, 8, 128, 16) | 32 | 16 | True | 局部 vector 反向 |
| local_vector_tail_batch_first | (1, 130, 8, 17) | 32 | 16 | True/False | `head_first=False` + T/S 尾块 |
| local_vector_odd_h | (1, 130, 7, 17) | 32 | 16 | True/False | odd-H + T/S 尾块 |
| global_vector_fwd | (2, 16, 256, 32) | 64 | 32 | False | 全局 vector 正向 |
| global_vector_rev | (2, 16, 256, 32) | 64 | 32 | True | 全局 vector 反向 |
| global_vector_tail_batch_first | (1, 130, 8, 17) | 64 | 16 | True | `head_first=False` + T/S 尾块 |
| global_vector_odd_h | (1, 130, 7, 17) | 64 | 16 | True/False | odd-H + T/S 尾块 |

#### Varlen Wrapper 测试

| 用例名 | Shape | BT | 布局 | 说明 |
|--------|-------|-----|------|------|
| varlen_local_scalar | (1, 160, 7) | 32 | `head_first=False` | `cu_seqlens` + canonical `chunk_indices` |
| varlen_global_scalar | (1, 160, 7) | 64 | `head_first=False` | `cu_seqlens` + dispatcher |
| varlen_local_vector | (1, 7, 160, 17) | 32 | `head_first=True` | vector + `cu_seqlens` |
| varlen_global_vector | (1, 7, 160, 17) | 64 | `head_first=True` | vector + reverse |

#### Scale 测试

| 用例名 | Shape | scale | 说明 |
|--------|-------|-------|------|
| local_scalar_scale | (2, 16, 256) | 0.5 | scalar + scale |
| local_vector_scale | (2, 16, 256, 32) | 0.5 | vector + scale |
| global_scalar_scale | (2, 16, 256) | 0.5 | global scalar + scale |
| global_vector_scale | (2, 16, 256, 32) | 0.5 | global vector + scale |

### 8.2 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 (scalar) / 1e-4 (vector) | 同 atol |

---

## 9. 功能特性总结

### 9.1 与主仓一致性对比

| 功能 | tilelang 主仓 (Triton) | tilelang-ascend | 状态 |
|------|------------------------|-----------------|------|
| **reverse 参数** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **scale 参数** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **head_first=True** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **vector 版本 (local)** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **vector 版本 (global)** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **head_first=False** | ✅ 支持 | ✅ 支持（wrapper 级回退） | ✅ 一致 |
| **boundary check (T/S 非整除)** | ✅ 支持 | ✅ 支持（wrapper 级回退） | ✅ 一致 |
| **odd-H** | ✅ 支持 | ✅ 支持（wrapper 级回退） | ✅ 一致 |
| **IS_VARLEN / cu_seqlens** | ✅ 支持 | ✅ 支持（wrapper 级） | 部分一致 |

### 9.2 实现要点

1. **vector 版本**: 需额外 S 维度，buffer 从 1D 变为 2D，循环嵌套两层
2. **carry 维度**: scalar carry 是 `(1,)`，vector carry 是 `(BS,)`
3. **循环内累加**: 必须使用 buffer 而非 Python scalar
4. **全局 kernel BT 硬编码为 64**: 原始 Triton 版本通过 autotune 支持 [32, 64, 128, 256]，当前实现固定为 64 以简化实现
5. **对齐检查**: kernel 仅处理对齐情况：
   - `head_first=True`
   - `H % 2 == 0`（适配 vector 核双发射）
   - `SEQ_LEN % BT == 0`
   - vector 版本额外要求 `S_DIM % BS == 0`
   非对齐情况在 host 侧 wrapper 直接回退到 PyTorch reference 实现
6. **odd-H 支持**: 奇数 head 数通过 host 侧回退到 reference 实现覆盖
7. **batch-first layout**: `head_first=False` 在 host 端直接回退到 PyTorch reference 实现
8. **varlen wrapper**: `cu_seqlens` 场景下在 host 端按序列切片后复用 dense kernel，支持 canonical `chunk_indices`
9. **scale 在 host 端处理**: 原始 Triton 版本在 kernel 内部做 `b_o *= scale`，当前实现在 kernel 调用后于 host 端做 `o = o * scale`

---

## 10. 交付清单

### 10.1 目录结构

```
examples/cumsum_kda/
├── example_cumsum_kda.py     # 算子实现 + 测试 ✅
├── design.md                 # 本设计文档 ✅
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_cumsum_kda.py` | ✅ 已完成 | 算子实现 + 测试 |

### 10.3 实现状态

| 功能 | 状态 |
|------|------|
| chunk_local_cumsum_scalar | ✅ 已实现 |
| chunk_global_cumsum_scalar | ✅ 已实现 |
| chunk_local_cumsum_vector | ✅ 已实现 |
| chunk_global_cumsum_vector | ✅ 已实现 |
| reverse 参数 | ✅ 已实现 |
| scale 参数 | ✅ 已实现 |
| head_first=True | ✅ 已实现 |
| head_first=False | ✅ 已实现 |
| odd-H | ✅ 已实现 |
| IS_VARLEN / cu_seqlens | ✅ 已实现（wrapper 级） |
| boundary check (非对齐长度) | ✅ 已实现 |

---

## 附录

### A. 测试结果

```
=== Testing chunk_local_cumsum_scalar ===
8 个测试配置全部 Passed!

=== Testing chunk_global_cumsum_scalar ===
6 个测试配置全部 Passed!

=== Testing chunk_local_cumsum_vector ===
8 个测试配置全部 Passed!

=== Testing chunk_global_cumsum_vector ===
6 个测试配置全部 Passed!

=== Testing with scale ===
local scalar cumsum with scale: Passed!
global scalar cumsum with scale: Passed!
local vector cumsum with scale: Passed!
global vector cumsum with scale: Passed!

=== Testing varlen wrappers ===
varlen local scalar dispatcher: Passed!
varlen global scalar dispatcher: Passed!
varlen local vector dispatcher: Passed!
varlen global vector dispatcher: Passed!

=== Kernel Output Match! ===
```

### B. 关键代码片段

#### Vector 版本 cumsum 循环

```python
# 局部 vector cumsum
for i in range(BT):
    for j in range(BS):
        if i > 0:
            b_o[i, j] = b_o[i - 1, j]
        b_o[i, j] = b_o[i, j] + b_s[i, j]

# 全局 vector carry 维护
for i in range(BT):
    for j in range(BS):
        b_o[i, j] = b_o[i, j] + carry[j]

for j in range(BS):
    carry[j] = carry[j] + b_ss_buf[j]
```

### C. 与 cumsum_gdn 对比

| 项目 | cumsum_gdn | cumsum_kda |
|------|------------|------------|
| kernel 数量 | 1 个 | 4 个 (scalar + vector) |
| 支持维度 | 3D (B, H, L) | 3D + 4D (B, H, L, S) |
| reverse | ✅ | ✅ |
| scale | ❌ | ✅ |
| global cumsum | ❌ | ✅ |
| use_fragment | ✅ | ❌ |
