# chunk_local_cumsum_scalar 算子设计文档

## 1. 概述

### 1.1 算子名称

chunk_local_cumsum_scalar (cumsum_gdn)

### 1.2 功能描述

分块局部累加和算子，用于 flash linear attention 中的 cumsum 操作。将输入序列按 chunk 分块，在每个 chunk 内计算累加和。

### 1.3 数学公式

正向 cumsum（reverse=False）：
$$
\text{output}[i] = \sum_{j=0}^{i} \text{input}[j], \quad \forall i \in \text{chunk}
$$

反向 cumsum（reverse=True）：
$$
\text{output}[i] = \text{total\_sum} - \text{prefix\_sum}[i] + \text{input}[i]
$$

其中 $\text{total\_sum} = \sum_{j=0}^{N-1} \text{input}[j]$，$N$ 为 chunk_size。

### 1.4 算法描述

1. 将输入张量按 sequence dimension 分成大小为 chunk_size 的块
2. 每个 kernel block 处理一个 chunk 的一个 batch-head
3. 在 chunk 内执行 cumsum 操作（正向或反向）
4. 输出张量形状与输入相同，每个 chunk 内的值为该 chunk 的局部累加和

### 1.5 数据流图

```
输入 G (B, H, L)
  ↓ T.copy (按 chunk 切片)
UB[g_ub] (chunk_size 元素)
  ↓ for循环 cumsum (+ reverse 转换)
UB[s_ub] (累加后)
  ↓ T.copy
输出 S (B, H, L)
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Expert 模式

### 2.2 选型理由

1. **T.cumsum API 后端不完整**: ascend 后端缺少 `tl.cumsum` intrinsic 的完整实现，需手写 for 循环
2. **需要手动同步**: `pass_configs={TL_ASCEND_AUTO_SYNC: True}` 必需
3. **精细内存控制**: 需显式 `T.alloc_ub` 分配 UB buffer
4. **reverse 功能需要额外处理**: 反向 cumsum 需计算总和后转换

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub` 显式分配 UB |
| 计算方式 | 手写 `for i in range(C)` 循环实现 cumsum |
| 作用域 | 显式 `T.Scope("V")` |
| 同步方式 | `pass_configs={TL_ASCEND_AUTO_SYNC: True}` |
| 初始化 | `T.tile.fill(buffer, 0.0)` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $g_{ub} = G[bs \times C : (bs+1) \times C]$ | 从 GM 加载一个 chunk 的数据到 UB |
| 2 | $s_{ub}[i] = s_{ub}[i-1] + g_{ub}[i]$ | 循环累加实现 cumsum |
| 3 | 若 reverse: $s_{ub}[i] = \text{total} - s_{ub}[i] + g_{ub}[i]$ | 反向转换 |
| 4 | $S[bs \times C] = s_{ub}$ | 将结果写回 GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 加载 chunk | `T.copy(G[...], g_ub)` | 索引计算 | Expert |
| 2 | 初始化 | `T.tile.fill(s_ub, 0.0)` | fill 值 0.0 | Expert |
| 3 | cumsum 循环 | `for i in range(C)` | Python range | Expert |
| 4 | reverse 转换 | 手写循环计算总和并转换 | `T.tile.fill(total_ub, 0.0)` | Expert |
| 5 | 写回结果 | `T.copy(s_ub, S[...])` | 累加结果 | Expert |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[-1], pass_configs={TL_ASCEND_AUTO_SYNC: True})
def cumsum_ker(B, H, L, C, reverse=False, head_first=True, use_fragment=False):
    chunk_num = T.ceildiv(L, C)
    VEC_NUM = 2
    shape = (B, H, L) if head_first else (B, L, H)

    @T.prim_func
    def main(G: T.Tensor(shape, "float"), S: T.Tensor(shape, "float")):
        with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
            bz = (cid // chunk_num) // (H // VEC_NUM)

            g_ub = T.alloc_ub([C], "float")
            s_ub = T.alloc_ub([C], "float")
            total_ub = T.alloc_ub([1], "float")
            if use_fragment:
                fragment_ub = T.alloc_ub([C], "float")

            with T.Scope("V"):
                T.tile.fill(s_ub, 0.0)
                
                if head_first:
                    T.copy(G[bz, by, bx * C], g_ub)
                else:
                    T.copy(G[bz, bx * C, by], g_ub)

                if use_fragment:
                    T.copy(g_ub, fragment_ub)
                    T.tile.fill(fragment_ub, 0.0)
                    for i in range(C):
                        if i > 0:
                            fragment_ub[i] = fragment_ub[i - 1]
                        fragment_ub[i] = fragment_ub[i] + g_ub[i]
                    if reverse:
                        T.tile.fill(total_ub, 0.0)
                        for i in range(C):
                            total_ub[0] = total_ub[0] + g_ub[i]
                        for i in range(C):
                            fragment_ub[i] = total_ub[0] - fragment_ub[i] + g_ub[i]
                    T.copy(fragment_ub, s_ub)
                else:
                    for i in range(C):
                        if i > 0:
                            s_ub[i] = s_ub[i - 1]
                        s_ub[i] = s_ub[i] + g_ub[i]
                    if reverse:
                        T.tile.fill(total_ub, 0.0)
                        for i in range(C):
                            total_ub[0] = total_ub[0] + g_ub[i]
                        for i in range(C):
                            s_ub[i] = total_ub[0] - s_ub[i] + g_ub[i]

                if head_first:
                    T.copy(s_ub, S[bz, by, bx * C])
                else:
                    T.copy(s_ub, S[bz, bx * C, by])

    return main
```

### 3.4 API 可行性确认

| API | 来源 | 状态 |
|-----|------|------|
| `T.copy` | 已验证 | ✅ 可用 |
| `T.alloc_ub` | 已验证 | ✅ 可用 |
| `T.tile.fill` | 已验证 | ✅ 可用 |
| `T.Scope("V")` | 已验证 | ✅ 可用 |
| `T.ceildiv` | tilelang/language/tir/op.py | ✅ 可用 |
| `T.cumsum` | tilelang/language/reduce.py | ❌ **后端不完整**，需手写循环 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| G | (B, H, L) | float32 | 输入张量，head_first=True 模式 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| S | (B, H, L) | float32 | 输出张量，shape 与输入相同 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| g_ub | (C,) | float32 | UB | chunk 数据缓冲 |
| s_ub | (C,) | float32 | UB | cumsum 结果缓冲 |
| total_ub | (1,) | float32 | UB | reverse 时存储总和 |
| fragment_ub | (C,) | float32 | UB | use_fragment 时的中间缓冲 |

### 4.4 内存搬运路径

```
纯 Vector 算子：

GM[G] --T.copy--> UB[g_ub]
  --for循环--> UB[s_ub] (或 UB[fragment_ub] --T.copy--> UB[s_ub])
  --T.copy--> GM[S]
```

### 4.5 UB 内存预算

| Buffer | Shape C=32 | Shape C=64 | dtype | 大小 |
|--------|-----------|-----------|-------|------|
| g_ub | (32,) | (64,) | float32 | 128B / 256B |
| s_ub | (32,) | (64,) | float32 | 128B / 256B |
| total_ub | (1,) | (1,) | float32 | 4B |
| fragment_ub | (32,) | (64,) | float32 | 128B / 256B |
| **总计** | | | | ~400B / ~800B << 128KB ✓ |

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅包含 cumsum 操作，无 matmul，仅涉及 Vector 核 UB。

### 5.2 Block 划分

```python
C = chunk_size  # 每个 block 处理一个 chunk
chunk_num = T.ceildiv(L, C)
VEC_NUM = 2  # 每个 vector 核处理部分 head

block_num = chunk_num * B * (H // VEC_NUM)
# 每个 block 处理: 一个 chunk 的一个 batch-head
```

### 5.3 约束分析

- **chunk_size 必须是 2 的幂**: `assert chunk_size == 2 ** (chunk_size.bit_length() - 1)`
- **UB 容量**: 所有 buffer << 128KB ✓
- **单维 Kernel**: ascend `T.Kernel` 只支持单一 block 维度

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| chunk | Block 级并行 | `T.Kernel(chunk_num * B * (H//VEC_NUM))` | 每个 block 处理一个 chunk |
| chunk 内元素 | Python for | `for i in range(C)` | cumsum 需顺序依赖 |
| reverse 总和计算 | Python for | `for i in range(C)` | 计算总和 |
| reverse 转换 | Python for | `for i in range(C)` | 转换为反向 cumsum |

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（pass_configs）

### 7.2 同步点说明

Expert 模式 + `TL_ASCEND_AUTO_SYNC: True`：
- `T.copy` 后自动同步
- `T.tile.fill` 后自动同步
- 循环内无需手动 `T.barrier_all`

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def ref_chunk_cumsum(g, C, reverse=False, head_first=True):
    B, H, L = g.shape
    chunk_num = (L + C - 1) // C
    g = g.view(B, H, chunk_num, C)
    if reverse:
        g_sum = torch.flip(torch.cumsum(torch.flip(g, dims=[3]), dim=3), dims=[3])
    else:
        g_sum = torch.cumsum(g, dim=-1)
    g_sum = g_sum.view(B, H, L)
    return g_sum
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | C | reverse | use_fragment | 说明 |
|--------|------|-------|---|---------|--------------|------|
| basic_fwd | Level 0 | (2, 32, 256) | 32 | False | False | 正向最小验证 |
| basic_rev | Level 0 | (2, 32, 256) | 32 | True | False | 反向验证 |
| basic_fragment | Level 0 | (2, 32, 256) | 32 | False | True | use_fragment 验证 |
| reverse_fragment | Level 1 | (2, 32, 256) | 32 | True | True | reverse + fragment |
| typical_fwd | Level 1 | (1, 16, 128) | 64 | False | False | 典型配置 |
| typical_rev | Level 1 | (1, 16, 128) | 64 | True | True | 典型反向 |
| large_fwd | Level 2 | (4, 8, 512) | 64 | False | False | 大规模 |
| large_rev | Level 2 | (4, 8, 512) | 64 | True | True | 大规模反向 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |

---

## 9. 功能特性总结

### 9.1 与主仓一致性对比

| 功能 | tilelang 主仓 | tilelang-ascend | 状态 |
|------|--------------|-----------------|------|
| **reverse 参数** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **use_fragment 参数** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| **head_first=True** | ✅ 支持 | ✅ 支持 | ✅ 一致 |
| head_first=False | ✅ 支持 | ❌ stride限制 | 平台限制 |
| T.cumsum API | ✅ 内置 | ❌ 手写循环 | 平台限制 |

### 9.2 实现要点

1. **reverse 实现**: 通过计算 chunk 总和，然后 `total - prefix + input` 转换
2. **use_fragment 实现**: 在 fragment buffer 中计算 cumsum，再 copy 到 s_ub
3. **循环内累加**: 必须使用 buffer（`T.alloc_ub`）而非 Python scalar

---

## 10. 交付清单

### 10.1 目录结构

```
examples/cumsum_gdn/
├── example_cumsum.py     # 算子实现 + 测试 ✅
├── design.md             # 本设计文档 ✅
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | ✅ 已完成 | 设计文档 |
| `example_cumsum.py` | ✅ 已完成 | 算子实现 + 测试 |

### 10.3 实现状态

| 功能 | 状态 |
|------|------|
| 正向 cumsum | ✅ 已实现 |
| reverse cumsum | ✅ 已实现 (P0) |
| use_fragment | ✅ 已实现 (P2) |
| head_first=True | ✅ 已实现 |
| head_first=False | ❌ 平台限制 |

---

## 附录

### A. 实际代码关键片段

```python
# reverse 功能实现 (P0)
if reverse:
    T.tile.fill(total_ub, 0.0)
    for i in range(C):
        total_ub[0] = total_ub[0] + g_ub[i]
    for i in range(C):
        s_ub[i] = total_ub[0] - s_ub[i] + g_ub[i]

# use_fragment 功能实现 (P2)
if use_fragment:
    T.copy(g_ub, fragment_ub)
    T.tile.fill(fragment_ub, 0.0)
    for i in range(C):
        if i > 0:
            fragment_ub[i] = fragment_ub[i - 1]
        fragment_ub[i] = fragment_ub[i] + g_ub[i]
    T.copy(fragment_ub, s_ub)
```

### B. 测试结果

```
=== Testing chunk_cumsum (cumsum_gdn) - P0: reverse ===
6 个测试配置全部 Passed!

=== Testing chunk_cumsum with use_fragment (P2) ===
6 个测试配置全部 Passed! (含 use_fragment=True 和 False)

=== All tests passed! ===
```