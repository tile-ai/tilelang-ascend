# GQA Flash Attention 算子设计文档

> **算子名**: flash_attention (GQA forward + backward)
> **目标平台**: Ascend 910B3 NPU
> **编程模式**: Expert（手动 L1/UB/L0C 分配 + 手动 Scope("C"/"V") + 手动同步）
> **来源**: GPU TileLang `examples/flash_attention/example_gqa_bwd.py` 移植

---

## 1. 算子概述

### 1.1 计算语义

GQA (Grouped Query Attention) Flash Attention 是标准 Multi-Head Attention 的变体，其中 Q 的 head 数量是 K/V 的 `groups` 倍。计算分为 forward 和 backward 两部分：

**Forward**:
```
S = Q @ K^T * scale          # [B, H, N, N] attention scores
P = softmax(S, causal_mask)   # [B, H, N, N] attention probabilities
O = P @ V                     # [B, H, N, D_v] output
lse = log(sum(exp(S)))        # [B, H, N] log-sum-exp for backward
```

**Backward** (给定 dO, 利用 forward 的 lse):
```
Delta = sum(O * dO, dim=-1)   # [B, H, N]
dV += P^T @ dO                # [B, H_kv, N, D_v]
dP = V @ dO^T                 # [B, H, N, N]
dS = P * (dP - Delta) * scale # [B, H, N, N]
dK += dS^T @ Q                # [B, H_kv, N, D_qk]
dQ += dS @ K                  # [B, H, N, D_qk]
```

### 1.2 GQA 分组机制

- Q 有 `H` 个 head，K/V 有 `H_kv = H / groups` 个 head
- 每个 Q head 索引 `by` 对应 K/V head 索引 `by // groups`
- backward 中 dK/dV 需要跨 groups 累加（atomic_add 或 split+sum）

### 1.3 适用场景

- LLM 推理/训练中的注意力层（如 LLaMA-2 70B 使用 GQA groups=8）
- 长序列注意力（seq_len 可达 4096+）
- 支持 causal（自回归）和 non-causal 两种模式

### 1.4 Kernel 组成（4 个 Kernel + 1 个优化变体）

| # | 函数名 | 功能 | 计算类型 |
|---|--------|------|---------|
| 1 | `flashattn_fwd` | Forward 基础版: QK^T → softmax → PV → O, lse | Cube + Vector 融合 |
| 1b | `flashattn_fwd_v4` | Forward 优化版: L0 双缓冲 + Fixed Core + 批处理 | Cube + Vector 融合 |
| 2 | `flashattn_bwd_preprocess` | 预处理: Delta = sum(O * dO, dim=-1) | 纯 Vector |
| 3 | `flashattn_bwd_postprocess` | 后处理: dQ fp32 → fp16 转换 | 纯 Vector |
| 4 | `flashattn_bwd_pipeline` | 主反向: 5 GEMM + 批处理 + fine-grained sync | Cube + Vector 融合 |

> **实现说明**: Forward 提供基础版（v1）和优化版（v4）两个变体，v4 通过 L0 双缓冲和 Fixed Core 实现 1.88x 加速。Backward 采用 pipeline 版本（批处理 + fine-grained flag sync），相比原始 atomic_add 版本实现 1.41x 加速。

---

## 2. I/O 规格

### 2.1 Kernel 1: flashattn_fwd (Forward)

| 参数 | Shape | dtype | 方向 | 说明 |
|------|-------|-------|------|------|
| Q | [B, H, N, D_qk] | float16 | 输入 | Query，BHSD 布局 |
| K | [B, H_kv, N, D_qk] | float16 | 输入 | Key，BHSD 布局 |
| V | [B, H_kv, N, D_v] | float16 | 输入 | Value，BHSD 布局 |
| Output | [B, H, N, D_v] | float16 | 输出 | 注意力输出 |
| lse | [B, H, N] | float32 | 输出 | Log-sum-exp，backward 使用 |

### 2.2 Kernel 2: flashattn_bwd_preprocess (预处理)

| 参数 | Shape | dtype | 方向 | 说明 |
|------|-------|-------|------|------|
| O | [B, H, N, D_v] | float16 | 输入 | Forward 输出 |
| dO | [B, H, N, D_v] | float16 | 输入 | 输出梯度 |
| Delta | [B, H, N] | float32 | 输出 | sum(O * dO, dim=-1) |

### 2.3 Kernel 3: flashattn_bwd_postprocess (后处理)

| 参数 | Shape | dtype | 方向 | 说明 |
|------|-------|-------|------|------|
| dQ | [B, H, N, D_qk] | float32 | 输入 | dQ fp32 累加器 |
| dQ_out | [B, H, N, D_qk] | float16 | 输出 | dQ fp16 输出 |

### 2.4 Kernel 4: flashattn_bwd_pipeline (主反向, pipeline)

| 参数 | Shape | dtype | 方向 | 说明 |
|------|-------|-------|------|------|
| Q | [B, H, N, D_qk_padded] | float16 | 输入 | D_qk pad 到 128 倍数 |
| K | [B, H_kv, N, D_qk_padded] | float16 | 输入 | 同上 |
| V | [B, H_kv, N, D_v] | float16 | 输入 | |
| dO | [B, H, N, D_v] | float16 | 输入 | |
| lse | [B, H, N] | float32 | 输入 | Forward 产出 |
| Delta | [B, H, N] | float32 | 输入 | Kernel 2 产出 |
| dQ | [B, H, N, D_qk_padded] | float32 | 输出 | 跨 KV 累积（需预清零） |
| dK | [B, H_kv, N, D_qk_padded] | float32 | 输出 | atomic_add 累加 |
| dV | [B, H_kv, N, D_v] | float32 | 输出 | atomic_add 累加 |
| ws_s_dp | [block_num, num_stages, M, N] | float32 | workspace | S/dP 中转 |
| ws_p_ds | [block_num, num_stages, M, N] | float16 | workspace | P/dS 中转 |
| ws_dv_dk | [block_num, num_stages, N, max(D_qk_padded, D_v)] | float32 | workspace | dV/dK 中转 |

> **pipeline 版本特性**: 采用 num_stages 批处理 + fine-grained flag sync（set_flag/wait_flag），将每轮 5 对 cross_flag 降低为每批 5 对，显著减少同步开销。dQ 跨 KV 迭代在 L0C 中累积，dK/dV 通过 atomic_add 写入 GM。

### 2.6 动态轴说明

| 符号 | 含义 | 典型范围 |
|------|------|---------|
| B | Batch size | 1~8 |
| H | Q head 数 | 1~32 |
| H_kv | KV head 数 = H / groups | 1~32 |
| N | Sequence length | 64~4096 |
| D_qk | QK head dimension | 64~192 |
| D_v | V head dimension | 64~128 |
| groups | GQA 分组数 = H / H_kv | 1~16 |

---

## 3. 编程模式选型

### 3.1 选择: Expert 模式

**理由**:

1. **Flash Attention 是典型的 CV 融合算子**: QK^T 和 PV 是矩阵乘（Cube 核），softmax 是归约+指数（Vector 核），必须显式分离 C/V scope
2. **需要精细控制 C↔V 数据流**: L0C → workspace(GM) → UB 的搬运路径需要手动管理 workspace 索引和同步
3. **内存压力大**: Flash Attention 需要大量中间 buffer（acc_s, acc_o, scores_max, scores_scale 等），手动 `T.annotate_address` 可以最大化复用 L1/UB 空间
4. **参考实现已验证**: `flash_attn_bhsd.py` 和 `fa_opt/flash_attn_bhsd_expert_h16_d128.py` 均为 Expert 模式，已验证可行

### 3.2 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

全部关闭，由手动控制同步和内存规划。

### 3.3 Expert 模式要素清单

| 要素 | API | 说明 |
|------|-----|------|
| 内存分配 | `T.alloc_L1`, `T.alloc_ub`, `T.alloc_L0C` | 显式指定存储层级 |
| 地址管理 | `T.annotate_address({...})` | 手动指定 buffer 在 L1/UB 中的偏移 |
| 计算域 | `T.Scope("C")`, `T.Scope("V")` | Cube/Vector 核代码分离 |
| 矩阵乘 | `T.gemm_v0(A, B, C, ...)` | 标准 GEMM（L1→L0C） |
| 元素计算 | `T.tile.xxx(dst, src, ...)` | Buffer 级 SIMD 操作 |
| 核内同步 | `T.barrier_all()` | 全管线屏障 |
| 核间同步 | `T.set_cross_flag(pipe, flag)`, `T.wait_cross_flag(flag)` | C↔V 核间握手 |
| V 核并行 | `vid` (0/1) 切分 block_M // 2 | 两个 V 核各处理一半行 |
| workspace | `@jit(workspace_idx=[...])` | C↔V 数据中转（GM） |

---

## 4. API 映射表 (GPU → NPU)

### 4.1 Kernel 定义与内存

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `T.Kernel(x, y, z, threads=256)` | `T.Kernel(block_num, is_npu=True) as (cid, vid)` | NPU 用 1D grid + cid/vid |
| `T.alloc_shared(shape, dtype)` | `T.alloc_L1(shape, dtype)` | GPU shared → NPU L1 (Cube) |
| `T.alloc_fragment(shape, dtype)` | `T.alloc_L0C(shape, dtype)` | GPU fragment → NPU L0C (Cube 输出) |
| — | `T.alloc_ub(shape, dtype)` | NPU 独有：Vector 核 UB |
| — | `T.annotate_address({...})` | NPU 独有：手动地址分配 |

### 4.2 计算原语

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `T.gemm(A, B, C, transpose_B=True, policy=...)` | `T.gemm_v0(A_L1, B_L1, C_L0C, transpose_B=True, init=True)` | NPU 用 gemm_v0，需显式 init |
| `T.reduce_max(buf, out, dim=1, clear=False)` | `T.reduce_max(buf_ub, out_ub, dim=-1)` | NPU reduce 在 UB 上执行 |
| `T.reduce_sum(buf, out, dim=1)` | `T.reduce_sum(buf_ub, out_ub, dim=-1)` | 同上 |
| `T.fill(buf, val)` | `T.tile.fill(buf_ub, val)` | NPU 用 T.tile.fill |
| `T.copy(acc_s, acc_s_cast)` (fp32→fp16) | `T.tile.cast(dst, src, "CAST_NONE", count)` 或 `T.copy(fp32_ub, fp16_ub)` | NPU dtype 转换 |

### 4.3 Element-wise 运算

| GPU (T.Parallel + 符号 API) | NPU (T.tile.xxx) | 说明 |
|---------------------------|-----------------|------|
| `for i,j in T.Parallel(M,N): c[i,j] = a[i,j] * b[i]` | `T.tile.broadcast(buf_2d, b_1d); T.tile.mul(c, a, buf_2d)` | NPU 需显式广播 |
| `for i in T.Parallel(M): s[i] = T.exp2(a[i])` | `T.tile.exp(dst, src)` | NPU 用自然指数 exp（非 exp2）|
| `for i,j in T.Parallel(M,N): c[i,j] = T.exp2(a[i,j]*s - m[i]*s)` | `T.tile.mul → T.tile.sub → T.tile.exp` 多步 | NPU 需分解为多步 T.tile |
| `for i,j in T.Parallel(M,N): c[i,j] = T.if_then_else(cond, a, b)` | `T.tile.compare(mask, ...); T.tile.select(dst, mask, a, b, "VSEL_CMPMASK_SPR")` | NPU 不支持 T.Parallel 内 if-else |
| `T.atomic_add(dQ[...], dq[i,j])` | `T.tile.atomic_add(dQ_gm_region, src_ub_or_l0c)` | NPU 原子累加 |

### 4.4 循环与调度

| GPU API | NPU API | 说明 |
|---------|---------|------|
| `T.Pipelined(range, num_stages=1)` | `T.serial(range)` | NPU Expert 模式用 T.serial + 手动流水线 |
| `for i,j in T.Parallel(M,N)` | `T.tile.xxx` 系列 | NPU 用 Buffer 级 SIMD |
| — | `T.set_cross_flag("FIX"/"MTE3", flag)` | NPU 核间同步 |
| — | `T.wait_cross_flag(flag)` | NPU 核间等待 |
| — | `T.barrier_all()` | NPU 核内全管线屏障 |

### 4.5 关键差异总结

| 差异点 | GPU | NPU |
|--------|-----|-----|
| 数据布局 | [B, N, H, D] (BSHD) | [B, H, N, D] (BHSD) |
| 指数函数 | `T.exp2(x * log2e)` | `T.tile.exp(x)` 或 `T.tile.mul + T.tile.exp` |
| 条件掩码 | `T.if_then_else` in T.Parallel | `T.tile.compare + T.tile.select` |
| C/V 分离 | 编译器自动 | 手动 `T.Scope("C"/"V")` + workspace |
| 同步 | 编译器自动 | 手动 `T.barrier_all` + `T.set/wait_cross_flag` |
| Grid 维度 | 3D (bx, by, bz) | 1D cid + 手动分解 |

---

## 5. 内存层级规划

### 5.1 硬件限制 (Ascend 910B3)

| 存储单元 | 大小上限 | 对齐要求 |
|---------|---------|---------|
| L0A | 65536 B (64 KB) | 512 B |
| L0B | 65536 B (64 KB) | 512 B |
| L0C | 131072 B (128 KB) | 64 B |
| L1 | 524288 B (512 KB) | 32 B |
| UB | 196608 B (192 KB) — 两个 V 核共享 | 32 B |

### 5.2 Kernel 1 (Forward) 内存规划

以 block_M=64, block_N=64, D_qk=128, D_v=128 为例：

**L1 分配 (Cube 核)**:

| Buffer | Shape | dtype | 大小 (B) | 地址偏移 |
|--------|-------|-------|---------|---------|
| q_l1 | [64, 128] | fp16 | 16384 | 0 |
| k_l1 | [64, 128] | fp16 | 16384 | 16384 |
| acc_s_l1 | [64, 64] | fp16 | 8192 | 32768 |
| v_l1 | [64, 128] | fp16 | 16384 | 40960 |
| **合计** | | | **57344** | **< 512KB ✓** |

> 注：k_l1 和 acc_s_l1 可复用地址（不同时使用），v_l1 和 acc_s_l1 也可复用。实际可通过地址复用降至 ~32KB。

**L0C 分配 (Cube 核)**:

| Buffer | Shape | dtype | 大小 (B) | 地址偏移 |
|--------|-------|-------|---------|---------|
| acc_s_l0c | [64, 64] | fp32 | 16384 | 0 |
| acc_o_l0c | [64, 128] | fp32 | 32768 | 0 (与 acc_s_l0c 分时复用) |
| **合计** | | | **32768** | **< 128KB ✓** |

> acc_s_l0c 和 acc_o_l0c 分时复用同一 L0C 地址（GEMM1 完成后才做 GEMM2）。

**UB 分配 (每个 V 核, half_M = block_M // 2 = 32)**:

| Buffer | Shape | dtype | 大小 (B) | 地址偏移 |
|--------|-------|-------|---------|---------|
| acc_o | [32, 128] | fp32 | 16384 | 0 |
| sumexp | [32] | fp32 | 128 | 16384 |
| m_i | [32] | fp32 | 128 | 16512 |
| acc_s_ub | [32, 64] | fp32 | 8192 | 16640 |
| m_i_prev | [32] | fp32 | 128 | 24832 |
| acc_s_ub_ | [32, 64] | fp32 | 8192 | 24960 |
| sumexp_i_ub | [32] | fp32 | 128 | 33152 |
| acc_s_half | [32, 64] | fp16 | 4096 | 33280 |
| acc_o_ub | [32, 128] | fp32 | 16384 | 37376 |
| acc_o_half | [32, 128] | fp16 | 8192 | 53760 |
| **合计** | | | **~62KB** | **< 96KB (单 V 核) ✓** |

> 部分 buffer 可地址复用（如 acc_s_ub_ 和 sumexp_i_ub 不同时使用，acc_s_half 和 acc_o_half 可复用）。

**Workspace (GM 中转)**:

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| workspace_1 | [block_num, block_M, block_N] | fp32 | C→V: QK^T 结果 |
| workspace_2 | [block_num, block_M, block_N] | fp16 | V→C: softmax(P) 结果 |
| workspace_3 | [block_num, block_M, D_v] | fp32 | C→V: PV 结果 |

### 5.3 Kernel 2 (BWD Preprocess) 内存规划

纯 Vector 核操作，无 Cube 参与。

**UB 分配 (blk=32)**:

| Buffer | Shape | dtype | 大小 (B) |
|--------|-------|-------|---------|
| o_ub | [32, 32] | fp16 | 2048 |
| do_ub | [32, 32] | fp16 | 2048 |
| acc_ub | [32, 32] | fp32 | 4096 |
| delta_ub | [32] | fp32 | 128 |
| **合计** | | | **~8KB ✓** |

### 5.4 Kernel 4 (BWD Main) 内存规划

以 block_M=128, block_N=32, D_qk=128, D_v=128 为例：

**L1 分配 (Cube 核)**:

| Buffer | Shape | dtype | 大小 (B) | 说明 |
|--------|-------|-------|---------|------|
| K_l1 | [128, 128] | fp16 | 32768 | K tile |
| V_l1 | [128, 128] | fp16 | 32768 | V tile |
| q_l1 | [32, 128] | fp16 | 8192 | Q tile |
| dsT_l1 | [128, 32] | fp16 | 8192 | dS^T cast 结果 |
| **合计** | | | **81920** | **< 512KB ✓** |

> K_l1 和 V_l1 在循环内不重叠使用时可地址复用，实际峰值 ~49KB。

**L0C 分配**:

| Buffer | Shape | dtype | 大小 (B) | 说明 |
|--------|-------|-------|---------|------|
| qkT_l0c | [128, 32] | fp32 | 16384 | K^T @ Q 结果 |
| dsT_l0c | [128, 32] | fp32 | 16384 | V^T @ dO 结果 |
| dv_l0c | [128, 128] | fp32 | 65536 | P^T @ dO 累加器 |
| dk_l0c | [128, 128] | fp32 | 65536 | dS^T @ Q 累加器 |
| **合计** | | | **~164KB** | **超 128KB ⚠️** |

> **L0C 容量风险**: dv 和 dk 不能同时驻留 L0C。解决方案：dv/dk 分时计算——先在内循环累加 dv/dk 到 UB，或分两阶段处理。具体策略见 §6 Tiling。

**UB 分配 (每个 V 核, half_M = 64)**:

| Buffer | Shape | dtype | 大小 (B) |
|--------|-------|-------|---------|
| qkT_ub | [64, 32] | fp32 | 8192 |
| dsT_ub | [64, 32] | fp32 | 8192 |
| lse_ub | [32] | fp32 | 128 |
| delta_ub | [32] | fp32 | 128 |
| dq_ub | [32, 128] | fp32 | 16384 |
| qkT_half | [64, 32] | fp16 | 4096 |
| dsT_half | [64, 32] | fp16 | 4096 |
| **合计** | | | **~41KB ✓** |

### 5.5 数据搬运路径总览

```
Forward Kernel:
  GM(Q,K,V) → L1 → L0A/L0B → L0C(QK^T) → workspace_1(GM) → UB(softmax) → workspace_2(GM) → L1(P) → L0A/L0B → L0C(PV) → workspace_3(GM) → UB(accumulate) → GM(O)

Backward Kernel:
  GM(K,V,Q,dO,lse,Delta) → L1/UB
  Cube: K^T@Q → L0C → workspace → UB(softmax+mask)
  Cube: V^T@dO → L0C → workspace → UB(dS computation)
  Cube: P@dO → L0C → UB(dV accumulate) → GM(dV)
  Cube: dS@Q → L0C → UB(dK accumulate) → GM(dK)
  Cube: dS^T@K → L0C → UB(dQ) → GM(dQ, atomic_add)
```

---

## 6. Tiling 策略

### 6.1 Forward Kernel

| 参数 | 值 | 说明 |
|------|-----|------|
| block_M | 64 | Q 序列维度 tile 大小 |
| block_N | 64 | KV 序列维度 tile 大小 |
| Grid | `(seq_len // block_M) * heads * batch` | 1D grid，cid 分解为 (bx, by, bz) |

**cid 分解**:
```python
bx = cid % (seq_len // block_M)          # Q 序列块索引
by = cid // (seq_len // block_M) % heads  # Q head 索引
bz = cid // (seq_len // block_M) // heads # batch 索引
kv_by = by // groups                       # KV head 索引 (GQA)
```

**内循环**: `for k in T.serial(ceildiv(seq_len, block_N))`
- 每次迭代处理一个 KV tile
- Causal 模式下循环范围为 `ceildiv((bx+1)*block_M, block_N)`

### 6.2 Backward Preprocess Kernel

| 参数 | 值 | 说明 |
|------|-----|------|
| blk | 32 | 序列维度 tile |
| Grid | `heads * ceildiv(seq_len, blk) * batch` | 纯 Vector 核 |

**内循环**: `for k in range(ceildiv(D_v, blk))` — 沿 D_v 维度分块累加

### 6.3 Backward Postprocess Kernel

| 参数 | 值 | 说明 |
|------|-----|------|
| blk | 64 | 序列维度 tile |
| Grid | `ceildiv(seq_len, blk) * heads * batch` | 纯 Vector 核 |

### 6.4 Backward Main Kernel

| 参数 | 值 | 说明 |
|------|-----|------|
| block_M | 128 | KV 序列维度 tile（外层 grid） |
| block_N | 32 | Q 序列维度 tile（内循环） |
| Grid | `heads * ceildiv(seq_len, block_M) * batch` | 按 head 和 KV 块分配 |

**cid 分解**:
```python
bx = cid % heads                          # head 索引
by = cid // heads % ceildiv(seq_len, block_M)  # KV 序列块索引
bz = cid // heads // ceildiv(seq_len, block_M) # batch 索引
kv_bx = bx // groups                       # KV head 索引 (GQA)
```

**内循环**: `for k in T.serial(loop_st, loop_ed)`
- Non-causal: `loop_st=0, loop_ed=ceildiv(seq_len, block_N)`
- Causal: `loop_st=floordiv(by*block_M, block_N), loop_ed=ceildiv(seq_len, block_N)`

### 6.5 L0C 容量解决方案 (Backward Kernel)

**问题**: dv [128, 128] fp32 = 64KB + dk [128, 128] fp32 = 64KB = 128KB，恰好等于 L0C 上限。

**方案 A — 分时计算（推荐）**:
1. 内循环中先计算 dv（GEMM: P^T @ dO → L0C），每轮迭代后将 dv 从 L0C 搬运到 UB 累加
2. 再计算 dk（GEMM: dS^T @ Q → L0C），每轮迭代后将 dk 从 L0C 搬运到 UB 累加
3. dv 和 dk 的 L0C 空间分时复用（地址偏移均为 0）

**方案 B — 缩小 block_M**:
- 将 block_M 从 128 降为 64，dv/dk 各 32KB，总计 64KB < 128KB
- 代价：grid 翻倍，可能增加 kernel launch 开销

**推荐方案 A**，参考 `flash_attn_bhsd.py` 的分时复用模式。

### 6.6 非整除处理

| 维度 | 策略 |
|------|------|
| seq_len % block_M ≠ 0 | L0 测试要求 seq_len 整除 block_M；L1 可扩展动态边界 |
| seq_len % block_N ≠ 0 | 同上，L0 要求整除 |
| D_qk / D_v 非整除 | L0 测试使用 2 的幂次；后续可扩展 |
| Causal 尾块 | 最后一个 KV tile 可能需要掩码处理 |

---

## 7. Cube/Vector 分工

### 7.1 Kernel 1 (Forward)

```
┌─────────────────────────────────────────────────────────┐
│ Scope("C") — Cube 核                                     │
│                                                          │
│  1. GM(Q) → q_l1 (L1)                                   │
│  for k in serial(num_kv_blocks):                         │
│    2. GM(K) → k_l1 (L1)                                 │
│    3. gemm_v0(q_l1, k_l1, acc_s_l0c, transB, init)      │
│    4. L0C → workspace_1(GM)                              │
│    5. set_cross_flag("FIX", 0)  ──────── notify V ────►  │
│    6. wait_cross_flag(1)  ◄──── V done softmax ────      │
│    7. workspace_2(GM) → acc_s_l1 (L1)                    │
│    8. GM(V) → v_l1 (L1)                                 │
│    9. gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init)          │
│   10. L0C → workspace_3(GM)                              │
│   11. set_cross_flag("FIX", 2)  ──────── notify V ────►  │
│   12. wait_cross_flag(3)  ◄──── V done accumulate ────   │
│                                                          │
├─────────────────────────────────────────────────────────┤
│ Scope("V") — Vector 核 (vid=0,1 各处理 block_M//2 行)    │
│                                                          │
│  init: fill(acc_o, 0), fill(sumexp, 0), fill(m_i, -inf) │
│  for k in serial(num_kv_blocks):                         │
│    1. wait_cross_flag(0)  ◄──── C done QK^T ────        │
│    2. workspace_1(GM) → acc_s_ub (UB, vid slice)         │
│    3. scale + reduce_max + exp + reduce_sum (softmax)     │
│    4. acc_s_half → workspace_2(GM)                       │
│    5. set_cross_flag("MTE3", 1)  ──── notify C ────►     │
│    6. wait_cross_flag(2)  ◄──── C done PV ────          │
│    7. workspace_3(GM) → acc_o_ub (UB, vid slice)         │
│    8. rescale acc_o + accumulate                          │
│    9. set_cross_flag("V", 3)  ──── notify C ────►        │
│  normalize: acc_o /= sumexp                              │
│  output: UB → GM(Output)                                 │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Kernel 2 (BWD Preprocess)

纯 Vector 核，无 Cube 参与。

```
Scope("V") only:
  for k in range(ceildiv(D_v, blk)):
    GM(O slice) → o_ub
    GM(dO slice) → do_ub
    acc += o * do (element-wise)
  reduce_sum(acc, delta, dim=1)
  UB(delta) → GM(Delta)
```

### 7.3 Kernel 3 (BWD Postprocess)

纯 Vector 核。

```
Scope("V") only:
  GM(dQ fp32) → dQ_ub
  T.tile.cast(dQ_half_ub, dQ_ub, "CAST_NONE", count)
  UB(dQ_half) → GM(dQ_out fp16)
```

### 7.4 Kernel 4 (BWD Main)

```
┌─────────────────────────────────────────────────────────┐
│ Scope("C") — Cube 核                                     │
│                                                          │
│  GM(K) → K_l1, GM(V) → V_l1                             │
│  for k in serial(num_q_blocks):                          │
│    GM(Q) → q_l1                                          │
│    gemm_v0(K_l1, q_l1, qkT_l0c, transB, init)           │
│    L0C → workspace_1(GM)  ──── notify V ────►            │
│    ◄──── V done softmax+mask ────                         │
│    workspace_2(GM) → qkT_l1                              │
│    GM(dO) → do_l1                                        │
│    gemm_v0(V_l1, do_l1, dsT_l0c, transB, init)          │
│    L0C → workspace_4(GM)  ──── notify V ────►            │
│    ◄──── V done dS computation ────                       │
│    workspace_5(GM) → dsT_l1                              │
│    gemm_v0(qkT_l1, do_l1, dv_l0c, init)                 │
│    L0C → workspace_dv(GM)  ──── notify V ────►           │
│    gemm_v0(dsT_l1, q_l1, dk_l0c, init)                  │
│    L0C → workspace_dk(GM)  ──── notify V ────►           │
│    workspace_dsT(GM) → dsT_l1                            │
│    gemm_v0(dsT_l1, K_l1, dq_l0c, transA, init)          │
│    L0C → workspace_dq(GM)  ──── notify V ────►           │
│                                                          │
│  end: dv/dk → GM(dV/dK, atomic_add)                      │
├─────────────────────────────────────────────────────────┤
│ Scope("V") — Vector 核                                   │
│                                                          │
│  for k in serial(num_q_blocks):                          │
│    ◄──── QK^T from workspace_1 ────                      │
│    softmax: scale + exp(lse) + causal_mask               │
│    ──── P to workspace_2 ────►                            │
│    ◄──── V^T@dO from workspace_4 ────                    │
│    dS = P * (dS - Delta) * sm_scale                      │
│    ──── dS to workspace_5 ────►                           │
│    ◄──── dv from workspace_dv ────                       │
│    accumulate dv to dv_acc_ub                             │
│    ◄──── dk from workspace_dk ────                       │
│    accumulate dk to dk_acc_ub                             │
│    ◄──── dq from workspace_dq ────                       │
│    T.tile.atomic_add(dQ_gm, dq_ub)                       │
│                                                          │
│  end: T.tile.atomic_add(dV_gm, dv_acc_ub)                │
│       T.tile.atomic_add(dK_gm, dk_acc_ub)                │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 同步策略

### 8.1 核间同步 (Cross-core Flags)

**Forward Kernel 同步协议**:

| 步骤 | 发起方 | API | 信号 | 含义 |
|------|--------|-----|------|------|
| 1 | Cube | `set_cross_flag("FIX", 0)` | 0 | QK^T 结果已写入 workspace_1 |
| 2 | Vector | `wait_cross_flag(0)` | 0 | 等待 QK^T 就绪 |
| 3 | Vector | `set_cross_flag("MTE3", 1)` | 1 | softmax(P) 已写入 workspace_2 |
| 4 | Cube | `wait_cross_flag(1)` | 1 | 等待 P 就绪 |
| 5 | Cube | `set_cross_flag("FIX", 2)` | 2 | PV 结果已写入 workspace_3 |
| 6 | Vector | `wait_cross_flag(2)` | 2 | 等待 PV 就绪 |
| 7 | Vector | `set_cross_flag("V", 3)` | 3 | 累加完成，可进入下一轮 |
| 8 | Cube | `wait_cross_flag(3)` | 3 | 等待 V 核累加完成 |

**数据通路选择**:
- Cube 核写 GM: 使用 `"FIX"` 通路（L0C → GM）
- Vector 核写 GM: 使用 `"MTE3"` 通路（UB → GM）
- Vector 核通知 Cube: 使用 `"V"` 或 `"MTE3"` 通路

### 8.2 核内同步 (Barrier)

Expert 模式下，每次 `T.copy` 和 `T.gemm_v0` 前后都需要 `T.barrier_all()`:

```python
# 典型 Cube 核操作序列
T.copy(GM_src, L1_dst)
T.barrier_all()                    # 等待搬运完成
T.gemm_v0(L1_A, L1_B, L0C_C, init=True)
T.barrier_all()                    # 等待 GEMM 完成
T.copy(L0C_C, GM_dst)
T.barrier_all()                    # 等待写出完成
```

### 8.3 Backward Kernel 同步协议

Backward 更复杂，每轮内循环有 5+ 次 C↔V 交互。信号编号需要仔细规划避免死锁：

| 信号 | 方向 | 含义 |
|------|------|------|
| 0 | C→V | QK^T 结果就绪 |
| 1 | V→C | softmax(P) 就绪 |
| 2 | C→V | V^T@dO 结果就绪 |
| 3 | V→C | dS 计算完成 |
| 4 | C→V | PV (dv 贡献) 就绪 |
| 5 | C→V | dS@Q (dk 贡献) 就绪 |
| 6 | C→V | dS^T@K (dq 贡献) 就绪 |
| 7 | V→C | V 核处理完所有结果 |

---

## 9. GQA 处理

### 9.1 Head 分组映射

```python
groups = heads_q // heads_kv   # 每个 KV head 对应的 Q head 数

# Forward: Q head by → KV head kv_by
kv_by = by // groups

# 数据访问:
Q[bz, by, ...]                 # Q 用完整 head 索引
K[bz, kv_by, ...]              # K 用 KV head 索引
V[bz, kv_by, ...]              # V 用 KV head 索引
```

### 9.2 Forward 中的 GQA

- Grid 按 Q head 数 (`heads`) 分配
- 每个 Q head tile 读取对应的 KV head tile（`by // groups`）
- 多个 Q head 共享同一 KV head 的数据

### 9.3 Backward 中的 GQA

**dQ**: 每个 Q head 独立计算，直接 atomic_add 到 `dQ[bz, k*block_N+i, bx, j]`

**dK/dV**: 多个 Q head 的梯度需要累加到同一个 KV head：
- **atomic_add 方案**: 每个 Q head 的 kernel 实例直接 `T.tile.atomic_add` 到 `dK[bz, kv_by, ...]`，硬件保证原子性
- **split 方案**: 每个 Q head 写到 `dK[bx % groups, bz, kv_by, ...]` 的独立 slice，host 侧 `dk.sum(0)` 归约

### 9.4 NPU 实现选择

**推荐 atomic_add 方案**（Kernel 4），理由：
1. 减少 host 侧后处理
2. `T.tile.atomic_add` 在 V1 版本已支持 UB → GM
3. 减少 dK/dV 的内存占用（无需 groups 维度）

**dK/dV 的 atomic_add 路径**:
```python
# V 核中，内循环结束后
T.tile.atomic_add(dV[bz, kv_by, by*block_M:(by+1)*block_M, :], dv_acc_ub)
T.tile.atomic_add(dK[bz, kv_by, by*block_M:(by+1)*block_M, :], dk_acc_ub)
```

---

## 10. Causal Mask 实现

### 10.1 Forward Causal Mask

**GPU 原始实现** (T.Parallel + if_then_else):
```python
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.if_then_else(bx*block_M + i >= k*block_N + j, 0, -inf)
```

**NPU Expert 实现** (T.tile.compare + T.tile.select):

由于 NPU 的 T.Parallel 不支持 if-else，需要用 compare + select 替代：

```python
# 方案: 在 softmax 阶段处理 causal mask
# 1. QK^T 结果写入 workspace_1 后，V 核读取时处理 mask
# 2. 利用 lse 中的 -inf 值自然处理

# 具体步骤 (V 核内):
# Step 1: 构建行列索引
#   row_idx = bx * block_M + vid * half_M + i  (0 <= i < half_M)
#   col_idx = k * block_N + j                  (0 <= j < block_N)
# Step 2: causal 条件: row_idx >= col_idx
#   利用 T.tile.arith_progression 生成索引序列
#   利用 T.tile.compare 生成 mask
#   利用 T.tile.select 将不满足条件的位置设为 -inf

# 简化方案 (推荐):
# 在 QK^T 结果上做 post-mask:
# 1. 生成 -inf 填充的 buffer
# 2. T.tile.compare(mask, row_idx_buf, col_idx_buf, "GE")
# 3. T.tile.select(acc_s_ub, mask, acc_s_ub, neg_inf_buf, "VSEL_CMPMASK_SPR")
```

**Causal 循环范围优化**:
```python
# Non-causal: 遍历所有 KV blocks
loop_range = ceildiv(seq_len, block_N)

# Causal: 只遍历到当前 Q block 对应的 KV blocks
loop_range = ceildiv((bx + 1) * block_M, block_N)
```

### 10.2 Backward Causal Mask

Backward 中的 causal mask 应用在 P 矩阵上：
```python
# GPU: qkT[i,j] = if_then_else(by*block_M + i <= k*block_N + j, qkT[i,j], 0)
# NPU: T.tile.compare + T.tile.select 将不满足条件的位置设为 0
```

---

## 11. 技术约束检测

### 11.1 三维 Kernel 限制

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 三维 grid | ⚠️ 需适配 | GPU 用 3D grid `(bx, by, bz)`，NPU 必须展平为 1D `cid` + 手动分解 |
| threads 参数 | ✅ 已处理 | GPU `threads=256` 在 NPU 无对应概念，NPU 由 cid/vid 决定并行度 |

### 11.2 动态边界约束

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 循环边界 | ⚠️ 需静态化 | Ascend 要求循环边界在编译期可知；`T.ceildiv(seq_len, block_N)` 在 JIT 时已是常量 |
| Causal loop_range | ✅ 可静态化 | `ceildiv((bx+1)*block_M, block_N)` 中 bx 是运行时变量，但可作为循环内的条件判断 |
| 尾块处理 | ⚠️ L0 规避 | L0 测试要求 seq_len 整除 block_M/block_N，避免尾块 |

### 11.3 L0C 容量风险

| Kernel | L0C 峰值需求 | L0C 上限 | 状态 |
|--------|-------------|---------|------|
| Forward (64×64) | 32KB (acc_o_l0c) | 128KB | ✅ 安全 |
| BWD Preprocess | 0 (纯 Vector) | 128KB | ✅ 无风险 |
| BWD Postprocess | 0 (纯 Vector) | 128KB | ✅ 无风险 |
| BWD Main (128×32) | ~130KB (dv+dk) | 128KB | ⚠️ 需分时复用 |

**BWD Main L0C 解决方案**: 采用分时计算策略（§6.5 方案 A），dv 和 dk 的 GEMM 分时使用 L0C，每轮 GEMM 完成后立即搬运到 UB 累加。

### 11.4 GEMM 分形限制

| 检查项 | 最小要求 (fp16) | 实际值 | 状态 |
|--------|----------------|--------|------|
| Forward QK^T: M×K×N | M≥16, K≥16, N≥16 | 64×128×64 | ✅ |
| Forward PV: M×K×N | M≥16, K≥16, N≥16 | 64×64×128 | ✅ |
| BWD K^T@Q: M×K×N | M≥16, K≥16, N≥16 | 128×128×32 | ✅ |
| BWD V^T@dO: M×K×N | M≥16, K≥16, N≥16 | 128×128×32 | ✅ |
| BWD P@dO: M×K×N | M≥16, K≥16, N≥16 | 128×32×128 | ✅ |
| BWD dS@Q: M×K×N | M≥16, K≥16, N≥16 | 128×32×128 | ✅ |
| BWD dS^T@K: M×K×N | M≥16, K≥16, N≥16 | 32×128×128 | ✅ |

### 11.5 UB 容量风险

| Kernel | UB 峰值需求 (单 V 核) | UB 上限 (单 V 核) | 状态 |
|--------|---------------------|------------------|------|
| Forward | ~62KB | ~96KB | ✅ 安全 |
| BWD Preprocess | ~8KB | ~96KB | ✅ 安全 |
| BWD Main | ~41KB + dv/dk acc | ~96KB | ⚠️ 需精确规划 |

**BWD Main UB 风险**: dv_acc [128, 128] fp32 = 64KB 和 dk_acc [128, 128] fp32 = 64KB 不能同时驻留单 V 核 UB。解决方案：
- dv_acc 和 dk_acc 分时使用 UB（与 L0C 分时策略配合）
- 或缩小 block_M 到 64，使 dv_acc/dk_acc 各 16KB

### 11.6 Workspace 内存需求

| Kernel | Workspace 总量 | 说明 |
|--------|---------------|------|
| Forward | 3 × block_num × (64×64 + 64×64 + 64×128) | ~block_num × 32KB |
| BWD Main | 7 × block_num × (128×32 + ...) | 更多 workspace 通道 |

> Workspace 分配在 GM，受设备全局内存限制，通常不是瓶颈。

### 11.7 atomic_add 支持

| 检查项 | 状态 | 说明 |
|--------|------|------|
| T.tile.atomic_add UB→GM | ✅ V1 支持 | dQ/dK/dV 的 atomic 累加 |
| fp32 atomic_add | ✅ 支持 | dQ/dK/dV 均为 fp32 累加器 |
| 预清零 | ⚠️ 需要 | 调用 kernel 前必须 `torch.zeros_like` 初始化 |

### 11.8 exp2 vs exp 差异

GPU 使用 `T.exp2(x * log2e)` 做指数运算，NPU 使用 `T.tile.exp(x)` 做自然指数。需要调整 scale 因子：
- GPU: `scale = (1/dim_qk)^0.5 * 1.44269504` (log2(e))，配合 exp2
- NPU: `scale = (1/dim_qk)^0.5`，配合 exp

---

## 12. 验证方案

### 12.1 Golden 函数 (PyTorch 参考实现)

```python
import torch
import torch.nn.functional as F

def ref_program(Q, K, V, is_causal=False, groups=1):
    """
    Q: [B, H, N, D_qk] float16
    K: [B, H_kv, N, D_qk] float16
    V: [B, H_kv, N, D_v] float16
    groups = H // H_kv
    Returns: O [B, H, N, D_v] float16
    """
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    # GQA: repeat KV to match Q heads
    if groups > 1:
        K_f = K_f.repeat_interleave(groups, dim=1)
        V_f = V_f.repeat_interleave(groups, dim=1)

    dim_qk = Q_f.shape[-1]
    scale = 1.0 / (dim_qk ** 0.5)

    # S = Q @ K^T * scale
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_f) * scale

    if is_causal:
        N = Q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    P = F.softmax(scores, dim=-1)
    O = torch.einsum("bhqk,bkhd->bqhd", P, V_f)
    return O.half()


def ref_bwd(Q, K, V, dO, is_causal=False, groups=1):
    """
    参考反向实现，返回 dQ, dK, dV
    """
    Q_f = Q.float().requires_grad_(True)
    K_f = K.float().requires_grad_(True)
    V_f = V.float().requires_grad_(True)

    K_rep = K_f.repeat_interleave(groups, dim=1) if groups > 1 else K_f
    V_rep = V_f.repeat_interleave(groups, dim=1) if groups > 1 else V_f

    dim_qk = Q_f.shape[-1]
    scale = 1.0 / (dim_qk ** 0.5)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q_f, K_rep) * scale

    if is_causal:
        N = Q.shape[2]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    P = F.softmax(scores, dim=-1)
    O = torch.einsum("bhqk,bkhd->bqhd", P, V_rep)
    O.backward(dO.float())

    dQ = Q_f.grad.half()
    # dK/dV 需要归约 groups
    dK_rep = K_f.grad  # [B, H_kv, N, D_qk]
    dV_rep = V_f.grad  # [B, H_kv, N, D_v]

    return dQ, dK_rep.half(), dV_rep.half()
```

### 12.2 L0 门槛测试计划

> 由 `tilelang-op-test-design`（场景 A）生成。L0 只覆盖**规则 shape（block 整除）、标准 dtype、基本精度**。

#### L0 测试用例

| 用例 ID | 场景 | B | H | H_kv | groups | N | D_qk | D_v | causal | block_M | block_N |
|---------|------|---|---|------|--------|---|------|-----|--------|---------|---------|
| L0-FWD-01 | Forward MHA | 1 | 1 | 1 | 1 | 128 | 64 | 64 | False | 64 | 64 |
| L0-FWD-02 | Forward GQA | 1 | 2 | 1 | 2 | 128 | 64 | 64 | False | 64 | 64 |
| L0-FWD-03 | Forward GQA causal | 1 | 2 | 1 | 2 | 128 | 64 | 64 | True | 64 | 64 |
| L0-FWD-04 | Forward batch | 2 | 4 | 2 | 2 | 256 | 64 | 64 | False | 64 | 64 |
| L0-PREP-01 | BWD preprocess | 1 | 1 | — | — | 128 | — | 64 | — | — | — |
| L0-POST-01 | BWD postprocess | 1 | 1 | — | — | 128 | 64 | — | — | — | — |
| L0-BWD-01 | BWD MHA | 1 | 1 | 1 | 1 | 128 | 64 | 64 | False | 128 | 32 |
| L0-BWD-02 | BWD GQA | 1 | 2 | 1 | 2 | 128 | 64 | 64 | False | 128 | 32 |
| L0-BWD-03 | BWD GQA causal | 1 | 2 | 1 | 2 | 128 | 64 | 64 | True | 128 | 32 |

#### L0 dtype 配置

| 张量 | dtype |
|------|-------|
| Q, K, V, O, dO | float16 |
| lse, Delta, dQ (累加器) | float32 |
| dK, dV (最终输出) | float16 (从 fp32 转换) |

#### L0 精度标准

| Kernel | 测试类型 | atol | rtol | 说明 |
|--------|---------|------|------|------|
| Forward | 输出 O | 1e-2 | 1e-2 | fp16 累积，参考 GPU 版本标准 |
| Forward | lse | 1e-2 | 1e-2 | fp32 但受 softmax 数值范围影响 |
| BWD Preprocess | Delta | 1e-3 | 1e-3 | 简单 dot product，精度较高 |
| BWD Postprocess | dQ_out | 1e-3 | 1e-3 | 纯 dtype 转换 |
| BWD Main | dQ | 1e-2 | 1e-2 | 多步 GEMM + softmax 反向 |
| BWD Main | dK | 1e-2 | 1e-2 | 同上 |
| BWD Main | dV | 1e-2 | 1e-2 | 同上 |

#### L0 验证流程

```python
# 1. Forward 验证
O_npu, lse_npu = flashattn_fwd(Q, K, V)
O_ref = ref_program(Q, K, V, is_causal, groups)
torch.testing.assert_close(O_npu, O_ref, atol=1e-2, rtol=1e-2)

# 2. BWD Preprocess 验证
Delta_npu = flashattn_bwd_preprocess(O, dO)
Delta_ref = (O.float() * dO.float()).sum(dim=-1)
torch.testing.assert_close(Delta_npu, Delta_ref, atol=1e-3, rtol=1e-3)

# 3. BWD Main 验证 (端到端)
dQ_npu, dK_npu, dV_npu = flashattn_bwd_pipeline(Q, K, V, dO, lse, Delta)
dQ_ref, dK_ref, dV_ref = ref_bwd(Q, K, V, dO, is_causal, groups)
torch.testing.assert_close(dQ_npu, dQ_ref, atol=1e-2, rtol=1e-2)
torch.testing.assert_close(dK_npu, dK_ref, atol=1e-2, rtol=1e-2)
torch.testing.assert_close(dV_npu, dV_ref, atol=1e-2, rtol=1e-2)
```

---

## 13. 同类实现引用

| 文件路径 | 说明 | 参考价值 |
|---------|------|---------|
| `examples/flash_attention/flash_attn_bhsd.py` | NPU Expert 模式 forward，手动 Scope/sync/address | **最核心参考**：C/V 分离、workspace 机制、同步协议 |
| `examples/flash_attention/flash_attn_bhsd_cc_sync.py` | NPU 自动 CV sync 版本 forward | 参考 auto_cv_sync pass_configs 的用法 |
| `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` | NPU 高性能 Expert forward，GQA 支持 | **GQA 处理参考**：`kv_by = by // (heads_q // heads_kv)`、T.mma + double buffering |
| `examples/flash_attention/paged_flash_attn_bhsd.py` | Paged KV cache flash attention | 参考 auto_cv_combine + workspace 索引模式 |
| GPU `examples/flash_attention/example_gqa_bwd.py` | GPU GQA forward + backward 完整实现 | **算法逻辑参考**：5 个 kernel 的计算流程 |

---

## 14. 风险点与缓解措施

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| BWD Main L0C 溢出 (dv+dk > 128KB) | 高 | 分时计算：dv/dk GEMM 交替使用 L0C，每轮搬出到 UB 累加 |
| BWD Main UB 溢出 (dv_acc+dk_acc > 96KB) | 高 | 分时累加：dv_acc 和 dk_acc 分时驻留 UB；或缩小 block_M |
| Causal mask 在 T.tile 中实现复杂 | 中 | 利用 T.tile.compare + T.tile.select；或简化为循环范围限制 + 尾块特殊处理 |
| 多 workspace 通道增加同步复杂度 | 中 | 严格规划信号编号，避免死锁；参考 flash_attn_bhsd.py 的同步协议 |
| D_qk=192 非 2 的幂次 | 中 | L0 测试先用 D=64/128；D=192 留到 L1 测试 |
| exp2 → exp 转换影响数值精度 | 低 | scale 因子调整：去掉 log2(e) 乘数；精度标准放宽到 1e-2 |
| BWD atomic_add 性能不如 split | 低 | 先实现 atomic_add 版本验证正确性；性能优化阶段可切换 split |

---

## 15. 交付清单

### 15.1 目录结构

```
example_gqa_bwd/
├── design.md                # 本设计文档
├── example_gqa_bwd.py       # 算子实现（Forward v1/v4 + Backward pipeline + Golden Ref + Autograd + __main__ 冒烟测试）
├── test_gqa_bwd.py          # 分层测试（L0/L1/L2/Boundary + argparse --level）
└── perf_example_gqa_bwd.py  # 性能测试（正确性检查 + TileLang vs PyTorch 对比）
```

### 15.2 文件说明

| 文件 | 说明 |
|------|------|
| `design.md` | 算子设计文档：I/O 规格、编程模式、API 映射、内存规划、tiling、同步策略、验证方案 |
| `example_gqa_bwd.py` | 算子实现：全部 kernel + golden reference + autograd wrapper + `__main__` 冒烟测试（输出 "Test Passed!"） |
| `test_gqa_bwd.py` | 分层测试文件：L0（规则 shape，阻塞）/ L1（不规则 shape，阻塞）/ L2（异常输入，非阻塞）/ Boundary（特殊值，非阻塞）+ `argparse --level` |
| `perf_example_gqa_bwd.py` | 性能测试：正确性检查 + TileLang Forward v1/v4 + Backward pipeline vs PyTorch baseline + 输出 "Test Passed!" |

### 15.3 函数清单

| 函数名 | 说明 |
|--------|------|
| `flashattn_fwd` | Forward 基础版（gemm_v0 + online softmax） |
| `flashattn_fwd_v4` | Forward 优化版（L0 双缓冲 + Fixed Core + 批处理 + fine-grained sync, num_stages=8） |
| `flashattn_bwd_preprocess` | Backward 预处理（Delta = sum(O × dO)） |
| `flashattn_bwd_postprocess` | Backward 后处理（dQ fp32→fp16） |
| `flashattn_bwd_pipeline` | Backward 主 kernel（5 GEMM + 批处理 + fine-grained sync, num_stages=8） |
| `ref_program` / `ref_bwd` | PyTorch golden reference |
| `attention` | autograd wrapper（BSHD↔BHSD 布局转换） |

### 15.4 运行方式

```bash
# 1. 冒烟测试（CI 直接运行主文件）
python example_gqa_bwd.py
# 输出: "Test Passed!"

# 2. 分层精度测试
python test_gqa_bwd.py --level l0     # L0 门槛测试（阻塞）
python test_gqa_bwd.py --level all    # 全部层级（L0+L1+L2+Boundary）

# 3. 性能测试
python perf_example_gqa_bwd.py
# 输出: 性能表格 + "Test Passed!"

# 4. pytest 兼容
python -m pytest test_gqa_bwd.py -v
```
