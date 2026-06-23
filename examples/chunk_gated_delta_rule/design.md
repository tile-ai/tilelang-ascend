# chunk_gated_delta_rule 算子设计文档

## 1. 概述

### 1.1 算子名称

`chunk_gated_delta_rule` - 基于 chunk 的门控 Delta Rule 前向传播算子，用于线性注意力机制（Linear Attention）中的隐藏状态递推计算。支持变长序列模式。

### 1.2 数学公式

该算子实现以下递推关系（按 chunk 分块计算）：

```
对于每个 chunk t (t = 0, 1, ..., NT-1):
  1. h[t] = h[t-1]  (累积状态，初始为 h0 或零)
  2. v_new[t] = v[t] - w[t] @ h[t]           # 残差计算 (GEMM)
  3. 若 USE_G:
       g_last = g[t_valid - 1]               # chunk 内最后一个有效 token 的 gate
       mask = (g_last - g[t] <= 0)           # 数值稳定掩码：仅保留 g <= g_last 的 token
       v_new[t] = v_new[t] * exp(where(mask, g_last - g[t], -inf))  # 门控缩放（超出部分置零）
       h[t] = h[t] * exp(g_last)              # 状态衰减
  4. h[t+1] = h[t] + k[t] @ v_new[t]          # 状态更新 (GEMM)
```

其中：
- `@` 表示矩阵乘法
- `g_last = g[t_valid - 1]` 为 chunk 内最后一个**有效** token 的 gate 值（支持未 padding 的变长序列）
- `where(mask, g_last - g[t], -inf)` 为数值稳定的指数运算：对 `g_last - g[t] > 0` 的 token（即 g > g_last）置为 -inf，使 `exp(-inf) = 0`，避免溢出
- 核心计算为 **两次 GEMM + 多步 element-wise 门控运算**

### 1.3 计算特征分析

| 维度 | 分析结果 |
|------|---------|
| **计算类型** | 混合（Cube GEMM + Vector element-wise） |
| **复杂度级别** | 多步（GEMM → sub → compare/select → exp → mul → GEMM → add，需中间缓冲） |
| **动态 shape** | N、NT、T_total 为符号维度；支持变长序列（cu_seqlens） |
| **核间协作** | 需要 Cube 核做 GEMM，Vector 核做 element-wise 门控 |
| **调度模式** | Work-queue：每个 Cube+Vector pair 处理多个 (i_n, i_h) 组合 |

### 1.4 典型配置示例

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 1 | 序列数 |
| T_total | 16384 | 总 token 数 |
| H | 32 | value head 数 |
| Hg | 16 | key head 数（GQA，H/Hg=2） |
| K | 128 | key dim |
| V | 128 | value dim |
| BT | 64 | chunk size（固定） |
| NT_max | 256 | 最大 chunk 数 = ceil(16384/64) |
| VEC_CORE_NUM | 48 | Vector 核总数 |
| CUBE_BLOCK_NUM | 24 | Cube+Vector pair 数 |

---

## 2. 编程模式选型

### 2.1 选型结论：**Expert 模式**

### 2.2 选型理由

| 因素 | 分析 |
|------|------|
| 含 GEMM 计算 | 需要 `T.gemm_v0`，涉及 L0A/L0B/L0C 寄存器管理 |
| 多步 element-wise | `v_new = v - w@h`、`compare/select`、`exp`、`mul`、`add` 需要中间 buffer |
| 状态累积 | `h` 在 chunk 间累积，需要精细管理 buffer 生命周期 |
| V 维度分块 | V 分为两半 [0:V//2] 和 [V//2:V]，由 `vid` 控制，需要显式分配 L1/UB/L0 buffer |
| 混合核计算 | Cube 核做 GEMM，Vector 核做 element-wise，需要显式 buffer 类型指定 |
| 数值稳定门控 | `T.tile.compare` + `T.tile.select` 实现数值稳定的 exp 运算，需要额外 padding buffer |

Expert 模式的优势：
- 使用 `T.alloc_L1/ub/L0C` 显式控制内存层级，避免编译器自动规划的不确定性
- 使用 `T.tile.*` 原语用于 element-wise 计算（含 compare/select 等高级原语）
- 显式控制数据搬运路径（GM → L1 → UB）
- Work-queue 调度模式，灵活分配计算资源

### 2.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
```

全部关闭，采用手动同步策略：
- 手动 barrier (`T.barrier_all` / `T.wait_flag` / `T.set_flag`)
- 手动 CV 分离 (`T.Scope("C")` / `T.Scope("V")`)
- 手动跨核同步 (`T.set_cross_flag` / `T.wait_cross_flag`)
- 手动内存管理 (`T.alloc_L1/ub/L0C` 显式分配)

---

## 3. API 映射设计

### 3.1 核心计算步骤 → TileLang API 映射

| 计算步骤 | PyTorch 参考 | TileLang API |
|---------|-------------|--------------|
| 加载初始状态 h0 | `h_state = h0.clone()` | `T.copy(h0[i_n, i_h, K//2*vid:..., j*V_half:...], h_state_ub_float[j])` （直接加载到 float32 缓冲） |
| 加载 w | `w_chunk = w[t_start:t_end]` | `T.copy(w[bos:bos+actual_len, i_h, :], w_chunk_l1[0, :, :])` （首 chunk 使用 actual_len） |
| 加载 k | `k_chunk = k[t_start:t_end]` | `T.copy(k[bos:bos+actual_len, k_head, :], k_chunk_l1[0, :, :])` （首 chunk 使用 actual_len） |
| 加载 v | `v_chunk = v[t_start:t_end]` | `T.copy(v[vec_global_start:..., i_h, j*V_half:...], v_chunk_ub[0, j])` （varlen-aware 切分） |
| 加载 g | `g_chunk = g[t_start:t_end]` | `T.copy(g[i_h, vec_global_start:...], g_chunk_ub[0])` （g shape 为 [H, T_total]，已转置） |
| **GEMM: w @ h** | `torch.matmul(w, h)` | `T.gemm_v0(w_chunk_l1[pid], h_state_l1[j], wh_frag[j], init=True)` |
| **v_new = v - wh** | `v - torch.matmul(w, h)` | `T.copy(v_chunk_ub[pid, j], v_chunk_ub_float[j])` → `T.tile.sub(v_chunk_ub_float[j], v_chunk_ub_float[j], wh_ub_float[j])` |
| **数值稳定 exp(g_last - g)** | `torch.where(g_last-g<=0, g_last-g, -inf)` → `torch.exp(...)` | `T.tile.fill(g_exp_ub, g_last)` → `T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid])` → `T.copy(g_exp_ub, g_exp_ub_pad[0:bt//2])` → `T.tile.compare(g_mask_ub_pad, g_exp_ub_pad, 0, "LE")` → `T.tile.select(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad, -inf, "VSEL_TENSOR_SCALAR_MODE")` → `T.copy(g_exp_ub_pad[0:bt//2], g_exp_ub)` → `T.tile.exp(g_exp_ub, g_exp_ub)` → `T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)` |
| **门控缩放** | `v_new * exp(g)` | `T.tile.mul(v_chunk_ub_float[j], v_chunk_ub_float[j], g_exp_ub_broc)` |
| **状态衰减** | `h * exp(g_last)` | `T.tile.fill(g_last_scalar, g_last)` → `T.tile.exp(g_last_scalar, g_last_scalar)` → `T.tile.mul(h_state_ub_float[j], h_state_ub_float[j], g_last_scalar[0])` （直接在 float32 缓冲上操作） |
| **GEMM: k @ v_new** | `torch.matmul(k.T, v_new)` | `T.gemm_v0(k_chunk_l1[pid], v_new_l1[j], hupd_frag[j], transpose_A=True, init=True)` |
| **状态累积** | `h = h + k.T @ v_new` | `T.tile.add(h_state_ub_float[j], h_state_ub_float[j], hupd_ub_float[j])` （直接在 float32 缓冲上操作） |
| h_state → Cube 传输 | N/A | `T.copy(h_state_ub_float[j], h_state_ub[j])` (float32→bf16) → `T.copy(h_state_ub[j], ws_h[...])` → `T.set_cross_flag("MTE3", SEM_H_V2C+j)` |
| 存储 h[t] | `h[chunk_idx] = h_state` | `T.copy(h_state_ub[j], h[i_n, i, i_h, K//2*vid:..., j*V_half:...])` （与 ws_h 传输在同一 j 循环内） |
| 存储 v_new | `v_new[t_start:t_end] = v_n` | `T.copy(v_chunk_ub[pid, j, :vec_chunk_len, :], v_new[...])` （使用 vec_chunk_len 而非全量） |
| 存储 ht | `final_state[i_n, i_h] = h_state` | `T.barrier_all()` → `T.copy(h_state_ub_float[j], ht[...])` （从 float32 缓冲直接存储） |

### 3.2 关键 API 说明

- **`T.gemm_v0(A, B, C, transpose_A, transpose_B, init)`**：标准 GEMM，`init=True` 表示清零累加器
- **`T.tile.sub/exp/mul/add(dst, src1, src2)`**：Vector 核 element-wise 原语
- **`T.tile.fill(dst, value)`**：Vector 核填充原语
- **`T.tile.broadcast(dst, src, axis)`**：Vector 核广播原语
- **`T.tile.compare(mask_dst, src, scalar, mode)`**：Vector 核比较原语，生成 uint8 mask
- **`T.tile.select(dst, mask, true_val, false_val, mode)`**：Vector 核条件选择原语，根据 mask 选择值
- **`T.copy(src, dst)`**：自动推断层级间数据搬运，支持 float32→bf16 类型转换
- **`T.if_then_else(cond, true_val, false_val)`**：运行时条件选择，支持动态 shape 计算

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 张量 | Shape (Kernel 参数) | Dtype | 说明 |
|------|---------------------|-------|------|
| h | `[N, NT_max, H, K, V]` | bfloat16 | 每个 chunk 的隐藏状态（既是输入 shape 参考也是输出） |
| k | `[T_total, Hg, K]` | bfloat16 | Key 向量（已 flatten） |
| v | `[T_total, H, V]` | bfloat16 | Value 向量（已 flatten） |
| w | `[T_total, H, K]` | bfloat16 | 门控权重（已 flatten） |
| g | `[H, T_total]` | float32 | 门控向量（已 transpose，可选） |
| v_new | `[T_total, H, V]` | bfloat16 | 更新后的 value（既是输入 shape 参考也是输出） |
| h0 | `[N, H, K, V]` | float32 | 初始状态（zero tensor 兜底） |
| ht | `[N, H, K, V]` | float32 | 最终隐藏状态（既是输入 shape 参考也是输出） |
| cu_seqlens | `[N+1]` | int32 | 变长序列边界 |

> **注意**：Python wrapper 层的输入仍为 `[1, T_total, Hg, K]` 等带 batch 维度的格式，wrapper 内部负责 flatten 和 transpose。g 在 wrapper 中执行 `.float().transpose(0, 1).contiguous()` 转为 `[H, T_total]`。

> **⚠️ g shape 为 `[H, T_total]` 而非 `[T_total, H]` 的原因**：`T.copy` 接口要求源张量的切片维度为最内层连续维度。若 g shape 为 `[T_total, H]`，加载时需使用 `T.copy(g[vec_global_start:vec_global_start+vec_chunk_len, i_h], ...)`，此时切片沿第一维（非连续维度）进行，`T.copy` 接口不支持此类非连续维度切片。将 g transpose 为 `[H, T_total]` 后，加载变为 `T.copy(g[i_h, vec_global_start:vec_global_start+vec_chunk_len], ...)`，切片沿最内层连续维度（T_total）进行，符合 `T.copy` 接口要求。

### 4.2 输出张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| h | `[N, NT_max, H, K, V]` | bfloat16 | 每个 chunk 的隐藏状态 |
| v_new | `[T_total, H, V]` | bfloat16 | 更新后的 value |
| ht | `[N, H, K, V]` | float32 | 最终隐藏状态（可选） |

> **注意**：Python wrapper 层对 h 输出会根据 `chunk_offsets` 切片拼接，最终返回 `[1, NT_total, H, K, V]` 格式。

### 4.3 内存层级规划（Expert 模式显式分配）

```
GM (全局内存)
  │
  ├─ T.copy → L1 (T.alloc_L1) — 双缓冲 [2,...]
  │   ├─ k_chunk_l1:    [2, BT, K]        — chunk 内 k 数据（Cube 核输入）
  │   ├─ w_chunk_l1:    [2, BT, K]        — chunk 内 w 数据（Cube 核输入）
  │   ├─ h_state_l1:    [2, K, V_half]    — 累积状态（Cube 核输入）
  │   └─ v_new_l1:      [2, BT, V_half]   — v_new 数据（Cube 核输入）
  │
  ├─ T.copy → UB (T.alloc_ub) — 双缓冲 [2,...]，V 分半 = V_half
  │   ├─ h_state_ub:         [2, K//2, V_half]  — 状态缓冲（bfloat16，仅用于 Cube 传输和输出存储）
  │   ├─ h_state_ub_float:   [2, K//2, V_half]  — 状态缓冲（float32，主计算缓冲）
  │   ├─ hupd_ub_float:      [2, K//2, V_half]  — h_update 缓冲（float32）
  │   ├─ wh_ub_float:        [2, BT//2, V_half] — w @ h 结果（float32）
  │   ├─ v_chunk_ub:         [2, 2, BT//2, V_half] — chunk v 数据（bfloat16, j 分半）
  │   ├─ v_chunk_ub_float:   [2, BT//2, V_half] — 用于 float32 计算的 v 缓冲
  │   └─ (门控相关 buffer，USE_G=True 时)
  │       ├─ g_chunk_ub:     [2, BT//2]          — chunk 内 g 数据（float32）
  │       ├─ g_last_scalar:  [1]                 — chunk 最后有效 token 的 g（float32）
  │       ├─ g_exp_ub:       [BT//2]             — exp(g_last - g) 中间值（float32）
  │       ├─ g_exp_ub_broc:  [BT//2, V_half]    — 广播后的 exp 值（float32）
  │       ├─ g_exp_ub_pad:   [BT]                — 256B 对齐的 exp 中间值（float32，用于 compare/select）
  │       └─ g_mask_ub_pad:  [BT//8]             — compare 生成的 uint8 mask（256B 对齐）
  │
  ├─ T.copy → L0C (T.alloc_L0C) — 双缓冲 [2,...]
  │   ├─ wh_frag:          [2, BT, V_half]  — GEMM 输出（w @ h）
  │   └─ hupd_frag:        [2, K, V_half]   — GEMM 输出（k.T @ v_new）
  │
  └─ Workspace (GM, workspace_idx=[9, 10, 11, 12])
      ├─ ws_wh:   [N, H, 2, BT, V_half] — w @ h 中间结果（float32）
      ├─ ws_vnew: [N, H, 2, BT, V_half] — v_new 中间结果（bfloat16）
      ├─ ws_hupd: [N, H, 2, K, V_half]  — h_update 中间结果（float32）
      └─ ws_h:    [N, H, 2, K, V_half]  — h_state 中间结果（bfloat16）
```

### 4.4 数据搬运路径

```
k: GM → k_chunk_l1[pid] (L1) → GEMM 输入（仅搬运 actual_len/next_len 行，非固定 BT）
w: GM → w_chunk_l1[pid] (L1) → GEMM 输入（仅搬运 actual_len/next_len 行）
v: GM → v_chunk_ub[pid, j] (UB) → v_chunk_ub_float[j] (UB) → element-wise 输入（varlen-aware 切分）
g: GM → g_chunk_ub[pid] (UB) → 门控计算（varlen-aware 切分）
h0: GM → h_state_ub_float[j] (UB) → 初始状态（直接加载到 float32 缓冲）

h_state → Cube: h_state_ub_float[j](UB,float32) → h_state_ub[j](UB,bf16) → ws_h(GM) → h_state_l1[j](L1) → GEMM 输入
同时: h_state_ub[j](UB,bf16) → h 输出(GM)

w @ h: w_chunk_l1[pid](L1) × h_state_l1[j](L1) → wh_frag[j](L0C) → ws_wh(GM) → wh_ub_float[j](UB)
v - wh: v_chunk_ub_float[j](UB) - wh_ub_float[j](UB) → v_chunk_ub_float[j](UB)
门控 compare/select: g_exp_ub(UB) → g_exp_ub_pad(UB,256B aligned) → compare → g_mask_ub_pad → select → g_exp_ub_pad → g_exp_ub
门控 exp/broadcast: g_exp_ub → exp → g_exp_ub → broadcast → g_exp_ub_broc
门控缩放: v_chunk_ub_float[j](UB) × g_exp_ub_broc(UB) → v_chunk_ub_float[j](UB)
状态衰减: h_state_ub_float[j](UB) × g_last_scalar → h_state_ub_float[j](UB)
k @ v_new: k_chunk_l1[pid](L1) × v_new_l1[j](L1, :chunk_len rows) → hupd_frag[j](L0C) → ws_hupd(GM) → hupd_ub_float[j](UB)
状态累积: h_state_ub_float[j](UB) + hupd_ub_float[j](UB) → h_state_ub_float[j](UB)

ht: h_state_ub_float[j](UB) → ht(GM)（float32 直接存储）
```

### 4.5 L1 容量估算（典型配置: K=128, V=128, BT=64）

```
k_chunk_l1:    2×64×128×2B = 32KB
w_chunk_l1:    2×64×128×2B = 32KB
h_state_l1:    2×128×64×2B  = 32KB  (K=128, V_half=64)
v_new_l1:      2×64×64×2B   = 16KB  (BT=64, V_half=64)
总计 ≈ 112KB < 524KB（安全范围内）
```

### 4.6 UB 容量估算（典型配置: K=128, V=128, BT=64）

```
h_state_ub:       2×64×64×2B = 16384B = 16KB
h_state_ub_float: 2×64×64×4B = 32768B = 32KB
hupd_ub_float:    2×64×64×4B = 32768B = 32KB
wh_ub_float:      2×32×64×4B = 16384B = 16KB
v_chunk_ub:       2×2×32×64×2B = 16384B = 16KB
v_chunk_ub_float: 2×32×64×4B = 16384B = 16KB
g_chunk_ub:       2×32×4B = 256B
g_last_scalar:    1×4B = 4B
g_exp_ub:         32×4B = 128B
g_exp_ub_broc:    32×64×4B = 8192B = 8KB
g_exp_ub_pad:     64×4B = 256B (BT=64, 256B aligned for compare)
g_mask_ub_pad:    8×1B = 8B (BT//8=8, uint8 mask)
总计 ≈ 137404B ≈ 134KB（UB 容量 192KB，使用率约 70%）
```

---

## 5. Tiling 策略

### 5.1 Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `T.Kernel(CUBE_BLOCK_NUM, is_npu=True)` | CUBE_BLOCK_NUM = VEC_CORE_NUM // VEC_NUM = 48 // 2 = 24 个 Cube+Vector pair |
| **Work-queue** | 每个 pair 处理 `num_pairs = min(pairs_left, pairs_per_core)` 个 (i_n, i_h) 组合 | pairs_per_core = ceildiv(N*H, total_tasks)，动态分配 |
| **Chunk** | `BT = 64` | 固定 chunk size |
| **V 分块** | `V_half = V // 2` | V=128 分为两半各 64，由 `vid` (Vector sub-block) 控制，vid=0 处理前 BT//2 个 token，vid=1 处理后 BT//2 个 token |
| **K 分块** | `K // 2` | h_state 在 UB 中按 K//2 分半，每个 Vector 核只存储 K//2 × V_half |

### 5.2 Tile Shape 设计（典型配置: K=128, V=128, BT=64）

| Buffer | Shape | 说明 |
|--------|-------|------|
| k_chunk_l1 | `[2, 64, 128]` | 双缓冲，BT=64 × K=128 |
| w_chunk_l1 | `[2, 64, 128]` | 双缓冲，BT=64 × K=128 |
| h_state_l1 | `[2, 128, 64]` | 双缓冲，K=128 × V_half=64 |
| wh_frag | `[2, 64, 64]` | 双缓冲，w[64,128] @ h[128,64] → [64,64] |
| v_new_l1 | `[2, 64, 64]` | 双缓冲，BT=64 × V_half=64 |
| hupd_frag | `[2, 128, 64]` | 双缓冲，k.T[128,64] @ v_new[64,64] → [128,64] |
| h_state_ub | `[2, 64, 64]` | 双缓冲，K//2 × V_half，V 分半处理（bfloat16） |
| h_state_ub_float | `[2, 64, 64]` | 双缓冲，K//2 × V_half，float32 主计算缓冲 |
| wh_ub_float | `[2, 32, 64]` | 双缓冲，BT//2 × V_half，float32 |
| v_chunk_ub | `[2, 2, 32, 64]` | 双缓冲 × j分半，BT//2 × V_half，bfloat16 |
| g_exp_ub_pad | `[64]` | BT=64，256B aligned，float32 |
| g_mask_ub_pad | `[8]` | BT//8=8，uint8 mask |

---

## 6. 循环与调度结构

### 6.1 Kernel 结构设计

```python
VEC_NUM = 2
VEC_CORE_NUM = 48
CUBE_BLOCK_NUM = VEC_CORE_NUM // VEC_NUM
total_tasks = CUBE_BLOCK_NUM

V_half = V // 2

N_sym = T.symbolic("n_batch")
NT_sym = T.symbolic("nt")
T_total_sym = T.symbolic("total_t")

SEM_WH_C2V = 0
SEM_VNEW_V2C = 2
SEM_HUPD_C2V = 4
SEM_H_V2C = 6

@tilelang.jit(workspace_idx=[9, 10, 11, 12], pass_configs=pass_configs)
def chunk_gated_delta_rule_fwd_kernel(
    H, Hg, K, V,
    input_dtype="bfloat16", accum_dtype="float32",
    bt=64, use_g=True,
    store_final_state=True, save_new_value=True,
):

    @T.prim_func
    def main(
        h: T.Tensor([N_sym, NT_sym, H, K, V], input_dtype),
        k: T.Tensor([T_total_sym, Hg, K], input_dtype),
        v: T.Tensor([T_total_sym, H, V], input_dtype),
        w: T.Tensor([T_total_sym, H, K], input_dtype),
        g: T.Tensor([H, T_total_sym], accum_dtype),
        v_new: T.Tensor([T_total_sym, H, V], input_dtype),
        h0: T.Tensor([N_sym, H, K, V], accum_dtype),
        ht: T.Tensor([N_sym, H, K, V], accum_dtype),
        cu_seqlens: T.Tensor([N_sym + 1], "int32"),
        ws_wh: T.Tensor([N_sym, H, 2, bt, V_half], accum_dtype),
        ws_vnew: T.Tensor([N_sym, H, 2, bt, V_half], input_dtype),
        ws_hupd: T.Tensor([N_sym, H, 2, K, V_half], accum_dtype),
        ws_h: T.Tensor([N_sym, H, 2, K, V_half], input_dtype),
    ):
        with T.Kernel(total_tasks, is_npu=True) as (cid, vid):
            total_pairs = N_sym * H
            pairs_per_core = T.ceildiv(total_pairs, total_tasks)
            pair_start = cid * pairs_per_core
            pairs_left = T.if_then_else(total_pairs > pair_start, total_pairs - pair_start, 0)
            num_pairs = T.if_then_else(pairs_left < pairs_per_core, pairs_left, pairs_per_core)

            # === Buffer 分配 ===
            h_state_ub = T.alloc_ub([2, K // 2, V_half], input_dtype)
            h_state_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            hupd_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            wh_ub_float = T.alloc_ub([2, bt // 2, V_half], accum_dtype)

            v_chunk_ub = T.alloc_ub([2, 2, bt // 2, V_half], input_dtype)
            v_chunk_ub_float = T.alloc_ub([2, bt // 2, V_half], accum_dtype)

            g_chunk_ub = T.alloc_ub([2, bt // 2], accum_dtype)
            g_last_scalar = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([bt // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([bt // 2, V_half], accum_dtype)

            g_exp_ub_pad = T.alloc_ub([bt], accum_dtype)  # 256B aligned for compare
            g_mask_ub_pad = T.alloc_ub([bt // 8], "uint8")

            k_chunk_l1 = T.alloc_L1([2, bt, K], input_dtype)
            w_chunk_l1 = T.alloc_L1([2, bt, K], input_dtype)
            h_state_l1 = T.alloc_L1([2, K, V_half], input_dtype)
            wh_frag = T.alloc_L0C([2, bt, V_half], accum_dtype)
            v_new_l1 = T.alloc_L1([2, bt, V_half], input_dtype)
            hupd_frag = T.alloc_L0C([2, K, V_half], accum_dtype)

            for pair_idx in T.serial(num_pairs):
                global_idx = pair_start + pair_idx
                i_n = global_idx // H
                i_h = global_idx % H
                hg_ratio = H // Hg
                k_head = i_h // hg_ratio

                T.barrier_all()

                # ==========================================
                # Cube 域：GEMM 计算
                # ==========================================
                with T.Scope("C"):
                    bos = cu_seqlens[i_n]
                    eos = cu_seqlens[i_n + 1]
                    T_len = eos - bos
                    NT_i = T.ceildiv(T_len, bt)

                    actual_len = T.if_then_else(T_len < bt, T_len, bt)
                    T.copy(w[bos : bos + actual_len, i_h, :], w_chunk_l1[0, :, :])
                    T.copy(k[bos : bos + actual_len, k_head, :], k_chunk_l1[0, :, :])
                    T.set_flag("mte2", "m", 0)

                    for i in T.serial(NT_i):
                        pid = i % 2
                        next_pid = (i + 1) % 2
                        chunk_start_next = bos + (i + 1) * bt

                        chunk_len = T.if_then_else(i * bt + bt > T_len, T_len - i * bt, bt)

                        # Prefetch w/k for next chunk (varlen-aware)
                        if i + 1 < NT_i:
                            next_len = T.if_then_else(
                                (i + 1) * bt + bt > T_len, T_len - (i + 1) * bt, bt)
                            T.copy(w[chunk_start_next : chunk_start_next + next_len, i_h, :],
                                   w_chunk_l1[next_pid, :, :])
                            T.copy(k[chunk_start_next : chunk_start_next + next_len, k_head, :],
                                   k_chunk_l1[next_pid, :, :])
                            T.set_flag("mte2", "m", next_pid)

                        # GEMM 1: w @ h_state → wh_frag → ws_wh
                        T.wait_flag("mte2", "m", pid)
                        for j in T.serial(2):
                            T.wait_cross_flag(SEM_H_V2C + j)
                            T.copy(ws_h[i_n, i_h, j, :, :], h_state_l1[j, :, :])
                            T.set_flag("mte2", "m", 2)
                            T.wait_flag("mte2", "m", 2)
                            T.gemm_v0(w_chunk_l1[pid, :, :], h_state_l1[j, :, :],
                                       wh_frag[j, :, :], init=True)
                            T.set_flag("m", "fix", 3)
                            T.wait_flag("m", "fix", 3)
                            T.copy(wh_frag[j, :, :], ws_wh[i_n, i_h, j, :, :])
                            T.set_cross_flag("FIX", SEM_WH_C2V + j)

                        # GEMM 2: k.T @ v_new → hupd_frag → ws_hupd
                        for j in T.serial(2):
                            T.wait_cross_flag(SEM_VNEW_V2C + j)
                            T.copy(ws_vnew[i_n, i_h, j, :chunk_len, :], v_new_l1[j, :, :])
                            T.set_flag("mte2", "m", 4)
                            T.wait_flag("mte2", "m", 4)
                            T.gemm_v0(k_chunk_l1[pid, :, :], v_new_l1[j, :, :],
                                       hupd_frag[j, :, :], transpose_A=True, init=True)
                            T.set_flag("m", "fix", 5)
                            T.wait_flag("m", "fix", 5)
                            T.copy(hupd_frag[j, :, :], ws_hupd[i_n, i_h, j, :, :])
                            T.set_cross_flag("FIX", SEM_HUPD_C2V + j)

                # ==========================================
                # Vector 域：element-wise 计算
                # ==========================================
                with T.Scope("V"):
                    bos = cu_seqlens[i_n]
                    eos = cu_seqlens[i_n + 1]
                    T_len = eos - bos
                    NT_i = T.ceildiv(T_len, bt)

                    # Load h0 into float32 primary buffer
                    for j in T.serial(2):
                        T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                                 j * V_half : (j + 1) * V_half],
                               h_state_ub_float[j, :, :])

                    # Prefetch v/g for chunk 0 (varlen-aware)
                    chunk_len = T.if_then_else(T_len < bt, T_len, bt)
                    vec_chunk_len = T.if_then_else(
                        vid == 0, T.min(bt // 2, chunk_len), T.max(chunk_len - bt // 2, 0))
                    vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)
                    vec_global_start = bos + vec_start_in_chunk

                    for j in T.serial(2):
                        T.copy(v[vec_global_start : vec_global_start + vec_chunk_len,
                                  i_h, j * V_half : (j + 1) * V_half],
                               v_chunk_ub[0, j, :, :])
                    if use_g:
                        T.copy(g[i_h, vec_global_start : vec_global_start + vec_chunk_len],
                               g_chunk_ub[0, :])

                    T.set_flag("mte2", "v", 2)

                    for i in T.serial(NT_i):
                        pid = i % 2
                        next_pid = (i + 1) % 2
                        v_flag_pid = pid + 2
                        v_flag_next = next_pid + 2
                        g_start = bos + i * bt
                        g_start_next = bos + (i + 1) * bt

                        chunk_len = T.if_then_else(i * bt + bt > T_len, T_len - i * bt, bt)
                        vec_chunk_len = T.if_then_else(
                            vid == 0, T.min(bt // 2, chunk_len), T.max(chunk_len - bt // 2, 0))
                        vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)

                        # Prefetch v/g for next chunk (varlen-aware)
                        if i + 1 < NT_i:
                            next_chunk_len = T.if_then_else(
                                (i + 1) * bt + bt > T_len, T_len - (i + 1) * bt, bt)
                            next_vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)
                            next_vec_chunk_len = T.if_then_else(
                                vid == 0, T.min(bt // 2, next_chunk_len),
                                T.max(next_chunk_len - bt // 2, 0))
                            next_vec_global_start = g_start_next + next_vec_start_in_chunk

                            for j in T.serial(2):
                                T.copy(v[next_vec_global_start : next_vec_global_start + next_vec_chunk_len,
                                          i_h, j * V_half : (j + 1) * V_half],
                                       v_chunk_ub[next_pid, j, :, :])
                            if use_g:
                                T.copy(g[i_h, next_vec_global_start : next_vec_global_start + next_vec_chunk_len],
                                       g_chunk_ub[next_pid, :])
                            T.set_flag("mte2", "v", v_flag_next)

                        # Pipeline flush before h_state transfer
                        T.set_flag("mte2", "v", 12)
                        T.wait_flag("mte2", "v", 12)

                        # h_state to Cube: float32 → bf16 → workspace + output
                        for j in T.serial(2):
                            T.copy(h_state_ub_float[j, :, :], h_state_ub[j, :, :])  # float32 → bf16
                            T.set_flag("v", "mte3", 11)
                            T.wait_flag("v", "mte3", 11)
                            T.copy(h_state_ub[j, :, :],
                                   ws_h[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :])
                            T.set_cross_flag("MTE3", SEM_H_V2C + j)
                            # Save h[t] to output
                            T.copy(h_state_ub[j, :, :],
                                   h[i_n, i, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                                     j * V_half : (j + 1) * V_half])

                        # Wait for v/g data ready
                        T.wait_flag("mte2", "v", v_flag_pid)

                        # Gating computation (numerically stable)
                        if use_g:
                            g_last = T.if_then_else(
                                i * bt + bt <= T_len,
                                g[i_h, g_start + bt - 1],
                                g[i_h, g_start + T_len - i * bt - 1])

                            T.tile.fill(g_exp_ub, g_last)
                            T.set_flag("mte2", "v", 4)
                            T.wait_flag("mte2", "v", 4)
                            T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid, :])
                            # Numerical stability: mask out g_last - g > 0 → -inf
                            T.copy(g_exp_ub, g_exp_ub_pad[0 : bt // 2])
                            T.tile.compare(g_mask_ub_pad, g_exp_ub_pad, T.float32(0), "LE")
                            T.tile.select(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad,
                                          -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")
                            T.copy(g_exp_ub_pad[0 : bt // 2], g_exp_ub)
                            T.tile.exp(g_exp_ub, g_exp_ub)
                            T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)

                            T.tile.fill(g_last_scalar, g_last)
                            T.tile.exp(g_last_scalar, g_last_scalar)

                        for j in T.serial(2):
                            T.copy(v_chunk_ub[pid, j, :, :], v_chunk_ub_float[j, :, :])

                            # v_new = v - w @ h
                            T.wait_cross_flag(SEM_WH_C2V + j)
                            T.copy(ws_wh[i_n, i_h, j,
                                          vec_start_in_chunk : vec_start_in_chunk + bt // 2, :],
                                   wh_ub_float[j, :, :])
                            T.set_flag("mte2", "v", 5)
                            T.wait_flag("mte2", "v", 5)
                            T.tile.sub(v_chunk_ub_float[j, :, :],
                                       v_chunk_ub_float[j, :, :], wh_ub_float[j, :, :])

                            # Save v_new to output
                            if save_new_value:
                                T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                                T.set_flag("v", "mte3", 6)
                                T.wait_flag("v", "mte3", 6)
                                T.copy(v_chunk_ub[pid, j, :vec_chunk_len, :],
                                       v_new[g_start + vec_start_in_chunk : g_start + vec_start_in_chunk + vec_chunk_len,
                                             i_h, j * V_half : j * V_half + V_half])

                            if use_g:
                                # v_new *= exp(g_last - g)
                                T.tile.mul(v_chunk_ub_float[j, :, :],
                                           v_chunk_ub_float[j, :, :], g_exp_ub_broc)
                                # h_state *= exp(g_last)
                                T.tile.mul(h_state_ub_float[j, :, :],
                                           h_state_ub_float[j, :, :], g_last_scalar[0])

                            # v_new to Cube workspace
                            T.set_flag("mte3", "v", 7)
                            T.wait_flag("mte3", "v", 7)
                            T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                            T.set_flag("v", "mte3", 8)
                            T.wait_flag("v", "mte3", 8)
                            T.copy(v_chunk_ub[pid, j, :, :],
                                   ws_vnew[i_n, i_h, j,
                                           vec_start_in_chunk : vec_start_in_chunk + bt // 2, :])
                            T.set_cross_flag("MTE3", SEM_VNEW_V2C + j)

                        # h += k.T @ v_new
                        for j in T.serial(2):
                            T.wait_cross_flag(SEM_HUPD_C2V + j)
                            T.copy(ws_hupd[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :],
                                   hupd_ub_float[j, :, :])
                            T.set_flag("mte2", "v", 9)
                            T.wait_flag("mte2", "v", 9)
                            T.tile.add(h_state_ub_float[j, :, :],
                                       h_state_ub_float[j, :, :], hupd_ub_float[j, :, :])

                # Epilogue: store final state ht
                if store_final_state:
                    T.barrier_all()
                    for j in T.serial(2):
                        T.copy(h_state_ub_float[j, :, :],
                               ht[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                                  j * V_half : (j + 1) * V_half])

    return main
```

### 6.2 调度选择

| 循环/操作 | 调度类型 | 理由 |
|----------|---------|------|
| pair 循环 | `T.serial(num_pairs)` | Work-queue 模式，每个 core 处理多个 (i_n, i_h) 对 |
| chunk 循环 | `T.serial(NT_i)` | chunk 间有状态依赖，必须串行；NT_i 为动态值（非固定 NT_max） |
| V 分块 + K 分块 | `vid` 控制，`for j in T.serial(2)` | V 和 K 各分两半，双缓冲 [2,...] |
| element-wise | `T.tile.*` | Vector 核自动向量化 |
| GEMM | `T.gemm_v0` | Cube 核矩阵乘 |
| 数值稳定门控 | `T.tile.compare` + `T.tile.select` | 仅保留 g_last - g <= 0 的值，其余置 -inf |
| varlen 长度计算 | `T.if_then_else` | 动态计算 actual_len, chunk_len, vec_chunk_len |

### 6.3 Python Wrapper 层

```python
def chunk_gated_delta_rule_fwd_h(
    k, w, u, g=None, initial_state=None,
    output_final_state=False, chunk_size=64,
    save_new_value=True, cu_seqlens=None, chunk_offsets=None,
):
    # 1. Flatten/transpose inputs
    #    k: [1, T_total, Hg, K] → [T_total, Hg, K]
    #    w: [1, T_total, H, K] → [T_total, H, K]
    #    u: [1, T_total, H, V] → [T_total, H, V]
    #    g: [1, T_total, H] → [H, T_total] (float32, transposed)
    # 2. Allocate outputs: h_out, v_new_flat, h0, ht
    # 3. Call kernel (workspace auto-allocated by workspace_idx)
    # 4. Format outputs: slice h by chunk_offsets, unsqueeze batch dim
```

---

## 7. 同步策略

### 7.1 手动同步方案

由于 `TL_ASCEND_AUTO_SYNC: False`，所有同步由代码显式控制：

| 同步类型 | API | 用途 |
|---------|-----|------|
| 全核同步 | `T.barrier_all()` | pair 循环开始前同步所有核；存储 ht 前同步 |
| xmb 同步 | `T.set_flag` / `T.wait_flag("mte2", "m" \| "v", id)` | Cube域/Vector域各自内部的数据搬运完成通知 |
| 标量同步 | `T.set_flag` / `T.wait_flag("m" \| "v", "fix" \| "mte3", id)` | GEMM 完成 → L0C 结果可用通知；float32→bf16 转换完成通知 |
| 跨核同步 | `T.set_cross_flag("FIX" \| "MTE3", sem)` / `T.wait_cross_flag(sem)` | Cube ↔ Vector 核间数据搬运完成握手 |
| 管线排空 | `T.set_flag("mte2", "v", 12)` / `T.wait_flag("mte2", "v", 12)` | Vector 域管线排空，确保门控计算完成后再搬运 h_state |

### 7.2 跨核数据流与握手协议

```
                                       Cube 域                      Vector 域
                                        │                              │
 [Vector] h_state_ub_float → h_state_ub → ws_h ──► wait_cross_flag(H_V2C)
                                        │  w @ h GEMM                   │
 [  Cube] ws_wh 搬运完成 ──► set_cross_flag(WH_C2V) ◄── wait_cross_flag(WH_C2V)
                                        │                              │ v - w@h → v_new
                                        │                              │ gating (compare/select/exp)
                                        │                              │ v_new *= exp(g_last-g)
                                        │                              │ h_state *= exp(g_last)
 [Vector] v_chunk_ub_float → v_chunk_ub → ws_vnew ──► set_cross_flag(VNEW_V2C)
                                        │  wait_cross_flag(VNEW_V2C)    │
                                        │  k.T @ v_new GEMM             │
 [  Cube] ws_hupd 搬运完成 ──► set_cross_flag(HUPD_C2V) ◄── wait_cross_flag(HUPD_C2V)
                                        │                              │ h_state_ub_float += hupd
                                        │                              │ → h_state_ub_float  (下一轮)
```

### 7.3 信号量分配

| 常量 | 值 | 方向 | 含义 |
|------|-----|------|------|
| `SEM_H_V2C + j` | 0, 1 | V → C | h_state 就绪，Cube 可开始 w @ h |
| `SEM_WH_C2V + j` | 0, 1 | C → V | w @ h 结果 ws_wh 就绪，Vector 可开始 v - wh |
| `SEM_VNEW_V2C + j` | 2, 3 | V → C | v_new 结果 ws_vnew 就绪，Cube 可开始 k.T @ v_new |
| `SEM_HUPD_C2V + j` | 4, 5 | C → V | h_update 结果 ws_hupd 就绪，Vector 可开始 h += h_upd |

> **注意**：`j ∈ {0, 1}` 对应 V 维度分半，每组 GEMM 用独立的信号量对。

### 7.4 Workspace 设计

为避免频繁的 GM → L1 → L0C → GM 搬运，使用 workspace buffer（GM 驻留），由 `workspace_idx=[9, 10, 11, 12]` 自动分配：

| Workspace | Shape | Dtype | 用途 |
|----------|-------|-------|------|
| ws_wh | `[N, H, 2, BT, V_half]` | float32 | 存储 w @ h 结果，供 Vector 核读取 |
| ws_vnew | `[N, H, 2, BT, V_half]` | bfloat16 | 存储 v_new，供 Cube 核读取 |
| ws_hupd | `[N, H, 2, K, V_half]` | float32 | 存储 k.T @ v_new 结果，供 Vector 核读取 |
| ws_h | `[N, H, 2, K, V_half]` | bfloat16 | 存储 h_state，供 Cube 核读取 |

---

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现）

```python
def ref_chunk_gated_delta_rule(k, w, u, g=None, initial_state=None,
                                output_final_state=False, chunk_size=64,
                                cu_seqlens=None, dtype=torch.bfloat16):
    BT = chunk_size

    k = k.float().squeeze(0)       # [T_total, Hg, K]
    w = w.float().squeeze(0)       # [T_total, H, K]
    u = u.float().squeeze(0)       # [T_total, H, V]
    g = g.float().squeeze(0) if g is not None else None  # [T_total, H]
    initial_state = initial_state.float().squeeze(0) if initial_state is not None else None  # [N, H, K, V]

    T_total, Hg, K = k.shape
    _, H, V = u.shape
    N = len(cu_seqlens) - 1

    NT_total = sum([(int(cu_seqlens[i + 1]) - int(cu_seqlens[i]) + BT - 1) // BT for i in range(N)])

    h = torch.zeros(NT_total, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.zeros(T_total, H, V, dtype=torch.float32, device=k.device)
    final_state = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device) if output_final_state else None

    chunk_offset = 0
    for i_n in range(N):
        bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
        T_len = eos - bos
        NT = (T_len + BT - 1) // BT

        for i_h in range(H):
            h_state = (initial_state[i_n, i_h].clone() if initial_state is not None
                       else torch.zeros(K, V, dtype=torch.float32, device=k.device))
            k_head = i_h // (H // Hg)

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T_len)

                h[chunk_offset + i_t, i_h] = h_state
                k_chunk, w_chunk, v_chunk = (
                    k[bos + t_start : bos + t_end, k_head, :],
                    w[bos + t_start : bos + t_end, i_h, :],
                    u[bos + t_start : bos + t_end, i_h, :],
                )

                v_n = v_chunk - torch.matmul(w_chunk, h_state)
                v_new[bos + t_start : bos + t_end, i_h, :] = v_n

                if g is not None:
                    g_chunk = g[bos + t_start : bos + t_end, i_h]
                    g_last = g_chunk[-1].item()
                    # Numerical stability: only keep g_last - g <= 0, replace positive with -inf
                    v_n = v_n * torch.exp(torch.where(g_last - g_chunk <= 0, g_last - g_chunk, float("-inf")))[:, None]
                    h_state = h_state * torch.exp(torch.tensor(g_last, device=k.device))

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)

            if output_final_state:
                final_state[i_n, i_h] = h_state
        chunk_offset += NT

    return h.to(dtype).unsqueeze(0), v_new.to(dtype).unsqueeze(0), final_state if final_state is not None else None
```

### 8.2 测试配置

| 模式 | 参数配置 | 测试组合 |
|------|---------|---------|
| **Varlen** | seqlens=[16384] (默认) | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[512,512,512,512] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[128,256,512,1024,128] | use_g=True/False, use_initial_state=True/False |

默认测试参数：H=32, Hg=16, K=128, V=128, dtype=bfloat16

输入生成：k/w/u 乘以 INPUT_SCALE=0.01，g 使用 `_chunk_local_cumsum_cpu` 生成 chunk 内累积和后乘以 GATE_SCALE=0.002。

### 8.3 命令行接口

```bash
python examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py \
  --use_g True --use_initial_state False \
  --seqlens 16384 --H 32 --Hg 16 --K 128 --V 128
```

### 8.4 精度容忍度

实际测试中采用 `rtol=1e-4, atol=1e-3`（bfloat16 级别），对所有 seqlens、use_g、use_initial_state 配置均适用。

---

## 9. 注意事项

### 9.1 当前限制

| 限制 | 说明 |
|------|------|
| **K = 128 固定** | 当前典型配置，L1 容量可容纳 |
| **V = 128 固定** | V_half=64 分半处理，需要 V 为大于 0 的偶数 |
| **BT = 64 固定** | chunk size 固定，双缓冲 slot 数 = 2 |
| **Hg ≤ H** | GQA 比例必须满足 |
| **input_dtype = bfloat16** | 默认数据类型为 bfloat16，非 float16 |
| **VEC_CORE_NUM = 48** | 硬件固定 Vector 核数量 |
| **g tensor 已转置** | Kernel 中 g 的 shape 为 `[H, T_total]`，Python wrapper 负责 transpose |

### 9.2 关键实现细节

| 细节 | 说明 |
|------|------|
| **h_state_ub_float 为主缓冲** | h0 直接加载到 float32 缓冲，所有计算在 float32 上进行，仅传输时转为 bf16 |
| **varlen-aware 数据搬运** | v/g 搬运使用 vec_chunk_len 和 vec_start_in_chunk 动态计算，而非固定 BT//2 |
| **数值稳定门控** | 使用 `T.tile.compare("LE")` + `T.tile.select(-inf)` 实现与 PyTorch `torch.where` 等价的数值稳定 exp 运算 |
| **g_exp_ub_pad 256B 对齐** | compare/select 操作需要 256B 对齐的 buffer，因此使用 `[BT]` 而非 `[BT//2]` 的 padding buffer |
| **Work-queue 调度** | 每个 Cube+Vector pair 处理多个 (i_n, i_h)，而非 1:1 映射 |

---

## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/chunk_gated_delta_rule/design.md` | 本文档 |
| 算子实现 | `examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py` | 已完成 |