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
       v_new[t] = v_new[t] * exp(g_last - g[t])  # 门控缩放
       h[t] = h[t] * exp(g_last)                  # 状态衰减
  4. h[t+1] = h[t] + k[t] @ v_new[t]          # 状态更新 (GEMM)
```

其中：
- `@` 表示矩阵乘法
- `g_last = g[t_valid - 1]` 为 chunk 内最后一个**有效** token 的 gate 值（支持未 padding 的变长序列）
- `exp(g_last - g[t])` 为数值稳定的指数运算（避免溢出）
- 核心计算为 **两次 GEMM + 多步 element-wise 门控运算**

### 1.3 计算特征分析

| 维度 | 分析结果 |
|------|---------|
| **计算类型** | 混合（Cube GEMM + Vector element-wise） |
| **复杂度级别** | 多步（GEMM → sub → exp → mul → GEMM → add，需中间缓冲） |
| **动态 shape** | T 为符号维度；支持变长序列（cu_seqlens） |
| **核间协作** | 需要 Cube 核做 GEMM，Vector 核做 element-wise 门控 |

### 1.4 典型配置示例

| 参数 | 值 | 说明 |
|------|-----|------|
| N | 1 | 序列数 |
| T_total | 2048 | 总 token 数 |
| H | 8 | value head 数 |
| Hg | 4 | key head 数（GQA，H/Hg=2） |
| K | 128 | key dim |
| V | 128 | value dim |
| BT | 64 | chunk size（固定） |
| NT_max | 32 | 最大 chunk 数 = ceil(2048/64) |

---

## 2. 编程模式选型

### 2.1 选型结论：**Expert 模式**

### 2.2 选型理由

| 因素 | 分析 |
|------|------|
| 含 GEMM 计算 | 需要 `T.gemm_v0`，涉及 L0A/L0B/L0C 寄存器管理 |
| 多步 element-wise | `v_new = v - w@h`、`exp`、`mul`、`add` 需要中间 buffer |
| 状态累积 | `h` 在 chunk 间累积，需要精细管理 buffer 生命周期 |
| V 维度分块 | V 分为两半 [0:V//2] 和 [V//2:V]，需要显式分配 L1/UB/L0 buffer |
| 混合核计算 | Cube 核做 GEMM，Vector 核做 element-wise，需要显式 buffer 类型指定 |

Expert 模式的优势：
- 使用 `T.alloc_L1/ub/L0C` 显式控制内存层级，避免编译器自动规划的不确定性
- 使用 `T.tile.*` 原语用于 element-wise 计算
- 显式控制数据搬运路径（GM → L1 → UB）

### 2.3 pass_configs 配置

```python
# 全部关闭，采用手动同步策略：
# - 手动 barrier (T.wait_flag / T.set_flag)
# - 手动 CV 分离 (T.Scope("C") / T.Scope("V"))
# - 手动跨核同步 (T.set_cross_flag / T.wait_cross_flag)
# - 手动内存管理 (T.alloc_L1/ub/L0C 显式分配)
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
```

---

## 3. API 映射设计

### 3.1 核心计算步骤 → TileLang API 映射

| 计算步骤 | PyTorch 参考 | TileLang API |
|---------|-------------|--------------|
| 加载初始状态 h0 | `h_state = h0.clone()` | `T.copy(h0[i_n, i_h, K//2*vid:K//2*vid+K//2, j*V_half:(j+1)*V_half], h_state_ub[j])` |
| 加载 w | `w_chunk = w[t_start:t_end]` | `T.copy(w[g_start:g_start+BT, i_h, :], w_chunk_l1[pid, :, :])` |
| 加载 k | `k_chunk = k[t_start:t_end]` | `T.copy(k[g_start:g_start+BT, k_head, :], k_chunk_l1[pid, :, :])` |
| 加载 v | `v_chunk = v[t_start:t_end]` | `T.copy(v[g_start+BT//2*vid:..., i_h, j*V_half:(j+1)*V_half], v_chunk_ub[pid, j])` |
| 加载 g | `g_chunk = g[t_start:t_end]` | `T.copy(g[g_start+BT//2*vid:..., i_h], g_chunk_ub[pid])` |
| **GEMM: w @ h** | `torch.matmul(w, h)` | `T.gemm_v0(w_chunk_l1[pid], h_state_l1[j], wh_frag[j], init=True)` |
| **v_new = v - wh** | `v - torch.matmul(w, h)` | `T.copy(v_chunk_ub[pid, j], v_chunk_ub_float[j])` → `T.tile.sub(v_chunk_ub_float[j], v_chunk_ub_float[j], wh_ub_float[j])` |
| **exp(g_last - g)** | `torch.exp(g_last - g)` | `T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid])` → `T.tile.exp(g_exp_ub, g_exp_ub)` |
| **门控缩放** | `v_new * exp(g)` | `T.tile.mul(v_chunk_ub_float[j], v_chunk_ub_float[j], g_exp_ub_broc)` |
| **状态衰减** | `h * exp(g_last)` | `T.tile.mul(h_state_ub_float[j], h_state_ub_float[j], g_last_scalar[0])` |
| **GEMM: k @ v_new** | `torch.matmul(k.T, v_new)` | `T.gemm_v0(k_chunk_l1[pid], v_new_l1[j], hupd_frag[j], transpose_A=True, init=True)` |
| **状态累积** | `h = h + k.T @ v_new` | `T.tile.add(h_state_ub_float[j], h_state_ub_float[j], hupd_ub_float[j])` |
| 存储 h[t] | `h[chunk_idx] = h_state` | `T.copy(h_state_ub[j], h[i_n, i, i_h, K//2*vid:K//2*vid+K//2, j*V_half:(j+1)*V_half])` |
| 存储 v_new | `v_new[t_start:t_end] = v_n` | `T.copy(v_chunk_ub[pid, j], v_new[g_start+BT//2*vid:..., i_h, j*V_half:(j+1)*V_half])` |
| 存储 ht | `final_state[i_n, i_h] = h_state` | `T.copy(h_state_ub[j], ht[i_n, i_h, K//2*vid:K//2*vid+K//2, j*V_half:(j+1)*V_half])` |

### 3.2 关键 API 说明

- **`T.gemm_v0(A, B, C, transpose_A, transpose_B, init)`**：标准 GEMM，`init=True` 表示清零累加器
- **`T.tile.sub/exp/mul/add(dst, src1, src2)`**：Vector 核 element-wise 原语
- **`T.tile.fill(dst, value)`**：Vector 核填充原语
- **`T.tile.broadcast(dst, src)`**：Vector 核广播原语
- **`T.copy(src, dst)`**：自动推断层级间数据搬运

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 张量 | Shape (变长) | Dtype | 说明 |
|------|-------------|-------|------|
| k | `[1, T_total, Hg, K]` | float16 | Key 向量 |
| w | `[1, T_total, H, K]` | float16 | 门控权重 |
| u (v) | `[1, T_total, H, V]` | float16 | Value 向量 |
| g | `[1, T_total, H]` | float32 | 门控向量（可选） |
| h0 | `[1, N, H, K, V]` | float16 | 初始状态（zero tensor 兜底） |
| cu_seqlens | `[N+1]` | int32 | 变长序列边界 |

### 4.2 输出张量

| 张量 | Shape (变长) | Dtype | 说明 |
|------|-------------|-------|------|
| h | `[1, NT_total, H, K, V]` | float16 | 每个 chunk 的隐藏状态 |
| v_new | `[1, T_total, H, V]` | float16 | 更新后的 value |
| ht | `[N, H, K, V]` | float16 | 最终隐藏状态（可选） |

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
  │   ├─ h_state_ub:         [2, K//2, V_half]  — 状态缓冲（float16）
  │   ├─ h_state_ub_float:   [2, K//2, V_half]  — 状态缓冲（float32）
  │   ├─ hupd_ub_float:      [2, K//2, V_half]  — h_update 缓冲（float32）
  │   ├─ wh_ub_float:        [2, BT//2, V_half] — w @ h 结果（float32）
  │   ├─ v_chunk_ub:         [2, 2, BT//2, V_half] — chunk v 数据（float16, j 分半）
  │   ├─ v_chunk_ub_float:   [2, BT//2, V_half] — 用于 float32 计算的 v 缓冲
  │   └─ (门控相关 buffer，USE_G=True 时)
  │       ├─ g_chunk_ub:     [2, BT//2]          — chunk 内 g 数据（float32）
  │       ├─ g_last_scalar:  [1]                 — chunk 最后有效 token 的 g
  │       ├─ g_exp_ub:       [BT//2]             — exp(g_last - g) 中间值
  │       ├─ g_exp_ub_broc:  [BT//2, V_half]    — 广播后的 exp 值
  │       └─ tmp_ub:         编译临时缓冲          — broadcast 内部临时空间
  │
  ├─ T.copy → L0C (T.alloc_L0C) — 双缓冲 [2,...]
  │   ├─ wh_frag:          [2, BT, V_half]  — GEMM 输出（w @ h）
  │   └─ hupd_frag:        [2, K, V_half]   — GEMM 输出（k.T @ v_new）
  │
  └─ Workspace (GM)
      ├─ ws_wh:   [N, H, 2, BT, V_half] — w @ h 中间结果（float32）
      ├─ ws_vnew: [N, H, 2, BT, V_half] — v_new 中间结果（float16）
      ├─ ws_hupd: [N, H, 2, K, V_half]  — h_update 中间结果（float32）
      └─ ws_h:    [N, H, 2, K, V_half]  — h_state 中间结果（float16）
```

### 4.4 数据搬运路径

```
k: GM → k_chunk_l1[pid] (L1) → GEMM 输入
w: GM → w_chunk_l1[pid] (L1) → GEMM 输入
v: GM → v_chunk_ub[pid, j] (UB) → v_chunk_ub_float[j] (UB) → element-wise 输入
g: GM → g_chunk_ub[pid] (UB) → 门控计算
h0: GM → h_state_ub[j] (UB) → 初始状态

w @ h: w_chunk_l1[pid](L1) × h_state_l1[j](L1) → wh_frag[j](L0C) → ws_wh(GM) → wh_ub_float[j](UB)
v - wh: v_chunk_ub_float[j](UB) - wh_ub_float[j](UB) → v_chunk_ub_float[j](UB)
门控: v_chunk_ub_float[j](UB) × g_exp_ub_broc(UB) → v_chunk_ub_float[j](UB)
状态衰减: h_state_ub_float[j](UB) × g_last_scalar → h_state_ub_float[j](UB)
k @ v_new: k_chunk_l1[pid](L1) × v_new_l1[j](L1) → hupd_frag[j](L0C) → ws_hupd(GM) → hupd_ub_float[j](UB)
状态累积: h_state_ub_float[j](UB) + hupd_ub_float[j](UB) → h_state_ub[j](UB)

h_state: UB → GM (h 输出)
v_new: UB → GM (v_new 输出)
ht: UB → GM (ht 输出)
```

### 4.5 L1 容量估算（典型配置: K=128, V=128）

```
k_chunk_l1:    2×64×128×2B = 32KB
w_chunk_l1:    2×64×128×2B = 32KB
h_state_l1:    2×128×64×2B  = 32KB  (K=128, V_half=64)
v_new_l1:      2×64×64×2B   = 16KB  (BT=64, V_half=64)
总计 ≈ 112KB < 524KB（安全范围内）
```

### 4.6 UB 容量估算（典型配置: K=128, V=128）

```
h_state_ub:       2×64×64×2B = 16384B = 16KB
v_chunk_ub:       2×2×32×64×2B = 16384B = 16KB
g_chunk_ub:       2×32×4B = 256B
g_exp_ub:         32×4B = 128B
g_exp_ub_broc:    32×64×4B = 8192B = 8KB
tmp_ub:           2048B = 2KB                    (broadcast 内部临时)
g_last_scalar:    1×4B = 4B
h_state_ub_float: 2×64×64×4B = 32768B = 32KB
v_chunk_ub_float: 2×32×64×4B = 16384B = 16KB
wh_ub_float:      2×32×64×4B = 16384B = 16KB
hupd_ub_float:    2×64×64×4B = 32768B = 32KB
(编译器内部 padding: ~28B)
总计 ≈ 141728B ≈ 139KB（UB 容量 192KB，使用率约 72%）
```

---

## 5. Tiling 策略

### 5.1 Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `T.Kernel(N * H, is_npu=True)` | 每个 (序列索引, head) 组合一个 block |
| **Chunk** | `BT = 64` | 固定 chunk size |
| **V 分块** | `V_half = V // 2` | V=128 分为两半各 64，由 `vid` (Vector sub-block) 控制 |

### 5.2 Tile Shape 设计（典型配置: K=128, V=128）

| Buffer | Shape | 说明 |
|--------|-------|------|
| k_chunk_l1 | `[2, 64, 128]` | 双缓冲，BT=64 × K=128 |
| w_chunk_l1 | `[2, 64, 128]` | 双缓冲，BT=64 × K=128 |
| h_state_l1 | `[2, 128, 64]` | 双缓冲，K=128 × V_half=64 |
| wh_frag | `[2, 64, 64]` | 双缓冲，w[64,128] @ h[128,64] → [64,64] |
| v_new_l1 | `[2, 64, 64]` | 双缓冲，BT=64 × V_half=64 |
| hupd_frag | `[2, 128, 64]` | 双缓冲，k.T[128,64] @ v_new[64,64] → [128,64] |
| h_state_ub | `[2, 64, 64]` | 双缓冲，K//2 × V_half，V 分半处理 |
| wh_ub_float | `[2, 32, 64]` | 双缓冲，BT//2 × V_half，float32 |
| g_chunk_ub | `[2, 32]` | 双缓冲，BT//2 |

---

## 6. 循环与调度结构

### 6.1 Kernel 结构设计

```python
@tilelang.jit(workspace_idx=[9, 10, 11, 12], pass_configs=pass_configs)
def chunk_gated_delta_rule_fwd_kernel(
    N, H, T_total_pad, Hg, K, V, NT_max,
    BT=64, USE_G=True,
    STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True,
    dtype="float16", accum_dtype="float32",
):
    V_half = V // 2

    # 跨核同步信号量编号（Cube ↔ Vector）
    SEM_WH_C2V = 0
    SEM_VNEW_V2C = 2
    SEM_HUPD_C2V = 4
    SEM_H_V2C = 6

    @T.prim_func
    def main(h, k, v, w, g, v_new, h0, ht, cu_seqlens, ws_wh, ws_vnew, ws_hupd, ws_h):
        with T.Kernel(N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H
            hg_ratio = H // Hg
            k_head = i_h // hg_ratio

            # === Buffer 分配（双缓冲 [2,...]） ===
            h_state_ub = T.alloc_ub([2, K // 2, V_half], dtype)
            h_state_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            hupd_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            wh_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)
            v_chunk_ub = T.alloc_ub([2, 2, BT // 2, V_half], dtype)
            v_chunk_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)
            g_chunk_ub = T.alloc_ub([2, BT // 2], accum_dtype)
            g_last_scalar = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([BT // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([BT // 2, V_half], accum_dtype)
            k_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            w_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            h_state_l1 = T.alloc_L1([2, K, V_half], dtype)
            wh_frag = T.alloc_L0C([2, BT, V_half], accum_dtype)
            v_new_l1 = T.alloc_L1([2, BT, V_half], dtype)
            hupd_frag = T.alloc_L0C([2, K, V_half], accum_dtype)

            # ==========================================
            # Cube 域：GEMM 计算
            # ==========================================
            with T.Scope("C"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, BT)

                # 预取 chunk 0 的 w/k 到 L1
                if NT_i > 0:
                    T.copy(w[i_h, bos : bos + BT, :], w_chunk_l1[0, :, :])
                    T.copy(k[k_head, bos : bos + BT, :], k_chunk_l1[0, :, :])
                    T.set_flag("mte2", "m", 0)

                for i in T.serial(NT_max):
                    if i < NT_i:
                        pid = i % 2
                        next_pid = (i + 1) % 2
                        g_start_next = bos + (i + 1) * BT

                        # 提前搬运 chunk i+1 的 w/k
                        if i + 1 < NT_i:
                            T.copy(w[i_h, g_start_next : g_start_next + BT, :], w_chunk_l1[next_pid, :, :])
                            T.copy(k[k_head, g_start_next : g_start_next + BT, :], k_chunk_l1[next_pid, :, :])
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
                            T.copy(ws_vnew[i_n, i_h, j, :, :], v_new_l1[j, :, :])
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
                NT_i = T.ceildiv(T_len, BT)

                # 加载初始状态 h0
                for j in T.serial(2):
                    T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                             j * V_half : (j + 1) * V_half], h_state_ub[j, :, :])

                # 预取 chunk 0 的 v/g
                if NT_i > 0:
                    for j in T.serial(2):
                        T.copy(v[i_h, bos + BT // 2 * vid : bos + BT // 2 * vid + BT // 2,
                                 j * V_half : (j + 1) * V_half], v_chunk_ub[0, j, :, :])
                    if USE_G:
                        T.copy(g[i_h, bos + BT // 2 * vid : bos + BT // 2 * vid + BT // 2],
                               g_chunk_ub[0, :])
                    T.set_flag("mte2", "v", 2)

                for i in T.serial(NT_max):
                    if i < NT_i:
                        pid = i % 2
                        next_pid = (i + 1) % 2
                        v_flag_pid = pid + 2
                        v_flag_next = next_pid + 2
                        g_start = bos + i * BT
                        g_start_next = bos + (i + 1) * BT

                        # 提前搬运 chunk i+1 的 v/g
                        if i + 1 < NT_i:
                            for j in T.serial(2):
                                T.copy(
                                    v[i_h, g_start_next + BT // 2 * vid :
                                          g_start_next + BT // 2 * vid + BT // 2,
                                      j * V_half : (j + 1) * V_half],
                                    v_chunk_ub[next_pid, j, :, :],
                                )
                            if USE_G:
                                T.copy(
                                    g[i_h, g_start_next + BT // 2 * vid :
                                          g_start_next + BT // 2 * vid + BT // 2],
                                    g_chunk_ub[next_pid, :]
                                )
                            T.set_flag("mte2", "v", v_flag_next)

                        # 将 h_state 搬运到 workspace，通知 Cube 核
                        for j in T.serial(2):
                            T.copy(h_state_ub[j, :, :],
                                   ws_h[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :])
                            T.set_cross_flag("MTE3", SEM_H_V2C + j)

                        # 保存当前 h[t] 到输出
                        for j in T.serial(2):
                            T.copy(h_state_ub[j, :, :],
                                   h[i_n, i, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                                     j * V_half : (j + 1) * V_half])

                        # 等待 v/g 数据就绪
                        T.wait_flag("mte2", "v", v_flag_pid)

                        # 门控计算
                        if USE_G:
                            # 提取最后一个有效 token 的 gate
                            g_last = T.if_then_else(
                                i * BT + BT <= T_len,
                                g[i_h, g_start + BT - 1],
                                g[i_h, g_start + T_len - i * BT - 1])
                            # exp(g_last - g)
                            T.tile.fill(g_exp_ub, g_last)
                            T.set_flag("mte2", "v", 4)
                            T.wait_flag("mte2", "v", 4)
                            T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid, :])
                            T.tile.exp(g_exp_ub, g_exp_ub)
                            T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)
                            # exp(g_last) 标量
                            T.tile.fill(g_last_scalar, g_last)
                            T.tile.exp(g_last_scalar, g_last_scalar)

                        for j in T.serial(2):
                            T.copy(v_chunk_ub[pid, j, :, :], v_chunk_ub_float[j, :, :])

                            # v_new = v - w @ h
                            T.wait_cross_flag(SEM_WH_C2V + j)
                            T.copy(ws_wh[i_n, i_h, j, BT // 2 * vid :
                                          BT // 2 * vid + BT // 2, :],
                                   wh_ub_float[j, :, :])
                            T.set_flag("mte2", "v", 5)
                            T.wait_flag("mte2", "v", 5)
                            T.tile.sub(v_chunk_ub_float[j, :, :],
                                       v_chunk_ub_float[j, :, :], wh_ub_float[j, :, :])

                            # 保存 v_new 到输出
                            if SAVE_NEW_VALUE:
                                T.copy(v_chunk_ub_float[j, :, :],
                                       v_chunk_ub[pid, j, :, :])
                                T.set_flag("v", "mte3", 6)
                                T.wait_flag("v", "mte3", 6)
                                T.copy(
                                    v_chunk_ub[pid, j, :, :],
                                    v_new[i_h, g_start + BT // 2 * vid :
                                              g_start + BT // 2 * vid + BT // 2,
                                          j * V_half : j * V_half + V_half],
                                )

                            if USE_G:
                                # v_new *= exp(g_last - g)
                                T.tile.mul(v_chunk_ub_float[j, :, :],
                                           v_chunk_ub_float[j, :, :], g_exp_ub_broc)
                                # h_state *= exp(g_last)
                                T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])
                                T.tile.mul(h_state_ub_float[j, :, :],
                                           h_state_ub_float[j, :, :], g_last_scalar[0])
                            else:
                                T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])

                            # 将 v_new 搬运到 workspace，通知 Cube 核
                            T.set_flag("mte3", "v", 7)
                            T.wait_flag("mte3", "v", 7)
                            T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                            T.set_flag("v", "mte3", 8)
                            T.wait_flag("v", "mte3", 8)
                            T.copy(v_chunk_ub[pid, j, :, :],
                                   ws_vnew[i_n, i_h, j, BT // 2 * vid :
                                               BT // 2 * vid + BT // 2, :])
                            T.set_cross_flag("MTE3", SEM_VNEW_V2C + j)

                        # 等待 h_update 结果：h += k.T @ v_new
                        for j in T.serial(2):
                            T.wait_cross_flag(SEM_HUPD_C2V + j)
                            T.copy(ws_hupd[i_n, i_h, j, K // 2 * vid :
                                               K // 2 * vid + K // 2, :],
                                   hupd_ub_float[j, :, :])
                            T.set_flag("mte2", "v", 9)
                            T.wait_flag("mte2", "v", 9)
                            T.tile.add(h_state_ub_float[j, :, :],
                                       h_state_ub_float[j, :, :], hupd_ub_float[j, :, :])
                            T.copy(h_state_ub_float[j, :, :], h_state_ub[j, :, :])

                        # flush 同步
                        T.set_flag("v", "mte3", 10)
                        T.wait_flag("v", "mte3", 10)

                # Epilogue: 存储最终状态 ht
                if STORE_FINAL_STATE:
                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :],
                               ht[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2,
                                  j * V_half : (j + 1) * V_half])

    return main
```

### 6.2 调度选择

| 循环/操作 | 调度类型 | 理由 |
|----------|---------|------|
| chunk 循环 | `T.serial(NT_max)` | chunk 间有状态依赖，必须串行 |
| 动态长度掩码 | `if i < NT_i` | 支持变长序列，仅处理有效 chunk |
| V 分块 + K 分块 | `vid` 控制，`for j in T.serial(2)` | V 和 K 各分两半，双缓冲 [2,...] |
| element-wise | `T.tile.*` | Vector 核自动向量化 |
| GEMM | `T.gemm_v0` | Cube 核矩阵乘 |

---

## 7. 同步策略

### 7.1 手动同步方案

由于 `TL_ASCEND_AUTO_SYNC: False`，所有同步由代码显式控制：

| 同步类型 | API | 用途 |
|---------|-----|------|
| xmb 同步 | `T.set_flag` / `T.wait_flag("mte2", "m" \| "v", id)` | Cube域 - Vector域各自内部的数据搬运完成通知 |
| 标量同步 | `T.set_flag` / `T.wait_flag("m" \| "v", "fix" \| "mte3", id)` | GEMM 完成 → L0C 结果可用通知 |
| 跨核同步 | `T.set_cross_flag("FIX" \| "MTE3", sem)` / `T.wait_cross_flag(sem)` | Cube ↔ Vector 核间数据搬运完成握手 |

### 7.2 跨核数据流与握手协议

```
                                      Cube 域                      Vector 域
                                       │                              │
[Vector] ws_h 搬运完成 ──────────────────► wait_cross_flag(H_V2C)
                                       │  w @ h GEMM                   │
[  Cube] ws_wh 搬运完成 ──► set_cross_flag(WH_C2V) ◄── wait_cross_flag(WH_C2V)
                                       │                              │ v - w@h → v_new
[Vector] ws_vnew 搬运完成 ──► set_cross_flag(VNEW_V2C) ◄──            │
                                       │  wait_cross_flag(VNEW_V2C)    │
                                       │  k.T @ v_new GEMM             │
[  Cube] ws_hupd 搬运完成 ──► set_cross_flag(HUPD_C2V) ◄── wait_cross_flag(HUPD_C2V)
                                       │                              │ h += h_upd
                                       │                              │ → ws_h  (下一轮)
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

为避免频繁的 GM → L1 → L0C → GM 搬运，使用 workspace buffer（GM 驻留）：

| Workspace | Shape | Dtype | 用途 |
|----------|-------|-------|------|
| ws_wh | `[N, H, 2, BT, V_half]` | float32 | 存储 w @ h 结果，供 Vector 核读取 |
| ws_vnew | `[N, H, 2, BT, V_half]` | float16 | 存储 v_new，供 Cube 核读取 |
| ws_hupd | `[N, H, 2, K, V_half]` | float32 | 存储 k.T @ v_new 结果，供 Vector 核读取 |
| ws_h | `[N, H, 2, K, V_half]` | float16 | 存储 h_state，供 Cube 核读取 |

---

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现）

```python
def ref_chunk_gated_delta_rule(k, w, u, g=None, initial_state=None, 
                                output_final_state=False, chunk_size=64, 
                                cu_seqlens=None):
    BT = chunk_size

    k = k.float().squeeze(0)       # [T_total, Hg, K]
    w = w.float().squeeze(0)       # [T_total, H, K]
    u = u.float().squeeze(0)       # [T_total, H, V]
    g = g.float().squeeze(0) if g is not None else None  # [T_total, H]
    initial_state = initial_state.float().squeeze(0) if initial_state is not None else None  # [N, H, K, V]

    T_total, Hg, K = k.shape
    _, H, V = u.shape
    N = len(cu_seqlens) - 1

    NT_total = sum((int(cu_seqlens[i + 1]) - int(cu_seqlens[i]) + BT - 1) // BT for i in range(N))

    h = torch.zeros(NT_total, H, K, V, dtype=torch.float32)
    v_new = torch.zeros(T_total, H, V, dtype=torch.float32)
    final_state = torch.zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    chunk_offset = 0
    for i_n in range(N):
        bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
        T_len = eos - bos
        NT = (T_len + BT - 1) // BT

        for i_h in range(H):
            h_state = (initial_state[i_n, i_h].clone() if initial_state is not None
                       else torch.zeros(K, V, dtype=torch.float32))
            k_head = i_h // (H // Hg)

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T_len)

                h[chunk_offset + i_t, i_h] = h_state
                k_chunk = k[bos + t_start : bos + t_end, k_head, :]
                w_chunk = w[bos + t_start : bos + t_end, i_h, :]
                v_chunk = u[bos + t_start : bos + t_end, i_h, :]

                v_n = v_chunk - torch.matmul(w_chunk, h_state)
                v_new[bos + t_start : bos + t_end, i_h, :] = v_n

                if g is not None:
                    g_chunk = g[bos + t_start : bos + t_end, i_h]
                    g_last = g_chunk[-1].item()
                    v_n = v_n * torch.exp(g_last - g_chunk)[:, None]
                    h_state = h_state * torch.exp(torch.tensor(g_last))

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)

            if output_final_state:
                final_state[i_n, i_h] = h_state
        chunk_offset += NT

    return h.half().unsqueeze(0), v_new.half().unsqueeze(0), final_state.half() if final_state is not None else None
```

### 8.2 测试配置

| 模式 | 参数配置 | 测试组合 |
|------|---------|---------|
| **Varlen** | seqlens=[512,512,512,512] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[128,256,512,1024,128] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[2048] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[1024,1024] | use_g=True/False, use_initial_state=True/False |

### 8.3 命令行接口

```bash
python examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py \
  --use_g True --use_initial_state False \
  --seqlens 512,512,512,512 --H 8 --Hg 4 --K 128 --V 128
```

### 8.4 精度容忍度

实际测试中采用 `rtol=1e-5, atol=1e-5`（float16 级别），对所有 len、use_g、use_initial_state 配置均适用。

---

## 9. 注意事项

### 9.1 当前限制

| 限制 | 说明 |
|------|------|
| **K = 128 固定** | 当前典型配置，L1 容量可容纳 |
| **V = 128 固定** | V_half=64 分半处理，需要 V 为大于 0 的偶数 |
| **BT = 64 固定** | chunk size 固定，双缓冲 slot 数 = 2 |
| **Hg ≤ H** | GQA 比例必须满足 |


## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/chunk_gated_delta_rule/design.md` | 本文档 |
| 算子实现 | `examples/chunk_gated_delta_rule/expert_chunk_gated_delta_rule.py` | 已完成 |
