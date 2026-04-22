# chunk_gated_delta_rule 算子设计文档

## 1. 概述

### 1.1 算子名称

`chunk_gated_delta_rule` - 基于 chunk 的门控 Delta Rule 前向传播算子，用于线性注意力机制（Linear Attention）中的隐藏状态递推计算。支持定长序列和变长序列两种模式。

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
| **动态 shape** | B、T 为符号维度；支持变长序列（cu_seqlens） |
| **核间协作** | 需要 Cube 核做 GEMM，Vector 核做 element-wise 门控 |

### 1.4 典型配置示例

| 参数 | 值 | 说明 |
|------|-----|------|
| B | 1 | batch size |
| T | 2048 | sequence length（定长模式） |
| H | 8 | value head 数 |
| Hg | 4 | key head 数（GQA，H/Hg=2） |
| K | 128 | key dim |
| V | 128 | value dim |
| BT | 64 | chunk size（固定） |
| NT | 32 | chunk 数 = ceil(2048/64) |

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
- 使用 `T.alloc_L1/ub/L0A/L0B/L0C` 显式控制内存层级，避免编译器自动规划的不确定性
- 使用 `T.tile.*` 原语用于 element-wise 计算
- 显式控制数据搬运路径（GM → L1 → L0 → UB）

### 2.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动插入 barrier
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # 自动核间同步
}
```

---

## 3. API 映射设计

### 3.1 核心计算步骤 → TileLang API 映射

| 计算步骤 | PyTorch 参考 | TileLang API |
|---------|-------------|--------------|
| 零初始化状态 | `torch.zeros([K, V])` | `T.tile.fill(h_state_ub, 0)` |
| 加载初始状态 h0 | `h_state = h0.clone()` | `T.copy(h0[i_n, i_h, K//2*vid:K//2*vid+K//2, :], h_state_ub)` |
| 加载 w | `w_chunk = w[t_start:t_end]` | `T.copy(w[g_start:g_start+BT, i_h, :], w_chunk_l1)` |
| 加载 k | `k_chunk = k[t_start:t_end]` | `T.copy(k[g_start:g_start+BT, k_head, :], k_chunk_l1)` |
| 加载 v | `v_chunk = v[t_start:t_end]` | `T.copy(v[g_start+BT//2*vid:..., i_h, :], v_chunk_ub)` |
| 加载 g | `g_chunk = g[t_start:t_end]` | `T.copy(g[g_start:g_start+BT, i_h], g_chunk_ub_all)` |
| **GEMM: w @ h** | `torch.matmul(w, h)` | `T.gemm_v0(w_chunk_l1, h_state_l1, wh_frag, init=True)` |
| **v_new = v - wh** | `v - torch.matmul(w, h)` | `T.copy(v_chunk_ub, v_chunk_ub_float)` → `T.tile.sub(v_new_ub_float, v_chunk_ub_float, wh_ub_float)` |
| **exp(g_last - g)** | `torch.exp(g_last - g)` | `T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub)` → `T.tile.exp(g_exp_ub, g_exp_ub)` |
| **门控缩放** | `v_new * exp(g)` | `T.tile.mul(v_new_ub_float, v_new_ub_float, g_exp_ub_broc)` |
| **状态衰减** | `h * exp(g_last)` | `T.tile.mul(h_state_ub_float, h_state_ub_float, g_last_scalar[0])` |
| **GEMM: k @ v_new** | `torch.matmul(k.T, v_new)` | `T.gemm_v0(k_chunk_l1, v_new_l1, hupd_frag, transpose_A=True, init=True)` |
| **状态累积** | `h = h + k.T @ v_new` | `T.tile.add(h_state_ub_float, h_state_ub_float, hupd_ub_float)` |
| 存储 h[t] | `h[chunk_idx] = h_state` | `T.copy(h_state_ub, h[i_n, i, i_h, K//2*vid:K//2*vid+K//2, :])` |
| 存储 v_new | `v_new[t_start:t_end] = v_n` | `T.copy(v_new_ub, v_new[g_start+BT//2*vid:..., i_h, :])` |
| 存储 ht | `ht[bz, by] = h_state` | `T.copy(h_state_ub, ht[i_n, i_h, K//2*vid:K//2*vid+K//2, :])` |

### 3.2 关键 API 说明

- **`T.gemm_v0(A, B, C, transpose_A, transpose_B, init)`**：标准 GEMM，`init=True` 表示清零累加器
- **`T.tile.sub/exp/mul/add(dst, src1, src2)`**：Vector 核 element-wise 原语
- **`T.tile.fill(dst, value)`**：Vector 核填充原语
- **`T.tile.compare(dst, src, scalar, mode)`**：Vector 核比较原语
- **`T.tile.select(dst, mask, src1, src2, mode)`**：Vector 核选择原语
- **`T.tile.broadcast(dst, src)`**：Vector 核广播原语
- **`T.copy(src, dst)`**：自动推断层级间数据搬运

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 张量 | Shape (定长) | Shape (变长) | Dtype | 说明 |
|------|-------------|-------------|-------|------|
| k | `[B, T, Hg, K]` | `[1, T_total, Hg, K]` | float16 | Key 向量 |
| w | `[B, T, H, K]` | `[1, T_total, H, K]` | float16 | 门控权重 |
| u (v) | `[B, T, H, V]` | `[1, T_total, H, V]` | float16 | Value 向量 |
| g | `[B, T, H]` | `[1, T_total, H]` | float32 | 门控向量（可选） |
| h0 | `[B, H, K, V]` | `[1, N, H, K, V]` | float16 | 初始状态（可选） |
| cu_seqlens | 无 | `[N+1]` | int32 | 变长序列边界（变长模式） |

### 4.2 输出张量

| 张量 | Shape (定长) | Shape (变长) | Dtype | 说明 |
|------|-------------|-------------|-------|------|
| h | `[B, NT, H, K, V]` | `[1, NT_total, H, K, V]` | float16 | 每个 chunk 的隐藏状态 |
| v_new | `[B, T, H, V]` | `[1, T_total, H, V]` | float16 | 更新后的 value |
| ht | `[B, H, K, V]` | `[1, N, H, K, V]` | float16 | 最终隐藏状态（可选） |

### 4.3 内存层级规划（Expert 模式显式分配）

```
GM (全局内存)
  │
  ├─ T.copy → L1 (T.alloc_L1)
  │   ├─ k_chunk_l1:    [BT, K]       — chunk 内 k 数据（Cube 核输入）
  │   ├─ w_chunk_l1:    [BT, K]       — chunk 内 w 数据（Cube 核输入）
  │   ├─ h_state_l1:    [K, V]        — 累积状态（Cube 核输入）
  │   └─ v_new_l1:      [BT, V]       — v_new 数据（Cube 核输入）
  │
  ├─ T.copy → UB (T.alloc_ub)
  │   ├─ h_state_ub:    [K//2, V]     — 状态缓冲（Vector 核处理，V 分半）
  │   ├─ wh_ub_float:   [BT//2, V]    — w @ h 结果（float32，V 分半）
  │   ├─ v_chunk_ub:    [BT//2, V]    — chunk 内 v 数据（V 分半）
  │   ├─ v_chunk_ub_float: [BT//2, V] — v 数据 float32 缓冲
  │   ├─ v_new_ub_float: [BT//2, V]   — v_new float32 缓冲
  │   ├─ v_new_ub:      [BT//2, V]    — v_new float16 缓冲（输出）
  │   ├─ hupd_ub:       [K//2, V]     — h_update 缓冲（V 分半）
  │   ├─ hupd_ub_float: [K//2, V]     — h_update float32 缓冲
  │   ├─ h_state_ub_float: [K//2, V]  — 状态 float32 缓冲
  │   └─ (门控相关 buffer，USE_G=True 时)
  │       ├─ g_chunk_ub_all: [BT]          — chunk 内 g 数据
  │       ├─ g_chunk_ub:     [BT//2]       — g 数据（V 分半对应）
  │       ├─ g_last_scalar:  [1]           — chunk 最后有效 token 的 g
  │       ├─ g_exp_ub:       [BT//2]       — exp(g_last - g)
  │       ├─ g_exp_ub_dim:   [1, BT//2]    — 用于广播的 g_exp
  │       ├─ g_exp_ub_broc:  [BT//2, V]    — 广播后的 g_exp
  │       └─ g_mask_ub_pad:  [BT//8]       — mask for 数值稳定
  │
  ├─ T.copy → L0C (T.alloc_L0C)
  │   ├─ wh_frag:        [BT, V]       — GEMM 输出（w @ h）
  │   └─ hupd_frag:      [K, V]        — GEMM 输出（k.T @ v_new）
  │
  └─ Workspace (GM)
      ├─ ws_wh:          [N, H, BT, V]  — w @ h 中间结果（float32）
      ├─ ws_vnew:        [N, H, BT, V]  — v_new 中间结果
      ├─ ws_hupd:        [N, H, K, V]   — h_update 中间结果
      └─ ws_h:           [N, H, K, V]   — h_state 中间结果
```

### 4.4 数据搬运路径

```
k: GM → k_chunk_l1 (L1) → GEMM 输入
w: GM → w_chunk_l1 (L1) → GEMM 输入
v: GM → v_chunk_ub (UB) → v_chunk_ub_float (UB) → element-wise 输入
g: GM → g_chunk_ub_all (UB) → 门控计算
h0: GM → h_state_ub (UB) → 初始状态

w @ h: w_chunk_l1(L1) × h_state_l1(L1) → wh_frag(L0C) → ws_wh(GM) → wh_ub_float(UB)
v - wh: v_chunk_ub_float(UB) - wh_ub_float(UB) → v_new_ub_float(UB)
门控: v_new_ub_float(UB) × g_exp_ub_broc(UB) → v_new_ub_float(UB)
状态衰减: h_state_ub_float(UB) × g_last_scalar → h_state_ub_float(UB)
k @ v_new: k_chunk_l1(L1) × v_new_l1(L1) → hupd_frag(L0C) → ws_hupd(GM) → hupd_ub(UB) → hupd_ub_float(UB)
状态累积: h_state_ub_float(UB) + hupd_ub_float(UB) → h_state_ub(UB)

h_state: UB → GM (h 输出)
v_new: UB → GM (v_new 输出)
ht: UB → GM (ht 输出)
```

### 4.5 L1 容量估算（典型配置: K=128, V=128）

```
k_chunk_l1:    64×128×2B = 16KB
w_chunk_l1:    64×128×2B = 16KB
h_state_l1:    128×128×2B = 32KB
v_new_l1:      64×128×2B = 16KB
总计 ≈ 80KB < 524KB（安全范围内）
```

### 4.6 UB 容量估算（典型配置: K=128, V=128）

```
h_state_ub:    64×128×2B = 16KB (×2 分块 = 32KB)
wh_ub_float:   32×128×4B = 16KB (×2 分块 = 32KB)
v_chunk_ub:    32×128×2B = 8KB (×2 分块 = 16KB)
v_chunk_ub_float: 32×128×4B = 16KB (×2 分块 = 32KB)
v_new_ub_float: 32×128×4B = 16KB (×2 分块 = 32KB)
v_new_ub:      32×128×2B = 8KB (×2 分块 = 16KB)
hupd_ub:       64×128×2B = 16KB (×2 分块 = 32KB)
hupd_ub_float: 64×128×4B = 32KB (×2 分块 = 64KB)
h_state_ub_float: 64×128×4B = 32KB (×2 分块 = 64KB)
g 相关 buffer: ~2KB
总计 ≈ 196KB（接近 UB 容量上限）
```

---

## 5. Tiling 策略

### 5.1 Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `T.Kernel(N * H, is_npu=True)` | 每个 (batch, head) 组合一个 block；定长 N=B，变长 N=序列数 |
| **Chunk** | `BT = 64` | 固定 chunk size |
| **V 分块** | `V // 2 = 64` | V=128 分为两半，由 `vid` (Vector sub-block) 控制 |
| **K 分块** | `K // 2 = 64` | K=128 分为两半，由 `vid` 控制（仅 h_state 相关） |

### 5.2 Tile Shape 设计（典型配置: K=128, V=128）

| Buffer | Shape | 说明 |
|--------|-------|------|
| k_chunk_l1 | `[64, 128]` | BT=64 × K=128 |
| w_chunk_l1 | `[64, 128]` | BT=64 × K=128 |
| h_state_l1 | `[128, 128]` | K=128 × V=128 |
| wh_frag | `[64, 128]` | w[64,128] @ h[128,128] → [64,128] |
| v_new_l1 | `[64, 128]` | BT=64 × V=128 |
| hupd_frag | `[128, 128]` | k.T[128,64] @ v_new[64,128] → [128,128] |
| h_state_ub | `[64, 128]` | K//2 × V，V 分半处理 |
| wh_ub_float | `[32, 128]` | BT//2 × V，float32 |
| g_chunk_ub_all | `[64]` | BT |

---

## 6. 循环与调度结构

### 6.1 Kernel 结构设计

```python
@tilelang.jit(workspace_idx=[9, 10, 11, 12], pass_configs=pass_configs)
def chunk_gated_delta_rule_fwd_kernel_unified(
    N, H, T_total_pad, Hg, K, V, NT_max,
    BT=64, USE_G=True, USE_INITIAL_STATE=True,
    STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True,
    dtype="float16", accum_dtype="float32",
):
    @T.prim_func
    def main(h, k, v, w, g, v_new, h0, ht, cu_seqlens, ws_wh, ws_vnew, ws_hupd, ws_h):
        with T.Kernel(N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H
            hg_ratio = H // Hg
            k_head = i_h // hg_ratio
            
            # 变长序列支持：动态计算每个序列的长度
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            T_len = eos - bos
            NT_i = T.ceildiv(T_len, BT)
            
            # Buffer 分配（显式指定层级）
            h_state_ub = T.alloc_ub([K // 2, V], dtype)
            h_state_ub_float = T.alloc_ub([K // 2, V], accum_dtype)
            ...
            
            # 初始状态加载
            if USE_INITIAL_STATE:
                T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, :], h_state_ub)
            else:
                T.tile.fill(h_state_ub, 0)
            
            # 主循环：遍历所有 chunk
            for i in T.serial(NT_max):
                if i < NT_i:  # 动态长度掩码
                    g_start = bos + i * BT
                    
                    # 1. w @ h (GEMM on Cube)
                    T.copy(h_state_ub, ws_h[i_n, i_h, K // 2 * vid, :])
                    T.copy(ws_h[i_n, i_h, :, :], h_state_l1)
                    T.copy(w[g_start : g_start + BT, i_h, :], w_chunk_l1)
                    T.gemm_v0(w_chunk_l1, h_state_l1, wh_frag, init=True)
                    T.copy(wh_frag, ws_wh[i_n, i_h, :, :])
                    
                    # 2. v_new = v - w @ h (Vector, float32 precision)
                    T.copy(ws_wh[i_n, i_h, BT // 2 * vid : ...], wh_ub_float)
                    T.copy(v[g_start + BT // 2 * vid : ...], v_chunk_ub)
                    T.copy(v_chunk_ub, v_chunk_ub_float)
                    T.tile.sub(v_new_ub_float, v_chunk_ub_float, wh_ub_float)
                    
                    # 3. 门控计算（可选）
                    if USE_G:
                        T.copy(g[g_start : g_start + BT, i_h], g_chunk_ub_all)
                        # 提取最后一个有效 token 的 g
                        if i * BT + BT <= T_len:
                            g_last_scalar[0] = g_chunk_ub_all[BT - 1]
                        else:
                            g_last_scalar[0] = g_chunk_ub_all[T_len - i * BT - 1]
                        
                        T.tile.fill(g_exp_ub, g_last_scalar[0])
                        T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub)
                        T.tile.compare(g_mask_ub_pad, g_exp_ub_pad, T.float32(0), "LE")
                        T.tile.select(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad, -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.exp(g_exp_ub, g_exp_ub)
                        T.tile.broadcast(g_exp_ub_broc, g_exp_ub_dim)
                        T.tile.mul(v_new_ub_float, v_new_ub_float, g_exp_ub_broc)
                        T.tile.exp(g_last_scalar, g_last_scalar)
                        T.copy(h_state_ub, h_state_ub_float)
                        T.tile.mul(h_state_ub_float, h_state_ub_float, g_last_scalar[0])
                    
                    # 4. k @ v_new (GEMM on Cube)
                    T.copy(v_new_ub_float, v_new_ub)
                    T.copy(v_new_ub, ws_vnew[i_n, i_h, BT // 2 * vid, :])
                    T.copy(ws_vnew[i_n, i_h, :, :], v_new_l1)
                    T.copy(k[g_start : g_start + BT, k_head, :], k_chunk_l1)
                    T.gemm_v0(k_chunk_l1, v_new_l1, hupd_frag, transpose_A=True, init=True)
                    T.copy(hupd_frag, ws_hupd[i_n, i_h, :, :])
                    
                    # 5. 状态累积 (Vector)
                    T.copy(ws_hupd[i_n, i_h, K // 2 * vid : ...], hupd_ub)
                    T.copy(hupd_ub, hupd_ub_float)
                    if not USE_G:
                        T.copy(h_state_ub, h_state_ub_float)
                    T.tile.add(h_state_ub_float, h_state_ub_float, hupd_ub_float)
                    T.copy(h_state_ub_float, h_state_ub)
                    
                    # 6. 存储输出
                    T.copy(h_state_ub, h[i_n, i, i_h, K // 2 * vid : ...])
            
            # Epilogue: 存储 ht
            if STORE_FINAL_STATE:
                T.copy(h_state_ub, ht[i_n, i_h, K // 2 * vid : ...])
    
    return main
```

### 6.2 调度选择

| 循环/操作 | 调度类型 | 理由 |
|----------|---------|------|
| chunk 循环 | `T.serial(NT_max)` | chunk 间有状态依赖，必须串行 |
| 动态长度掩码 | `if i < NT_i` | 支持变长序列，仅处理有效 chunk |
| V 分块 | 编译器自动（vid） | 2 个 Vector sub-block 并行处理 V//2 |
| element-wise | `T.tile.*` | Vector 核自动向量化 |
| GEMM | `T.gemm_v0` | Cube 核矩阵乘 |

---

## 7. 同步策略

### 7.1 自动同步（pass_configs）

由于使用 `TL_ASCEND_AUTO_SYNC: True`，编译器自动在以下位置插入同步：

| 同步点 | 位置 |
|--------|------|
| 数据加载后 | `T.copy` → GEMM 之间 |
| GEMM 后 | GEMM → `T.copy` 之间 |
| element-wise 后 | `T.tile.*` → GEMM 之间 |

### 7.2 核间同步（CV 分离）

由于使用 `TL_ASCEND_AUTO_CV_SYNC: True`，编译器自动处理：
- Cube 核完成 GEMM 后 → `set_cross_flag`
- Vector 核等待 → `wait_cross_flag`
- Vector 核完成 element-wise → `set_cross_flag`
- Cube 核等待 → `wait_cross_flag`

### 7.3 Workspace 设计

为避免频繁的 GM → L1 → L0C → GM 搬运，使用 workspace buffer：

| Workspace | Shape | Dtype | 用途 |
|----------|-------|-------|------|
| ws_wh | `[N, H, BT, V]` | float32 | 存储 w @ h 结果，供 Vector 核读取 |
| ws_vnew | `[N, H, BT, V]` | float16 | 存储 v_new，供 Cube 核读取 |
| ws_hupd | `[N, H, K, V]` | float16 | 存储 k.T @ v_new 结果 |
| ws_h | `[N, H, K, V]` | float16 | 存储 h_state，供 Cube 核读取 |

---

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现）

```python
def ref_chunk_gated_delta_rule(k, w, u, g=None, initial_state=None, 
                               output_final_state=False, chunk_size=64, 
                               cu_seqlens=None):
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    
    k = k.float()
    w = w.float()
    u = u.float()
    g = g.float() if g is not None else None
    initial_state = initial_state.float() if initial_state is not None else None
    
    if not is_varlen:
        # 定长模式
        B, T_len, Hg, K = k.shape
        _, _, H, V = u.shape
        NT = (T_len + BT - 1) // BT
        
        h = torch.zeros(B, NT, H, K, V, dtype=torch.float32)
        v_new = torch.zeros(B, T_len, H, V, dtype=torch.float32)
        
        for bz in range(B):
            for by in range(H):
                h_state = initial_state[bz, by].clone() if initial_state else torch.zeros(K, V)
                k_head = by // (H // Hg)
                
                for i in range(NT):
                    t_start = i * BT
                    t_end = min((i + 1) * BT, T_len)
                    
                    h[bz, i, by] = h_state
                    k_chunk, w_chunk, v_chunk = ...
                    
                    v_n = v_chunk - torch.matmul(w_chunk, h_state)
                    v_new[bz, t_start:t_end, by, :] = v_n
                    
                    if g is not None:
                        g_chunk = g[bz, t_start:t_end, by]
                        g_last = g_chunk[-1].item()
                        v_n = v_n * torch.exp(g_last - g_chunk)[:, None]
                        h_state = h_state * torch.exp(torch.tensor(g_last))
                    
                    h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)
    else:
        # 变长模式
        ...
    
    return h.half(), v_new.half(), final_state.half() if final_state else None
```

### 8.2 测试配置

| 模式 | 参数配置 | 测试组合 |
|------|---------|---------|
| **Fixed-length** | B=1, T=2048, H=8, Hg=4, K=128, V=128 | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[512,512,512,512] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[128,256,512,1024,128] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[2048] | use_g=True/False, use_initial_state=True/False |
| **Varlen** | seqlens=[1024,1024] | use_g=True/False, use_initial_state=True/False |

### 8.3 命令行接口

```bash
# Fixed-length 模式
python examples/chunk_gated_delta_rule/chunk_gated_delta_rule_varlen.py \
  --use_g True --use_initial_state False --varlen False \
  --B 1 --T 2048 --H 8 --Hg 4 --K 128 --V 128

# Varlen 模式
python examples/chunk_gated_delta_rule/chunk_gated_delta_rule_varlen.py \
  --use_g True --use_initial_state False --varlen True \
  --seqlens 512,512,512,512 --H 8 --Hg 4 --K 128 --V 128
```

### 8.4 精度容忍度

| 配置 | rtol | atol |
|------|------|------|
| T ≤ 128 | 1e-2 | 1e-3 |
| 128 < T ≤ 512 | 1e-2 | 5e-3 |
| 512 < T ≤ 2048 | 5e-2 | 5e-2 |

---

## 9. 风险点与注意事项

### 9.1 已解决风险

| 风险 | 原状态 | 解决方案 |
|------|-------|---------|
| **无 g 精度问题** | float16 精度累积误差 | v_new 计算改用 float32 精度，ws_wh 改为 float32 |
| **变长序列 g_last** | 取 padding 的零值 | 使用条件判断提取最后一个有效 token 的 g |

### 9.2 当前限制

| 限制 | 说明 |
|------|------|
| **K = 128 固定** | 当前典型配置，L1 容量可容纳 |
| **V = 128 固定** | 分为两半各 64，需要 V 为偶数 |
| **BT = 64 固定** | chunk size 固定 |
| **Hg ≤ H** | GQA 比例必须满足 |


---

## 10. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/chunk_gated_delta_rule/design.md` | 本文档 |
| 算子实现 | `examples/chunk_gated_delta_rule/chunk_gated_delta_rule_varlen.py` | 已完成 |
| 性能脚本 | `examples/chunk_gated_delta_rule/profile_all.sh` | 待创建 |

---

## 附录 A：精度修复说明

### A.1 问题背景

当 `USE_G=False` 时，`v_new = v - w @ h` 的计算出现精度问题：
- `wh_frag` (L0C) 为 float32
- 拷贝到 `ws_wh` 时被截断为 float16（原设计）
- `T.tile.sub` 在 float16 上计算，导致精度损失

### A.2 修复方案

1. **ws_wh 类型改为 float32**：`ws_wh: T.Tensor([N, H, BT, V], accum_dtype)`
2. **减法计算改用 float32**：
   ```python
   T.copy(v_chunk_ub, v_chunk_ub_float)  # float16 → float32
   T.tile.sub(v_new_ub_float, v_chunk_ub_float, wh_ub_float)  # float32 计算
   T.copy(v_new_ub_float, v_new_ub)  # float32 → float16（输出）
   ```
3. **添加中间 buffer**：`v_chunk_ub_float` 用于类型转换

---

## 附录 B：变长序列 g_last 提取

### B.1 问题背景

变长序列的 chunk 可能未完全填充，原实现 `g_last = g_chunk_ub_all[BT - 1]` 取的是 padding 的零值。

### B.2 修复方案

使用条件判断提取最后一个有效 token 的 g：

```python
if i * BT + BT <= T_len:
    # chunk 完全填充
    g_last_scalar[0] = g_chunk_ub_all[BT - 1]
else:
    # chunk 未完全填充
    g_last_scalar[0] = g_chunk_ub_all[T_len - i * BT - 1]
```

---

## 附录 C：测试用例总览

| 模式 | use_g | use_initial_state | seqlens | 组合数 |
|------|-------|-------------------|---------|--------|
| Fixed | True/False | True/False | 2048 | 4 |
| Varlen | True/False | True/False | 512,512,512,512 | 4 |
| Varlen | True/False | True/False | 128,256,512,1024,128 | 4 |
| Varlen | True/False | True/False | 2048 | 4 |
| Varlen | True/False | True/False | 1024,1024 | 4 |
| **总计** | | | | **20 组** |