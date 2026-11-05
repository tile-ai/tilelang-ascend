# Fused Sigmoid Gating Delta Rule SSM Scan — 设计文档

## 功能概述

本算子实现 Mamba-style 线性注意力中的融合 sigmoid 门控 + delta rule 状态更新的多序列扫描计算（SSM Scan）。支持变长序列、多核并行、ping-pong 数据预取，将门控参数（softplus + sigmoid）与 delta rule 状态更新融合在单个 kernel 中执行。

**编程模式**: Expert（显式内存分配 + 手动同步 + `T.tile.xxx` 原语）  
**目标硬件**: 昇腾 NPU（24 核 Vector）

---

## 数学公式

### 记号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $T$ | 序列总 token 数 | — |
| $H_k$ | Query/Key head 数 (`nk`) | — |
| $H_v$ | Value head 数 (`nv`) | — |
| $d_k$ | Q/K 维度 (`dk`) | — |
| $d_v$ | V 维度 (`dv`) | — |
| $S$ | 序列数 (`num_seqs`) | — |
| $B_v$ | V 维度分块大小 (`block_v`) | — |
| $\beta_s$ | softplus 参数 (`softplus_beta`) | — |
| $\epsilon$ | L2 归一化 $\varepsilon$ | — |
| $\text{scale}$ | Q 缩放因子 $\;= 1/\sqrt{d_k}$ | — |

$H_v$ 必须是 $H_k$ 的整数倍，`v_per_k` $= H_v / H_k$。

---

### 门控参数计算

对每个 v_head $h$、每个 token $t$：

$$
\begin{aligned}
\text{softplus}(x;\; \beta_s) &=
\begin{cases}
x & \text{if } \beta_s \cdot x > 20 \\
\displaystyle \frac{\log(1 + e^{\beta_s x})}{\beta_s} & \text{otherwise}
\end{cases} \\[8pt]
x_t &= a_{t,h} + \text{dt\_bias}_{h} \\[4pt]
\alpha_{t,h} &= \exp\!\big(-e^{A_{\text{log},h}} \cdot \text{softplus}(x_t; \beta_s)\big) \quad\in (0,1) \\[4pt]
g_{t,h} &= \sigma(b_{t,h}) = \frac{1}{1 + e^{-b_{t,h}}}
\end{aligned}
$$

---

### 每时间步状态更新（SSM Scan）

状态矩阵 $H \in \mathbb{R}^{B_v \times d_k}$ 按行存储（每行是 V 维度的一段）。对每条序列逐 token 执行：

$$
\begin{aligned}
\text{(可选 L2 Norm)} \quad &\hat{q}_t = \frac{q_t}{\sqrt{\|q_t\|^2 + \epsilon}}, \quad
\hat{k}_t = \frac{k_t}{\sqrt{\|k_t\|^2 + \epsilon}} \\[4pt]
\text{(Scale)} \quad &\hat{q}_t \leftarrow \hat{q}_t \cdot \text{scale} \\[4pt]
\text{(State Decay)} \quad &H \leftarrow H \cdot \alpha_{t,h} \\[4pt]
\text{(Prediction)} \quad &\text{pred} = H \cdot \hat{k}_t \quad\in \mathbb{R}^{B_v} \\[4pt]
\text{(Delta Rule)} \quad &\delta = (v_t - \text{pred}) \odot g_{t,h} \quad\in \mathbb{R}^{B_v} \\[4pt]
\text{(State Update)} \quad &H \leftarrow H + \hat{k}_t \otimes \delta \qquad\big(H[i,j] \mathrel{+}= \hat{k}_t[j] \cdot \delta[i]\big) \\[4pt]
\text{(Output)} \quad &o_t = H \cdot \hat{q}_t \quad\in \mathbb{R}^{B_v}
\end{aligned}
$$

---

## 输入 / 输出

### 输入张量

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `A_log` | $(H_v,)$ | fp16 | 每个 v_head 的 $\log(A)$ 参数 |
| `a` | $(T_{\text{pad}}, H_v)$ | fp16 | 输入门控参数 $a$ |
| `dt_bias` | $(H_v,)$ | fp16 | 每个 head 的 $\Delta t$ bias |
| `query` | $(T_{\text{pad}}, H_k, d_k)$ | fp16 | Query 张量 |
| `key` | $(T_{\text{pad}}, H_k, d_k)$ | fp16 | Key 张量 |
| `value` | $(T_{\text{pad}}, H_v, d_v)$ | fp16 | Value 张量 |
| `beta` | $(T_{\text{pad}}, H_v)$ | fp16 | Sigmoid 输入 $b$ |
| `init_state` | $(C, H_v, d_v, d_k)$ | fp16 | 缓存的状态池，内部运算转置为 $(C,H_v,d_k,d_v)$ |
| `ssm_state_indices` | $(S,)$ | int32 | 每条序列在状态缓存中的索引，$-1$ 表示空 |
| `cu_seqlens` | $(S+1,)$ | int32 | 累积序列长度，`cu_seqlens[i+1] - cu_seqlens[i]` 为序列长度 |

### 输出张量

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `out` | $(T_{\text{pad}}, H_v, d_v)$ | fp16 | 每 token 输出（含 padding） |
| `final_state` | $(S, H_v, d_v, d_k)$ | fp16 | 每条序列最终状态（内部转置为 $d_k \times d_v$） |

---

## Tiling 策略

$$
\text{num\_v\_tiles} = \big\lceil d_v / B_v \big\rceil, \quad
\text{vec\_block\_v} = B_v / 2
$$

- **V 维度分块**: `block_v` 沿 $d_v$ 分块，`block_v` 为 32 的整数倍且整除 `VEC_NUM=2`
- **Vector 子分块**: 每个 vector unit 处理 `vec_block_v = block_v / 2` 行
- **K 维度**: 不做分块，整 $d_k$ 维放入 UB（$d_k \leq 128$）
- **任务粒度**: 每个任务为 `(seq_idx, v_head_idx, v_tile_idx)` 三元组

---

## 工作分配（24 核并行）

设总任务数 `block_num = num_seqs * H_v * num_v_tiles`，均匀分配给 24 个核：

$$
\text{max\_work\_per\_block} = \begin{cases}
\lfloor \text{block\_num} / 24 \rfloor + 1 & \text{前 } (\text{block\_num} \bmod 24) \text{ 个核} \\
\lfloor \text{block\_num} / 24 \rfloor & \text{剩余核}
\end{cases}
$$

每个核按 `flat_idx → (seq, v_head, v_tile)` 映射执行子任务。

---

## 内存层级与 UB 布局

### 数据流

```
GM (global memory)
  ↓ T.copy (异步)
UB (unified buffer)  — ping-pong: Q [2, dk], K [2, dk], V [2, vec_block_v]
  ↓ T.copy
UB (fp32 workspace)  — q_f, k_f, v_f; H [vec_block_v, dk]
  ↓ tile compute
UB (fp32 result)     — out ping-pong [2, vec_block_v]
  ↓ T.copy
GM (out)
```

### UB 分配明细

| Buffer | Shape | dtype | 用途 |
|--------|-------|-------|------|
| `q_buf`, `k_buf` | $(2, d_k)$ | fp16 | Q/K ping-pong 预取 |
| `v_buf` | $(2, B_v/2)$ | fp16 | V ping-pong 预取 |
| `q_f`, `k_f` | $(d_k,)$ | fp32 | 当前 token Q/K |
| `v_f` | $(B_v/2,)$ | fp32 | 当前 token V |
| `h_vec` | $(B_v/2, d_k)$ | fp32 | 状态矩阵 $H$ |
| `h_load_vec` | $(B_v/2, d_k)$ | fp16 | 从 GM 加载 $H_0$ |
| `h_store_vec` | $(B_v/2, d_k)$ | fp16 | 回存 $H_T$ 到 GM |
| `pred_vec` | $(B_v/2,)$ | fp32 | $\text{pred} = H \hat{k}$ |
| `delta_vec` | $(B_v/2,)$ | fp32 | $\delta$ |
| `o_half_buf` | $(2, B_v/2)$ | fp16 | 输出 ping-pong |
| `k_broadcasted` | $(B_v/2, d_k)$ | fp32 | $\hat{k}$ 广播 |
| `compute_buffer` | $(B_v/2, d_k)$ | fp32 | 中间 workspace |

---

## 同步策略

| 同步点 | Flag 类型 | 方向 | 用途 |
|--------|-----------|------|------|
| 加载 init_state | `mte2→v` | MTE2→Vector | 等待 $H_0$ 到达 UB |
| 加载 A_log | `mte2→v` | MTE2→Vector | 等待 $A_{\log}$ 到达 |
| 写回 out 后 | `v→mte3` | Vector→MTE3 | out 写回完成确认 |
| QKV 预取 | `mte2→v` | MTE2→Vector | 等待下一 token Q/K/V 到达 |

---

## 约束与限制

1. $H_v$ 必须是 $H_k$ 的整数倍（支持 Grouped Query）
2. `block_v` 必须是 `VEC_NUM=2` 的整数倍且整除 $d_v$
3. 仅支持 `float16` 类型（Ascend C 编译器限制）
4. 时间步串行（状态更新存在数据依赖）
5. Token 总数必须 padding 至 64 的整数倍

---

## 验证标准

与 PyTorch 参考实现对比，精度标准: `rtol=2e-2, atol=2e-2`。
