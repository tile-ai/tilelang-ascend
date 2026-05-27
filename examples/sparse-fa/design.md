# MTGR Ragged Segment Attention 算子设计文档

## 1. 概述

### 1.1 算子名称

`mtgr_ragged_segment_attention`（MTGR 多段式稀疏注意力，ragged batch）

### 1.2 功能描述

在 LLM 推理 Decode / Prefill 阶段实现多段掩码（Segment Mask）的 Flash Attention。
将逻辑序列划分为 4 段（或 3 段），每段使用不同的掩码规则（Causal / Full / Diagonal），
支持 ragged batch（每个 request 长度不同）及 Paged KV Cache 前缀匹配。

### 1.3 数学公式

$$
\text{Output}[q, h, d] = \sum_k \text{Softmax}\left(\text{Mask}\left(
\frac{Q[q,h,:] \cdot K[k,h,:]^T}{\sqrt{d_{\text{head}}}}, \; \text{seg\_rules}, \text{seg\_offsets}
\right)\right) \cdot V[k,h,d]
$$

其中 Mask 运算根据查询行所在段的规则（Rule）决定可见范围：

| 规则 | 含义 | visible_end[q] | 对角线可见 |
|------|------|----------------|-----------|
| 0 (Causal) | 下三角 | q_pos + 1 | 无 |
| 1 (Full) | 全算 | seg_end | 无 |
| 2 (Diagonal) | 对角线 | seg_start | q_pos（自身可见） |

- Score[k] 保留当且仅当：`k < visible_end[q]` **或** `k == diag_col[q]`
- 其余位置置为 `-inf`，Softmax 后自然为 0

### 1.4 算法描述

采用 **Online Safe Softmax**（在线迭代 softmax）：

```
初始化: O = 0, l = 0, m = -inf

对每个 KV tile (kv_start..kv_start+Bn):
  S = Q × K^T × sm_scale                    // GEMM QK
  对 S 施加 segment mask（置 -inf）          // Mask
  m_new = row_max(S)                         // 新 max
  m' = max(m, m_new)
  P = exp(S - m')                             // 安全 exp
  l_new = sum(P)                              // 新 sumexp
  l' = l * exp(m - m') + l_new               // 在线更新 ∑exp
  O = O * exp(m - m') + P × V               // 在线更新 O
  m = m'
  l = l'

完成后: O = O / l                             // 归一化
```

### 1.5 段结构示例（四段式）

```
├── history (causal)  ~2000 tokens
├── context (full)       8 tokens
├── realtime (causal) ~200 tokens
└── target  (diagonal) ~1000 tokens
```

### 1.6 数据流图

```
segment_offsets ──┬──(Host预处理)──> visible_end, diag_col (per-request 局部坐标)
segment_rules  ───┘
                          │
Q[TND packed]───────┐
q_seq_starts[. ]────┤
                    ├──> packed offset 计算 ──> [Cube] Q×K^T GEMM ──> scores S
K[TND packed]───────┤                              │
kv_seq_starts[. ]───┤                     [Vector] segment mask (局部坐标比较)
                    │                              │
                    │                     [Vector] online softmax
                    │                              │
                    │                     [Cube]   P×V GEMM ──> O_partial
V[TND packed]───────┤                              │
                    │                     [Vector] O accum + update
                    │                              │
                    └──────> Output[TND packed]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（自动化模式）

### 2.2 选型理由

1. 算子含 GEMM + Element-wise 后处理，属 CV 融合算子，AUTO_CV_COMBINE 可自动分离 Cube/Vector 操作
2. 参考实现 `examples/sparse_flash_attention/example_sparse_flash_attn.py` 使用 Developer 模式 + AUTO passes 成功
3. Developer 模式降低开发复杂度：无需手动管理 T.Scope、T.set_cross_flag/T.wait_cross_flag
4. 使用 `MEMORY_PLANNING` 自动管理地址分配，避免手动 T.annotate_address

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_L1 / T.alloc_ub / T.alloc_L0C（显式声明，MEMORY_PLANNING 分配地址） |
| 计算方式 | T.gemm_v0（Cube） + T.tile.* / T.reduce_*（Vector） |
| 作用域 | AUTO_CV_COMBINE 自动分离 Cube / Vector |
| 同步方式 | AUTO_SYNC + AUTO_CV_SYNC 自动管理 |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | Q × K^T × sm_scale | GEMM 计算注意力分数，分第一个 GEMM 和尾部 GEMM |
| 2 | Mask(S, visible_end, diag_col) | 按段规则遮蔽不可见位置 |
| 3 | m = max(S, dim=-1) | 在线 Safe Softmax：行内 max |
| 4 | m' = max(m, m_prev) | 与历史 m 取 max |
| 5 | P = exp(S - m') | 安全指数 |
| 6 | l = sum(P, dim=-1) | 行内 sum |
| 7 | l_prev = l_prev * exp(m_prev - m') + l | 在线更新 ∑exp |
| 8 | O = O * exp(m_prev - m') + P × V | 在线更新输出 + GEMM PV |
| 9 | O = O / l | 归一化 |

### 3.2 TileLang API 映射

| 步骤 | TileLang API | 参数 | 备注 |
|------|-------------|------|------|
| Q 加载 | `T.copy(Q[q_packed_start:q_packed_start+block_M, h, :], q_l1)` | TND packed 偏移 | |
| K 加载 | `T.copy(K[kv_packed_start:kv_packed_start+block_N, h_kv, :], k_l1)` | TND packed 偏移 | |
| GEMM QK | `T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)` | | L0C 累加器 |
| S→workspace | `T.copy(acc_s_l0c, workspace_s[cid, :, :])` | | C→V 传输 |
| P←workspace | `T.copy(workspace_s[cid, vid*v_block:..., :], acc_s_ub_)` | | V 侧读取 |
| 乘 sm_scale | `T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)` | | |
| Mask 比较 | `T.tile.compare(mask_ub, kv_col_float, visible_end_float[i], "LT")` | mode="LT", 局部坐标比较 | 生成可见掩码 |
| Mask 对角线 | `T.tile.compare(mask_diag_ub, kv_col_float, diag_col_float[i], "EQ")` | mode="EQ" | 对角掩码 |
| Mask 合并 | `T.tile.bitwise_or(mask_ub, mask_ub, mask_diag_ub)` | | |
| Mask 应用 | `T.tile.select(acc_s_ub[i,:], mask_ub, acc_s_ub[i,:], -T.infinity(...), "VSEL_TENSOR_SCALAR_MODE")` | | |
| 行 max | `T.reduce_max(acc_s_ub, m_i, dim=-1)` | | |
| m' = max(m, m_prev) | `T.tile.max(m_i, m_i, m_i_prev)` | | |
| α = exp(m_prev - m') | `T.tile.sub(m_i_prev, m_i_prev, m_i)` → `T.tile.exp(m_i_prev, m_i_prev)` | | 修正因子 α |
| S - m' | `T.tile.sub(acc_s_ub[h_i,:], acc_s_ub[h_i,:], m_i[h_i])` | per-row | |
| exp(S - m') | `T.tile.exp(acc_s_ub, acc_s_ub)` | | P |
| sum(P) | `T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)` | | l_new |
| l = l * α + l_new | `T.tile.mul(sumexp, sumexp, m_i_prev)` → `T.tile.add(sumexp, sumexp, sumexp_i_ub)` | | |
| O = O * α | `T.tile.mul(acc_o[h_i,:], acc_o[h_i,:], m_i_prev[h_i])` | per-row | |
| P→workspace | `T.copy(acc_s_ub, acc_s_half)` → `T.copy(acc_s_half, workspace_p[cid, vid*v_block:..., :])` | dtype cast | V→C 传输 |
| P←L1 | `T.copy(workspace_p[cid, :, :], acc_s_l1)` | | C 侧 |
| V 加载 | `T.copy(V[kv_packed_start:kv_packed_start+block_N, h_kv, :], v_l1)` | TND packed 偏移 | |
| GEMM PV | `T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)` | | L0C |
| O_partial→ws | `T.copy(acc_o_l0c, workspace_o[cid, :, :])` | | C→V 传输 |
| O += O_partial | `T.copy(workspace_o[cid, vid*v_block:..., :], acc_o_ub)` → `T.tile.add(acc_o, acc_o, acc_o_ub)` | | V 侧 |
| O = O / l | `T.tile.div(acc_o[h_i,:], acc_o[h_i,:], sumexp[h_i])` | per-row | |
| 输出写入 | `T.copy(acc_o, acc_o_half)` → `T.copy(acc_o_half, Output[output_packed_start:..., h, :])` | TND packed 偏移 | |

### 3.3 计算伪代码（Developer 模式，TND Packed Layout）

```python
# ==== Host 预计算 ====
num_seq_tiles_per_request = [ceildiv(actual_q_len[b], block_M) for b in range(batch)]
cum_seq_tiles_per_request = prefix_sum(num_seq_tiles_per_request)
total_tasks = cum_seq_tiles_per_request[-1] * num_heads

# ==== Kernel ====
with T.Kernel(core_num, is_npu=True) as (cid, vid):
    # ==== 内存分配 ====
    q_l1 = T.alloc_L1([block_M, head_dim], dtype)
    k_l1 = T.alloc_L1([block_N, head_dim], dtype)
    v_l1 = T.alloc_L1([block_N, head_dim], dtype)
    acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
    acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
    acc_o_l0c = T.alloc_L0C([block_M, head_dim], accum_dtype)
    # UB (Vector)
    acc_o = T.alloc_ub([v_block, head_dim], accum_dtype)
    sumexp = T.alloc_ub([v_block], accum_dtype)
    m_i = T.alloc_ub([v_block], accum_dtype)
    acc_s_ub = T.alloc_ub([v_block, block_N], accum_dtype)
    m_i_prev = T.alloc_ub([v_block], accum_dtype)
    acc_s_ub_ = T.alloc_ub([v_block, block_N], accum_dtype)
    sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
    acc_s_half = T.alloc_ub([v_block, block_N], dtype)
    acc_o_ub = T.alloc_ub([v_block, head_dim], accum_dtype)
    acc_o_half = T.alloc_ub([v_block, head_dim], dtype)
    # Mask buffers
    visible_end_ub = T.alloc_ub([block_M], "int32")
    diag_col_ub = T.alloc_ub([block_M], "int32")
    kv_col_ub = T.alloc_ub([block_N], "int32")
    mask_ub = T.alloc_ub([block_N // 8], "uint8")
    mask_diag_ub = T.alloc_ub([block_N // 8], "uint8")

    # ==== 固定核任务循环 ====
    for core_index in T.serial(T.ceildiv(total_tasks, core_num)):
        pid = core_index * core_num + cid
        if pid < total_tasks:
            # pid → (b_i, s_local, h_i) 解析
            # flat_seq_tile = 跨所有 request 的序列 tile 索引
            flat_seq_tile = pid // num_heads
            h_i = pid % num_heads

            # 用前缀和 cum_seq_tiles 线性扫描找到所属 request（无累加变量）
            b_i = -1
            for _b in T.serial(batch):
                if flat_seq_tile < cum_seq_tiles_per_request[_b]:
                    b_i = _b
                    break
            cum_before = T.if_then_else(b_i == 0, 0, cum_seq_tiles_per_request[b_i - 1])
            s_local = flat_seq_tile - cum_before  # query tile index within request

            # 加载 Q tile（TND packed: 全局偏移 q_seq_starts[b_i] + 局部偏移）
            q_packed_start = q_seq_starts[b_i] + s_local * block_M
            T.copy(Q[q_packed_start : q_packed_start + block_M, h_i, :], q_l1)

            # 初始化 softmax 状态
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2.0**30))

            # kv_len for this request
            kv_len_b = actual_kv_len[b_i]
            kv_tiles_b = T.ceildiv(kv_len_b, block_N)

            # 加载 per-row mask 元数据（局部坐标）
            T.copy(visible_end[b_i, s_local * block_M : s_local * block_M + block_M],
                   visible_end_ub)
            T.copy(diag_col[b_i, s_local * block_M : s_local * block_M + block_M],
                   diag_col_ub)

            # 遍历 KV tiles（局部坐标）
            for k_i in range(kv_tiles_b):
                kv_local_start = k_i * block_N
                kv_packed_start = kv_seq_starts[b_i] + kv_local_start
                # 当前 tile 的有效列数（最后一 tile 可能不足 block_N）
                valid_cols = T.if_then_else(kv_local_start + block_N > kv_len_b,
                                            kv_len_b - kv_local_start, block_N)

                # ---- V scope: 加载 K (packed offset → GM) ----
                T.copy(K[kv_packed_start : kv_packed_start + block_N,
                         h_i // kv_group, :], k_l1)

                # ---- C scope: QK GEMM ----
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_s[cid, :, :])

                # ---- V scope: Mask + Softmax ----
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(workspace_s[cid, vid * v_block: vid * v_block + v_block, :],
                       acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                # 生成 kv_col 局部索引数组，多余列设哨兵值 INT_MAX（永不匹配掩码）
                for c in T.Parallel(block_N):
                    kv_col_ub[c] = T.if_then_else(c < valid_cols,
                                                   kv_local_start + c,
                                                   2147483647)
                T.barrier_all()  # 等待 T.Parallel 写 UB 完成

                # 逐行施加 mask（局部坐标比较）
                for row in T.serial(v_block):
                    row_i = vid * v_block + row
                    ve_float = T.cast(visible_end_ub[row_i], "float32")
                    dc_float = T.cast(diag_col_ub[row_i], "float32")
                    T.tile.compare(mask_ub, kv_col_ub, ve_float, "LT")
                    T.tile.compare(mask_diag_ub, kv_col_ub, dc_float, "EQ")
                    T.tile.bitwise_or(mask_ub, mask_ub, mask_diag_ub)
                    T.tile.select(acc_s_ub[row, :], mask_ub,
                                  acc_s_ub[row, :], -T.infinity(accum_dtype),
                                  "VSEL_TENSOR_SCALAR_MODE")

                # Online safe softmax
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)
                for row in T.serial(v_block):
                    T.tile.sub(acc_s_ub[row, :], acc_s_ub[row, :], m_i[row])
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                # 注意：当 m_i[row] 仍为 -inf 时（该行全 -inf），exp(-inf) = 0，
                # alpha = exp(m_i_prev - m_i) = exp(-inf - (-inf)) = exp(0) = 1，
                # 所以 sumexp *= 1, acc_o *= 1，保持不变。
                # 最终 sumexp 为 0 → div 不会执行（跳过无效行）。

                # P → workspace (V→C, dtype cast)
                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_p[cid, vid * v_block:
                                                vid * v_block + v_block, :])

                # ---- C scope: PV GEMM ----
                T.copy(workspace_p[cid, :, :], acc_s_l1)
                T.copy(V[kv_packed_start : kv_packed_start + block_N,
                         h_i // kv_group, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_o[cid, :, :])

                # ---- V scope: 累加输出 ----
                for row in T.serial(v_block):
                    T.tile.mul(acc_o[row, :], acc_o[row, :], m_i_prev[row])
                T.copy(workspace_o[cid, vid * v_block: vid * v_block + v_block, :],
                       acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # 归一化并输出（TND packed 全局偏移）
            # visible_end=-1 的无效行，sumexp 始终为 0，跳过 div
            for row in T.serial(v_block):
                T.tile.div(acc_o[row, :], acc_o[row, :], sumexp[row])
            T.copy(acc_o, acc_o_half)
            output_packed_start = q_packed_start + vid * v_block
            T.copy(acc_o_half,
                   Output[output_packed_start : output_packed_start + v_block,
                          h_i, :])
```

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.gemm_v0` | `examples/sparse_flash_attention/*.py` 广泛使用 | ✅ 已验证 |
| `T.tile.add/sub/mul/div/exp/max` | `examples/flash_attention/flash_attn_bhsd.py` | ✅ 已验证 |
| `T.tile.select` | `example_sparse_flash_attn_mask.py:214` | ✅ 已验证 |
| `T.tile.compare` | `example_sparse_flash_attn_mask.py:196` | ✅ 已验证 |
| `T.tile.bitwise_or` | `example_sparse_flash_attn_mask_pa.py:224` | ✅ 已验证 |
| `T.reduce_max / T.reduce_sum` | `examples/flash_attention/flash_attn_bhsd.py:142,162` | ✅ 已验证 |
| `T.Kernel(一维, is_npu=True)` | 所有 reference 均使用 | ✅ 已验证 |
| `AUTO_CV_COMBINE + AUTO_CV_SYNC + AUTO_SYNC` | `example_sparse_flash_attn.py` |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 是否涉及 | 处理方案 |
|------|----------|----------|
| 不支持三维 Kernel | No | 使用 `T.Kernel(core_num, is_npu=True)` 一维 + 固定核任务循环 |
| threads 参数限制（仅 1 或 2） | No | 使用默认值（2），不显式指定 |
| 动态循环边界不支持 | Yes | 使用固定 `core_num` + `T.serial(max_iters)` + `if pid < total_tasks` |
| 流水线不支持动态边界 | No | 使用 `T.serial`，不启用 `T.Pipelined` |
| GEMM 要求 M,N 为 block 整数倍 | Yes | `block_M=64` 要求 `seq_len` 在 host 侧对齐/裁剪 |
| L0C 容量限制 | No | `64*128*4=32KB < 128KB` ✓ |

### 3.5.2 参考实现差异说明

| 差异项 | GPU 参考（gpu.cpp） | 本项目（Ascend） | 转换方案 |
|--------|---------------------|-----------------|----------|
| Kernel 维度 | Hopper CTA grid (2D/3D) | 一维 `T.Kernel(core_num)` | 固定核 + 任务循环 |
| GEMM 方式 | WGMMA (TMA + SM90 MMA) | `T.gemm_v0` | `examples/flash_attention/flash_attn_bhsd.py` |
| 线程模型 | CUDA warp cooperative | Ascend C Cube/Vector 核 | `AUTO_CV_COMBINE` |
| Mask 方式 | GPU shared memory + CUDA | `T.tile.compare` + `T.tile.select` | `example_sparse_flash_attn_mask.py` |
| KV Cache | GPU global memory + TMA | `T.copy` from GM | `example_sparse_flash_attn_mask_pa.py` |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/flash_attention/flash_attn_bhsd.py` | 高度 | Online softmax 结构、GEMM API、同步模式 |
| `examples/sparse_flash_attention/example_sparse_flash_attn.py` | 高度 | Developer + AUTO passes、CV 融合、workspace 模式 |
| `examples/sparse_flash_attention/example_sparse_flash_attn_mask.py` | 高度 | T.tile.compare/select mask 生成、固定核循环 |
| `examples/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py` | 高度 | Paged KV cache、block_table lookup、动态 shape |
| `examples/sparse_flash_attention/example_sparse_flash_attn_gqa.py` | 中等 | GQA 模式、head padding 处理 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

> **Layout 说明**：严格遵循接口说明中的 TND packed 格式。
> Q/K/V 第 0 维为 `total_live_q`（所有 request 的 live token 总数，跨 request 物理拼接）。
> `q_seq_starts` / `kv_seq_starts` 给出每个 request 在 packed 维度的起始偏移。
>
> visible_end / diag_col 为 per-request **局部坐标**（以 request 内偏移为基准），掩码比较时转为局部坐标。

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Q | [total_live_q, num_heads, head_dim] | bf16 | Packed live Query（TND layout） |
| K | [total_live_kv, num_kv_heads, head_dim] | bf16 | Packed live Key（TND layout） |
| V | [total_live_kv, num_kv_heads, head_dim] | bf16 | Packed live Value（TND layout） |
| q_seq_starts | [batch] | int32 | 每个 request 在 packed Q 维的起始偏移 |
| kv_seq_starts | [batch] | int32 | 每个 request 在 packed KV 维的起始偏移 |
| actual_q_len | [batch] | int32 | 每个 request 的 live Q 长度 |
| actual_kv_len | [batch] | int32 | 每个 request 的 live KV 长度 |
| num_seq_tiles_per_request | [batch] | int32 | **Host 预计算**：每个 request 的 `ceildiv(actual_q_len[b], block_M)`（仅 Host 侧用于算 total_tasks，不传入内核） |
| cum_seq_tiles_per_request | [batch] | int32 | **Host 预计算**：前缀和 `cum[b] = Σ_{i≤b} num_seq_tiles[i]`，传入内核供 pid→(b,s,h) 映射 |
| total_tasks | 标量 | int32 | **Host 预计算**：`T.symbolic("total_tasks")`，`= Σ_b num_seq_tiles[b] * num_heads` |
| visible_end | [batch, max_request_len] | int32 | **Host 预计算**：每个 request 内每行的局部 visible_end，无效行填 **-1**（避免全 -inf 导致 NaN） |
| diag_col | [batch, max_request_len] | int32 | **Host 预计算**：每个 request 内每行的局部对角列索引，-1 表示无 |
| sm_scale | 标量 | float | Attention scale = 1/√head_dim |

**Phase 2 追加（Paged KV Cache）**：

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| key_cache | [num_blocks, block_size, num_kv_heads, head_dim] | bf16 | Paged K cache |
| value_cache | [num_blocks, block_size, num_kv_heads, head_dim] | bf16 | Paged V cache |
| block_table | [batch, max_blocks] | int32 | request→physical block 映射 |
| matched_prefix_lens | [batch] | int32 | 每个 request 的 prefix 长度 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | [total_live_q, num_heads, head_dim] | bf16 | Packed attention 输出（TND layout，与 Q 同结构） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 层级 | 用途 |
|-----------|-------|-------|------|------|
| q_l1 | [block_M, head_dim] | bf16 | L1 | Q tile |
| k_l1 | [block_N, head_dim] | bf16 | L1 | K tile |
| v_l1 | [block_N, head_dim] | bf16 | L1 | V tile |
| acc_s_l1 | [block_M, block_N] | bf16 | L1 | P = softmax(S) for PV GEMM |
| acc_s_l0c | [block_M, block_N] | fp32 | L0C | QK^T accumulator |
| acc_o_l0c | [block_M, head_dim] | fp32 | L0C | PV accumulator |
| acc_o | [v_block, head_dim] | fp32 | UB | 累积输出 O |
| sumexp | [v_block] | fp32 | UB | 运行 ∑exp |
| m_i | [v_block] | fp32 | UB | 运行 max |
| m_i_prev | [v_block] | fp32 | UB | 上一轮 max（计算 α 因子） |
| acc_s_ub | [v_block, block_N] | fp32 | UB | 当前 tile 的 attention scores |
| acc_s_ub_ | [v_block, block_N] | fp32 | UB | 从 workspace 读取的 scores |
| sumexp_i_ub | [v_block] | fp32 | UB | 当前 tile 的 ∑exp |
| acc_s_half | [v_block, block_N] | bf16 | UB | dtype cast 后的 P（写 workspace） |
| acc_o_ub | [v_block, head_dim] | fp32 | UB | 从 workspace 读取的 O_partial |
| acc_o_half | [v_block, head_dim] | bf16 | UB | dtype cast 后的输出 |
| visible_end_ub | [block_M] | int32 | UB | 当前 tile 的 per-row visible_end |
| diag_col_ub | [block_M] | int32 | UB | 当前 tile 的 per-row diag_col |
| kv_col_ub | [block_N] | int32 | UB | KV 列坐标 (用于 mask 生成) |
| mask_ub | [block_N // 8] | uint8 | UB | 主掩码 bitmask |
| mask_diag_ub | [block_N // 8] | uint8 | UB | 对角线掩码 bitmask |

### 4.4 内存搬运路径

```
[GM] Q ──T.copy──> [L1] q_l1
[GM] K ──T.copy──> [L1] k_l1
  [L1] q_l1 + k_l1 ──T.gemm_v0(transpose_B=True)──> [L0C] acc_s_l0c
  [L0C] acc_s_l0c ──T.copy──> workspace_s (GM) ──T.copy──> [UB] acc_s_ub_
  [UB] acc_s_ub_ ──mask, softmax──> [UB] acc_s_ub ──cast──> [UB] acc_s_half
  [UB] acc_s_half ──T.copy──> workspace_p (GM) ──T.copy──> [L1] acc_s_l1
[GM] V ──T.copy──> [L1] v_l1
  [L1] acc_s_l1 + v_l1 ──T.gemm_v0──> [L0C] acc_o_l0c
  [L0C] acc_o_l0c ──T.copy──> workspace_o (GM) ──T.copy──> [UB] acc_o_ub
  [UB] acc_o_ub ──accumulate──> [UB] acc_o
  [UB] acc_o ──normalize, cast──> [UB] acc_o_half ──T.copy──> [GM] Output
```

### 4.5 UB 内存预算

| Buffer | Shape | sizeof | 大小 (Bytes) |
|--------|-------|--------|-------------|
| acc_o | [32, 128] | fp32 | 16384 |
| sumexp | [32] | fp32 | 128 |
| m_i | [32] | fp32 | 128 |
| m_i_prev | [32] | fp32 | 128 |
| acc_s_ub | [32, 64] | fp32 | 8192 |
| acc_s_ub_ | [32, 64] | fp32 | 8192 |
| sumexp_i_ub | [32] | fp32 | 128 |
| acc_s_half | [32, 64] | bf16 | 4096 |
| acc_o_ub | [32, 128] | fp32 | 16384 |
| acc_o_half | [32, 128] | bf16 | 8192 |
| visible_end_ub | [64] | int32 | 256 |
| diag_col_ub | [64] | int32 | 256 |
| kv_col_ub | [64] | int32 | 256 |
| mask_ub | [8] | uint8 | 8 |
| mask_diag_ub | [8] | uint8 | 8 |
| **总计** | | | **~63KB** / 192KB (A2/A3) |

> MEMORY_PLANNING 将进一步复用非同时活跃的 buffer 地址，实际占用 < 63KB。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| batch | T.symbolic("batch") | 1~16 |
| total_live_q | T.symbolic("total_live_q") | sum(actual_q_len)，每 request ~3200 |
| total_live_kv | T.symbolic("total_live_kv") | sum(actual_kv_len)，与 total_live_q 同量级 |
| max_request_len | T.symbolic("max_request_len") | 对齐到 block_M 倍数，~3200 |
| total_tasks | T.symbolic("total_tasks") | `= cum_seq_tiles[-1] * num_heads`，~batch * 50 * H |

> `total_tasks` / `cm_seq_tiles_per_request` 在 Host 侧由 `actual_q_len` 计算后传入内核（避免 DSL 内对 tensor 值做累加）。

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(
    out_idx=[3],             # Output 参数索引
    workspace_idx=[11, 12, 13, 14],  # workspace 参数索引
    pass_configs=pass_configs,
)
def mtgr_ragged_segment_attention_fwd(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    block_M=64,
    block_N=64,
    core_num=24,
):
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（CV 融合）

**判定依据**: 算子包含 `T.gemm_v0`（Cube：QK 和 PV 两个 GEMM）+ `T.tile.*`（Vector：mask、softmax 更新）

### 5.2 Block 划分

```python
block_M = 64   # Query 行 tile（匹配 reference，≥ 16 ✓）
block_N = 64   # KV 列 tile（匹配 reference，≥ 16 ✓）
head_dim = 128 # 匹配 gpu.cpp HeadDim，2 的幂 ✓
v_block = block_M // 2  # 每个 Vector 核处理的 query 行数 (= 32)

# TND packed 任务数（跨 request 累积）
# Host 侧计算后传入内核（total_tasks 和 cum_seq_tiles 均为 T.symbolic）
num_seq_tiles_per_request[b] = ceildiv(actual_q_len[b], block_M)
cum_seq_tiles_per_request[b] = Σ_{i=0}^b num_seq_tiles_per_request[i]
total_tasks = cum_seq_tiles_per_request[batch-1] * num_heads
core_num = 24  # A2 设备默认核数，A3 为 20
```

### 5.3 约束分析

- **对齐约束**: block_M=64, block_N=64，fp16 尾轴 128，均满足分形对齐（≥16）
- **L0C 容量**: 
  - `acc_s_l0c[64, 64] * 4(f32) = 16KB < 128KB` ✓
  - `acc_o_l0c[64, 128] * 4(f32) = 32KB < 128KB` ✓
- **L1 容量**: 
  - `q_l1[64, 128] * 2(bf16) = 16KB + k_l1[64, 128] * 2 = 16KB + v_l1[64, 128] * 2 = 16KB = 48KB < 512KB` ✓
- **UB 容量**: ~63KB < 192KB ✓
- **非整除**: `ceildiv(seq_len, block_M)` 保证全覆盖，最后一 tile 不满时由 actual_q_len 控制有效行

### 5.4 非整除处理策略

**Query 维度**：
- `actual_q_len[b]` 按 `block_M=64` 取 `ceildiv` 分 tile
- 最后一 tile 不满 64 行时，超额行（local_row ≥ actual_q_len - s_local*block_M）的 `visible_end` 在 Host 侧预填 **-1**
- `kv_col < -1` 对任何 kv_col≥0 永不成立 → 整行 score 全 -inf → `m_i` 保持 -inf → `sumexp` 保持 0 → `acc_o` 保持初始值 0（`div` 跳过无效行）
- Output 仅写回有效行，由 `actual_q_len[b]` 控制范围

**KV 维度**：
- `actual_kv_len[b]` 按 `block_N=64` 取 `ceildiv` 分 tile
- 最后一 tile 不满 64 列时，多余列的 `kv_col_ub` 设哨兵值 `INT_MAX`（`2147483647`）
- 哨兵值永不满足 `kv_col < visible_end` 或 `kv_col == diag_col`，被自然遮蔽

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| (batch, seq_tile, head) 三维 | 核心任务循环 | `T.serial(max_iters)` + `if pid < total_tasks` | 固定核数，轮询所有 request 的 query tile |
| Request 查找 | T.serial(batch) 内线性扫描 | `T.serial(batch)` + if 条件 | batch ≤ 16，线性扫描代价可忽略 |
| KV 维度 | 迭代 | `T.serial(kv_tiles_b)` | 按 request 的 kv_len 分 tile，标准 FA 结构 |
| 行内 mask/exp | 迭代 | `T.serial(v_block)` | T.tile.* 不支持自动 broadcasting |

### 6.2 循环伪代码

```python
# 固定核心任务循环 (TND packed 版本)
for core_index in T.serial(T.ceildiv(total_tasks, core_num)):
    pid = core_index * core_num + cid
    if pid < total_tasks:
        # pid → (b_i, s_local, h_i): 线性扫描 request 定位
        b_i = find_request(pid // num_heads, num_seq_tiles_per_request)
        s_local = local_seq_tile(pid // num_heads, num_seq_tiles_per_request)

        # Q offset: q_seq_starts[b_i] + s_local * block_M
        q_packed_start = q_seq_starts[b_i] + s_local * block_M
        T.copy(Q[q_packed_start : q_packed_start + block_M, h_i, :], q_l1)

        # Visible_end / diag_col mask: 局部坐标 [b_i, s_local*block_M : ...]
        T.copy(visible_end[b_i, s_local*block_M : ...], visible_end_ub)

        # KV tile 迭代（局部坐标范围）
        for k_i in range(T.ceildiv(actual_kv_len[b_i], block_N)):
            kv_local_start = k_i * block_N
            kv_packed_start = kv_seq_starts[b_i] + kv_local_start

            # Cube: QK GEMM
            T.copy(K[kv_packed_start : ..., :], k_l1)
            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
            T.copy(acc_s_l0c, workspace_s[cid, :, :])

            # Vector: mask (局部坐标) + softmax
            for row in T.serial(v_block):
                # kv_col_ub[c] = kv_local_start + c (局部坐标)
                # compare against visible_end_ub[row_i] (局部坐标)
                T.tile.select(acc_s_ub[row, :], mask_ub, ..., -T.infinity(...))
            ... (online softmax) ...

            # Cube: PV GEMM
            T.copy(workspace_p[cid, :, :], acc_s_l1)
            T.copy(V[kv_packed_start : ..., :], v_l1)
            T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
            T.copy(acc_o_l0c, workspace_o[cid, :, :])

            # Vector: accumulate O
            T.copy(workspace_o[cid, vid * v_block:, :], acc_o_ub)
            T.tile.add(acc_o, acc_o, acc_o_ub)

        # 归一化 + 输出（TND packed 偏移）
        for row in T.serial(v_block):
            T.tile.div(acc_o[row, :], acc_o[row, :], sumexp[row])
        T.copy(acc_o_half, Output[q_packed_start + vid * v_block : ..., h_i, :])
```

### 6.3 流水线优化

不使用 `T.Pipelined`，保持 `T.serial`。Developer 模式的 `AUTO_CV_COMBINE` 自动处理核间流水线调度。

### 6.4 尾块处理

- Query 维度：`actual_q_len[b]` 按 `block_M=64` 取 `ceildiv` 分 tile。最后一 tile 不满 64 行时，超额行的 `visible_end` 在 Host 侧预填 `-1`，导致 score 全 -inf，`sumexp=0`，输出保持 0
- KV 维度：`actual_kv_len[b]` 按 `block_N=64` 取 `ceildiv` 分 tile。最后一 tile 多余列的 `kv_col_ub` 设哨兵值 `INT_MAX`，自然被 mask 遮蔽

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

全部由 `AUTO_SYNC` + `AUTO_CV_SYNC` 自动管理：
- Cube→Vector 数据传输前后（workspace 读写）
- GEMM 计算前后
- L0C 数据就绪后

**无需手动 `T.barrier_all()` / `T.set_cross_flag()` / `T.wait_cross_flag()`**。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步点插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动地址分配
}
```

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是

**判定依据**: 算子包含两个 GEMM（QK 和 PV）和 element-wise post-processing（mask、softmax、accumulation），需 Cube/Vector 核间协作。

### 8.2 workspace 设计

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| workspace_kv | [core_num, block_N, head_dim] | bf16 | V 核加载 K 后传输给 C 核（V→C） |
| workspace_s | [core_num, block_M, block_N] | fp32 | C 核 QK GEMM 结果 S（C→V） |
| workspace_p | [core_num, block_M, block_N] | bf16 | V 核 softmax 结果 P（V→C） |
| workspace_o | [core_num, block_M, head_dim] | fp32 | C 核 PV GEMM 结果 O_partial（C→V） |

**workspace_idx**: `[11, 12, 13, 14]`（参数位置：Q=0, K=1, V=2, Output=3, q_seq_starts=4, kv_seq_starts=5, actual_q_len=6, actual_kv_len=7, cum_seq_tiles_per_request=8, visible_end=9, diag_col=10, ws_kv=11, ws_s=12, ws_p=13, ws_o=14）

### 8.3 数据流时序（CV Pipeline）

```
迭代 k_i:
  │
  ├── V 核：加载 K[i] → workspace_kv[cid]
  │   AUTO_CV_SYNC
  ├── C 核：workspace_kv[cid] → k_l1
  │   C 核：QK GEMM → acc_s_l0c → workspace_s[cid]
  │   AUTO_CV_SYNC
  ├── V 核：workspace_s[cid] → acc_s_ub_
  │   V 核：mask + softmax → acc_s_half → workspace_p[cid]
  │   AUTO_CV_SYNC
  ├── C 核：workspace_p[cid] → acc_s_l1
  │   C 核：V[i] → v_l1 → PV GEMM → acc_o_l0c → workspace_o[cid]
  │   AUTO_CV_SYNC
  └── V 核：workspace_o[cid] → acc_o_ub
      V 核：acc_o += acc_o_ub, 更新 sumexp
```

### 8.4 pass_configs

Developer 模式下 `AUTO_CV_COMBINE` 自动处理上述 Pipeline 的执行分配和同步。

### 8.5 注意事项

- `workspace_idx` 必须与函数签名中 workspace 参数位置一致
- 固定核模式下 workspace 第一维为 `core_num`（不是 total_tasks），支持多任务复用
- Phase 2 引入 KV cache 时需要额外 workspace（V 数据也需要 V→C 传输）

---

## 9. 验证方案

### 9.1 Golden 函数

**来源**: `examples/sparse-fa/golden.py` 中的 `run_mtgr_torch_mask_attention_reference` - 基于 PyTorch 的 C++ 参考实现。

待转换为 Python golden：
```python
def golden_mtgr_attention(q, k, v, segment_offsets, segment_rules,
                           matched_prefix_lens, sm_scale):
    """Segment-based mask attention reference"""
    # 1. Build full visible mask from segment_offsets and rules
    # 2. Standard attention: softmax(QK^T * mask) * V
    # 3. Crop output to live tokens
    pass
```

### 9.2 测试用例

> Tensor 为 TND packed 格式（total_live = sum over batch of live_lens）。
> 下表用 (batch, max_len_per_request, heads, dim) 描述请求规格。

| 用例名 | Level | 请求规格 (B, max_len, H, D) | Segment 配置 | 说明 |
|--------|-------|-----------------|-------------|------|
| basic_4seg | Level 0 | (1, 128, 8, 128) | his=64,cxt=8,rt=32,tgt=24 | 最小四段功能验证 |
| basic_3seg | Level 0 | (1, 128, 8, 128) | his_cxt=72,rt=32,tgt=24 | 最小三段功能验证 |
| typical_4seg | Level 1 | (1, 3200, 32, 128) | his=2000,cxt=8,rt=192,tgt=1000 | 客户典型四段配置 |
| typical_3seg | Level 1 | (1, 3200, 32, 128) | his_cxt=2008,rt=192,tgt=1000 | 客户典型三段配置 |
| multi_batch | Level 1 | (2, 3200, 32, 128) | 各不同 ragged 长度 | 多 batch ragged 验证 |
| boundary_1tok | Level 2 | (1, 64, 1, 128) | his=32,cxt=8,rt=16,tgt=8 | 极小 token 边界 |
| boundary_nopad | Level 2 | (1, 63, 4, 128) | his=31,cxt=8,rt=16,tgt=8 | 非 block_M 对齐边界 |
| large_batch | Level 3 | (16, 3200, 32, 128) | 客户规格上限 | 性能测试 |
| ragged_3_5_7 | Level 2 | (3, 500, 4, 128) | len=500, 300, 100 | 三请求不等长 ragged |

### 9.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| bfloat16 | 1e-2 | 1e-2 |
| float16 | 1e-2 | 1e-2 |

---

## 10. 风险点与注意事项

### 10.1 已知约束

1. **Phase 1 不支持前缀缓存**：仅限 match_mode=0（no_match），所有 KV 来自 live memory（`kv_seq_starts` = `q_seq_starts` 在此模式下成立）
2. **`max_request_len` 需对齐到 block_M=64**：通过 `ceildiv(actual_q_len[b], block_M)` 分 tile，Host 负责将 `visible_end` 无效行填 0
3. **head_dim 必须为 2 的幂**：GEMM 对齐要求
4. **segment_rules 不支持动态调整**：规则数组为常量，编译期确定
5. **第三段仅支持 1 个子段**（530 截止）：后续需支持多子段

### 10.2 常见风险

| 风险 | 触发场景 | 影响 | 缓解措施 |
|------|----------|------|----------|
| UB 溢出 | 增加新 buffer 未重算 | 编译/运行时错误 | 每次修改前计算 UB 预算 |
| Mask 精度损失 | float32↔int32 cast | 边界比较误差 | 统一使用 float32 比较 |
| 全 -inf 行 NaN | `visible_end=0` 导致 score 全 -inf | 除零 / NaN 传播 | `visible_end=-1` 使其永不匹配，sumexp=0 时跳过 div |
| 各 batch 不等长 | 直接取 batch[0] 长度 | 短序列算到无效数据 | 使用 `actual_q_len` + `visible_end=-1` 跳过无效行 |
| L0C 溢出 | block_M × head_dim × sizeof(fp32) > 128KB | segfault | 验证 block_M ≤ 256 (当 head_dim=128) |
| AUTO_CV_COMBINE 不支持固定核循环 | 纯 AUTO passes 在固定核 + 动态任务判断下可能无法正确分离 CV | 编译失败或运行时错误 | 实现时优先尝试纯 AUTO，若失败则 fallback 到手动 `T.Scope("C"/"V")` + `T.set_cross_flag`（参考 `example_sparse_flash_attn_mask.py` 模式） |

### 10.3 Phase 2 扩展考虑

- 引入 `block_table` / `key_cache` / `value_cache` / `matched_prefix_lens`
- 需增加 workspace_v（V 从 V 核传输到 C 核）
- Mask 逻辑考虑前缀缓存带来的跨段 KV 访问

---

## 11. 交付清单

### 11.1 目录结构

```
examples/sparse-fa/
├── golden.py                           # 已存在: C++ PyTorch 参考实现
├── gpu.cpp                             # 已存在: CUDA Hopper WGMMA kernel 参考
├── 接口说明.txt                         # 已存在: 接口语义文档
├── 接口限制.txt                         # 已存在: 规格约束文档
├── design.md                           # 本设计文档
├── example_mtgr_ragged_segment_attn.py  # 待实现: 算子实现 + 基础测试
└── test_mtgr_ragged_segment_attn.py     # 待实现: 测试文件
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | 已完成 | 设计文档 |
| `example_mtgr_ragged_segment_attn.py` | 待实现 | 算子实现 |
| `test_mtgr_ragged_segment_attn.py` | 待实现 | 测试文件 |

### 11.3 命名规范

- 目录名: `sparse-fa`（已存在）
- 实现文件: `example_mtgr_ragged_segment_attn.py`
- 测试文件: `test_mtgr_ragged_segment_attn.py`

### 11.4 实现顺序

1. ✅ 设计文档（design.md）
2. ⬜ Host 预处理函数（segment → visible_end + diag_col） + Golden 函数
3. ⬜ 算子实现（example_mtgr_ragged_segment_attn.py）— Phase 1（无前缀缓存）
4. ⬜ 基础测试（Level 0：四段式 + 三段式小规模）
5. ⬜ 典型配置测试（Level 1：客户规格）
6. ⬜ 边界测试（Level 2：非对齐、极小 shape）
7. ⬜ Phase 2 扩展（前缀缓存支持，match_mode=1,2）
