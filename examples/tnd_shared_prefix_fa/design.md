# TND-Shared-Prefix FA 算子设计文档

## 1. 概述

### 1.1 算子名称

`tnd_shared_prefix_fa`（TND Shared-Prefix FlashAttention）

### 1.2 功能描述

大模型推理 prefill 阶段，同一 batch 内多条 request 共享完全相同的 prompt 前缀（shared-prefix）。本算子在该场景下执行 GQA 结构的 FlashAttention：shared-prefix 的 K/V 在全局内存只存一份，各 request 私有 K/V 独立存储；算子内部不涉及 KV-Cache 读写，仅做 prefill 前向计算。

> **TND 含义**：T=Token 维度，N=heads 头维度，D=head_dim。所有张量按 `[token, head, dim]` 紧凑排布（packed），无 padding 浪费。

### 1.3 数学公式

对 batch 内每一条 request `b`，完整输入序列 = `shared_prefix + private_b`。其完整 Q 的每一行做 attention 时，K/V 由公共前缀段与私有段拼接而成：

$$
O_{i} = \text{softmax}\!\left(\frac{Q_{i} \cdot [K_{\text{shared}};\, K_{\text{private}_b}]^{\top}}{\sqrt{d}}\right) \cdot [V_{\text{shared}};\, V_{\text{private}_b}]
$$

GQA 规则：`h_kv = h_q // group_size`，其中 `group_size = q_head // kv_head`。`group_size` 个 Query Head 共享同一套 KV Head。

### 1.4 算法描述

计算分两类独立 task：

1. **shared_prefix task**：Q tile 位于 `[0, shared_prefix_len)` 范围（shared-prefix 的 Q 只存一份，对所有 request 相同）。K/V 仅来自 `K_shared / V_shared`（causal mask 下 shared-prefix 的 Q 只能看到 shared-prefix 段的 K/V）。此 task 只算一次，输出写入 `Output[0:shared_prefix_len]`。
2. **private task**：Q tile 位于 request `b` 的私有段 `[shared_prefix_len + offset_b, ...)`。K/V = `K_shared + K_private_b`（先迭代 shared 段，再迭代 private 段）。causal mask 下 private Q 的所有行可看到全部 shared 段 K/V（位置在前），private 段需逐行 causal 判断。输出写入 `Output[shared_prefix_len + offset_b:...]`。

两类 task 各自独立做 online softmax（各自维护 `m_i / sumexp / acc_o`），无跨 task 数据依赖。

> **⚠️ Host 侧 Buffer 操作约束**（详见 SKILL.md §3.2）：host 侧只允许经证明共享原 storage、只改 metadata 的 view 操作，以及 kernel 调用和验证；禁止真实数据搬运和 aclnn 计算。本算子 host 侧仅负责输入张量准备、`block_metadata` 元数据构造（新建 int32 tensor，不触碰输入 buffer）和 kernel 调用，不涉及 `.contiguous() / torch.cat / torch.nn.functional.* / .to(dtype) / .clone()` 等禁止操作。

### 1.5 数据流图

```
GM[Q]                ──T.copy──→ L1[q_l1]  ──T.copy──→ L0A
GM[K_shared]         ──T.copy──→ L1[k_l1]  ──T.copy──→ L0B ──T.gemm_v0(Q,K^T)──→ L0C[acc_s] ──T.copy──→ UB
GM[K_private]        ──T.copy──→ L1[k_l1]
                                                              UB[acc_s] ──online softmax──→ UB[acc_s_half]
GM[block_metadata]   ──read──→  q_packed_start / q_valid / private_kv_start / private_kv_len (int32 scalars)
                                                              UB[acc_s_half] ──T.copy──→ L1[acc_s_l1] ──T.copy──→ L0A
GM[V_shared]         ──T.copy──→ L1[v_l1]  ──T.copy──→ L0B ──T.gemm_v0(P,V)──→ L0C[acc_o] ──T.copy──→ UB
GM[V_private]        ──T.copy──→ L1[v_l1]
                                                              UB[acc_o] ──normalize──→ UB[acc_o_half] ──T.copy──→ GM[Output]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

| 算子特征 | 分析 |
|----------|------|
| 计算类型 | CV 融合（GEMM + online softmax + GEMM，即 FlashAttention 结构） |
| 是否含 matmul | 是（两次 `T.gemm_v0`） |
| 是否含归约 | 是（`T.reduce_max / T.reduce_sum`） |
| 是否需要流水线 | 是（KV tile 迭代 + online softmax 累积） |
| 变长序列 | 是（TND packed + block_metadata 展平任务） |

选择 Developer 模式理由：
1. 参考实现 `examples/developer_mode/flash_attn_bshd_developer.py` 已验证 Developer 模式下 FA 的 `alloc_shared/alloc_fragment` + `threads=2` + pass_configs 全开写法可行，可直接复用 online softmax 结构。
2. Developer 模式由 pass 自动处理 CV 分离与同步，代码量约为 Expert 模式的 1/3，利于 L0 快速精度收敛。
3. 性能瓶颈预期在 KV 两段拼接的分支判断和 block_metadata 查表，而非 CV 同步粒度——Developer 模式足以覆盖。
4. 后续如性能不达标，可按升级路径切 Expert 模式（参考 `examples/gqa_fwd_varlen/gqa_fwd_varlen.py` 的 `T.mma` + 双缓冲 + 手动 flag 方案）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器映射 L1/UB）+ `T.alloc_fragment`（映射 L0C） |
| 计算方式 | `T.gemm_v0`（Cube）+ `T.tile.*` / `T.reduce_*`（Vector） |
| 作用域 | 编译器自动分离 CV 核（无显式 `T.Scope`） |
| 同步方式 | `AUTO_SYNC` + `AUTO_CV_SYNC` + `AUTO_CV_COMBINE` 自动处理 |
| threads | `threads=2`（消 vid 前提，单 `cid` 轴） |
| workspace | Developer 默认消除：片上 `alloc_shared/alloc_fragment` 直连，装饰器无 `workspace_idx` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `S = Q @ K^T` | Q tile 与 KV tile 的矩阵乘，输出 `[block_M, block_N]` score |
| 2 | `S = S * sm_scale` | 缩放 |
| 3 | `m_i = max(S_row, m_i_prev)` | online softmax 行最大值更新 |
| 4 | `P = exp(S - m_i)` | softmax 分子（逐行减最大值后 exp） |
| 5 | `sumexp = sumexp * exp(m_i_prev - m_i) + sum(P_row)` | online softmax 分母累积 |
| 6 | `acc_o = acc_o * exp(m_i_prev - m_i) + P @ V` | 输出累积（rescale + GEMM2） |
| 7 | `O = acc_o / sumexp` | 最终归一化 |
| 8 | K/V 两段拼接 | shared 段 + private 段，各自独立迭代 step 1-6，状态跨段共享 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | `S = Q @ K^T` | `T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)` | q_l1 `[block_M, dim]`, k_l1 `[block_N, dim]` → acc_s_l0c `[block_M, block_N]` | Developer |
| 2 | `S *= scale` | `T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)` | element-wise 乘标量 | Developer |
| 3 | 行最大值 | `T.reduce_max(acc_s_ub, m_i, dim=-1)` + `T.tile.max(m_i, m_i, m_i_prev)` | dim=-1 行归约 | Developer |
| 4 | exp | `T.tile.exp(acc_s_ub, acc_s_ub)` + 逐行 `T.tile.sub` | 先减 m_i 再 exp | Developer |
| 5 | 分母累积 | `T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)` + `T.tile.mul/add` | 行归约 + 标量运算 | Developer |
| 6 | `P @ V` | `T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)` | acc_s_l1 `[block_M, block_N]`, v_l1 `[block_N, dim]` → acc_o_l0c `[block_M, dim]` | Developer |
| 7 | 归一化 | 逐行 `T.tile.div(acc_o[h_i,:], acc_o[h_i,:], sumexp[h_i])` | 逐行除以 sumexp | Developer |
| 8 | KV 拼接 | `if kv_start < shared_prefix_len: T.copy(K_shared[...])` else `T.copy(K_private[...])` | 运行时 if 分支 | Developer |
| - | buffer 分配 | `T.alloc_shared` / `T.alloc_fragment` | Developer 自动映射 L1/UB/L0C | Developer |
| - | 尾块搬运 | `T.copy(src[start:start+valid], dst[:, :])` | 源端动态 valid extent，目标端完整 tile | Developer |
| - | 任务查表 | `block_metadata[tile_id, col]` | int32 GM tensor 索引 | Developer |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[6], pass_configs=pass_configs)
def tnd_shared_prefix_fa(
    q_head, kv_head, head_dim,
    shared_prefix_len, max_private_kv_len,
    total_q, total_private_kv, total_q_blocks,
    block_M=128, block_N=64,
    causal_mask=False,
    sm_scale=None,
):
    sm_scale = (1.0 / head_dim) ** 0.5 if sm_scale is None else sm_scale
    dtype = "float16"
    accum_dtype = "float32"
    group_size = q_head // kv_head

    max_shared_iters = (shared_prefix_len + block_N - 1) // block_N   # 编译期常量
    max_private_iters = (max_private_kv_len + block_N - 1) // block_N  # 编译期常量
    block_num = total_q_blocks * q_head                                # 编译期常量

    @T.prim_func
    def main(
        Q:            T.Tensor([total_q, q_head, head_dim], dtype),
        K_shared:     T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),
        V_shared:     T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),
        K_private:    T.Tensor([total_private_kv, kv_head, head_dim], dtype),
        V_private:    T.Tensor([total_private_kv, kv_head, head_dim], dtype),
        block_metadata: T.Tensor([total_q_blocks, 4], "int32"),
        Output:       T.Tensor([total_q, q_head, head_dim], dtype),
    ):
        with T.Kernel(block_num, threads=2, is_npu=True) as (cid):
            tile_id = cid // q_head
            h_q     = cid % q_head
            h_kv    = h_q // group_size

            q_packed_start   = block_metadata[tile_id, 0]
            q_valid          = block_metadata[tile_id, 1]
            private_kv_start = block_metadata[tile_id, 2]
            private_kv_len   = block_metadata[tile_id, 3]

            # ---- Buffer 分配（Developer: alloc_shared/alloc_fragment）----
            q_l1        = T.alloc_shared([block_M, head_dim], dtype)
            k_l1        = T.alloc_shared([block_N, head_dim], dtype)
            v_l1        = T.alloc_shared([block_N, head_dim], dtype)
            acc_s_l1    = T.alloc_shared([block_M, block_N], dtype)
            acc_s_l0c   = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c   = T.alloc_fragment([block_M, head_dim], accum_dtype)
            acc_o       = T.alloc_shared([block_M, head_dim], accum_dtype)
            sumexp      = T.alloc_shared([block_M], accum_dtype)
            m_i         = T.alloc_shared([block_M], accum_dtype)
            acc_s_ub    = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_ub_   = T.alloc_shared([block_M, block_N], accum_dtype)
            m_i_prev    = T.alloc_shared([block_M], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)
            acc_s_half  = T.alloc_shared([block_M, block_N], dtype)
            acc_o_ub    = T.alloc_shared([block_M, head_dim], accum_dtype)
            acc_o_half  = T.alloc_shared([block_M, head_dim], dtype)

            # ---- 初始化 ----
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2**30))

            # ---- 加载 Q tile ----
            T.copy(Q[q_packed_start : q_packed_start + block_M, h_q, :], q_l1)

            # ================ Shared 段 KV 迭代 ================
            for k in T.serial(max_shared_iters):
                kv_start = k * block_N
                if kv_start < shared_prefix_len:
                    # GEMM1: S = Q @ K_shared^T
                    T.copy(K_shared[kv_start : kv_start + block_N, h_kv, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, acc_s_ub_)

                    # Online softmax
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    for h_i in range(block_M):
                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    # GEMM2: O += P @ V_shared
                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(acc_s_half, acc_s_l1)
                    T.copy(V_shared[kv_start : kv_start + block_N, h_kv, :], v_l1)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, acc_o_ub)
                    for h_i in range(block_M):
                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                    T.tile.add(acc_o, acc_o, acc_o_ub)

            # ================ Private 段 KV 迭代 ================
            for k in T.serial(max_private_iters):
                kv_start = k * block_N
                if kv_start < private_kv_len:
                    private_offset = private_kv_start + kv_start
                    # GEMM1: S = Q @ K_private^T
                    T.copy(K_private[private_offset : private_offset + block_N, h_kv, :], k_l1)
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.copy(acc_s_l0c, acc_s_ub_)

                    # Online softmax（与 shared 段相同逻辑）
                    T.tile.fill(acc_s_ub, 0.0)
                    T.copy(m_i, m_i_prev)
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    for h_i in range(block_M):
                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i_ub)

                    # GEMM2: O += P @ V_private
                    T.copy(acc_s_ub, acc_s_half)
                    T.copy(acc_s_half, acc_s_l1)
                    T.copy(V_private[private_offset : private_offset + block_N, h_kv, :], v_l1)
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.copy(acc_o_l0c, acc_o_ub)
                    for h_i in range(block_M):
                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                    T.tile.add(acc_o, acc_o, acc_o_ub)

            # ---- 最终归一化 ----
            for h_i in range(block_M):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            # ---- 输出 ----
            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[q_packed_start : q_packed_start + block_M, h_q, :])

    return main
```

> **causal_mask 设计（L1 启用）**：在 `T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)` 之后、`T.reduce_max` 之前插入 mask 应用逻辑。参考 `examples/generative_recommendation/mtgr_ragged_segment_attention.py:508-536` 的 per-row `T.tile.fill(buf[row, col_start:col_end], NEG_INF)` 方式：
> - shared_prefix task 的 shared 段：Q 位置 `[0, shared_prefix_len)`，KV 位置 `[0, shared_prefix_len)`，需 causal mask
> - private task 的 shared 段：Q 位置 `>= shared_prefix_len`，KV 位置 `< shared_prefix_len`，全部可见（无需 mask）
> - private task 的 private 段：需 causal mask（`shared_prefix_len + kv_start + col <= q_packed_start + row`）
> - 尾块 KV（`kv_valid < block_N`）：列 `[kv_valid, block_N)` 设 `NEG_INF`

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|----------|----------|
| `T.alloc_shared` | `examples/developer_mode/flash_attn_bshd_developer.py:52-71` | ✅ 已验证 |
| `T.alloc_fragment` | 同上 :58-59 | ✅ 已验证 |
| `T.gemm_v0(…, transpose_B=True, init=True)` | 同上 :81, :109 | ✅ 已验证 |
| `T.copy(GM_slice, L1_buf)` | 同上 :76, :80, :108 | ✅ 已验证 |
| `T.tile.fill / add / mul / sub / exp / max / div` | 同上 :73-119 | ✅ 已验证 |
| `T.reduce_max / T.reduce_sum` | 同上 :90, :100 | ✅ 已验证 |
| `for h_i in range(block_M): T.tile.sub(…)` 逐行 | 同上 :96-97 | ✅ 已验证 |
| `T.Kernel(block_num, threads=2, is_npu=True) as (cid)` | 同上 :47 | ✅ 已验证 |
| `T.serial(max_iters)` + `if k < actual` 条件判断 | `examples/generative_recommendation/mtgr_ragged_segment_attention.py:265-267` | ✅ 已验证（mtgr 用 `T.serial` + `if` 分支） |
| GM int32 tensor 索引 `block_metadata[tile_id, col]` | `examples_experiment/grouped_gemm/example_grouped_gemm_fwd.py:69-71` | ✅ 已验证（grouped_gemm 用 `block_metadata[bx, 0/1]`） |
| `T.copy(src[start:start+valid], dst[:, :])` 动态 valid extent | `mtgr_ragged_segment_attention.py:260, 289-292` | ✅ 已验证（mtgr 用 `q_tile_size_live` 动态切片） |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(total_q_blocks * q_head)` + `cid // q_head` / `cid % q_head` 分解 |
| threads 参数限制（仅 1 或 2） | Yes | `threads=2`（Developer 模式消 vid） |
| 动态循环边界不支持 | Yes | `max_shared_iters` / `max_private_iters` 为编译期常量（Python int 上界），kernel 内 `T.serial(max_iters)` + `if k * block_N < actual_len` 条件判断 |
| 流水线不支持动态边界 | Yes | 不使用 `T.Pipelined`（shared/private 两段各自 `T.serial`），pass 自动优化 |
| L0C 容量上限（128KB） | Yes | `block_M × block_N × sizeof(float32) = 128×64×4 = 32KB < 128KB` ✓ |
| T.copy 列方向 strided 切片 | No | 所有 T.copy 切片为 2D `[start:end, head_idx, :]`，最内维 `head_dim` 连续 |

### 3.5.2 参考实现差异说明

**重要**：外部参考实现（GPU 版 FA / vLLM shared-prefix FA）不可直接使用，必须转换为 Ascend 兼容方案。

| 差异项 | 参考实现（GPU / 通用） | 本项目（Ascend） | 转换方案 |
|--------|----------------------|-----------------|----------|
| Kernel 维度 | 三维 `T.Kernel(m_num, h_num, batch)` | 一维 `T.Kernel(total_q_blocks * q_head)` + cid 分解 | 参考 `flash_attn_bshd_developer.py:47-50` |
| 循环边界 | 动态 `T.Pipelined(kv_len // block_N)` | 静态 `T.serial(max_iters)` + `if` 条件 | 参考 `mtgr_ragged_segment_attention.py:265-267` |
| GEMM API | `T.gemm` 通用版 | `T.gemm_v0` | 参考 `flash_attn_bshd_developer.py:81` |
| 内存分配 | `T.alloc_shared` 自动映射 | `T.alloc_shared` + `T.alloc_fragment`（Developer） | 参考 `flash_attn_bshd_developer.py:52-71` |
| threads | `threads=128` | `threads=2` | NPU 限制 |
| KV 拼接 | host 侧 `torch.cat([k_shared, k_private])` | kernel 内两段循环 `if kv_start < shared_len` 分支 | §3.2 Host 侧禁止 `torch.cat`（aclnn），改为 kernel 内分支 |
| 任务调度 | 三维并行 `(batch, q_block, head)` | block_metadata 展平 + cid 查表 | 参考 `grouped_gemm/example_grouped_gemm_fwd.py` |

### 3.5.3 本项目同类实现参考

**必须列出**：本项目 examples/ 中最相似的实现

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/developer_mode/flash_attn_bshd_developer.py` | 高度相似 | Developer 模式 FA 完整结构：`threads=2` + `alloc_shared/alloc_fragment` + online softmax + `T.gemm_v0` + `T.Pipelined` |
| `examples/generative_recommendation/mtgr_ragged_segment_attention.py` | 高度相似 | TND packed `[total_q, heads, dim]` + shared prefix K/V 两段拼接（`if kv_start < prefix_len` 分支）+ GQA（`h_kv = h_i // kv_group`）+ 变长序列（`q_seq_starts`） |
| `examples/varlen_paged_flash_attention/DESIGN_v1_simple.md` | 中度相似 | 变长 + shared prefix FA 的设计模板（6 决策点矩阵、host 端构造函数、升级路径） |
| `examples_experiment/grouped_gemm/example_grouped_gemm_fwd_ptr.py` | 中度相似 | block_metadata 3 列结构与 host 端 padding 构造（`[batch_idx, m_start, valid_m]`） |
| `examples/gqa_fwd_varlen/gqa_fwd_varlen.py` | 中度相似 | GQA + varlen + mask + Expert CV pipeline（mask 同步、apply_mask 开关可移植） |
| `examples/generative_recommendation/golden.py` | 辅助 | shared prefix K/V 拼接的 golden reference 写法（`torch.cat([k_prefix, k_live])`） |

### 3.5.4 分派覆盖审计

```text
[DISPATCH-COVERAGE]
supported_domain:
  - dtype: float16, bfloat16
  - head_dim: 64（编译期常量，后续可扩展为 32/128/256）
  - q_head, kv_head: 任意正整数，要求 q_head % kv_head == 0（GQA 约束）
  - batch, total_q, total_private_kv: 任意正整数（编译期常量传入）
  - shared_prefix_len: 任意非负整数（编译期常量，=0 退化为标准 varlen GQA FA）
  - causal_mask: True/False（编译期开关）
generic_fallback: 单 kernel 路径（无特化分派，所有合法输入走同一路径）
specializations: none
unsupported_inputs: head_dim 非 64、q_head % kv_head != 0、shared_prefix_len < 0
result: pass
```

判定：`generic_fallback == none` 时所有分支谓词并集覆盖 supported_domain。本算子无特化分支，单一路径覆盖全部声明域。✓

### 3.5.5 Host 侧 Buffer 操作审计

```text
[HOST-METADATA-AUDIT]
operation: torch.tensor(metadata_list, dtype=torch.int32, device=device)
input_stride -> output_stride: N/A（新建 tensor，非 view 操作）
shares_storage / same_data_ptr: false（新建 tensor，不共享任何输入 buffer 的 storage）
aclnn_or_physical_copy: false（torch.tensor 从 Python list 构造 int32 tensor，在 device 上创建，不触发 aclnn 算子）
result: allow

operation: kernel 调用 func(Q, K_shared, V_shared, K_private, V_private, block_metadata, Output)
input_stride -> output_stride: N/A（直接传入，无 host 侧 tensor 操作）
shares_storage / same_data_ptr: true（Output 由调用方预分配，kernel 写入）
aclnn_or_physical_copy: false
result: allow

operation: 结果验证 torch.testing.assert_close(ref_output, output, ...)
input_stride -> output_stride: N/A（只读比较，不修改 buffer）
shares_storage / same_data_ptr: N/A
aclnn_or_physical_copy: false（CPU 侧比较，或 .cpu() 只读拷贝用于验证）
result: allow
```

全部 host 侧操作通过审计。无 `.contiguous()` / `torch.cat` / `torch.nn.functional.*` / `.to(dtype)` / `.clone()` 等禁止行为。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Q | `[total_q, q_head, head_dim]` | float16 | 全 batch token 的 Query，排布 `[shared_prefix Q \| B0 private Q \| B1 private Q \| …]` |
| K_shared | `[shared_prefix_len, kv_head, head_dim]` | float16 | 共享前缀 K（仅 1 份，全 batch 复用） |
| V_shared | `[shared_prefix_len, kv_head, head_dim]` | float16 | 共享前缀 V（仅 1 份） |
| K_private | `[total_private_kv, kv_head, head_dim]` | float16 | 各 request 私有 K，紧凑排布 `[B0 private K \| B1 private K \| …]` |
| V_private | `[total_private_kv, kv_head, head_dim]` | float16 | 各 request 私有 V，同上排布 |
| block_metadata | `[total_q_blocks, 4]` | int32 | 任务元数据表（详见 §4.6） |

> **dtype 说明**：§4.1 只填代表性 dtype（float16）。完整 dtype 全集（float16, bfloat16）见 §9.3 精度表与 `proto.yaml`。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| Output | `[total_q, q_head, head_dim]` | float16 | Attention 输出，token 排布与输入 Q 完全对齐 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| q_l1 | `[block_M, head_dim]` | float16 | L1（alloc_shared） | Q tile 缓冲 |
| k_l1 | `[block_N, head_dim]` | float16 | L1 | K tile 缓冲（shared/private 复用） |
| v_l1 | `[block_N, head_dim]` | float16 | L1 | V tile 缓冲（shared/private 复用） |
| acc_s_l1 | `[block_M, block_N]` | float16 | L1 | softmax P 矩阵（GEMM2 输入） |
| acc_s_l0c | `[block_M, block_N]` | float32 | L0C（alloc_fragment） | Q@K^T 累积器 |
| acc_o_l0c | `[block_M, head_dim]` | float32 | L0C | P@V 累积器 |
| acc_o | `[block_M, head_dim]` | float32 | UB（alloc_shared） | 输出累积（online softmax 状态） |
| sumexp | `[block_M]` | float32 | UB | softmax 分母累积 |
| m_i | `[block_M]` | float32 | UB | 行最大值（当前） |
| m_i_prev | `[block_M]` | float32 | UB | 行最大值（上一 tile） |
| acc_s_ub | `[block_M, block_N]` | float32 | UB | score 缓冲（scale 后） |
| acc_s_ub_ | `[block_M, block_N]` | float32 | UB | score 缓冲（GEMM1 输出） |
| sumexp_i_ub | `[block_M]` | float32 | UB | 当前 tile 分母 |
| acc_s_half | `[block_M, block_N]` | float16 | UB | P 矩阵 fp16 版（L1 中转） |
| acc_o_ub | `[block_M, head_dim]` | float32 | UB | P@V 输出（当前 tile） |
| acc_o_half | `[block_M, head_dim]` | float16 | UB | 输出 fp16 版 |

### 4.4 内存搬运路径

```
GM[Q]                ──T.copy──→ L1[q_l1]
GM[K_shared/private] ──T.copy──→ L1[k_l1]          # 两段复用同一 k_l1
GM[V_shared/private] ──T.copy──→ L1[v_l1]          # 两段复用同一 v_l1

L1[q_l1] ──────────────────────→ L0A (隐式, T.gemm_v0 内部)
L1[k_l1] ──────────────────────→ L0B (隐式, T.gemm_v0 内部)
L0A + L0B ──T.gemm_v0(Q,K^T)──→ L0C[acc_s_l0c] ──T.copy──→ UB[acc_s_ub_]

UB[acc_s_ub_] ──scale+softmax──→ UB[acc_s_ub] ──T.copy──→ UB[acc_s_half] ──T.copy──→ L1[acc_s_l1]

L1[acc_s_l1] ──────────────────→ L0A (隐式)
L1[v_l1] ──────────────────────→ L0B (隐式)
L0A + L0B ──T.gemm_v0(P,V)─────→ L0C[acc_o_l0c] ──T.copy──→ UB[acc_o_ub]

UB[acc_o] ──rescale+add──→ UB[acc_o]  (跨 tile 累积)
UB[acc_o] ──normalize────→ UB[acc_o_half] ──T.copy──→ GM[Output]
```

> 内存层级不可跨级：GM → L0 直接搬运违规。GM 必须先 → L1/UB，再 → L0A/L0B。

### 4.5 UB 内存预算

以 `block_M=128, block_N=64, head_dim=64, dtype=float16, accum_dtype=float32` 计算：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| q_l1 | 128×64 | float16 | 16,384 (16 KB) |
| k_l1 | 64×64 | float16 | 8,192 (8 KB) |
| v_l1 | 64×64 | float16 | 8,192 (8 KB) |
| acc_s_l1 | 128×64 | float16 | 16,384 (16 KB) |
| acc_o | 128×64 | float32 | 32,768 (32 KB) |
| acc_s_ub | 128×64 | float32 | 32,768 (32 KB) |
| acc_s_ub_ | 128×64 | float32 | 32,768 (32 KB) |
| acc_s_half | 128×64 | float16 | 16,384 (16 KB) |
| acc_o_ub | 128×64 | float32 | 32,768 (32 KB) |
| acc_o_half | 128×64 | float16 | 16,384 (16 KB) |
| m_i / m_i_prev / sumexp / sumexp_i_ub | 128×1 each | float32 | 2,048 (2 KB) |
| **简单累加** | | | **213 KB** |

> **⚠️ 简单累加超过 192KB**。但 Developer 模式下 `TL_ASCEND_MEMORY_PLANNING=True` 会自动进行生命周期分析并复用 UB 地址（参考 `flash_attn_bshd_developer.py` 简单累加 ~344KB 仍可运行）。pass 会识别 `acc_s_ub` 与 `acc_s_ub_` 的不重叠生命周期、`acc_s_half` 与 `acc_o_half` 的不重叠生命周期等进行地址复用。预计实际 UB 占用 ~120-150 KB。
>
> **备选方案**（若 MEMORY_PLANNING 仍不足）：降级 `block_M=64, block_N=64`，简单累加 ~114 KB < 192KB，无需依赖 pass 优化。L0 先用 `block_M=128` 验证，若编译失败切 `block_M=64`。

### 4.6 block_metadata 结构

```python
block_metadata: T.Tensor([total_q_blocks, 4], "int32")
# 列定义:
#   [0] q_packed_start   — 该 Q tile 在 total_q 中的起始偏移
#   [1] q_valid          — 该 Q tile 有效行数（尾块 < block_M 时用于裁剪输出）
#   [2] private_kv_start — 该 task 对应 request 的私有 KV 在 total_private_kv 中的起始偏移
#                          （shared_prefix task = 0，不使用）
#   [3] private_kv_len   — 该 task 对应 request 的私有 KV 长度
#                          （shared_prefix task = 0，private 段循环跳过）
```

**Host 端构造逻辑**（纯 Python 整数运算，新建 int32 tensor，不触碰输入 buffer）：

```python
def build_block_metadata(
    shared_prefix_len: int,
    private_q_lens: list[int],   # 每个 request 的私有 Q/KV 长度（两者相同）
    block_M: int,
    q_head: int,
):
    metadata_list = []
    # 1. shared_prefix tasks
    for i in range(math.ceil(shared_prefix_len / block_M)):
        q_start = i * block_M
        q_valid = min(block_M, shared_prefix_len - q_start)
        metadata_list.append([q_start, q_valid, 0, 0])  # private_kv_start=0, private_kv_len=0

    # 2. private tasks
    priv_offset = 0  # 在 total_private_kv 中的累积偏移
    for b, priv_len in enumerate(private_q_lens):
        q_packed_offset = shared_prefix_len + priv_offset  # Q 中 private 段的起始
        for i in range(math.ceil(priv_len / block_M)):
            q_start = q_packed_offset + i * block_M
            q_valid = min(block_M, priv_len - i * block_M)
            metadata_list.append([q_start, q_valid, priv_offset, priv_len])
        priv_offset += priv_len

    return torch.tensor(metadata_list, dtype=torch.int32, device=device)
```

### 4.7 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| total_q | Python int 编译期常量传入 | shared_prefix_len + Σ private_q_len_b |
| total_private_kv | Python int 编译期常量传入 | Σ private_q_len_b |
| total_q_blocks | Python int 编译期常量传入 | ceildiv(shared_prefix_len, block_M) + Σ ceildiv(private_q_len_b, block_M) |

> **说明**：以上三个值作为 Python int 传入 `@tilelang.jit` 函数（编译期常量），不同输入规模触发重新编译（tilelang cache 后无额外开销）。如需运行时灵活变长，后续版本可改为 `T.symbolic` + 固定 `core_num` + 动态任务分配（参考 `mtgr_ragged_segment_attention.py:196-205`）。

### 4.8 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动 CV 跨核同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
}

@tilelang.jit(
    out_idx=[6],                # Output 是第 7 个参数（index 6）
    pass_configs=pass_configs,
)
def tnd_shared_prefix_fa(...):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: CV 融合（GEMM + online softmax + GEMM）

**判定依据**: 算子包含两次 `T.gemm_v0`（Q@K^T 和 P@V）+ 中间 element-wise softmax 后处理，判定为 CV 融合算子。

### 5.2 Block 划分

```python
block_M = 128   # Q tile 行数（参考 flash_attn_bshd_developer.py）
block_N = 64    # KV tile 行数（= head_dim，L0B/L0C 最优分形）
                  # 注意: block_N >= 32（GEMM2 的 K 维 = block_N >= 32 分形限制）

block_num = total_q_blocks * q_head   # 编译期常量
```

**业务典型场景任务数估算**（batch=10, q_head=14, shared_prefix_len=24, private_kv_len_avg=150, block_M=128, block_N=64）：

| task 类型 | tile 数 | × heads | 任务数 |
|-----------|---------|---------|--------|
| shared_prefix | ceildiv(24, 128) = 1 | × 14 | 14 |
| private | ceildiv(150, 128) × 10 = 2 × 10 | × 14 | 280 |
| **总计** | | | **294** |

### 5.3 约束分析

- **分形限制**: block_M=128 ≥ 16 ✓, block_N=64 ≥ 32（GEMM2 K 维）✓, head_dim=64 ≥ 32（GEMM1 K 维）✓
- **L0C 容量**: `block_M × block_N × sizeof(float32) = 128×64×4 = 32KB < 128KB` ✓
- **UB 容量**: 简单累加 213KB，依赖 `MEMORY_PLANNING` pass 复用后预计 ~120-150KB < 192KB ✓（备选 block_M=64 → 114KB）
- **L1 容量**: q_l1(16KB) + k_l1(8KB) + v_l1(8KB) + acc_s_l1(16KB) = 48KB < 512KB ✓

### 5.4 注意事项

**非整除处理**（shared_prefix_len / private_kv_len 不被 block_N 整除）：

- **KV 尾块**：`kv_valid = min(block_N, kv_total - kv_start)`，源端用 `T.copy(K[...:...+kv_valid, ...], k_l1[:, :])` 动态 extent（参考 `mtgr_ragged_segment_attention.py:289-292`）。无效 KV 列在 acc_s 上设 `NEG_INF`（`T.tile.fill(mask_ub[row, kv_valid:block_N], NEG_INF)` + `T.tile.add(acc_s_ub, acc_s_ub, mask_ub)`），L1 启用。
- **Q 尾块**：`q_valid = min(block_M, q_len - tile_idx * block_M)`，输出时用 `T.copy(acc_o_half[:q_valid, :], Output[q_start:q_start+q_valid, ...])` 限制有效行（参考 `mtgr_ragged_segment_attention.py:622-630`），L1 启用。
- **L0 简化**：L0 用例要求所有长度被 block_M / block_N 整除，不触发尾块逻辑。

> **⚠️ 非整除必须显式设计**（详见 SKILL.md §3.2）：输入、输出 GM 两侧使用 `valid_*` extent 的 BufferRegion，前端按动态切片裁剪搬运。不得使用标量 GM 起点配完整 UB tile；host 侧不允许 padding + crop。

### 5.5 数据搬运性能可行性

本算子不涉及数据重排（无 permute/transpose/reshape 物理化），TND packed 布局直接使用。所有 `T.copy` 均为 2D 行连续切片（最内维 `head_dim` 连续），无列方向 strided access。

| 路径 | 代表性最大 case | GM pass | DMA 数/平均字节 | GM 标量访问 | 地址 div/mod | AIV 并行度 | 结论 |
|------|-----------------|---------|------------------|-------------|--------------|------------|------|
| 单一 TND packed 路径 | batch=10, total_q=2205, head_dim=64, fp16 | 每 tile: Q(1R) + K(1R×5iters) + V(1R×5iters) + O(1W) = 12 pass/tile | ~294 tiles × 12 = 3528 DMA, avg 8KB | 0（无标量 GM 访问） | 0（无 div/mod，cid 整数分解） | 294 tasks / 24 cores ≈ 12 serial/core | 可行 |

> 无大张量逐元素 strided GM 主路径。block_metadata 查表为 int32 GM 访问（每 task 4 个 int32 = 16B），可忽略。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| task 级（block 间） | 隐式并行 | `T.Kernel(block_num, threads=2)` | 每个 block 处理一个 (q_tile, head) task |
| shared 段 KV 迭代 | 串行 | `T.serial(max_shared_iters)` | 编译期常量上界 + `if` 条件跳过空迭代 |
| private 段 KV 迭代 | 串行 | `T.serial(max_private_iters)` | 编译期常量上界 + `if k < actual` 条件 |
| 逐行 softmax | 串行 | `for h_i in range(block_M): T.tile.sub/mul/div` | `T.tile` 不支持行广播，需逐行（参考 `flash_attn_bshd_developer.py:96-98`） |

### 6.2 循环伪代码

```python
with T.Kernel(block_num, threads=2, is_npu=True) as (cid):
    # 查表获取 task 上下文
    tile_id = cid // q_head
    h_q     = cid % q_head
    h_kv    = h_q // group_size
    q_packed_start   = block_metadata[tile_id, 0]
    q_valid          = block_metadata[tile_id, 1]
    private_kv_start = block_metadata[tile_id, 2]
    private_kv_len   = block_metadata[tile_id, 3]

    # 加载 Q tile + 初始化 online softmax 状态
    T.copy(Q[q_packed_start:..., h_q, :], q_l1)
    T.tile.fill(acc_o, 0.0); T.tile.fill(sumexp, 0.0); T.tile.fill(m_i, -(2**30))

    # Shared 段循环
    for k in T.serial(max_shared_iters):
        if k * block_N < shared_prefix_len:
            T.copy(K_shared[...], k_l1)
            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
            # ... online softmax + GEMM2 ...

    # Private 段循环（shared_prefix task 的 private_kv_len=0，全部跳过）
    for k in T.serial(max_private_iters):
        if k * block_N < private_kv_len:
            T.copy(K_private[private_kv_start + ...], k_l1)
            T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
            # ... online softmax + GEMM2 ...

    # 归一化 + 输出
    for h_i in range(block_M):
        T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
    T.copy(acc_o, acc_o_half)
    T.copy(acc_o_half, Output[q_packed_start:..., h_q, :])
```

### 6.3 流水线优化

**当前版本**：不使用 `T.Pipelined`（shared/private 两段各自 `T.serial`，因两段循环边界不同且需条件判断，`T.Pipelined` 不支持动态边界）。Developer 模式下 `AUTO_CV_COMBINE` + `AUTO_CV_SYNC` pass 会自动插入 CV 流水线。

**升级路径**（Expert 模式）：参考 `mtgr_ragged_segment_attention.py` 的 `T.mma` + L0A/L0B/L0C 双缓冲 + `num_stages=14` + 6 跨核 semaphore 手动流水线。

### 6.4 尾块处理

- **KV 尾块**（`shared_prefix_len % block_N ≠ 0` 或 `private_kv_len % block_N ≠ 0`）：源端动态 `T.copy(K[start:start+kv_valid, ...], k_l1[:, :])`，无效列 mask `NEG_INF`（L1 启用）
- **Q 尾块**（`q_len % block_M ≠ 0`）：输出端 `T.copy(acc_o_half[:q_valid, :], Output[start:start+q_valid, ...])`（L1 启用）
- **L0 简化**：所有长度被 block_M / block_N 整除

> **⚠️ 尾块必须显式设计**（详见 SKILL.md §3.2）：输入、输出 GM 两侧都使用 `valid_*` extent 的 BufferRegion；前端按这些动态切片裁剪搬运。不得用标量 GM 起点配完整 UB tile，也不得在 host 侧 padding + crop。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下由 pass_configs 自动处理，无需手动同步：

| Pass | 作用 |
|------|------|
| `TL_ASCEND_AUTO_SYNC` | 自动插入核内同步（Cube/Vector 同核内） |
| `TL_ASCEND_AUTO_CV_SYNC` | 自动插入 Cube ↔ Vector 跨核同步 |
| `TL_ASCEND_AUTO_CV_COMBINE` | 自动合并 CV 同步点（减少 barrier 开销） |
| `TL_ASCEND_MEMORY_PLANNING` | 自动内存规划（UB/L1 地址分配与复用） |

> 本算子不使用 `T.Scope("C"/"V")`、`T.set_flag` / `T.wait_flag`、`T.set_cross_flag` / `T.wait_cross_flag`、`T.barrier_all`——全部由 pass 自动处理。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是

**判定依据**: 算子包含两次 GEMM（Q@K^T + P@V）+ 中间 online softmax element-wise 后处理，判定为 CV 融合算子。

### 8.2 CV 交互设计（Developer 模式，默认消除 workspace/vid）

- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（单轴 + `threads=2`）
- 装饰器无 `workspace_idx`，签名无 `workspace_*` 参数
- Cube↔Vector 改片上 `alloc_shared/alloc_fragment` 直连，中转/同步交给四个 pass
- 模板见 `tilelang-programming-model-guide mode-examples.md §6`

**不产出 workspace 规格**（Developer 默认消除）。如后续切 Expert/混合模式回退，再填写 workspace 表。

### 8.3 Cube 核计算流程

```python
# Developer（推荐）：Cube 输出直连片上 buffer，无 workspace
T.copy(Q[...], q_l1)                        # GM → L1
T.copy(K_shared[...] / K_private[...], k_l1) # GM → L1（两段复用 k_l1）
T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)  # L0A+L0B → L0C
T.copy(acc_s_l0c, acc_s_ub_)                # L0C → UB（片上直连）

T.copy(acc_s_half, acc_s_l1)                # UB → L1
T.copy(V_shared[...] / V_private[...], v_l1) # GM → L1
T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)  # L0A+L0B → L0C
T.copy(acc_o_l0c, acc_o_ub)                 # L0C → UB（片上直连）
```

### 8.4 Vector 核计算流程

```python
# Developer（推荐）：从片上 buffer 直读，无 workspace、无 vid 偏移
T.tile.fill(acc_s_ub, 0.0)
T.copy(m_i, m_i_prev)
T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
# ... online softmax (reduce_max, exp, reduce_sum, rescale) ...
T.copy(acc_s_ub, acc_s_half)                # fp32 → fp16（L1 中转）

# O 累加
for h_i in range(block_M):
    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
T.tile.add(acc_o, acc_o, acc_o_ub)

# 归一化 + 输出
for h_i in range(block_M):
    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
T.copy(acc_o, acc_o_half)
T.copy(acc_o_half, Output[...])             # UB → GM
```

### 8.5 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动 CV 跨核同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 内存规划
}
```

### 8.6 注意事项

- Developer 模式默认消除 workspace/vid：`threads=2` 是消 vid 前提，消 vid 是消 workspace 前提
- 本算子 shared/private 两段循环复用同一组 L1/UB buffer（k_l1, v_l1, acc_s_l1 等），pass 会识别生命周期并复用地址
- 两段循环的 online softmax 状态（m_i, sumexp, acc_o）跨段共享——**不可**在两段之间重置，否则 online softmax 累积断裂

---

## 9. 验证方案

### 9.1 Golden 函数

```python
def ref_tnd_shared_prefix_fa(
    Q: torch.Tensor,            # [total_q, q_head, head_dim]
    K_shared: torch.Tensor,     # [shared_prefix_len, kv_head, head_dim]
    V_shared: torch.Tensor,     # [shared_prefix_len, kv_head, head_dim]
    K_private: torch.Tensor,    # [total_private_kv, kv_head, head_dim]
    V_private: torch.Tensor,    # [total_private_kv, kv_head, head_dim]
    shared_prefix_len: int,
    private_q_lens: list[int],  # 每个 request 的私有 Q/KV 长度
    q_head: int,
    kv_head: int,
    head_dim: int,
    sm_scale: float = None,
    causal_mask: bool = True,
):
    """对每条 request 用完整序列（shared + private）计算 attention，结果与 Q 排布对齐。"""
    import math
    sm_scale = (1.0 / head_dim) ** 0.5 if sm_scale is None else sm_scale
    group_size = q_head // kv_head
    total_q = Q.shape[0]
    dtype = Q.dtype
    Q = Q.float(); K_shared = K_shared.float(); V_shared = V_shared.float()
    K_private = K_private.float(); V_private = V_private.float()

    O = torch.zeros((total_q, q_head, head_dim), dtype=torch.float32)

    # 1. shared_prefix 段：Q[0:shared_prefix_len] 的 attention，KV 只用 shared
    if shared_prefix_len > 0:
        q_seg = Q[:shared_prefix_len]                          # [sp_len, q_head, dim]
        k_seg = K_shared                                       # [sp_len, kv_head, dim]
        v_seg = V_shared
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]                               # [sp_len, dim]
            k = k_seg[:, h_kv, :]                              # [sp_len, dim]
            v = v_seg[:, h_kv, :]
            scores = torch.matmul(q, k.T) * sm_scale           # [sp_len, sp_len]
            if causal_mask:
                mask = torch.triu(torch.ones(sp_len, sp_len), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            O[:shared_prefix_len, h_q, :] = torch.matmul(attn, v)

    # 2. private 段：每条 request 的 Q 做 attention，KV = shared + private
    priv_offset = 0
    for b, priv_len in enumerate(private_q_lens):
        if priv_len == 0:
            continue
        q_start = shared_prefix_len + priv_offset
        q_seg = Q[q_start : q_start + priv_len]                # [priv_len, q_head, dim]
        k_priv = K_private[priv_offset : priv_offset + priv_len]  # [priv_len, kv_head, dim]
        v_priv = V_private[priv_offset : priv_offset + priv_len]
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]                               # [priv_len, dim]
            k = torch.cat([K_shared[:, h_kv, :], k_priv[:, h_kv, :]], dim=0)  # [sp_len+priv_len, dim]
            v = torch.cat([V_shared[:, h_kv, :], v_priv[:, h_kv, :]], dim=0)
            scores = torch.matmul(q, k.T) * sm_scale           # [priv_len, sp_len+priv_len]
            if causal_mask:
                total_kv = shared_prefix_len + priv_len
                q_pos = torch.arange(shared_prefix_len, shared_prefix_len + priv_len).unsqueeze(1)
                kv_pos = torch.arange(total_kv).unsqueeze(0)
                mask = kv_pos > q_pos                          # [priv_len, total_kv]
                scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            O[q_start : q_start + priv_len, h_q, :] = torch.matmul(attn, v)
        priv_offset += priv_len

    return O.to(dtype)
```

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**，供 Stage 2 快速精度收敛。L1（功能扩展）/ L2（异常输入）/ Boundary（特殊值）的完整分层套件由 `tilelang-op-test-design` 生成。

| 用例名 | 级别 | 配置 | dtype | 说明 |
|--------|------|------|-------|------|
| l0_business | L0 | batch=10, q_head=14, kv_head=2, head_dim=64, shared_prefix_len=24, private_q_lens=[150]*10, block_M=128, block_N=64, causal_mask=False | float16 | 业务典型场景起步（avg seq=150, prefix=24）。非整除：shared 24%64=24 尾块、private 150%128=22 尾块。total_q=24+1500=1524, total_tasks=ceildiv(24,128)*14+ceildiv(150,128)*10*14=14+280=294 |
| l0_p99 | L0 | batch=10, q_head=14, kv_head=2, head_dim=64, shared_prefix_len=24, private_q_lens=[218]*10, block_M=128, block_N=64, causal_mask=False | float16 | 业务 p99 场景（seq=218, prefix=24）。非整除尾块。total_q=24+2180=2204, total_tasks=14+280=294, 每 private task KV 迭代=ceildiv(24,64)+ceildiv(218,64)=1+4=5 |
| l0_causal | L0 | batch=2, q_head=4, kv_head=2, head_dim=64, shared_prefix_len=64, private_q_lens=[128, 128], block_M=128, block_N=64, causal_mask=True | float16 | causal mask 开启验证，block 整除简化（先验证 mask 逻辑正确性） |
| l0_bf16 | L0 | 同 l0_business 但 dtype=bfloat16 | bfloat16 | bfloat16 精度验证 |

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。阈值**仅按 dtype**，L0/L1/Boundary 套用精度比对；L2 为非法输入负向测试，不比精度。完整定义见 `tilelang-op-test-design/references/precision-standard.md`。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |

> 本算子不支持 NaN/Inf 输入（FlashAttention 输入为有限浮点值）。若后续需支持，须按位置契约补充特殊值混合输入用例。

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_p99 | batch=10, q_head=14, kv_head=2, head_dim=64, shared_prefix_len=32, private_q_lens=[218]*10, block_M=128, block_N=64, causal_mask=False | float16 | 单一 TND packed 路径 | 120 秒 | 业务 p99 最大序列长度，total_q=32+2180=2212, total_tasks=ceildiv(32,128)*14 + ceildiv(218,128)*10*14 = 14+280=294, 每 private task KV 迭代=ceildiv(32,64)+ceildiv(218,64)=1+4=5，最大 DMA/任务数 |

测试数据、随机数和 golden 物理重排均在 CPU 完成；运行阶段只做 H2D → TileLang kernel → D2H，避免测试 harness 引入 aclnn 依赖。

---

## 10. 风险点与注意事项

### 10.1 已知约束

| 约束 | 影响 | 缓解方案 |
|------|------|----------|
| `total_q / total_private_kv / total_q_blocks` 为编译期常量 | 不同输入规模需重新编译 | tilelang cache 后无额外开销；后续可改 `T.symbolic` + 固定 core_num + 动态任务分配 |
| `shared_prefix_len` 为编译期常量 | 不同 shared_prefix_len 需重新编译 | 同上；或改为 `max_shared_prefix_len` 上界 + 运行时 scalar + 条件判断 |
| UB 简单累加 213KB > 192KB | 依赖 MEMORY_PLANNING pass 复用 | 备选 block_M=64 → 114KB；L0 先用 block_M=128 验证 |
| `for h_i in range(block_M)` 逐行 softmax | block_M=128 时 128 次串行迭代 | Developer pass 自动优化；后续 Expert 版用 `T.tile.broadcast` 批量处理 |
| shared/private 两段循环不用 `T.Pipelined` | 无法利用核内流水线 | Developer pass 自动插入 CV 流水线；后续 Expert 版用 `T.mma` + 双缓冲 |
| causal mask 的 per-row `T.tile.fill` 动态切片 | 运行时动态列偏移 | 参考 mtgr 已验证可行；L1 启用 |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_M=128 + pass 未充分复用 | 编译失败 / segfault | 降级 block_M=64 |
| block_N < 32 | GEMM2 的 K 维 = block_N < 32 | 分形限制报错 | block_N ≥ 32 |
| shared/private 两段间重置 online softmax 状态 | 在 private 段循环前 `T.tile.fill(acc_o, 0.0)` | 累积断裂，结果错误 | 两段共享 m_i/sumexp/acc_o，不可重置 |
| host 侧 `torch.cat([K_shared, K_private])` 拼接 KV | host 侧 aclnn 调用 | §3.2 违规，cann-bench 环境失败 | kernel 内两段循环分支 |
| host 侧 `.contiguous()` / `.reshape().contiguous()` | 对非 contiguous Q 做 reshape | §3.2 违规 | Q 是 TND packed contiguous，直接使用 |
| `T.ceildiv(private_kv_len, block_N)` 作为循环边界 | private_kv_len 是 tensor 值 | 动态循环边界不支持 | 用 `max_private_iters` 编译期上界 + `if` 条件 |
| shared_prefix task 输出重复写入 | 多个 shared_prefix task 写同一 Output 区域 | 写入竞态 | shared_prefix tasks 按 tile_id 去重（每个 tile 只一个 task） |

### 10.3 特殊场景处理

| 场景 | 处理方式 |
|------|----------|
| `shared_prefix_len = 0`（无共享前缀） | shared 段循环 `max_shared_iters=0` 不执行，退化为标准 varlen GQA FA |
| `private_kv_len = 0`（某 request 只有 shared prefix） | 不为该 request 生成 private task（metadata 中无对应行），自然跳过 |
| `q_head = kv_head`（MHA，group_size=1） | `h_kv = h_q // 1 = h_q`，退化为标准 MHA FA |
| `batch = 1`（单 request） | 只有 shared_prefix tasks + 1 组 private tasks，退化为单序列 FA |
| 非整除（shared_prefix_len=24, private_kv_len=150） | 尾块用 valid extent + mask（L1 启用） |
| causal_mask=True | per-row mask 生成 + `T.tile.fill(mask[row, col_start:col_end], NEG_INF)`（L1 启用） |

### 10.4 性能收益预估

以业务典型场景（batch=10, shared_prefix_len=24, private_kv_len_avg=150, head_dim=64, q_head=14, kv_head=2, dtype=float16）估算：

| 指标 | 原生 FA（每 request 独立补全序列） | 本算子（shared prefix 复用） | 节省 |
|------|-----------------------------------|---------------------------|------|
| shared K/V 内存 | 10 × 24 × 2 × 64 × 2B = 61.4 KB（重复 10 份） | 1 × 24 × 2 × 64 × 2B = 6.1 KB（1 份） | 55.3 KB（90%） |
| shared 段 QK^T FLOPs | 10 × (24×24×64) × 2 = 737K | 1 × (24×24×64) × 2 = 74K | 663K（90%） |
| shared 段 PV FLOPs | 10 × (24×24×64) × 2 = 737K | 1 × (24×24×64) × 2 = 74K | 663K（90%） |
| shared 段总 FLOPs 占比 | 1.5M / 33M ≈ 4.5% | — | 总 FLOPs 节省 ≈ 4.5% |

> **收益评估**：由于业务场景 shared_prefix_len（20~25）占总序列（~174）比例小（~14%），总 FLOPs 节省约 4.5%，内存节省约 90%（shared K/V 从 10 份降为 1 份）。若 shared_prefix 更长（如 system prompt 1000+ tokens），收益按比例放大。核心价值在**内存节省**和**消除冗余计算**，而非绝对吞吐量提升。

---

## 11. 交付清单

### 11.1 目录结构

```
examples/tnd_shared_prefix_fa/
├── tnd_shared_prefix_fa.py    # 纯 kernel（@tilelang.jit，可 import，无 golden/测试/__main__）
├── test_tnd_shared_prefix_fa.py  # from tnd_shared_prefix_fa import kernel + golden + 分层测试 + main
├── proto.yaml                 # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── design.md                  # 本设计文档
└── README.md                  # 使用说明（可选）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | 已完成 | 设计文档 |
| `proto.yaml` | 已完成 | 算子接口规格（dtype 全集取自 §9.3 精度表、attr 取自 §4/§1，机械派生），覆盖门禁 `coverage_check.py --proto` 用 |
| `tnd_shared_prefix_fa.py` | 待实现 | 纯 kernel（@tilelang.jit） |
| `test_tnd_shared_prefix_fa.py` | 待实现 | golden + 分层测试 + main（`from tnd_shared_prefix_fa import tnd_shared_prefix_fa`） |

### 11.3 命名规范

- 目录名: `tnd_shared_prefix_fa`（snake_case）
- kernel 文件: `tnd_shared_prefix_fa.py`
- 测试文件: `test_tnd_shared_prefix_fa.py`（顶部 `from tnd_shared_prefix_fa import tnd_shared_prefix_fa`）

### 11.4 实现顺序

1. ✅ 设计文档（design.md）+ proto.yaml（算子接口规格）+ L0 门槛测试计划（本文件 §9.2）
2. ⬜ kernel 实现（`tnd_shared_prefix_fa.py`，纯 @tilelang.jit）
3. ⬜ 测试文件（`test_tnd_shared_prefix_fa.py`）：import kernel + Golden 函数 + L0 用例 + main
4. ⬜ L0 门槛测试通过（精度收敛）
5. ⬜ 扩展分层套件（L1 功能 / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表** + **§4/§1** 的 attr/shape 机械派生，是覆盖门禁 `coverage_check.py --proto` 的权威 dtype/attr 来源。

```yaml
operator:
  name: TndSharedPrefixFA
  category: FlashAttention
  formula: |
    O = softmax(Q @ [K_shared; K_private]^T / sqrt(d)) @ [V_shared; V_private]
    with GQA: h_kv = h_q // (q_head / kv_head)
  attrs:
    - name: q_head
      type: Int
      default: null
      required: true
    - name: kv_head
      type: Int
      default: null
      required: true
    - name: head_dim
      type: Int
      default: 64
      required: true
    - name: shared_prefix_len
      type: Int
      default: null
      required: true
    - name: max_private_kv_len
      type: Int
      default: null
      required: true
    - name: total_q
      type: Int
      default: null
      required: true
    - name: total_private_kv
      type: Int
      default: null
      required: true
    - name: total_q_blocks
      type: Int
      default: null
      required: true
    - name: block_M
      type: Int
      default: 128
      required: false
    - name: block_N
      type: Int
      default: 64
      required: false
    - name: causal_mask
      type: Bool
      default: false
      required: false
    - name: sm_scale
      type: Float
      default: null
      required: false
  inputs:
    - name: Q
      dtype: [float16, bfloat16]
    - name: K_shared
      dtype: [float16, bfloat16]
    - name: V_shared
      dtype: [float16, bfloat16]
    - name: K_private
      dtype: [float16, bfloat16]
    - name: V_private
      dtype: [float16, bfloat16]
    - name: block_metadata
      dtype: [int32]
  outputs:
    - name: Output
      dtype: [float16, bfloat16]
  schema: |
    tnd_shared_prefix_fa(
      Tensor Q, Tensor K_shared, Tensor V_shared,
      Tensor K_private, Tensor V_private, Tensor block_metadata,
      int q_head, int kv_head, int head_dim,
      int shared_prefix_len, int max_private_kv_len,
      int total_q, int total_private_kv, int total_q_blocks,
      int block_M=128, int block_N=64,
      bool causal_mask=false, float sm_scale=null
    ) -> Tensor Output
```

> **一致性约束**：`inputs[].dtype` 与 §9.3 精度表的 dtype 行一致（float16, bfloat16 全集）；`attrs[].name` 覆盖所有影响计算路径的属性（`q_head/kv_head/head_dim/shared_prefix_len/max_private_kv_len/total_q/total_private_kv/total_q_blocks/block_M/block_N/causal_mask/sm_scale`）。
