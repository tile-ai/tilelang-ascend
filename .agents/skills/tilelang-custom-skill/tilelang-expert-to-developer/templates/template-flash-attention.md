# 模板 D：Flash Attention（Developer + 混合模式）

**适用于：** 标准 FlashAttention, Sparse Attention, 各类 Attention 变体

## 核心算法

```
O = softmax(Q @ K^T / sqrt(d)) @ V
```

使用 **Online Softmax** 分块计算，避免一次性 materialization 全部 attention 矩阵。

## pass_configs

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

## 完整模板

```python
import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(0)
tilelang.disable_cache()

B, S, H, D = 1, 128, 1, 512

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(heads, dim):
    block_M, block_N = 32, 64
    batch = B
    seq_len = S
    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / dim) ** 0.5
    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_M * heads * batch

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        # workspace 用于 Cube→Vector 中转
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),  # S 矩阵
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),        # P 矩阵（半精度）
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),      # O 矩阵
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            # =====================
            # Block 坐标计算
            # =====================
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            # =====================
            # Buffer 分配
            # =====================
            q_l1 = T.alloc_shared([block_M, dim], dtype)
            k_l1 = T.alloc_shared([block_N, dim], dtype)
            v_l1 = T.alloc_shared([block_N, dim], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, dim], accum_dtype)

            # Vector 侧 buffer（每个 vid 处理 block_M//2 行）
            acc_o = T.alloc_shared([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_shared([block_M // 2], accum_dtype)
            m_i = T.alloc_shared([block_M // 2], accum_dtype)
            acc_s_ub = T.alloc_shared([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_shared([block_M // 2, block_N], accum_dtype)
            tmp_ub = T.alloc_shared(
                [3 * DataType(accum_dtype).bits // 8 * block_M // 2 * block_N], "uint8")
            sumexp_i_ub = T.alloc_shared([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_shared([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_shared([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_shared([block_M // 2, dim], dtype)

            # =====================
            # 初始化（混合：T.tile.fill）
            # =====================
            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -2**30)
            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)

            # =====================
            # 主循环：Online Softmax + Attention
            # =====================
            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
                # --- Step 1: S = Q @ K^T ---
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, workspace_1[cid, :, :])

                # --- Step 2: Online Softmax ---
                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.copy(
                    workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                    acc_s_ub_)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)

                # scale
                for i, j in T.Parallel(block_M // 2, block_N):
                    acc_s_ub[i, j] = acc_s_ub[i, j] * sm_scale

                # update max
                T.reduce_max(acc_s_ub, m_i, tmp_ub, dim=-1)
                for i in T.Parallel(block_M // 2):
                    m_i[i] = T.max(m_i[i], m_i_prev[i])
                    m_i_prev[i] = T.exp(m_i_prev[i] - m_i[i])

                # softmax numerator
                for h_i, j in T.Parallel(block_M // 2, block_N):
                    acc_s_ub[h_i, j] = T.exp(acc_s_ub[h_i, j] - m_i[h_i])

                # update sumexp
                T.reduce_sum(acc_s_ub, sumexp_i_ub, tmp_ub, dim=-1)
                for i in T.Parallel(block_M // 2):
                    sumexp[i] = sumexp[i] * m_i_prev[i] + sumexp_i_ub[i]

                # --- Step 3: P @ V ---
                T.copy(acc_s_ub, acc_s_half)
                T.copy(
                    acc_s_half,
                    workspace_2[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :])
                T.copy(workspace_2[cid, :, :], acc_s_l1)
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], v_l1)
                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, workspace_3[cid, :, :])

                # --- Step 4: Update acc_o ---
                for h_i, j in T.Parallel(block_M // 2, dim):
                    acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]
                T.copy(
                    workspace_3[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
                    acc_o_ub)
                T.tile.add(acc_o, acc_o, acc_o_ub)

            # =====================
            # 最终归一化
            # =====================
            for h_i, j in T.Parallel(block_M // 2, dim):
                acc_o[h_i, j] = acc_o[h_i, j] / sumexp[h_i]

            T.copy(acc_o, acc_o_half)
            T.copy(
                acc_o_half,
                Output[bz, by,
                       bx * block_M + vid * block_M // 2:bx * block_M + vid * block_M // 2 + block_M // 2, :])

    return main


# 实例化 & 测试
func = flash_attention_fwd(heads=H, dim=D)

q = torch.randn((B, H, S, D), dtype=torch.float16)
k = torch.randn((B, H, S, D), dtype=torch.float16)
v = torch.randn((B, H, S, D), dtype=torch.float16)

torch.npu.synchronize()
output = func(q, k, v)

def ref_flash_attn(q, k, v):
    q, k, v = q.float(), k.float(), v.float()
    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)

ref_output = ref_flash_attn(q, k, v)
torch.npu.synchronize()
torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
print("Test Passed!")
```

## 关键设计要点

### workspace 设计

Flash Attention 需要 **3 个 workspace**：
1. `workspace_1`：存放 S = Q@K^T 的结果（`accum_dtype`）
2. `workspace_2`：存放 softmax(S) 的半精度结果（`dtype`）
3. `workspace_3`：存放 P@V 的结果（`accum_dtype`）

每个 workspace 的第一维为 `block_num`，确保每个核有独立空间。

### Online Softmax 流程

```
1. m_new = max(m_old, rowmax(S_j * scale))
2. correction = exp(m_old - m_new)
3. P_j = exp(S_j * scale - m_new)
4. sumexp = sumexp * correction + rowsum(P_j)
5. O = O * correction + P_j @ V_j
```

最终：`O = O / sumexp`

### vid 分工

每个 Vector 线程处理 `block_M // 2` 行：
- `vid=0` 处理行 `[0, block_M//2)`
- `vid=1` 处理行 `[block_M//2, block_M)`

通过 `vid * block_M // 2` 计算偏移。

### 参数建议

- `block_M`：32（小 dim）或 128（大 dim）
- `block_N`：64 或 128
- `dim`：128, 256, 512 等
- `num_stages=2`：Pipelined 双级流水
