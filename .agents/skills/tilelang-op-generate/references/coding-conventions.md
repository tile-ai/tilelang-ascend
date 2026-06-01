# 关键编码规范

## 目录

- [1. Buffer 分配](#1-buffer-分配)
- [2. 数据搬运索引](#2-数据搬运索引)
- [3. 同步](#3-同步)
- [4. 广播](#4-广播)
- [5. 测试模板](#5-测试模板)

---

## 1. Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

Developer 模式下：
```python
# Vector 核 buffer（编译器映射到 UB）
packed_ub = T.alloc_shared([block_M // VEC_NUM, block_N], dtype)

# Cube 核 buffer（编译器映射到 L1/L0）
A_L1 = T.alloc_shared([block_M, block_K], dtype)
B_L1 = T.alloc_shared([block_N, block_K], dtype)
C_L0 = T.alloc_fragment([block_M, block_N], accum_dtype)
```

## 2. 数据搬运索引

```python
# 标准索引模式（纯 Vector 算子）
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

**✅ CV 融合场景 — Developer 模式（推荐，默认消除 workspace/vid）**：
```python
# T.Kernel(block_num, threads=2, is_npu=True) as (cid)  —— threads=2，无 vid 轴
for bi_i in range(BI):                       # 整程，无 vid 偏移
    T.copy(KV[..., idx[bi_i], ...], kv_ub)
    T.copy(kv_ub, kv_l1[bi_i, :])            # gather 直连片上 L1，无 workspace
...
T.copy(acc_s_l0c, acc_s_ub_)                 # L0C → shared 直连，无 GM 往返
```
前提链：`threads=2` → 消 vid → 消 workspace；完整映射表见 [mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。

**⚠️ 回退写法 — workspace 索引一致性（Expert/混合或复杂场景）**：
```python
VEC_NUM = 2
block_N_2 = block_N // VEC_NUM

for row in T.serial(block_N_2):
    actual_row = bn * block_N + vid * block_N_2 + row  # 关键索引
    
    # 读数据和写 workspace 都必须用 actual_row
    T.copy(B_packed[actual_row, chunk_offset], packed_ub)  # ✓
    # ... 处理 ...
    T.copy(output_ub, workspace[actual_row, chunk_offset * 2])  # ✓（必须一致）

# Cube 核读取完整 block_N（不涉及 vid）
T.copy(workspace[bn * block_N, k_offset], B_L1)  # 完整 block_N
```

**易错点（仅回退写法）**：workspace 写入时忘记使用 `actual_row`，导致数据错乱。

## 3. 同步

```python
# Expert 模式：手动同步
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Developer 模式 + 自动同步：无需手动 barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

## 4. 广播

```python
# 归约结果 [M, 1] 广播到 [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```

## 5. 测试模板

```python
# golden 对比
ref_output = torch.nn.functional.softmax(input_data, dim=-1)  # 或手写 golden
torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)
```
