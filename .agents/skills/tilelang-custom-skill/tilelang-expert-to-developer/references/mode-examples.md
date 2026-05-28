# Developer vs Expert 模式代码对比

---

## 1. GEMM — Developer 模式

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # Developer 模式：alloc_shared / alloc_fragment
            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                # Developer 模式：无需 T.barrier_all()，编译器自动插入
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**特点**：
- 无 `T.Scope`、无 `T.barrier_all`、无 `T.set_flag`
- 使用 `alloc_shared` / `alloc_fragment`
- 全靠 pass_configs 自动处理同步和内存

---

## 2. GEMM — Expert 模式

```python
import tilelang
import tilelang.language as T

# Expert 模式：无 pass_configs（或全 False）
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # Expert 模式：显式指定 L1/L0C
            A_L1 = T.alloc_L1([block_M, block_K], dtype)
            B_L1 = T.alloc_L1([block_K, block_N], dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                # Expert 模式：手动插入 barrier
                T.barrier_all()
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                T.barrier_all()

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**特点**：
- 手动 `T.barrier_all()` 同步
- 使用 `alloc_L1` / `alloc_L0C` 显式指定存储层级
- 无 pass_configs

---

## 3. Flash Attention — Expert 模式 pass_configs

Expert 模式极致性能场景，**全部关闭**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

## 4. Flash Attention — Developer 核间流水线 pass_configs

核间流水线场景，**全部开启**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

---

## 5. 混合模式 — Softmax

混合模式典型场景：Developer pass_configs + Ascend 专属 `T.tile` 原语（`T.tile.fill/max/sub/exp/div`）

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# kernel 内部混用 Developer 和 Expert API
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    # Expert API：T.tile.fill, T.tile.max, T.tile.sub, T.tile.exp 等
    T.tile.fill(acc_ub, 0.0)
    T.reduce_max(scores_ub, row_max_ub, dim=-1)
    T.tile.sub(scores_ub, scores_ub, row_max_ub)
    T.tile.exp(scores_ub, scores_ub)
    T.reduce_sum(scores_ub, row_sum_ub, dim=-1)
    T.tile.div(scores_ub, scores_ub, row_sum_ub)
    # 使用 Developer 的 pass_configs 自动处理同步
```

**关键点**：`T.tile.xxx` 和 `T.reduce_*` 可以在 Developer pass_configs 下正常工作，无需手写同步。

---

## 6. CV 融合 — Developer 模式（W4A8 GEMM）

CV 融合典型场景：Vector 核解量化 + Cube 核 GEMM。

```python
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2
BLOCK_K_HALF = 128

@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def w4a8_gemm_cv(M, N, K):
    K_half = K // 2
    block_M = 64
    block_N = 16  # 满足 L0B/L0C 分形限制（必须 ≥ 16）
    block_N_2 = block_N // VEC_NUM  # 每个 V 核处理 8 行
    block_K_chunk = BLOCK_K_HALF * 2

    k_num = T.ceildiv(K_half, BLOCK_K_HALF)
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "int8"),
        B_packed: T.Tensor((N, K_half), "uint8"),
        workspace: T.Tensor((N, K), "int8"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            # ===== Vector 核部分：W4 解量化 =====
            # 使用 alloc_shared，编译器自动映射到 UB
            packed_ub = T.alloc_shared((BLOCK_K_HALF,), "uint8")
            output_ub = T.alloc_shared((BLOCK_K_HALF * 2,), "int8")
            # ... 其他临时 buffer ...

            # 每个 V 核处理 block_N_2 行
            for row in T.serial(block_N_2):
                actual_row = bn * block_N + vid * block_N_2 + row  # 关键索引

                for k_chunk in T.serial(k_num):
                    chunk_offset = k_chunk * BLOCK_K_HALF
                    
                    # 读数据（用 actual_row）
                    T.copy(B_packed[actual_row, chunk_offset], packed_ub)
                    
                    # ... W4 解量化逻辑（T.tile.bitwise_and/rshift/cast/add）...
                    
                    # 写 workspace（必须用 actual_row！）
                    T.copy(output_ub, workspace[actual_row, chunk_offset * 2])

            # ===== Cube 核部分：GEMM =====
            # 使用 alloc_shared/fragment，编译器自动映射到 L1/L0
            A_L1 = T.alloc_shared((block_M, block_K_chunk), "int8")
            B_L1 = T.alloc_shared((block_N, block_K_chunk), "int8")
            C_L0 = T.alloc_fragment((block_M, block_N), "int32")

            for k_chunk in T.serial(k_num):
                k_offset = k_chunk * BLOCK_K_HALF * 2
                
                # Cube 核读取完整 block_N（不涉及 vid）
                T.copy(A[bm * block_M, k_offset], A_L1)
                T.copy(workspace[bn * block_N, k_offset], B_L1)  # 完整 16 行
                
                # init=(k_chunk == 0)：第一次调用清零 C_L0
                T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k_chunk == 0))

            T.copy(C_L0, C[bm * block_M, bn * block_N])

    return main
```

**特点**：
- **无 `T.Scope`、无手动同步**：AUTO_CV_COMBINE 和 AUTO_CV_SYNC 自动处理
- **V 核并行化**：`vid` 分配任务，每个 V 核处理 8 行
- **workspace 索引一致性**：读写都使用 `actual_row`
- **Cube 核读取完整 block_N**：GEMM 不涉及 vid
- **满足分形限制**：`block_N = 16`（≥ L0B/L0C 最小要求）

**关键 pass_configs**：
- `AUTO_CV_COMBINE`：编译器识别 Vector 解量化 + Cube GEMM 并自动分离
- `AUTO_CV_SYNC`：编译器自动在 Vector 写完 workspace 后通知 Cube 读取

### 6.1 CV 融合算子特征

**CV 融合算子** = Vector 核预处理/后处理 + Cube 核 GEMM

典型场景：
- **W4A8 GEMM**：Vector 核解量化（W4 → int8），Cube 核做 GEMM
- **Flash Attention**：Vector 核 Softmax，Cube 核做两次 GEMM
- **量化 GEMM**：Vector 核反量化/量化，Cube 核做 GEMM

### 6.2 Developer 模式下 CV 融合的关键点

**必须开启 4 个 pass_configs**：
- `AUTO_CV_COMBINE`：编译器自动识别 Cube/Vector 操作并分离到不同核
- `AUTO_CV_SYNC`：编译器自动在 Cube/Vector 写入 workspace 后插入核间同步
- **不要手写 `T.Scope("C")` / `T.Scope("V")`**（会与 AUTO_CV_COMBINE 冲突）

### 6.3 V 核并行化（避免算力浪费）

Ascend NPU C:V = 1:2，两个 V 核默认执行相同工作。正确使用 `vid` 可让两个 V 核分担任务。

**易错点**：
- workspace 写入时忘记使用 `actual_row`（导致数据错乱）
- Cube 核读取时使用 vid 切分（Cube 不涉及 vid）

### 6.4 编译器警告解读

Developer 模式下可能出现：
```
Warning: Cube loop times (= X) is not enough to catch up vec loop times (= Y)
```

**解读**：
- Vector 循环次数 = `block_N_2 × k_num`
- Cube 循环次数 = `k_num`
- 此警告可忽略，AUTO_CV_SYNC 会确保同步正确
