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

混合模式典型场景：Developer pass_configs + Expert 计算原语（`T.tile.fill/max/sub/exp/div`）

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
