# Expert → Developer 完整转换示例

本文档提供多个完整的转换示例，展示如何将 Expert 模式算子转换为 Developer 模式。

---

## 示例 1：GEMM（矩阵乘法）

### Expert 模式

```python
@tilelang.jit(out_idx=[-1])    # 无 pass_configs
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

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)            # ← 显式 L1
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)            # ← 显式 L1
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)  # ← 显式 L0C

            with T.Scope("C"):                                     # ← 显式 Scope
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()                                # ← 手动 barrier
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()                                # ← 手动 barrier
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

### Developer 模式

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
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

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)        # ← alloc_shared
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)        # ← alloc_shared
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype) # ← alloc_fragment

            # 无 T.Scope，无 barrier
            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**转换要点**：
1. 添加完整 `pass_configs`
2. `alloc_L1` → `alloc_shared`，`alloc_L0C` → `alloc_fragment`
3. 删除 `with T.Scope("C")`
4. 删除所有 `T.barrier_all()`

---

## 示例 2：ElementWise Add（逐元素加法）

### Expert 模式

```python
@tilelang.jit(out_idx=[-1])    # 无 pass_configs
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)   # ← alloc_ub
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):                                         # ← 显式 Scope
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()                                        # ← 手动 barrier
                T.tile.add(c_ub, a_ub, b_ub)                          # ← T.tile.add
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### Developer 模式

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            # ← T.Parallel + 符号运算
            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**转换要点**：
1. 添加 `pass_configs`
2. `alloc_ub` → `alloc_shared`
3. `T.tile.add` → `T.Parallel` + `+`
4. 删除 `with T.Scope("V")`
5. 删除所有 `T.barrier_all()`

---

## 示例 3：MatMul + Add（Cube + Vector 混合）

### Expert 模式

```python
@tilelang.jit(out_idx=[-2])    # 无 pass_configs, 无 workspace_idx
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype), D: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("C"):                                          # ← 显式 Cube Scope
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])         # ← 直接写到 C（GM中转）
                T.set_cross_flag("FIX", 0)                            # ← 手动核间同步

            with T.Scope("V"):                                          # ← 显式 Vector Scope
                T.wait_cross_flag(0)                                    # ← 等待 Cube 完成
                T.copy(C[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)
                T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N], d_ub)
                T.barrier_all()
                T.tile.add(c_ub, c_ub, d_ub)
                T.barrier_all()
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### Developer 模式

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

@tilelang.jit(out_idx=[2], workspace_idx=4, pass_configs=pass_configs)
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype), D: T.Tensor((M, N), dtype),
             workspace: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
            d_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # Cube 部分（自动分离）
            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.copy(C_L0, workspace[bx * block_M, by * block_N])

            # Vector 部分（自动分离 + 自动核间同步）
            T.copy(workspace[bx * block_M + vid * block_M // VEC_NUM, by * block_N], c_ub)
            T.copy(D[bx * block_M + vid * block_M // VEC_NUM, by * block_N], d_ub)
            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = c_ub[i, j] + d_ub[i, j]
            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**转换要点**：
1. 添加完整 `pass_configs`（需要 AUTO_CV_COMBINE 和 AUTO_CV_SYNC）
2. 内存分配：`alloc_L1` / `alloc_ub` → `alloc_shared`，`alloc_L0C` → `alloc_fragment`
3. 删除 `T.Scope("C")` 和 `T.Scope("V")`
4. 删除 `T.set_cross_flag` / `T.wait_cross_flag`
5. 删除所有 `T.barrier_all()`
6. `T.tile.add` → `T.Parallel` + `+`
7. 使用 workspace tensor 进行 Cube→Vector 数据中转

---

## 示例 4：行广播转换

### Expert 模式

```python
# 需要手动逐行处理
for h_i in range(sub_block_M):
    T.tile.sub(a[h_i, :], a[h_i, :], tile_max[h_i])

for h_i in range(sub_block_M):
    T.tile.div(a[h_i, :], a[h_i, :], prev_sum[h_i])
```

### Developer 模式

```python
# 2D buffer - 1D buffer 自动广播
for h_i, j in T.Parallel(sub_block_M, block_N):
    a[h_i, j] = a[h_i, j] - tile_max[h_i]

for h_i, j in T.Parallel(sub_block_M, block_N):
    a[h_i, j] = a[h_i, j] / prev_sum[h_i]
```

---

## 转换检查清单

### Expert → Developer

- [ ] 添加完整 `pass_configs`（4 个开关全部开启）
- [ ] `T.alloc_L1` / `T.alloc_ub` → `T.alloc_shared`
- [ ] `T.alloc_L0C` → `T.alloc_fragment`
- [ ] 删除所有 `T.Scope("C")` / `T.Scope("V")`
- [ ] 删除所有 `T.barrier_all()`
- [ ] 删除所有 `T.set_cross_flag` / `T.wait_cross_flag`
- [ ] 删除所有 `T.set_flag` / `T.wait_flag`
- [ ] `T.tile.add/sub/mul/div` → `T.Parallel` + 符号运算
- [ ] `T.tile.exp/log/abs/sqrt` → `T.Parallel` + `T.exp/log/abs/sqrt`
- [ ] 逐行循环 → `T.Parallel` 自动广播
- [ ] 删除 `T.annotate_address`