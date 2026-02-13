# Tilelang-Ascend Vid Reduction & Auto CV Ratio Feature

## 1. Design Goals
由于晟腾架构中存在cube和vector数量比1:1, 1:2的情况，所以TileLang前端需要显式地用vid来做vector上的数据切分，这样带来的问题是前段表达上暴露了架构细节。为此，本特性旨在实现CV自动配比，从而将前端表达中的vid消除，使用户不再感知，达到屏蔽硬件细节，简化编程的目的。

## 2. Usage Guide
非常简单，您只需要通过参数threads=VECNUM来控制您是要启动一个V核还是两个V核即可，其中VEC_NUM取值为1或2，分别对应C:V=1:1, C:V=1:2. 详见例子[matmul_add_developer.py](../../examples/developer_mode/matmul_add_developer.py).
```python
    # 通过threads参数指定CV配比，仅支持1和2.
    with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
        bx = cid // n_num
        by = cid % n_num
        A_L1 = T.alloc_shared((block_M, block_K), dtype)
        B_L1 = T.alloc_shared((block_K, block_N), dtype)

        C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

        d_ub = T.alloc_shared((block_M, block_N), dtype)
        c_ub = T.alloc_shared((block_M, block_N), dtype)

        loop_k = T.ceildiv(K, block_K)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * block_K], A_L1)
            T.copy(B[k * block_K, by * block_N], B_L1)

            if k == 0:
                T.gemm_v0(A_L1, B_L1, C_L0, init=True)
            else:
                T.gemm_v0(A_L1, B_L1, C_L0)

        T.copy(C_L0, workspace[bx * block_M, by * block_N])

        T.copy(workspace[bx * block_M, by * block_N], c_ub)
        T.copy(D[bx * block_M, by * block_N], d_ub)

        T.tile.add(c_ub, c_ub, d_ub)

        T.copy(c_ub, C[bx * block_M, by * block_N])
```

## 3. Important Limitations
目前，该特性基于静态维度规则，适用于GM与完整ub_buffer的数据传输

