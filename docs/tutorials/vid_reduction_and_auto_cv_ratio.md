# Tilelang-Ascend Vid Reduction & Auto CV Ratio Feature

## 1. Design Goals
Because the Ascend architecture sometimes has a 1:1 or 1:2 ratio of cubes to vectors, the TileLang frontend needs to explicitly use vids to partition data on the vectors. This exposes architectural details in the frontend representation. Therefore, this feature aims to achieve automatic CV matching, thereby eliminating vids in the frontend representation, making them imperceptible to the user, thus shielding hardware details and simplifying programming.

## 2. Usage Guide
It's very simple. You only need to use the parameter `threads=VECNUM` to control whether you want to start one or two V-cores. `VEC_NUM` can be 1 or 2, corresponding to C:V=1:1 and C:V=1:2 respectively. When setting the `threads` parameter, the return value will only contain `cid` and not `vid`. See the example for details[matmul_add_developer.py](../../examples/developer_mode/matmul_add_developer.py).
```python
    # The CV ratio is specified via the `threads` parameter; only 1 and 2 are supported.
    with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
        bx = cid // n_num
        by = cid % n_num
        A_L1 = T.alloc_shared((block_M, block_K), dtype)
        B_L1 = T.alloc_shared((block_K, block_N), dtype)

        C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
        # When allocating memory for the ub, block_M does not need to be divided by 2 to allocate to two vid cores; the compilation pass will perform the division by 2 operation.
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
        # When copying gm to ub, the parameters in gm do not need to be split by adding vid * block_M // VEC_NUM; the compilation pass will handle the relevant operations.
        T.copy(workspace[bx * block_M, by * block_N], c_ub)
        T.copy(D[bx * block_M, by * block_N], d_ub)

        T.tile.add(c_ub, c_ub, d_ub)
        # When copying ub to gm, the parameters in gm do not need to be split by adding vid * block_M // VEC_NUM; the compilation pass will handle the relevant operations.
        T.copy(c_ub, C[bx * block_M, by * block_N])
```

## 3. Important Limitations
Currently, this feature is based on static dimensional rules and is suitable for data transfer between the GM and the complete ub_buffer. Elimination involves data chunking; if multiple VID cores have different responsibilities, such as index array movement, then this elimination mode is not suitable.

automatic reduction range：
- UB allocation processing: Only UBs are processed.
- UB transfer processing: Suitable for copying GM to UB and UB to GM, it performs parameter offset processing on the starting position of parameters in GM, supports GM slicing, multi-dimensional, etc. UB needs to be a complete buffer.

