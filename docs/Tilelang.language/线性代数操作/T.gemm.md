# Tilelang.language.gemm

## 1. OP概述

简介：`tilelang.language.gemm` 返回输入tensor的矩阵乘计算结果

```
T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) [Developer mode]
T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) [Expert mode]
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `src1` | `tensor`| 输入tensor, `fp16`  |
| `src2` | `tensor`| 输入tensor, `fp16` |
| `dst` | `tensor` | 输出tensor, `fp32` |
|`size`|`shape`|如果size=[a, b, c], 则 `src1`的shape为[a, b]或[b, a], `src2`的shape为[b, c]或[c, b], `dst`的shape为[a, c]|
| `initC` | `bool` | 是否对dst清零。`initC`=True表示dst=src1@src2; `initC`=False表示dst=src1@src2+dst|
| `a_transpose` | `bool` |是否对src1进行转置 |
| `b_transpose` | `bool` |是否对src2进行转置|

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Ascend | ×  | × |  × |  × | ×  | ×  | ×  | ×  | √  | √ |  ×  | ×  |

#### 2.2.2 Shape支持

结论：`src1` 为2维，`src2`为2维，`dst`为2维|

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了一个形状为(M,K)的tensor和一个形状为(K,N)的tensor矩阵乘

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::gemmOp**将被下降为**hivm.hir.mmadL1**