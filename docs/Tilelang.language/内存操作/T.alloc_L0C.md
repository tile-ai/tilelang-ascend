# tilelang.language.alloc_L0C

## 1. 概述

简介: `tilelang.language.alloc_L0C` 用于在 AIC 核的 `L0C` buffer上申请内存

```python
T.alloc_L0C(shape, dtype)
```

## 2. 规格

### 2.1 参数说明

| 参数名  | 类型         | 描述     |
| ------- | ------------ | -------- |
| `shape` | `list/tuple` | 形状     |
| `dtype` | `str`        | 数据类型 |

### 2.2 shape支持

支持 1~5 维tensor

### 2.3 DataType支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :----------- | :--: | :---: | :---: | :---: | :----: | :----: | :----: | :---: | :--: | :--: | :--: | :--: | :--: |
| Ascend A2/A3 |  √   |   √   |   √   |   ×   |   ×    |   ×    |   ×    |   √   |  √   |  √   |  ×   |  √   |  √   |

### 2.4 使用方法

```python
@tilelang.jit(target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num*n_num, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bx = cid // n_num * block_M
                by = cid % n_num * block_N
                A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                for i in T.serial(T.ceildiv(K, K_L1)):
                    T.npuir_load_nd2nz(A[bx, i * K_L1], A_BUF, [block_M, K_L1])
                    T.npuir_load_nd2nz(B[i * K_L1, by], B_BUF, [K_L1, block_N])

                    if i == 0:
                        T.npuir_dot(A_BUF, B_BUF, C_BUF, initC=True, 
                            size=[block_M, K_L1, block_N])
                    else:
                        T.npuir_dot(A_BUF, B_BUF, C_BUF, initC=False, 
                            size=[block_M, K_L1, block_N])

                    T.npuir_store_fixpipe(C_BUF, C[bx, by],
                        size=[block_M, block_N], enable_nz2nd=True)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::alloc_L0C**将被下降为`memref::alloc`Op