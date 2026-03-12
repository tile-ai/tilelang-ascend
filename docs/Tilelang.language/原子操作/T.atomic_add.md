# Tilelang.language.atomic_add

## 1. OP概述

简介：`tilelang.language.atomic_add` 为原子加法操作，在指定的内存位置执行原子加法：

```
T.atomic_add(dst, src)
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor`,  `scalar`| 输入tensor或scalar |
| `dst` | `tensor` | 输出tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | ×    | ×   | ×     | ×    | ×     | ×    | ×     | ×    | √   | √   | ×   | ×        |

#### 2.2.2 Shape支持

结论：atomic_add对形状无特殊限制；

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了一个二维张量的原子加法计算：

```
def vec_atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def vecAtomicAdd2D(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            bx = blockx * block_M
            blocky = cid % n_num
            by = blocky * block_N
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            t0 = shape_M - bx
            tile_size_M = T.min(block_M, t0)        
            t0 = shape_N - by
            tile_size_N = T.min(block_N, t0)   
            T.copy(A[bx, by], A_VEC, [tile_size_M, tile_size_N]) 
            T.atomic_add(B[bx, by], A_VEC, [tile_size_M, tile_size_N])           
    return vecAtomicAdd2D
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::atomic_addOp**将被降级转换为hivm::StoreOp
