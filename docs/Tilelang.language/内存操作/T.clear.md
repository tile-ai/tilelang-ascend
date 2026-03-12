# Tilelang.language.clear

## 1. OP概述

简介：`tilelang.language.clear` 用于对给定内存位置的值进行全零初始化：

```
T.clear(buffer)
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `buffer` | `buffer` | 	给定的内存空间 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | ×    | ×   | ×     | ×    | ×     | ×    | ×     | ×    | √   | √   | ×   | ×        |

#### 2.2.2 Shape支持

结论：支持 1\~5 维tensor

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例通过T.clear实现了将指定内存位置的值进行全零初始化：

```
def vec_clear(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 20
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.clear(A_VEC)
                    T.copy(A_VEC, A[bx, by])
    return main
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::clearOp**将被降级转换为hivm::VBrcOp。
