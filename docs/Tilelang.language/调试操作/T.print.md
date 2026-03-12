# Tilelang.language.print

## 1. OP概述

简介：`tilelang.language.debug_print_var` 和`tilelang.language.debug_print_buffer_value` 分别用于打印特定变量或缓冲区的值：

```
T.print(obj, msg, hex)
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `obj` | `var`,  `buffer`, `bufferLoad`,  `bufferRegion`| 要打印的对象，可以是缓冲区或原始表达式：1. 若obj为var，则直接打印变量值 ；2. 若obj为buffer相关，则在第一个线程上打印其值|
| `msg` | `str` | 可选的提示信息，包含在打印语句中 |
| `hex` | `bool` | 是否以十六进制格式打印，默认False |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | ×    | ×   | ×     | √    | ×     | √    | ×     | √    | √   | √   | ×   | ×        |

#### 2.2.2 Shape支持

结论：在shape方面，print无特殊要求；

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例通过T.print实现了指定缓冲区的打印：

```
import tilelang
import tilelang.language as T
def vec_add_2d(block_M, block_N, dtype="float32"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    @T.prim_func
    def vecAdd2D(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            blockx = cid % T.ceildiv(N, block_N)
            bx = blockx * block_M
            blocky = cid // T.ceildiv(N, block_N)
            by = blocky * block_N
            A_VEC = T.alloc_shared([block_M, block_N], dtype)
            B_VEC = T.alloc_shared([block_M, block_N], dtype)
            C_VEC = T.alloc_shared([block_M, block_N], dtype)
            T.copy(A[bx:bx + block_M, by:by + block_N], A_VEC[:block_M, :block_N])
            T.copy(B[bx:bx + block_M, by:by + block_N], B_VEC[:block_M, :block_N])
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.print(C_VEC[:4,:4])
            T.copy(C_VEC[:block_M, :block_N], C[bx:bx + block_M, by:by + block_N])
    return vecAdd2D
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::printOp**将被降级转换为hivm::DebugOp。
