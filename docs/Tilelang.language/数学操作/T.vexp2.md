# Tilelang.language.vexp2

## 1. OP概述

简介：`tilelang.language.vexp2`执行逐元素底数为2的指数计算 $2^{src}$

由于底层不支持硬件级别的exp2操作，实际上使用的是 $B = exp(A*ln2)$，其中 tmpBuffer 用来存储 $A*ln2$ 这一中间结果

```
T.vexp2(input, output, tmpBuffer)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `input` | `tensor` | 输入tensor |
| `output`                            | `tensor` | 输出tensor |
| `tmpBuffer`                         | `tensor` | 临时空间tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

input, output, tmpBuffer 三者形状需要一致

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了两个形状为(M,N)的输入tensor进行min计算：

```python
import torch
import torch_npu
import tilelang
import tilelang.language as T
def vec_exp2(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    BLOCK_SIZE = 20
    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num*n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.copy(B[bx, by], B_VEC)
                    T.vexp2(A_VEC, B_VEC, C_VEC)
                    T.copy(C_VEC, C[bx, by])
    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

tilelang::vexp2Op将被下降为`hivm::VMulOp`和`hivm::VExpOp`
