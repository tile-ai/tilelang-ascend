# Tilelang.language.vmin

## 1. OP概述

简介：`tilelang.language.vmin`对向量元素取最小值操作，该算子对比两个输入源的对应元素，并将较小值写入输出目标。

```
T.vmin(src1, src2, dst)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src1` | `tensor` | 输入tensor1 |
| `src2`                              | `tensor or scalar` | 输入tensor2或标量 |
| `dst`                               | `tensor` | 输出tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | √ | × | √ | × | √ | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：在shape方面：

1. 当src1和src2具有相同shape时，min直接执行elementwise运算；
2. 当src1和src2 shape不一致，且二者有且仅有一个维度的shape不同，且在该维度上，其中一个的size为1时，执行带brc的min，即此时vmin操作会自动广播；
   示例：
   ✅ 合法：
   src1: (M, N, K)  src2: (M, N, K) → dst: (M, N, K)
   src1: (M, 1, K)  src2: (M, N, K) → dst: (M, N, K)
   src1: (1, N, K)  src2: (M, N, K) → dst: (M, N, K)
   ❌ 非法：
   src1: (1, 1, K)  src2: (M, N, K) → dst: (M, N, K)（两个维度不同，违反“仅一个维度不同”）
3. src2可以使用标量进行计算

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了两个形状为(M,N)的输入tensor进行min计算：

```python
def vec_min(M, N, block_M, block_N):
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
                    T.vmin(A_VEC, B_VEC, C_VEC)
                    T.copy(C_VEC, C[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

tilelang::vminOp将被转换为`hivm::VMinOp`
