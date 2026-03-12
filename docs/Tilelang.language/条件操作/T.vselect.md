# Tilelang.language.vselect

## 1. OP概述

简介：`tilelang.language.vselect`根据条件Tensor（Cond）的布尔值，按元素从两个输入Tensor（A或B）中选择值并输出。

```
T.vselect(Cond, A, B, Out)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `Cond` | `tensor` | 条件Tensor |
| `A`  | `tensor` | 第一输入源，当条件为真时，输出此对应位置的值 |
| `B`  | `tensor` | 第二输入源，当条件为假时，输出此对应位置的值 |
| `Out`  | `tensor` | 输出Tensor |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

`Cond`、`A`、`B`和`Out`的Shape一致

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例实现了一个vselect计算

```python
def select_kernel(N, block_M, dtype="float16"):
    grid = (N + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid, is_npu=True) as (bx, _):
            cond_ub = T.alloc_shared((block_M,), "bool")
            acc_A = T.alloc_shared((block_M,), dtype)
            acc_B = T.alloc_shared((block_M,), dtype)
            out_ub = T.alloc_shared((block_M,), dtype)

            for i in T.serial(grid):
                start = i * block_M
                end = T.min(start + block_M, N)
                cur_size = end - start

                T.copy(A[start:end], acc_A[0:cur_size])
                T.copy(B[start:end], acc_B[0:cur_size])

                T.vcmp(acc_A[0:cur_size], acc_B[0:cur_size], cond_ub[0:cur_size], "ge")
                T.vselect(cond_ub[0:cur_size], acc_A[0:cur_size], acc_B[0:cur_size], out_ub[0:cur_size])
                T.copy(out_ub[0:cur_size], Out[start:end])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::selectOp**将被编译为hivm::VSelOp