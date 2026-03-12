# Tilelang.language.Parallel

# 1. OP概述

简介：`tilelang.language.Parallel` 实现并行语义，循环体内要求标量计算。

```
T.Parallel(ub_0, ub_1, ...,  ub_N)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `ub_i` | `scalar` | 第`i`个循环的循环次数 |

### 2.2 支持规格

#### 2.2.1 DataType支持

`T.Parallel` 的参数为scalar，主要类型限制取决于循环体内的标量计算算子，目前已支持标量操作为：

* 指数计算：`T.exp`
* 加减乘除：使用 `+`, `-`, `*`, `/` 即可
* sigmoid: `T.sigmoid`
* 广播：`T.vbrc`
* 比较操作：使用 `==`, `!=`, `<`, `<=`, `>`, `>=` 即可
* 条件分支：`T.if_then_else`，当前不支持直接使用循环变量的条件语句，例如 `T.if_then_else(i>j, xxx, xxx)`

#### 2.2.2 Shape支持

无

### 2.3 特殊限制说明

* `T.Parallel` 内下标不能做变换，即对于一条语句 `C[i]=A[i]+B[i]`，其中的下标 `i` 不能做任何变换，例如 `i+1` 等
* 循环体内的变量必须是alloc申请出的UB上的Buffer，不能是直接传入的GM上的Tensor变量，以 2.4 使用方法中的代码为例，必须使用 `A_shared`, `B_shared`, `C_local`, 而不能直接使用传入的函数参数变量 `A`, `B`, `C`

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的并行加法计算

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def elementwise_add(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def elemAdd(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    return elemAdd
```
