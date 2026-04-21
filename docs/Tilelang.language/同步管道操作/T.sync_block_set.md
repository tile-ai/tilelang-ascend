# Tilelang.language.sync_block_set

## 1. OP概述

简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。

```markup
T.sync_block_set(i)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型  | 说明                     |
| ------ | ----- | ------------------------ |
| `id`   | `int` | 同步标志位 ID（Flag ID） |

### 2.2 支持规格

#### 2.2.1 DataType支持

不涉及

#### 2.2.2 Shape支持

不涉及

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例包含了sync_block_set同步指令的使用

```python
def simple_sync(M, N, block_M, block_N, dtype="float16", inner_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, subid):

            with T.Scope("Cube"):
                bx = (cid // n_num) * block_M
                by = (cid % n_num) * block_N

                A_BUF = T.alloc_L1((block_M, block_N), dtype)
                T.copy(A[bx, by], A_BUF)

                C_BUF = T.alloc_L0C((block_M, block_N), inner_dtype)

                with T.rs("PIPE_FIX"):
                    T.sync_block_wait(1)  
                    T.npuir_store_fixpipe(C_BUF, B[bx, by], [block_M, block_N], enable_nz2nd=True)
                    T.sync_block_set(0)    

                with T.rs("PIPE_FIX"):
                    for i in range(0, FFTS_FLAG_THRESHOLD):
                        T.sync_block_wait(1)

            with T.Scope("Vector"):
                bx = (cid // n_num) * block_M
                by = (cid % n_num) * block_N

                with T.rs("PIPE_MTE2"):
                    for i in range(0, FFTS_FLAG_THRESHOLD):
                        T.sync_block_set(1)   

                C_VEC = T.alloc_ub((block_M, block_N), dtype)

                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)  
                    T.copy(B[bx, by], C_VEC)
                    T.sync_block_set(1)   
    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::sync_block_setOP**将被下降为hivm::SyncBlockSetOp
