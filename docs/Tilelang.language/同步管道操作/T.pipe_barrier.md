# Tilelang.language.pipe_barrier

## 1. OP概述

简介：`tilelang.language.pipe_barrier` 在指定管道上设置屏障，用于同步管道操作

```
T.pipe_barrier(pipe)
```

## 2. OP规格

### 2.1 参数说明

| 参数名  | 类型  | 说明  |
| ------------ | ------------ | ------------ |
| `pipe` | `str` | 管道类型标识符，用于指定需要同步的管道 |

### 2.2 支持规格

#### 2.2.1 DataType支持

不涉及数据类型

#### 2.2.2 Shape支持

不涉及shape

### 2.3 特殊限制说明

- pipe参数必须是有效的字符串标识符
- 该操作用于同步管道级别的操作，确保管道中的操作按预期顺序执行
- 通常与set_flag、wait_flag等同步操作配合使用

### 2.4 使用方法

以下示例展示了在管道操作中使用pipe_barrier进行同步

```
import torch
import torch_npu
import tilelang
import tilelang.language as T

def pipe_barrier_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_shared((M, N), dtype)

            T.copy(src, src_ub)

            T.pipe_barrier("VEC_IN")

            T.vadd(src_ub, src_ub, dst_ub)

            T.pipe_barrier("VEC_OUT")

            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::barrierOp**将被下降为`mlir::hivm::PipeBarrierOp`