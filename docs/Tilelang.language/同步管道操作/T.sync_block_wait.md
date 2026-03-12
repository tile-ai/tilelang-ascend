# Tilelang.language.sync_block_wait

## 1. OP概述

简介：`tilelang.language.sync_block_wait`用于Block内部的同步。让当前的执行处于等待状态，直到指定的事件标志位被对应的sync_block_set指令激活

```
T.sync_block_wait(id)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `id` | `int` | 同步标志位ID（Flag ID） |

### 2.2 支持规格

#### 2.2.1 DataType支持

不涉及

#### 2.2.2 Shape支持

不涉及

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下代码示例了sync_block_wait同步指令的使用

```python
def simple_sync_demo(A, Workspace, Output):
    with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):

        with T.Scope("Cube"):
            for i in T.serial(num_blocks):
                with T.rs("PIPE_FIX"):
                    T.npuir_store_fixpipe(l0_result, Workspace[cid * block_m, i * block_n])
                    T.sync_block_set(i)
        
        with T.Scope("Vector"):
            for i in T.serial(num_blocks):
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(i)
                    T.copy(Workspace[cid * block_m, i * block_n], cross_kernel_ub)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::sync_block_waitOp**将被编译为hivm::SyncBlockWaitOp