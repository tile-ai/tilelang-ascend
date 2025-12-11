# Tilelang-Ascend Workspace Auto-Allocation Feature

## 1. Design Goals
This feature aims to achieve automated workspace memory management with the following core objectives:
1. **Automated Memory Management**: The framework automatically handles workspace allocation and deallocation internally
2. **Simplified User Interface**: Users don't need to be aware of workspace existence, focusing only on business logic parameters
3. **Maintained Flexibility**: Provides clear declaration mechanisms allowing developers to control workspace usage

## 2. Usage Guide

### 2.1 Operator Development Declaration Method
**When developing operators**, you need to explicitly declare workspace parameter positions using the `workspace_idx` parameter of the `@tilelang.jit` decorator:

```python
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7])  # Declare parameters 4-7 as workspace
def sparse_attention_fwd(...):
    @T.prim_func
    def main(
            # --- Input tensors ---
            Q: T.Tensor(q_shape, dtype),  
            KV: T.Tensor(kv_shape, dtype),  
            Indices: T.Tensor(indices_shape, indices_dtype), 

            # --- Auto-allocated output (index 3 in out_idx) --- 
            Output: T.Tensor(o_shape, dtype),  

            # --- Auto-allocated workspaces (indices 4-8 in workspace_idx) ---
            # These are temporary buffers managed by the runtime
            workspace_1: T.Tensor([block_num, BI, D], dtype),
            workspace_2: T.Tensor([block_num, BI, D_tail], dtype),
            workspace_3: T.Tensor([block_num, H_per_block, BI], accum_dtype),
            workspace_4: T.Tensor([block_num, H_per_block, BI], dtype),
            workspace_5: T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
    # Operator implementation 
    ...
```

**Note**: For now, workspace parameters and their types should be declared in the function definition. The framework handles the memory allocation based on your declared shapes automatically.

### 2.2 Parameter Description
- `out_idx`: Specifies the position of output parameters in the function signature(0-based indexing, negative values count from the end)
- `workspace_idx`: Specifies the list of workspace parameter positions in the function signature(0-based indexing, negative values count from the end)

### 2.3 User Calling Method
**When users call operators**, they only need to pass input tensor, with workspace being fully automatically managed by the framework:

```python
# Users only need to pass input parameters
q = ...  # Query tensor
kv = ...  # Key-Value tensor
indices = ...  # Indices tensor

# Workspace is completely transparent to users
output = sparse_attention_op(q, kv, indices)
```



## 3. Important Limitations

### 3.1 Execution Backend Limitation
**Please note**: Currently, the workspace auto-management feature is only available in the following execution backend:
- **Cython Backend**: Fully supported

Before using this feature, please ensure your execution environment is configured for the Cython backend. Other backends(e.g. ctypes) have not been supported yet.