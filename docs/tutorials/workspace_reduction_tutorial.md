# TileLang-Ascend Workspace Reduction Tutorial

## 1. Design Goals

In Ascend architecture, Cube and Vector cores cannot directly exchange data. When Cube computation results need to be processed by Vector, the traditional approach requires:

1. Cube writes results to Global Memory (GM)
2. Vector reads data from GM
3. Potential synchronization overhead between cores

This traditional approach introduces **Programming complexity**: Users must manually write two-stage copy scripts with workspace management, offset calculation, and must explicitly declare workspace buffers and manage their lifecycle, significantly increasing frontend script writing difficulty and reducing code maintainability

**Workspace Reduction** aims to:

- **Automate workspace allocation**: Compiler automatically creates GM workspace buffers for cross-core data transfer
- **Transform copy statements**: Convert direct `copy_l0c_to_ub` and `copy_ub_to_l1` into two-stage GM-based copies
- **Support multi-core scenarios**: Each AI Core gets independent workspace region via `cid` offset
- **Handle vector core splitting**: When `threads=2`, workspace stores complete data while UB is split by `vid`
- **Simplify programming model**: Users write single copy statement, compiler handles workspace management

## 2. Usage Guide

Workspace Reduction is **automatically enabled** in the compilation pipeline. No manual configuration required.

### 2.1 Basic Example

**Original Code (Expert Mode)**:
```python
@T.prim_func
def kernel(...):
    with T.Kernel(..., threads=2, is_npu=True) as (cid):
        # Cube computation
        T.gemm_v0(q_l1, k_l1, acc_s_l0c, ...)
        
        # Direct copy: L0C → UB (requires cross-core transfer)
        T.copy(acc_s_l0c, acc_s_ub)  # ← Workspace Reduction triggered
        
        # Vector computation
        T.tile.exp(acc_s_ub, acc_s_ub)
```

### 2.2 Practical Example: FlashAttention

See complete example: [flash_attn.py](../../examples/developer_mode/flash_attn_bshd_developer.py)

```python
@T.prim_func
def flash_attn(...):
    with T.Kernel(..., threads=2, is_npu=True) as (cid):
        # Attention score computation (Cube)
        T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
        
        # L0C → UB transfer (triggers Workspace Reduction)
        T.copy(acc_s_l0c, acc_s_ub)
        
        # Softmax computation (Vector)
        T.tile.exp(acc_s_ub, acc_s_ub)
        T.reduce_max(acc_s_ub, m_i, dim=-1)
        T.tile.sub(acc_s_ub, acc_s_ub, m_i)
        ...
```

## 3. Important Limitations

### 3.1 Current Constraints

- **Static shape requirement**: Workspace shapes must be statically determinable at compile time
- **Vid Reduction dependency**: Requires coordination with Vid Reduction, i.e., `threads` parameter must be set to 1 or 2

### 3.2 Known Limitations

- Multi-dimensional workspace: Currently supports 2D shapes `[M, N]`, future extension to higher dimensions if needed.
- Dynamic shapes: Not supported yet, requires compile-time constants
- Skip UB scenario: Currently only supports indices array transfer in Sparse Flash Attention (SFA) operator

### 3.3 Future Enhancements

- **Scoped copy-back insertion**: Support region-based copy insertion strategy where Stage 2 copy statements are inserted at region boundaries before all data consumers, reducing interleaving of copy and compute operations, and avoiding efficiency degradation from redundant cross-core synchronization overhead

