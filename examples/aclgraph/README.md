# Using AclGraph Mode in TileLang

This example demonstrates how to combine TileLang operators with `torch.npu.NPUGraph` (AclGraph), using the Graph Capture mechanism to fuse multiple operators for execution, reducing kernel launch overhead and improving end-to-end performance.

## Background

AclGraph is a graph execution mode. Its core idea is:

1. **Capture phase**: Record a series of operator calls into a computation graph
2. **Replay phase**: Submit the entire graph for execution at once, avoiding the overhead of dispatching kernels one by one

When multiple TileLang operators need to be executed sequentially (e.g., RMS Norm followed by RoPE), using AclGraph can significantly reduce scheduling latency.

## Quick Start

### Run the Example

```bash
# Run with default parameters
python rms_rope_aclgraph.py
```

## Code Walkthrough

Using [rms_rope_aclgraph.py](rms_rope_aclgraph.py) as an example, the complete workflow consists of three steps: **Define operators → Graph capture → Graph replay**.

### Step 1: Configure Compilation Options

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}
```

| Config Key | Description |
|------------|-------------|
| `TL_ASCEND_AUTO_SYNC` | Automatically insert synchronization between read/write operations on shared buffers |
| `TL_ASCEND_MEMORY_PLANNING` | Enable Ascend memory planning optimization to reduce memory usage |
| `TL_ASCEND_AUTO_CV_SYNC` | Automatically insert Cube/Vector inter-core synchronization |
| `TL_ASCEND_AUTO_CV_COMBINE` | Automatically combine cross-core communication to reduce synchronization overhead |

### Step 2: Define Operators with `@tilelang.jit`

#### RMS Norm Operator

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def rms_norm_kernel(M, head_dim, block_M, eps, dtype="float16"):
    @T.prim_func
    def main_rms(
        x: T.Tensor((M, head_dim), dtype),
        out: T.Tensor((M, head_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # 1. Copy from Global Memory to Unified Buffer
            T.copy(x[row_x : row_x + row_per_vec, :], x_ub)
            # 2. Compute sum of squares → reduce → mean → sqrt
            T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)
            T.reduce_sum(sum_square_ub, rms_ub, dim=-1)
            T.tile.div(rms_ub, rms_ub, head_dim)
            T.tile.add(rms_ub, rms_ub, eps)
            T.tile.sqrt(rms_ub, rms_ub)
            # 3. Normalize and write back
            for i in T.serial(0, row_per_vec):
                T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
            T.copy(x_ub, out[row_x : row_x + row_per_vec, :])
    return main_rms
```

**Key parameter notes**:

- `out_idx=[-1]`: Designates the last Tensor parameter (`out`) as the output; TileLang will automatically allocate and return this Tensor
- `pass_configs`: Passes the compilation optimization options defined above
- `is_npu=True`: Declares that this Kernel runs on the Ascend NPU

#### RoPE Operator (In-place)

```python
@tilelang.jit(pass_configs=pass_configs)
def rope_kernel_in_place(M, block_M, batch_size, hidden_size, rope_dim, head_num, dtype="float16"):
    @T.prim_func
    def main_rope(
        x: T.Tensor([M, hidden_size], dtype),
        sin: T.Tensor([batch_size, rope_dim], dtype),
        cos: T.Tensor([batch_size, rope_dim], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # Modify x in-place, no extra output Tensor needed
            ...
    return main_rope
```

Note that **`out_idx` is not specified** here, because RoPE is an in-place operation that directly modifies the input `x`.

### Step 3: AclGraph Capture and Replay

This is the core step for using AclGraph mode:

```python
# 1. Create an NPUGraph object
g = torch.npu.NPUGraph()

# 2. Capture the operator call sequence
with torch.npu.graph(g):
    q = tilelang_rms_norm(q, variance_epsilon)   # RMS Norm
    q = tilelang_apply_rope(q, sin, cos)          # RoPE

# 3. Replay execution (can be called multiple times)
g.replay()
```

**Notes**:

- Operations inside `with torch.npu.graph(g)` are only **recorded**, not executed immediately
- The recorded operations are actually executed when `g.replay()` is called
- After graph capture, `replay()` can be called multiple times, executing the same computation flow each time

### Step 4: Verify Results

```python
# Compare against PyTorch reference implementation
torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
```

## Verifying AclGraph is Enabled

Check the info-level Host compilation logs and search for the `capture model` keyword. A value of 0 (default) means AclGraph mode is disabled; a value of 1 means it is enabled. Configure the environment variables as follows:

```bash
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=logs
export ASCEND_GLOBAL_LOG_LEVEL=1
```

## Complete Workflow Summary

```
Define pass_configs (compilation options)
        │
        ▼
Define operators with @tilelang.jit
        │
        ▼
Write Python wrapper functions to call operators
        │
        ▼
torch.npu.NPUGraph() — create graph object
        │
        ▼
with torch.npu.graph(g): — capture operator calls
        │
        ▼
g.replay() — execute the computation graph
```
