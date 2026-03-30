# T.Pipelined on TileLang-Ascend

## Overview

`T.pipelined` is a high-level abstraction in TileLang-Ascend designed to express and optimize pipelined parallelism on Ascend AI accelerators. It enables fine-grained overlapping of computation, memory access within a single core (intra-core), and synchronization across multiple cores (inter-core).

## Usage

### Interface

```python
for var in T.Pipelined(loop_iterations: int, num_stages: int, cross_interval: int = 1):
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loop_iterations` | int | Required | Total number of loop iterations |
| `num_stages` | int | Required | Number of pipeline stages for double buffering |
| `cross_interval` | int | 1 | Interval for cross-core synchronization (only effective in inter-core pipeline) |

### Intra-core Pipeline

Intra-core pipeline overlaps memory access and computation within a single core. When `num_stages=2`, the execution pattern is:

```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)

    T.barrier_all()
    if k == 0:
        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    else:
        T.gemm_v0(A_L1, B_L1, C_L0)

    T.barrier_all()
```

Assuming `loop_k = 4`, the execution timeline:

| Time | Copy A       | Copy B       | Compute    |
|------|--------------|--------------|------------|
| t₀   | **copy_A_0** | **copy_B_0** |            |
| t₁   | **copy_A_1** | **copy_B_1** |            |
| t₂   | **copy_A_2** | **copy_B_2** | **gemm_0** |
| t₃   | **copy_A_3** | **copy_B_3** | **gemm_1** |
| t₄   |              |              | **gemm_2** |
| t₅   |              |              | **gemm_3** |

With `num_stages=2`:
- Prefetch: `copy_A_0 copy_A_1` and `copy_B_0 copy_B_1`
- Main body: `copy_A_2 copy_B_2 gemm_0` and `copy_A_3 copy_B_3 gemm_1`
- Epilogue: `gemm_2` and `gemm_3`

### Inter-core Pipeline

Inter-core pipeline overlaps computation between Cube and Vector cores through workspace buffers.

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    # Cube: write to workspace
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    # Vector: read from workspace
    T.copy(workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :], acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
```

Assuming `T.ceildiv(seq_len, block_N) = 4`, the execution timeline with `num_stages=2`:

| Time | Write Workspace | Read Workspace |
|------|-----------------|-----------------|
| t₀   | **write_0**    |                 |
| t₁   | **write_1**     | **read_0**      |
| t₂   | **write_2**    | **read_1**      |
| t₃   | **write_3**    | **read_2**      |
| t₄   |                 | **read_3**      |

### Cross-interval (Inter-core Only)

`cross_interval` controls how often cross-core synchronization occurs in inter-core pipeline.

```python
# cross_interval=1: sync every iteration (default)
for k in T.Pipelined(num_iters, num_stages=4, cross_interval=1):
    # CrossCoreSetFlag executed every iteration

# cross_interval=2: sync every 2 iterations
for k in T.Pipelined(num_iters, num_stages=4, cross_interval=2):
    # CrossCoreSetFlag executed at i=1,3,5...
    # CrossCoreWaitFlag executed at i=0,2,4...
```

| cross_interval | Sync Frequency | Use Case |
|----------------|----------------|----------|
| 1 | Every iteration | Default, highest parallelism |
| N | Every N iterations | Reduced sync overhead, multi-KV cache |

**Generated code example** (cross_interval=2, num_stages=4):

```c
// Writer (Cube): set flag when i % 2 == 1 or last iteration
if (((i % 2) == 1) || (i == 3)) {
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(0);
}

// Reader (Vector): wait flag when i % 2 == 0
if ((i % 2) == 0) {
    AscendC::CrossCoreWaitFlag(2);
}
```

> **Note**: `cross_interval` is only effective when inter-core pipeline is enabled (i.e., `TL_ASCEND_AUTO_CV_SYNC=True` and operations are distributed across Cube/Vector cores). In intra-core pipeline, this parameter has no effect.

## Pipeline Pattern: Nested vs. Flat

### Nested Pattern (Not Supported)

In nested pattern, both inter-core and intra-core pipeline are enabled using `T.Pipelined`:

```python
# ❌ Not supported: T.Pipelined nested, enabling both inter-core and intra-core
for k in T.Pipelined(num_iters, num_stages=4):  # Inter-core pipeline
    # T.Pipelined enables inter-core sync

    # Nested T.Pipelined for intra-core double buffering
    for side in T.Pipelined(2, num_stages=2):  # ❌ Not supported
        T.copy(K[side], k_l1)
        T.copy(k_l1, l0b[side])
        T.mma(l0a[side], l0b[side], l0c[side], init=(side == 0))
```

This pattern is not supported and may cause undefined behavior.

### Flat Pattern (Recommended)

In flat pattern, use `T.Pipelined` for inter-core pipeline, and manually implement intra-core pipeline:

```python
# ✅ Recommended: T.Pipelined for inter-core, manual for intra-core
for k in T.Pipelined(num_iters, num_stages=4):
    # T.Pipelined handles inter-core sync automatically

    # Manual intra-core double buffering (side = 0, 1)
    for side in T.serial(2):
        T.copy(K[side], k_l1)
        T.set_flag("MTE2", "MTE1", SIG_K_L1)
        T.wait_flag("MTE1", "MTE2", SIG_K_L1)

        T.copy(k_l1, l0b[side])
        T.set_flag("MTE1", "MTE2", SIG_K_L1)
        T.set_flag("MTE1", "M", SIG_L0AB + side)

        T.wait_flag("MTE1", "M", SIG_L0AB + side)
        T.wait_flag("FIX", "M", SIG_L0C + side)
        T.mma(l0a[side], l0b[side], l0c[side], init=(side == 0))
        T.set_flag("M", "MTE1", SIG_L0AB + side)
        T.set_flag("M", "FIX", SIG_L0C + side)
```

Benefits of flat pattern:
- **Clear separation**: `T.Pipelined` handles inter-core sync, manual code handles intra-core
- **Maintainable**: Easier to understand and debug
- **Automatic optimization**: Compiler can optimize inter-core pipeline better

## Constraints

- **Intra-core pipeline and inter-core pipeline cannot be nested**. Use flat pattern as shown above.
- Multiple inter-core pipelines are not supported within a single program.
- When using inter-core pipeline, automatic CV separation and synchronization must be enabled:
  ```python
  pass_configs = {
      tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
      tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
  }
  ```
