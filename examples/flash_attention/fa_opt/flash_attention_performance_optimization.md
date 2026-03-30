# CV Fusion Operator Performance Analysis and Tuning Methods

[中文](flash_attention_performance_optimization_zh.md) | **English**

## Table of Contents

- [I. Performance Analysis Tools](#i-performance-analysis-tools)
- [II. Overall Performance Optimization Strategy](#ii-overall-performance-optimization-strategy)
- [III. Intra-Core Optimization](#iii-intra-core-optimization)
- [IV. Inter-Core Optimization (CV Fusion)](#iv-inter-core-optimization-cv-fusion)
- [V. TileLang Performance Optimization Primitives](#v-tilelang-performance-optimization-primitives)
- [VI. Common Performance Issues and Solutions](#vi-common-performance-issues-and-solutions)

---

## I. Performance Analysis Tools

### 1.1 msprof Tool Overview

CANN provides a built-in performance analysis tool `msprof` that supports two modes:

| Mode | Purpose | Output |
|------|------|------|
| **op mode** | Collect real hardware performance data | kernel execution time, bandwidth utilization, etc. |
| **simulator mode** | Collect pipeline simulation data | Visualized pipeline diagrams |

### 1.2 Collecting Performance Data

```bash
msprof op --kernel-name="main_kernel" --output=<output_path> python3 xxx.py
```

**Example Scripts**:
- [Batch Testing Script](bench.sh)

### 1.3 Collecting Pipeline Diagrams (Simulator Mode)

```bash
msprof op simulator --soc-version=Ascend910B4 --kernel-name="main_kernel" --output=<output_path> python3 xxx.py
```

**Note**:
- `--soc-version` should be replaced with the actual NPU model (e.g., Ascend910B1, Ascend910B4, etc.)
- Use tool prompts to get the list of supported models

**Pipeline Diagram Viewing Methods**:
1. Access `chrome://tracing/` in Chrome browser and load the output file
2. Download [MindStudio Insight](https://gitcode.com/Ascend/msinsight/releases/tag_MindStudio_26.0.0-alpha.1) tool

---

## II. Overall Performance Optimization Strategy

Taking FA operator as an example, this section introduces the overall approach for CV fusion operator performance optimization.

### 2.1 Optimization Hierarchy Architecture

| Inter-Core Optimization | Intra-Core Optimization |
|:---:|:---:|
| · num_stages | Cube Core: L1 resident, DB |
| · Load balance | Vector Core: MTE2/VEC/MTE3 DB |
| · Sync optimize | · num_stages |
| | · Instruction vectorization |

### 2.2 Optimization Process

```
1. Performance benchmark → 2. Identify bottlenecks → 3. Targeted optimization → 4. Verify gains
              ↑                                                                   ↓
              └─────────────────── Iterative optimization ←───────────────────────┘
```

### 2.3 Core Principles

| Principle | Description |
|------|------|
| **Mask shorter pipelines** | Try to mask shorter pipelines with longer ones |
| **Reduce bubbles** | Optimize task scheduling to reduce inter-core wait time |
| **Single Bound** | Ideally optimize to a single type of pipeline bound (e.g., fixPipe bound) |

---

## III. Intra-Core Optimization

### 3.1 Double Buffer Principle

**Background**: AI Core instruction queues are independent and can execute in parallel:
- **MTE Queue**: Memory transfer instructions
- **Vector Queue**: Vector computation instructions
- **Cube Queue**: Matrix computation instructions

**Example**: Vector core execution order `MTE2 → VEC → MTE3`

| Mode | Execution Method | Characteristics |
|------|----------|------|
| **Serial Mode** | Data blocks execute sequentially | Long single block time, waiting exists |
| **Double Buffer** | Data blocks split, pipeline parallel | Reduced total time, high resource utilization |

```
Serial Mode:
  Block0: [MTE2][VEC][MTE3]
  Block1:        ----------[MTE2][VEC][MTE3]
  
Double Buffer:
  Block0: [MTE2][VEC][MTE3]
  Block1:   [MTE2][VEC][MTE3]
```

### 3.2 Cube Core Optimization

#### 3.2.1 L1 Memory Resident

Reduce the number of data transfers between GM and L1.

| Strategy | Applicable Scenario | Implementation |
|------|----------|----------|
| **Large Reuse** | Sufficient L1 memory | Q persists in L1 across multiple basic blocks, not released during P@V |
| **Small Reuse** | Limited L1 memory | Q persists in L1 for one basic block, released during P@V |

**Code Example (Large Reuse)**:
```python
T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
    for n_i in T.serial(n_num):
        T.copy(K[bz, by, k * block_N + n_i * block_K : k * block_N + (n_i + 1) * block_K, :], k_l1)
        T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
        T.copy(acc_s_l0c, workspace_1[cid, :, n_i * block_K : (n_i + 1) * block_K])
```

#### 3.2.2 L1 → L0 Double Buffer

When L0 space is smaller than L1, multiple transfers are needed with overlap between rounds.

#### 3.2.3 Optimize to Single Bound

**Goal**: Use the longest pipeline as the bound, and mask other pipelines with it.

```
Before optimization (pipelines execute serially):
Timeline →
MTE:  [==]        [==]
M:          [===]      [===]
FIX:                         [====================]

After optimization (one FIX masks two MTE and M):
Timeline →
      |------ One FIX cycle ------|
MTE:  [==]   [==]               ← 2x MTE masked by FIX
M:    [===]  [===]              ← 2x M masked by FIX
FIX:  [====================]     ← bound pipeline
```

### 3.3 Vector Core Optimization

#### 3.3.1 Intra-Core Double Buffer

MTE2, VECTOR, MTE3 of different data blocks mask each other.

#### 3.3.2 Algorithm Optimization

**(1) Scalar Vectorization**

Transform multiple scalar operations in for loops into tile operations:

```python
# Before: Multiple scalar operations in loop
for h_i in range(block_M // 2):
    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])

# After: Single tile operation
T.tile.broadcast(m_i_2d, m_i, tmp_ub)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

**(2) Reduce Instruction Dispatch Count**

Use Axpy algorithm to merge instructions:

```python
# Before: Two instructions
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)

# After: One instruction (if applicable)
# Use fused instructions like Axpy based on actual scenarios
```

---

## IV. Inter-Core Optimization (CV Fusion)

### 4.1 Finding the Optimal num_stages

**Principle**: With more blocks, increasing `num_stages` can reduce bubbles when CV task execution times are uneven.

```
num_stages = 2:
  C-core: [C0]----[C1]--------[C2]
  V-core:      [V0]----[V1]--------[V2]
               ↑ C1 waits for V0 to complete

num_stages = 3:
  C-core: [C0][C1]----[C2]----[C3]
  V-core:     [V0][V1]----[V2]----[V3]
               ↑ Bubbles reduced
```

**Tuning Recommendations**:
- Start with `num_stages=2` and gradually increase
- Observe C/V core time ratio and choose the value that minimizes bubbles
- Note that excessive `num_stages` increases memory usage

### 4.2 Inter-Core Synchronization Optimization

**Problem**: Too many inter-core synchronizations may cause scalar bound.

**Solution**: Reduce synchronization frequency, e.g., synchronize every two tasks:

```python
# Before: Synchronize every task
for i in range(n):
    process()
    sync()

# After: Synchronize after multiple tasks
for i in range(n):
    process()
    if i % 2 == 1:
        sync()
```

---

## V. TileLang Performance Optimization Primitives

### 5.1 T.pipelined Primitive

Used to enable intra-core or inter-core pipeline masking.

**Syntax**:
```python
for i in T.pipelined(loop_range, num_stages=N):
    # Task processing
```

**Parameter Description**:
| Parameter | Description |
|------|------|
| `loop_range` | Loop range |
| `num_stages` | Number of pipeline stages, controls task parallelism |

**Detailed Documentation**: [T.pipelined Tutorial](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/tutorials/t_pipelined.md)

### 5.2 Usage Restrictions

**Important**: `T.pipelined` does not support nested usage.

```python
# Method 1: Enable inter-core pipeline (recommended)
for i in T.pipelined(loop, num_stages):
    process_on_cube()
    process_on_vector()

# Method 2: Enable inter-core/intra-core pipeline separately
for i in T.pipelined(loop, num_stages):  # Intra-core
    process_on_cube()
for i in T.pipelined(loop, num_stages):  # Intra-core
    process_on_vector()

# Wrong: Nested usage
for i in T.pipelined(outer, num_stages):
    for j in T.pipelined(inner, num_stages):  # Not supported!
        ...
```

### 5.3 Recommended Practices

| Scenario | Recommended Method |
|------|----------|
| **Inter-core pipeline** | Use `T.pipelined` primitive |
| **Intra-core pipeline** | Manual implementation at frontend (manually split data blocks + synchronization) |
| **Synchronization interval control** | Use `cross_interval` parameter (coming soon) |

---

## VI. Common Performance Issues and Solutions

### 6.1 Problem Diagnosis Process

```
Performance below expectations
    │
    ├─→ Collect msprof data
    │       │
    │       ├─→ C/V core time imbalance → Adjust num_stages
    │       │
    │       ├─→ Large intra-core pipeline bubbles → Enable Double Buffer
    │       │
    │       └─→ Many scalar operations → Vectorization refactoring
    │
    └─→ Check inter-core synchronization
            │
            └─→ Too many synchronizations → Reduce sync frequency
```

### 6.2 Common Issues Quick Reference

| Symptom | Possible Cause | Solution |
|------|----------|----------|
| Large bubbles in C-core | V-core takes long, `num_stages` too small | Increase `num_stages` |
| Memory overflow | `num_stages` too large or buffer too large | Reduce block parameters or `num_stages` |
| Slow instruction dispatch | Too many scalar operations | Use `T.tile` vectorized operations |
| GM bandwidth not saturated | Low data transfer efficiency | Enable L1 resident, Double Buffer |

### 6.3 Tuning Checklist

- [ ] Collect msprof performance data
- [ ] Analyze C/V core time ratio
- [ ] Try different `num_stages` values
- [ ] Check L1/L0 memory utilization
- [ ] Confirm Double Buffer is enabled
- [ ] Optimize scalar operations to vectorization
- [ ] Reduce unnecessary inter-core synchronizations

---

## Appendix: Related Resources

- [T.pipelined Detailed Tutorial](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/tutorials/t_pipelined.md)
- [TileLang-Ascend Programming Guide](../../../docs/TileLang-Ascend%20Programming%20Guide.md)
- [MindStudio Insight Download](https://gitcode.com/Ascend/msinsight/releases/tag_MindStudio_26.0.0-alpha.1)