# Flash Attention for Ascend NPU using TileLang-Ascend

## Overview

This repository provides an implementation of **Flash Attention** forward pass using **TileLang-Ascend** DSL for Huawei Ascend NPU. The implementation leverages the hardware characteristics of Ascend NPU through block-wise computation, pipeline optimization, and automatic memory planning to achieve high-performance attention computation.

## Flash Attention Mathematical Formulation

### Standard Attention Computation

The standard attention computation can be expressed as:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- `Q, K, V ∈ ℝ^{batch×heads×seq_len×dim}` are input tensors
- `d_k = dim` is the dimension of the key vectors
- `softmax` is applied along the last dimension

### Online Softmax Algorithm for Block-Wise Computation

For long sequences, computing the full attention matrix `S = Q·K^T ∈ ℝ^{seq_len×seq_len}` is memory-intensive. Flash Attention uses a block-wise algorithm with online softmax updates:

#### State Variables for Row i:
- `m_i`: maximum value encountered so far for row i
- `ℓ_i`: sum of exponentials for row i (scaled by previous max)

#### Online Update Rules:

For each new block `S_new ∈ ℝ^{block_M×block_N}` of attention scores:

1. **Update maximum**:
   ```
   m_i_new = max(m_i_old, max(S_new[i, :]))
   ```

2. **Update exponential sum**:
   ```
   scale = exp(m_i_old - m_i_new)
   ℓ_i_new = ℓ_i_old × scale + sum(exp(S_new[i, :] - m_i_new))
   ```

3. **Update output**:
   ```
   O_i_new = O_i_old × scale + exp(S_new[i, :] - m_i_new) · V_block
   ```

4. **Final normalization**:
   ```
   O_i_final = O_i / ℓ_i
   ```

This algorithm allows processing the attention computation in blocks while maintaining numerical stability.