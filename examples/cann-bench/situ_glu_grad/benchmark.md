**English** | [中文](benchmark_zh.md)

# SiTU-GLU Grad (SituGluGrad) Benchmark

## 1. Operator

```
t = tanh(gate / beta),  s = sigmoid(gate),  situ_a = beta * t * s
up' = linear_beta * tanh(up / linear_beta)   (linear_beta > 0), else up
grad_gate = (grad_y * up') * s * [(1 - t^2) + beta * t * (1 - s)]
grad_up   = (grad_y * situ_a) * (1 - tanh^2(up / linear_beta))   (linear_beta > 0)
grad_up   = (grad_y * situ_a)                                   (linear_beta <= 0)
```

- Backward of the Kimi K3 Stable LatentMoE gated activation (SiTU-GLU); `x` is
  split along `dim` into gate/up halves according to `activate_left`
- Input: x [..., cut] (cut even), grad_y [..., cut/2]; fp32 / fp16 / bf16
- Output: grad_x, same shape and dtype as x
- Config: dim=-1, beta=4.0, linear_beta=25.0, activate_left=True (Kimi K3 production)
- Reference: CANN ops-nn `activation/situ_glu_grad` (aclnnSituGluGrad / `cann_ops_nn.situ_glu_grad`)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B3 (Atlas A2) |
| CANN | 9.2.0 |
| Framework | torch 2.7.1 + torch_npu 2.7.1.post4 |
| Baseline | `cann_ops_nn.situ_glu_grad` (official AscendC kernel) |
| Measurement | NPUGraph capture-replay, kernel-only |

## 3. Architecture (single kernel, pure Vector)

| Component | Choice | Reason |
|-----------|--------|--------|
| Core usage | dual V-core row split per block (`vid` splits the row tile) | elementwise op, bandwidth-bound |
| Tiling | 16x128 tiles with `T.Pipelined(num_stages=2)` over a column strip | overlaps MTE2 load / vector math / MTE3 store |
| Math | fp32 intermediate compute, `CAST_RINT` write-back | matches the official op's internal precision strategy |
| tanh | `2 * sigmoid(2x) - 1` | no tanh tile primitive; numerically stable |
| Specialization | compile-time (dtype, beta, linear_beta > 0, activate_left) | the identity branch computes no `dtu` ops |
| Host adapter | flatten to [rows, cut*inner], zero-pad to block multiples, trim | aligned shapes (incl. Kimi) pay no padding |

## 4. Performance Tuning

- **Strip selection**: `tiles_per_block` is chosen by a shape-aware cost model
  (`waves * (c * per_tile + fill)`) that trades long strips (lower per-task
  pipeline fill overhead) against wave count. Verified on device: M=512 -> 4
  (47.2 µs vs 50.1 µs at 24), M=1024/2048 -> 8, M=4096 -> 12, M=8192 -> 24.
  A naive "keep every core busy" rule (shorter strips) regresses all shapes.
- **Bandwidth-bound**: reads 1.5x cut elements (x + grad_y), writes 1x cut
  (grad_x); tuning targets memory throughput rather than arithmetic.

## 5. Performance (kernel-only, Kimi K3 shapes)

Shapes: x [M, 6144] bf16, grad_y [M, 3072], dim=-1, beta=4.0, linear_beta=25.0,
activate_left=True. Protocol: NPUGraph capture-replay (host overhead removed),
warmup 5 / rep 20 per config, interleaved A/B, median.

| M | tilelang kernel | aclnn baseline | Speedup |
|---|-----------------|----------------|---------|
| 512 | 47.3 µs | 42.6 µs | 0.90x |
| 1024 | 83.9 µs | 80.3 µs | 0.96x |
| 2048 | 161.6 µs | 156.3 µs | 0.97x |
| 4096 | 305.3 µs | 305.9 µs | 1.00x |
| 8192 | 595.2 µs | 605.4 µs | 1.02x |

Notes:
- M=512 is a small-batch shape: the per-task pipeline fill/launch overhead
  dominates, giving 0.90x kernel-only; M>=1024 stays within 4% of the official
  kernel and M=4096/8192 reach parity or slightly above.

## 6. Accuracy (910B3, vs independent fp64 golden)

Mixed tolerance: `abs_err <= atol + rtol * |golden|`, `matched_ratio >= 0.99`;
atol/rtol = 2^-10 / 2^-6 (bf16), 2^-14 / 2^-9 (fp16), 2^-16 / 2^-10 (fp32).

| Layer | Coverage | Result |
|-------|----------|--------|
| L0 | bf16 / fp16 / fp32 x Kimi config + default params | 4/4 |
| L1 | dim variants (0/1/-2/-1, 2D-4D), activate_left=False, linear_beta=0/negative, beta=100, unaligned cut/rows, non-contiguous inputs, M=1, small shapes | 17/17 |
| L2 | odd cut / beta=0 / grad_y shape mismatch / bad dim (rejected) | 4/4 |
| Boundary | empty tensor / saturation / tiny values | 2/2 |

Key numbers: bf16 Kimi max_abs = 1.56e-2 (matched_ratio = 1.0); fp16 = 1.95e-3;
fp32 = 9.06e-6. Cross-check against the official aclnn kernel: max relative
diff ~3e-3 (bf16), consistent with bf16 rounding.

## 7. Notes & Limitations

- Non-divisible shapes are zero-padded on the host; padded regions produce zero
  gradients and are trimmed away, so aligned shapes pay no overhead.
- The strip cost-model constants are calibrated on A2 (910B3, 40 AIV cores).
- M=512 stays below 1.0x vs the official kernel (small-batch overhead); it is
  reported as-is rather than traded off against the prefill-scale shapes.
