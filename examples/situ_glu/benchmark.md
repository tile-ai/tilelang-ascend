**English** | [中文](benchmark_zh.md)

# SituGlu Benchmark

## 1. Operator

```
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
up'    = linear_beta * tanh(up / linear_beta)   (linear_beta > 0), else up
y      = situ_a * up'
```

- SiTU gated linear unit, the forward gated activation used by Kimi K3's
  Stable LatentMoE; `x` is split along `dim` into gate/up halves according to
  `activate_left`
- Input: x [..., cut] (cut even), 1-8 dims, fp32 / fp16 / bf16, ND
- Output: y [..., cut/2], same dtype as x
- Config: dim=-1, beta=1.0, linear_beta=0.0, activate_left=True (defaults)
- Reference: CANN ops-nn `activation/situ_glu` (aclnnSituGlu / `cann_ops_nn.situ_glu`)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910_9362 (Atlas A3), 40 AIV cores |
| CANN | 9.0.0 |
| Framework | torch 2.7.1 + torch_npu 2.7.1.post4 |
| Baseline | official AscendC kernel built standalone from the ops-nn source (`op_kernel/situ_glu.hpp` verbatim + learning-hub direct-launch harness + ported op_host tiling); `aclnnSituGlu` ships with CANN 9.2.0 and is unavailable under 9.0.0 |
| Measurement | aclrtEvent pairs around N in-flight launches, kernel-only, median of 5 rounds |

## 3. Architecture (single kernel, pure Vector)

| Component | Choice | Reason |
|-----------|--------|--------|
| Core usage | one grid block per output tile, dual V-core row split (`vid`) | elementwise op, bandwidth-bound |
| Tiling | (block_M, block_N) output tiles walked in `sub_M`-row chunks | chunk = the pipelining unit; see below |
| Pipeline | explicit flag-based double buffering: chunk mm+1's MTE2 load overlaps chunk mm's vector math and chunk mm-1's MTE3 store (`set_flag`/`wait_flag`, `elementwise_add_pipeline.py` pattern) | no auto-sync passes are enabled (auto barriers serialize the loop and interfere with manual flags) |
| Shape policy | `M`/`N` are symbolic by default (one compile serves any shape); concrete ints give a static specialization with lighter dispatch | serving-friendly while allowing fixed-shape deployments |
| Math | fp32 intermediate compute, `CAST_RINT` write-back for bf16 (`CAST_NONE` for fp16) | matches the official op's internal precision strategy; CAST_RINT is required for bf16 on non-A2 SoCs |
| tanh | `2 * sigmoid(2x) - 1` | no tanh tile primitive; numerically stable (saturates instead of producing `inf/inf = NaN` for large inputs, unlike the exp-difference form) |
| Tail handling | ceildiv grid + full-tile `T.copy` with scalar start; the frontend clamps copy extents to tensor bounds and zero-pads reads | no host-side padding or cropping |
| Specialization | compile-time on (dtype, beta, linear_beta > 0, activate_left) | the `linear_beta <= 0` branch computes no bounded-tanh ops; `beta == 1` folds scalar multiplies |

## 4. Performance Tuning

Tile calibration on (4096, 18432) bf16 (static specialization, kernel-only):

| Tile (block_M x block_N, sub_M) | Time | Bandwidth |
|---|---|---|
| 128 x 128, 32 | 514.9 us | 440 GB/s |
| 256 x 128, 32 | 455.0 us | 498 GB/s |
| 256 x 128, 64 | 420.7 us | 538 GB/s |
| 512 x 128, 32 | 432.3 us | 524 GB/s |
| 256 x 256, 32 | 360.3 us | 629 GB/s |
| 512 x 64, 32 | 572.3 us | 396 GB/s |
| **512 x 256, 32** | **352.5 us** | **643 GB/s** |

Wider `block_N` wins by lengthening contiguous DMA segments; deeper blocks
(larger `block_M / sub_M`) amortize the pipeline prologue/epilogue. The host
adapter picks tiles from the input shape (metadata-only heuristic):
M >= 4096 -> (512, 256); M >= 1024 -> (256, 256); otherwise (128, 128).

## 5. Performance (kernel-only)

Shapes use the LLM FFN profile (cut = 2 x hidden). Protocol as in Section 2.

| Shape | dtype | tilelang | official kernel | speedup | tilelang BW | unfused torch |
|---|---|---|---|---|---|---|
| (4096, 18432) | bf16 | 352.5 us | 727.6 us | 2.06x | 643 GB/s | 1421 us |
| (8192, 18432) | bf16 | 678.5 us | 1444.0 us | 2.13x | 668 GB/s | 2746 us |
| (4096, 18432) | fp16 | 348.1 us | 723.1 us | 2.08x | 651 GB/s | 1402 us |
| (4096, 18432) | fp32 | 520.1 us | 816.6 us | 1.57x | 871 GB/s | 2158 us |

"unfused torch" = `chunk + tanh + sigmoid + mul` as separate torch ops
(the pre-fusion form this operator replaces).

Host-side note: symbolic-shape calls pay a fixed ~0.57 ms dispatch cost per
call (per-call shape inference in the JIT adapter), static specializations
~0.18 ms. Kernel-only numbers above exclude this; for small batches the
dispatch dominates end-to-end latency.

## 6. Accuracy

Ascend 910_9362, vs an fp64 PyTorch golden. Layered suite: L0 (9) / L1 (24) /
L2 negative (9) / Boundary (8) — dtype coverage, unaligned/prime/edge shapes,
rank 1-8, non-default attrs, sparse ±inf / NaN (position-sensitive mask
comparison), dtype bounds, empty tensors. 3 repeated full runs, no flakes.

Mixed tolerance `abs_err <= atol + rtol * |golden|`, `matched_ratio >= 0.99`:

| dtype | atol | rtol |
|---|---|---|
| float16 | 2^-14 | 2^-9 |
| bfloat16 | 2^-10 | 2^-6 |
| float32 | 2^-16 | 2^-10 |
