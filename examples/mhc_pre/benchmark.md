# MHC Pre Benchmark & Optimization Path

## 1. Operator

```
mHC Pre forward pipeline:
  1. out = x @ fn.T, sqrsum = x^2.sum(-1)
  2. mixes = out * rsqrt(sqrsum / (hc * hidden) + rms_eps)
  3. pre/post/comb = split(mixes) + Sinkhorn normalization
  4. layer_input = sum over hc of (residual * pre_mix)
```

- Input: residual [n, hc, hidden] bf16, fn [hc_mult3, hc*hidden] fp32, hc_scale [3] fp32, hc_base [hc_mult3] fp32
- Output: post_mix [n, hc, 1] fp32, comb_mix [n, hc, hc] fp32, layer_input [n, hidden] bf16
- Constraint: hc = 4 (B3 AXPY specialization)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| Tool | do_bench (Python) |
| Dtype | bf16 input, fp32 accumulate |

## 3. Architecture (5-kernel pipeline)

| Kernel | Function | Core | Key Parameters |
|--------|----------|------|----------------|
| A1 | GEMM: out = x @ fn.T | Cube (T.gemm_v0) | token_block=128, h_blk=512, T.Pipelined |
| A2 | sqrsum = x^2.sum(-1) | Vector (dual-V-core) | h_blk=4096, T.Pipelined |
| B1 | RMSNorm | Vector (dual-V-core) | - |
| B2 | split + Sinkhorn | Vector (dual-V-core) | adapted from hc_split_sinkhorn.py |
| B3 | apply pre_mix | Vector (AXPY, dual-V-core) | hc=4 specialized, h_blk=2048 |

A1 uses Cube GEMM (single bid dimension). A2/B1/B2/B3 use dual-V-core partitioning (bid = cid * 2 + vid).

## 4. Optimization Path

| Step | Change | Effect |
|------|--------|--------|
| A1 token_block | 16 -> 128 | Better Cube utilization |
| A1 h_blk | 128 -> 512 | K-tile sweep optimum |
| A2 h_blk | 128 -> 4096 | Fewer loop iterations |
| A2 dual-V-core | cid*2+vid | 2x Vector throughput |
| B1/B2/B3 dual-V-core | cid*2+vid | All Vector kernels utilize both V-cores |
| B3 AXPY | hc=4 specialized | From mhc_post experience, avoids broadcast+reduce |
| Skip unnecessary padding | h aligned -> no pad | Eliminate wasted compute |
| A1 output -> B1 direct | padded output passed directly | Eliminate 32->24->32 repad |
| fn prepack/cache | prepare_fn + fn_packed | Avoid repeated cast/transpose at inference |
| kernel compile cache | _kernel_cache dict | Avoid repeated JIT lookup |

## 5. Final Performance (E2E, do_bench, warmup=10, rep=50, prepacked fn)

| n | h | hc | TileLang | PyTorch (CANN) | Speedup |
|---|---|---|----------|----------------|---------|
| 512 | 2560 | 4 | 2.08 ms | 1.35 ms | 0.65x |
| 4096 | 2560 | 4 | 2.13 ms | 2.29 ms | 1.07x |
| 4096 | 7168 | 4 | 2.90 ms | 5.22 ms | 1.80x |

Small shape (512x2560) is slower than CANN due to multi-kernel launch overhead. Large shapes benefit from the pipeline.

## 6. Kernel Breakdown (n=4096, h=7168)

| Kernel | Latency | Share |
|--------|---------|-------|
| A1 GEMM | 0.56 ms | 21.5% |
| A2 sqrsum | 0.52 ms | 20.3% |
| B1 RMSNorm | 0.24 ms | 9.1% |
| B2 Sinkhorn | 0.74 ms | 28.5% |
| B3 apply | 0.53 ms | 20.5% |

B2 (Sinkhorn) is the largest Vector hotspot (28-41% of kernel time).

## 7. Known Limitation

B2 (Sinkhorn) is the largest Vector hotspot. A static 1D-buffer specialization was evaluated, but the current Ascend backend encounters an AICore failure for the required 2D-to-1D T.copy slice pattern. The generic verified Sinkhorn path is retained for correctness and compiler stability.

## 8. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 7/7 passed (including distinct-eps parameter test) |
| Tolerance | rtol=1e-2, atol=1e-2 |
| Max diff | 0.0156 (layer_input, n=4096) |
| Source of diff | BF16 quantization and different accumulation order of AXPY-based apply kernel |

## 9. Stop Condition

| Condition | Status |
|-----------|--------|
| E2E > CANN (large shape) | Yes (1.07x - 1.80x) |
| Largest hotspot identified | Yes (B2 Sinkhorn, 28.5%) |
| B2 optimization blocked | Yes (compiler 2D-to-1D T.copy limitation) |
| All parameters routed correctly | Yes (verified by distinct-eps test) |

Optimization stopped: B2 Sinkhorn is the primary bottleneck but cannot be further optimized due to the Ascend backend 2D-to-1D T.copy slice limitation. Other kernels are reasonably balanced (9-21% each).
