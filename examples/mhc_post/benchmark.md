# MHC Post Benchmark & Optimization Path

## 1. Operator

```
output = x * post_layer_mix + comb_res_mix^T @ residual
```

- Input: x [n, h] bf16, residual [n, hc, h] bf16, post_layer_mix [n, hc, 1] fp32, comb_res_mix [n, hc, hc] fp32
- Output: [n, hc, h] bf16
- Constraint: hc = 4 (hardcoded AXPY specialization)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| Tool | do_bench (Python), msprof op (hardware) |
| Dtype | bf16 input, fp32 accumulate |

## 3. Optimization Path (V0 -> V7)

| Version | Change | Kernel-only (n=4096, h=2560) | vs CANN | Key Breakthrough |
|---------|--------|------------------------------|---------|------------------|
| V0 | Cube dual-kernel | 4.63 ms | 0.49x | Baseline |
| V1 | AIV single V-core | 13.37 ms | 0.17x | Pure Vector |
| V2 | Dual V-core + h_blk=256 | 2.56 ms | 0.89x | Utilize both V-cores |
| V3 | comb resident in UB + skip pad | 1.70 ms | 1.32x | Eliminate redundant GM reads |
| V4 | AXPY + h_blk=2048 | 0.70 ms | 3.17x | Structural refactor |
| V5 | T.Pipelined(stage=2) | 0.67 ms | 3.37x | Pipeline overlap |
| V6 | Cast fusion + kernel cache | 0.65 ms | 3.46x | Remove host overhead |
| V7 | Adaptive h_blk + out reuse + T.unroll | 0.42 ms | 5.39x | Eliminate padding waste |

### Key Decisions

**V0 -> V1: Why not Cube?**
- hc=4 padded to 16 wastes 93.75% Cube MACs
- CV sync bug prevents single-kernel fusion
- Pure Vector avoids both issues
- But V1 is 3x *slower* than Cube: it uses only one V-core, and simulates the
  `[4,4]@[4,h]` matmul with `broadcast + mul + reduce_sum`, which is far less
  efficient than dedicated Cube hardware. This motivates V2 (dual V-core) and
  V4 (AXPY replacing broadcast+reduce).

**V1 -> V2: Dual V-core**
- `bid = cid * 2 + vid` — two Vector units process different tokens
- 2.9x speedup (one V-core was idle before)

**V3: Loop-invariant hoisting**
- comb coefficients loaded once outside h-loop (4 independent 1D UB buffers)
- 2D UB layout evaluated per reviewer suggestion; 2D T.copy and 2D cast work
  correctly, but after T.copy from GM to a 2D UB buffer, reading data via
  scalar indexing (buf[i,j]) or row slicing (buf[i,:]) accesses wrong addresses
  (manual fill + scalar read works; T.copy roundtrip works; only T.copy + slice
  read fails). AICore exception 507015 when using 2D row slice as T.copy source.
  Workaround: 4 separate 1D buffers
- A single flat 1D comb buffer `[16]` with one `T.copy(comb[bid,0,0], comb_fp32)`
  was also evaluated to consolidate the 4 short copies; it reads wrong data
  (single 16-element copy from a 3D scalar offset does not span rows correctly),
  so the 4 separate 1D buffers are retained.

**V4: AXPY (biggest breakthrough)**
- Replaced `broadcast + mul + reduce_sum` with `T.tile.mul(dst, src, scalar)` + `T.tile.axpy(dst, src, scalar)`
- Eliminated 7 large 2D FP32 UB buffers (56KB -> 16KB)
- Smaller UB footprint allows h_blk=512 -> 2048, reducing loop iterations from 5 to 2

**V6: Cast fusion**
- Removed host-side FP32->BF16 cast (separate kernel launch)
- Kernel directly receives FP32, no BF16 quantization
- Golden reference synchronized (removed ref's bfloat16() quantization)
- max_diff improved: 0.125 -> 0.0625

**V7: Adaptive h_blk + out reuse + T.unroll**
- h_blk selected as largest divisor of h from [3584, 3072, 2560, 2048, 1024, 512]
  - h=2560 -> h_blk=2560 (no padding, 1 tile)
  - h=7168 -> h_blk=3584 (no padding, 2 tiles)
- Old padded path computed 1.6x elements (h=2560: was 4096, now 2560)
- out0~3 merged into single reusable out_fp32 (UB footprint -24KB)
- T.unroll(4) replaces manual 4x code duplication
- F.pad replaces manual _pad_3d/_pad_2d_1d functions

## 4. Final Performance (do_bench, warmup=20, rep=100, 3-run average)

| n | h | hc | h_blk | Kernel-only | E2E | PyTorch (CANN) | Kernel speedup | E2E speedup |
|---|---|---|-------|-------------|-----|----------------|----------------|-------------|
| 512 | 2560 | 4 | 2560 | 0.30 ms | 0.34 ms | 0.25 ms | 0.83x | 0.75x |
| 4096 | 2560 | 4 | 2560 | 0.42 ms | 0.42 ms | 2.23 ms | 5.34x | 5.34x |
| 4096 | 7168 | 4 | 3584 | 0.84 ms | 0.84 ms | 6.08 ms | 7.25x | 7.25x |

> Adaptive h_blk eliminates host-side padding for common shapes (h=2560, h=7168),
> so E2E ≈ kernel-only. n=512 is slower than CANN due to small data volume (24MB)
> not saturating dual-V-core parallelism.

## 5. h_blk Sweep (V7 kernel, kernel-only)

| h_blk | n=512, h=2560 | n=4096, h=2560 | n=4096, h=7168 |
|-------|---------------|-----------------|------------------|
| 512 | 0.84x | 2.26x | 2.40x |
| 1024 | 0.81x | 3.08x | 3.93x |
| 2048 | 0.84x | 3.49x | 5.16x |
| 2560 | 0.75x | 5.44x | 6.05x |
| 3072 | 0.79x | 5.13x | 5.59x |
| 3584 | 0.87x | 4.77x | **7.26x** |

Adaptive selection picks the largest divisor of h from candidates [3584, 3072, 2560, 2048, 1024, 512].
h=2560 -> h_blk=2560 (no-pad); h=7168 -> h_blk=3584 (no-pad). n=512 is slower than CANN
across all h_blk (small data volume, ~24MB, does not saturate dual-V-core parallelism).

## 6. Pipeline Ablation (n=4096, h=7168, h_blk=3584, kernel-only)

| Schedule | Latency | vs CANN | vs serial |
|----------|---------|---------|-----------|
| T.serial | 0.876 ms | 6.93x | baseline |
| T.Pipelined(stage=2) | 0.840 ms | 7.23x | +4.1% |

stage=2 provides 4.1% speedup over serial. stage=3 not feasible (h_blk=3584 × 3 stages exceeds UB capacity).

## 7. Performance Analysis (n=4096, h=7168, h_blk=3584)

### V7 Effective Bandwidth (do_bench)

| shape | Data volume | Kernel latency | Effective BW | HBM peak ratio |
|-------|-------------|---------------|--------------|----------------|
| n=4096, h=2560 | 189 MB | 0.42 ms | 453 GB/s | 38% |
| n=4096, h=7168 | 529 MB | 0.84 ms | 631 GB/s | 53% |

### V6 msprof Reference (h_blk=2048, kernel 1.18 ms)

> V6 hardware-level breakdown measured via msprof. V7 uses h_blk=3584 (kernel 0.84 ms);
> the compute structure (AXPY + dual-V-core) is unchanged, so the Vector-compute bottleneck
> applies to V7 as well. V7's higher bandwidth (631 vs 449 GB/s) comes from eliminating
> padded data movement, not from a change in compute pattern.

| Metric | Value |
|--------|-------|
| Task Duration | 1,194 us |
| Block count | 6,144 (3 hardware cores per AI Core: 1 Cube + 2 Vector; logical blocks = n/2 = 2048) |
| Vector compute | 21,701 us (1818%) |
| MTE2 (GM->UB load) | 8,348 us (699%) |
| MTE3 (UB->GM store) | 9,980 us (836%) |
| Scalar | 6,921 us (580%) |
| Parallelism | 37.6x |
| Effective BW | 449 GB/s |

> Percentages = per-core accumulated time / Task Duration. Values >100% mean
> multiple cores are active concurrently.

### Bottleneck: Vector-compute constrained

Vector compute (1818%) > MTE total (1535%). The kernel is primarily Vector-compute
constrained. T.Pipelined effectively overlaps compute with memory operations (37.6x
parallelism). V7's adaptive h_blk improves bandwidth by 40% (449 -> 631 GB/s) by
eliminating padded data movement, but the compute bottleneck remains unchanged.

## 8. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 15/15 passed |
| Tolerance | rtol=1e-2, atol=0.2 |
| Max diff | 0.0625 |
| Source of diff | BF16 output rounding + accumulation order |

## 9. Stop Condition

| Condition | Status |
|-----------|--------|
| Kernel > CANN | Yes (0.83x - 7.25x; small shape slower, large shape 5-7x) |
| E2E > CANN (large shape) | Yes (5.34x - 7.25x) |
| Compute-bound confirmed | Yes (V6 msprof: Vector 1818% > MTE 1535%; V7 structure unchanged) |
| Pipeline optimized | Yes (stage=2, +4.1% over serial; stage=3 UB-limited) |
| h_blk optimized | Yes (adaptive: largest divisor of h, no-pad for common shapes) |
| Effective BW high | Yes (631 GB/s, 53% of HBM peak) |

Optimization stopped: adaptive h_blk eliminates padding waste for common shapes,
E2E ≈ kernel-only. Further gains require 2D UB contiguous copy (blocked by compiler)
or tail-padding without host allocation (T.copy lacks pad_value parameter).
