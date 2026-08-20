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
- h_blk selected as largest divisor of h from [4096, 3584, 3072, 2560, 2048, 1024, 512]
  - h=2560 -> h_blk=2560 (no padding, 1 tile)
  - h=7168 -> h_blk=3584 (no padding, 2 tiles)
- Eliminates 60% wasted computation on padded elements (h=2560: was 4096, now 2560)
- out0~3 merged into single reusable out_fp32 (UB footprint -24KB)
- T.unroll(4) replaces manual 4x code duplication
- F.pad replaces manual _pad_3d/_pad_2d_1d functions

## 4. Final Performance (do_bench, warmup=20, rep=100)

| n | h | hc | h_blk | Kernel-only | E2E | PyTorch (CANN) | Kernel speedup | E2E speedup |
|---|---|---|-------|-------------|-----|----------------|----------------|-------------|
| 512 | 2560 | 4 | 2560 | 0.23 ms | 0.34 ms | 0.25 ms | 1.08x | 0.75x |
| 4096 | 2560 | 4 | 2560 | 0.42 ms | 0.43 ms | 2.24 ms | 5.39x | 5.26x |
| 4096 | 7168 | 4 | 3584 | 0.84 ms | 0.86 ms | 6.08 ms | 7.25x | 7.06x |

> Adaptive h_blk eliminates host-side padding for common shapes (h=2560, h=7168),
> so E2E ≈ kernel-only. n=512 E2E includes ~0.10ms Python dispatch overhead.

## 5. h_blk Sweep (V7 kernel, kernel-only)

| h_blk | n=512, h=2560 | n=4096, h=2560 | n=4096, h=7168 |
|-------|---------------|-----------------|------------------|
| 512 | 1.06x | 2.26x | 2.40x |
| 1024 | 1.04x | 3.08x | 3.93x |
| 2048 | 1.09x | 3.49x | 5.16x |
| 2560 | **1.12x** | **5.44x** | 6.05x |
| 3072 | 1.10x | 5.13x | 5.59x |
| 3584 | 1.06x | 4.77x | **7.26x** |
| 4096 | 1.06x | 4.49x | 6.75x |

Adaptive selection (bold) picks the largest divisor of h, eliminating padding waste.
h=2560 -> h_blk=2560 (no-pad); h=7168 -> h_blk=3584 (no-pad).

## 6. Pipeline Ablation (n=4096, h=7168, kernel-only)

| Schedule | Latency | vs CANN | vs serial |
|----------|---------|---------|-----------|
| T.serial | 1.216 ms | 4.99x | baseline |
| T.Pipelined(stage=2) | 1.180 ms | 5.14x | +3.0% |
| T.Pipelined(stage=3) | 1.172 ms | 5.17x | +3.6% |

stage=2 captures most pipeline benefit. stage=3 adds only 0.6% — not worth the extra UB pressure.

## 7. msprof Analysis (n=4096, h=7168)

| Metric | Value |
|--------|-------|
| Task Duration | 1,194 us |
| Block count | 6,144 (msprof counts 3 hardware cores per AI Core: 1 Cube + 2 Vector; logical blocks = n/2 = 2048) |
| Vector compute | 21,701 us (1818%) |
| MTE2 (GM->UB load) | 8,348 us (699%) |
| MTE3 (UB->GM store) | 9,980 us (836%) |
| Scalar | 6,921 us (580%) |
| Parallelism | 37.6x |
| Effective BW | 449 GB/s |

> Percentages = per-core accumulated time / Task Duration. Values >100% mean
> multiple cores are active concurrently (e.g. Vector compute 1818% ≈ 18 vector
> units working in parallel across all blocks).

### Bottleneck: Compute-bound

```
Vector compute (1818%) > MTE total (1535%)
```

The current implementation is primarily Vector-compute constrained according to the measured profiler breakdown (Vector 1818% > MTE 1535%). T.Pipelined effectively overlaps compute with memory operations (37.6x parallelism). Remaining optimization opportunities are mainly in data-movement/layout efficiency; see the reviewer discussion on 2D UB contiguous copy in the PR thread.

## 8. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 11/11 passed |
| Tolerance | rtol=1e-2, atol=0.2 |
| Max diff | 0.0625 |
| Source of diff | BF16 output rounding + accumulation order |

## 9. Stop Condition

| Condition | Status |
|-----------|--------|
| Kernel > CANN | Yes (1.08x - 7.25x) |
| E2E > CANN (large shape) | Yes (5.26x - 7.06x) |
| Compute-bound confirmed | Yes (Vector 1818% > MTE 1535%) |
| Pipeline optimized | Yes (stage=2 optimal) |
| h_blk optimized | Yes (adaptive: largest divisor of h) |
| Effective BW high | Yes (449 GB/s) |

Optimization stopped: adaptive h_blk eliminates padding waste for common shapes,
E2E ≈ kernel-only. Further gains require 2D UB contiguous copy (blocked by compiler)
or tail-padding without host allocation (T.copy lacks pad_value parameter).
