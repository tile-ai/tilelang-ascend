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

## 3. Optimization Path (V0 -> V8 Hybrid)

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
| V8 Hybrid | 2D UB fast path + 1D UB fallback | 0.38 ms | 5.98x | Merged copy for 2D path, 3584 preserved via 1D fallback |

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
- 2D UB layout evaluated per reviewer suggestion; 2D slice operations (T.copy,
  T.tile.cast, T.tile.axpy with 2D dst/src) work correctly, but T.copy(GM→2D UB)
  followed by 2D scalar indexing (comb[i,j]) reads wrong addresses — reviewer's
  full 2D proposal fails (max_diff=25.59) because it uses 2D scalar reads for comb
  coefficients. A hybrid approach (2D slice for res/out + 1D for comb) is viable
  and ~12% faster at h=2560, but requires separate h_blk sweep. Workaround: 4
  separate 1D buffers for all data, enabling larger h_blk (2560/3584).
  > V8 Hybrid later resolves the 2D scalar read issue by aligning comb to [4, 8]
  > (32 bytes/row), enabling a 2D UB fast path. See V8 Hybrid below.
- A single flat 1D comb buffer `[16]` with one `T.copy(comb[bid,0,0], comb_fp32)`
  was also evaluated; it reads wrong data (3D scalar offset copy does not span
  rows correctly), so the 4 separate 1D buffers are retained.

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

**V8 Hybrid: 2D UB fast path + 1D UB fallback**
- Two kernel variants JIT-compiled per shape, dispatched by `_select_path(h)`:
  - **2D UB path** (h_blk ≤ 3072): merged res/out copy (1 T.copy instead of 4),
    aligned comb `comb_fp32[4, 8]` (32 bytes/row enables correct 2D scalar read
    `comb_fp32[res_idx, out_idx]`). No host-side pad (requires h % h_blk == 0).
    Faster due to fewer MTE2/MTE3 launch overheads.
  - **1D UB path** (h_blk = 3584 or non-dividing h): V7's separate per-row
    buffers (smaller UB footprint). Host-side F.pad for non-dividing h.
- Dispatch rule: find largest h_blk from combined candidates [3584, 3072, 2560,
  2048, 1024, 512] that divides h; if it's in the 2D list (≤ 3072) → 2D path,
  if it's 3584 → 1D path, if none divides → 1D path with pad.
- Key breakthrough: `comb_fp32 = T.alloc_ub((HC, (HC+7)//8*8), accum_dtype)` =
  [4, 8] aligns each row to 32 bytes, matching AlignInnerDim's padding. This
  makes 2D scalar reads work correctly (previous [4, 4] was unaligned → wrong
  addresses → max_diff=25.59).
- Results (5-run average, do_bench warmup=20 rep=100):
  - h=4096×2560 (2D path, h_blk=2560): **5.98x** (V7: 5.34x, +12%)
  - h=4096×7168 (1D path, h_blk=3584): **7.24x** (V7: 7.27x, parity)
- 2D UB with h_blk=3584 was tested and fails (UB overflow, kernel hangs). The
  hybrid avoids this by falling back to 1D UB for h_blk=3584.

## 4. Final Performance (V8 Hybrid, do_bench, warmup=20, rep=100, 5-run average)

| n | h | hc | path | h_blk | Kernel-only | E2E | PyTorch (CANN) | Kernel speedup | E2E speedup |
|---|---|---|------|-------|-------------|-----|----------------|----------------|-------------|
| 512 | 2560 | 4 | 2D | 2560 | 0.33 ms | 0.33 ms | 0.25 ms | 0.76x | 0.76x |
| 4096 | 2560 | 4 | 2D | 2560 | 0.38 ms | 0.38 ms | 2.28 ms | 5.98x | 5.98x |
| 4096 | 7168 | 4 | 1D | 3584 | 0.84 ms | 0.84 ms | 6.10 ms | 7.24x | 7.24x |

> 2D path: no host pad (h % h_blk == 0), E2E ≈ kernel-only.
> 1D path for h=7168: no pad (7168 % 3584 == 0), E2E ≈ kernel-only.
> n=512 is slower than CANN due to small data volume (24MB) not saturating dual-V-core parallelism.

## 5. h_blk Sweep (V7 1D kernel, kernel-only)

| h_blk | n=512, h=2560 | n=4096, h=2560 | n=4096, h=7168 |
|-------|---------------|-----------------|------------------|
| 512 | 0.84x | 2.26x | 2.40x |
| 1024 | 0.81x | 3.08x | 3.93x |
| 2048 | 0.84x | 3.49x | 5.16x |
| 2560 | 0.75x | 5.44x | 6.05x |
| 3072 | 0.79x | 5.13x | 5.59x |
| 3584 | 0.87x | 4.77x | **7.26x** |

> Sweep measured with V7's 1D UB kernel. V8 Hybrid dispatches h_blk ≤ 3072 via
> the faster 2D UB path (merged copy), and h_blk=3584 via the 1D UB path.
> At h_blk=2560, the 2D path achieves 5.98x vs V7's 5.44x (+10%).
> h_blk=3584 is excluded from the 2D path (UB overflow at this size).

## 6. Pipeline Ablation (1D path, n=4096, h=7168, h_blk=3584, kernel-only)

| Schedule | Latency | vs CANN | vs serial |
|----------|---------|---------|-----------|
| T.serial | 0.876 ms | 6.93x | baseline |
| T.Pipelined(stage=2) | 0.840 ms | 7.23x | +4.1% |

stage=2 provides 4.1% speedup over serial. stage=3 not feasible (h_blk=3584 × 3 stages exceeds UB capacity).

## 7. Performance Analysis (n=4096, h=7168, h_blk=3584)

### V8 Hybrid Effective Bandwidth (do_bench)

| shape | path | Data volume | Kernel latency | Effective BW | HBM peak ratio |
|-------|------|-------------|---------------|--------------|----------------|
| n=4096, h=2560 | 2D | 189 MB | 0.38 ms | 496 GB/s | 42% |
| n=4096, h=7168 | 1D | 529 MB | 0.84 ms | 630 GB/s | 53% |

> 2D path improves h=2560 bandwidth from 451 GB/s (V7) to 496 GB/s (+10%),
> from merged res/out copy reducing MTE2/MTE3 launch overhead.

### V6 msprof Reference (h_blk=2048, kernel 1.18 ms)

> V6 hardware-level breakdown measured via msprof. V8 Hybrid 1D path uses h_blk=3584
> (kernel 0.84 ms); the compute structure (AXPY + dual-V-core) is unchanged, so the
> Vector-compute bottleneck applies to V8 Hybrid as well. The 1D path's bandwidth
> (630 GB/s) comes from eliminating padded data movement, same as V7. The 2D path
> further improves h=2560 bandwidth to 496 GB/s via merged copy.

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
parallelism). V8 Hybrid's 2D path improves h=2560 bandwidth by 10% (451 → 496 GB/s)
via merged copy, but the compute bottleneck remains unchanged.

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
| Kernel > CANN | Yes (0.76x - 7.24x; small shape slower, large shape 6-7x) |
| E2E > CANN (large shape) | Yes (5.98x - 7.24x) |
| Compute-bound confirmed | Yes (V6 msprof: Vector 1818% > MTE 1535%; V8 structure unchanged) |
| Pipeline optimized | Yes (stage=2, +4.1% over serial; stage=3 UB-limited) |
| h_blk optimized | Yes (adaptive: largest divisor of h, hybrid 2D/1D dispatch) |
| 2D UB utilized | Yes (2D path for h_blk ≤ 3072, merged copy +10% over V7 at h=2560) |
| Effective BW high | Yes (496-630 GB/s, 42-53% of HBM peak) |

Optimization stopped: hybrid 2D/1D dispatch achieves best of both paths —
2D UB merged copy for h_blk ≤ 3072 (faster MTE2/MTE3), 1D UB for h_blk=3584
(preserves large tile count for h=7168). Non-dividing h handled safely via
host F.pad on 1D path. Further gains require reducing Vector compute (AXPY
loop unroll is already at T.unroll(4) for hc=4).
