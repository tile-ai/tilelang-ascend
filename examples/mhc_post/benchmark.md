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

## 3. Optimization Path (V0 -> V6)

| Version | Change | Kernel-only (n=4096, h=2560) | vs CANN | Key Breakthrough |
|---------|--------|------------------------|---------|------------------|
| V0 | Cube dual-kernel | 4.63 ms | 0.49x | Baseline |
| V1 | AIV single V-core | 13.37 ms | 0.17x | Pure Vector |
| V2 | Dual V-core + h_blk=256 | 2.56 ms | 0.89x | Utilize both V-cores |
| V3 | comb resident in UB + skip pad | 1.70 ms | 1.32x | Eliminate redundant GM reads |
| V4 | AXPY + h_blk=2048 | 0.70 ms | 3.17x | Structural refactor |
| V5 | T.Pipelined(stage=2) | 0.67 ms | 3.37x | Pipeline overlap |
| V6 | Cast fusion + kernel cache | 0.65 ms | 3.46x | Remove host overhead |

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
  correctly, but 2D UB row/scalar slices used as src operands in T.tile.axpy
  produce incorrect results (max_diff > 12), and 2D UB row slice as T.copy
  source triggers AICore exception 507015. Workaround: 4 separate 1D buffers

**V4: AXPY (biggest breakthrough)**
- Replaced `broadcast + mul + reduce_sum` with `T.tile.mul(dst, src, scalar)` + `T.tile.axpy(dst, src, scalar)`
- Eliminated 7 large 2D FP32 UB buffers (56KB -> 16KB)
- Smaller UB footprint allows h_blk=512 -> 2048, reducing loop iterations from 5 to 2

**V6: Cast fusion**
- Removed host-side FP32->BF16 cast (separate kernel launch)
- Kernel directly receives FP32, no BF16 quantization
- Golden reference synchronized (removed ref's bfloat16() quantization)
- max_diff improved: 0.125 -> 0.0625

## 4. Final Performance (E2E, do_bench, warmup=20, rep=100)

| n | h | hc | TileLang (AIV) | PyTorch (CANN) | Speedup |
|---|---|---|-----------------|----------------|---------|
| 512 | 2560 | 4 | 0.40 ms | 0.25 ms | 0.63x |
| 4096 | 2560 | 4 | 1.00 ms | 2.27 ms | 2.28x |
| 4096 | 7168 | 4 | 1.96 ms | 6.08 ms | 3.10x |

> **E2E vs kernel-only**: §3/§5/§6 report **kernel-only** latency (pure NPU
> execution); this section reports **E2E** latency including host-side tensor
> preprocessing (`_pad_3d` + `comb.mT.contiguous()`). For n=4096,h=2560 the gap
> is 0.65ms (kernel-only) → 1.00ms (E2E), i.e. ~0.35ms host overhead from the
> 134MB pad + transpose-contiguous. This host preprocessing is the main
> remaining E2E bottleneck (see §9).

Note: n=512 E2E includes ~0.17ms Python dispatch overhead. Kernel-only latency is 0.23ms (1.08x CANN).

## 5. h_blk Sweep (AXPY version, kernel-only)

| h_blk | n=512 | n=4096, h=2560 | n=4096, h=7168 |
|-------|-------|-----------------|------------------|
| 256 | 1.09x | 1.29x | 1.32x |
| 512 | 1.05x | 2.19x | 2.32x |
| 1024 | 1.12x | 2.90x | 3.81x |
| 2048 | 1.11x | 3.17x | 5.03x |

h_blk=2048 is optimal across all shapes. AXPY's small UB footprint enables large tile size.

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
| Block count | 6,144 (msprof-reported; logical dual-V-core blocks = n/2 = 2048) |
| Vector compute | 21,701 us (1818%) |
| MTE2 (GM->UB load) | 8,348 us (699%) |
| MTE3 (UB->GM store) | 9,980 us (836%) |
| Scalar | 6,921 us (580%) |
| Parallelism | 37.6x |
| Effective BW | 443 GB/s |

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
| Kernel > CANN | Yes (1.08x - 5.03x) |
| E2E > CANN (large shape) | Yes (2.28x - 3.10x) |
| Compute-bound confirmed | Yes (Vector 1818% > MTE 1535%) |
| Pipeline optimized | Yes (stage=2 optimal) |
| h_blk optimized | Yes (2048 via sweep) |
| Effective BW high | Yes (443 GB/s) |

Optimization stopped: the kernel is primarily Vector-compute constrained per profiler breakdown. Further arithmetic-side optimization has diminishing returns; remaining opportunities are mainly in data-movement/layout efficiency.
