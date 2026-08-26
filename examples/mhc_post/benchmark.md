# MHC Post Benchmark & Optimization Path

## 1. Operator

```
output = x * post_layer_mix + comb_res_mix^T @ residual
```

- Input: x [n, h] bf16, residual [n, hc, h] bf16, post_layer_mix [n, hc, 1] fp32, comb_res_mix [n, hc, hc] fp32
- Output: [n, hc, h] bf16
- Constraint: 1 <= hc <= 8 (JIT parameter, tested range)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| Tool | do_bench (Python), msprof op (hardware) |
| Dtype | bf16 input, fp32 accumulate |

## 3. Optimization Path (V0 -> V10 Generic HC + Tail Mask)

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
| V9 Unified | single kernel: 2D res + 1D out | 0.38 ms | 5.88x | One kernel covers all h_blk incl 3584 |
| V10 Generic | remove host pad + generic hc + merged bf16 store | 0.38 ms | 5.98x | In-kernel tail (pad_value + TAIL_MASK), hc 1-8, 2D bf16 merged MTE3 |

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
  - **1D UB path** (h_blk = 3584 or non-dividing h): separate per-row res/out
    buffers (smaller UB footprint) + aligned comb [4,8] (single copy, same as 2D
    path). Host-side F.pad for non-dividing h.
- Dispatch rule: find largest h_blk from combined candidates [3584, 3072, 2560,
  2048, 1024, 512] that divides h; if it's in the 2D list (≤ 3072) → 2D path,
  if it's 3584 → 1D path, if none divides → 1D path with pad.
- Key breakthrough: `comb_fp32 = T.alloc_ub((HC, (HC+7)//8*8), accum_dtype)` =
  [4, 8] aligns each row to 32 bytes, matching AlignInnerDim's padding. This
  makes 2D scalar reads work correctly (previous [4, 4] was unaligned → wrong
  addresses → max_diff=25.59).
- Results (5-run average, do_bench warmup=20 rep=100):
  - h=4096×2560 (2D path, h_blk=2560): **5.98x** (V7: 5.34x, +12%)
  - h=4096×7168 (1D path, h_blk=3584): **7.40x** (V7: 7.27x, +2%)
- 2D UB with h_blk=3584 was tested and fails (UB overflow, kernel hangs). The
  hybrid avoids this by falling back to 1D UB for h_blk=3584.

**V9 Unified: single kernel (2D res + 1D out)**
- Merged V8's two paths into one kernel: 2D res (1 T.copy for all 4 rows) +
  1D out (streaming write-back, out buffer reused across out_idx).
- 2D res keeps MTE efficiency (1 copy vs 4); 1D out keeps UB footprint low
  (~126KB) so h_blk=3584 fits — V8's 2D path overflowed because it kept a 2D
  `out_fp32`/`out_bf16` (`[4, h_blk]`, ~189KB footprint).
- Removes `_select_path` dispatch; host only picks the largest dividing h_blk
  and pads otherwise. Code: 2 kernels + dispatch → 1 kernel (~-97 lines).
- Correctness 15/15. Perf: h=7168 7.42x (parity/slightly faster), h=2560 5.88x.
- Engineering trade-off vs V8: h=2560 is ~1.7% slower (5.88x vs 5.98x) but the
  single-kernel structure removes one kernel variant and the dispatch logic
  (~-97 lines); h=7168 is slightly faster. Accepted for the code simplification.
- T.Persistent (reviewer-suggested) was evaluated but NOT adopted: single-shape
  runs show +4%~17%, but full-shape testing triggers vector-core exceptions
  (unaligned UUB addresses, err 0x10) on small-n / padded / single-tile shapes
  and multi-shape runs. Left as a known future direction pending backend fix.
- AUTO_SYNC=False manual pipeline was also evaluated but NOT adopted. As a pure
  AIV kernel, setting `AUTO_SYNC=False` alone gives +13%~21% (7168: 7.44x→8.98x)
  because `AUTO_SYNC=True` inserts redundant sync that suppresses MTE2/V/MTE3
  overlap. Realizing that gain requires hand-written `set_flag`/`wait_flag`, which
  hit three blockers:
  1. `set_flag` event id must be a compile-time constant. Passing the `T.serial`
     loop var `i % 2` makes codegen drop the `% 2`, producing a runtime event id
     (`i` = 0,1,2,...) that never pairs up → deadlock. Fix: use a `stages` variable
     + `T.serial(0, h_num - 1)` + an explicit epilogue so the loop unrolls to
     constant 0/1.
  2. `T.tile.mul`'s scalar read (`post_fp32[out_idx]`) emits `PipeBarrier<PIPE_ALL>`,
     and the global barrier forms a deadlock cycle with the flag ring. Fix: replace
     `mul` with `T.tile.fill(dst, 0.0)` + `T.tile.axpy(dst, src, scalar)`, whose
     scalar read inlines without a barrier.
  3. A full pipeline needs a double-buffered output (`out_bf16[2, 4, h_blk]`) to
     avoid tile-to-tile races, but that overflows UB (~210KB > 192KB) at
     h_blk=3584; a single buffer (`[4, h_blk]`) races between tile i's MTE3 write
     and tile i+1's V cast. Blockers 1 and 2 are solvable; blocker 3 is a hard UB
     constraint at h_blk=3584. Left as a known direction.

**V10 Generic: remove host pad + generic hc + merged bf16 store**
- Three changes responding to reviewer feedback:
  1. **Remove host F.pad**: tail tile merged into the same `T.Pipelined` loop
     (`total_tiles = ceildiv(h, h_blk)`). Every tile copy passes `pad_value=0.0`
     (no-op for full tiles, zero-fills the gap on the tail tile).
     `TL_ASCEND_TAIL_MASK` pass enabled to rewrite vector ops on tail tiles to
     compute only the valid region. No host-side padding needed.
  2. **Generic hc (1-8)**: `hc` is a JIT parameter, no longer hardcoded to 4.
     `comb_row_stride = (hc + 7) // 8 * 8` aligns each row to 32B for correct 2D
     scalar read. `_max_h_blk(hc)` bounds h_blk from UB budget per hc.
  3. **2D bf16 merged store**: `out_bf16 = T.alloc_ub((hc, h_blk), dtype)`, cast
     per row into `out_bf16[out_idx, :]`, then 1 merged `T.copy(out_bf16, output)`
     instead of hc separate MTE3 stores.
- The separate tail block (V9's `if tail > 0` after the pipeline loop) conflicted
  with `T.Pipelined` double-buffering and the tail mask pass's buffer tracking.
  Merging the tail into the main loop resolved this.
- Correctness 22/22 (15 hc=4 cases + 7 hc=1/2/3/8 cases covering tail path).
- Perf: parity with V9 at h=2560 (5.98x); h=7168 improved 0.82->0.75 ms (7.42x
  -> 8.10x), likely from simpler loop structure. 5-run average.

## 4. Final Performance (V10 Generic, do_bench, warmup=20, rep=100, 5-run average)

| n | h | hc | h_blk | Kernel-only | E2E | PyTorch (CANN) | Kernel speedup | E2E speedup |
|---|---|---|-------|-------------|-----|----------------|----------------|-------------|
| 512 | 2560 | 4 | 2560 | 0.34 ms | 0.34 ms | 0.25 ms | 0.74x | 0.74x |
| 4096 | 2560 | 4 | 2560 | 0.38 ms | 0.38 ms | 2.25 ms | 5.98x | 5.98x |
| 4096 | 7168 | 4 | 3584 | 0.75 ms | 0.75 ms | 6.07 ms | 8.10x | 8.10x |

> Single unified kernel. h=2560/7168 divide h_blk exactly (no tail), so
> E2E ≈ kernel-only. V10 removes host pad and enables TL_ASCEND_TAIL_MASK;
> performance is at parity with V9 for h=2560, and improved for h=7168
> (0.82→0.75 ms, simpler loop structure).
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

> Sweep measured with V7's 1D UB kernel. V9 Unified uses a single kernel
> (2D res merged copy + 1D out) covering all h_blk including 3584.
> At h_blk=2560, unified achieves 5.88x vs V7's 5.44x (+8%).

## 6. Pipeline Ablation (n=4096, h=7168, h_blk=3584, kernel-only)

| Schedule | Latency | vs CANN | vs serial |
|----------|---------|---------|-----------|
| T.serial | 0.876 ms | 6.93x | baseline |
| T.Pipelined(stage=2) | 0.840 ms | 7.23x | +4.1% |

stage=2 provides 4.1% speedup over serial. stage=3 not feasible (h_blk=3584 × 3 stages exceeds UB capacity).

## 7. Performance Analysis (n=4096, h=7168, h_blk=3584)

### V10 Generic Effective Bandwidth (do_bench)

| shape | Data volume | Kernel latency | Effective BW | HBM peak ratio |
|-------|-------------|---------------|--------------|----------------|
| n=4096, h=2560 | 189 MB | 0.38 ms | 497 GB/s | 41% |
| n=4096, h=7168 | 529 MB | 0.75 ms | 705 GB/s | 59% |

> Unified 2D-res merged copy improves h=2560 bandwidth from 451 GB/s (V7) to
> 497 GB/s by reducing MTE2/MTE3 launch overhead. V10's unified pipeline
> (merging tail into main loop) further improves h=7168 from 647 GB/s (V9)
> to 705 GB/s.

### V6 msprof Reference (h_blk=2048, kernel 1.18 ms)

> V6 hardware-level breakdown measured via msprof (h_blk=2048, kernel 1.18 ms).
> V10 uses h_blk=3584 (kernel 0.75 ms) with a different data-movement layout
> (2D-res merged copy + 2D bf16 merged store); a fresh V10 msprof profile has
> not been collected, so the V6 result is retained as historical reference
> rather than a definitive V10 bottleneck characterization. The h=7168
> bandwidth (705 GB/s) comes from eliminating padded data movement, same as
> V7; the merged 2D-res copy improves h=2560 bandwidth to 497 GB/s.

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

### Bottleneck: Vector-compute constrained (V6 historical)

V6 msprof (h_blk=2048) shows Vector compute (1818%) > MTE total (1535%), i.e.
the earlier AXPY implementation was primarily Vector-compute constrained, with
T.Pipelined overlapping compute and memory (37.6x parallelism). V10 retains the
same arithmetic structure while improving the data-movement layout (2D-res
merged copy + 2D bf16 merged store). V10 itself has not been re-profiled, so
the Vector-compute bottleneck is treated as historical evidence, not a
definitive V10 characterization.

## 8. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 22/22 passed |
| Tolerance | rtol=1e-2, atol=0.2 |
| Max diff | 0.0625 |
| Source of diff | BF16 output rounding + accumulation order |

> 15 hc=4 cases (various h, including tail path) + 7 hc=1/2/3/8 cases
> (covering hc<4, hc>4, non-8-aligned comb_row_stride, and tail path).

## 9. Stop Condition

| Condition | Status |
|-----------|--------|
| Kernel > CANN | Yes (0.74x - 8.10x; small shape slower, large shape 6-8x) |
| E2E > CANN (large shape) | Yes (5.98x - 8.10x) |
| Compute-bound evidence | V6 msprof (historical): Vector 1818% > MTE 1535%; V10 not re-profiled |
| Pipeline optimized | Yes (stage=2, +4.1% over serial; stage=3 UB-limited) |
| h_blk optimized | Yes (adaptive: largest divisor of h, single kernel) |
| 2D res merged copy | Yes (2D res merged copy +10% over V7 at h=2560) |
| 2D bf16 merged store | Yes (1 merged MTE3 instead of hc stores) |
| In-kernel tail | Yes (pad_value + TL_ASCEND_TAIL_MASK, no host pad) |
| Generic hc | Yes (hc 1-8, JIT parameter) |
| Effective BW high | Yes (496-647 GB/s, 42-54% of HBM peak) |

Optimization stopped: a single unified kernel (2D res merged load + 2D bf16
merged store) covers all h_blk including 3584 and all hc 1-8. Non-dividing h
handled in-kernel via pad_value + TL_ASCEND_TAIL_MASK (no host-side padding).
Further gains require reducing Vector compute (AXPY loop unroll is already at
T.unroll(hc) for the tested hc range).
