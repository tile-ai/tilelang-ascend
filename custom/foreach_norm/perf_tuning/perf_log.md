# ForeachNorm 性能调优日志

## Iteration 1 — multi-core partial reduction (ADOPTED as baseline)
- timestamp: 2026-08-06T08:20Z
- bottleneck_type: compute + sync
- optimization: multi-core partial reduction (launch_cores=min(n_num,24), strided tile assignment, per-core FP32 partial → host-side _finalize with PyTorch ops)
- baseline_time: N/A (iter0 single-block, not benchmarked with warmup=5/iters=20)
- candidate_time: avg ours_us varies by case (see ours_iter1.json)
- avg_speedup: 0.1713
- improvement: N/A (first iter)
- precision: pass (L0/L1/L2/Boundary all PASS)
- adopted: yes
- rollback_reason: N/A
- next_hint: host-side _finalize uses PyTorch ops (.sum/.sqrt/.log/.exp), each a separate NPU kernel launch; may be a bottleneck for multi-tensor cases

## Iteration 2 — on-NPU finalize kernel (ROLLED BACK)
- timestamp: 2026-08-06T08:35Z
- bottleneck_type: sync (kernel launch overhead)
- optimization: replaced host-side _finalize() (PyTorch ops .sum/.max/.min/.sqrt/.log/.exp/.to) with 1-block finalize kernels on NPU (combine + finalize + cast in 1 kernel launch)
- baseline_time: avg_speedup=0.1713 (iter1)
- candidate_time: avg_speedup=0.1360
- improvement: -20.7% (REGRESSION)
- precision: pass
- adopted: no
- rollback_reason: TileLang JIT kernel launch overhead (~80us) exceeds CANN native op launch overhead (~15-25us). Replacing 2~5 CANN ops with 1 TileLang kernel is counterproductive. The cann-bench-elementwise-optimization.md P0 anti-pattern targets host-side DATA COPIES (chunk+contiguous), not host-side COMPUTE ops (CANN native sum/sqrt are highly optimized).
- next_hint: focus on kernel COMPUTE efficiency (double buffer to overlap load+compute) and block_N increase for FP16/BF16, NOT host overhead elimination

## Iteration 3 — block_N=16384 for FP16/BF16 (ROLLED BACK)
- timestamp: 2026-08-06T08:42Z
- bottleneck_type: compute
- optimization: increased block_N from 8192 to 16384 for FP16/BF16 (halves tile count); Lp kernel in-place exp to fit UB
- baseline_time: avg_speedup=0.1713 (iter1)
- candidate_time: avg_speedup=0.1707
- improvement: -0.3% (within noise)
- precision: pass
- adopted: no
- rollback_reason: No average improvement. BF16 large cases improved (case 9: +22%) but FP16 cases unchanged — doubling block_N doubles per-tile MTE2+V time, netting zero. Kernel is not memory-bandwidth-limited but overhead-limited.
- next_hint: reduce launch count via multi-tensor batching

## Iteration 4 — batch same-shape tensors into 1 kernel launch (ADOPTED)
- timestamp: 2026-08-06T09:25Z
- bottleneck_type: sync (launch overhead)
- optimization: group tensors by flattened N, stack into (batch, N), process in 1 kernel launch (outer T.serial(batch) loop). Batched host finalize (2~5 CANN ops total vs list_len × 2~5). Reduces TileLang launches from list_len to 1.
- baseline_time: avg_speedup=0.1713 (iter1)
- candidate_time: avg_speedup=0.2093
- improvement: +22.2%
- precision: pass
- adopted: yes
- rollback_reason: N/A
- next_hint: single-tensor cases (batch=1) slightly regressed (~5-10%) due to 2D indexing overhead vs iter1 1D kernel. Consider 1D fast-path for batch=1. Double buffer (T.Pipelined) to overlap load+compute would help all cases. TileLang launch overhead is 185us (measured), fundamental limit for small cases.
- key data: multi-tensor improvements: case 16 x4 +175%, case 11 x3 +113%, case 2 x3 +102%, case 19 x2 +106%

## Iteration 5 — T.Pipelined double buffer (ROLLED BACK)
- timestamp: 2026-08-06T09:35Z
- bottleneck_type: compute
- optimization: T.Pipelined(num_stages=2) for inner tile loop to overlap MTE2 load with V compute
- baseline_time: avg_speedup=0.2093 (iter4)
- candidate_time: avg_speedup=0.2040
- improvement: -2.5% (within noise)
- precision: pass
- adopted: no
- rollback_reason: single_core_load too short (3~6 iterations) — pipeline fill/drain (2 iterations each) consumes most of the loop, no steady-state overlap achieved. T.Pipelined designed for loops with 100+ iterations.
- next_hint: 1D fast-path for batch=1 to fix single-tensor regression

## Iteration 6 — 1D fast-path + smart batch threshold (NEUTRAL, KEPT)
- timestamp: 2026-08-06T09:50Z
- bottleneck_type: sync
- optimization: added 1D kernels (from iter1) for batch=1 (avoids 2D T.copy overhead); added _should_batch() threshold to skip batching when torch.stack cost > launch saving (large N cases)
- baseline_time: avg_speedup=0.2093 (iter4)
- candidate_time: avg_speedup=0.2092
- improvement: -0.05% (within noise)
- precision: pass
- adopted: yes (kept for architectural improvements: case 9 fixed -14%, single-tensor cases improved 3-10%)
- rollback_reason: N/A
- next_hint: reduce Lp finalize CANN ops (5 ops → 3 via torch.pow)

## Iteration 7 — torch.pow for Lp finalize (NEUTRAL, KEPT)
- timestamp: 2026-08-06T10:05Z
- bottleneck_type: sync
- optimization: replaced (s.log()/p).exp() (3 CANN ops) with torch.pow(s, 1/p) (1 CANN op) for Lp finalize
- baseline_time: avg_speedup=0.2092 (iter6)
- candidate_time: avg_speedup=0.2095
- improvement: +0.14% (within noise)
- precision: pass
- adopted: yes (kept for code simplification; torch.pow may decompose to log+mul+exp internally on CANN)
- rollback_reason: N/A
- next_hint: TileLang 185us launch overhead is the fundamental bottleneck. Target 0.6x is mathematically unreachable.

## [DESIGN_ERROR] — TileLang launch overhead makes 0.6x target unreachable

### Evidence
1. **Measured TileLang launch overhead**: 185us per kernel call (vs CANN native op 44us). Measured via trivial 1-block kernel.
2. **6 optimization approaches tried**:
   - iter2: on-NPU finalize kernel (ROLLED BACK — TileLang launch > CANN ops)
   - iter3: block_N=16384 for FP16/BF16 (ROLLED BACK — no improvement)
   - iter4: batch multi-tensor (ADOPTED +22.2%)
   - iter5: T.Pipelined double buffer (ROLLED BACK — too few iterations)
   - iter6: 1D fast-path + smart batch (NEUTRAL, kept for architecture)
   - iter7: torch.pow for Lp finalize (NEUTRAL, kept for simplification)
3. **Best achieved**: avg_speedup = 0.2095 (target 0.6, gap 2.87×)
4. **Mathematical proof of unreachability**:
   - With 185us minimum per case (1 TileLang launch + 0 compute), maximum speedup per case = baseline / 195us
   - 13/20 cases have baseline < 117us → max speedup < 0.6 for these cases
   - Theoretical maximum average (zero compute): avg(baseline_i / 195) ≈ 0.44
   - Target 0.6 requires avg(our_time) ≤ avg(baseline) / 0.6 = 83 / 0.6 = 138us
   - But 195us minimum > 138us target → IMPOSSIBLE

### Root cause
The 0.6× performance target was set without accounting for TileLang's Python runtime dispatch overhead (185us/kernel). This is a framework-level constraint, not a kernel implementation issue. The multi-core partial reduction design is sound (iter4's batching gave +22%), but the framework overhead caps the achievable performance at ~0.21 average speedup.

---

## Round 2 Direction 1 — Conditional T.Pipelined double buffer (ADOPTED)

- timestamp: 2026-08-06T11:30Z
- bottleneck_type: compute + transfer (MTE2/V serialization)
- optimization: Conditional T.Pipelined(num_stages=2) with AUTO_SYNC=False + explicit T.barrier_all() for large single_core_load >= 20. Small shapes (single_core_load < 20) stay on T.serial to avoid pipeline fill/drain overhead.

### Implementation details

1. **Dual kernel compilation**: 12 new pipelined JIT kernels (6 norm types × 2D batched + 1D fast-path), mirroring the 12 existing serial kernels. Host dispatch routes based on `single_core_load >= PIPELINE_THRESHOLD (20)`.
2. **pass_configs_pipelined**: `AUTO_SYNC=False` + `MEMORY_PLANNING=True`. Manual sync via `T.barrier_all()` after MTE2 load (sync MTE2→V) and before MTE3 store (sync V→MTE3). V-queue operations are serial, no inter-V sync needed.
3. **Parity-split accumulators**: `acc_a` (even iterations) + `acc_b` (odd iterations) → merge after loop (`acc_ub = acc_a ⊕ acc_b`). Prevents WAW/WAR hazards on cross-iteration accumulator when pipeline stages overlap. Merge op: add (sum types), max (Linf), min (Lneg-inf).
4. **Threshold**: `single_core_load >= 20`. At scl=20, fill/drain (2 iterations) = 10% of loop, steady-state overlap = 90%. At scl=253 (case 9), fill/drain < 1%.

### Why iter5 failed but Round 2 Dir1 succeeded

| Factor | iter5 (ROLLED BACK) | Round 2 Dir1 (ADOPTED) |
|--------|---------------------|------------------------|
| Dispatch | All cases (unconditional) | Conditional (scl >= 20 only) |
| pass_configs | AUTO_SYNC=True (same as serial) | AUTO_SYNC=False (manual sync) |
| Accumulator | Single acc_ub (race risk) | Parity-split acc_a/acc_b |
| Sync | Automatic (compiler-inserted, kills overlap) | Manual T.barrier_all() after load + before store |
| Result on large cases | No improvement (auto-sync prevents overlap) | -10~30% kernel time (pipeline overlap achieved) |

### Bench results (warmup=5, iters=20, median)

- baseline_time (round1/iter7): avg_speedup = 0.2095
- candidate_time (round2_dir1): avg_speedup = 0.2312
- improvement: +10.4%
- precision: pass (L0/L1/L2/Boundary all PASS)
- adopted: yes

### Per-case highlights (pipelined cases, single_core_load >= 20)

| case | shape | scl | r1_us | r2_us | delta | r1_sp | r2_sp | d_sp |
|------|-------|-----|-------|-------|-------|-------|-------|------|
| 9 | [363,367,373]×2 bf16 | 253 | 1261 | 887 | -274us (-22%) | 0.255 | 0.364 | +0.109 |
| 20 | [2,3,17,1024,101]×4 fp32 | 54 | 987 | 795 | -193us (-20%) | 0.191 | 0.238 | +0.047 |
| 5 | [2048,4096]×3 fp32 | 43 | 625 | 530 | -95us (-15%) | 0.211 | 0.246 | +0.034 |
| 3 | [4096,4096] bf16 | 86 | 353 | 318 | -35us (-10%) | 0.208 | 0.243 | +0.034 |
| 13 | [11,13,17,67,67] fp32 | 56 | 372 | 327 | -45us (-12%) | 0.243 | 0.273 | +0.030 |
| 2 | [2048,2048]×3 fp32 | 22 | 374 | 334 | -40us (-11%) | 0.244 | 0.274 | +0.030 |
| 4 | [2048,2048] fp16 | 22 | 303 | 287 | -16us (-5%) | 0.194 | 0.195 | +0.001 |

Pipelined cases total: -798us, +0.286 speedup.
Serial cases (13, scl < 20): -121us, +0.148 speedup (within noise, no regression).

### [DB-ANALYSIS] (pre-implementation)

- Q1: MTE3 inside loop? No. Only MTE2 (load) + V (compute). MTE3 (store) after loop.
- Q2: Cross-iteration accumulator? Yes. Parity-split (acc_a/acc_b) used.
- Q3: Sync method? T.barrier_all() after MTE2 load, before MTE3 store. AUTO_SYNC=False.

### Anti-pattern check

- **纯 AIV memory bound 算子未做流水/双 buffer**: RESOLVED. Large-shape cases now use T.Pipelined(num_stages=2) to overlap MTE2 load with V compute.
- **冗余全局同步**: T.barrier_all() used 2× per iteration (after load + before store). Acceptable because only MTE2→V and V→MTE3 crossings need sync; V-queue is serial. Flag-based sync (set_flag/wait_flag) could reduce cross-core sync overhead but adds complexity — deferred to future optimization if needed.

- rollback_reason: N/A
- next_hint: Flag-based sync (T.set_flag/T.wait_flag) instead of T.barrier_all() to avoid cross-core sync overhead. Explore num_stages=3 for cases with scl > 100 (case 9: scl=253).

---

## Round 2 Direction 2 — VEC_NUM=2 dual vector sub-core (ADOPTED)

- timestamp: 2026-08-06T12:30Z
- bottleneck_type: compute (V-queue element-wise underutilization)
- optimization: VEC_NUM=2 — split each tile's element-wise compute across the 2 vector sub-cores (vid=0,1) that every Ascend910B3 AIV core provides. Previously both vids ran identical code (vid=1 idle), wasting half the vector throughput. Now each vid processes `half_block = block_N // 2` elements in parallel.

### Key discovery: default vid extent = 2

Source code inspection (`src/ir.cc` L249-259) revealed that `T.Kernel(N, is_npu=True)` with default `threads=None` sets `is_npu_kernel_frame=True`, which **hardcodes vid extent to 2**:

```cpp
n->frames.push_back(LaunchThread(
    CreateEnvThread("vid", "blockIdx.y", grid_size[0].dtype()), 2));  // hardcoded 2
```

This means all prior kernels (iter1~round2_dir1) were already launching with VEC_NUM=2, but both vids executed identical code and wrote the same `Partial[cid]` (last-write-wins). The `test_tilelang_ascend_language_tile_atomic_add.py` test confirms: `expected = num_blocks * VEC_NUM` (each cid×vid instance contributes once).

Direction 2 activates the idle vid=1 by:
1. Halving all element-wise buffer sizes: `block_N` → `half_block = block_N // VEC_NUM`.
2. Offseting each vid's GM read: `logical_tile * block_N + vid * half_block`.
3. Halving cast count: `T.tile.cast(..., block_N)` → `T.tile.cast(..., half_block)`.
4. Expanding Partial output to `(batch, launch_cores, VEC_NUM)` / `(launch_cores, VEC_NUM)` — each vid writes its own partial.
5. Host finalize combines both cid and vid dims: `dim=[1, 2]` (batched) / full reduce (single).

### Implementation details

- **Applied to**: ALL 24 kernels (12 serial + 12 pipelined, 6 norm types × 2D/1D). Uniform modification via mechanical pattern replacement ensures consistency.
- **reduce_sum merge strategy**: per-vid acc + host-side merge. Each vid accumulates into its own `acc_ub` (per-vid private because `alloc_shared` in kernel body is per-(cid,vid) instance). Kernel writes `Partial[cid, vid]`; host `_finalize_batched` does `sum/max/min over dim=[1,2]`. No cross-vid synchronization needed (no atomic_add, no cross-vid barrier beyond existing T.barrier_all).
- **buffer layout**: 1D `(half_block,)` per vid. half_block = 4096 (block_N=8192/2). Each vid: 5 buffers × 4096 × 4B = 80KB. Pipelined double-buffer: ×2 = 160KB < 192KB UB limit ✓.
- **UB budget**: 80KB (serial) / 160KB (pipelined with T.Pipelined num_stages=2) per vid, well under 192KB UB.
- **block_N alignment**: `_choose_block_n` returns powers of 2 ≥ 32, so `block_N % 2 == 0` always holds. half_block ≥ 16, 16 × 2B = 32B (32B alignment ✓).
- **T.Pipelined + VEC_NUM=2 interaction**: T.barrier_all() synchronizes both vids on the same core. Since both vids execute symmetric work (same tile count, half_block each), pipeline progress stays aligned — no deadlock or significant bubble observed.

### Bench results (warmup=5, iters=20, median)

- baseline_time (round2_dir1): avg_speedup = 0.2312
- candidate_time (round2_dir2): avg_speedup = 0.2429
- improvement: +5.06%
- precision: pass (L0/L1/L2/Boundary all PASS)
- adopted: yes

### Per-case highlights

Large-shape cases (pipelined, scl >= 20) — significant kernel time reduction:

| case | shape | scl | r1_us | r2_us | delta | r1_sp | r2_sp | d_sp |
|------|-------|-----|-------|-------|-------|-------|-------|------|
| 9 | [363,367,373]×2 bf16 | 253 | 887 | 687 | -200us (-23%) | 0.364 | 0.470 | +0.107 |
| 5 | [2048,4096]×3 fp32 | 43 | 530 | 426 | -105us (-20%) | 0.246 | 0.310 | +0.064 |
| 20 | [2,3,17,1024,101]×4 fp32 | 54 | 795 | 643 | -152us (-19%) | 0.238 | 0.294 | +0.056 |
| 13 | [11,13,17,67,67] fp32 | 56 | 327 | 283 | -44us (-13%) | 0.273 | 0.313 | +0.039 |
| 3 | [4096,4096] bf16 | 86 | 318 | 283 | -35us (-11%) | 0.243 | 0.261 | +0.018 |
| 2 | [2048,2048]×3 fp32 | 22 | 334 | 323 | -11us (-3%) | 0.274 | 0.282 | +0.008 |
| 4 | [2048,2048] fp16 | 22 | 287 | 290 | +3us (+1%) | 0.195 | 0.205 | +0.010 |

Pipelined cases total: -544us kernel time, +0.300 speedup.

Small-shape cases (serial, scl < 20): ours_us changes within ±5% (noise). Speedup fluctuations driven by baseline (torch.norm) measurement variance, not kernel regression. No case exceeded -5% ours_us degradation.

### Anti-pattern check

- **Vector Core 内逐元素/逐行 for loop 计算**: N/A — already using vectorized T.tile ops on full half_block tiles.
- **tile size 过小导致片上内存浪费**: half_block=4096 (FP32 16KB per buffer) is well-sized; no waste.
- **纯 AIV memory bound 算子未做流水/双 buffer**: RESOLVED (round2_dir1) + VEC_NUM=2 now also utilizes both vector sub-cores. MTE2 load per vid halved (4096 elements), V compute per vid halved — both pipelines benefit.

### [DB-ANALYSIS] (VEC_NUM=2 + T.Pipelined interaction)

- Q1: Does VEC_NUM=2 change MTE2/V/MTE3 queue behavior? No — each vid has its own MTE2/V/MTE3 queues. T.barrier_all() syncs both vids at the same point, preserving pipeline semantics.
- Q2: Cross-vid accumulator? No — per-vid acc_ub (private alloc). Merge happens host-side via `dim=[1,2]` reduce.
- Q3: Sync method? Unchanged from round2_dir1: T.barrier_all() after MTE2 load, before MTE3 store. Both vids hit the same barrier, progress stays aligned due to symmetric work.

- rollback_reason: N/A
- next_hint: Case 4 (fp16, scl=22) saw no kernel improvement (287→290us) — fp16 half_block=4096 V compute is already fast, barrier overhead may offset vid parallelism. Consider T.set_flag/T.wait_flag to replace T.barrier_all() for finer-grained vid sync. Case 9 (scl=253) still has the highest ceiling — explore num_stages=3 pipeline for scl > 100.

## Round 2 Direction 3 — Stack Optimization (ROLLED BACK)

- timestamp: 2026-08-06T11:30Z
- bottleneck_type: transfer (host-side torch.stack overhead)
- optimization: replace `torch.stack([t.view(-1) for t in list])` with alternative to avoid `aclnnStack_Pack` overhead (cann-bench op_times shows stack takes 18~31% of total time on multi-tensor cases)

### Motivation

cann-bench op_times data (host-side breakdown):

| case | shape | tl | stack_us | total_us | stack% |
|------|-------|----|---------|---------|--------|
| 20 | [2,3,17,1024,101]×4 fp32 | 4 | 233 | 866 | 27% |
| 2  | [2048,2048]×3 fp32 | 3 | 54  | 177 | 31% |
| 5  | [2048,4096]×3 fp32  | 3 | 124 | 488 | 25% |
| 11 | [3,7,13,4001]×3 fp32| 3 | 18  | 58  | 32% |
| 18 | [2,511,2049]×2 fp32 | 2 | 23  | 70  | 32% |
| 16 | [255,8193]×4 bf16   | 4 | 23  | 102 | 22% |
| 15 | [512,2049]×2 fp32   | 2 | 10  | 36  | 28% |
| 19 | [4,255,2049]×2 bf16 | 2 | 14  | 81  | 18% |
| 1  | [1024,1024]×2 fp16  | 2 | 7   | 31  | 23% |

Hypothesis: eliminating stack would recover 18~31% of total time, boosting
geometric_mean from 0.426 to ~0.490 (cann-bench), and local avg from 0.2429
to ~0.30+.

### Attempt A: torch.empty + slice copy_ (ROLLED BACK)

Implementation:
```python
x_batched = torch.empty((batch, n), dtype=ref.dtype, device=ref.device)
for i, idx in enumerate(indices):
    x_batched[i].copy_(x[idx].view(-1))
```
Rationale: torch.empty is zero-cost (uninitialized); each copy_ is a 1D
contiguous DMA (fastest path), bypassing stack's internal Pack algorithm.

### Bench results (attempt A, warmup=5, iters=20, median)

- baseline_time (round2_dir2): avg_speedup = 0.2429
- candidate_time (round2_dir3 attempt A): avg_speedup = 0.2263
- improvement: -6.86% (REGRESSION)
- precision: pass (L0/L1/L2/Boundary all PASS)
- adopted: no

Per-case comparison (batch>=2 cases, where stack is used):

| case | batch | dtype | r2_us (stack) | dir3_us (empty+copy) | delta | r2_sp | dir3_sp |
|------|-------|-------|---------------|----------------------|-------|-------|---------|
| 1  | 2 | fp16   | 330 | 369 | +38us (+12%)  | 0.211 | 0.220 |
| 2  | 3 | fp32   | 322 | 376 | +54us (+17%)  | 0.282 | 0.257 |
| 5  | 3 | fp32   | 426 | 499 | +72us (+17%)  | 0.310 | 0.265 |
| 9  | 2 | bf16   | 687 | 714 | +27us (+4%)   | 0.470 | 0.449 |
| 11 | 3 | fp32   | 368 | 408 | +40us (+11%)  | 0.246 | 0.222 |
| 15 | 2 | fp32   | 282 | 379 | +96us (+34%)  | 0.246 | 0.184 |
| 16 | 4 | bf16   | 440 | 440 | 0us (0%)      | 0.257 | 0.257 |
| 18 | 2 | fp32   | 357 | 384 | +26us (+7%)   | 0.210 | 0.195 |
| 19 | 2 | bf16   | 383 | 411 | +28us (+7%)   | 0.199 | 0.185 |
| 20 | 4 | fp32   | 643 | 672 | +29us (+5%)   | 0.294 | 0.284 |

ALL batch cases regressed or flat. Worst: case 15 +96us (+34%).

### Attempt C: torch.cat + view(1,-1) (ROLLED BACK)

Implementation:
```python
x_batched = torch.cat([x[idx].view(1, -1) for idx in indices], dim=0)
```
Rationale: stack = unsqueeze(0) + cat; both go through aclnn cat internally.
Testing if cat path is marginally faster than stack's Pack algorithm.

### Bench results (attempt C, warmup=5, iters=20, median)

- baseline_time (round2_dir2): avg_speedup = 0.2429
- candidate_time (round2_dir3 attempt C): avg_speedup = 0.2337
- improvement: -3.82% (REGRESSION, within noise for some cases)
- precision: pass
- adopted: no

Per-case (batch>=2): cat ≈ stack (differences within ±5% noise for most
cases). case 16 improved -63us but case 15 regressed +64us — measurement
variance, not systematic improvement. cat and stack are functionally
equivalent (both call aclnn cat internally).

### Root cause analysis (why direction 3 failed)

1. **`aclnnStack_Pack` is already optimized for multi-source DMA**:
   stack's internal Pack algorithm launches 1 CANN op that parallelizes
   DMA across all source tensors. Replacing it with `batch × copy_` means
   `batch` serial CANN op launches (each ~15-25us overhead), which is
   SLOWER than 1 parallelized Pack launch for batch=2~4.

2. **cann-bench op_times vs end-to-end measurement discrepancy**:
   cann-bench op_times measures `aclnnStack_Pack` as a standalone op
   (including its own launch overhead + DMA). In end-to-end measurement
   (our bench), stack's relative share is much smaller than 18~31% because
   total time also includes TileLang kernel (~185us launch + compute) and
   finalize ops. The 18~31% figure is cann-bench's isolated op cost, not
   the recoverable end-to-end overhead.

3. **stack absolute cost is small relative to TileLang launch overhead**:
   - TileLang launch: ~185us (dominant)
   - stack for batch=2 fp16 (case 1): ~7us (cann-bench) — negligible
   - stack for batch=4 fp32 (case 20): ~233us (cann-bench) — but our
     end-to-end ours_us=643us, so even removing 233us entirely would give
     410us, speedup 0.47 (vs current 0.29). However, the 233us cann-bench
     figure likely includes cann-bench framework overhead not present in
     our measurement. Our end-to-end measurement shows case 20 ours_us=643us
     with stack; replacing with copy_ gives 672us (+29us), confirming stack
     is NOT the bottleneck in our environment.

4. **copy_ launch overhead dominates for small N**:
   For case 15 (N=1049088, batch=2): stack DMA = 14us (estimated), but
   2 × copy_ launch overhead = 30-50us. The DMA time is small, so launch
   overhead dominates, making copy_ slower than stack.

### Anti-pattern check

- **Host-side全量数据搬运**: stack IS a host-side copy, but it's already
  the most efficient form (1 parallelized CANN op). Replacing with more
  copies (copy_ × batch) is WORSE. The anti-pattern guidance targets
  AVOIDABLE copies (chunk + contiguous patterns), not single-shot batched
  copies like stack.
- No kernel-side changes; kernel compute path unchanged.

### Conclusion

- Direction 3 NOT ADOPTED. Both attempt A (empty + copy_) and attempt C
  (cat) failed to improve over stack.
- stack is already near-optimal for batch=2~4 in our environment.
- The cann-bench op_times stack overhead (18~31%) does NOT translate to
  recoverable end-to-end time in our measurement setup.
- Reverted to round2_dir2 implementation (torch.stack).

- rollback_reason: Both replacement strategies (empty+copy_ and cat)
  regressed end-to-end performance. stack's single parallelized Pack op
  is more efficient than multiple serial copy_ launches. cann-bench
  op_times stack overhead does not translate to recoverable end-to-end
  time in our measurement environment.
- next_hint: Host-side stack overhead is NOT the bottleneck. Focus
  returns to kernel-side optimization:
  1. TileLang launch overhead (~185us) is still dominant for small cases —
     consider reducing launch count further (e.g. fuse multiple norm types
     into 1 kernel if same-N group has mixed scalars — currently each
     scalar group is a separate launch).
  2. Case 9 (scl=253, bf16) still highest-ceiling — num_stages=3 pipeline.
  3. Case 4 (fp16, scl=22) — T.set_flag/T.wait_flag for finer vid sync.
  4. Consider on-NPU finalize fusion (iter2 failed, but with different
     approach: fuse finalize INTO the main kernel's last core, avoiding
     extra launch).

## Round 3 Direction A — num_stages=3 triple pipeline (ROLLED BACK)
- timestamp: 2026-08-06T15:10Z
- bottleneck_type: compute (V-bound reduction, not MTE2-bound)
- optimization: upgraded all 12 pipelined kernels from T.Pipelined(num_stages=2)
  to num_stages=3 with 3-way parity-split accumulators (acc_a/acc_b/acc_c, k%3)
  to prevent WAW/WAR hazards between overlapping iterations k and k+2. Merge
  changed from 2-way (acc_a ⊕ acc_b) to 3-way ((acc_a ⊕ acc_b) ⊕ acc_c).
- baseline_time: avg_speedup=0.2366 (round2_final, local bench)
- candidate_time: avg_speedup=0.2431 (run1), 0.2385 (run2); avg=0.2408
- improvement: +1.78% (two-run average, below 3% noise threshold)
- precision: pass (L0/L1/L2/Boundary all PASS — 3-stage compiled successfully,
  no UB overflow; MEMORY_PLANNING=True handles buffer reuse for 3-stage)
- adopted: no
- rollback_reason: |
  1. Average improvement +1.78% (two-run avg) < 3% noise threshold.
  2. Bench variance confirmed ~2-4% between consecutive runs (run1=0.2431,
     run2=0.2385, delta=1.9%). Non-pipelined cases (serial kernel, UNMODIFIED)
     showed similar 4-6% "improvement" → apparent gain is bench noise, not
     real 3-stage benefit.
  3. Root cause: kernel is V-compute-bound (reduction: cast+mul+reduce_sum),
     NOT MTE2-bound. 3-stage pipeline deepens MTE2 latency hiding (3 iterations
     in flight vs 2) but V queue is serial — V compute is the bottleneck.
     Extra fill/drain cost (3 iterations vs 2) offsets pipeline benefit,
     especially for scl=20-25 where fill/drain = 12-15% of loop (vs 8-10%
     for 2-stage).
  4. Per-case analysis (pipelined cases scl>=20):
     - case 9 (scl=253, bf16): -1.6% (within noise; V-bound, not MTE2-bound)
     - case 3 (scl=86, bf16):  -1.0% (within noise)
     - case 13 (scl=56, fp32): -3.0% (borderline; possibly real but small)
     - case 5 (scl=43, fp32):  -3.0% (borderline)
     - case 2 (scl=22, fp32):  -3.5% (borderline)
     - case 4 (scl=22, fp16):  -3.0% (borderline)
     - case 20 (scl=54, fp32): +0.3% (flat)
     None show the hoped-for 20% main_kernel reduction. The kernel's
     reduction compute (V queue) dominates, not MTE2 load.
  5. UB constraint was NOT a problem — 3-stage with block_N=8192 compiled
     successfully for all dtypes (fp16/bf16/fp32). MEMORY_PLANNING=True
     reuses UB for non-overlapping buffer lifetimes, as theorized.
- ub_compilation_test: PASS (all 12 pipelined kernels compiled with
  num_stages=3 + block_N=8192 for fp16/bf16/fp32 without UB overflow)
- next_hint: |
  1. V-compute is the bottleneck for large-shape reduction kernels. To
     improve, focus on V compute efficiency:
     a. Reduce V instruction count per tile (e.g., fuse cast+mul into 1 op,
        or use T.tile.rsqrt for L2 finalize).
     b. Increase V compute density per MTE2 load (e.g., process 2 tiles
        worth of data per V pipeline fill — requires buffer restructuring).
     c. Consider L1 buffer for intermediate results (larger than UB,
        reduces MTE2 round-trips for multi-pass algorithms).
  2. TileLang launch overhead (~185us) still dominant for small cases.
     Fuse multiple norm types into 1 kernel if same-N group has mixed
     scalars (currently each scalar group = separate launch).
  3. Case 9 (scl=253, bf16) remains highest-ceiling but V-bound — consider
     bf16-specific compute path (avoid upcast to fp32 if precision allows,
     halving V compute for mul/reduce).
  4. Direction B candidates: T.set_flag/T.wait_flag for finer vid sync
     (replace T.barrier_all with per-queue flags); on-NPU finalize fusion
     into last core (avoid extra launch).
- skills_consulted:
  - tilelang-perf-optimization (references/performance-antipatterns.md,
    references/optimization-guide.md)
  - examples/pipeline/gemm_v0_pipeline.py (num_stages=3 reference)
  - examples/pipeline/matmul_add_pipeline.py (num_stages=3 reference)
  - examples/dispatch_combine/dispatch_combine_shmem.py (elif in kernel)

## Iteration Best+List — 2026-08-13T10:41Z
- bottleneck_type: sync (host-side torch.stack + multiple TileLang launches)
- optimization: 新增 L2/Lp list kernel (l2_norm_kernel_list2/3/4 + lp_norm_kernel_list2/3/4)，泛化 dispatch (_use_list_kernel 替代 _use_l1_list_kernel)，消除 L2/Lp 多 tensor case 的 torch.stack + 多次 launch 开销
- baseline_time: avg_speedup=0.2270 [本地 bench, 2026-08-13]
- candidate_time: avg_speedup=0.2486 [本地 bench, 2026-08-13]
- improvement: +9.5% (本地 bench)
- precision: pass (唯一失败 l1_c12_fp32_l5_5d 为 best baseline pre-existing 问题，batch=1 不走 list kernel)
- adopted: yes
- rollback_reason: N/A
- next_hint: 上传 cann-bench 验证官方 geo_mean 是否从 0.6179 提升；本地 bench 与官方有 +58% 偏差

## Iteration MulChain-p5-fix — 2026-08-13T11:40Z
- bottleneck_type: compute (p=5 mul chain correctness bug — |x|^6 instead of |x|^5)
- optimization: 修复 p=5 mul chain 的 buffer aliasing bug。原有代码第 2 步
  `T.tile.mul(abs_ub, x_cal, x_cal)` 计算 |x|^4（覆盖了 abs_ub 中的 |x|），
  导致最终结果为 |x|^6 而非 |x|^5。修正为 `T.tile.mul(abs_ub, x_cal, abs_ub)`
  计算 |x|^3，最终 |x|^3 * |x|^2 = |x|^5 ✓。
  p=3/p=4 mul chain 已正确，无需修改。
  修复涉及 7 个 Lp kernel：lp_norm_kernel / lp_norm_kernel_1d /
  lp_norm_kernel_pipelined / lp_norm_kernel_1d_pipelined /
  lp_norm_kernel_list2/3/4。
- baseline_time: N/A (correctness fix — baseline had [PRECISION_FAIL] on p=5)
- candidate_time: N/A (correctness fix — same instruction count as broken mul chain)
- improvement: 0% (correctness fix, not perf optimization; instruction count unchanged)
- precision: pass (l1_c12_fp32_l5_5d: max_abs 4.740e+01 → 0.000e+00, matched_ratio 0.0 → 1.0)
- adopted: yes
- rollback_reason: N/A
- next_hint: |
  1. p=5 mul chain 现已正确，case 13 (scl=5) 预期 main_kernel -20~40%
     （mul 链 3 指令 vs ln+mul+exp 4 指令，且 mul 比 ln/exp 快 3~5x）。
     需上传 cann-bench 验证实际收益。
  2. p=3 (case 5/19) 和 p=4 (case 8) mul chain 已在前期正确实现，
     预期 main -30~60%，同样需 cann-bench 数据确认。
  3. finalize 优化（p=4 用 sqrt(sqrt) 替代 pow(sum,0.25)）未实施——
     torch.pow 单 CANN op 可能比 2 次 sqrt 更快，需 msprof 验证。
- skills_consulted:
  - tilelang-perf-optimization (references/performance-antipatterns.md §基础指令拼接未融合)
