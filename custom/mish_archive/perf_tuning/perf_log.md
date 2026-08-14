# Mish Stage 3 Performance Tuning Log

## Iteration 1 — 2026-08-05T17:45:00Z

- bottleneck_type: transfer + sync (memory-bound element-wise, host launch overhead dominates for small shapes; NPU kernel near torch for large shapes)
- optimization: [#1] 关闭 AUTO_CV_COMBINE（纯 Vector 算子消除空 AIC，bench 无变化但反模式修复保留）+ [#3] Fixed Core（已回滚：大 shape 严重退化 +25-36%）
- baseline_time: 0.1867 ms (bench S_aligned_fp16) / 0.9500 ms (bench L_aligned_fp16) / mean speedup 0.449x
- candidate_time: 0.1873 ms (bench S_aligned_fp16, [#1] only) / 0.9595 ms (bench L_aligned_fp16, [#1] only) / mean speedup 0.449x
- improvement: +0.3% (bench S_aligned_fp16, 噪声范围) / mean speedup 无变化 (0.449x → 0.449x)
- precision: pass (L0 8/8, L1 15/15, Boundary 4/4, L2 1 PASS + 1 WARN 非阻塞)
- adopted: yes (基于反模式修复保留 [#1]；[#3] Fixed Core 回滚因大 shape 严重退化)
- rollback_reason: [#3] Fixed Core 大 shape (8192,8192) 退化 +25-36%，mean speedup 下降 12.2%
- next_hint: 0.6x 目标未达（0.449x），但大 shape 已接近 torch（0.92-0.96x）。优化空间有限：UB 压力限制 tile size，Expert 双缓冲不可行，Fixed Core 对重计算算子退化。建议中止或接受当前性能。

### 详细数据

| 指标 | baseline (iter1 start) | candidate [#1] only | candidate [#1]+[#3] (回滚) |
|------|----------------------|---------------------|---------------------------|
| bench S_aligned_fp16 (1024,1024) median | 0.1867 ms | 0.1873 ms (+0.3%) | 0.1896 ms (+1.6%) |
| bench S_aligned_fp32 (1024,1024) median | 0.1881 ms | 0.1841 ms (-2.1%) | 0.1866 ms (-0.8%) |
| bench S_aligned_bf16 (1024,1024) median | 0.1879 ms | 0.1868 ms (-0.6%) | 0.1888 ms (+0.5%) |
| bench M_aligned_fp16 (2048,2048) median | 0.2177 ms | 0.2174 ms (-0.1%) | 0.2384 ms (+9.5%) |
| bench M_aligned_fp32 (2048,2048) median | 0.2177 ms | 0.2209 ms (+1.5%) | 0.2312 ms (+6.2%) |
| bench L_aligned_fp16 (8192,8192) median | 0.9500 ms | 0.9595 ms (+1.0%) | 1.2882 ms (+35.6%) |
| bench L_aligned_fp32 (8192,8192) median | 0.9381 ms | 0.9325 ms (-0.6%) | 1.1764 ms (+25.5%) |
| bench S_nonalign_bf16 (1023,1023) median | 0.1996 ms | 0.1917 ms (-4.0%) | 0.1950 ms (-2.3%) |
| bench S_prime_fp32 (1537,769) median | 0.1995 ms | 0.1967 ms (-1.4%) | 0.2036 ms (+2.1%) |
| **mean speedup** | **0.449x** | **0.449x** | **0.394x** |
| target | 0.6x | 0.6x | 0.6x |
| target met | NO | NO | NO |

### 瓶颈诊断

- Op Type: pure Vector (12-step T.tile.xxx element-wise)
- 大 shape (8192,8192): NPU kernel 时间占比大，bench 端到端 0.92-0.96x 接近 torch
- 小 shape (1024,1024): host runtime 开销 ~137us dominates（tilelang ~187us vs torch ~50us）
- 中 shape (2048,2048): 部分 host 开销，bench 0.40-0.42x
- UB 压力: 5 个 fp32 buffer + 1 原 dtype buffer = 168-176KB（接近 192KB 上限），无法放大 tile

### 采纳判定

- [#1] 关闭 AUTO_CV_COMBINE: bench 无变化（< 3% 噪声），但基于反模式修复保留（performance-antipatterns.md 明确指出纯 Vector + AUTO_CV_COMBINE 是反模式；sigmoid 先例保留）
- [#3] Fixed Core: 大 shape 严重退化 +25-36%，回滚
- 最终 mish.py: 保留 [#1]（关闭 AUTO_CV_COMBINE），不加 [#3]（保持 T.Kernel(m_num*n_num) launch）
- 0.6x 目标未达（0.449x），但大 shape 已接近 torch，优化空间有限

## Iteration 2 — 2026-08-06T11:03:00Z (cann-bench official, direction 1)

- bottleneck_type: transfer + dispatch (1D prime shape → 1954 tiny blocks → per-block overhead dominates)
- optimization: block_N cap increase from 512 to 8192 for M≤2 (1D shapes) in `_select_tiling`; case 12 drops from 1954 blocks / 82 iters to 123 blocks / 6 iters
- baseline_time: 177.48 us (case 12, cann-bench msprof, baseline job_fbfcb466435b)
- candidate_time: 23.6 us (case 12, cann-bench msprof, run 20260806_110344)
- improvement: +650% (case 12 speedup 0.076 → 0.570); mean speedup 0.5932 → 0.6737 (+13.6%)
- precision: pass (case 12 MERE=0 MARE=0 inf_match=True; 17/20 cases pass; 3 fp32 pre-existing failures not caused by this change)
- adopted: yes
- rollback_reason: none
- next_hint: 0.6x target MET (0.6737). 3 fp32 precision failures (cases 5, 11, 20) are pre-existing (device difference vs 910C baseline); consider investigating fp32 large-value precision on current device if those cases need to pass.

### cann-bench official results (run 20260806_110344)

| Case | Shape | dtype | speedup | kernel_us | baseline_us | precision |
|------|-------|-------|---------|-----------|-------------|-----------|
| 1 | [1024,1024] | fp16 | 0.752 | 14.5 | 10.9 | PASS |
| 2 | [2048,2048] | fp32 | 0.912 | 46.9 | 42.7 | PASS |
| 3 | [4096,4096] | bf16 | 0.909 | 189.1 | 171.8 | PASS |
| 4 | [8192,8192] | fp16 | 0.909 | 741.1 | 673.4 | PASS |
| 5 | [8192,8192] | fp32 | N/A | N/A | 640.4 | FAIL (pre-existing) |
| 6 | [1023,1023] | bf16 | 0.795 | 17.5 | 13.9 | PASS |
| 7 | [1009,1021] | fp16 | 0.621 | 17.2 | 10.7 | PASS |
| 8 | [1537,769] | fp32 | 0.542 | 26.7 | 14.5 | PASS |
| 9 | [363,367,373] | bf16 | 0.595 | 841.4 | 500.8 | PASS |
| 10 | [2049,513] | fp16 | 0.465 | 23.4 | 10.9 | PASS |
| 11 | [3,7,13,4001] | fp32 | N/A | N/A | 13.5 | FAIL (pre-existing) |
| 12 | [1000003] | bf16 | **0.570** | **23.6** | 13.5 | PASS |
| 13 | [11,13,17,67,67] | fp32 | 0.220 | 480.7 | 105.8 | PASS |
| 14 | [3,7,11,13,1009] | fp16 | 0.740 | 42.7 | 31.6 | PASS |
| 15 | [512,2049] | fp32 | 0.679 | 19.6 | 13.3 | PASS |
| 16 | [255,8193] | bf16 | 0.773 | 31.3 | 24.2 | PASS |
| 17 | [4097,511] | fp16 | 0.663 | 32.9 | 21.8 | PASS |
| 18 | [2,511,2049] | fp32 | 0.627 | 36.2 | 22.7 | PASS |
| 19 | [4,255,2049] | bf16 | 0.682 | 35.4 | 24.1 | PASS |
| 20 | [2,3,17,1024,101] | fp32 | N/A | N/A | 102.1 | FAIL (pre-existing) |

**Mean speedup (17 passing): 0.6737** (target ≥ 0.6: MET)

## Iteration 3 — 2026-08-06T11:16:00Z (cann-bench official, direction 2)

- bottleneck_type: transfer + dispatch (high-dim small-N shapes → adapter's fixed "merge all but last dim into M" yields huge M + tiny N → num_blocks blows up → per-block launch/DMA overhead dominates)
- optimization: ND smart-flatten — replace the fixed "merge all dims except last into M" with a search over all split points, picking the (M, N) that minimizes num_blocks (via new `_estimate_num_blocks` helper that wraps `_select_tiling`). On num_blocks tie, prefer larger split_idx (closer to original logic, smaller N) to avoid surprising regressions on already-well-tiled shapes.
- baseline_time: case 13 kernel 480.7 us (cann-bench msprof, iter2) / case 20 kernel 331.3 us (NPU event bench, iter2)
- candidate_time: case 13 kernel 148.0 us (cann-bench msprof, iter3) / case 20 kernel 179.2 us (NPU event bench, iter3)
- improvement: case 13 speedup 0.220 → 0.714 (+224.6%); case 20 kernel speedup 0.308 → 0.570 (+85.1%); mean speedup 0.6737 → 0.7168 (+6.40%)
- precision: pass (case 13 max_diff=2.384e-07 identical to iter2; case 20 max_diff=1.305e-06 identical to iter2; 3 fp32 pre-existing failures (5/11/20) unchanged, not caused by this change)
- adopted: yes
- rollback_reason: none
- next_hint: 0.6 target MET with margin (0.7168). Remaining bottlenecks: case 10 (0.457, 2D non-align fp16) and case 8 (0.534, 2D prime fp32) are host-overhead-bound small 2D shapes — smart-flatten does not apply (2D has only one split point). Further gains would require host-runtime launch overhead reduction (out of kernel scope).

### cann-bench official results (run 20260806_111606)

| Case | Shape | dtype | iter2 speedup | iter3 speedup | change | iter3 kernel_us | precision |
|------|-------|-------|---------------|---------------|--------|-----------------|-----------|
| 1 | [1024,1024] | fp16 | 0.752 | 0.747 | -0.7% | 14.6 | PASS |
| 2 | [2048,2048] | fp32 | 0.912 | 0.917 | +0.6% | 46.6 | PASS |
| 3 | [4096,4096] | bf16 | 0.909 | 0.907 | -0.2% | 189.3 | PASS |
| 4 | [8192,8192] | fp16 | 0.909 | 0.909 | -0.0% | 740.9 | PASS |
| 5 | [8192,8192] | fp32 | N/A | N/A | — | N/A | FAIL (pre-existing) |
| 6 | [1023,1023] | bf16 | 0.795 | 0.803 | +1.0% | 17.4 | PASS |
| 7 | [1009,1021] | fp16 | 0.621 | 0.624 | +0.4% | 17.2 | PASS |
| 8 | [1537,769] | fp32 | 0.542 | 0.534 | -1.5% | 27.2 | PASS |
| 9 | [363,367,373] | bf16 | 0.595 | 0.597 | +0.3% | 839.3 | PASS |
| 10 | [2049,513] | fp16 | 0.465 | 0.457 | -1.7% | 23.8 | PASS |
| 11 | [3,7,13,4001] | fp32 | N/A | N/A | — | N/A | FAIL (pre-existing) |
| 12 | [1000003] | bf16 | 0.570 | 0.568 | -0.3% | 23.7 | PASS |
| 13 | [11,13,17,67,67] | fp32 | **0.220** | **0.714** | **+224.6%** | **148.0** | PASS |
| 14 | [3,7,11,13,1009] | fp16 | 0.740 | 0.741 | +0.1% | 42.7 | PASS |
| 15 | [512,2049] | fp32 | 0.679 | 0.683 | +0.6% | 19.5 | PASS |
| 16 | [255,8193] | bf16 | 0.773 | 0.784 | +1.4% | 30.8 | PASS |
| 17 | [4097,511] | fp16 | 0.663 | 0.667 | +0.6% | 32.7 | PASS |
| 18 | [2,511,2049] | fp32 | 0.627 | **0.854** | **+36.2%** | 26.6 | PASS |
| 19 | [4,255,2049] | bf16 | 0.682 | 0.680 | -0.2% | 35.4 | PASS |
| 20 | [2,3,17,1024,101] | fp32 | N/A | N/A | — | 179.2* | FAIL (pre-existing) |

\* case 20 kernel time measured via NPU event bench (cann-bench skips prof for precision-failed cases): iter2 331.3 us → iter3 179.2 us (-45.9%), kernel speedup vs PyTorch baseline (102.1 us) 0.308 → 0.570.

**Mean speedup (17 passing): 0.7168** (iter2: 0.6737, +6.40%)

### Smart-flatten selection per ND case

| Case | dims | OLD (M, N) | OLD num_blocks | SMART (M, N) | SMART num_blocks | change |
|------|------|------------|----------------|--------------|------------------|--------|
| 9 | [363,367,373] | (133221, 373) | 3123 | (133221, 373) | 3123 | unchanged (split_idx=1 = old) |
| 13 | [11,13,17,67,67] | (162877, 67) | 2546 | (2431, 4489) | 684 | **-73.1%** |
| 14 | [3,7,11,13,1009] | (3003, 1009) | 188 | (3003, 1009) | 188 | unchanged (split_idx=3 = old) |
| 18 | [2,511,2049] | (1022, 2049) | 144 | (2, 1047039) | 128 | -11.1% (block_N 256→8192) |
| 19 | [4,255,2049] | (1020, 2049) | 144 | (1020, 2049) | 144 | unchanged (split_idx=1 = old) |
| 20 | [2,3,17,1024,101] | (104448, 101) | 1632 | (2, 5274624) | 644 | **-60.5%** |

2D cases (1,2,3,4,6,7,8,10,15,16,17): unchanged (only one split point).

## Iteration 4 — 2026-08-06T11:32:00Z (cann-bench official, direction 3, ROLLED BACK)

- bottleneck_type: sync + dispatch (host launch overhead dominates small shapes — cann-bench elapsed_us 14-27us but t_hw_us 1-2.5us; hypothesized Fixed Core could reduce dispatch overhead)
- optimization: Fixed Core 分 shape dispatch — new `_mish_kernel_fixed_core` (T.Kernel(launch_cores=min(block_num,24)) + T.serial(single_core_load) + striped logical_cid + bounds guard `if logical_cid < block_num`) for num_blocks < 100; large shapes keep original `_mish_kernel_default` (T.Kernel(m_num*n_num)). Threshold=100 cleanly separates small (nb 64-91) from large (nb 123+).
- baseline_time: iter3 mean speedup 0.7168 (17 passing cases); small shape cases 7/8/10/15/1/6 elapsed 17-27us
- candidate_time: iter4 mean speedup 0.6854 (-4.4%); small shapes AVG +15.8% time (case 8 +28.9%, case 10 +27.2%, case 15 +21.0%, case 6 +8.3%, case 7 +6.2%, case 1 +3.2%)
- improvement: -13.0% avg speedup on small shape cases; -4.4% overall mean speedup (0.7168→0.6854)
- precision: pass (all 17 passing cases retain identical max_diff; 3 pre-existing fp32 failures unchanged)
- adopted: NO — rolled back
- rollback_reason: Fixed Core 对 mish 小 shape 严重退化（AVG +15.8% time, case 8/10/15 退化 21-29%）。T.serial 循环 + bounds guard 开销超过 launch 数减少收益。host 开销瓶颈是 tilelang runtime (Python→C++→ACL chain)，非 NPU thread block 数量，Fixed Core 不减少 host 开销。
- next_hint: 方向 3（Fixed Core dispatch）已证伪。小 shape 瓶颈是 host runtime launch overhead (~12-25us)，超出 kernel 优化范围。iter3 mean speedup 0.7168 已达标（≥0.6），建议中止 Stage 3。

### cann-bench official results (run 20260806_113224, iter4 Fixed Core dispatch — ROLLED BACK)

| Case | Shape | dtype | iter3 speedup | iter4 speedup | sp_chg | iter3_us | iter4_us | time_chg | dispatch | precision |
|------|-------|-------|---------------|---------------|--------|----------|----------|----------|----------|-----------|
| 1 | [1024,1024] | fp16 | 0.747 | 0.724 | -3.1% | 14.6 | 15.1 | +3.2% | Fixed Core (nb=64) | PASS |
| 6 | [1023,1023] | bf16 | 0.803 | 0.742 | -7.7% | 17.4 | 18.8 | +8.3% | Fixed Core (nb=64) | PASS |
| 7 | [1009,1021] | fp16 | 0.624 | 0.587 | -5.8% | 17.2 | 18.2 | +6.2% | Fixed Core (nb=64) | PASS |
| 8 | [1537,769] | fp32 | 0.534 | 0.414 | -22.4% | 27.2 | 35.0 | +28.9% | Fixed Core (nb=91) | PASS |
| 10 | [2049,513] | fp16 | 0.457 | 0.360 | -21.4% | 23.8 | 30.3 | +27.2% | Fixed Core (nb=85) | PASS |
| 15 | [512,2049] | fp32 | 0.683 | 0.564 | -17.4% | 19.5 | 23.6 | +21.0% | Fixed Core (nb=72) | PASS |
| 2 | [2048,2048] | fp32 | 0.917 | 0.901 | -1.8% | 46.6 | 47.4 | +1.8% | default (nb=256) | PASS |
| 3 | [4096,4096] | bf16 | 0.907 | 0.906 | -0.2% | 189.3 | 189.6 | +0.2% | default (nb=1024) | PASS |
| 4 | [8192,8192] | fp16 | 0.909 | 0.908 | -0.1% | 740.9 | 741.6 | +0.1% | default | PASS |
| 9 | [363,367,373] | bf16 | 0.597 | 0.595 | -0.2% | 839.3 | 841.4 | +0.2% | default | PASS |
| 12 | [1000003] | bf16 | 0.568 | 0.565 | -0.7% | 23.7 | 23.9 | +0.7% | default (nb=123) | PASS |
| 13 | [11,13,17,67,67] | fp32 | 0.714 | 0.711 | -0.4% | 148.1 | 148.7 | +0.4% | default (nb=684) | PASS |
| 14 | [3,7,11,13,1009] | fp16 | 0.741 | 0.732 | -1.2% | 42.7 | 43.2 | +1.2% | default | PASS |
| 16 | [255,8193] | bf16 | 0.784 | 0.770 | -1.8% | 30.8 | 31.4 | +1.8% | default | PASS |
| 17 | [4097,511] | fp16 | 0.667 | 0.653 | -2.0% | 32.7 | 33.4 | +2.1% | default | PASS |
| 18 | [2,511,2049] | fp32 | 0.854 | 0.844 | -1.2% | 26.6 | 26.9 | +1.2% | default (nb=128) | PASS |
| 19 | [4,255,2049] | bf16 | 0.680 | 0.677 | -0.5% | 35.4 | 35.6 | +0.5% | default | PASS |
| 5 | [8192,8192] | fp32 | N/A | N/A | — | N/A | N/A | — | default | FAIL (pre-existing) |
| 11 | [3,7,13,4001] | fp32 | N/A | N/A | — | N/A | N/A | — | default | FAIL (pre-existing) |
| 20 | [2,3,17,1024,101] | fp32 | N/A | N/A | — | N/A | N/A | — | default | FAIL (pre-existing) |

**Mean speedup (17 passing): 0.6854** (iter3: 0.7168, **-4.4%** — REGRESSION, rolled back)

### Regression analysis

Small shape cases (Fixed Core dispatch, nb < 100): AVG +15.8% time, -13.0% speedup.
  - case 8/10/15: severe regression (+21-29% time) — these have larger block_M (64-128) and fewer blocks (72-91), so single_core_load=3-4 and the T.serial loop overhead per tile is proportionally higher.
  - case 1/6/7: moderate regression (+3-8% time) — block_M=32, nb=64, single_core_load=3.

Large shape cases (default kernel, nb >= 100): all within ±2% (noise) — dispatch correctly routes to original kernel.

Root cause: Fixed Core's T.serial loop + bounds guard adds per-tile overhead that exceeds any dispatch reduction. The host launch overhead bottleneck (cann-bench elapsed_us >> t_hw_us) is tilelang runtime (Python→C++→ACL chain), NOT NPU thread-block count — Fixed Core changes the latter but not the former. iter1 already showed Fixed Core was flat-to-negative for small shapes; iter4 confirms this with isolated dispatch.

## Iteration 5 — 2026-08-06T11:47:00Z (cann-bench official, direction 4, ROLLED BACK)

- bottleneck_type: other (buffer-count optimization attempted, but MEMORY_PLANNING already aliases dead buffers → no runtime effect)
- optimization: [#4] bf16 cast path in-place — eliminate b_ub via `T.tile.mul(a_ub, a_ub, t0_ub)` (dst=src0 in-place, safe under AUTO_SYNC=True); raised UB budgets (FP32 9000→11000, CAST 8500→9800) to reflect 4-buffer layout
- baseline_time: iter3 mean speedup 0.7168 (17 passing cases); bf16 cases mean 0.690
- candidate_time: iter5 mean speedup 0.7140 (-0.4%); bf16 cases mean 0.6875 (-0.18%)
- improvement: -0.4% overall / -0.18% bf16 (within ±3% noise threshold)
- precision: pass (bf16 MERE all 0.000000 identical to iter3; 17/17 pass; 3 pre-existing fp32 failures unchanged)
- adopted: NO — rolled back
- rollback_reason: 性能无变化（< 3% 噪声阈值）。MEMORY_PLANNING=True 已将 b_ub（在 a_ub 死亡时诞生）别名化到 a_ub 的内存，显式 in-place 产生相同编译结果。预算提升未改变任何 case 的 tiling（block_M 32 对齐吸收了增量）。
- next_hint: 方向 4 已证伪。4 个方向全部探索完毕（方向1+2 采纳，方向3+4 回滚）。iter3 mean speedup 0.7168 为最终最优结果（目标 ≥0.6 达标）。建议中止 Stage 3。

### cann-bench official results (run 20260806_114645, iter5 bf16 in-place — ROLLED BACK)

| Case | Shape | dtype | iter3 speedup | iter5 speedup | sp_chg | precision |
|------|-------|-------|---------------|---------------|--------|-----------|
| 3 | [4096,4096] | bf16 | 0.907 | 0.9025 | -0.5% | PASS (MERE=0) |
| 6 | [1023,1023] | bf16 | 0.803 | 0.8039 | +0.1% | PASS (MERE=0) |
| 9 | [363,367,373] | bf16 | 0.597 | 0.5943 | -0.5% | PASS (MERE=0) |
| 12 | [1000003] | bf16 | 0.568 | 0.5654 | -0.5% | PASS (MERE=0) |
| 16 | [255,8193] | bf16 | 0.784 | 0.7819 | -0.3% | PASS (MERE=0) |
| 19 | [4,255,2049] | bf16 | 0.680 | 0.6770 | -0.4% | PASS (MERE=0) |
| **bf16 mean** | — | — | **0.690** | **0.6875** | **-0.18%** | — |
| 1 | [1024,1024] | fp16 | 0.747 | 0.7385 | -1.1% | PASS |
| 2 | [2048,2048] | fp32 | 0.917 | 0.9009 | -1.8% | PASS |
| 4 | [8192,8192] | fp16 | 0.909 | 0.9046 | -0.5% | PASS |
| 7 | [1009,1021] | fp16 | 0.624 | 0.6178 | -1.0% | PASS |
| 8 | [1537,769] | fp32 | 0.534 | 0.5343 | +0.1% | PASS |
| 10 | [2049,513] | fp16 | 0.457 | 0.4599 | +0.6% | PASS |
| 13 | [11,13,17,67,67] | fp32 | 0.714 | 0.7131 | -0.1% | PASS |
| 14 | [3,7,11,13,1009] | fp16 | 0.741 | 0.7366 | -0.6% | PASS |
| 15 | [512,2049] | fp32 | 0.683 | 0.6859 | +0.4% | PASS |
| 17 | [4097,511] | fp16 | 0.667 | 0.6663 | -0.1% | PASS |
| 18 | [2,511,2049] | fp32 | 0.854 | 0.8553 | +0.2% | PASS |
| **all-17 mean** | — | — | **0.7168** | **0.7140** | **-0.4%** | — |

### Why no improvement (root cause)

1. **MEMORY_PLANNING already aliases b_ub**: b_ub is born at step 12 (final mul) exactly when a_ub dies. The compiler's dead-buffer reuse maps b_ub to a_ub's memory automatically. Explicit in-place produces identical compiled code.
2. **Tiling unchanged**: block_M rounded to 32-multiples + clamped [32,128]. Budget increase (8500→9800) doesn't cross any boundary (bn=512: raw 33→32 old, 39→32 new, both round to 32).
3. **Optimization is correct but redundant**: source cleaner (5 vs 6 buffers) but compiled binary effectively identical.
