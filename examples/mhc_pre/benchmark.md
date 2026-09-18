**English** | [中文](benchmark_zh.md)

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
- Constraint: 1 <= hc <= 8 (B3 AXPY, JIT parameter, tested range)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| Tool | do_bench (Python), msprof op (hardware) |
| Dtype | bf16 input, fp32 accumulate |

## 3. Architecture (3-kernel pipeline, fused from original 5)

| Kernel | Function | Core | Key Parameters |
|--------|----------|------|----------------|
| A1 | GEMM: out = x @ fn.T | Cube (T.gemm_v0) | token_block=128, h_blk=512, T.Pipelined |
| A2+B1 (fused) | sqrsum + RMSNorm | Vector (dual-V-core) | sqr_h_blk=4096, T.Pipelined, in-kernel tail |
| B2+B3 (fused) | split + Sinkhorn + apply pre_mix | Vector (dual-V-core) | T.alloc_shared for Sinkhorn, UB for apply, T.unroll(hc) |

Original 5-kernel pipeline (A1 + A2 + B1 + B2 + B3) fused to 3 kernels by combining
A2+B1 (sqrsum result stays in UB) and B2+B3 (pre_mix stays in shared/L1).

A1 uses Cube GEMM (single bid dimension). A2+B1 and B2+B3 use dual-V-core
partitioning (bid = cid * 2 + vid).

## 4. Optimization Path

| Step | Change | Effect |
|------|--------|--------|
| A1 token_block | 16 -> 128 | Better Cube utilization |
| A1 h_blk | 128 -> 512 | K-tile sweep optimum |
| A1 remove guard | delete h_num=1 T.serial guard | Guard broke CI on 910B |
| A2 h_blk | 128 -> 4096 | Fewer loop iterations |
| A2 in-kernel tail | pad_value + TAIL_MASK | Delete host sqrsum pad |
| B3 2D merged load | 4 separate 1D -> 2D res_ub[hc, h_blk] | 1 T.copy vs hc copies |
| B3 hc generalize | hc=4 hardcoded -> hc 1-8 JIT | T.unroll(hc), assert 1<=hc<=8 |
| B3 in-kernel tail | pad_value + TAIL_MASK | Delete host _pad_3d |
| A2+B1 fusion | sqrsum + RMSNorm in one kernel | Save 1 launch + sqrsum GM round-trip |
| B2+B3 fusion | Sinkhorn + apply in one kernel | Save 1 launch + pre_mix GM round-trip |
| B2 T.unroll(hc) | T.serial(hc) -> T.unroll(hc) | Compile-time unroll (no perf change, cleaner) |
| pass_configs | add TL_ASCEND_TAIL_MASK | Enable pad_value for in-kernel tail |
| fn prepack | prepare_fn + fn_packed | Avoid repeated cast/transpose at inference |
| remove _kernel_cache | drop _get_kernel/_KERNEL_BUILDERS dict, call jit kernels directly | tilelang.jit already caches kernels in-memory; -34 lines, no perf impact |
| developer mode | drop manual T.Scope("C"/"V"), alloc_L1/L0C -> alloc_shared/fragment | combineCV builds scopes automatically, no perf regression, cleaner code |
| shape test expansion | 11 -> 17 shapes (add hc 5/6/7, n 1024/2048, h 7168) | Full hc 1-8 + mid/large shape coverage |
| tests moved to pytest | bulk shape tests moved to test_example_mhc_pre.py, registered in operator_test_manifest; example keeps a single simple case | Runs in CI; 17 shapes + distinct-eps (18 tests), low_priority cases only run in full/scheduled jobs |
| CI case budget | default (per-PR) set cut to 5 cases chosen by kernel compile key (per-PR kernel compiles 42 -> 15), the other 13 shapes low_priority | Fixes the Benchmark Tests 60-min timeout: --forked re-compiles every case's 3 kernels, and a manifest change triggers the full CI run; follows the moe/quant precedent (#1711) |

## 5. Final Performance (E2E, do_bench, warmup=20, rep=100, 5-run average, prepacked fn)

| n | h | hc | TileLang | PyTorch (CANN) | Speedup |
|---|---|---|----------|----------------|---------|
| 512 | 2560 | 4 | 1.59 ms | 1.23 ms | 0.77x |
| 4096 | 2560 | 4 | 1.65 ms | 2.31 ms | **1.40x** |
| 4096 | 7168 | 4 | 1.99 ms | 5.22 ms | **2.62x** |

Small shape (512x2560) is slower than CANN due to 3-kernel launch overhead.
Large shapes benefit from fused pipeline. 4096x2560 improved from 0.96x to 1.40x
after kernel fusion.

## 6. Pipeline Advantage Analysis

The E2E win over a per-op baseline (eager CANN ops or an ascendc-style
5-kernel chain) comes from four sources, in order of impact:

| # | Source | Evidence |
|---|--------|----------|
| 1 | Kernel fusion (5 -> 3) | -28.3% E2E; host overhead 0.32 -> 0.16 ms (section 7) |
| 2 | On-chip intermediates | sqrsum stays in UB (A2+B1), pre_mix stays in shared (B2+B3); 2 GM round-trips saved per token |
| 3 | Dual-V-core partition | A2+B1 / B2+B3 run `bid = cid * 2 + vid`, both vector cores of one AI Core in parallel |
| 4 | T.Pipelined + tail mask | double-buffering hides MTE2 latency; pad_value + TL_ASCEND_TAIL_MASK removes all host-side padding copies |

Fusion dominates: the intermediate tensors of a 5-op chain are small but their
launch + GM round-trip cost is fixed per token-block, so removing two launches
plus two GM round-trips is what moved 4096x2560 from 0.96x to 1.40x vs CANN.

## 7. Kernel Breakdown (n=4096, h=2560, after fusion)

| Kernel | Latency | Share |
|--------|---------|-------|
| A1 GEMM | 0.30 ms | 18.7% |
| A2+B1 sqrsum+RMSNorm | 0.30 ms | 18.5% |
| B2+B3 sinkhorn+apply | 0.86 ms | 52.7% |
| Host overhead (3 launches) | 0.16 ms | 10.0% |

B2+B3 fused kernel is the dominant component (52.7%). Host overhead reduced
from 0.32 ms (5 launches) to 0.16 ms (3 launches) after fusion.

## 8. B2 Sinkhorn Optimization Attempts

B2 (Sinkhorn) was the #1 bottleneck (28-39% before fusion). Six optimization
approaches were attempted, all blocked by codegen limitations:

| Approach | Result | Blocker |
|----------|--------|---------|
| T.alloc_shared -> T.alloc_ub | 507015 | 1D UB slice as T.copy source/dst |
| T.tile.cast for 1D->2D-row | Precision error | Column reduce doesn't support narrow real_shape (dim=0) |
| T.Scope("V") + T.alloc_shared | 507015 | V scope assigns shared to UB |
| T.serial(hc) -> T.unroll(hc) | No change (+0.1%) | Compiler already unrolls short loops |
| T.unroll(sinkhorn_iters) | No change (-0.6%) | Noise range |
| Eliminate workspace GM round-trip | 507015 | 1D shared slice as T.copy source |

msprof analysis shows B2 is **balanced** (Vec 18.8% / MTE 19.4% / Scalar 18.3% /
Wait 25.2%) — no single dominant component. The bottleneck is scalar dispatch
overhead from ~130 small operations on 4-8 element buffers, not compute or memory.

## 9. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 18/18 passed (17 shapes covering hc 1-8, n 4-4096, h 100-7168 + distinct-eps parameter test) |
| Tolerance | rtol=1e-2, atol=1e-2 |
| Max diff | 0.0156 (layer_input, n=4096) |
| Source of diff | BF16 quantization and different accumulation order of AXPY-based apply kernel |

## 10. Stop Condition

| Condition | Status |
|-----------|--------|
| E2E > CANN (large shape) | Yes (1.40x - 2.62x) |
| Kernel fusion done | Yes (5 -> 3 kernels, -28.3% latency) |
| B2 Sinkhorn optimized | Blocked by codegen (6 approaches tried) |
| All parameters routed correctly | Yes (verified by distinct-eps test) |
| Host overhead minimized | Yes (3 launches, 10%) |

Optimization stopped: 5-kernel pipeline fused to 3 kernels (A1 + A2B1 + B2B3).
B2+B3 fused kernel is the primary bottleneck (52.7%) but B2 Sinkhorn is blocked
by codegen limitations (1D UB/shared slice 507015, column reduce narrow real_shape).
Further optimization requires codegen support for 1D buffer slice T.copy or
narrow column reduce.
