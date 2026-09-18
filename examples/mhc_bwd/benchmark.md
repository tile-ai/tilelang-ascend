**English** | [中文](benchmark_zh.md)

# MHC BWD (Sinkhorn implicit CG) Benchmark

## 1. Operator

```
Forward:  R = sinkhorn(M)                    (doubly stochastic matrix)
Backward: dL/dM = (dR - x1 - x2^T) * R
          where x1, x2 are solved from the linear system by 2*n_stream
          Conjugate Gradient iterations (implicit differentiation)
```

- Input: out (R) [seqlen, n_stream, n_stream] fp32, dout (dR) same shape
- Output: res (dL/dM) [seqlen, n_stream, n_stream] fp32
- CG solver is exact to fp32 tolerance (max_abs_diff < 1e-3 vs autograd)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B3 |
| CANN | 9.2.0 |
| Tool | do_bench (tilelang.profiler), warmup=10 ms, rep=100 ms, median |
| Dtype | fp32 in / fp32 accumulate |

## 3. Architecture (single kernel, pure Vector)

| Component | Choice | Reason |
|-----------|--------|--------|
| Core usage | single V core per AI Core block (`vid == 0` guard) | CG pipeline is dispatch-bound, not compute-bound |
| Buffers | batched [tilesize, NS]-shaped UB buffers | one instruction drives tilesize rows, amortizing ~15 small ops per CG iteration |
| CG loop | T.serial(2 * n_stream) with T.Parallel element-wise updates | compile-time unroll of short loops, no conv check needed |
| Tail handling | host pad to multiple of tilesize, then trim | padded rows are degenerate (zero grads), pad-run-trim is exact |

The kernel tiles seqlen by tilesize rows per block. The host adapter `sinkhorn_bwd`
pads non-divisible seqlen with zeros: padded rows give b1 = b2 = 0, so the CG
solution degenerates to exactly zero grads, and the padded rows are trimmed off.

## 4. Optimization Path

| Step | Change | Effect |
|------|--------|--------|
| baseline | PR head: batched single-V-core kernel | 1.12 ms @ seqlen=256 (do_bench) |
| tail fix | host pad in `sinkhorn_bwd` adapter | Non-divisible seqlen safe (was silent out-of-bounds read/write) |
| shape tests | 1 -> 6 cases (seqlen 100-512, n_stream 8/16/32) | 6/6 passed |
| attempted: dual-V-core per-tile | restructure to per-tile [NS, NS] buffers + vid 0/1 split | rejected: +12-23% slower, batched ops win |
| dispatch trim | hoist matvec "+x" out of the per-tile loop (2 whole-block updates instead of 2*tilesize), skip the initial `A @ 0` matvec (r0 = b directly) | -3.1% @ seqlen 2048/4096 (interleaved A/B, 2 rounds averaged); within noise below 1024 |
| tests moved to pytest | bulk shape tests moved to test_example_mhc_bwd.py, registered in operator_test_manifest; example keeps a single simple case (seqlen=100, non-divisible, pad path) | Runs in CI; 6 shapes total, default set chosen by compile key (3 cases = 3 kernel compiles under --forked), the other 3 shapes low_priority |

The dual-V-core per-tile variant also exposed two codegen limits, both worked
around but not worth the perf cost: `T.copy` on a [1]-element UB buffer faults
(aicore exception 507015), and iteration-carried scalar dependencies through
`T.Parallel` updates produce NaN (fixed by rewriting as `T.tile` ops). The
batched design needs neither.

## 5. Final Performance (do_bench, warmup=10 ms, rep=100 ms, median)

| seqlen | kernel | torch autograd (NPU) | Speedup |
|--------|--------|----------------------|---------|
| 256 | 1.12 ms | 8.85 ms | **7.88x** |
| 512 | 1.08 ms | 9.32 ms | **8.61x** |
| 1024 | 1.15 ms | 9.12 ms | **7.91x** |
| 2048 | 1.95 ms | 9.19 ms | **4.71x** |
| 4096 | 3.90 ms | 9.04 ms | **2.32x** |

Notes:
- The kernel is dispatch-bound below seqlen=1024 on 910B3 (~1.1 ms fixed
  floor: 32-128 small blocks of ~500 scalar instructions each). From 2048 on,
  latency scales with rows and matches the original PR's 910B do_bench numbers
  (2.19 ms @ 2048, 4.38 ms @ 4096 — this build is slightly faster).
- `torch autograd` is dominated by backpropagation through the 20 forward
  Sinkhorn iterations (fixed launch overhead, nearly flat in seqlen), which
  is exactly what the implicit-CG formulation avoids.

## 6. Accuracy

| Metric | Value |
|--------|-------|
| Test cases | 6/6 passed (seqlen 100/250/256/512, n_stream 8/16/32, incl. non-divisible seqlen) |
| Tolerance | max_abs_diff < 1e-3 vs torch autograd |
| Max diff | 8.16e-07 (seqlen=250, n_stream=8) |
| vs manual-CG ref | 1.79e-07 max |
| Source of diff | fp32 reduce accumulation order in the CG iteration |

## 7. Known Limitation

Single V core per block (vid 1 idle): the CG pipeline is scalar-dispatch-bound
(~15 small ops per iteration on 8-16 element rows), so the batched single-core
design beats the dual-core per-tile alternative by 12-23%. Exploiting the second
V core would require wider per-instruction rows (larger tilesize), which the
batched design already does through y1/y2 [tilesize, NS] reductions.