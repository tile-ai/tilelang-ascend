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
| Tool | torch timing (Python), warmup=10, rep=50, median |
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
| baseline | PR head: batched single-V-core kernel | 1.23 ms @ seqlen=256 |
| tail fix | host pad in `sinkhorn_bwd` adapter | Non-divisible seqlen safe (was silent out-of-bounds read/write) |
| shape tests | 1 -> 6 cases (seqlen 100-512, n_stream 8/16/32) | 6/6 passed |
| attempted: dual-V-core per-tile | restructure to per-tile [NS, NS] buffers + vid 0/1 split | rejected: +12-23% slower, batched ops win |

The dual-V-core per-tile variant also exposed two codegen limits, both worked
around but not worth the perf cost: `T.copy` on a [1]-element UB buffer faults
(aicore exception 507015), and iteration-carried scalar dependencies through
`T.Parallel` updates produce NaN (fixed by rewriting as `T.tile` ops). The
batched design needs neither.

## 5. Final Performance (torch timing, median of 50, warmup=10)

| seqlen | kernel | torch autograd (NPU) | Speedup |
|--------|--------|----------------------|---------|
| 256 | 1.29 ms | 8.58 ms | **6.66x** |
| 512 | 1.55 ms | 8.87 ms | **5.72x** |
| 1024 | 2.13 ms | 9.29 ms | **4.37x** |
| 2048 | 3.03 ms | 9.34 ms | **3.08x** |

Note: `torch autograd` time is dominated by backpropagation through the 20
forward Sinkhorn iterations (fixed launch overhead, nearly flat in seqlen),
which is exactly what the implicit-CG formulation avoids.

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