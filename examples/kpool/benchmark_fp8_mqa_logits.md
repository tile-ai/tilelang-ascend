**English** | [中文](benchmark_fp8_mqa_logits_zh.md)

# GLM-5.3 K-Pool FP8 MQA Logits Benchmark (Ascend)

## 1. Operator

Paged pooled-history scoring of the GLM-5.3-Flash sparse-attention indexer.
For every query row the kernel walks the row's valid pool range
`[pool_start, pool_end)`, gathers the pooled keys from a caller-owned paged FP8
cache through `pool_page_table` (E4M3 bytes carried in `uint8`), dequantizes
them with the same bit-level scheme the compressor writes, scores the 32
indexer heads (`keys @ query^T`), applies the cached per-pool scale, ReLU-gates
the head scores, weights them and reduces them to one FP32 logit per pool.
Pools outside the range or backed by an invalid page emit `-inf`.

```
blk         = pool_page_table[page_table_rows[r], p // 64]
off         = p % 64
logits[r,p] = sum_h relu( dot(q[r,h,:], decode(k_cache[blk, off, :])) * scale_cache[blk, off] ) * w[r,h]
              for p in [pool_start[r], pool_end[r]) with a valid page
            = -inf otherwise
```

- Input: `query` [R, 32, 128] bf16 (E4M3 values load losslessly; the caller
  folds the per-head query scale into `weights`), `k_cache` [num_blocks, 64,
  128] uint8 (E4M3 byte patterns written by the compressor), `scale_cache`
  [num_blocks, 64] fp32, `weights` [R, 32] fp32, `pool_page_table`
  [num_page_rows, max_pages] int32, `page_table_rows` / `pool_starts` /
  `pool_ends` [R] int32
- Output: `logits` [R, max_num_pools] fp32 (kernel writes whole 64-pool tiles;
  the host returns the `[:, :max_num_pools]` view)
- Reference: upstream `examples/kpool` in tile-ai/tilelang
  (`glm53_kpool_fp8_mqa_logits_reference`); accuracy criteria in §7

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B3 (Atlas A2) |
| CANN | 9.2.0 |
| Framework | torch 2.7.1 + torch_npu 2.7.1.post4 |
| Baseline | none published for this operator on Ascend (absolute numbers below) |
| Measurement | NPUGraph capture-replay, kernel-only; 10/4/1 calls per graph depending on the shape, 20 replays; medians of three runs |

## 3. Architecture (one kernel per call)

| Component | Choice | Reason |
|-----------|--------|--------|
| Main kernel | one block owns `pages_per_block` consecutive 64-pool pages of one row; two AIVs each own a 32-pool half of every page: clamped page-table lookup, bulk 32x128B gather, whole-half E4M3 decode; Cube runs one 64x32x128 GEMM per page; Vector does scale, ReLU, per-head weights, 32-head reduce and the mask; one static-extent write per half-page | batching pages amortizes the per-block query load, page-table metadata and cross-core handshake; the page-sized GEMM satisfies the L0 fractal constraints with an 8KB L0C accumulator |
| Page batching | `pages_per_block` defaults to enough pages to cover the device's AI Cores once per row, capped at 8 | 8 pages/block measured faster than 1, 2, 4 and 16 pages/block on 910B3; the auto cross-core sync emits one flag pair per page, and the flag counter saturates above 16 pages |
| Validity mask | built in-kernel on a `[32 pools, 32 heads]` tile (0 inside the range with a valid page, -inf otherwise) and added after the head reduce; fully valid tiles skip it on a fast path | removes a separate mask kernel (previously ~25% of device time) and keeps every compare/select above the 256B vector-op floor |
| CV exchange | explicit `workspace_keys` / `workspace_scores` GM buffers, one producer/consumer pair each; auto cross-core sync | the `threads=2` vid-reduction path mis-splits this kernel's sub-region copies and small compare/broadcast/index ops (confirmed with probes) |
| FP8 path | `k_cache` uint8 + in-kernel bit-level E4M3 decode | A2/A3 has no FP8 data path; bytes match the reference cache exactly |
| Host adapter | metadata-only validation (no `.item()`/device reads); padded output + view | keeps the D2H round-trip out of the hot path |

## 4. Performance (kernel-only, graph replay)

Shape: page_size=64, head_dim=128, 32 indexer heads, full valid pool ranges,
page table backing every page (the pattern used by the upstream tests).

| R (rows) | max_num_pools | grid (blocks) | pages/block | kernel-only (us) | ns / pool-row |
|----------|---------------|---------------|-------------|------------------|---------------|
| 2 | 128 | 4 | 1 | 8.9 | 34.8 |
| 8 | 1,024 | 16 | 8 | 28.3 | 3.5 |
| 4 | 32,768 | 256 | 8 | 283.5 | 2.2 |
| 32 | 32,768 | 2,048 | 8 | 2,166.6 | 2.1 |

The current numbers include the fast-path mask and bit-sign decode (§8); they
are 14.2% (R=4) and 15.3% (R=32) faster than the pre-optimization values
(330.5 / 2,557.5 us). The R=2/R=8 rows were re-measured on the same build.

## 5. Performance Analysis

msprof (`op`, R=4 / 32,768 pools, **pre-optimization baseline**: Task Duration
338.6 us, Block Dim 256):

| Pipe (AIV, normalized by 40 AIVs) | Utilization |
|-----------------------------------|-------------|
| scalar | 69.2% |
| vector | 52.8% |
| MTE2 | 16.3% |
| MTE3 | 8.3% |

- The AIV is busy ~94% of the block duration on the baseline; scalar and vector
  are both well above 50%, so the kernel is instruction-issue bound, not memory
  bound. At the current 283.5 us, effective bandwidth is roughly
  120 MB / 283.5 us ~ 423 GB/s, about 35% of the 1.2 TB/s HBM peak.
- The E4M3 bit-level decode is the largest single cost: replacing it with a
  trivial cast drops the baseline R=4 / 32,768 pools from 330 us to 232 us,
  i.e. the decode chain is ~30% of the kernel.  After the fast-path mask and
  bit-sign decode the kernel is 283.5 us, with the decode chain still the
  largest single cost.  The Cube is nearly idle (cube 1.8%, MTE 6.7%, FIX 5.4%
  of the AIC time) because one 64x32x128 GEMM per page is tiny.
- Batching pages is what removed the per-block overhead: at one page per block
  the same kernel takes 543 us; at 8 pages per block the baseline was 330 us
  and the current build is 283.5 us.  Larger batches are capped by the
  cross-core flag counter (16 pages hang, 8 are safe).

## 6. End-to-end (host adapter included, wall clock)

| R (rows) | max_num_pools | E2E (us) |
|----------|---------------|----------|
| 2 | 128 | 387 |
| 8 | 1,024 | 401 |
| 4 | 32,768 | 354 |
| 32 | 32,768 | 2,565 |

For the small shapes E2E is host-bound: the kernel itself is 9.6-32 us and one
call costs ~350-400 us on this stack (validation + JIT dispatch + workspace
allocation + launch).  The host adapter performs metadata-only validation (no
device reads), so the wall clock contains no D2H round-trip beyond the launch
path itself.  This table is the baseline host measurement; it is host-bound and
not re-benchmarked per kernel change, so the kernel-only rows above are the
performance gate.

## 7. Accuracy

All tests in `test_example_glm53_kpool_fp8_mqa_logits.py` pass (11/11). The
comparison is position-sensitive: the exact `-inf` mask is compared first, then
the finite logits are checked with `rtol=2^-10`, `atol=2^-16` (the smoke case
reports max diff 1.953e-3, within tolerance). Metadata (`uint8` bytes, `int32`
indices) is compared exactly. The cases cover full ranges, partial/empty/
page-unaligned ranges, invalid page entries, non-multiple `max_num_pools`,
batched pages (including a non-page-aligned batched tail), an exhaustive
sweep of every finite E4M3 byte pattern, the empty-input no-launch paths and
metadata rejection. Test data uses non-negative head
weights (the gated-indexer regime), where the 32-head sum has no cancellation
and the fp32 tolerance holds; mixed-sign weights can cancel and are out of the
documented tolerance regime.

## 8. Optimization notes

- Fully valid tiles take a fast path that skips the `-inf` mask; partial and
  invalid tiles keep the general path. This removed ~12% of the kernel time at
  R=4 / 32,768 pools.
- The E4M3 sign is applied by setting the FP32 sign bit instead of a sign
  multiply, saving one vector op and 16 KB of UB.
- Decode double buffering via `T.Pipelined` is currently blocked by a codegen
  limitation: the pipelined body re-emits the `T.reinterpretcast` locals
  (`dfnorm` / `dmag` / `dfval`) and fails to compile at `pages_per_block > 1`.
  Manual double buffering (`AUTO_SYNC=False` + manual flags) remains future work.
