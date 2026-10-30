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

## 3. Architecture (two kernels per call)

| Component | Choice | Reason |
|-----------|--------|--------|
| Mask kernel | one AIV per 64-pool tile builds a per-lane 0 / -inf mask from `pool_starts`/`pool_ends` and the page validity | the 64-lane compare/select and the index generator are exact on a single AIV; small-vector compare needs 256B alignment |
| Main kernel | one block per (row, 64-pool page); two AIVs each own a 32-pool half: clamped page-table lookup, bulk 32x128B gather, whole-half E4M3 decode; Cube runs one 64x32x128 GEMM per block; Vector does scale, ReLU, per-head weights and the 32-head reduce; one static-extent write | bulk half-page copies keep gmem scalar accesses O(pages); the page-sized GEMM satisfies the L0 fractal constraints with an 8KB L0C accumulator |
| CV exchange | explicit `workspace_keys` / `workspace_scores` GM buffers, one producer/consumer pair each; auto cross-core sync | the `threads=2` vid-reduction path mis-splits this kernel's sub-region copies and small compare/broadcast/index ops (confirmed with probes) |
| FP8 path | `k_cache` uint8 + in-kernel bit-level E4M3 decode | A2/A3 has no FP8 data path; bytes match the reference cache exactly |
| Host adapter | metadata-only validation (no `.item()`/device reads); padded output + view | keeps the D2H round-trip out of the hot path |

## 4. Performance (kernel-only, graph replay)

Shape: page_size=64, head_dim=128, 32 indexer heads, full valid pool ranges,
page table backing every page (the pattern used by the upstream tests).

| R (rows) | max_num_pools | num_blocks | pool-tiles (R x tiles) | kernel-only (us) | ns / pool-row |
|----------|---------------|------------|------------------------|------------------|---------------|
| 2 | 128 | 4 | 4 | 13.6 | 53.1 |
| 8 | 1,024 | 16 | 128 | 55.1 | 6.7 |
| 4 | 32,768 | 512 | 2,048 | 664.9 | 5.1 |
| 32 | 32,768 | 512 | 16,384 | 4,978.9 | 4.7 |

## 5. Performance Analysis

- Device split (graph replay, R=4 / 32,768 pools): mask kernel **163.9 us** +
  main kernel **476.2 us**. The mask kernel is ~1/4 of the device time even
  though its arithmetic is tiny: it launches one AIV per 64-pool tile and each
  block is dominated by the scalar metadata reads and small compare/select ops.
- The main kernel is CV: per block two 32-pool E4M3 decodes (Vector), one
  64x32x128 GEMM (Cube), then the 32x32 post-processing (scale by row, ReLU,
  per-head weights by column, 32-head reduce) and one 128B write per half.
- Time scales linearly with the number of pool tiles: 2,048 -> 16,384
  blocks costs 665 -> 4,979 us (7.5x).
- Under msprof the two kernels of one call report 270 us / 537 us (profiler
  overhead included); the Cube pipe is active only in the main kernel.

## 6. End-to-end (host adapter included, wall clock)

| R (rows) | max_num_pools | E2E (us) |
|----------|---------------|----------|
| 2 | 128 | 1,196 |
| 8 | 1,024 | 1,224 |
| 4 | 32,768 | 1,225 |
| 32 | 32,768 | 5,312 |

E2E is dominated by the two per-call kernel launches on this stack (~0.45 ms
per call host overhead) for the small and mid shapes; it is not part of the
kernel contribution. The host adapter performs metadata-only validation (no
device reads), so the wall clock contains no D2H round-trip beyond the launch
path itself.

## 7. Accuracy

All tests in `test_example_glm53_kpool_fp8_mqa_logits.py` pass (8/8). The
comparison is position-sensitive: the exact `-inf` mask is compared first, then
the finite logits are checked with `rtol=2^-10`, `atol=2^-16` (the smoke case
reports max diff 1.953e-3, within tolerance). Metadata (`uint8` bytes, `int32`
indices) is compared exactly. The cases cover full ranges, partial/empty/
page-unaligned ranges, invalid page entries, non-multiple `max_num_pools`, the
empty-input no-launch paths and metadata rejection. Test data uses non-negative
head weights (the gated-indexer regime), where the 32-head sum has no
cancellation and the fp32 tolerance holds; mixed-sign weights can cancel and
are out of the documented tolerance regime.
