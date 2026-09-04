# Cross Entropy Loss - Benchmark & Optimization Path

## Optimization Path

### Baseline (original)
- Per-row serial loop: `for n_idx: T.tile.sub(x[n_idx,:], max[n_idx])` — 64 serial vector ops per block
- No `pad_value`, risk of OOB when C not divisible by block_C
- Single dtype path (cast through `x_ub` even for float32 input)

### Step 1: pad_value tail handling
- `T.copy(..., pad_value=-inf)` on both passes
- Eliminates OOB reads when `C % block_C != 0`
- Enables correct reduction on tail tiles

### Step 2: broadcast replaces per-row loop
- `T.tile.broadcast(max_2d, tile_max, axis=1)` + `T.tile.sub(x_32, x_32, max_2d)` — one bulk op instead of 64 serial
- float16 block_C=128: 8.1ms -> 3.9ms (2.08x)

### Step 3: broadcast axis=1 fix (reviewer)
- Added `axis=1` to all 6 broadcast calls
- Fixes wrong-dim inference when `block_C == block_N_2` (59/64 rows mismatched)

### Step 4: logsum_2d hoist (reviewer)
- `prev_max`/`prev_sum` broadcast moved out of second `bc` loop
- With C=131072, block_C=128: 2 broadcasts/iter -> 2 broadcasts total (saves 2048 ops)
- Introduced `logsum_2d` buffer; two-step subtraction kept for FP32 numerical stability

### Step 5: FP32 dedicated kernel
- float32 input skips `x_ub`, loads directly into `x_32` (saves 64KB UB)
- block_C can open to 192
- Removes redundant `l_n` buffer, writes `l_n_32` directly to `loss`
- float32 block_C=192: 8.1ms -> 2.9ms (2.79x)

### Step 6: data race fix (reviewer)
- Replaced `bn = (cid * VEC_NUM + vid) % n_2_num` with linear `bn = cid * VEC_NUM + vid` + `if bn < n_2_num` guard
- Eliminates concurrent GM writes when surplus tasks wrap around

### Step 7: y_dtype cleanup (reviewer)
- Removed `int64` from `y_dtype` Literal (AscendC does not support int64 Adds)

### Step 8: CombineCV pass (reviewer)
- Added `TL_ASCEND_AUTO_CV_COMBINE: True` to pass_configs
- Removed manual `with T.Scope("V"):` — pass handles scope automatically

## Internal Benchmark (N=4, C=131072)

### float16

| Version | block_C | Latency | vs Baseline |
|---------|---------|---------|-------------|
| Baseline (per-row) | 128 | 8.1ms | 1.0x |
| Baseline (per-row) | 256 | 4.6ms | 1.76x |
| +broadcast | 128 | 3.9ms | 2.08x |
| +broadcast+axis+logsum_2d | 128 | 3.7ms | 2.19x |

### float32

| Version | block_C | Latency | vs Baseline |
|---------|---------|---------|-------------|
| Baseline (per-row) | 128 | 8.1ms | 1.0x |
| +broadcast+FP32 dedicated | 128 | 3.5ms | 2.31x |
| +broadcast+FP32 dedicated | 192 | 2.9ms | 2.79x |

## CANN Comparison (block_N=128, block_C=128)

| Shape (N, C) | This kernel | CANN operator | Ratio |
|--------------|-------------|---------------|-------|
| (1024, 1024) | 35.00us | 44.64us | 0.78x (faster than CANN) |
| (4, 131072) | 3333.14us | 26.66us | 125x (slower than CANN) |

- **Large-batch small-vocab (N=1024, C=1024)**: 1.27x faster than CANN
- **Small-batch large-vocab (N=4, C=131072)**: significantly slower; tiling design issue

## Known Limitation

Current tiling uses `block_N x block_C` 2D blocks. For small-N/large-C (e.g. N=4, C=131072):
- `block_N=128` causes heavy padding waste
- `c_num=1024` serial loops make memory bandwidth the bottleneck

Future plan: add a dedicated small-N/large-C kernel branch that splits parallelism along C dimension.

## Test Coverage (23 cases)

- dtype: float16 / float32 / bfloat16
- block_C: 16 / 32 / 64 / 128 / 192
- block_N: 16 / 32 / 64 / 128
- square block / c_num=1 / tail handling / batch 4~1024
