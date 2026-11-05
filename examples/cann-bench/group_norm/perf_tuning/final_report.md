# group_norm perf optimization — final report

## Baseline (msprof op task duration, Ascend 910B3)

| case | shape | dtype | baseline μs |
|------|-------|-------|-------------|
| cb1  | [8,32,64,64] | fp16 | 15.5 |
| cb13 | [1023,257] | fp16 | 134.4 |
| cb15 | [4,31,251,251] | bf16 | 332.5 |
| cb20 | [1,513,63,63] | fp32 | 271.2 |

## Optimizations applied

1. **[#2] v5 two-tier UB-budget formula** (`_find_block_S`): the split>2
   serial path's `data_buf_db` is dead-code-eliminated and the planner
   reliably frees Pass-1 buffers on it (verified: cb15@224, cb20@32 both
   run) — a lighter per-unit bound applies there, while the split<=2 path
   keeps the original battle-tested bound.  cb20 271→186μs, cb15 332→296μs.
2. **[#1] 2D row-tiled fast path** (`group_norm_kernel_2d_tiled`):
   num_groups==1 2D inputs tile block_M rows per block (VEC_NUM=2 cores
   split rows) with [ROWS, cpg_pad] MTE bursts instead of one row per
   block; the padded-row output (R_pad) returns a contiguous [:R] view.
   cb13 134→10.7μs (-92%).

## Final (msprof op task duration)

| case | baseline | final | delta |
|------|----------|-------|-------|
| cb1  | 15.5 | 15.6 | ~0 |
| cb13 | 134.4 | **10.7** | **-92%** |
| cb15 | 332.5 | **295.8** | -11% |
| cb20 | 271.2 | **186.6** | **-31%** |

Precision: full cann-bench 20-case suite + boundary/L2 pass on BOTH the
default AscendC and TILELANG_BACKEND=pto backends (Test Passed!).

Rejected during tuning (documented dead ends):
- Host-side `.contiguous()` on the padded slice: aclnnInplaceCopy
  ERR00100 crash on the eval env (torch 2.9/CANN 9.1) + reads
  uninitialized memory; also violates the zero-aclnn host contract.
- Pass-1 per-block row reduction ([cpg] accumulators): saves 2 big
  buffers but adds 2 reduces + 2 adds per s-block — net slower on
  high-s_num cases (cb15 +36μs). Reverted.
- tile-op region arguments (`scale_bc_full[0:cpg, 0:block_S]`):
  codegen falls back to a broken scalar path (`int4` undeclared).
- Narrow strided MTE copies: src row pitch quantizes to
  ceil(width*elem, 32B) — only exact when (block_S-tail)*elem < 32;
  otherwise per-row 1D copies are used (quantization-safe).
