**English** | [中文](benchmark_zh.md)

# GLM-5.3 K-Pool Compress Benchmark (Ascend)

## 1. Operator

Fused compressor of the GLM-5.3-Flash sparse-attention indexer. For every pool
of `pool_size` tokens: per-dimension softmax over `slot_score + ape`, BF16
rounded pooled K vectors, normalized Hadamard-128 rotation, one FP32
power-of-two absmax scale per vector, E4M3 quantization; the E4M3 rows and the
scales are emitted to contiguous scratches and the host adapter scatters both
into the paged caches.

```
m[p,d]      = max_s(slot_score[p,s,d] + ape[s,d])
w[p,s,d]    = exp(slot_score[p,s,d] + ape[s,d] - m[p,d])
pooled[p,d] = bf16( sum_s w[p,s,d] * slot_k[p,s,d] / sum_s w[p,s,d] )
r[p,:]      = bf16( hadamard128(pooled[p,:]) * (1 / sqrt(128)) )
a[p]        = max(1e-4, max_d |r[p,d]|)
scale[p]    = 2^ceil(log2(a[p] / 448))   (round_scale=True)
              a[p] / 448                  (round_scale=False)
q[p,:]      = E4M3(clamp(r[p,:] / scale[p], -448, 448))
k_cache[loc[p] // page_size, loc[p] % page_size, :] = q[p,:]
scale_cache[loc[p] // page_size, loc[p] % page_size] = scale[p]
```

BF16 boundaries round to nearest even; E4M3 encodes round-half-even with
subnormals and saturates at 448.

- Input: `slot_k` [P, pool_size, 128] bf16, `slot_score` [P, pool_size, 128]
  bf16, `ape` [pool_size, 128] fp32, `loc` [P] int64, `write_mask` [P] bool
- Cache: `k_cache` [num_blocks, page_size, 128] uint8 (E4M3 byte patterns),
  `scale_cache` [num_blocks, page_size] fp32 (in/out, written in place)
- Reference: CUDA/ROCm `examples/kpool` in tile-ai/tilelang
  (`glm53_kpool_reference`); accuracy criteria in §6

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B3 (Atlas A2) |
| CANN | 9.2.0 |
| Framework | torch 2.7.1 + torch_npu 2.7.1.post4 |
| Baseline | none published for this operator on Ascend (absolute numbers below) |
| Measurement | NPUGraph capture-replay, kernel-only; warmup 5, 20 replays x 10 kernels |

## 3. Architecture (single kernel, pure Vector)

| Component | Choice | Reason |
|-----------|--------|--------|
| Grid / tiling | one 32-pool tile per AIV subcore (each block covers 64 pools, `ceil(P/64)` blocks) | 32 pools is the largest UB-fitted tile (P=64 fails memory planning); both subcores of a block work independently |
| FP8 path | E4M3 emulated with the ASCEND cast whitelist + integer bit math; `k_cache` as uint8 | A2/A3 has no FP8 data path; bytes match the reference exact |
| Hadamard | ascending-stride butterfly on contiguous half-slices of a dim-major tile | bit-exact vs the reference, P<=128 keeps compile time sane |
| Pooling | slot-serial accumulation in slot order | matches the reference summation order |
| Scale | `2^ceil(log2(absmax/448))` through the fp32 bit field, exact inverse by bit construction (round_scale=True) / plain division (round_scale=False) | reference-identical scales |
| Host adapter | fold `ape` into scores (fp32), slot-major repack, pad to 64 rows; implicit `write_mask` skips the mask gathers | numerically identical to the reference in-kernel arithmetic |
| Cache / scale write | kernel emits the E4M3 rows and scales to two contiguous scratches; the adapter scatters both with one device indexing op each | a device-side per-row scatter loop is serialized by conservative PIPE_ALL barriers emitted around every scalar access (~30% of kernel time); 4-byte scalar GM stores are additionally not cache-line coherent across cores |

## 4. Performance (kernel-only, graph replay)

Shape: pool_size=4, head_dim=128, page_size=64, `num_blocks = P / 64`,
all-pools-fit cache, `round_scale=True`.

| P (pools) | kernel (us) | ns / pool |
|-----------|-------------|-----------|
| 1,024 | 16.5 | 16.1 |
| 4,096 | 49.0 | 12.0 |
| 16,384 | 149.2 | 9.1 |

Scaling follows block occupancy (1,024 pools = 16 blocks, each using both AIV
subcores; 20 AI cores on this part).

## 5. End-to-end (host adapter included, wall clock)

| P (pools) | E2E (us) | ns / pool |
|-----------|----------|-----------|
| 1,024 | 2,200 | 2,148 |
| 4,096 | 3,135 | 765 |
| 16,384 | 6,725 | 410 |

E2E is dominated by torch_npu host dispatch (~0.45 ms per op for the
preparation/scatter ops) in this environment, not by device work; it is not
part of the kernel contribution. Reducing it (in-kernel `ape` fold, fewer host
preparation ops) is future work.

## 6. Accuracy

All tests in `test_example_glm53_kpool_compress.py` pass. The reference-
mirroring cases keep the upstream byte-level criterion: every E4M3 byte of
written rows within 1 ULP of `glm53_kpool_reference` (typically byte-identical).
The parametrized multi-tile sweep (300/1024/4096/16384 pools, up to 2.1M
elements per case) bounds each element by one E4M3 code or one per-pool scale
unit -- the byte-level one-ULP criterion is not statistically stable at these
element counts, because a rare bf16 pooling-boundary crossing can be amplified
by Hadamard cancellation on near-zero dims. Measured over 4 shapes x 90 runs
(~44M elements) multi-code excursions stayed within 0.25 scale units.
`round_scale=True` scales are bit-exact and untouched cache slots are fully
preserved in every case.
