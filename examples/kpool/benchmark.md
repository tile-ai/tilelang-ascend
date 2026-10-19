**English** | [中文](benchmark_zh.md)

# GLM-5.3 K-Pool Compress Benchmark (Ascend)

## 1. Operator

Fused compressor of the GLM-5.3-Flash sparse-attention indexer. For every pool
of `pool_size` tokens: per-dimension softmax over `slot_score + ape`, BF16
rounded pooled K vectors, normalized Hadamard-128 rotation, one FP32
power-of-two absmax scale per vector, E4M3 quantization and a paged-cache
write.

- Input: `slot_k` [P, pool_size, 128] bf16, `slot_score` [P, pool_size, 128]
  bf16, `ape` [pool_size, 128] fp32, `loc` [P] int64, `write_mask` [P] bool
- Cache: `k_cache` [num_blocks, page_size, 128] uint8 (E4M3 byte patterns),
  `scale_cache` [num_blocks, page_size] fp32 (in/out, written in place)
- Reference: CUDA/ROCm `examples/kpool` in tile-ai/tilelang
  (`glm53_kpool_reference`); tolerance: every E4M3 byte within 1 ULP of the
  reference, scales bit-exact

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
| Grid / tiling | one 32-pool tile per block, `ceil(P/32)` blocks | 32 pools is the largest UB-fitted tile (P=64 fails memory planning) |
| FP8 path | E4M3 emulated with the ASCEND cast whitelist + integer bit math; `k_cache` as uint8 | A2/A3 has no FP8 data path; bytes match the reference exact |
| Hadamard | ascending-stride butterfly on contiguous half-slices of a dim-major tile | bit-exact vs the reference, P<=128 keeps compile time sane |
| Pooling | slot-serial accumulation in slot order | matches the reference summation order |
| Scale | `2^ceil(log2(absmax/448))` through the fp32 bit field, exact inverse by bit construction (round_scale=True) / plain division (round_scale=False) | reference-identical scales |
| Host adapter | fold `ape` into scores (fp32), slot-major repack, pad to 32 with inactive rows, scatter scales | numerically identical to the reference in-kernel arithmetic |
| Scale write | kernel emits scales to a contiguous scratch; the adapter scatters them with one device indexing op | 4-byte scalar GM stores are not cache-line coherent across cores (toolchain disables automatic DCCI for scalar accesses); 128-byte row copies are |

## 4. Performance (kernel-only, graph replay)

Shape: pool_size=4, head_dim=128, page_size=64, `num_blocks = P / 64`,
all-pools-fit cache, `round_scale=True`.

| P (pools) | kernel (us) | ns / pool |
|-----------|-------------|-----------|
| 1,024 | 22.5 | 22.0 |
| 4,096 | 70.0 | 17.1 |
| 16,384 | 230.2 | 14.0 |

Scaling follows block occupancy (1,024 pools = 32 blocks < 40 AIVs).

## 5. End-to-end (host adapter included, wall clock)

| P (pools) | E2E (us) | ns / pool |
|-----------|----------|-----------|
| 1,024 | 3,220 | 3,145 |
| 4,096 | 3,920 | 957 |
| 16,384 | 6,770 | 413 |

E2E is dominated by torch_npu host dispatch (~0.45 ms per op for the 7-8
preparation/scatter ops) in this environment, not by device work; it is not
part of the kernel contribution. Reducing it (in-kernel ape fold, fewer host
ops, in-kernel scale placement) is future work.

## 6. Accuracy

All tests in `test_example_glm53_kpool_compress.py` pass: every E4M3 byte of
written rows within 1 ULP of `glm53_kpool_reference` (typically byte-identical),
scales bit-exact for `round_scale=True`, untouched cache fully preserved.
