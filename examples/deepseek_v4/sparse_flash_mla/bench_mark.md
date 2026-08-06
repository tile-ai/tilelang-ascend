Sparse flash MLA is the attention operator of DeepSeek V4. It reads one shared paged KV cache
through three access patterns — a sliding window over the original KV (`swa`), original plus
compressed KV (`hca`), and original plus compressed KV selected by top-k sparse indices (`csa`) —
so all three variants share the same kernel skeleton and differ only in which KV blocks they visit.

### Performance Testing

Input parameter definitions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 1 | Batch size |
| N1 | 64 | Query heads |
| N2 | 1 | KV heads (so 64 query heads share one KV head) |
| D | 512 | Head dimension |
| layout | TND | Query layout |
| dtype | bfloat16 | |
| block_size | 128 | Paged KV block size (original and compressed) |
| window | 127 | Sliding window, left only (`ori_win_left=127`, `ori_win_right=0`) |
| K | 512 | Top-k compressed indices (`csa` / `hca`; `swa` uses none) |
| cmp_ratio | 4 / 128 | Compression ratio — 4 for `swa`/`csa`, 128 for `hca` |
| S (prefill) | 8192 | Query and KV sequence length |
| S (decode) | 1 | One query step against a KV cache of 8193 |

Measurement: `msprof` **device Task Duration** from `op_summary`, first launch dropped
(cold start) and the rest averaged. Both sides run the same shapes and the same dtype.

Best performance results:

| Scenario | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| swa prefill | 1850.954u | 1490.930u | 124.1% |
| hca prefill | 3005.296u | 3141.853u | 95.7% |
| csa prefill | 6882.407u | 6022.700u | 114.3% |
| swa decode | 23.916u | 13.824u | 173.0% |
| hca decode | 24.007u | 18.644u | 128.8% |
| csa decode | 38.759u | 31.659u | 122.4% |

A ratio above 100% means the TileLang kernel is faster than the hand-written AscendC operator it
is ported from. Five of the six scenarios are at or above parity; `hca prefill` is the one that
still trails, at 95.7%.

### Optimization Strategies

The kernel drives the cube pipeline itself instead of calling one all-in-one gemm helper:

1. **KV ring in L1**: the paged KV blocks rotate through a small ring of L1 slots, so the
   `GM -> L1` fetch of the next block overlaps the current block's matmul
2. **L0A/L0B double buffering**: the `L1 -> L0` load of the next K chunk runs while the current
   chunk is being multiplied
3. **Kernel-driven `L1 -> L0 -> mma -> fixpipe`**: the load / matmul / writeback sequence is issued
   step by step with `real_k` / `real_n` / `k_actual` / `n_actual` / `unit_flag`, which removes the
   per-call full-pipe drain that a self-contained gemm helper has to emit
4. **Runtime contraction and output width**: `k_actual` / `n_actual` cut the matmul down to the
   window actually covered this round, instead of computing the full buffer width
5. **Hardware-paired writeback**: the last matmul of a band and its fixpipe both carry
   `unit_flag=0b11`, so the writeback of one band overlaps the matmul of the next through the
   L0C ping-pong, with no software handshake in between
6. **Directional flags instead of full barriers**: synchronization is expressed as pipe-to-pipe
   flags on individual buffers, so unrelated stages keep running

### Files

| File Name | Description |
|--------|------|
| sparse_flash_mla_kernel.py | TileLang kernels (three `@tilelang.jit` builders: swa / hca / csa) |
| sparse_flash_mla_api.py | The `sparse_flash_mla(...)` entry point; running it checks all three variants |
| sparse_flash_mla_golden.py | Reference implementation and accuracy checks |
