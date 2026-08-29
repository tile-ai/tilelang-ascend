# TileLang LightningIndexer for Ascend NPU

This example implements **LightningIndexer** — a sparse-attention index selector — for Ascend NPUs using [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend).

For each query token it scores every key position and emits the Top-K key indices (and optionally scores) to be consumed by a subsequent sparse-attention kernel.

## Algorithm

For each query row `s1` (per batch `b`, kv-group `n2`):

```
score[s2] = Σ_g relu(Q[s1, n2*G+g, :] · K[s2, n2, :]) * W[s1, n2*G+g]      # float32
```

- `G = N1 // N2` (group-query factor)
- Positions with `s2 >= s2_valid` are masked to `-inf`:
  - default: `s2_valid = actual_k_len[b]`
  - `sparse_mode == 3` (rightDownCausal): `s2_valid = actual_k_len - actual_q_len + s1 + 1` (when `> 0`)
- The `sparse_count` highest-scoring key positions are returned as indices; slots beyond the valid key count are filled with `-1`.

## 🚀 Key Features

- **Four layouts**: `BSND+BSND`, `BSND+PA_BSND`, `TND+TND`, `TND+PA_BSND` (paged key via `block_table`).
- **Cube/Vector pipeline**: Cube scope runs the `Q·K` mma (L1/L0 ping-pong, 3-slot K pipeline, per-BSN Q reuse); Vector scope does the G-reduce, mask, per-block Top-K sort, deferred 3-slot merge, and a dispersed cross-core Phase-2 merge.
- **C/V overlap**: counting-semaphore cross-core flags with manual sync (`auto_sync=False`).
- **CPU golden**: a pure-torch reference (`lightning_indexer_golden.py`) reproduces the algorithm in float32 for precision verification.

## ⚠️ Constraints

- `D == 128` (head dim).
- `N1 % N2 == 0` (group-query factor `G = N1 // N2`).
- `sparse_count ∈ [1, 2048]`.
- `sparse_mode ∈ {0, 3}` (`0` = defaultMask, `3` = rightDownCausal).
- `layout_key` should equal `layout_query` except for the paged case (`PA_BSND`).

## Interface

```python
from lightning_indexer import lightning_indexer

indices, values = lightning_indexer(
    query, key, weights,
    actual_seq_lengths_query=...,   # [B] int32, per-batch valid length
    actual_seq_lengths_key=...,     # [B] int32, per-batch valid length
    block_table=...,                # PA_BSND only, [B, block_num] int32
    layout_query="BSND",            # "BSND" | "TND"
    layout_key="BSND",              # "BSND" | "PA_BSND" | "TND"
    sparse_count=2048,
    sparse_mode=0,
    return_value=False,             # whether to return scores
)
```

**Returns** a 2-tuple `(indices, values)`:

- `indices`: `int32`, shape `[B, S1, N2, K]` (BSND) or `[T, N2, K]` (TND); invalid slots are `-1`.
- `values`: Top-K scores in the input dtype when `return_value=True` (`-inf` for invalid slots); an empty placeholder tensor when `return_value=False`.

**PA returns indices only**: for `layout_key="PA_BSND"`, `return_value=True` raises `ValueError`. Use `return_value=False` — `values` is an empty placeholder.

`actual_seq_lengths_query` / `actual_seq_lengths_key` are per-batch. For `TND`, host-side offsets are computed as prefix sums.

## 🛠 Usage

```shell
cd examples/deepseek_v4/attention/lightning_indexer
python lightning_indexer.py            # run all four scenarios
python lightning_indexer.py bsnd_bsnd  # one of: bsnd_bsnd | bsnd_pa | tnd_tnd | tnd_pa
```

Each scenario verifies the kernel against the CPU golden (set-based index comparison, ≥95% threshold; non-PA scenarios also compare scores). On success:

> Kernel Output Match!

## Files

| File | Description |
|---|---|
| `lightning_indexer.py` | TileLang-Ascend kernel + host wrapper + four example scenarios |
| `lightning_indexer_golden.py` | Pure-CPU torch golden reference |
