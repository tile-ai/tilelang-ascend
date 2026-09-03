# KDA Benchmark

Kimi Delta Attention (KDA) is the linear-attention block of the Kimi Linear
architecture. It is Gated DeltaNet with one change: the forget gate carries a
channel index, so the decay inside a chunk is a `K`-wide vector rather than one
scalar per token. This directory implements the decode recurrence and the
chunkwise prefill forward on Ascend with TileLang.

## Performance Testing

Input parameter definitions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 1 | Batch size |
| SEQ | 4096 | Sequence length |
| H | 4 / 8 / 16 / 32 / 96 | Query heads (`HV = H`, no GVA) |
| K, V | 128 | Key and value head dimension |
| C | 64 | Chunk size |
| BC | 16 | Anchor block inside a chunk |
| dtype | float16 | Gate `g` is fp32 |

`H = 96` is the head count Kimi K3 runs.

Best performance results:

| H | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| 4 | 1141.78u | 1183.30u | 96.5% |
| 8 | 1884.56u | — | — |
| 16 | 3437.45u | — | — |
| 32 | 6855.00u | — | — |
| 96 | 20180.92u | 21508.20u | 93.8% |

The ratio without `route_b` (the default, see below) is 79.4% at `H = 4` and
72.0% at `H = 96`. It settles from `H = 16` upward: the reference carries a
larger fixed cost, which at small `H` is 19.0% of its runtime against 13.2% of
ours, and that difference is what lifts the ratio at `H = 4`. Fitting
`T = fixed + rate * waves` over the head ladder gives 190.3u + 90.57u/wave for
this implementation against 217.2u + 64.93u/wave for the reference, so 71.7% is
the steady-state ratio the two marginal rates imply.

Long sequences, `H = 4`:

| SEQ | AscendC | tileLang | Ratio |
|------|------|------|------|
| 4096 | 1123.11u | 1416.43u | 79.3% |
| 8192 | 2199.11u | 2818.35u | 78.0% |
| 16384 | 4434.27u | 5500.72u | 80.6% |

## Optimization Strategies and Impact Analysis

Trajectory at `H = 4`, each step measured on board:

| Step | Optimization | Time | Ratio |
|---|---|---|---|
| 0 | Correct, unoptimized | 5992.20u | 19.1% |
| 1 | **Instruction vectorization** — materialise the broadcasts that lower to one narrow instruction per row | 2417.41u | 47.2% |
| 2 | **kkt on the cube** — anchored `BC` decomposition puts the off-diagonal strips in a plain matmul | 1688u | 67.6% |
| 3 | **solve_tril on the cube** — a doubling Neumann series replaces 62 rows of serial forward substitution with 8 matmuls | 1584.83u | 72.0% |
| 4 | **Redundancy removal** — five cuts, each one a piece of work another stage already did | 1438.16u | 79.4% |
| 5 | **`route_b`** — the diagonal blocks join the strips on the cube | 1183.30u | **96.5%** |

Notes on the two that carry most of the gain:

**Instruction vectorization.** An operand that is missing one of the
`T.Parallel` indices is a broadcast, and this dialect lowers it to one narrow
instruction per row inside a loop the compiler names `outer_broadcast_idx`,
each preceded by a barrier. `T.tile.broadcast` into a tile that is dead at that
point spreads it in one wide instruction instead, at no extra UB. Measured
1866.80 -> 122.46us on an isolated micro-benchmark, bit-identical.

**Putting the Gram matrix on the cube.** With a per-channel gate the decay sits
inside the sum over `d`, so `sum_d k_i[d] k_j[d] exp(g_i[d] - g_j[d])` is not a
matmul. Splitting the exponent at an anchor row `a` factors it into a term in
`i` and a term in `j`, which fold into the two operands and leave a plain
`X Y^T`. For the off-diagonal strips both factors are bounded; for the diagonal
blocks the column factor is not, which is why they stayed on the vector unit
until `route_b`.

`route_b` raises the clamp on that column factor and moves the cube operands to
bfloat16, whose exponent range holds it. It is off by default: it is an
approximation, and a gate steep enough to span more than the clamp inside one
block saturates. Callers whose gate is ordinary get 2.3x on this stage by
passing `route_b=True`; callers who do not ask keep the numerics this operator
shipped with.

## Correctness

Every stage is checked against two independent CPU goldens: the token-by-token
recurrence, and a chunkwise reference written from the paper's factorisation.
Agreement between the two goldens is reported alongside, since it is the floor
the kernel can reach.

Three properties are asserted exactly, not within a tolerance:

- a varlen batch is bit-identical to running each sequence on its own;
- a sequence split at a chunk boundary and run in two halves is bit-identical
  to running it whole;
- passing an all-zero `initial_state` is bit-identical to passing none.

Relative error against the token-by-token golden at `SEQ = 4096`: 8.4e-4 in
fp16 and 6.7e-3 in bf16, both about 1.7x each format's own output quantisation
floor.

## Reference

*AscendC reference implementation source:*
https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_kda_fwd

The reference is not part of the CANN binary release; it is built from that
source repository. Both sides run the same shapes and the same dtype, and the
shape used here hits its `TilingKey 2`, the compile-time specialisation for
`chunkSize == 64 && kDim == 128 && vDim == 128`, so the comparison is against
its best path on this part rather than a fallback. It is invoked with
`safeGate = 0`, which is its faster configuration.

Measurement: `msprof` device Task Duration from `op_summary`, first launch
dropped (cold start) and the warm launches taken as the median. Board is
`Ascend910_9362` (910_93), 20 AI cores. Collections of the identical
configuration vary by up to 25us per stage, so every A/B in the table above is
the median of at least three collections.
