**English** | [中文](bench_mark_zh.md)

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

Best performance results, against the reference's `safeGate = 0`
configuration (see **Reference** below -- it is the fallback path, not the
fast one):

| H | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| 4 | 1141.78u | 1183.30u | 96.5% |
| 8 | 1884.56u | — | — |
| 16 | 3437.45u | — | — |
| 32 | 6855.00u | — | — |
| 96 | 20180.92u | 21508.20u | 93.8% |

The ratio without `route_b` (the default, see below) is 79.4% at `H = 4` and
71.0% at `H = 96`. It settles from `H = 16` upward: the reference carries a
larger fixed cost, which at small `H` is 19.0% of its runtime against 13.2% of
ours, and that difference is what lifts the ratio at `H = 4`. Fitting
`T = fixed + rate * waves` over the head ladder gives 190.3u + 90.57u/wave for
this implementation against 217.2u + 64.93u/wave for the reference, so 71.7% is
the steady-state ratio the two marginal rates imply.

Against the reference's `safeGate = 1` -- the configuration that is actually
faster, and equally accurate -- at the two shapes collected for it:

| H | AscendC `safeGate=1` | tileLang | Ratio (AC/TL) | tileLang (`route_b`) | Ratio (AC/TL) |
|------|------|------|------|------|------|
| 4 | 783.05u | 1438.16u | 54.4% | 1183.30u | 66.2% |
| 96 | 11414.79u | 28406.60u | 40.2% | 21508.20u | 53.1% |

Those four rows are one collection, taken together; the `safeGate = 0` figure
in the same run was 1130.66u at `H = 4` and 20117.76u at `H = 96`, about 1%
under the table above, which is the run-to-run spread on this board.

The whole of that difference is software pipelining, which this implementation
does not do at all: the reference's fast path carries a 4-deep pipelined solve
and a software-pipelined task loop, and both are on this repository's own
optimization list (`examples/flash_attention/fa_opt/`, items 3 and 5) as work
not yet started here. What this implementation did instead is algorithm-level:
the anchored decomposition and the doubling Neumann series put two stages on the
cube that were not matmuls to begin with, which is not on that list because
flash attention is a matmul already.

Long sequences, `H = 4`:

| SEQ | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| 4096 | 1123.11u | 1416.43u | 79.3% |
| 8192 | 2199.11u | 2818.35u | 78.0% |
| 16384 | 4434.27u | 5500.72u | 80.6% |

### The decode path

Reported separately, and on its own terms: the vendor package ships
`chunk_kda_fwd` and no recurrent operator, so there is no reference to divide
by. The axis that matters is the batch rather than the length -- a decode step
is one token per sequence -- and the grid is `B * HV`, so at the K3 head count
`B = 1` already puts 96 blocks on 20 cores.

`H = HV = 96`, `K = V = 128`, fp16, with a non-zero `initial_state`:

| Configuration | Time per step | Blocks | Against `B = 1` |
|------|------|------|------|
| `B = 1`, `SEQ = 1` | 115.50u | 96 | 1.00x for 1x the work |
| `B = 8`, `SEQ = 1` | 896.72u | 768 | 7.76x for 8x the work |
| `B = 32`, `SEQ = 1` | 3583.99u | 3072 | 31.03x for 32x the work |
| `B = 64`, `SEQ = 1` | 7152.52u | 6144 | 61.93x for 64x the work |
| `B = 1`, `SEQ = 8` | 859.40u | 96 | a short prompt through this path |

Scaling is linear to within 3%, and slightly better than linear, so the path is
work-bound rather than launch-bound even at `B = 1`.

Pipe occupancy, and the reason this path is the next thing worth optimising:

| Configuration | vec | scalar | mte2 | mte3 | cube mac |
|------|------|------|------|------|------|
| `B = 1`, `SEQ = 1` | 0.465 | **0.517** | 0.034 | 0.022 | 0.000 |
| `B = 64`, `SEQ = 1` | 0.458 | **0.504** | 0.054 | 0.022 | 0.000 |

The cube figure is zero by construction: every token-level operation here is
matrix-vector shaped (`M = 1`) and cannot fill the 16x16x16 fractal, so this
kernel allocates no L0/L1 and issues no `T.gemm_v0`. Decode is memory-bound in
any case.

Scalar occupancy is the finding. At 51.7% it exceeds the vector pipe's own
46.5%, and it is well past the 35% that this repository's tuning notes give as
the line past which a kernel is issuing instructions rather than computing --
the same signature that, on the prefill side, was worth 3575u when the
broadcasts were materialised. This path has never been through that pass.

## Optimization Strategies and Impact Analysis

Trajectory at `H = 4`, each step measured on board.  The ratio column is
against `safeGate = 0`, as the first table is:

| Step | Optimization | Time | Ratio (AscendC/tileLang) |
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
1866.80u -> 122.46u on an isolated micro-benchmark, bit-identical.

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

## Optimization Results

Rows are the two paths this operator ships; columns are the optimizations, taking
the seven of `examples/flash_attention/fa_opt/bench_mark.md` and adding the one
axis that is not on that list.

| Configuration | Instruction vectorization | Algorithm to cube | Redundancy removal | L1 residency | Multi-buffer | CV pipelined | Sync frequency | vs `safeGate=0` | vs `safeGate=1` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| default (`route_b=False`) | + | strips | + | per task | x | x | x | 71.0% | 40.2% |
| `route_b=True` | + | strips + diagonal | + | per task | x | x | x | 93.8% | 53.1% |

Ratios at `H = 96`. "Algorithm to cube" is this operator's own axis and is absent
from that list because flash attention is a matmul to begin with: here stage 2 and
stage 3 are not, and were rewritten until they were. The three crosses are the
reference's fast path and the bulk of the remaining gap.

| File | Description |
|---|---|
| `kda_chunk_cumsum.py` | stage 1, chunk-local cumsum of the log gate |
| `kda_chunk_scaled_dot_kkt.py` | stage 2, the gated Gram matrix; carries `route_b` |
| `kda_solve_tril.py` | stage 3 dispatch, forward substitution |
| `kda_solve_tril_cube.py` | stage 3 on the cube, doubling Neumann series |
| `kda_wy_fast.py` | stage 4, the UT transform |
| `kda_chunk_h.py` | stage 5, inter-chunk state recurrence |
| `kda_chunk_o.py` | stage 6, output |
| `kda_recurrent.py` | the decode path, one token at a time |
| `kda_varlen.py` | `cu_seqlens` bookkeeping, shared by both layers |
| `kda_chunk_ref.py`, `kda_ref.py` | the two CPU goldens |
| `bench.sh` | the `msprof` harness that produces the tables above |

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
source repository. Both sides run the same shapes and the same dtype.

The reference has two configurations and they are far apart. `safeGate = 1`
switches the score operand from fp16 to bfloat16
(`chunk_kda_fwd_prepare.h:167`), raises the triangular solve's pipeline depth
from 1 to 4 (`:224`), and takes a software-pipelined task loop (`:2331`).
Measured on this board it is 1.76x faster than `safeGate = 0` at `H = 96`
(11414.79u against 20117.76u) and 1.44x faster at `H = 4` (783.05u against
1130.66u), while its output agrees with `safeGate = 0` to fp16 quantisation:

```
safeGate=0   attn_out[0:4] = -0.000518 0.000986 0.022263 -0.002504
safeGate=1   attn_out[0:4] = -0.000509 0.000983 0.022263 -0.002506
```

and it asks for *more* workspace (109.39 MB against 104.47 MB), so it is doing
more buffering rather than less work. `safeGate = 0` is the fallback, not the
fast path, and both are reported above.

The shape used here hits `TilingKey 2`, the compile-time specialisation for
`chunkSize == 64 && kDim == 128 && vDim == 128`. On this part that key still
dispatches to the generic implementation: the arch35 specialisation is gated on
`__CCE_AICORE__ == 310`, which is Ascend950.

Measurement: `msprof` device Task Duration from `op_summary`, first launch
dropped (cold start) and the warm launches taken as the median. The six stages
compile to a `prim_func` named `main` and so share one Op Name; they are told
apart by launch order, one prefill being six launches in a fixed sequence, with
`Block Num` corroborating. `bench.sh` is a different instrument -- it profiles
each stage's own correctness sweep, a mix of shapes, and answers "did this stage
regress", not "what is this shape worth". Board is
`Ascend910_9362` (910_93), 20 AI cores. Collections of the identical
configuration vary by up to 25u per stage, so every A/B in the table above is
the median of at least three collections.
