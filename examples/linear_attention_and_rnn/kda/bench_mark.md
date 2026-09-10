**English** | [中文](bench_mark_zh.md)

Kimi Delta Attention (KDA) is the linear-attention block of Kimi Linear: Gated
DeltaNet with the forget gate carrying a channel index, so the decay inside a
chunk is a `K`-wide vector rather than one scalar per token.

### Performance Testing

Input parameter definitions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 1 | Batch size |
| SEQ | 4096 / 8192 / 16384 | Sequence length |
| H | 4 / 8 / 16 / 32 / 96 | Query heads (`HV = H`, no GVA) |
| K, V | 128 | Key and value head dimension |
| C | 64 | Chunk size |
| BC | 16 | Anchor block inside a chunk |
| dtype | float16 / bfloat16 | Gate `g` is fp32 |

float16 throughout. `route_b` and `KDA_WY_FIXEDCORE` are both opt-in, so the
default column is what a caller who asks for nothing gets:

| H | SEQ | AscendC `safeGate=1` | tileLang default | ratio | tileLang `route_b` + `KDA_WY_FIXEDCORE` | ratio |
|------|------|------|------|------|------|------|
| 4 | 4096 | 788.14u ±7.8 | 1422.57u ±3.1 | 55.4% | 854.68u ±4.9 | **92.2%** |
| 8 | 4096 | 1197.12u ±12.3 | 2457.73u ±9.7 | 48.7% | 1239.85u ±7.4 | **96.6%** |
| 16 | 4096 | 2010.82u ±14.1 | 4799.50u ±13.8 | 41.9% | 2374.75u ±34.3 | **84.7%** |
| 32 | 4096 | 3959.94u ±30.5 | 9559.25u ±7.0 | 41.4% | 4820.22u ±17.5 | **82.2%** |
| 96 | 4096 | 11423.01u ±13.4 | 28239.44u ±39.8 | 40.5% | 14270.49u ±68.4 | **80.0%** |
| 4 | 8192 | 1493.69u ±18.6 | 2807.97u ±24.7 | 53.2% | 1657.49u ±7.9 | **90.1%** |
| 4 | 16384 | 2995.36u ±39.7 | 5698.17u ±6.1 | 52.6% | 3369.31u ±50.6 | **88.9%** |

`H = 96` is the head count Kimi K3 runs. There the median is 80.0%, and the
three collections span 79.80% to 80.57% -- it sits *on* the line rather than
above it, and a batch taken on another day could land either side. The two
trends either side of it are the more useful reading: the ratio falls with head
count (96.6% at `H = 8` down to 80.0% at `H = 96`) and holds with sequence
length (92.2% at 4096, 88.9% at 16384). What degrades this operator is
parallelism, not work.

`route_b` is worth more at `H = 96` than at `H = 4` -- 1.96x against 1.63x --
which is the opposite of what a fixed-cost argument would predict. The reason is
share: the two stages it changes are 75% of the pipeline at `H = 96` and 65% at
`H = 4`. Any ratio quoted for this operator has to name its head count.

bfloat16, at the two shapes it was taken on. There is no bf16 reference
collection, so no ratio is given:

| H | default | `route_b` |
|------|------|------|
| 4 | 1445.55u ±9.5 | 897.42u ±8.0 |
| 96 | 28380.11u ±2.2 | 14671.21u ±23.2 |

bf16 reaches `route_b` through a float32 round trip in stage 6, because the part
has no bfloat16 vector select. It costs 1.5% end to end at `H = 96` and 3.4% at
`H = 4`.

Per stage at `H = 96`, fp16:

| | cumsum | kkt | solve_tril | wy_fast | chunk_h | chunk_o |
|------|------|------|------|------|------|------|
| default | 963.2u | 10084.9u | 1356.9u | 1704.8u | 3106.2u | 11117.1u |
| `route_b` | 955.7u | 3307.0u | 1353.8u | 1698.5u | 3084.7u | 4053.4u |
| `+ KDA_WY_FIXEDCORE` | 937.9u | 3326.6u | 1356.9u | **1506.2u** | 3087.8u | 4009.9u |

The decode path is reported on its own terms, since the vendor package ships no
recurrent operator to divide by. At `H = HV = 96`, `K = V = 128`, fp16, one step
costs 115.50u at `B = 1` and 7152.52u at `B = 64` — linear to within 3%, so it is
work-bound rather than launch-bound. Its scalar pipe sits at 51.7% against the
vector pipe's 46.5%, which is the next thing worth optimising there.

**Measurement method.** `msprof` in its full form (`msprof op` returns Task
Duration 0.000000 on this box), device Task Duration read from `op_summary`.
Every figure is the median of **three independent collections**, each of 6 to 12
iterations depending on the shape, with the first iteration dropped — it
carries first-touch cost and is always the outlier. The half-range across
collections is quoted so the reader can see what the number is worth; collections
of an identical configuration vary by up to 25u per stage, which is why nothing
here rests on a single run. The six stages all compile to a `prim_func` named
`main` and so share one Op Name; they are told apart by launch order, one prefill
being six launches in a fixed sequence, with `Block Num` corroborating. Board is
`Ascend910_9362` (910_93), 20 AI cores.

`bench.sh` in this directory is a different instrument: it profiles each stage's
own correctness sweep, a mix of shapes, and answers "did this stage regress", not
"what is this shape worth".

### Optimization Strategies and Impact Analysis

For the KDA operator we adopted the following optimizations, in this order. Each
marker is the absolute time after that step, measured on board at `H = 4`, and
its ratio against the reference:

0. **Correct, unoptimized** — the six-stage pipeline straight from the paper's
   factorisation  **--- 5992.20u, 13.2%**
1. **Instruction Vectorization**: materialise the broadcasts that otherwise lower
   to one narrow instruction per row  **--- 2417.41u, 32.6%**
2. **Algorithm to Cube (kkt)**: an anchored `BC` decomposition puts the
   off-diagonal strips of the gated Gram matrix into a plain matmul
   **--- 1688u, 46.7%**
3. **Algorithm to Cube (solve_tril)**: a doubling Neumann series replaces 62 rows
   of serial forward substitution with 8 matmuls  **--- 1584.83u, 49.7%**
4. **Redundant Computation Elimination**: five cuts, each one a piece of work
   that turned out to be unnecessary -- mostly a duplicate of something the same
   kernel already did a line above  **--- 1438.16u, 54.8%**
5. **`route_b`**: the diagonal blocks join the strips on the cube
   **--- 874.74u, 90.1%**
6. **`KDA_WY_FIXEDCORE`**: stage 4 on the physical core count rather than the
   task count  **--- 854.68u, 92.2%**

Notes on the two that carry most of the gain:

**Instruction vectorization.** An operand missing one of the `T.Parallel` indices
is a broadcast, and this dialect lowers it to one narrow instruction per row
inside a loop the compiler names `outer_broadcast_idx`, each preceded by a
barrier. `T.tile.broadcast` into a tile that is dead at that point spreads it in
one wide instruction instead, at no extra UB. Measured 1866.80u -> 122.46u on an
isolated micro-benchmark, bit-identical.

**Putting the Gram matrix on the cube.** With a per-channel gate the decay sits
inside the sum over `d`, so `sum_d k_i[d] k_j[d] exp(g_i[d] - g_j[d])` is not a
matmul. Splitting the exponent at an anchor row `a` factors it into a term in `i`
and a term in `j`, which fold into the two operands and leave a plain `X Y^T`.
For the off-diagonal strips both factors are bounded; for the diagonal blocks the
column factor is not, which is why they stayed on the vector unit until
`route_b`. `route_b` raises the clamp on that column factor and moves the cube
operands to bfloat16, whose exponent range holds it. It is off by default because
it is an approximation: a gate steep enough to span more than the clamp inside
one block saturates. It is also unavailable under `cu_seqlens` -- a varlen call
that asks for it falls back to route A and warns.

**Why there is no operator fusion.** Three things were tried and measured, and
each is recorded here so it is not tried again:

- **On this platform a Cube-Vector handoff costs a GM round trip whether or not
  the two halves share a kernel.** `T.copy(ub, l1)` emits no UB→L1 move here: the
  generated AscendC contains zero `copy_ub_to_l1`, and so does every one of the
  72 kernels in this box's cache (against 267 `copy_gm_to_l1`). It is not
  silent -- `AscendWorkspaceReduction` (`tilelang/engine/phase.py:83`) rewrites
  the copy into a two-stage GM one, and the tutorial documents it as always on.
  The gate is one line, `needs_gm_workspace_ = (platform != "A5")`
  (`src/transform/ascend_workspace_reduction.cc:137`): **A5 keeps the direct
  path, A3 -- this board -- does not**, and the `pto_use_pipe` flag cannot change
  that because the workspace test is an `||`. So the cost sits on the AIC/AIV
  boundary rather than the kernel boundary, and fusing two stages into one kernel
  does not remove it. On A5 this bullet would have to be re-measured.
- **There is no inter-stage gap to recover anyway.** The six launches are
  serialised on one stream, and the five gaps between them measure 2.5-10us each,
  about 21us against a 14459.51u prefill -- **0.15%**. Fusion would be reclaiming
  idle that is not there. The vendor operator already
  overlap; fusion would be removing a cost that is not being
  paid. The official PyPTO implementation reaches the same conclusion by
  construction — it also lands `gk`, `aqk`, `akk`, `w`, `u`, `qg`, `kg`, `v_new`
  and `h` in GM.
- **Cube-side multi-buffer is correct but worth nothing here.** `aic_mac`
  occupancy is 4.2%, so a deeper pipeline has nothing to hide behind. Occupancy
  evidence beat instruction-shape evidence.

What is actually left is synchronisation and per-block setup. Stage 6 spends
30.8% of itself on synchronisation — priced by rebuilding it with the sync
inserter off, which is numerically wrong but times the barriers — and all
eighteen of its `SetFlag` / `WaitFlag` pairs are a set followed immediately by
its own wait, so nothing overlaps anything. Across all six stages 4561u of the
14459u is on the scalar unit, and it is per-block setup rather than arithmetic:
stage 4's 6144 blocks each run 277 ns against a 164 ns prologue.
`KDA_WY_FIXEDCORE` is that observation applied to one stage; the other five have
not had it.

### Optimization Results

Rows are the configurations this operator ships. The first seven columns are the
seven optimizations of `examples/flash_attention/fa_opt/bench_mark.md`, in its
order; the last two are axes absent from that list because flash attention is a
matmul to begin with, whereas stages 2 and 3 here were not and had to be
rewritten until they were.

| Configuration | L1 Residency | Instruction Vectorization | Multi-Buffer | Sync Elimination | CV pipelined | Optimized Sync Frequency | Reduced Instructions | Algorithm to Cube | Redundancy Removal | Performance (`H = 96`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| default | × | √ | × | × | × | × | × | strips | √ | 40.5% |
| `route_b` | × | √ | × | × | × | × | × | strips + diagonal | √ | 79.3% |
| `route_b` + `KDA_WY_FIXEDCORE` | × | √ | × | × | × | × | × | strips + diagonal | √ | **80.0%** |

One of the crosses is a measured dead end rather than unstarted work:
**Multi-Buffer** on the cube side, which has three A/B collections behind it and
came out slightly slower, at an unchanged 4.2% MAC occupancy. **L1 Residency** is
a cross because what this operator does is per-task residency rather than the
across-basic-block residency that column means -- but no attempt at the latter
has been priced, so it is unstarted work, not a dead end. **Sync Elimination** is
the largest single item left.

Operator implementation: https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/linear_attention_and_rnn/kda

| File Name | Description |
|---|---|
| `kda_chunk_cumsum.py` | stage 1, chunk-local cumsum of the log gate |
| `kda_chunk_scaled_dot_kkt.py` | stage 2, the gated Gram matrix; carries `route_b` |
| `kda_solve_tril.py` | stage 3, the unit lower triangular inverse |
| `kda_wy_fast.py` | stage 4, the UT transform; carries `KDA_WY_FIXEDCORE` |
| `kda_chunk_h.py` | stage 5, inter-chunk state recurrence |
| `kda_chunk_o.py` | stage 6, output; carries `route_b` |
| `kda_recurrent.py` | the decode path, one token at a time |
| `kda_varlen.py` | `cu_seqlens` bookkeeping, shared by both layers |
| `kda_chunk_ref.py`, `kda_ref.py` | the two CPU goldens |
| `bench.sh` | per-stage `msprof` regression sweep |

### Reference

The AscendC operator compared against is `chunk_kda_fwd`:
https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_kda_fwd

It is not part of the CANN binary release; it is built from that source
repository. Both sides run the same shapes and the same dtype.

It is run in its `safeGate = 1` configuration — its fast path, and the
denominator of every ratio in this file. That flag switches the score operand
from fp16 to bfloat16 (`chunk_kda_fwd_prepare.h:167`), raises the triangular
solve's pipeline depth from 1 to 4 (`:224`), and takes a software-pipelined task
loop (`:2331`).

The shape used here hits `TilingKey 2`, the compile-time specialisation for
`chunkSize == 64 && kDim == 128 && vDim == 128`. On this part that key still
dispatches to the generic implementation: the arch35 specialisation is gated on
`__CCE_AICORE__ == 310`, which is Ascend950.
