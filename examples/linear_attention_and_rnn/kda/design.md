# KDA design notes

Why the kernels are shaped the way they are.  The README documents *what* the
operator does and how to run it; this file documents *why* each stage
partitions and moves data the way it does, so a reviewer does not have to
re-derive the constraints.

Two forward paths ship here and they are deliberately ordered:

* **`kda_recurrent.py` — the recurrent decode path.** One token at a time,
  carrying the `[K, V]` state.  Section 0 below.
* **the six chunkwise stages — the prefill path.** Sections 1–5.

The recurrent path was written, validated on hardware and frozen *first*, and
the chunkwise pipeline then uses it as its acceptance golden
(`kda_full.py` compares against `kda_ref.kda_ref`, the CPU twin of the
recurrent kernel).  That ordering is a requirement, not a preference: a
chunkwise decomposition checked only against another chunkwise implementation
can be consistently wrong.

Reference parameters used throughout: `B=1, SEQ=128, H=2, HV=4, K=V=128, C=64`,
`VEC_NUM=2`, so `N = SEQ/C = 2` chunks and `GRP = HV/H = 2`.
910B limits: UB 196,352 B **per AIV**, L1 524,032 B, L0A/L0B 65,536 B each,
L0C 131,072 B.

---

## 0. The recurrent path (`kda_recurrent.py`)

Three steps per token, exactly the recurrence in section 1 below:

```
S <- Diag(exp(g_t)) S                     per-channel row scaling
S <- S + beta_t * k_t (v_t - S^T k_t)^T   delta rule: only the residual is written
o_t = S^T (scale * q_t)
```

**Grid is `B * HV`, one block per (batch, value head); the two vector cores
split the V axis.**  The choice of axis is the whole design.  `S^T k` in the
delta rule reduces along K, so splitting the state along K would force a
cross-block reduction *every token*.  Splitting along V leaves each core with a
self-contained `[K, BV]` half that never has to talk to the other one.

**No Cube.**  Every token-level operation is matrix-vector shaped, `M = 1`,
which cannot fill the Cube's 16x16x16 fractal — the padding waste would exceed
the work.  The state never leaves UB across the whole `T.serial(SEQ)` loop, and
decode is memory-bound in any case, so this is not a concession.  Concretely
the kernel needs only two primitives: a row scaling `Diag(c) S` and a reduction
along the K axis; the outer product `k (x) u` is built by broadcast rather than
by a matmul.

Three dialect constraints show up here and are worth stating, because two of
them cost real debugging time:

* **`beta` is padded to 8 fp32 slots** (`[B, SEQ, HV, 8]`, value in lane 0).
  A `[1]` UB buffer is 4 B and skews the address of every allocation after it —
  it surfaces as "The UB address accessed by the VEC instruction is not
  aligned", far from the actual cause.  One read of 8 fp32 = 32 B fixes it.
  Padding the *head* axis instead (`[B, SEQ, HV+8]`) does not: head `hv` would
  start at byte `4*hv`, still unaligned.
* **Broadcasting a 1-D buffer by an outer loop variable must be in place.**
  `T.copy(s_ub, prod_ub)` and then scaling `prod_ub` works; writing the scaled
  result into a different buffer in one statement does not.
* **The outer product needs two passes.**  First tile `u` across the buffer
  using the inner variable, then scale by `k` using the outer variable.  The
  one-liner `ku[i,j] = k[i]*u[j]` is an out-of-place broadcast by an outer
  variable and does not lower.

`initial_state` is always materialised, as zeros when the caller passes none,
so the kernel reads it unconditionally rather than compiling a second variant.

---

## 1. The one mathematical difference from GDN

KDA is Gated DeltaNet with the scalar forget gate replaced by a per-channel
vector gate:

```
GDN:  S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T   alpha_t scalar
KDA:  S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
                                   ^^^^^^^^^^^^^ alpha_t in R^K
```

Everything else in this directory is a consequence of that one change.  The
reason it costs so much is that `Diag(alpha)` does **not** commute with
`k k^T`, while a scalar does:

```
scalar:  (a I) k k^T - k k^T (a I) = 0
vector:  Diag(a) k k^T - k k^T Diag(a) != 0
```

Non-commuting means `exp(Gamma_i - Gamma_j)` cannot be pulled out of the sum
over channels in stage 2, which is why that stage cannot be expressed as a
matmul.  See §3.2.

---

## 2. Stage table

| # | file | grid | chunk axis in grid | `vid` splits | engines | flags | gemms |
|---|---|---|:---:|---|:---:|---:|---:|
| 1 | `kda_chunk_cumsum.py` | `B*HV*chunk_num` | yes | K channels | V | 0 | 0 |
| 2 | `kda_chunk_scaled_dot_kkt.py` | `B*HV*chunk_num` | yes | output rows | V | 0 | 0 |
| 3 | `kda_solve_tril.py` | `ceil(B*HV*N / VEC_NUM)` | yes | **whole tasks** | V | 0 | 0 |
| 4 | `kda_wy_fast.py` | `B*HV*chunk_num` | yes | token rows | V+C | 1 | 2 |
| 5 | `kda_chunk_h.py` | `B*HV*BV_NUM` | **no** | state rows **and** token rows | C+V | 4 | 2 |
| 6 | `kda_chunk_o.py` | `B*HV*N` | yes | output rows (contiguous) **and** anchor blocks (interleaved) | V+C | 3 | 3 |

---

## 3. What forces each partitioning

Every stage cuts the way it does because of exactly one of three things:
a **data dependency**, a **numerical range**, or **on-chip capacity**.

### 3.1 Stage 1 (cumsum) -- data dependency

The prefix sum `s_i = s_{i-1} + g_i` is chained, so the token axis cannot be
split.  Batch, head and chunk are already in the grid, which leaves only the
channel axis for `vid`: `BK = K // VEC_NUM`.  UB usage is 16.7% -- capacity is
not a factor here.

The chunk length `C` itself exists for a numerical reason, not a performance
one: restarting the cumulative sum at every chunk boundary caps how far the
exponents can travel, which is what keeps every `exp()` in the five later
stages in range.

### 3.2 Stage 2 (kkt) -- numerical range

`L[i,j] = beta_i sum_d k[i,d] k[j,d] exp(G[i,d] - G[j,d])`.

Algebraically the factor splits as `exp(G_i) * exp(-G_j)` and the whole thing
becomes a single matmul.  Numerically that is a trap: `G` is a cumulative sum
of non-positive gates, so `exp(-G_j)` grows without bound down the chunk.
Measured on CPU with the reference layer, the naive fold produces **2624
non-finite entries** at the `forget` gate (min `Gamma_C = -209`) and **3840**
at `extreme` (`-841`).

So this stage keeps the element-wise form and folds the causal mask **into the
exponent before `exp()`**, not after.  Applying it afterwards would leave
`0 * inf = NaN`: a multiply cannot discard an infinity.

Consequence: the cube is idle for the whole stage.  This is the slowest kernel
of the six and the obvious target for future work -- see §5.

### 3.3 Stage 3 (solve_tril) -- data dependency

`A = (I + L)^{-1}` by forward substitution.  The row recurrence is serial, so
the two vector cores take **two whole chunk matrices** rather than splitting
one: splitting by rows would need `C - 1 = 63` cross-core handshakes per
matrix, whereas moving the parallelism up one level makes the dependency
disappear entirely (the matrices are chunk-local and independent).

That is why this is the only stage whose grid is `ceil(total / VEC_NUM)`
rather than `total`: `vid` picks a task, it does not cut an axis.

Two properties make the substitution cheap: `I + L` has a **unit diagonal**, so
there is no division and no pivoting; and `L` is nilpotent, so the inverse is a
finite sum.  UB usage is 29.3%.

### 3.4 Stage 4 (wy_fast) -- no hard constraint

Nothing forces this stage's partitioning; it follows from core count and the
16-element fractal alignment.  Worth noting is that the same index `j` is
uncoupled while *building* the operands (each row needs only its own
`beta_j, k_j, Gamma_j`) and coupled while *consuming* them (the matmul sums over
all `C` rows).  That is why the two halves must be glued back together in GM
before the cube reads them.

The two gemms share the same left operand `A` and do not depend on each other,
so one flag suffices.

### 3.5 Stage 5 (chunk_h) -- data dependency **and** capacity

This is the only stage that is genuinely serial along the chunk axis:
`S_{n+1}` needs `S_n`.  A dependent axis cannot go into the grid, so it becomes
`for ci in T.serial(N_CHUNK)` and the V axis takes its place
(`BV_NUM` replaces `chunk_num`).  Splitting V is safe because column `j` of
`S_{n+1}` depends only on column `j` of `S_n`.

`BV = min(V, 64)` is then forced by UB: at `C=64, K=128, BV=V=128` the
footprint is **180,992 B** against a **179,968 B** budget -- over by **1,024 B**.

Note the consequence: this stage's grid does not grow with sequence length.
Every other stage carries `chunk_num` in its grid and scales with `SEQ`.

### 3.6 Stage 6 (chunk_o) -- numerical range

Same trap as stage 2, but here it is solvable.  `A_qk[i,j]` needs
`exp(G_i - G_j)`, and folding it as `exp(+G_i) * exp(-G_j)` again sends one
factor to `+inf`.  Inserting the identity `exp(G_ar) * exp(-G_ar)` at an
**anchor row** `ar = a * BC` splits it into two factors whose exponents are both
`<= 0` over the off-diagonal strip:

```
qf[i] = q[i] exp(G_i  - G_ar)     i >= ar  =>  exponent <= 0
kf[j] = k[j] exp(G_ar - G_j )     j <  ar  =>  exponent <= 0
```

Both are bounded by 1, so `qf @ kf^T` is a safe cube matmul.  Underflow to zero
is the correct answer here ("that contribution decayed away"); overflow is not.

The diagonal block (`ar <= j < ar + BC`) has no anchor bounding both sides and
stays on the vector cores.  With `BC = 16` that puts **74%** of `A_qk` on the
cube (1,536 of 2,080 non-zero entries) and leaves 26% element-wise.

`BC = 16` is **not** a capacity choice -- `BC = 32` also fits UB (186,112 B).
It is the trade between how much lands on the cube and the 16-element fractal
granularity below which gemm efficiency falls off.

---

## 4. Data movement

Vector and cube reach different memories:

```
Vector:  GM <-> UB
Cube:    GM -> L1 -> L0A/L0B -> cube -> L0C -> GM
```

UB is **per-AIV**, so anything one vector core produces for the other core, or
for the cube, is handed over through GM.  That is what every `ws_*` workspace
is for; the number of workspaces per stage tracks the number of cross-engine
handoffs (0, 0, 0, 2, 4, 5).

A `copy_ub_to_l1` primitive does exist in the runtime, but it is `half`-only,
and several of the intermediates have to be written to GM anyway because they
are also outputs, so the current implementation routes through GM uniformly.

Workspaces are listed in `workspace_idx` and are framework-allocated, which
means they arrive as **dirty memory**.  Every one of them is fully written
before it is read inside the same iteration; a workspace that needed zeroing
could not be declared this way.

---

## 5. Known gaps

* **No performance data.** No msprof run has been made, so this directory makes
  no speed claim anywhere.  The `Optimize Results` table that the sibling GDN
  README carries is deliberately absent rather than filled with estimates.
* **Stage 2 leaves the cube idle.** The anchored decomposition already used by
  stage 6 applies here as well: anchor each row sub-block, hand the off-diagonal
  strips to `T.gemm_v0`, keep the element-wise path for the diagonal blocks.
* **No tail block, no varlen.** All six host wrappers assert `SEQ % C == 0`;
  ragged batches and `cu_seqlens` are not handled.  The reference layer does
  handle a tail, by zero-padding on the host — that route is not open to the
  kernels, because doing the padding host-side is exactly the hidden cost the
  acceptance gate forbids.

  The route that *is* open, and is the next round, is to opt into the tail-block
  support the framework already provides.  `compute_valid_extent`
  (`src/op/ascend.cc:410`) clamps `validRow` / `validCol` on every GM copy to
  `shape - offset`, and `T.copy(..., pad_value=)`
  (`tilelang/language/copy_op.py:262`) fills the unused part of the destination
  — the same `valid_shape` + `fillpad` pair the official PyPTO implementation
  uses, already wired up and covered by
  `testing/python/language/test_tilelang_ascend_language_tail_block.py`.
  `examples/gemm/example_gemm_tail_block_developer.py` is a working example:
  a `T.ceildiv` grid, full-size tiles, ordinary `bx * block_M` indexing, and no
  special-casing anywhere.

  So the change here is `SEQ // C` → `ceildiv(SEQ, C)` plus explicit handling at
  the three places the framework does *not* cover: single-row reads whose token
  index sits on a unit-extent axis (`find_active_dim_indices` only bounds-checks
  the last two *active* dims, so those are unguarded), UB tail rows that reach
  `exp()` before being consumed by the cube, and the chunk decay in stage 5,
  which must be read at the last *valid* token rather than at row `C - 1`.
* **Zero-length sequences are supported**, and are worth calling out separately
  because they are *not* covered by the rule above: `0 % C == 0`, so `SEQ == 0`
  passes every divisibility guard.  Left alone it would launch a zero-block grid
  per stage and return allocated-but-never-written memory — a silent wrong
  answer rather than a loud failure.  Each host wrapper therefore tests for it
  explicitly and returns without touching the device.  The contract is that no
  token was consumed: token-axis outputs are empty, and `final_state` is
  `initial_state` unchanged (zeros when none was given), copied rather than
  aliased so a caller relaying it cannot mutate its own input.
* **Six separate launches.** The stages are not fused.
* **Forward only.**
