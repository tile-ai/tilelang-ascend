# Kimi Delta Attention (KDA)

Kimi Delta Attention is the linear-attention layer of Kimi Linear and Kimi K3. It keeps a
`[K, V]` hidden state $\mathbf S_t$ and updates it once per token with a gated
delta rule, so a whole sequence is processed in $O(L)$ state updates instead of
an $O(L^2)$ attention matrix:

$$\mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^{\top})\,\mathrm{Diag}(\alpha_t)\,\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^{\top},\qquad \mathbf o_t=\mathbf S_t^{\top}(s\cdot\mathbf q_t).$$

The forget gate enters in the log domain, $g=\ln\alpha\le 0$, and the query is
pre-scaled by $s=K^{-1/2}$.

**KDA = [GDN](../gdn) with the scalar gate replaced by a per-channel vector
gate — that is the only mathematical difference.** GDN carries one $\alpha_t$
per token; KDA carries $K$ of them, one per state row, so $\mathrm{Diag}(\alpha_t)$
replaces GDN's scalar multiply. Everything downstream follows from that single
change: the cumsum widens from a scalar chain to a $K$-wide vector chain, the
decay factor $e^{\Gamma_{i,d}-\Gamma_{j,d}}$ moves *inside* the sum over $d$ and
can no longer be hoisted out of a matmul, and the row broadcasts of `wy_fast` /
`chunk_h` become full elementwise products. `solve_tril` is unchanged, because
the gate is already baked into $L$ before it runs.

---

## The six-stage chunked pipeline

The sequence is cut into chunks of $C$ tokens. Stages 1–4 are chunk-parallel,
stage 5 carries the state serially across chunks, stage 6 reads it back.
`kda_full.py` chains all six.

| # | Stage | File | Computes | Engine |
|:-:|---|---|---|:-:|
| 1 | `chunk_cumsum` | `kda_chunk_cumsum.py` | $\Gamma_{t,d}=\sum_{s=t_0}^{t}g_{s,d}$, restarted at every chunk boundary | Vector |
| 2 | `chunk_scaled_dot_kkt` | `kda_chunk_scaled_dot_kkt.py` | $L_{ij}=\beta_i\sum_d k_{i,d}k_{j,d}\,e^{\Gamma_{i,d}-\Gamma_{j,d}}$ for $j<i$ | Vector + Cube |
| 3 | `solve_tril` | `kda_solve_tril.py` | $\mathbf A=(\mathbf I+\mathbf L)^{-1}$ by a doubling Neumann series on the cube; the row-wise forward substitution in the same file is the fp32 / $C<16$ fallback | Vector + Cube |
| 4 | `wy_fast` | `kda_wy_fast.py` | UT transform: $\mathbf U=\mathbf A\,\mathrm{Diag}(\beta)\mathbf V$, $\mathbf W=\mathbf A\,\mathrm{Diag}(\beta)(\mathbf K\odot e^{\Gamma})$ | Vector + Cube |
| 5 | `chunk_h` | `kda_chunk_h.py` | $\mathbf V'=\mathbf U-\mathbf W\mathbf S$, then $\mathbf S\leftarrow\mathrm{Diag}(e^{\Gamma_C})\mathbf S+\mathrm{kg}^{\top}\mathbf V'$ with $\mathrm{kg}=\mathbf K\odot e^{\Gamma_C-\Gamma}$ | Vector + Cube |
| 6 | `chunk_o` | `kda_chunk_o.py` | $\mathbf O=(s\mathbf Q\odot e^{\Gamma})\mathbf S_n+\mathbf A^{qk}\mathbf V'$, $A^{qk}_{ij}=\sum_d q_{i,d}k_{j,d}e^{\Gamma_{i,d}-\Gamma_{j,d}}$ for $j\le i$ | Vector + Cube |

The pipeline contains sixteen `T.gemm_v0` calls on the default path: one in
`chunk_scaled_dot_kkt`, eight in `solve_tril`, two in `wy_fast`, two in
`chunk_h`, three in `chunk_o`. Only stage 1 has none.

* **Stage 2 is not a matmul in its natural form.** With a
  per-channel gate the decay sits inside $\sum_d$, and the causal mask has to be
  folded into the exponent *before* `exp()` — masking after `exp()` lets the
  $j>i$ half overflow to $\pm\infty$ and then $0\times\infty=\mathrm{NaN}$
  poisons the half that is kept. Folding the mask into the exponent destroys
  row/column separability, so in that form the contraction has to be evaluated
  one output row at a time on the vector cores. Anchored blocking recovers the
  cube anyway -- the same construction stage 6 uses, described next -- which is
  what moved this stage from 19.1% to 67.6% of the reference (`bench_mark.md`).
  The diagonal blocks are the part that stays on the vector cores, and `route_b`
  moves those too -- in this stage and in stage 6, which uses the same
  construction.
* **Stage 6 recovers the Cube by anchored blocking.** Each block of `BC = 16`
  rows is anchored at its first row; on the strictly-below-anchor columns both
  folded factors $e^{\Gamma_i-\Gamma_{ar}}$ and $e^{\Gamma_{ar}-\Gamma_j}$ are
  bounded by 1, so the off-diagonal strips go to `T.gemm_v0` and only the
  diagonal blocks fall back to the vector cores.
* **Stage 5 is the only chunk-serial stage.** Its grid is `B * HV * BV_NUM` and
  deliberately contains no chunk axis; the state stays resident in UB across the
  whole `T.serial(N)` loop.
* Stages 4–6 hand operands from Vector to Cube through GM workspaces guarded by
  `set_cross_flag` / `wait_cross_flag`. A `copy_ub_to_l1` template does exist
  (`src/tl_templates/ascend/common.h`, `half` only), but nothing in the language
  surface reaches it: a `T.copy` from a UB buffer to an L1 buffer lowers to
  `copy_ub_to_gm` followed by `copy_gm_to_l1`. Checked by dumping the generated
  AscendC for a kernel that writes UB straight into L1 -- the emitted count of
  `copy_ub_to_l1` is zero. So a Vector to Cube handoff costs a GM round trip
  whether or not the two halves live in the same kernel, which is why fusing
  stages would not remove one.

---

## Tensor layout

The external interface follows **FLA's `[B, SEQ, HV, D]` layout, not upstream
GDN's `[B, H, L, D]`**. This is the layout the KDA model code already hands
over, so the host wrappers do no transposes, reshapes or state staging at all —
they only pad `beta` to a 32B slot, build the constant masks and look up the
dtype. Moving the layout adaptation onto the host would hide kernel cost there.

The price is that the head axis sits *between* the token axis and the head-dim
axis, so every `[C, D]` tile is a strided transfer: `D` contiguous elements per
row, `HV * D` (or `H * K` for the qk-head tensors) elements between rows. Every
tile load therefore writes the token range out as an explicit slice:

```python
T.copy(G[bz, t0 : t0 + C, hv, 0:K], g_ub)   # region [1, C, 1, K] -> one strided DataCopyPad
```

> ⚠️ Writing `T.copy(G[bz, t0, hv, 0], g_ub)` instead **compiles, runs, and
> produces wrong results**: the region is inferred from the *trailing* dims, so
> the `C` extent lands on the head axis and the copy reads `C` consecutive heads
> of one token. Single-row (1-D) reads need no slice and use the bare form.

### Inputs and outputs

`N = SEQ // C` is the chunk count, `GRP = HV // H`; value head `hv` reads qk head
`hq = hv // GRP` (GVA).

| Tensor | Shape | dtype |
|---|---|---|
| `q`, `k` | `[B, SEQ, H, K]` | fp16 / bf16 |
| `v` | `[B, SEQ, HV, V]` | fp16 / bf16 |
| `g` | `[B, SEQ, HV, K]` | **fp32**, log-domain, `g <= 0` |
| `beta` | `[B, SEQ, HV]` | fp16 / bf16, read as fp32 |
| `initial_state` | `[B, HV, K, V]` | **fp32**, optional |
| `o` | `[B, SEQ, HV, V]` | same as `q` |
| `final_state` | `[B, HV, K, V]` | **fp32** |
| `scale` | scalar | defaults to `K ** -0.5`, applied to `q` |

### Inter-stage tensors

| Tensor | Shape | dtype | Produced by | Consumed by |
|---|---|:-:|:-:|---|
| `G` ($\Gamma$) | `[B, SEQ, HV, K]` | fp32 | 1 | 2, 4, 5, 6 |
| `L` | `[B, SEQ, HV, C]` | dtype | 2 | 3 |
| `A` | `[B, SEQ, HV, C]` | dtype | 3 | 4 |
| `W` / `U` | `[B, SEQ, HV, K]` / `[B, SEQ, HV, V]` | dtype | 4 | 5 |
| `states` | `[B, HV, N, K, V]` | dtype | 5 | 6 |
| `V'` | `[B, SEQ, HV, V]` | dtype | 5 | 5 (Cube read-back), 6 |
| `SF` | `[B, HV, K, V]` | fp32 | 5 | user (relay) |

`G` stays fp32 from stage 1 to stage 6 and is never rounded. `SF` is fp32
because it is the user-facing relay value — rounding it would make a two-segment
run disagree with a one-shot run; `states` is dtype because it only ever feeds
the Cube.

`beta` is padded on the host to `[B, SEQ, HV, 8]` fp32 with the value in lane 0.
A 4-byte `[1]` UB buffer misaligns every allocation after it ("The UB address
accessed by the VEC instruction is not aligned"); the seven padding zeros are
load-bearing, since the kernels recover lane 0 as the row sum.

---

## Directory contents

The layout mirrors the GDN example next door: the stage kernels live in a
subdirectory, the driver that chains them sits one level up.

```
linear_attention_and_rnn/
├── gdn_full.py                    # upstream, for reference
├── kda_full.py                    # the six stages chained, checked against two goldens
└── kda/
    ├── __init__.py                # empty; makes kda_full.py's imports a package import
    ├── kda_chunk_cumsum.py        # stage 1  + self-test
    ├── kda_chunk_scaled_dot_kkt.py# stage 2  + self-test
    ├── kda_solve_tril.py           # stage 3 dispatch  + self-test
    ├── kda_wy_fast.py             # stage 4  + self-test
    ├── kda_chunk_h.py             # stage 5  + self-test
    ├── kda_chunk_o.py             # stage 6  + self-test
    ├── kda_chunk_ref.py            # chunkwise PyTorch reference + per-stage goldens
    ├── kda_varlen.py               # cu_seqlens bookkeeping, shared by both layers
    │
    ├── kda_recurrent.py           # the recurrent decode kernel  + self-test
    ├── test_kda_recurrent.py      # decode acceptance test (incl. the FLA cross-check)
    ├── kda_ref.py                 # pure-PyTorch token-by-token recurrence + make_inputs
    │
    ├── bench.sh                    # msprof harness, per stage and for the pipeline
    ├── bench_mark.md               # measured results against the AscendC operator
    ├── design.md                  # why each kernel partitions and moves data the way it does
    └── README.md                  # this file
```

`kda_full.py` imports the stages as `from kda.kda_chunk_h import chunk_h`, which
is what `__init__.py` is for. The stage files themselves import each other and
the reference layer **flat** (`import kda_chunk_ref`), because CI executes every
`.py` in the tree as a standalone script and a package-relative import would not
resolve that way. `kda_full.py` puts `kda/` on `sys.path` before importing them
so both styles resolve to the same module objects.

Two forward paths ship here. `kda_recurrent.py` is the **decode** path: one
token at a time, carrying the `[K, V]` state, grid `B * HV`, entirely on the
vector cores because every token-level operation is matrix-vector shaped
(`M = 1`) and cannot fill the Cube. The six `kda_chunk_*` stages are the
**prefill** path. `kda_ref.py` is the CPU twin of the decode kernel and is the
acceptance golden for the chunkwise pipeline — the decode path was frozen
first, and the chunkwise decomposition is checked against it rather than only
against another chunkwise implementation.

`kda_chunk_ref.py` and `kda_full.py` both use the token-by-token recurrence in
`kda_ref.py` as their ground truth, and take `make_inputs` from it. It ships in
this directory, so nothing outside `linear_attention_and_rnn/` has to be present
for the tests to run.

---

## Usage

### Full pipeline

```python
from kda_full import kda_chunk_fwd

o, final_state = kda_chunk_fwd(q, k, v, g, beta, C=64, BC=16,
                               scale=None,            # defaults to K ** -0.5
                               initial_state=None,    # [B, HV, K, V] fp32
                               output_final_state=True)
```

All inputs must be contiguous. The wrapper asserts contiguity rather than
calling `.contiguous()` for you: a token-axis slice such as `q[:, :cut]` keeps
the original `stride[0]` and is a non-contiguous view, and repairing it on the
host would be a full copy of every input hidden behind the kernel.

### Running the tests

Every file is executable and prints `Kernel Output Match!` on success, or exits
non-zero. Stages 1–6 and `kda_full.py` require an Ascend NPU;
`kda_chunk_ref.py` is pure PyTorch and runs on CPU alone.

```bash
cd examples/linear_attention_and_rnn

# reference layers only, no NPU needed:
#   chunkwise vs the recurrence, state relay, zero-length sequences, and a
#   demonstration that the naive one-shot exponent fold goes non-finite
python kda/kda_chunk_ref.py
python kda/kda_ref.py

# the decode path
python kda/kda_recurrent.py
python kda/test_kda_recurrent.py     # also cross-checks FLA, if it is installed

# per-stage self-tests, each against its golden from kda_chunk_ref.stage_tensors()
python kda/kda_chunk_cumsum.py
python kda/kda_chunk_scaled_dot_kkt.py
python kda/kda_solve_tril.py
python kda/kda_wy_fast.py
python kda/kda_chunk_h.py
python kda/kda_chunk_o.py

# the six stages chained, vs both goldens, plus the bit-exactness invariants
python kda_full.py
```

Goldens are always computed on CPU in fp32. On device, `einsum` dispatches to a
matmul with reduced-precision accumulation and drifts two references that should
be bit-identical by roughly `3e-4` — the same order as the quantity being
measured.

`kda_chunk_ref.make_inputs(B, SEQ, H, HV, K, V, device=..., dtype=..., gate=...)`
builds test inputs at four gate settings: `keep` ($\alpha\to1$), `normal`
(logsigmoid), `forget` (bounded, $\min\Gamma_C\approx-209$) and `extreme`
(unbounded, $\min\Gamma_C\approx-841$).

---

## Supported configurations

**dtypes.** `q`, `k`, `v`, `beta` in fp16 or bf16; `g`, `initial_state` and
`final_state` in fp32. The dtype is threaded from the inputs into the kernel
templates, never hardcoded. `solve_tril` additionally accepts fp32 in and fp32
out when called on its own.

**Constraints asserted by the host wrappers** (`VEC_NUM = 2`: one Cube and two
Vector cores per AI Core on 910B):

| Constraint | Where |
|---|---|
| `SEQ % C == 0` | **lifted** -- a ragged tail chunk is zero-filled in the kernel |
| `HV % H == 0` (GVA) | stages 2, 4, 5, 6 and `kda_chunk_fwd` |
| `K % (VEC_NUM * 8) == 0`, i.e. `K % 16 == 0` | stage 1 (UB row pitch must stay 32B-aligned) |
| `K % 16 == 0` | stage 2 |
| `K % 16 == 0` and `V % 16 == 0` | stage 6 |
| `C % 2 == 0` and `C % 16 == 0` | stage 2 |
| `C % 16 == 0` and `C <= 64` | stage 3 |
| `C % (VEC_NUM * 16) == 0`, i.e. `C % 32 == 0` | stage 4 |
| `C % (BC * VEC_NUM) == 0`, i.e. `C % 32 == 0` at `BC = 16` | stage 6 |
| `C % 2 == 0`, `K % 2 == 0` | stage 5 (the two vector cores split `C` and `K`) |
| `K % BK == 0`, `BK % 16 == 0`, `V % BV == 0`, `BV % 16 == 0` | stage 4 (`BK`/`BV` default to `K`/`V`) |
| `V % BV == 0`, `BV % 16 == 0` | stage 5 (`BV` defaults to `min(V, 64)`) |
| `Kt`, `W`, `U` share one dtype; `A.dtype == k.dtype` | stages 4, 5 |
| `Q`, `Kt`, `V'`, `states` share one dtype | stage 6 |
| inputs contiguous | stages 3, 5 and `kda_chunk_fwd` |
| UB footprint within `196352` B (stage 5 keeps a `16384` B margin for compiler temporaries) | stages 2, 5, 6 |

Taken together this leaves **`C ∈ {32, 64}`**. `HV` may be odd or not a power of
two (`HV = 1, 3, 6` are all exercised); stage 1 splits the `K` axis rather than
the head axis precisely so that no parity constraint on `HV` exists.

Shapes exercised by the tests: `B = 1, 2`; `H = 1, 2`; `HV = 1 … 6`;
`K = V = 64`, `K = V = 128` (the K3 spec), and `K != V` (64/128);
`SEQ = 32 … 256`.

> `C = 128` with `K = 128` does not fit: the three `[C, K]` fp32 tiles alone
> need 196 608 B against the 196 352 B UB limit. The host asserts instead of
> letting it become an aicore exception.

---

## Accuracy status

Verified on `Ascend910_9362` (A3, 20 Cube / 40 Vector cores).  Nothing here has
been run on an A2-class 910B1/B2.  Measured results are in `bench_mark.md`.

**Full pipeline vs the L0 token-by-token recurrence** (`test_vs_both_goldens`):
relative error below **`1e-3` in fp16** and **`7e-3` in bf16**.

What ships here is three configurations -- a GVA workhorse, a ragged tail, and
the bfloat16 pass -- because a distinct shape costs about 6.2s of JIT compile on
board and a repeat costs 0.2s, so the wall time of an example is its
distinct-shape count and little else. The wider sweep that the paragraph above
rests on -- both gate extremes, `K != V`, the K3 spec at `H = 96`, `B = 4`, the
single-chunk case `SEQ == C`, a non-zero `initial_state` -- is the local
regression rather than the shipped file, and its results are in `bench_mark.md`. The
two goldens — the L0 recurrence and the chunkwise reference — sit
`3e-7 … 4e-6` apart, and the kernel output is *equidistant* from both. Since the
kernel shares its decomposition with the chunkwise reference and not with the
recurrence, equidistance says the residual is fp16 accumulation noise rather
than algorithmic bias.

**Bit-exact invariants.** All three are asserted as exact equality, not as a
tolerance — a test that permits drift cannot support a bit-identical claim.

| Test | Cases | Criterion | Result |
|---|:-:|---|---|
| whole sequence vs two-segment relay through `final_state` | 4 | `rel == 0.0`, no tolerance | bit-identical |
| zero `initial_state` vs no `initial_state` | 2 | `max\|diff\| == 0.0`, no tolerance | bit-identical |
| zero-length sequence: `final_state` vs `initial_state` | 2 | `max\|diff\| == 0.0`, no tolerance | bit-identical |

The relay is exact rather than merely close because every cut lands on a chunk
boundary: chunks are independent given their entry state, so the segmented run
performs the same arithmetic in the same order as the one-shot run. Any
difference at all would mean the entry state did not survive the round trip
through `final_state` / `initial_state`.

**Zero-length sequences** (`test_empty_sequence`) are checked at both levels:
`kda_chunk_fwd` and each of the six stage wrappers, for output shape, for the
bit-identical state pass-through, and for `final_state` being a copy rather than
an alias of `initial_state`. `0 % C == 0`, so this case passes every
divisibility guard and would otherwise launch zero-block grids over unwritten
memory.

**Per-stage self-tests**, each fed the golden inputs from
`kda_chunk_ref.stage_tensors()` and compared against the matching entry:

`Cases` is what the shipped `__main__` validates. Each file covers three
distinct shapes -- a GVA workhorse at `C = 64`, a ragged tail, and a varlen batch
with an empty sequence in the middle -- and takes its second gate and its
bfloat16 pass on a shape already compiled, which is why the counts exceed three
while the wall time does not. Stage 6 is larger because it ships two routes and
`route_b` builds a different kernel.

| Stage | Golden | Threshold | Cases |
|---|---|---|:-:|
| 1 `chunk_cumsum` | `["G"]` | `rel < 1e-5`, all finite | 4 |
| 2 `chunk_scaled_dot_kkt` | `["L"]` | fp16 `5e-3` / bf16 `3e-2` | 5 |
| 3 `solve_tril` | `ref_solve_tril()` and `["A"]` | adaptive (below) | 6 |
| 4 `wy_fast` | `["W"]`, `["U"]` | fp16 `5e-3` / bf16 `3e-2` | 4 |
| 5 `chunk_h` | `["states"]`, `["Vt"]`, `["SF"]` | fp16 `2e-2` / bf16 `6e-2` | 5 |
| 6 `chunk_o` | `["o"]` | fp16 `3e-2` / bf16 `6e-2` | 11 |

bf16 tolerances are roughly 8× the fp16 ones because bf16 keeps 8 mantissa bits
against fp16's 11 and both gemm operands are rounded once.

`solve_tril` is the one stage with an adaptive bound, because matrix inversion
is condition-number sensitive: with $\hat L$ the rounded input the kernel
actually received, it requires
$e_{\text{kern}}=\mathrm{rel}(\text{got},\mathrm{ref}(\hat L))<\text{TOL}$ **and**
$e_{\text{gold}}=\mathrm{rel}(\text{got},A)\le 4\,e_{\text{sens}}+\text{TOL}$,
where $e_{\text{sens}}=\mathrm{rel}(\mathrm{ref}(\hat L),A)$ isolates the
amplification of input rounding through $A\,\mathrm dL\,A$. `TOL` is
`5e-3` / `3e-2` / `1e-5` for fp16 / bf16 / fp32; on the two fp32 cases the input
is exact, $e_{\text{sens}}=0$, and the criterion tightens to `1e-5` on its own.

---

## Environment switches

Four, all opt-in, none of them changing the default: with nothing set the
operator runs the numerics it shipped with. They exist so an A/B can be taken
without editing source, which is how every figure in `bench_mark.md` was
produced.

All four parse the same way -- `1` / `true` / `yes` / `on`, case-insensitively --
so an unrecognised value means *off*. That is the safe direction in both places
it matters: an opt-in approximation stays off, and a cube path that has an exact
fallback falls back.

| Switch | Default | Read at | Effect |
|---|---|---|---|
| `KDA_ROUTE_B` | off | call time | Overrides the `route_b` argument in both directions. Forced off under `cu_seqlens`, and asking for it there warns rather than silently returning the route A result. |
| `KDA_WY_FIXEDCORE` | off | call time | Overrides `wy_fast`'s `fixed_core` argument. Stage 4 with the grid set to the physical core count; worth 192.3u on that stage at `H = 96`. Fixed length only. |
| `KDA_SOLVE_CUBE` | **on** | call time | Stage 3's cube solver. Off falls back to the row-wise forward substitution in the same file, which is also the fp32 path. |
| `KDA_SOLVE_STEPS` | `2` | import time | Doubling steps in stage 3's Neumann series, 1 to 3. The default covers `L^7` and is measured, not picked: `1` (covering `L^3`) fails the keep gate in the pipeline at 8.260e-03 against a 5e-3 tolerance. |

`route_b` and `fixed_core` are also plain keyword arguments -- `kda_chunk_fwd`
forwards `route_b` to stages 2 and 6, and `wy_fast` takes `fixed_core` -- so a
caller never has to reach for the environment. The variables exist for the
benchmark harness, which has to flip a caller it does not control.

## Status and what is not done

* **Performance is measured, not claimed.** Every figure in `bench_mark.md`
  comes from an `msprof` collection on board, against the hand-written AscendC
  operator built from `gitcode.com/cann/ops-transformer`. No number in this
  directory is an estimate.
* **Two things about the CPU goldens that cost a CI cycle each, so they are
  written down here.**

  `kda_ref.py` and `kda_chunk_ref.py` set `torch.backends.mkldnn.enabled = False`
  at import. They are the goldens, and a golden that quietly computes at reduced
  precision cannot judge a kernel. On an x86 runner oneDNN may take fp32 matmul
  through bfloat16, selected by `ONEDNN_DEFAULT_FPMATH_MODE` in the environment
  rather than by anything here, and it hits only the batched forms a chunkwise
  reference is made of: 2-D `a @ b` is untouched at 5.4e-07 while 3-D `bmm` goes
  to 2.1e-03, 5-D `matmul` to 2.8e-03 and `einsum` to 1.2e-03 -- which lands as
  4.4e-03 to 7.3e-03 against a 1e-5 acceptance threshold. Note that
  `torch.set_float32_matmul_precision("highest")` does *not* defeat it: that is
  already the default and governs a different path.

  `test_varlen_equals_fixed_batch` in `kda_chunk_ref.py` is bounded relatively
  rather than by `torch.equal`. It compares N calls at `B = 1` against one call
  at `B = N`, and the folded batch extent is what BLAS blocks its 5-D matmuls
  over, so the reduction order is a property of the machine rather than of the
  algebra. Measured at 40 threads against an output scale near 0.8: exactly 0 on
  the three shapes it ships, 7.3e-11 on `[128, 128]` at `C = 64` and 1.8e-9 on
  `[512, 512]`. The analogous check in `kda_ref.py` stays exact on purpose --
  that reference is a token-by-token recurrence whose reduction axis is `K`, not
  the batch.

  Both files also print the roll-up of which checks failed as their last line,
  because `examples/bench_test.sh` reports only `tail -n 1` of a failing script.
* **A ragged tail chunk and varlen are both supported.** `SEQ % C != 0` is
  handled by zero-filling the pad rows inside the kernel -- a garbage gate row
  exponentiates to `+inf`, and `0 * inf` is `NaN` landing in a *valid* row's
  reduction, so the fill is load-bearing rather than tidy. `cu_seqlens` goes
  through `kda_varlen.py`, and a batched varlen run is asserted bit-identical to
  running each sequence on its own.
* **`route_b` is off by default.** It puts the diagonal blocks of stages 2 and 6
  on the cube and is worth 3.05x on stage 2 and 2.74x on stage 6 at `H = 96`
  (2.24x and 2.4x at `H = 4` -- this operator's ratios always have to name a head
  count), but it saturates a gate that spans more than
  its clamp inside one block, so it is opt-in rather than automatic. The
  approximation itself is not unusual -- the reference makes the same one, and
  harder: it clamps the same exponent two-sided at 55.45 nats
  (`chunk_kda_fwd_post_wu.h:40`) against 80 nats one-sided here. No exact cube
  route exists for these blocks: writing `exp(G_i - G_j)` as a product of a row
  factor and a column factor forces the row factor's dynamic range to equal the
  block's gate span, and 80 nats is already what a bf16 exponent holds. What is
  opt-in here is the approximation, not the speed -- the exact path is the one
  that ships. See `bench_mark.md`.
* The six stages are six kernel launches; every inter-stage tensor and the
  cross-core workspaces round-trip through GM. The hand-written operator does
  the same -- its `gk`, `aqk`, `akk`, `w`, `u`, `qg`, `kg`, `v_new` and `h` are
  all GM tensors -- so fusion is not where the remaining gap is. It does size
  its Cube-to-Vector scratch by physical core count rather than by logical task
  count, and that was tried here: the grid becomes the core count and each core
  walks its own slice of the tasks. Stage 4 ships it behind `KDA_WY_FIXEDCORE`,
  where it is worth 192.3u -- 11.3% of that stage -- and takes that stage's own
  scratch from `[6144, 64, 128] x 2` fp16 (192.00 MiB) to `[20, 64, 128] x 2`
  (640 KiB), the factor 6144/20 = 307.2. Enumerating every stage's
  `workspace_idx` at this shape gives 1.796 GiB across the six, and 7.5 MiB if
  all six were converted.

  The reason it pays is not the workspace size. Stage 4's grid is
  `B * HV * chunk_num`, 6144 blocks at `H = 96`, and each block runs about 277 ns
  against a per-block prologue -- nine `GlobalTensor` and five `TBuf`
  constructions -- of about 164 ns, all of it on the scalar unit. Both figures are
  core-time per logical task (stage duration / `Block Num`), not the wall time of
  one block. The generated kernel contains no `GetValue` at any shape tried: the
  work is not scalar, the
  *setup* is, and Fixed Core pays it once per core instead of once per task.
  Across the six stages that setup is 4561u of 14459u at `H = 96`, so the same
  change applies to the other five and has not been made.

  One caution for whoever does them. Under Fixed Core a workspace slice belongs to
  the CORE, not to the task, so a core that walks more than one task overwrites
  what the previous one published. A single Vector-to-Cube flag is not enough; a
  back-edge is needed so the cube can say it has finished reading, with the two
  counts balanced exactly or the kernel deadlocks. Without it the failure is
  intermittent and shape-dependent: it appears only once the task count exceeds
  the core count, so a suite whose cases are all smaller than 20 tasks passes it.
* **The gap is software pipelining.** The reference ships two configurations:
  `safeGate = 1` carries a 4-deep pipelined triangular solve and a
  software-pipelined task loop, and runs 1.76x faster than its own
  `safeGate = 0` fallback at `H = 96` while agreeing with it to fp16
  quantisation.

  Double buffering was tried on both halves of stage 6 and neither arm is worth
  shipping, which is itself the useful result. On the cube side it is correct and
  gains nothing -- that half runs at 4.2% mac occupancy and has the slack to
  absorb its own stalls. On the vector side it gains 63u but does not fit UB at
  `K = 128`.

  What the generated code shows is more useful than either number. All eighteen
  `SetFlag` / `WaitFlag` pairs in that kernel are a set followed immediately by
  its own wait, so nothing overlaps anything by construction. Rebuilding the
  stage with the sync inserter switched off -- numerically wrong, but it times
  the barriers -- runs it at 2785.6u against 4024.0u at `H = 96`, so 30.8% of the
  stage is synchronisation. Capturing that needs the flags placed by hand, not a
  second buffer, and that is the next thing to try.

* Backward is not part of this directory.
