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
`kda_l1_full.py` chains all six.

| # | Stage | File | Computes | Engine |
|:-:|---|---|---|:-:|
| 1 | `chunk_cumsum` | `kda_chunk_cumsum.py` | $\Gamma_{t,d}=\sum_{s=t_0}^{t}g_{s,d}$, restarted at every chunk boundary | Vector |
| 2 | `chunk_scaled_dot_kkt` | `kda_chunk_scaled_dot_kkt.py` | $L_{ij}=\beta_i\sum_d k_{i,d}k_{j,d}\,e^{\Gamma_{i,d}-\Gamma_{j,d}}$ for $j<i$ | Vector |
| 3 | `solve_tril` | `kda_solve_tril.py` | $\mathbf A=(\mathbf I+\mathbf L)^{-1}$ by row-wise forward substitution | Vector |
| 4 | `wy_fast` | `kda_wy_fast.py` | UT transform: $\mathbf U=\mathbf A\,\mathrm{Diag}(\beta)\mathbf V$, $\mathbf W=\mathbf A\,\mathrm{Diag}(\beta)(\mathbf K\odot e^{\Gamma})$ | Vector + Cube |
| 5 | `chunk_h` | `kda_chunk_h.py` | $\mathbf V'=\mathbf U-\mathbf W\mathbf S$, then $\mathbf S\leftarrow\mathrm{Diag}(e^{\Gamma_C})\mathbf S+\mathrm{kg}^{\top}\mathbf V'$ with $\mathrm{kg}=\mathbf K\odot e^{\Gamma_C-\Gamma}$ | Vector + Cube |
| 6 | `chunk_o` | `kda_chunk_o.py` | $\mathbf O=(s\mathbf Q\odot e^{\Gamma})\mathbf S_n+\mathbf A^{qk}\mathbf V'$, $A^{qk}_{ij}=\sum_d q_{i,d}k_{j,d}e^{\Gamma_{i,d}-\Gamma_{j,d}}$ for $j\le i$ | Vector + Cube |

The pipeline contains exactly seven `T.gemm_v0` calls: two in `wy_fast`, two in
`chunk_h`, three in `chunk_o`. Stages 1–3 have none.

* **Stage 2 has no matmul by construction, not by omission.** With a
  per-channel gate the decay sits inside $\sum_d$, and the causal mask has to be
  folded into the exponent *before* `exp()` — masking after `exp()` lets the
  $j>i$ half overflow to $\pm\infty$ and then $0\times\infty=\mathrm{NaN}$
  poisons the half that is kept. Folding the mask into the exponent destroys
  row/column separability, so the contraction is evaluated one output row at a
  time on the vector cores.
* **Stage 6 recovers the Cube by anchored blocking.** Each block of `BC = 16`
  rows is anchored at its first row; on the strictly-below-anchor columns both
  folded factors $e^{\Gamma_i-\Gamma_{ar}}$ and $e^{\Gamma_{ar}-\Gamma_j}$ are
  bounded by 1, so the off-diagonal strips go to `T.gemm_v0` and only the
  diagonal blocks fall back to the vector cores.
* **Stage 5 is the only chunk-serial stage.** Its grid is `B * HV * BV_NUM` and
  deliberately contains no chunk axis; the state stays resident in UB across the
  whole `T.serial(N)` loop.
* Stages 4–6 hand operands from Vector to Cube through GM workspaces guarded by
  `set_cross_flag` / `wait_cross_flag`, because there is no UB → L1 path on 910B.

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

```
kda/
├── __init__.py                   # empty; marks the package and keeps CI from executing it
├── kda_chunk_cumsum.py           # stage 1  + self-test
├── kda_chunk_scaled_dot_kkt.py   # stage 2  + self-test
├── kda_solve_tril.py             # stage 3  + self-test
├── kda_wy_fast.py                # stage 4  + self-test
├── kda_chunk_h.py                # stage 5  + self-test
├── kda_chunk_o.py                # stage 6  + self-test
├── kda_l1_ref.py                 # pure-PyTorch chunkwise reference + per-stage goldens
├── kda_l1_full.py                # the six stages chained, checked against two goldens
│
├── kda_recurrent.py              # the recurrent decode kernel  + self-test
├── test_kda_recurrent.py         # decode acceptance test (incl. the FLA cross-check)
├── kda_ref.py                    # pure-PyTorch token-by-token recurrence + make_inputs
│
├── bench.sh                      # msprof harness -- provided, never run (see "Not yet done")
├── design.md                     # why each stage partitions and moves data the way it does
└── README.md
```

Two forward paths ship here. `kda_recurrent.py` is the **decode** path: one
token at a time, carrying the `[K, V]` state, grid `B * HV`, entirely on the
vector cores because every token-level operation is matrix-vector shaped
(`M = 1`) and cannot fill the Cube. The six `kda_chunk_*` stages are the
**prefill** path. `kda_ref.py` is the CPU twin of the decode kernel and is the
acceptance golden for the chunkwise pipeline — the decode path was frozen
first, and the chunkwise decomposition is checked against it rather than only
against another chunkwise implementation.

`kda_l1_ref.py` and `kda_l1_full.py` both use the token-by-token L0 recurrence in
`kda_ref.py` as their ground truth, and take `make_inputs` from it. It ships in
this directory, so nothing outside the directory has to be present for the tests
to run. (The loader still falls back to `../kda`, `../../examples/kda` and
`../../kda` for a tree that keeps L0 as a separate example.)

---

## Usage

### Full pipeline

```python
from kda_l1_full import kda_chunk_fwd

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
non-zero. Stages 1–6 and `kda_l1_full.py` require an Ascend NPU;
`kda_l1_ref.py` is pure PyTorch and runs on CPU alone.

```bash
# reference layer only, no NPU needed:
#   chunkwise vs L0 recurrence, state relay, and a demonstration that the
#   naive one-shot exponent fold produces non-finite values
python kda_l1_ref.py

# per-stage self-tests, each against its golden from kda_l1_ref.stage_tensors()
python kda_chunk_cumsum.py
python kda_chunk_scaled_dot_kkt.py
python kda_solve_tril.py
python kda_wy_fast.py
python kda_chunk_h.py
python kda_chunk_o.py

# the six stages chained, vs both goldens, plus the two bit-exactness invariants
python kda_l1_full.py
```

Goldens are always computed on CPU in fp32. On device, `einsum` dispatches to a
matmul with reduced-precision accumulation and drifts two references that should
be bit-identical by roughly `3e-4` — the same order as the quantity being
measured.

`kda_l1_ref.make_inputs(B, SEQ, H, HV, K, V, device=..., dtype=..., gate=...)`
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
| `SEQ % C == 0` | all six stages and `kda_chunk_fwd` |
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

Verified on Ascend 910B. Correctness only — see "Not yet done" below.

**Full pipeline vs the L0 token-by-token recurrence** (`test_vs_both_goldens`,
18 configurations covering both gate extremes, GVA, `K != V`, the K3 spec,
`B = 4`, the single-chunk case `SEQ == C` and a non-zero `initial_state`): all
pass, with a relative error below **`1e-3` in fp16** and **`7e-3` in bf16**. The
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
`kda_l1_ref.stage_tensors()` and compared against the matching entry:

| Stage | Golden | Threshold | Cases |
|---|---|---|:-:|
| 1 `chunk_cumsum` | `["G"]` | `rel < 1e-5`, all finite | 12 |
| 2 `chunk_scaled_dot_kkt` | `["L"]` | fp16 `5e-3` / bf16 `3e-2` | 16 |
| 3 `solve_tril` | `ref_solve_tril()` and `["A"]` | adaptive (below) | 16 |
| 4 `wy_fast` | `["W"]`, `["U"]` | fp16 `5e-3` / bf16 `3e-2` | 8 |
| 5 `chunk_h` | `["states"]`, `["Vt"]`, `["SF"]` | fp16 `2e-2` / bf16 `6e-2` | 10 |
| 6 `chunk_o` | `["o"]` | fp16 `3e-2` / bf16 `6e-2` | 15 |

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

## Not yet done

* **No performance data whatsoever.** `msprof` has never been run on any stage
  or on the pipeline. There is therefore no latency, no throughput, no bandwidth
  figure and **no speed claim anywhere in this README or in the source
  comments** — not against GDN, not against any other KDA implementation. Any
  such number would have to be measured first.
* **No tail block, no varlen / `cu_seqlens` support.** All six host wrappers
  assert `SEQ % C == 0`, so ragged tails are rejected rather than padded. The
  pure-PyTorch `kda_l1_ref.kda_chunk_ref` does pad the token axis and is tested
  at `SEQ = 70` and `SEQ = 33`, but `stage_tensors()` and the kernels do not.
  Host-side padding is not an option for the kernels: it is exactly the hidden
  cost the acceptance gate rules out.
* The six stages are six kernel launches; every inter-stage tensor and all
  eleven cross-core workspaces (2 in stage 4, 4 in stage 5, 5 in stage 6)
  round-trip through GM. Fusing them is future work.
* Backward is not part of this directory.
