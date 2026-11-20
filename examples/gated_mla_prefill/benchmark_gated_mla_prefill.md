**English** | [中文](benchmark_gated_mla_prefill_zh.md)

# Gated MLA Prefill Benchmark

## 1. Operator

```
ql[b,t,h,:] = q_content[b,t,h,:] @ w_k[h]                               # [128] @ [128,512] -> [512]
s[b,t,h,k]  = scale * ( ql[b,t,h,:] . c[b,k,:] + qa[b,t,h,:] . e[b,k,:] )
p[b,t,h,k]  = softmax_k( s[b,t,h,k] )                                    # over visible k only
z[b,t,h,:]  = sum_k p[b,t,h,k] * c[b,k,:]                                # [512], FP32
y[b,t,h,:]  = ( z[b,t,h,:] @ w_v[h]^T ) * sigmoid( gate_logits[b,t,h,:] ) # [128]
```

- The attention block of the Kimi K3 Gated MLA layer, from the seven already
  projected inputs up to but not including W_O. The layer is NoPE, so the 64-wide
  auxiliary cache carries plain key channels rather than rotary ones, and the
  output gate is enabled in the released configuration.
- `scale = (D + E)^-0.5 = 192^-0.5`, matching the official `KimiMLAAttention.scaling`.
- Visibility: key `k` is visible to query row `t` exactly when
  `k < min(query_start + t + 1, kv_len)`. Rows past the valid query length write
  exact zeros.
- dtype: scores, the online softmax state and `z` are FP32; `ql`, the caches, the
  weights, the gate and the output share the input dtype.
- Reference model: [Kimi-K3 `modeling_kimi_linear.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py)
  at revision `c5d1dd4c`, whose released configuration sets `mla_use_nope: true`,
  `mla_use_output_gate: true`, `num_attention_heads: 96`, `kv_lora_rank: 512`,
  `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`. The model
  source is linked for provenance and is not redistributed here.

## 2. Hardware and software

| Item | Value |
|---|---|
| NPU | Ascend 910B3 |
| CANN | 9.2.0 |
| Framework | torch 2.7.1 + torch_npu 2.7.1 |
| Programming mode | Expert (explicit Cube/Vector ownership, L1/L0/UB and cross-core sync) |

## 3. Architecture (three kernel launches)

Unlike a single-kernel operator such as kpool, the three stages here differ
enough in shape and in compute character that they are not forced into one
launch.

| Component | Choice | Reason |
|---|---|---|
| Q absorption | Vector kernel, FP32 accumulation with explicit rounding back to the input dtype | `[128] @ [128,512]` is independent per head; the Cube fractal does not repay the transfers |
| Attention core | Persistent Cube over 20 physical blocks; the online softmax state stays in UB | A grid fixed to the physical core count amortises the per-block prologue |
| Value recovery and gate | Vector kernel, `(Z @ W_V^T) * sigmoid(G)` fused | The gate is per-channel elementwise and rides the same pass as value recovery |
| `block_n` selection | the host selector `public_prefill_block_n`, **not exposed as a tuning knob** | Only the one measured profile selects 128; every other shape selects 64 |
| Mask | whole-tile packed mask when heads are aligned, row-wise otherwise | The whole-tile path removes the per-row comparison |
| Two entry points | `main_pipelined` when `iterations == 1`, `heads % block_m == 0` and `batch == 1`; `main` otherwise | See below |

### Cross-wave Cube/Vector software pipeline

The baseline used a single-buffer four-flag handshake, so Cube and Vector
strictly alternated. PipeUtilization on the same shape, per-core medians:

| | Baseline | Pipelined |
|---|---:|---:|
| Cube blocked on flags | 57.63 of 129.66 us (44%) | **1.19 us (1%)** |
| Vector blocked on flags | 66.95 of 130.55 us (51%) | **23.28 us (25%)** |
| `cube_time` | 14.66 us | **14.66 us** |
| `mte1_time` | 29.07 us | **29.07 us** |

Each core's wait matched the other's busy time, so the two never overlapped. In
the same sample GM->L1 bandwidth was 2.5% and vector bank conflicts 1%, so
**neither bandwidth nor conflicts bound the kernel**.

The pipeline skews the Cube loop by one wave: **the QK of wave w is issued before
the PV of wave w-1**, so the Vector softmax between them overlaps a Cube GEMM.
The three workspaces gain a two-slot dimension and the four flags become six
ready/consumed semaphores paired per workspace, with the consumer priming one
slot before the loop. Both scopes drain their own primed semaphores after the
loop, so no cross flag survives a launch.

`cube_time` and `mte1_time` are unchanged: **only the schedule moved**.

**The domain is a correctness boundary, not a tuning choice.** C and E are
L1-resident under a batch tag, and the skewed loop would let the QK of wave w
reload them into the buffers the pending PV still reads. Lifting `batch == 1`
needs a second C/E slot, about 144 KiB more L1, which has to be confirmed on
hardware. A negative test pins this boundary.

## 4. Performance

Protocol: `msprof op` single-kernel Task Duration, three independent collections
per cell of eight launches each with the first discarded, reported as the median
and half-range. FP16 / PTO. `waves = ceil(Tq * H / block_m / blocks)` with
`blocks = min(20, tasks)`.

| shape | waves | before | this PR | speedup |
|---|---:|---:|---:|---:|
| H32 / Tq128 / Tk128 | 7 | 56.04 ±0.43 | 41.28 ±0.53 | 1.358x |
| H96 / Tq64 / Tk128 | 10 | 76.12 ±0.52 | 55.29 ±0.25 | 1.377x |
| H64 / Tq128 / Tk128 | 13 | 96.73 ±0.20 | 69.70 ±0.19 | 1.388x |
| H96 / Tq128 / Tk64 | 20 | 73.11 ±0.65 | 51.87 ±0.22 | 1.409x |
| H96 / Tq128 / Tk128 | 20 | 140.87 ±0.86 | 99.35 ±0.99 | 1.418x |
| H96 / Tq256 / Tk128 | 39 | 263.36 ±0.48 | 182.68 ±0.83 | 1.442x |
| B2 / H96 / Tq128 / Tk128 | — | 267.22 ±0.61 | 265.16 ±1.74 | 1.008x |
| H96 / Tq128 / **Tk256** | — | 180.85 ±1.32 | 180.44 ±0.32 | 1.002x |
| **H48** / Tq128 / Tk128 | — | 91.45 ±0.62 | 93.09 ±0.39 | 0.982x |

**The trend says more than any single point.** The speedup rises monotonically
with wave count (7 -> 1.358, 10 -> 1.377, 13 -> 1.388, 20 -> 1.418, 39 -> 1.442),
while the two cells at equal wave count (Tk128 and Tk64, both 20) are flat at
1.418 and 1.409 although absolute time differs by nearly 2x. **The gain follows
parallelism, not workload** — exactly the shape a one-wave skew predicts, since
fill and drain cost one wave each for an overhead of `1/waves`.

The last three rows are shapes outside the guard. All land within 0.982–1.008x,
the size of their own half-ranges. **Zero gain outside the domain is the design,
not an omission.**

### Both dtypes and both backends, measured on this file

The table above was collected during development. This one is measured on the
example exactly as submitted, over both dtypes and both backends, by NPUGraph
capture and replay: a graph holding ten calls is captured once, then replayed
twenty times under a pair of device events, three independent runs per cell.
Capture takes compilation and host dispatch out of the measurement.

`msprof op` was tried first and abandoned: all three stages are generated from a
`prim_func` named `main` and therefore share one kernel name, so `--kernel-name`
cannot separate them, and the default `--launch-count 1` captures only the cold
first launch. Graph capture needs no name filter.

| dtype | backend | shape | block_n | attention stage (us) |
|---|---|---|---:|---:|
| FP16 | PTO | B1 / Tq128 / Tk128 | 128 | 92.68 ±0.09 |
| FP16 | AscendC | B1 / Tq128 / Tk128 | 128 | **84.70** ±0.40 |
| BF16 | PTO | B1 / Tq128 / Tk128 | 128 | 116.63 ±0.21 |
| BF16 | AscendC | B1 / Tq128 / Tk128 | 128 | 97.08 ±0.20 |
| FP16 | PTO | B1 / Tq64 / Tk128 | 64 | 79.29 ±0.27 |
| FP16 | AscendC | B1 / Tq64 / Tk128 | 64 | 80.33 ±0.26 |
| BF16 | PTO | B1 / Tq64 / Tk128 | 64 | 89.71 ±0.24 |
| BF16 | AscendC | B1 / Tq64 / Tk128 | 64 | 90.98 ±0.62 |
| FP16 | PTO | B2 / Tq64 / Tk128 | 64 | 145.21 ±0.13 |
| FP16 | AscendC | B2 / Tq64 / Tk128 | 64 | 147.49 ±0.16 |
| BF16 | PTO | B2 / Tq64 / Tk128 | 64 | 167.87 ±0.53 |
| BF16 | AscendC | B2 / Tq64 / Tk128 | 64 | 168.91 ±0.79 |

Three readings, all consistent with the design:

* **BF16 costs more than FP16, and only where the pipeline runs** — 26% more on
  PTO and 15% on AscendC at the measured profile. BF16 takes the P-residual path
  (`use_p_residual = dtype == "bfloat16"`), which adds vector work to the stage
  the skew is hiding.
* **AscendC leads PTO by 9% on FP16 and 17% on BF16 at the measured profile, and
  the two converge to within 1.6% on every other row.** The backends differ only
  where the skew is active, which is the expected signature: off the pipelined
  entry they are issuing the same work in the same order.
* The B2 and Tq64 rows are on the scalar entry, as the `entry` column of the
  ratio table records for the same shapes.

These cells set `query_start = 0`, so **the Tq64 rows here are not comparable
with the Tq64 rows of the ratio table**, which set `query_start = kv_len -
valid_q`. The kernel skips cache blocks that the mask makes entirely invisible
(`k * block_n < c_max_visible`): at Tq64 with `query_start = 0` only the first of
two 64-wide blocks is ever live, while `query_start = 64` makes both live. That
is the whole of the 79.29 us against 139.80 us difference, and it is a real
property of the kernel rather than a discrepancy between the tables.

The same harness measured the full three-launch chain at 15.9–18.1 ms per call
across these cells, dominated by the Q absorption and value/gate stages that this
change does not touch. **That figure has not been attributed to its stages and
should not be read as one** — it is reported only to place the attention core,
which is what this change moves, in the context of the whole chain.

### Step by step (FP16 / PTO, B1/H96/Tq=Tk128)

| Step | Optimization | Task |
|---|---|---:|
| 0 | before | 208.16 us |
| 1 | whole-tile causal mask over a uniform query tile | 193.96 us |
| 2 | single KV block resident across waves within a batch | 176.98 us |
| 3 | exact single-block state identity elimination | 165.86 us |
| 4 | QK L0 K grain 128 | 159.58 us |
| 5 | prefetch into the L1 region of queries whose life has ended | 151.66 us |
| 6 | PTO PV64 / AscendC PV32 | 143.50 us |
| 7 | **cross-wave Cube/Vector software pipeline** | **98.55 us** |
| 8 | **single-step carry of the Cube stage condition** | **97.80 us** |

Steps before 7 are single-process pilot snapshots while 7 and 8 are medians of
three collections, so the two cannot be subtracted as strict increments.

### Ratio against the official reference

Attention core only, against an absorbed `npu_fused_infer_attention_score_v2`,
FP16 / PTO. NPUGraph capture and replay, device events around the replays, three
independent runs per cell; the ratio is taken inside each run so the two arms are
always compared against numbers measured beside each other.

**Both arms qualify before either is timed.** Each is checked against an FP32
reference of the same stage, and a cell that misses the budget reports no ratio
rather than a flattering number. All eight cells below qualified, the candidate
within 1.2–2.4e-04 normalised and the reference within 3.4–6.3e-04.

One detail decides whether this comparison means anything. FIA implements
right-down aligned causal masking only — the device refuses the alternatives with
`Only support sparse 0/3/4/9 when ifa mla is enabled`, and of those only
`sparse_mode 3` accepts this layout. This operator expresses the same alignment
natively through `metadata[b,2]`, `query_start`, so **every cell sets
`query_start = kv_len - valid_q` and both arms then compute the same mask.**
Comparing a top-left aligned candidate against FIA instead compares two different
problems, and shows a normalised error near 1.0 for every shape with Tq != Tk.
Right-down alignment is undefined for Tq > Tk, so no such cell appears.

| shape | block_n | cache iters | entry | this PR | FIA v2 | ratio |
|---|---:|---:|---|---:|---:|---:|
| B1 / H96 / Tq128 / Tk128 | 128 | 1 | **pipelined** | 92.55 | 69.10 | **74.9%** ±0.5 |
| B1 / H96 / Tq64 / Tk128 | 64 | 2 | scalar | 139.80 | 49.39 | 35.3% ±0.2 |
| B1 / H96 / Tq128 / Tk256 | 64 | 4 | scalar | 449.65 | 82.74 | 18.4% ±0.3 |
| B1 / H96 / Tq64 / Tk256 | 64 | 4 | scalar | 264.49 | 56.80 | 21.4% ±0.1 |
| B1 / H96 / Tq128 / Tk512 | 64 | 8 | scalar | 935.48 | 119.75 | 12.8% ±0.2 |
| B1 / H32 / Tq128 / Tk128 | 64 | 2 | scalar | 82.44 | 50.71 | 61.5% ±0.2 |
| B1 / H64 / Tq128 / Tk128 | 64 | 2 | scalar | 140.87 | 57.88 | 41.1% ±0.0 |
| B2 / H96 / Tq128 / Tk128 | 64 | 2 | scalar | 393.41 | 96.45 | 24.5% ±0.1 |

Times are microseconds per call, median of three runs. An independent repeat of
the whole matrix reproduced every ratio within 1.2 points.

**The `entry` column is why these rows must not be read against each other.** The
pipelined entry needs the cache covered in one iteration and a single batch, and
`public_prefill_block_n` returns 128 only for the one profile it has measured, so
**seven of the eight rows run the conservative fallback.** Shape and entry move
together down the table, which is exactly what makes the raw ratios misleading on
their own — H96/Tq128/Tk128 is faster in absolute terms than H64/Tq128/Tk128
despite half again the work, purely because it is the only row on the pipeline.

### What the pipeline is worth, at a fixed shape

To separate the pipeline from the shape, hold the shape still and move `block_n`
alone. Whenever `block_n` covers the cache in one iteration the pipelined entry is
reachable, so one shape can be timed on both entries. `block_n` is a parameter of
`cube_prefill`, so every point here is a legal call; the shipped selector simply
declines to pick the larger values outside its measured profile.

| shape | block_n | cache iters | entry | us | normalised error |
|---|---:|---:|---|---:|---:|
| B1 / H96 / Tq128 / Tk128 | 64 | 2 | scalar | 204.54 ±0.10 | 1.2e-04 |
| B1 / H96 / Tq128 / Tk128 | 128 | 1 | **pipelined** | **92.98** ±0.09 | 1.2e-04 |
| B1 / H96 / Tq64 / Tk128 | 64 | 2 | scalar | 141.94 ±0.40 | 2.1e-04 |
| B1 / H96 / Tq64 / Tk128 | 128 | 1 | **pipelined** | **54.31** ±0.09 | 2.1e-04 |
| B1 / H96 / Tq128 / Tk256 | 64 | 4 | scalar | 452.97 ±0.96 | 2.1e-04 |
| B1 / H96 / Tq128 / Tk256 | 128 | 2 | scalar | 330.05 ±0.52 | 2.0e-04 |
| B1 / H96 / Tq128 / Tk256 | 256 | 1 | — | refused: unsupported attention tile | — |
| B1 / H96 / Tq128 / Tk512 | 64 | 8 | scalar | 942.43 ±12.44 | 2.1e-04 |
| B1 / H96 / Tq128 / Tk512 | 128 | 4 | scalar | 636.84 ±0.61 | 2.3e-04 |
| B1 / H96 / Tq128 / Tk512 | 256 | 2 | — | refused: unsupported attention tile | — |
| B1 / H96 / Tq128 / Tk512 | 512 | 1 | — | refused: unsupported attention tile | — |

**At a fixed shape the pipelined entry is worth 2.20x at Tq128 and 2.61x at
Tq64**, with the normalised error identical on both entries to the digit shown —
the two entries compute the same thing. That is the honest measure of this
change; the ratio table's spread is mostly the selector and the tile limit.

Two consequences follow, and neither is cosmetic:

* **`block_n` >= 256 is refused outright**, so for Tk > 128 the pipelined entry is
  structurally unreachable, not merely untuned. The low ratios on the Tk256 and
  Tk512 rows are a ceiling, and raising them needs a larger attention tile rather
  than a better selector.
* **At B1/H96/Tq64/Tk128 the shipped selector leaves 2.61x on the table.** That
  configuration qualifies at 2.1e-04 and runs in 54.31 us, against the 141.94 us
  the selector's `block_n = 64` actually gets. Widening the selector is not done
  here: its contract is deliberately "measured profile only", the newly reachable
  path carries no test coverage today, and earning that coverage means re-running
  the whole verification chain. It is recorded as the first performance item to
  take up, with the measurement already in hand.

**No ratio here should be read as every calling domain meeting a bar.** Ordinary
non-Graph calls in the same scope are dominated by host overhead and sit near 66%
of FIA; this optimization does not improve them by design.

The matrix and the ablation are driven by harnesses this example does not carry.
The protocol, shapes, mask convention and environment are given here so both can
be reproduced; the example does not claim to ship them.

### What is left

Of the 89.28 us the Cube side is busy, `cube_time` accounts for only 14.66 us and
`mte1_time` for 29.07 us, so **about 45 us is neither arithmetic nor transfer** —
instruction issue and cross-core sync dispatch are the candidates. The ladder
recorded in `examples/flash_attention/fa_opt` attributes 60% -> 72% to reducing
the number of cross-core synchronisations and 72% -> 80% to cutting instruction
count with axpy. Neither is implemented here. Since Cube is the critical path
after step 7 (Cube busy 89.28 us against Vector 68.53 us), the former ranks first.

## 5. Accuracy

### The floor: two independent references

The module carries two reference implementations that **share no contraction
order**:

- **absorbed**, the form the kernel implements: fold `W_K` into the query to get
  `ql`, then contract directly against the latent cache
- **expanded**: materialise per-head keys and values of width 128 from the
  latent cache first, then run ordinary attention

```
absorbed  s = scale * ( (q @ w_k) . c  + qa . e )
expanded  k = w_k @ c ;  s = scale * ( q . k + qa . e )
```

They are algebraically the same, since the inner sum over the latent rank can be
taken in either order, but they are computed by different code. Agreement
between them is therefore evidence that **the absorbed decomposition itself is
right**, not merely that the kernel matches one way of writing it.

Measured on CPU, with no device involved, normalised by the output magnitude:

| Case | dtype | max_abs | normalised | ULP |
|---|---|---:|---:|---:|
| measured profile | fp16 | 6.10e-05 | 1.95e-04 | 0.20 |
| measured profile | bf16 | 2.44e-04 | 7.81e-04 | 0.10 |
| row-wise mask shape | fp16 | 6.10e-05 | 2.08e-04 | 0.21 |
| short query, clamped cache, shifted | fp16 | 3.05e-05 | 3.07e-04 | 0.31 |
| two batches | fp16 | 6.10e-05 | 2.35e-04 | 0.24 |
| single visible key | fp16 | 1.22e-04 | 3.61e-04 | 0.37 |

**The widest gap is 0.37 ULP**, all of them under one unit in the last place.

**This is the floor.** A kernel tolerance tighter than two correct references
disagree by would be measuring the references rather than the kernel. The
`rtol=2^-9`, `atol=2^-10` used below sits well above it and was chosen from it,
not tuned to whatever the implementation happens to produce. A CPU test pins the
floor at one ULP.

### Gates

`test_example_gated_mla_prefill.py`, 28 cases, six of which are the CPU
reference-agreement gate above. Four kinds of gate, held to deliberately
different standards.

**Budgeted** — the chained forward against the FP32 reference in the example
module, position sensitively: rows past the valid query length must be **exactly
zero** before `rtol=2^-9`, `atol=2^-10` is applied to the rest. Covers
FP16/BF16 x PTO/AscendC.

**Analytic** — with `QL = QA = 0` every visible score is zero and a visible key's
probability is `exp(0)`; with `latent_kv[k,:] = k` the PV accumulation is
`sum(0..n-1)` and the divide by the normaliser yields `(n-1)/2`. **Every expected
value is a multiple of 0.5, and a mask that admits one key too many or too few
moves a whole row by exactly 0.5**, so the assertion rounds to the nearest half
and compares exactly: there is no tolerance to argue about. A second assertion
bounds the device drift at 8 ULP of FP32 so rounding cannot hide it. Covers both
mask paths, `query_start` shifts and `kv_len` clamping.

**Bit-exact** — a batch computed beside another must be bit-identical to
computing it alone, and an empty batch must not disturb a live neighbour. A
tolerance would be worse than no test here: the fault these catch is a tile
writing past its own rows into a neighbour, which produces a bounded,
plausible-looking difference.

**Negative** — an invalid shape, a strided operand and a wrong dtype must each
raise a specific exception matching a message fragment, not a generic one; and
the pipelined entry **must not** be compiled at `batch >= 2` or for more than one
KV block. If that ever starts passing, the documented domain has changed.

### What the gates cost CI

| Entry | Content | Wall clock |
|---|---|---:|
| `test_example_gated_mla_prefill.py` | 28 cases, 6 of them CPU-only | 200 s |
| `example_gated_mla_prefill.py` `__main__` | 5 configurations x 2 backends | 233 s |

The two are independent, so the wall clock is about four minutes. The test
figure is from a cold JIT cache, which is what CI starts from; a warm cache
brings it to 173 s. A new shape
costs roughly 6 s of JIT compilation and a repeated one about 0.2 s, so the
figure tracks the number of **distinct** shapes rather than the number of cases;
the configurations above reuse shapes wherever a case only needs a different
dtype or backend.

### What is covered, and what was cut

The shipped cases cover both dtypes against both backends, both mask paths
(whole-tile and row-wise), `query_start` in {0, 1, 7, 64}, `kv_len` in
{1, 40, 127, 128}, one and two batches, an inactive neighbour, the domain guard,
and four rejection cases.

Two axes are deliberately **not** covered by the shipped gates, and both are
exercised in section 4 instead: cache lengths beyond 128, and head counts other
than 96. The ratio matrix qualifies Tk256 and Tk512 at 2.1e-04 against the FP32
reference, and H32/H64 likewise, so the evidence exists — but no shipped case
pins it, because each additional distinct shape costs a JIT compilation and
these shapes cannot reach the pipelined entry anyway (section 6).

## 6. Known limits

- **The pipelined entry is compiled only when `iterations == 1`,
  `heads % block_m == 0` and `batch == 1`.** The first two are properties of the
  guard; the third is a correctness boundary (section 3). Shapes outside it take
  the original entry and measure zero gain.
- **`block_n` >= 256 is refused with `unsupported attention tile`**, so for
  Tk > 128 the pipelined entry cannot be reached at any `block_n`. The low ratios
  on the Tk256 and Tk512 rows are a ceiling set by the available tile, not a
  tuning gap.
- **`public_prefill_block_n` is deliberately narrower than what qualifies.** At
  B1/H96/Tq64/Tk128 the pipelined `block_n = 128` qualifies at 2.1e-04 and runs
  2.61x faster than the `block_n = 64` the selector returns. The selector's
  contract is "measured profile only", and the newly reachable path has no test
  coverage, so widening it is left as the first performance item rather than done
  here.
- **BF16 is timed but not ratioed.** Section 4 reports BF16 on both backends; no
  BF16 ratio is given because no external reference passes the same accuracy
  qualification in that dtype.
- **Ordinary non-Graph calls show no significant speedup**; that path is
  dominated by host overhead.
- **The ratio against the official FIA covers eight shapes in FP16 / PTO, core
  scope, Graph timing**, with both arms qualified against an FP32 reference
  before timing. AscendC and BF16 are timed but not ratioed, and the before/after
  table earlier in section 4 reports **speedups**, not ratios against a qualified
  reference — the two tables answer different questions.
- **The FIA comparison requires right-down aligned causal masking** on both
  sides, so the ratio cells all set `query_start = kv_len - valid_q` and no cell
  with Tq > Tk can be compared at all.
- Decode, cache append, the packed layout adapter and full model integration are
  not included.
