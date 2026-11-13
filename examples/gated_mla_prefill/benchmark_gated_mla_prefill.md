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

Common low-precision core scope, against an absorbed
`npu_fused_infer_attention_score_v2`, Graph amortised timing, B1/H96/Tq=Tk128,
FP16, 24 independent trials:

| Backend | This PR | vs official FIA v2 |
|---|---:|---:|
| PTO | 97.23 us | **75.07%** |
| AscendC | 88.67 us | **82.86%** |

**These ratios come from measuring the same kernel under a separate comparison
harness that this example does not carry.** The protocol, shape and environment
are given here so the measurement can be reproduced; the example does not claim
to ship that harness.

**The Graph ratio must not be read as every calling domain meeting a bar.**
Ordinary non-Graph calls in the same scope are dominated by host overhead and sit
near 66% of FIA; this optimization does not improve them by design.

### What is left

Of the 89.28 us the Cube side is busy, `cube_time` accounts for only 14.66 us and
`mte1_time` for 29.07 us, so **about 45 us is neither arithmetic nor transfer** —
instruction issue and cross-core sync dispatch are the candidates. The ladder
recorded in `examples/flash_attention/fa_opt` attributes 60% -> 72% to reducing
the number of cross-core synchronisations and 72% -> 80% to cutting instruction
count with axpy. Neither is implemented here. Since Cube is the critical path
after step 7 (Cube busy 89.28 us against Vector 68.53 us), the former ranks first.

## 5. Accuracy

`test_example_gated_mla_prefill.py`, 22 cases, with three kinds of gate held to
deliberately different standards.

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

## 6. Known limits

- **The pipelined entry is compiled only when `iterations == 1`,
  `heads % block_m == 0` and `batch == 1`.** The first two are properties of the
  guard; the third is a correctness boundary (section 3). Shapes outside it take
  the original entry and measure zero gain.
- **BF16 has no external performance reference that passes the same accuracy
  qualification**, so no BF16 ratio is reported.
- **Ordinary non-Graph calls show no significant speedup**; that path is
  dominated by host overhead.
- The ratio against the official FIA is obtained at one working point only
  (B1/H96/Tq=Tk128, FP16, core scope, Graph timing). The multi-shape table in
  section 4 reports **before/after speedups**, not ratios against a qualified
  reference.
- Decode, cache append, the packed layout adapter and full model integration are
  not included.
