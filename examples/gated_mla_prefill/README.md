# Kimi K3 Gated MLA prefill

The prefill forward of the Gated MLA attention block in Kimi K3, for Ascend
910B. It does absorbed multi-head latent attention over a latent KV cache, then
value recovery and a per-channel sigmoid output gate, and returns the result
before `W_O`.

"Absorbed" is the part worth naming. The ordinary way to compute MLA expands the
512-wide latent cache into a per-head key and value of width 128 and then runs
normal attention. This operator folds `W_K` into the query instead and
contracts against the latent cache directly, using it as both key and value, so
no `[B, Tk, H, 128]` intermediate is ever materialised. The two are
algebraically the same because the sum over the latent rank can be taken in
either order; the module ships both, and their agreement is what pins the
decomposition (see **Accuracy**).

## Interface

```python
from example_gated_mla_prefill import gated_mla_prefill

# target is "pto" or "ascendc"
y = gated_mla_prefill(q_content, q_aux, latent_kv, key_aux, w_k, w_v, gate_logits, metadata, target="pto")
```

| Tensor | Shape | dtype |
|---|---|---|
| `q_content` | `[B, Tq, H, 128]` | fp16 / bf16 |
| `q_aux` | `[B, Tq, H, 64]` | fp16 / bf16 |
| `latent_kv` | `[B, Tk, 512]` | fp16 / bf16 |
| `key_aux` | `[B, Tk, 64]` | fp16 / bf16 |
| `w_k` | `[H, 128, 512]` | fp16 / bf16 |
| `w_v` | `[H, 128, 512]` | fp16 / bf16 |
| `gate_logits` | `[B, Tq, H, 128]` | fp16 / bf16 |
| `metadata` | `[B, 8]` | int32 |
| **returns** | `[B, Tq, H, 128]` | same as `q_content` |

**`metadata` is a runtime input, not a compile-time parameter.** Its first three
columns are the valid query length, the valid cache length and `query_start`, so
one compiled kernel serves different valid lengths. A key `k` is visible to
query row `t` when `k < min(query_start + t + 1, kv_len)` — causal, with a
`kv_len` clamp, and `query_start` placing a query chunk at an offset into a
longer cache.

Rows past the valid query length are written as exact zeros, which the gates
check before applying any tolerance.

## Directory contents

| File | Lines | What it is |
|---|---:|---|
| `example_gated_mla_prefill.py` | 1598 | The operator, two independent references, and a `__main__` that runs five configurations against both backends |
| `test_example_gated_mla_prefill.py` | 337 | 28 cases in four kinds — budgeted, analytic, bit-exact and negative — plus six CPU-only reference-agreement cases |
| `design.md` | 354 | How it works: programming mode, the cross-wave pipeline, the on-chip budget, `pass_configs`, and the mapping onto the K3 released config |
| `benchmark_gated_mla_prefill.md` | 422 | Measurements and their protocol |
| `benchmark_gated_mla_prefill_zh.md` | 345 | The same in Chinese |

## Usage

Run the example directly to check the chain against the PyTorch reference. Five
configurations, each exercising a different property rather than a larger
version of the same one, on both backends:

```bash
python example_gated_mla_prefill.py
```

Run the gates:

```bash
# everything (needs an NPU; 22 of the 28 cases launch kernels)
pytest test_example_gated_mla_prefill.py

# the six reference-agreement cases only -- pure CPU, no device needed
pytest test_example_gated_mla_prefill.py -k the_two_references_agree
```

The test file is registered in `ci/operator_test_manifest.yaml`, so CI runs it
under Pytest rather than as a script.

## Supported configurations

**dtype** — fp16 or bf16 throughout; bf16 takes a probability-residual path, so
it costs more than fp16 where the pipeline is active.

**backend** — `target="pto"` or `target="ascendc"`. Both are exercised by the
gates and timed in the benchmark.

**The pipelined entry is compiled only when the cache is covered in one
`block_n` iteration, `heads % block_m == 0`, and `batch == 1`.** The first two
are structural; the third is a correctness boundary, because `c_l1`/`e_l1` are
L1-resident under a batch tag and the skewed loop would let one wave's QK
overwrite what the previous wave's PV is still reading. Shapes outside the
domain take the scalar entry and measure no gain. A negative case pins this: if
the pipelined entry ever starts compiling at `batch >= 2`, the documented domain
has changed.

`block_n` comes from `public_prefill_block_n`, a host selector that returns 128
only for the one measured profile `(1, 128, 96, 128)` and 64 otherwise. It is
not a caller-facing knob. `block_n >= 256` is refused by the compiler with
`unsupported attention tile`, which is what caps `Tk > 128`.

## Accuracy

The module carries **two references that share no contraction order** — the
absorbed form the kernel implements, and an expanded form that materialises
per-head keys and values and then runs ordinary attention. Over the six cases in
the test file they differ by at most **0.37 ULP** of the storage format. That is
the floor: a kernel tolerance tighter than two correct references disagree by
would be measuring the references, not the kernel. The gates use `rtol = 2⁻⁹`
and `atol = 2⁻¹⁰`, chosen above that floor rather than fitted to the kernel's
output.

Details, including the analytic gate whose expected values are all multiples of
0.5 so a mask off by one key moves the answer by exactly 0.5, are in
`benchmark_gated_mla_prefill.md` section 5.

## Performance

At the measured profile (`B1/H96/Tq128/Tk128`, the K3 working point) the
attention core reaches **74.9% of the official absorbed
`npu_fused_infer_attention_score_v2`** in fp16 on PTO, with both arms qualified
against an FP32 reference before either is timed.

Holding the shape still and moving only `block_n` separates the pipeline from
the shape: the pipelined entry is worth **2.20x at Tq128 and 2.61x at Tq64**, at
identical normalised error.

The full eight-shape ratio matrix, the `block_n` ablation, both dtypes on both
backends, and the step-by-step optimisation ladder are in
`benchmark_gated_mla_prefill.md` section 4. Read the `entry` column before
comparing rows: shape and entry move together down that table.

## Not included

Decode, cache append, the packed layout adapter and full model integration.
Ordinary non-Graph calls show no significant speedup — that path is dominated by
host overhead and this optimisation does not improve it by design.
