# Gated MLA comparison tools

These tools compare the normally installed seven-input package with two explicit
NPU compositions. They do not time an entire model or replace a missing candidate
with a Torch/FIA fallback.

| Method | Computation |
|---|---|
| `candidate` | Installed `gmla.build_runtime().forward` |
| `historical_fp32` | FP32 Torch-NPU projections/gate plus official FIA |
| `source_decode` / `source_prefill` | Adapted unquantized vLLM-Ascend path, preserving its layouts and low-precision rounding |

The source path is pinned to vLLM-Ascend
[`e4d533d`](https://github.com/vllm-project/vllm-ascend/tree/e4d533d33211f0696bcdf65eda47245d52e428c5).
Decode includes the original head padding, FIA v2 and transpose-BMM. Prefill uses
one original `F.linear` function to expand K/V, the old FIA TND path, its FP16 to
BF16 conversions and int8 causal mask. The first prefill adapter admits only
B1/H96/Tq=Tk128 with all queries active and no historical prefix.

The seven supplied inputs already contain the external projections, normalized
latent cache and gate logits. Those operations, cache append, W_O and distributed
communication are outside this comparison. Static weight packing, block tables
and mask creation are recorded as preparation; per-call conversions remain timed.

## Run

Activate a built TileLang/CANN environment first. The tested hardware is 910B3.
The source-gate measurements used CANN9.2.0, Torch2.7.1, TorchNPU2.7.1.post11 and
Triton-Ascend3.2.2.post2. Configure compatible dependencies separately; these
tools never install or upgrade the environment.

From the repository root:

```bash
PKG="$PWD/examples_experiment/gated_mla/package"
BENCH="$PKG/benchmark"
python -m pip install --no-deps --no-build-isolation "$PKG"
SITE=$(python -c 'from pathlib import Path; import gmla; print(Path(gmla.__file__).resolve().parent.parent)')
python "$BENCH/verify_installed.py" --source-root "$PKG" --output /tmp/gmla-installed.json

python "$BENCH/source_decode_benchmark.py" \
  --installed-root "$SITE" --inventory /tmp/gmla-installed.json \
  --protocol-root "$BENCH/protocol" --visible 4096 --dtype float16 \
  --output /tmp/gmla-decode-fp16

python "$BENCH/source_prefill_benchmark.py" \
  --installed-root "$SITE" --inventory /tmp/gmla-installed.json \
  --protocol-root "$BENCH/protocol" --dtype float16 --legacy-route mla_paged \
  --output /tmp/gmla-prefill-fp16
```

Use a fresh output directory for each run. Repeat decode with `--visible 17` and
both commands with `--dtype bfloat16`. PTO is the default; use `--target ascendc`
for the other candidate backend.

The default source adapter names its unfused gate `torch_rounding`. To execute
the isolated original Triton gate, add both arguments:

```bash
--gate-module source_gate.isolated_gate \
--gate-label vllm_e4d533d_original_gate_direct_launcher
```

The kernel and selected helper functions are preserved by AST. Only the vLLM
warmup/host dispatcher is replaced by a direct Triton launcher; this is not full
vLLM host-performance evidence. Source attribution and hashes are in
`source_gate/source_manifest.json`; Apache-2.0 code retains its notices and
`LICENSE.apache-2.0`.

## Measurement and qualification

Three seeds are checked against independent expanded and absorbed FP64 references
before timing. The unchanged budget is rtol0.01, atol0.002 and maximum error divided
by reference maximum magnitude <=0.005. Inputs and prepared constants are checked
for mutation. Outputs are rechecked after Event and graph measurements.

Event, synchronized host wall and graph-replay distributions are separate. Graph
uses 50 calls for decode and5 for prefill, with5 replay warmups and20 samples;
inputs/weights are reused without an L2 flush. A failed reference has no qualified
performance ratio. `--time-unqualified` records diagnostic timing only.

For hardware counters, first save a successful benchmark report, then use the
same arguments with `--mode profile --profile-method candidate` (or the named
reference) and `--validated-report <report.json>` under `msprof op`. Keep the
application output in a separate `entry/` subdirectory of the profiler output.
Inspect the entry status, kernel records and generated sources: a profiler exit
code of zero alone does not prove the application ran successfully.

## Balanced order and host diagnostics

[Paired testing](PAIRED_TESTING.md) adds serial independent-process trials, all six
method orders, separate Event/wall/graph statistics and per-boundary qualification.
[The 2026-11-12 report](PERFORMANCE_BALANCED_20261112.md) records the twelve
case/dtype/backend combinations; historical results above retain their original
protocol. The production numerical implementation is unchanged.

The independent `host_decode_diagnostic.py` keeps uninstrumented timing separate
from cProfile attribution; see [its boundary and commands](HOST_DIAGNOSTIC.md).
