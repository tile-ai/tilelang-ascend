# Experimental Gated MLA forward package

## Current A9/A10 evidence, 2026-11-13

The current prefill source is a9a96f9e, reviewed from parent a1723d4/a0e1c997.
Only this production file changes; the other24 production Python files remain.
The batch-one, head-aligned, single-KV-block pipeline retains the original precision policy and fallback.

- [Current source-pinned report](benchmark/PREFILL_A9_CURRENT_20261113.md):24 fresh processes,10800 verified samples.
  FP16 Graph core reaches75.07%/82.86% of qualified FIA(PTO/AscendC);
  full-chain reaches87.92%/100.54% of the historical projection/FIA/gate composition.
  These are one-shape Graph ratios, not all-shape or model-throughput acceptance.
- Directory-parallel tests include original L0:243 core and44 extensions pass, with artifact checks.
  Core248.367s exceeds the extra local120s target(wrapper1/pytest0/artifacts pass);
  extensions201.839s. Hosted CI/review remain separate.
- [BF16 model diagnostic](benchmark/MODEL_BF16_DIAGNOSTIC_20261113.md):
  FP64 gates pass in the tested B2 fallback fixture, but original cross-official gates still fail identically in parent/candidate.
  This is not a new B1-pipeline model acceptance pass.
- Current normal-install8 passed with production sources/CPP matching all formal trials; current hardware6 profiles passed source/CPP/precision checks.
  Final documentation was added after the tested frozen wheel; A8 simulator records remain historical.
  [Current validation](benchmark/PREFILL_A9_VALIDATION_20261113.json) separates these states.

Older environment/count/performance paragraphs below retain their original historical scope.
This current run uses Torch-NPU2.7.1.post4; do not substitute earlier post11/prototype/7c-control measurements.

Dense causal inference with shared compressed C/E cache. This export starts from
the selected r1.7 implementation and includes the reviewed 2026-11-12 prefill-only
revision. Of the 25 production Python files, only `gmla/kernels/prefill.py` changes
from parent `de16598`; the separate identity-view experiment remains unselected.

The selected implementation is Q2 r2, CubeValue R3, S6 split-KV decode and C4
native vector-mask prefill. Default target is PTO; `target="ascendc"` is explicit.
The selected Cube grid is 20, with M32 attention. The seven-input public runtime
uses N128 only for B1/H96/physical Tq=Tk128 at the default scale; all other public
shapes, decode and custom scales retain N64. Metadata values never select the
profile or enter a JIT key. The base `runtime.kernel_bundle` remains explicit N64
for controls; the profile uses a separate immutable cached bundle. Packed/model
adapters retain N64. Query/Value specialize
B1/2/4 and H96; other applicable shapes retain TileLang Vector implementations.
Value uses Cube for Tq>=16. No Torch/FIA attention fallback is present.

## Environment and use

Component evidence uses CANN9.2.0, TileLang-Ascend
`cc9e2681c17c7a8479e2ec9b3b91124fca819c69`, Python3.12.9,
Torch2.7.1/Torch-NPU2.7.1.post4 and Ascend910B3. Activate the matching source-runtime
environment first. This package does not install or replace those dependencies.
Installation: `python -m pip install --no-deps --no-build-isolation .`.
The declared Python>=3.10 is a language/package constraint; the verified native
stack above uses3.12.9. Installation neither supplies nor validates other native
versions or hardware. CPU metadata/reference tests can run without an NPU; actual
operator execution requires Torch-NPU, a built TileLang Ascend runtime and NPU.

Preload TileLang/TVM before positive NPU work:

```python
import torch
from gmla import build_runtime, prepare_metadata

runtime = build_runtime()  # default PTO
_ = runtime.kernel_bundle  # imports only; no factory or NPU launch
torch.npu.set_device(0)
# Create/transfer the seven input tensors after preload.
cpu_meta = prepare_metadata([[17, 81, 64]], batch=1, query_capacity=17, cache_capacity=81)
device_meta = cpu_meta.to(q_content.device)  # explicit32 bytes/batch control H2D
with torch.no_grad():
    output, stages, audit = runtime.forward(
        q_content,
        q_aux,
        latent_kv,
        key_aux,
        w_k,
        w_v,
        gate_logits,
        device_meta,
        return_intermediates=True,
    )
```

This native-library order does not establish compatibility with every legacy
TBE operation or arbitrary valid late-import sequence. NZ-negative tests use the
public `empty_with_format` allocator; their uninitialized NZ payload is never
read, filled, copied or computed.

## Contract

|Input|Shape|
|---|---|
|Q content / auxiliary|`[B,Tq,H,128]` / `[B,Tq,H,64]`|
|Shared C / E cache|`[B,Tk,512]` / `[B,Tk,64]`|
|WK / WV|each `[H,128,512]`|
|Gate logits|`[B,Tq,H,128]`|

All seven inputs share FP16/BF16 dtype and one indexed NPU, are logically dense
and physically ND/NCHW. That NPU must be the current device. Forward never changes
devices or copies inputs to repair a mismatch; actual multi-card execution is
not validated. Output is `[B,Tq,H,128]`, in input dtype, after sigmoid gate and
before W_O. All physical operands including padding must be finite; no payload
scan or clipping is inserted. Autograd-enabled inputs requiring gradients are
rejected; no training/backward path is supplied.

One complete host preflight precedes lazy kernel loading: shape/dtype/device,
grad intent, physical format, metadata layout and actual int32 element spans.
Public and direct-bundle paths share it without repeating full validation.
No public trusted/bypass option exists.

Metadata is contiguous device int32 `[B,8]`: q_len, kv_len, query_start, then
five zero columns. CPU validates `0<=q_len<=Tq`, `0<=kv_len<=Tk`, `0<=query_start`
and `query_start+q_len<=kv_len`, then explicitly uploads it. No device values are
read back for routing or JIT keys. Length updates reuse capacity-specialized
kernels; caller owns stream ordering/lifetime.

Scale defaults to1/sqrt(192). Explicit host values must convert to finite nonzero
float32; the original host value/default constants and JIT behavior are retained.
Tq1/default scale uses decode; other nonempty calls use prefill. Tq0 returns an
empty tensor. Tk0 with CPU-validated inactive metadata invokes the zero kernel.
There is no arbitrary128/16k cap: actual dense input/latent/weight, split-partial
and tail-buffer element spans must fit int32. Static acceptance establishes
neither hardware coverage nor available memory.

## Packed cache adapter

```python
from gmla import create_cache, build_packed_adapter

with torch.no_grad():
    owner = create_cache(B, capacity, dtype=torch.bfloat16, device="npu:0")
    core = build_packed_adapter(owner, heads=H)
    y = core.append_and_forward(raw_query, raw_kv_weight, normalized_c, raw_kv, gate_logits, q_lens)
```

Q may be `[B,Tq,H*192]` or `[B,Tq,H,192]`; KV weight may be `[H*256,512]` or
`[H,256,512]`. Views preserve storage. Query emits QA and Value reads the second128
weight rows directly. Cache append consumes normalized C and rawKV's last64
channels. Initialization and append use NPU kernels, without host payload copies.

The owner is synchronous, fixed-capacity and bound to one thread/device/stream.
Direct append/read require no_grad/inference_mode and validate physical formats
before mutation. Reset/grow/reorder/eviction and captured/asynchronous ownership
are unsupported. Packed forward requires positive physical Tq. After committed
append followed by failure, `CacheCommittedForwardError` exposes lengths/generation;
do not blindly append the same tokens again.

External projections, norms, gate-logit generation, append/control transfers and
W_O costs remain explicit. The optional frozen Kimi bridge requires separately
supplied pinned source/layer; it is not a general Transformers Cache replacement.

## Numerics and evidence

Value R3 uses scale4 for FP16 and2 for BF16. Hi is cast from Z/scale; Lo retains
the residual in Z's original scale. Two separately initialized FP32 GEMMs produce
A/B; Vector combines A+B/scale, gates, then rescales. This intentional numerical
change from R2 is recorded in source_manifest.

FP16 high representation covers `abs(Z)<=262016`; scaled BF16 high can represent
every finite FP32 Z. These are representation bounds, not universal error
guarantees for arbitrary finite values. Extreme weights can overflow FP32 partial
products; the existing FP32 sigmoid may underflow for extreme negative logits.
No thresholds are widened to hide these limitations.

Independent expanded/absorbed FP64 references consume the original seven inputs.
The frozen budget is elementwise rtol0.01/atol0.002 and normalized max error<=0.005,
with finite/dtype/shape checks. Official FIA/HF comparators are qualified
independently; failing comparators receive no speedup ratio.

R3 range/normal cases and the N128 profile have separate device evidence. The PR's
performance report must identify each tested shape, source hash, timing boundary
and qualified baseline. Results from different runs or boundaries must not be
combined into a new ratio. An approximate70% reference rate is the current project
target, not a claimed property of every shape; unqualified comparators are not scored.
See [validation results](VALIDATION.md) and [remaining acceptance](REGRESSION_GAPS.md)
for implemented regressions, recorded device evidence and the additional local cold-time target.

## Reproduction

Run the commands below from this source checkout, with the required runtime
already active. `source_manifest.json` verifies packaged file/function AST hashes;
historical origin IDs are identifiers, not filesystem paths or runtime dependencies.
`FILELIST.sha256` covers the exported source files and is also checked by the audit
tool. No D-drive paths or development prototypes are needed for these commands.
After reviewed test/tool edits, follow [source-record maintenance](MAINTENANCE.md)
before rerunning the audit.

```bash
# From repository root; use the owned NPU and compatible built runtime:
PKG="$PWD/examples_experiment/gated_mla/package"
python "$PKG/tools/audit_package.py"
python "$PKG/validation/run_core_ci.py" --repo-root "$PWD" \
  --runtime-root /path/to/built/tilelang-ascend --output /new/cold-evidence --workers 8
```

The wrapper resolves the active manifest's ten files and uses `--forked -n8`.
Its full boundary includes collection, compilation, loading, tests and audit.
The wrapper additionally enforces a strict local 120s cold-time target; it is not
a hosted pytest timeout. The repository template permits a core/daily split when
full tests exceed two minutes. This package retains 145 core and 22 low_priority
cases, measured at 187.6994s and 172.8109s respectively, without changing tests or
markers. See [VALIDATION](VALIDATION.md) for correctness and timing boundaries.

The wheel installs `gmla` and its kernel design note; source tests, audit tools
and root manifests remain source-checkout artifacts. Source pytest adds this
checkout to Python's search path and therefore does not prove that a separately
installed wheel was imported. Verify wheel imports from outside the checkout and
record `gmla.__file__` plus `gmla.package_sources()` for installed-artifact checks.

The optional `load_frozen_layer(source_path)` fixture requires an explicitly
supplied official source file whose SHA256 is checked; it is not needed by the
core API or the bundled tests and is not downloaded automatically.

NPU failures are not silently skipped. Active repository CI registration covers
nine package-suite anchors plus the preserved L0 suite. Kernel Task, graph, eager
entry and whole-layer costs remain separate. The recorded finite single-layer
smoke and the outstanding cold/hosted CI boundary are detailed in [VALIDATION](VALIDATION.md).

## Why this is under examples_experiment

The files in `gmla/kernels` are real executable TileLang kernels. The experimental
location describes the integration/acceptance status, not a Torch simulation.
Only the documented forward layouts, shapes and environments have evidence;
suite registrations and range regressions are present; the additional local
cold-time target, hosted CI and broader deployment acceptance remain separate.
Moving this package to `examples/` should follow
the repository's performance, test and review gates, not merely the existence of
working kernels. The retained single-domain L0 example outside this package is a
separate regression entry.

[Kernel design and data flow](gmla/kernels/design.md) summarizes the packaged
implementation without referring to private prototype folders.

## Review evidence

- [Complete-chain performance and reference qualification](benchmark/PERFORMANCE.md).
- [Benchmark commands and source attribution](benchmark/README.md).
- [Validation summary, extended cases and model boundaries](VALIDATION.md).
- [Historical eight-case direct/simulator evidence and reproduction](validation/README.md).
- [Frozen one-layer model reproduction](validation/MODEL_SMOKE.md).
- [Source integrity and maintenance](MAINTENANCE.md).

Repository CI selects the preserved L0 suite and nine package suites. The local test conftest prepares a shared CPU compile pool only for selected Gated MLA device tests, then scopes cache/compiler checks to those tests. Kernel-cache preparation is included in the cold-test wall time. Ordinary collection and CPU-only selection do not compile kernels; unrelated operator execution restores its original context.

## Reviewed prefill revision

The prefill revision preserves public interfaces, original FP64 budgets and BF16 query/probability residuals. It adds head-aligned whole-tile masks, invocation-local KV residency and query prefetch, exact one-KV-block state identities and backend-safe L0 grains. Generic heads and multi-KV cases retain the guarded recurrence. See [performance/qualification](benchmark/PREFILL_OPTIMIZATION_20261112.md) and [reproduction](benchmark/PREFILL_OPTIMIZATION_TESTING.md); scope-specific speedups are not whole-model throughput claims. The following revision adds a cross-wave Cube/Vector pipeline for head-aligned single-KV-block shapes at batch one, leaving arithmetic and ordering unchanged and every other shape on the original entry; see [pipeline performance](benchmark/PREFILL_CV_PIPELINE_20261112.md).


Current closeout: final core243 after3 additional CPU symbol regressions passed separately from the earlier44 extensions. Core248.367132s exceeds the unchanged local120s target. Normal-install8 and current hardware6 passed; see current report and validation JSON. New PTO/FP16 simulator timed out at900s after prepare/direct; no new-source simulator pass is claimed. BF16 original model cross-official gates remain failed; hosted CI/review require the new published SHA.
