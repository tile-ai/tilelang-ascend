# Gated MLA decode measurements

Measured on 2026-10-30 with Ascend 910B3 (20 Cube/40 Vector), CANN 9.2.0,
driver 25.5.0, Python 3.12.9, PyTorch 2.7.1 and torch_npu 2.7.1.post4.
The TileLang runtime was built from `463c234ce3087205c092e931207ca87b4cd5494c`.
The interface is the fixed FP16 domain documented in README.md; the default scale is
`1/sqrt(192)` and the output is after value restoration and gate, before output projection.

## Protocol

- Reuse inputs and weights; no L2 flush. All reported latency values are microseconds.
- Eager: 10 warmup calls followed by 20 samples. An NPU Event interval encloses the complete
  public call and can include device idle gaps caused by host dispatch. Synchronized wall
  time includes recording the end event and waiting for completion.
- Graph: capture 50 actual calls, warm up with 5 replays, then measure 20 replays and divide
  each sample by 50. Report the median. Python metadata validation and allocation dispatch
  occur during capture. Graph and eager describe different invocation scenarios.
- Independent stage graphs use CPU-reference FP32 intermediate inputs. The full chain uses
  intermediate outputs from the five real kernels; isolated stage times are not additive.
- JIT and correctness checks run before the timed samples. Before and after measurements
  use the same protocol but were separate runs, not randomized interleaved trials.

## Complete chain

| Visible keys | Implementation | Before graph | After graph | After eager Event | After synchronized wall |
|---:|---|---:|---:|---:|---:|
| 64 | TileLang five stages | 36.20 | 35.34 | 2564.72 | 2646.85 |
| 64 | PyTorch-NPU FP32 composition | 26.87 | 26.42 | 1006.65 | 1088.83 |
| 64 | CANN FIA v2 composition | 29.88 | 29.03 | 942.35 | 1021.17 |
| 17 | TileLang five stages | 77.50 | 35.84 | 2588.82 | 2668.66 |
| 17 | PyTorch-NPU FP32 composition | 25.25 | 24.49 | 1016.09 | 1091.63 |
| 17 | CANN FIA v2 composition | 28.05 | 27.10 | 909.02 | 986.28 |

The visible17 TileLang graph interval decreases by approximately 53.8% (2.16x faster).
The visible64 graph result is essentially unchanged. Host load also affects the reference
eager results between runs; do not attribute the eager differences solely to the mask fix.
The ordinary TileLang call still has substantial adapter/host dispatch overhead.

The PyTorch-NPU composition uses FP32 operands/intermediates, then converts the result to
FP16. The FIA composition performs FP32 query absorption, rounds that result to FP16,
calls the A2 MLA path in `npu_fused_infer_attention_score_v2`, then performs FP32 value
restoration and sigmoid gate before final FP16 conversion. FIA latent output is FP16.
The compositions include the per-call projections, casts, arithmetic and output work.

FIA uses an identity block table with block size16, prepared once outside timing. Cache
views are created inside each call without copying the contiguous cache. A valid query's
visible prefix is passed as the actual KV length. Both reference compositions passed
5 seeds x visible lengths1/17/64 against the expanded FP64 reference using the frozen
FP16 budget. These checks cover the documented synthetic inputs and default scale.

## Latent readout diagnosis

| Visible17 latent readout | Before | After |
|---|---:|---:|
| Independent stage graph median | 47.55 | 6.61 |
| msprof Task Duration | 47.30 | 6.04 |
| Slowest AIV scalar time | 42.511 | 1.624 |
| Slowest AIV vector time | 0.958 | 0.996 |

The original generated code used a 64x64 scalar mask loop. The new generated code uses
one `Duplicate<float>` over 3008 elements starting at FP32 element1088 for visible17.
The remaining row-scalar probability broadcast is unchanged. A fresh `msprof op` capture
uses one matching `main_kernel`, BlockDim4/MixBlockDim8, warmup10, and reports frequency1800.
Cube instructions/FLOPs remain0. The profiler values are a single diagnostic sample;
the graph results above are repeated measurements with a separate protocol.

## Reproduce

```bash
python examples_experiment/gated_mla/tools/benchmark/measure_decode.py \
  --warmup 10 --repeats 20 --graph-batch 50 --output /tmp/gated_mla_timing

env -u ASCEND_RT_VISIBLE_DEVICES msprof op \
  --warm-up=10 --launch-count=1 --kernel-name=main_kernel \
  --aic-metrics=PipeUtilization,Memory,ArithmeticUtilization \
  --output=/tmp/gated_mla_profile \
  python examples_experiment/gated_mla/tools/benchmark/measure_decode.py \
  --mode profile --stage latent_readout --visible 17 --output /tmp/gated_mla_profile_entry
```

Both output roots must be new for each experiment. The entry saves environment, source
hashes and generated code; timing mode also saves raw samples and reference errors.
The profile entry launches the selected stage once; `--launch-count` does not repeat it.

## Review status

This report does not claim that every case/invocation mode meets the upstream performance
threshold. Graph FIA/TileLang time ratios are approximately82.1% (visible64) and75.6%
(visible17); eager ratios remain below50%. Baseline precision differs as described above.
Representative production shapes, PTO validation, simulator materials and maintainer
acceptance of this limited experimental scope remain open. Real-device profiling is not
presented as simulator evidence. The PR should remain Draft until the applicable review
requirements are resolved or explicitly accepted by maintainers.


## Correctness and CI scope

The original development workspace passed all 100 cases (82 CPU and 18 NPU), including
five isolated prefix-mask regressions. The self-contained example passes 29 tests in
two lanes: 25 PR-core cases in 116.77s using a new empty TileLang JIT cache,
then 4 extended cases in 64.81s with that cache. Both lanes have zero
skipped tests. The core result has little margin below two minutes on this host; hosted
CI timing is not yet established.

Core coverage retains readout prefixes1/17/63/64 with nonzero tail weights, full-chain
causal/length checks, gate saturation, inactive queries, input immutability, and explicit
device-side negative-infinity checks. The four `low_priority` cases add default-scale
single-key and near-full complete chains, an extra seven-key readout, and the extreme-scale
complete-chain stress case. A finite-mask counterexample remains in the CPU core tests.

The experimental worktree uses the built runtime from the same base commit. Its local
pytest command selected that runtime with `-o pythonpath=<built-tilelang-checkout>`;
the worktree's own TileLang dependencies were not rebuilt. This affects runtime lookup,
not test selection or the operator imports. The candidate is imported from the new
example directory. The normal repository-root commands assume that checkout is built.

The final packaged benchmark was rerun after entry-point fixes. All ten generated kernel
sources (five stages x two prefixes) are byte-identical to the mask-optimized implementation
used for the hardware profile. The single-sample profile and repeated graph measurements
therefore describe the same device code, under their separately stated protocols.

The before prototype used the same five stages with this tail clearing before the row
probability multiplication; the current implementation replaces only this mask operation:

```python
for i, j in T.Parallel(64, 64):
    cf[i, j] = T.if_then_else(i < visible_keys, cf[i, j], 0.0)
```
