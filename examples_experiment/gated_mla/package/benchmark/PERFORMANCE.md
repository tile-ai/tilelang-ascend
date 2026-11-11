# Gated MLA performance evidence

The measured candidate is the packaged seven-input forward, before W_O. Input
projections/norm, gate-logit generation, cache append and communication are
outside this scope. Measurements are not whole-model serving results.

## Environment and comparison

Ascend910B3 (20 Cube/40 Vector cores, 192MiB L2), CANN9.2.0, Python3.12.9,
Torch2.7.1+cpu, TileLang `cc9e2681c17c7a8479e2ec9b3b91124fca819c69`.
The source-derived comparisons use isolated TorchNPU2.7.1.post11 and
Triton-Ascend3.2.2.post2. Each table row compares the same inputs in one process.

The independent FP64 budget is unchanged: rtol0.01, atol0.002 and maximum absolute
error divided by reference maximum magnitude <=0.005. Seeds0/1/17 are qualified
before timing; inputs/constants and post-timing outputs are checked again.
The compact machine-readable results, qualification records and measured source
inventory are in [measurements.json](measurements.json).

The vLLM-Ascend source path is pinned in [benchmark documentation](README.md).
Its original output-gate kernel runs through a direct Triton launcher, rather
than the complete vLLM warmup/host dispatcher. The historical comparison uses
FP32 Torch-NPU projections/gate plus FIA. The two references intentionally retain
different intermediate rounding and are reported separately.

## Complete-chain graph replay

Median microseconds; reused inputs/weights, no L2 flush. Decode uses50 calls per
graph and prefill5; both use5 replay warmups and20 replay samples. The ratio is
reference time / candidate time, not latency reduction or hardware utilization.

| Shape / visibility | Dtype | Candidate | Qualified reference | Reference | Ratio |
|---|---|---:|---|---:|---:|
| B1/H96/Tq1/Tk4096, full | FP16 | 157.738 | Source-derived decode + original gate | 110.253 | 69.90% |
| Same physical shape, visible17 | FP16 | 91.986 | Source-derived decode + original gate | 73.941 | 80.38% |
| B1/H96/Tq1/Tk4096, full | BF16 | 188.956 | Historical FP32 FIA composition | 128.881 | 68.21% |
| Same physical shape, visible17 | BF16 | 97.796 | Neither reference qualified | — | — |
| B1/H96/Tq128/Tk128, causal | FP16 | 451.086 | Historical FP32 FIA MLA composition | 313.042 | 69.40% |
| Same prefill shape | BF16 | 540.136 | Neither reference qualified | — | — |

The BF16 source-derived decode also fails the frozen budget for full visibility.
Both dtype variants of the source-derived prefill fail this budget; their
diagnostic times, approximately162/134us, are not qualification ratios. These
results describe this input/precision contract, not a claim that the upstream
operators are generally incorrect. No GPU speed comparison was performed.

### Ordinary calls

Event and synchronized host wall samples are retained separately. In the same
source-derived FP16 decode runs, candidate/reference Event medians were
2129/941us (full) and2134/980us (visible17): approximately44%/46%. Graph results
therefore do not establish the same ordinary-call performance. Host checking,
allocation and dispatch overhead remain a limitation.

## Adopted optimization

Only the public seven-input B1/H96/physical Tq=Tk128/default-scale profile selects
BN128. Device metadata values never enter dispatch/JIT keys. Decode, other
shapes, explicit/base bundles and packed/model dispatch retain their previous
configuration. Query/Value numerical kernels are unchanged by this selection.

Controlled configuration comparison used the original TorchNPU post4 environment
and one unchanged no-grad harness; it is not mixed with the post11 reference rows:

| Dtype | Complete graph BN64→128 | Latency reduction | Attention msprof Task BN64→128 | Task reduction |
|---|---:|---:|---:|---:|
| FP16 | 486.896→455.778us | 6.39% | 248.580→212.400us | 14.55% |
| BF16 | 578.674→540.018us | 6.68% | 332.260→291.020us | 12.41% |

The Task values are individual hardware samples, not graph medians. Do not sum
stage measurements and call the result an independently measured full chain.
Both backends/dtypes and full, prefix17 and inactive metadata passed12 follow-up
cases with three repeats and readonly checks. The installed public route's
generated kernel hashes match the successful configuration experiment.

The unadopted identity-view host experiment has identical numerical kernels;
its FP16 host timings varied with the references, so no stable host speedup is
claimed or included in the selected package.

Reproduction commands are in [README.md](README.md). The submitted-tree test,
simulator, source-identity and CI results are documented separately from these
measurements; older component results are not counted as hosted CI acceptance.
