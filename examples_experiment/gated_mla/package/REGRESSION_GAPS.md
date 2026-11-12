# Coverage and remaining acceptance

The current source includes a reviewed prefill-only revision from parent
`de16598`; the other 24 production Python files remain unchanged. Numerical
correctness, performance, simulator execution and the local cold-time target
remain separate acceptance results. The [current report](benchmark/PREFILL_OPTIMIZATION_20261112.md)
and [validation summary](benchmark/PREFILL_VALIDATION_20261112.json) record source
and test fingerprints. [Historical validation](VALIDATION.md) retains earlier
r1.7 direct/simulator and frozen-model evidence under its original boundaries.

## Current regression evidence

- 104 independent old/new numerical cases passed: 13 shape/tile templates, two
  seeds, two dtypes and two backends. Four Graph binding groups also passed
  same-storage C/E and E-only input updates. Production BF16 query/P residuals
  and the original FP64 budgets remain intact.
- Eight normal-install combinations passed in core/full-chain scopes, both
  dtypes and both backends. Generated CPP matched the selected independent
  variant and the formal performance trials.
- The full registered directory collected 255 cases across the original16 test
  files: 233 core and22 `low_priority` extended cases. Both real-NPU selections
  passed without skips. This includes CPU benchmark tests and preserved L0;
  the 233 count is not233 independent NPU numerical cases.
- The change strengthens six existing representative instances and adds three
  core NPU nodes plus one CPU capacity test. It preserves generic-head,
  physical-tail, multi-KV, continuation, inactive and public-profile coverage,
  plus metadata/Q/KV updates and repeat/Graph checks. No manifest change or new
  `ci_skip` hides a required regression.
- Core and extended runs used fresh CPU compile/audit directories, eight
  workers,99/73 unique factories and141/90 requests. Worker/parent ABI and
  loaded sources matched, published caches stayed unchanged, and no unexpected
  native compilation occurred. CPU-only package selection passed178 tests;
  46 device tests were explicitly deselected there.

## Remaining boundaries

The core full cold process took217.205s (pytest209.88s), exceeding the unchanged
extra local120s target: wrapper exit1, pytest exit0, artifact checks passed.
The extended process took158.517s. Hosted CI, other operators and reviewer
approval remain separate checks. The representative updated-source simulator
matrix passed four prefill combinations with three physical launches each; see
the [matrix and actual log render](benchmark/PREFILL_SIMULATOR_20261112.md). Native
source-line mapping remains unavailable; this is not hardware timing or a claim
that all shapes were simulated.

FP16 shared-LP prefill core Graph performance remains below70% of qualified FIA
(49.64% PTO and54.87% AscendC). Full-chain FP16 Graph ratios to the historical
FP32-projection/FIA/gate composition are79.12%/92.44%, while ordinary Event/wall
full-chain calls show no meaningful improvement. BF16 unqualified references
receive no ratio. Scope-specific Graph gains do not establish all performance
requirements, whole-model throughput, general deployment or all legal shapes.
