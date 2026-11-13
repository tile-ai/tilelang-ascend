# Validation record

## Current A9/A10 acceptance, 2026-11-13

This section supersedes historical counts/statuses below for the current a9a96f9e prefill source.
The other24 production Python sources are unchanged from parent a1723d4.
Historical simulator/model/installation records are not promoted to the new source.

|Boundary|Actual result|
|---|---|
|Source-pinned timing|24 completed independent processes;10800 raw samples, source/CPP/qualification/round-trip checks pass|
|Directory core including original L0|243 passed;pytest0/artifact pass;248.367132s;wrapper1 because extra local120s target failed|
|Disjoint low_priority including L0|44 passed;pytest0/artifact pass;201.838996s|
|Package serial --forked|213 core/253 full passed;contains CPU tools, excludes original L0|
|CPU package only|178 passed,75 NPU cases deselected;9.29s, no NPU execution|
|Five changed Python files|Ruff0.17.4 lint/format passed;documentation decoder checked separately|
|Normal installed artifact8|all8 source/precision/CPP combinations passed;matches all3 formal trials per cell|
|Current hardware profile|6 source/CPP/precision-verified PTO/FP16 collections;Task140.64±0.90→98.98±0.47µs|
|New A9 simulator|PTO/FP16 prepare/direct passed;simulator900s timeout,audit not run,overall failed;other3 old-source records remain historical|
|Hosted CI/review/merge|not claimed for this unpublished revision|

The complete directory tests used forked/parallel execution and audited the CPU compile pool,
loaded factory inventories, unchanged published cache and absence of unplanned native compilation.
They are not284 independent NPU numerical examples; CPU/tool tests and original L0 are included.
The120s wrapper is not invoked by hosted CI and does not replace the repository's test-cost requirements.

The real H96/hidden7168/qrank1536 random-weight BF16 model fixture remains diagnostic_only_failed:
parent/candidate on both backends have the same saved operand/output bits and CPP in its B2 fallback domain,
and all three FP64 phases pass, while original cross-official gates remain failed.
There is no checkpoint, complete-model quality or new B1-pipeline model integration acceptance.

See [current timing](benchmark/PREFILL_A9_CURRENT_20261113.md),
[validation fingerprints/status](benchmark/PREFILL_A9_VALIDATION_20261113.json) and
[BF16 diagnostic](benchmark/MODEL_BF16_DIAGNOSTIC_20261113.md).

The current source includes the reviewed 2026-11-12 prefill-only revision from
parent `de16598`; 24 other production Python files remain unchanged. Current
normal-install and full-suite results are in the [prefill report](benchmark/PREFILL_OPTIMIZATION_20261112.md)
and [validation summary](benchmark/PREFILL_VALIDATION_20261112.json): eight installed
combinations, 233 core tests and 22 extended tests passed. The core cold process
took 217.21s and missed the unchanged extra local 120s target. Updated-source
representative simulator runs passed all four prefill dtype/backend combinations;
see the [current simulator matrix](benchmark/PREFILL_SIMULATOR_20261112.md). Native
source-line mapping remains unavailable. The older matrix below retains its
historical source/runtime boundary.

The following revision adds a cross-wave Cube/Vector pipeline to the same
single production file. It is compiled only for head-aligned shapes whose cache
fits one KV block at batch one; every other shape keeps the previous entry, whose
generated code is byte-identical. Arithmetic and ordering are unchanged and the
two entries agree bit-for-bit over 32 paired cases spanning both backends, both
dtypes, the pipelined and fallback domains and two seeds. Core-scope Graph
performance against the qualified official FIA is 77.58% (PTO) and 83.11%
(AscendC), medians of three independent trials; the same-scope eager_wall ratios
are 66.5% and 66.2% and did not move. See the
[pipeline report](benchmark/PREFILL_CV_PIPELINE_20261112.md).

## Historical r1.7 record

The remaining sections record the previously selected r1.7 implementation, before
the reviewed prefill revision. All 25 production Python files matched that frozen
candidate at the time. The separate later identity-view experiment was excluded.
These results preserve their original source/runtime boundaries.

## Recorded environment

Ascend910B3/A2, CANN9.2.0, Python3.12.9, Torch2.7.1+cpu and
Torch-NPU2.7.1.post4. TileLang source is
`cc9e2681c17c7a8479e2ec9b3b91124fca819c69`; TVM is
`c2921fdaf795b1103d21abc962e83a209c7258d7` with the repository's verified
backtrace-symbol and slice-step build patches. Loaded native libraries, source
and cache identity were audited. The separate post11/Triton source-benchmark
environment is not substituted for this functional/simulator/model record.

## Eight direct/simulator combinations

PTO/AscendC × FP16/BF16 × decode/prefill: **8/8 passed** direct and simulator
precision, readonly and source/kernel/configuration checks.

|Path|B/H/Tq/physical Tk|Actual public attention BN|Filtered launches|
|---|---|---:|---:|
|Decode|1/96/1/128|64|4|
|Prefill|1/96/128/128|128|3|

Prepare/direct/simulator kernel objects and generated hashes matched; exported
CPP hashes were independently checked. Prepare did not initialize an NPU.
The [matrix summary](validation/simulator_matrix.json) contains all eight records;
[reproduction instructions](validation/README.md) preserve the scope and timeout
units. The screenshot is a browser rendering of actual logs, not a native
MindStudio GUI. Native source-line mapping was unavailable. Simulator completion
is not a hardware-performance measurement.

## Original 18 extended cases

**18 passed,0 failures/errors/skips** in the retained extended JUnit report
(612.625s suite time; approximately 612.68s pytest wall). This includes:

- 12 prefill cases: Tq127/128/129, physical Tk129, both dtypes/backends.
- 4 H96 full-chain16k decode cases, both dtypes/backends.
- 2 BF16 custom-scale cases, both backends.

Original node IDs and the executed source snapshot are retained. Tq128/Tk129
remains an N64/KV-tail case and is not the new Tq=Tk128 core profile. Later test
organization does not retroactively change these18 recorded cases. Their duration
is not the cold PR CI budget or a kernel latency.

## Final registered extended suites

The final registered low_priority selection also passed on the real NPU:
**22 passed, zero failures/skips**, comprising the original four L0 cases and all
18 package extensions. Pytest reported160.57s; the complete process took172.8109s.
The automatic CI path compiled from an empty kernel cache, and all eight workers
passed source/ABI, forbidden-recompile and unchanged-cache audits. See
[extended_ci_result.json](validation/extended_ci_result.json).

Together, separate core and extended invocations executed all167 collected cases;
this is not a claim that a single combined invocation took either reported time.
The extended marker remains for daily/full runs. Test bodies and markers were
unchanged.

## Frozen one-layer model smoke: 12 candidate FP64 phases

Four jobs use B2/H96/hidden7168/qrank1536, capacity65, random weights, and phases
prefill17 → continuation3 → decode1. Cache lengths progress0→17→20→21. This is
one exact frozen official class, with no checkpoint or whole-model claim. It
normally imports the installed package; no prototype provider is injected.
Prefill confirms actual Cube Query and Value R3. Packed/model attention retains
its configured N64.

|Backend / dtype|Original job status / exit|Candidate before/after W_O vs FP64,3 phases|External inputs / readonly / cache|
|---|---|---|---|
|PTO FP16|passed /0|pass|pass|
|AscendC FP16|passed /0|pass|pass|
|PTO BF16|diagnostic_only_failed /1|pass|pass|
|AscendC BF16|diagnostic_only_failed /1|pass|pass|

Thus **12/12 candidate phases pass both before- and after-W_O FP64 checks**.
This does not mean12 full-model passes. Both BF16 jobs retain their official-HF
comparison failures: before-W_O pairwise comparison fails in prefill/continuation
and passes in decode; after-W_O pairwise comparison fails in all three phases.
The official BF16 before-W_O output also fails the same FP64 oracle in all three
phases; its after-W_O oracle checks pass. These original failed gates, exit1,
bad indices and diagnostics remain recorded. No threshold was relaxed and no
unqualified comparator receives a performance score.

The budget remains rtol0.01/atol0.002 and normalized maximum error<=0.005, with
finite/dtype/shape checks. Expanded/absorbed FP64 references agree. See the
[public one-layer reproduction](validation/MODEL_SMOKE.md) and
[compact model summary](validation/model_summary.json). Cost fields are
instrumented single-call, cold-inclusive wall/stream intervals covering external
projections/norm, append/control H2D and W_O; they are not steady-state gates.

## Final merged-tree cold core validation

All **145 core tests passed, zero failures/skips** with default PTO and explicit
AscendC coverage. The active manifest selected nine package suites and the
preserved L0 suite. All eight workers passed native/source/ABI and unchanged-cache
audits, with no unplanned native compile. The original test bodies, numerical
budgets and markers were retained.

The full pytest process took **187.6994 seconds**, including collection, empty
kernel-cache compilation, loading, device/CPU test bodies and final audits;
pytest itself reported 174.86s. **The extra strict local 120s target was not met.**
This is not a hosted per-operator timeout. Native/Cython
toolchain caches were retained. A faster CPU-only rehearsal is not substituted
for this NPU result. Details are in [core_ci_result.json](validation/core_ci_result.json).

The first attempt (143 passed/2 failed;178.28s) exposed a CI registry mismatch:
packed and separate-input defaults differ by one host floating-point ULP, which
enters the compilation key. The registry now uses the actual packed default.
Production code, references, thresholds and test selection were unchanged; the
corrected second cold attempt passed all 145 assertions and artifact audits.

~~~bash
# From repository root, with the compatible runtime and a real NPU:
PKG="$PWD/examples_experiment/gated_mla/package"
python "$PKG/validation/run_core_ci.py" --repo-root "$PWD" \
  --runtime-root /path/to/built/tilelang-ascend --output /new/cold-evidence --workers 8
~~~

The wrapper uses the ordinary hosted flags **--forked -n8**, but additionally
exits nonzero if its strict local 120s target is missed, even when correctness and
artifact auditing pass. The recorded exit and raw `120s_gate_passed=false` remain
unchanged. Hosted CI invokes pytest directly and does not call this wrapper.
The current PR template permits the implemented core/daily marker split when
full tests exceed two minutes; 145 core and 22 extended retain all tests, labels
and measured times above. In this split instance,
the existing full py1.11.0 dependency had to be made visible ahead of pytest's
py.py shim in an isolated directory; no global runtime or operator dependency was
changed. Hosted CI needs its normal pytest-forked dependencies, including py.process.

|Gate|Recorded result|
|---|---|
|Core assertions|145 passed,0 failed,0 skipped|
|Native/source/ABI and full-cache audit|8 workers passed;0 unplanned compiles|
|Full process wall|187.6994s|
|Extra strict local <=120s target|Not met; not a hosted hard gate|
|Hosted CI and reviewer approval|Separate checks; not established by local results|

## Evidence identifiers and payload boundary

- Matrix archive `matrix_evidence.tar.gz`:342865bytes,
  SHA256 `d11d0f4ab0e1b554f950d9787b6651c440def8f7984088080e67f754136190ea`.
- Extended/model archive `extended_model_evidence.tar.gz`:108437bytes,
  SHA256 `ee5243fc6a8aa5bd14998e5b94714cc62c0a0a8b0b4ca930bbf201d3d119211e`.
- Executed model runner SHA256:
  `1b649c1cdbf15dea0c7d22a5f5e3f0a3986298d21accea3034bf20c773b88ac9`.
- External official model source SHA256:
  `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`,
  revision `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`.

Compact archives contain reports/logs/source snapshots/generated CPP, not the
large simulator dumps, compiled caches or model failure tensors. Those retained
payloads have a separate inventory; they are not silently claimed as bundled.
No Moonshot model source, checkpoint or vendor runtime is distributed here.

## Performance-test follow-up, 2026-11-12

The balanced benchmark added62 CPU regression tests. The integrated source
collects229 unique cases:207 core and the original22 extended. Package CPU
selection passed155 tests with 43 device cases deselected; these deselections
are not new NPU executions. Numerical production sources, the historical
protocol and the original numerical tests are unchanged.

The additional NPU comparison executed36 fresh-process trials (12 combinations,
three repetitions), with108 candidate initial seed checks and432 post-round
checks passing. Host attribution and a four-Task hardware profile are separate
records, not added to Event/graph measurements. See the
[measured report](benchmark/PERFORMANCE_BALANCED_20261112.md).

## 2026-11-12 benchmark boundary alignment

The 25 production Python sources and retained numerical definitions are unchanged.
The new tools passed 84 CPU tests; the package CPU selection passed 177 tests,
with 43 NPU cases deselected on CPU (8.10s). The 16 registered test files collect
251 unique cases: 229 core and 22 unchanged low_priority extensions.

The aligned hardware experiment completed 72 fresh-process trials (36 per scope),
with seeds 0/1/17 per trial, unchanged FP64 budgets, post-boundary/read-only checks,
and 27,000 independently verified timing samples. Full-chain seeds also require
public/prevalidated byte equality and successful live-payload update probes.
Two msprof applications passed numerical validation and produced 6 checked Task
CSVs. Eighteen FP16 core trials use the same generated attention/merge code as
production full-chain trials. Source qualification and BF16 component boundaries
remain explicit; no failed reference gets a performance ratio.

This is representative benchmark validation, not a new full 229-core/22-extension
cold NPU run. The prior 145 numerical-core and 22-extension execution records remain
historical evidence for unchanged production code. Hosted CI and review status
must be checked separately. See [the aligned report](benchmark/PERFORMANCE_ALIGNED_20261112.md)
and [scope/reproduction](benchmark/ALIGNMENT.md).


Current closeout: final core243 after3 additional CPU symbol regressions passed separately from the earlier44 extensions. Core248.367132s exceeds the unchanged local120s target. Normal-install8 and current hardware6 passed; see current report and validation JSON. New PTO/FP16 simulator timed out at900s after prepare/direct; no new-source simulator pass is claimed. BF16 original model cross-official gates remain failed; hosted CI/review require the new published SHA.
