# MLA benchmark boundaries

The original public-forward reports remain historical evidence. The alignment
tools add separate experiments; they do not modify production kernels, remove
public validation, or reinterpret a boundary change as an optimization.

| Experiment | Candidate | Reference | Preparation outside timing |
|---|---|---|---|
| Existing public forward | Unchanged public API | Existing source-derived or historical full-chain compositions | As recorded in the original report |
| `prepared-full-chain` | Original public preflight and profile selection once, then the unchanged private execution path on each call | Unchanged bound source-derived decode/prefill composition | Metadata validation/H2D, static weight layout, import/compile; candidate public preflight and route selection |
| `shared-lp-core` | Existing attention entry on common low-precision QL/QA/C/E | Absorbed official FIA v2, with the same logical inputs | Query projection, query/layout packing, decode zero-head padding, page table/mask preparation and H2D |

The prevalidated candidate still performs dynamic query projection, attention,
decode split merge, value recovery, gate, internal guards, and native per-call
output/workspace allocation. It is a benchmark binding, not a new public API or
a pure-kernel timer. Both public and prevalidated callables bind fixed tensors;
the source composition also owns prepared copies of static weights. Rebuild
bindings when shapes, dtype/device/storage, weights or metadata contracts change.

Each full-chain seed checks public/prevalidated output equality with zero
tolerance. A separate untimed probe changes live Q/cache/gate payloads in place,
checks that the result changes and still equals public forward, restores the
inputs, and checks their bytes. The original independent expanded/absorbed FP64
references and precision budget remain unchanged.

The shared-core QL is independently computed from original CPU inputs in FP64,
rounded once to the input dtype, and copied to the device. Its independent FP64
golden uses those actual rounded operands. Candidate output is FP32; FIA output
retains its original FP16/BF16 dtype. Both are checked against that same golden.
The output boundary is `[B,Tq,H,512]`, before value recovery and gate.

**BF16 shared-LP is a component experiment, not production BF16 performance.**
The production full chain retains its HiLo query and probability compensation.
The shared-LP experiment uses the existing `ql_residual=None` specialization;
it does not cast a reconstructed HiLo query down and claim identical inputs.
Its results cannot replace production BF16 full-chain acceptance.

Source-prefill attention operates on expanded K/V and returns V128. It is not
the latent512 component used here. The prefill core comparator is explicitly
the separately extracted, historical absorbed FIA-v2 route. Decode preserves
the source adapter's logical H96 / physical H128 padding, prepared outside the
core timer. Candidate heads remain H96. This physical-layout difference is
reported rather than hidden.

## Timing and qualification

- Three cases: physical Tk4096/full decode; the same capacity with prefix17;
  and causal Tq=Tk128 prefill. B1/H96; FP16/BF16; PTO/AscendC.
- Three independent processes per case/dtype/backend, with seeds0/1/17 for
  qualification in every process. Only seed0 is timed.
- Full chain measures candidate prevalidated / public control / source in all
  six permutations. Core measures two methods in balanced two-order cycles.
  The existing three-method order algorithm and protocol identifier are retained.
- Eager Event, synchronized wall, and amortized graph samples are separate.
  Event includes dispatch gaps; none is hardware Task time. Graph P95 is a tail
  of amortized samples, not individual request P95. Compile, qualification,
  preparation, and read-only checks occur outside each timer.
- Qualification before and after each boundary uses rtol0.01, atol0.002,
  max-amplitude-normalized error0.005. No threshold is relaxed for a comparator.
  Failed references can be timed only as explicitly requested diagnostics and
  receive no ratio. Candidate numerical failure aborts the trial.
- `public_forward / candidate` is an interface-overhead control. Only the
  separately named source/FIA rows compare with an external implementation.
  The ratio is reference time / candidate time, not model throughput.
- Summaries preserve process-level medians and P95s. Scope, prepared operand
  hashes, source/helper hashes and method definitions must match before trial
  aggregation. Raw samples from different scopes are never pooled.
- Timing and execution helpers load from exact sibling file paths, independently
  of the source adapter directory and preexisting module names. The actual
  timing module hash is asserted against its recorded identity before sampling.

Run each scope into a new output directory, using the same normally installed
candidate and verified source adapter as the existing report:

```bash
python run_aligned_suite.py \
  --scope prepared-full-chain \
  --installed-root <site> --inventory <pinned_inventory.json> \
  --benchmark-root <package/benchmark> \
  --output <new-full-chain-output> --trials 3 --time-unqualified

python run_aligned_suite.py \
  --scope shared-lp-core \
  --installed-root <site> --inventory <pinned_inventory.json> \
  --benchmark-root <package/benchmark> \
  --output <new-core-output> --trials 3 --time-unqualified
```

The default source gate is the isolated original Triton launcher. The existing
`--torch-rounding-gate` flag explicitly selects a different diagnostic gate.
For CPU-only fixture/golden checks use `aligned_benchmark.py --mode cpu-reference`
with one scope/case/dtype. This mode asserts that neither compiler nor device was
initialized; it produces no hardware performance result.

## Precedents and limits

- [FlashMLA #35](https://github.com/deepseek-ai/FlashMLA/pull/35) prepares scheduler
  metadata and FlashInfer `plan()` before execution timing. Its GPU performance
  is not an Ascend baseline.
- [TileLang-Ascend #1644](https://github.com/tile-ai/tilelang-ascend/pull/1644)
  and [#1645](https://github.com/tile-ai/tilelang-ascend/pull/1645) demonstrate
  decode-core and paged-core scope; neither is a full seven-input gated chain.
- [vLLM-Ascend #17298](https://github.com/vllm-project/vllm-ascend/pull/17298)
  measures gate-only eager/graph paths and reports the observed regression.
  Fusion alone is not evidence of speedup.

These precedents inform experiment design; they do not waive current repository
requirements or establish performance for this candidate.
