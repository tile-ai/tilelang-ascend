# FP16 visible17 decode host diagnostic

This independent tool uses the normally installed public
`gmla.build_runtime(target=...).forward`. It does not modify the PR, inject a
provider, bypass guards, or alter global NPU format settings. The case is fixed:
`B=1, H=96, Tq=1, Tk=4096`, metadata `[1,4096,16,0,0,0,0,0]`, FP16,
head-specific gate, default scale `1/sqrt(192)`.

The shared `benchmark/protocol/decode_fixture.py` constructs the seven inputs.
Its expanded and absorbed FP64 references must agree. The original output
budgets remain `rtol=0.01`, `atol=0.002`, global normalized error `<=0.005`.
This tool is a one-seed host diagnostic, not a replacement for the multi-seed
source-comparison benchmark or hardware acceptance matrix.

## Measurement boundaries

1. Verify installed source inventory and frozen protocol digests; build CPU
   inputs/references outside timing. Preload the public kernel bundle before
   initializing the NPU, matching `source_decode_benchmark.py` ordering.
2. Qualify the actual output plus query/latent stages. Check all seven inputs
   and metadata are unchanged.
3. Reuse the frozen `eager_timing` helper for **uninstrumented** Event interval
   and wall-with-end-event-sync measurements. Retain the production public
   validation path. Recheck output and read-only inputs after timing.
4. Warm again, then use cProfile on the same public callable. Pre/post-call NPU
   synchronization is outside cProfile, and every profiled call starts after
   the previous call has drained. Validate the output/read-only inputs again.

The cProfile report is elapsed host attribution, not CPU hardware cycles. It
includes Python, driver/launch waits and instrumentation disturbance. Inclusive
rows overlap; do not sum them or interpret their total as removable validator
cost. Cython internals may be opaque. The tool does **not** derive validator
cost from Event-minus-graph, and does not capture a graph. `post_call_drain`
is recorded separately and is not claimed to be device kernel duration.

Any exception stops further device work. Formal measurements saved before a
later diagnostic failure remain in the JSON with overall `status=failed`.
Partial cProfile statistics are retained; their denominator/status must be
read before interpreting them.

## CLI

Use a clean process with the owner-provided built TileLang/CANN environment.
Normal import must resolve to a non-editable installed package; the tool never
adds a package/prototype directory to `sys.path`. `--inventory` has the existing
`{"sources": gmla.package_sources()}` format. Keep the inventory pinned to the
actual selected candidate rather than silently regenerating it after changes.

```bash
export TL_PLATFORM=A2 TL_RUN_MODE=npu
python host_decode_diagnostic.py \
  --installed-root /path/to/installed/site \
  --inventory /path/to/pinned_installed_inventory.json \
  --protocol-root /path/to/package/benchmark/protocol \
  --output /path/to/new/host_visible17 \
  --mode diagnose --target pto --seed 0 \
  --warmup 10 --repeats 20 --profile-calls 20
```

The script uses the current NPU device; it does not switch devices. The NPU
owner must run it serially with other benchmarks. CPU validation, which never
loads TileLang/torch_npu or initializes an NPU:

```bash
python -m pytest -q test_host_decode_diagnostic.py
python host_decode_diagnostic.py \
  --installed-root /path/to/installed/site \
  --inventory /path/to/pinned_installed_inventory.json \
  --protocol-root /path/to/package/benchmark/protocol \
  --output /path/to/new/cpu_reference --mode cpu-reference
```

## Outputs and shared-driver hook

- `host_decode_diagnostic.json`: source hashes, exact argv, environment, CPU
  input/reference digests, metadata, stage/output accuracy and read-only checks,
  uninstrumented measurements and separate cProfile diagnostics.
- `SUMMARY.md`: small uninstrumented timing table and top 15 inclusive rows.
- `host_profile.pstats` and `host_profile_functions.json`: complete attribution;
  JSON also records completed/requested calls if profiling fails.

A unified driver can reuse
`run_host_profile(callable, synchronize, calls=20, output_dir=new_dir)`.
It must warm and qualify its callable before invoking the helper, then qualify
the returned output and check inputs afterward. The helper deliberately has
no fixture, device initialization or guard-bypass behavior.

The fixed FP16/PTO visible17 device diagnostic has passed numerical and readonly
checks before timing, after uninstrumented timing and after all 30 profiled calls.
See [the 2026-11-12 report](PERFORMANCE_BALANCED_20261112.md). CPU-reference
validation remains a separate check and does not establish hardware performance.
