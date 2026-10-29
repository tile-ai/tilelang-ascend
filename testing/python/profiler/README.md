# Profiler consistency regressions

Run from the repository root with Python, PyTorch, and pytest installed:

```bash
python -m pytest testing/python/profiler/test_assert_consistent.py -q
```

The tests load the actual profiler module from source and use real PyTorch tensors.
Only unused TVM/TileLang imports are stubbed, within a scoped `sys.modules` patch
that is restored after loading. The 17 CPU cases require no TVM build, CANN
installation, NPU, or kernel compilation. The tests are discovered by the existing
`testing/python/` pytest invocation; no operator manifest entry is needed.

The regressions cover reused tensor storage and mutable output containers,
single tensor/list/tuple returns, scalar tensors, changing output counts and
shapes, and unchanged results. `repeat=N` still performs one initial call and
N comparisons. The first result is copied once; this costs one extra set of
output tensors for the duration of the check, not in the benchmarking path.

## Optional NPU coverage

With a working TorchNPU/CANN environment, the same command also runs 20 device
cases. Only these NPU cases are skipped when TorchNPU or an NPU is unavailable;
the original CPU regressions remain in normal discovery.

The NPU matrix covers stable and changing reused tensors, bare/list/tuple returns,
scalar tensors, and output count/shape changes. Each case runs once on the current
stream and once with an explicitly ordered producer stream. Before each write,
the producer waits for an event on the consumer stream, protecting the profiler's
previous snapshot. Before returning, the consumer waits for a producer event,
protecting its subsequent reads. Output storage, streams and events stay alive
until final synchronization, including assertion paths. No unsynchronized
producer/consumer race is introduced as a test oracle.

## Recorded device evidence

On 2026-10-25, a separate frozen harness checked the complete profiler modules at
PR head `27c1fa257c987801a48f795ad082b68f7c18b3ee` and base
`43b0b69720f479894a13c02746599eccee5c5769` on an Ascend 910B3, CANN 9.0.0,
PyTorch 2.7.1 and TorchNPU 2.7.1.post4. All 24 expected outcomes passed: the 20
patched-source cases above and four base-source stable/changing-buffer controls.
The original source accepted changing aliased buffers on both stream paths;
the patched source rejected them. The harness used the same native-import
isolation as these tests.

That recorded harness run is distinct from executing this newly adapted pytest
file on NPU. This file still needs its own device/CI run. The checks validate the
Python comparison contract on real device tensors; they do not compile a native
TileLang kernel, validate either generated-kernel backend, or prove that kernels
are race-free.
