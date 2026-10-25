# Profiler consistency regressions

Run from the repository root with Python, PyTorch, and pytest installed:

```bash
python -m pytest testing/python/profiler/test_assert_consistent.py -q
```

The tests load the actual profiler module from source and use real CPU tensors.
Only unused TVM/TileLang imports are stubbed, within a scoped `sys.modules` patch
that is restored after loading. No TVM build, CANN installation, NPU, or kernel
compilation is required. The tests are discovered by the existing
`testing/python/` pytest invocation; no operator manifest entry is needed.

The regressions cover reused tensor storage and mutable output containers,
single tensor/list/tuple returns, scalar tensors, changing output counts and
shapes, and unchanged results. `repeat=N` still performs one initial call and
N comparisons. The first result is copied once; this costs one extra set of
output tensors for the duration of the check, not in the benchmarking path.

These tests validate the Python comparison logic. They do not validate NPU
stream ordering, either kernel backend, or prove the absence of device races.
