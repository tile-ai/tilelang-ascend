# NPUIR Pytest Conventions

This directory uses a narrow pytest contract on purpose. The goal is to keep test
selection predictable for contributors and keep execution behavior derived from a
single source of truth.

## Design

The test model is split into three layers:

- Marker metadata: `op` and `mode`
- Runtime controls: `--op`, `--mode`, `--npu-device`
- Test matrix: `dtype`, shapes, and other case dimensions via `@pytest.mark.parametrize(...)`

That split is intentional:

- `op` and `mode` are the only custom markers. They define what a test is.
- CLI options only choose which tests to run and which NPU device to use.
- Data-oriented coverage stays inside the test matrix instead of growing more CLI flags.

## Marker Rules

Only two custom markers are valid:

- `@pytest.mark.op("<real-op>")`
- `@pytest.mark.mode("<mode>")`

Use file-level `pytestmark` when a whole file shares the same metadata:

```python
import pytest

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]
```

Use test-level markers only when a file intentionally mixes different ops or modes:

```python
import pytest

@pytest.mark.op("copy")
@pytest.mark.mode("Developer")
def test_copy_dev():
    ...

@pytest.mark.op("copy")
@pytest.mark.mode("Expert")
def test_copy_release():
    ...
```

The closest marker wins. In practice that means a test-level marker overrides a
file-level marker of the same kind.

## Runtime Rules

`--npu-device` is the only supported device selector in this pytest layer. The
session hook sets the current device once before tests run.

`mode` is marker-driven. Contributors should not manually wrap tests with
`with ascend_mode(...)`. The pytest runtime reads the closest `mode` marker and
applies `ascend_mode(mode)` automatically around the test body.

## Test Matrix Rules

Use `@pytest.mark.parametrize(...)` for case dimensions such as:

- input/output dtype
- shapes and block sizes
- index selections
- other data-dependent coverage dimensions

Example:

```python
import pytest

pytestmark = [pytest.mark.op("copy")]

DTYPE_CASES = [
    ("float16", "float16"),
    ("float16", "float32"),
    ("float32", "float32"),
]

SHAPE_CASES = [
    (256, 1024, 32, 32),
    (512, 512, 64, 64),
]

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_CASES)
@pytest.mark.parametrize("M, N, block_M, block_N", SHAPE_CASES)
def test_copy_shape(M, N, block_M, block_N, in_dtype, out_dtype):
    ...
```

## Contributor Rules

- Do not add custom markers beyond `op` and `mode`.
- Do not add `@pytest.mark.dtype(...)`.
- Do not add folder-category markers such as `@pytest.mark.memory`.
- Do not call `torch.npu.set_device(...)` inside tests.
- Prefer file-level `pytestmark` for shared `op` / `mode`.
- Use test-level markers only when a file intentionally needs overrides.
- Keep compile and execution work inside test functions, not at module import time.

The directory name remains the category signal for humans. For example,
`memory_ops/` tells readers the family of tests, while `@pytest.mark.op("copy")`
identifies the real operation used by CLI filtering.

## CLI

```bash
# all NPUIR tests
pytest testing/npuir

# one folder
pytest testing/npuir/memory_ops

# one file
pytest testing/npuir/memory_ops/test_copy_shape_dev.py

# filtered by op
pytest testing/npuir --op=copy

# filtered by mode
pytest testing/npuir --mode=Developer

# combined selection
pytest testing/npuir --op=copy --mode=Developer --npu-device=0
```

`--op` and `--mode` accept comma-separated values.

## How Filtering Works

- `--op` matches the closest `@pytest.mark.op(...)`
- `--mode` matches the closest `@pytest.mark.mode(...)`
- `--npu-device` sets the session device before tests execute

Tests without a matching marker are excluded when that selector is provided.

## Minimal Template

```python
import pytest

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]
CASES = [
    (256, 1024, 32, 32),
]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", CASES)
def test_copy_shape_dev(M, N, block_M, block_N, dtype):
    ...
```
