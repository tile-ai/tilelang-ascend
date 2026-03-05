# NPUIR Pytest Infra Guide

This directory uses a unified pytest infra for NPUIR tests, including:

- shared tensor/assert helpers in `testcommon.py`
- unified CLI filters in `conftest.py`
- marker registration in `pytest.ini`

## Infra Components

- `testcommon.py`
  - dtype mapping (`resolve_dtype`, `dtype_name`)
  - tensor generation (`gen_tensor`)
  - numeric check (`assert_close`)
  - mode context (`ascend_mode`)
  - environment setup helpers (`set_seed`, `set_npu_device`, `clear_tilelang_cache`)
- `conftest.py`
  - custom CLI options: `--op`, `--dtype`, `--mode`, `--npu-device`, `--seed`
  - collection filtering by marker values
  - fallback op inference from test name pattern `test_*_...`
  - session-level setup fixture (seed/device/cache)
- `pytest.ini`
  - `python_files = test_*.py`
  - default options (`-ra`)
  - all marker registrations used by this suite

## Directory and Naming Conventions

- Test file name: `test_*.py`
- Typical op files:
  - `testing/npuir/copy/test_copy_*.py`
  - `testing/npuir/math/test_*.py`
  - other category folders follow same style
- Recommended test function naming:
  - `test_<op>_<case>()`
  - example: `test_copy_shape_2d_3d_dynamic()`

This naming helps `--op` fallback matching when explicit `@pytest.mark.op(...)` is absent.

## How to Run

From repo root:

```bash
pytest testing/npuir
```

Run one folder:

```bash
pytest testing/npuir/copy
```

Run one file:

```bash
pytest testing/npuir/copy/test_copy_simple_dev.py
```

Run one test:

```bash
pytest testing/npuir/copy/test_copy_simple_dev.py::test_copy_simple_2d_dev
```

## CLI Filters

### `--op`

Filter by `@pytest.mark.op("...")` (comma-separated):

```bash
pytest testing/npuir --op=copy
pytest testing/npuir --op=copy,sigmoid
```

If `mark.op` is missing, infra infers op from `test_*_...`.

### `--dtype`

Filter by `@pytest.mark.dtype("...")`:

```bash
pytest testing/npuir --dtype=float16
pytest testing/npuir --dtype=float16,float32
```

Matching rule:

- `--dtype=float16,float32` means "keep tests whose dtype marker contains at least one of them".
- If `dtype` marker is absent on a test, that test is not selected by `--dtype` unless its `nodeid` text happens to match.

### `--mode`

Filter by `@pytest.mark.mode("...")`:

```bash
pytest testing/npuir --mode=Developer
pytest testing/npuir --mode=Expert
```

### `--npu-device`

Choose NPU device id (default `0`):

```bash
pytest testing/npuir --npu-device=0
```

### `--seed`

Set random seed for full session (default `42`):

```bash
pytest testing/npuir --seed=123
```

### Combined Filters

All filters are AND-combined:

```bash
pytest testing/npuir --op=copy --dtype=float16 --mode=Developer --npu-device=0
```

## Marker Writing Styles

The following styles are all valid and recommended depending on scope.

### 1) Function-level markers (most common)

```python
import pytest

@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_simple_2d_dev():
    ...
```

### 2) Module-level markers via `pytestmark` (apply to all tests in file)

```python
import pytest

pytestmark = [
    pytest.mark.copy,
    pytest.mark.op("copy"),
    pytest.mark.dtype("float16"),
]
```

You can still add extra markers on specific test functions.

### 3) Class-level markers

```python
import pytest

@pytest.mark.copy
@pytest.mark.op("copy")
class TestCopySuite:
    def test_case_a(self):
        ...
```

### 4) Param-level markers (`pytest.param`)

```python
import pytest

@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float16", marks=pytest.mark.dtype("float16")),
        pytest.param("float32", marks=pytest.mark.dtype("float32")),
    ],
)
def test_copy_dtype(dtype):
    ...
```

This is useful when one test function covers multiple dtypes/modes and you still want infra filtering.

## DType Marking Cookbook

### Case A: test only one dtype

```python
@pytest.mark.dtype("float16")
def test_copy_f16():
    ...
```

Run:

```bash
pytest testing/npuir --dtype=float16
```

### Case B: same test body supports multiple dtypes (recommended)

```python
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float16", marks=pytest.mark.dtype("float16")),
        pytest.param("float32", marks=pytest.mark.dtype("float32")),
        pytest.param("bfloat16", marks=pytest.mark.dtype("bfloat16")),
    ],
)
def test_copy_multi_dtype(dtype):
    ...
```

Run only some dtypes:

```bash
pytest testing/npuir --dtype=float16,bfloat16
```

### Case B2: input/output dtype Cartesian product (no manual enumeration)

```python
import pytest
from testcommon import build_dtype_param_combos

IN_DTYPES = ["float16", "float32"]
OUT_DTYPES = ["float16", "float32"]

DTYPE_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES)

@pytest.mark.parametrize("in_dtype, out_dtype", DTYPE_COMBOS)
def test_copy_io_dtype_combo(in_dtype, out_dtype):
    ...
```

This pattern scales directly to more dtypes without manually listing each pair.

### Case B3: add accum_dtype with automatic combination

```python
import pytest
from testcommon import build_dtype_param_combos

IN_DTYPES = ["float16", "float32"]
OUT_DTYPES = ["float16", "float32"]
ACCUM_DTYPES = ["float32"]

TRIPLE_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES, ACCUM_DTYPES)

@pytest.mark.parametrize("in_dtype, out_dtype, accum_dtype", TRIPLE_COMBOS)
def test_copy_io_accum_combo(in_dtype, out_dtype, accum_dtype):
    ...
```

Optional: provide custom id labels when needed:

```python
TRIPLE_COMBOS = build_dtype_param_combos(
    IN_DTYPES, OUT_DTYPES, ACCUM_DTYPES, names=["input", "output", "accum"]
)
```

### Case C: one test function should match multiple dtypes as a group

Use repeated `dtype` markers on the same test:

```python
@pytest.mark.dtype("float16")
@pytest.mark.dtype("float32")
def test_copy_shared_logic():
    ...
```

Now both commands select this test:

```bash
pytest testing/npuir --dtype=float16
pytest testing/npuir --dtype=float32
```

### Case D: module-level default dtype + per-test override

```python
pytestmark = [pytest.mark.dtype("float16")]

@pytest.mark.dtype("float32")
def test_special_case_f32():
    ...
```

Both markers exist for the second test, so `--dtype=float16` and `--dtype=float32` can both select it.
If you need strict separation, prefer Case B (`pytest.param` per dtype) instead of stacking defaults + overrides.

## Recommended Marker Policy

For each test case, prefer at least:

- one category marker (`copy`, `math`, `reduce`, ...)
- one op marker (`@pytest.mark.op("<op>")`)
- dtype marker when dtype is explicit (`@pytest.mark.dtype("<dtype>")`)
- mode marker only when mode-specific behavior is tested

For `copy` tests specifically:

- use `@pytest.mark.op("copy")` consistently
- if missing, ensure function name follows `test_copy_*`

## Practical Examples

Run all copy tests:

```bash
pytest testing/npuir --op=copy
```

Run copy + Developer mode:

```bash
pytest testing/npuir --op=copy --mode=Developer --npu-device=0
```

Run a single dynamic copy file:

```bash
pytest testing/npuir/copy/test_copy_shape_dynamic.py
```

Generate JUnit report for CI:

```bash
pytest testing/npuir --op=copy --junitxml=report-copy.xml
```

## Notes

- `conftest.py` performs session-level setup once per pytest session.
- Keep device setup centralized in infra (`--npu-device`) instead of hardcoding device id in test modules.
- `__pycache__` folders are Python bytecode cache and do not affect test discovery.
