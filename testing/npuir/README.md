# NPUIR Pytest Manual

This directory uses shared pytest infra from:

- `conftest.py`: CLI filters and session setup
- `testcommon.py`: tensor/assert/mode helpers
- `pytest.ini`: marker registration and default options

## Quick Start

```bash
# all npuir tests
pytest testing/npuir

# one op folder
pytest testing/npuir/copy

# one file
pytest testing/npuir/copy/test_copy_simple_dev.py

# one test
pytest testing/npuir/copy/test_copy_simple_dev.py::test_copy_simple_2d_dev
```

## CLI Filters

```bash
# by op
pytest testing/npuir --op=copy
pytest testing/npuir --op=copy,sigmoid

# by dtype
pytest testing/npuir --dtype=float16
pytest testing/npuir --dtype=float16,float32

# by mode
pytest testing/npuir --mode=Developer

# device/seed
pytest testing/npuir --npu-device=0 --seed=42

# combined (AND)
pytest testing/npuir --op=copy --dtype=float16 --mode=Developer --npu-device=0
```

Filter behavior:

- `--op`: first matches `@pytest.mark.op("...")`; if missing, infers from test name `test_*_...`.
- `--dtype`: keeps tests with at least one matching `@pytest.mark.dtype(...)`.
- `--mode`: matches `@pytest.mark.mode("...")`.

## Marker Patterns

Function-level:

```python
import pytest

@pytest.mark.copy
@pytest.mark.op("copy")
@pytest.mark.dtype("float16")
@pytest.mark.mode("Developer")
def test_copy_simple_2d_dev():
    ...
```

Module-level (`pytestmark`):

```python
import pytest

pytestmark = [
    pytest.mark.copy,
    pytest.mark.op("copy"),
]
```

Param-level dtype:

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

## DType Combo Helper (Recommended)

Use `build_dtype_param_combos` from `testcommon.py` to avoid manual enumeration.

```python
import pytest
from testcommon import build_dtype_param_combos

IN_DTYPES = ["float16", "float32"]
OUT_DTYPES = ["float16", "float32"]
ACCUM_DTYPES = ["float32"]

# 2 lists
IO_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES)

# 3 lists (or more)
IOA_COMBOS = build_dtype_param_combos(IN_DTYPES, OUT_DTYPES, ACCUM_DTYPES)

# optional custom id prefixes
IOA_COMBOS_NAMED = build_dtype_param_combos(
    IN_DTYPES, OUT_DTYPES, ACCUM_DTYPES, names=["input", "output", "accum"]
)

@pytest.mark.parametrize("in_dtype, out_dtype", IO_COMBOS)
def test_io(in_dtype, out_dtype):
    ...
```

## Minimal Conventions

- File name: `test_*.py`
- Prefer test name pattern: `test_<op>_<case>`
- Add `@pytest.mark.op("<op>")` for reliable `--op` filtering
- Keep device selection centralized via `--npu-device` (avoid hardcoded device ids in tests)
