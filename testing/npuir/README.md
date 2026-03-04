# NPUIR Pytest Infra Usage

This directory uses a unified pytest infrastructure for NPU IR tests.

## Files

- `testcommon.py`: shared helpers
  - dtype resolution
  - tensor generation (`gen_tensor`)
  - numeric check (`assert_close`)
  - ascend mode context (`ascend_mode`)
  - seed/device/cache helpers
- `conftest.py`: pytest integration
  - custom CLI: `--op`, `--dtype`, `--mode`, `--npu-device`, `--seed`
  - collection-time filtering by markers and nodeid fallback
  - session fixture for NPU device, seed, and cache clear
- `pytest.ini`: marker registration and default pytest options

## Run Basics

From repo root:

```bash
pytest testing/npuir
```

From inside `testing/npuir`:

```bash
pytest .
```

Run one file:

```bash
pytest testing/npuir/test_copy_simple_dev.py
```

Run one test function:

```bash
pytest testing/npuir/test_copy_simple_dev.py::test_copy_simple_2d_dev
```

## Infra CLI Options

### `--npu-device`

Select NPU device id (default `0`):

```bash
pytest testing/npuir --npu-device=0
```

### `--seed`

Set random seed for the whole test session (default `42`):

```bash
pytest testing/npuir --seed=123
```

### `--op`

Filter by `@pytest.mark.op("...")` values (comma-separated):

```bash
pytest testing/npuir --op=copy_simple
pytest testing/npuir --op=copy_sliced,copy_strided
```

If a test has no `op` marker, fallback matching uses `nodeid` substring.

### `--dtype`

Filter by `@pytest.mark.dtype("...")` values:

```bash
pytest testing/npuir --dtype=float16
```

### `--mode`

Filter by `@pytest.mark.mode("...")` values:

```bash
pytest testing/npuir --mode=Developer
pytest testing/npuir --mode=Expert
```

### Combined filtering

All filters are combined with logical AND:

```bash
pytest testing/npuir --op=copy_sliced --dtype=float16 --mode=Developer --npu-device=0
```

## Marker Usage In Tests

Recommended markers per test:

- `@pytest.mark.copy` (category marker)
- `@pytest.mark.op("copy_xxx")` (op identity)
- `@pytest.mark.dtype("float16")` (dtype identity)
- `@pytest.mark.mode("Developer")` (when mode-specific)

## Copy Test Coverage

Current `test_copy_*` files already support this infra and can be filtered directly, e.g.:

```bash
pytest testing/npuir --op=copy_shape_dynamic
pytest testing/npuir --op=copy_shape_dynamic_cube
pytest testing/npuir --op=copy_simple
pytest testing/npuir --op=copy_sliced
pytest testing/npuir --op=copy_sliced_cube
pytest testing/npuir --op=copy_sliced_extended
pytest testing/npuir --op=copy_strided
```

## Practical CI Examples

Smoke (copy only):

```bash
pytest testing/npuir -k "copy_" --npu-device=0 --seed=42
```

Mode-specific smoke:

```bash
pytest testing/npuir --op=copy_simple,copy_sliced --mode=Developer --npu-device=0
```

Generate JUnit XML:

```bash
pytest testing/npuir --op=copy_sliced --junitxml=report-copy-sliced.xml
```
