# TileLang-Ascend AI Core Exception Dump Technical White Paper

## 1. Overview

When executing custom operators on Ascend NPUs, AI Core hardware exceptions (such as illegal memory access, MTE errors, Cube/Vector compute overflow, etc.) cause kernel execution to abort. Traditional debugging methods struggle to capture the kernel's input tensor state at the moment of exception — developers are often left with only a hardware error code, unable to determine the actual data of each input tensor when the exception occurred.

The TileLang-Ascend AI Core Exception Dump feature leverages CANN's exception callback mechanism to automatically capture and save the kernel's input tensor data when an AI Core hardware exception is triggered, enabling developers to recover and analyze the exception scene post-mortem.

### Core Capabilities

- **Automatic capture**: When an AI Core exception occurs, CANN invokes the registered callback asynchronously — no manual intervention required
- **Tensor-level precision**: Saves complete input tensor data (not truncated register snapshots)
- **Kernel-level isolation**: Each kernel's dump file is prefixed with the kernel name, avoiding interference between kernels
- **Zero-intrusion switch**: Controlled by the `TL_ASCEND_EXCEPTION_DUMP` compile-time config, disabled by default, with zero impact on compiled artifacts when not enabled
- **Dual-backend support**: Covers both the AscendC (`ascendc`) and PTO (`pto`) codegen backends
- **End-to-end toolchain**: Provides Python functions that wrap dump file discovery, CANN msaicerr tool invocation, and tensor data reading

### Use Cases

| Scenario | Description |
|----------|-------------|
| Kernel debugging | Recover input tensors after a hardware exception for local issue reproduction |
| CI automated testing | Trigger exceptions in test cases and verify dump data matches expected inputs |
| Production troubleshooting | Deploy with the feature enabled; dumps are auto-saved on exception without reproduction |

---

## 2. Technical Principles

### 2.1 Workflow

```
 kernel execution ──→ AI Core hardware exception
                          │
                          ▼
           CANN invokes registered exception callback
                          │
          ┌───────────────┴───────────────┐
          │                               │
   Extract kernel args buffer      Callback searches for
   from exception info             MAGIC marker on Host
   (device memory)                 to locate ParamSizeInfo
          │                               │
          └───────────┬───────────────────┘
                      ▼
     Build acldumpTensorInfo array
     (tensor addresses, sizes, data types)
                      │
                      ▼
     acldumpSaveExceptionInfo(kernel_name, ...)
                      │
                      ▼
     <ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/
       <kernel_name>.custom.<timestamp>
                      │
                      ▼ Post-mortem parsing
     msaicerr.py parses → per-tensor .bin files
                      │
                      ▼
     read_msaicerr_bin() → np.ndarray
```

### 2.2 Exception Callback Mechanism

TileLang registers an exception callback function with the CANN runtime at compile time. When an AI Core hardware exception occurs, CANN automatically invokes this callback, passing in exception information. The callback uses CANN APIs to extract the kernel's argument buffer (in device memory) from the exception info, copies it to the host, and then parses it.

### 2.3 ParamSizeInfo and MAGIC Search

The argument buffer received by the callback contains all of the kernel's raw parameters (tensor pointers, tiling parameters, etc.), but the callback itself does not know which positions correspond to tensors, or what the size and data type of each tensor are.

TileLang's solution is to append a `ParamSizeInfo` struct to the end of the kernel parameters at compile time, containing:

- **MAGIC value**: A unique 8-byte marker (`0x474e414c454c4954`, i.e., ASCII "TILELANG")
- **kernel name**: The operator name, used as the dump file name prefix
- **Tensor metadata**: Byte size, device address, and ACL data type code for each tensor

The callback searches the argument buffer in 8-byte strides for the MAGIC value. Once located, it can read all tensor descriptor information from the `ParamSizeInfo` struct. This approach fully decouples the callback code from the kernel's specific parameter layout.

### 2.4 Dump File Generation

The callback builds a CANN `acldumpTensorInfo` array from the `ParamSizeInfo` and calls `acldumpSaveExceptionInfo` to save the tensor data to a file. The dump file path is controlled by CANN environment variables:

```
<ASCEND_DUMP_PATH>/extra-info/data-dump/<devId>/<kernel_name>.custom.<timestamp>
```

Each kernel's dump file is prefixed with the kernel name, supporting file isolation in multi-kernel scenarios.

### 2.5 Configuration Control

The feature is controlled by the `TL_ASCEND_EXCEPTION_DUMP` compile-time config, **disabled by default**. Reasons for default-off:

1. The feature depends on CANN APIs such as `aclrtSetExceptionInfoCallback` and `acldumpSaveExceptionInfo`, which may not be available in older CANN versions
2. Enabling it appends an extra parameter to the kernel signature, changing the compiled artifact's ABI
3. Exception callback registration alters CANN runtime behavior

The config is propagated through TVM's standard `PassContext` mechanism: Python `pass_configs` → `PassContext.config` → Codegen reads it → conditionally generates C++ code. When disabled, no related code is generated at all, and the compiled artifact is identical to one produced without this feature.

---

## 3. Environment Dependencies

### 3.1 CANN Version Requirements

**CANN >= 9.3.0**

The following CANN APIs are required (see `acl/acl_rt.h` and `acl/acl_dump.h`):

| API | Purpose |
|-----|---------|
| `aclrtSetExceptionInfoCallback` | Register exception callback |
| `aclrtGetArgsFromExceptionInfo` | Extract kernel args from exception info |
| `acldumpSaveExceptionInfo` | Save tensor data to dump file |

### 3.2 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ASCEND_DUMP_PATH` | Yes | Root path for dump files |
| `ASCEND_DUMP_SCENE` | Yes | Set to `aic_err_brief_dump` to enable exception dump |
| `ASCEND_HOME_PATH` | At parse time | Locates CANN's `msaicerr.py` tool |

> **Critical**: `ASCEND_DUMP_PATH` and `ASCEND_DUMP_SCENE` must be set **before** `import torch` / ACL initialization, otherwise CANN will not read these settings.

---

## 4. Usage Guide

### 4.1 Enable the Config

Pass it via `pass_configs` in the `tilelang.jit` decorator or `tilelang.compile`:

```python
import tilelang

@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
    },
)
def my_kernel(M, N, dtype="float16"):
    ...
```

### 4.2 Set Environment Variables

```python
import os

# Must be set before importing torch
os.environ["ASCEND_DUMP_PATH"] = "/tmp/exc_dump"
os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"

import torch
import tilelang
```

The `ASCEND_DUMP_SCENE` environment variable enables the feature — if not set, no tensor dump will occur even when an exception happens.

The `ASCEND_DUMP_PATH` environment variable sets the storage path for exception dump files. If not set, it defaults to the current working directory.

### 4.3 Catch Exceptions and Parse Dumps

Call `parse_exception_dump()` in the `except` block to automatically handle dump file discovery, msaicerr parsing, and tensor data reading:

```python
from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

try:
    # Execute a kernel call that may trigger an AI Core exception
    result = kernel(a, b)
except Exception as e:
    print(f"AI Core exception: {e}")

    # Parse the dump file and return a list of tensor data
    tensors = parse_exception_dump(
        dump_path="/tmp/exc_dump",
        kernel_name="main_kernel",
    )

    for t in tensors:
        data = t["data"]                # np.ndarray
        print(f"  {t['type']}[{t['index']}] dtype={t['dtype']}, "
              f"shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
```

### 4.4 Return Value

`parse_exception_dump()` returns `list[dict]`, where each dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `np.ndarray` | Tensor data (1-D; reshape as needed) |
| `type` | `str` | `"input"` / `"output"` / `"workspace"` |
| `index` | `int` | Tensor index within its type group |
| `dtype` | `str` | Data type string, e.g. `"float16"` |
| `file` | `str` | Path to the `.bin` file |

### 4.5 Manual Parsing (Optional)

If you prefer not to use `parse_exception_dump()`, you can do it step by step:

```bash
# 1. Parse the dump file with CANN's msaicerr.py
python $ASCEND_HOME_PATH/tools/msaicerr/msaicerr.py \
    -d /tmp/exc_dump/extra-info/data-dump/0/main_kernel.custom.<timestamp> \
    -out /tmp/parsed
```

```python
# 2. Read the .bin file
from tilelang.tools.ascend_exception_dump_bin import read_msaicerr_bin
import numpy as np

data = read_msaicerr_bin(
    "/tmp/parsed/main_kernel.custom.xxx.input.0.float16.bin",
    dtype=np.float16,
    shape=(128, 128),
)
```

### 4.6 Complete Example

See `examples/exception_dump_test/example_exception_dump.py`.

---

## 5. Dump File Path Convention

```
<ASCEND_DUMP_PATH>/
└── extra-info/
    └── data-dump/
        └── <devId>/                              # Device ID
            └── <kernel_name>.custom.<timestamp>   # Dump file
```

After msaicerr.py parsing, per-tensor `.bin` files are generated with the naming convention:

```
<kernel_name>.custom.<timestamp>.<tensor_type>.<tensor_index>.<dtype>.bin
```

Example: `main_kernel.custom.20260731191027177.input.0.float16.bin`

---

## 6. API Reference

### Python Config

| API | Description |
|-----|-------------|
| `tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP` | Config key, value is `"tl.ascend_exception_dump"`, default `False` |

### Python Tools

| Function | Description |
|----------|-------------|
| `parse_exception_dump(dump_path, kernel_name, output_dir, wait_seconds)` | All-in-one: find dump file → msaicerr parse → read numpy arrays |
| `read_msaicerr_bin(file_path, dtype, shape, header_size)` | Read a single `.bin` file as a numpy array |

### CANN Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `ASCEND_DUMP_PATH` | Path | Root path for dump files |
| `ASCEND_DUMP_SCENE` | `aic_err_brief_dump` | Enable AI Core exception dump |
| `ASCEND_HOME_PATH` | CANN install path | Locates the msaicerr.py tool |
