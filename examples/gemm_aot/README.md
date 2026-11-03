# TileLang-Ascend AOT Operator Example

This directory provides an example of implementing an **Ahead-of-Time (AOT)** compiled operator, using **GEMM (General Matrix Multiplication)** as a demonstration.  
It shows how an operator can be generated, compiled, and tested through a simple automated workflow.


## Overview

The example demonstrates the full workflow of building and testing an AOT-compiled operator:
1. **Operator generation** (from Python to device source)
2. **Compilation** of the generated operator code into `kernel_lib.so`
3. **Validation** of results through test scripts

Generation and compilation are both driven by the framework: `example_gemm.py`
lowers the kernel and compiles it through the same `LibraryGenerator` used by the
JIT path, so the build stays consistent with the framework's compiler flags
across CANN versions (no standalone build script to keep in sync).

---

## Directory Structure

```
gemm_aot/
├── example_gemm.py           # Lowers the GEMM kernel and compiles it into kernel_lib.so
├── test_example_gemm.py      # Script to test the compiled AOT operator
├── run_example_gemm_aot.sh   # One-click script to run the full example
└── README.md
```

`example_gemm.py` accepts `--target {ascendc,pto}` (default `ascendc`),
`--platform` (default auto-detected) and `-o/--output` (default
`./kernel_lib.so`). The generated `kernel_lib.so` is not checked into the repo.

---

## Running the Example

We assume that you have already
[installed tilelang-ascend](../../README.md#tilelang-ascend-installation).

To run the complete AOT GEMM example:

```bash
cd examples/gemm_aot
bash run_example_gemm_aot.sh
```

