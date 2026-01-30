# Example of integrating TileLang-Ascend operators in C++ (using libtorch)

This directory provides an example of integrating TileLang-Ascend operators into a C++ application using libtorch and libtorch_npu.

At present, the example operator in [`examples/flash_attention/flash_attn_bhsd`](../../flash_attention/flash_attn_bhsd.py) is supported.

The example operator is [pre-compiled at building](prepare.py) and dynamically loaded at runtime using `dlopen`.

## Build

With the following prerequisites:
- [TileLang-Ascend installed](../../../README.md#tilelang-ascend-installation)
- [libtorch_npu compiled from source](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/building_libtorch_npu.md)

Edit libtorch_npu-related paths in `build.sh`:
```bash
...
CMAKE_PREFIX_TORCH_NPU="path/to/pytorch/libtorch_npu"  # root path to compiled libtorch_npu
TORCH_NPU_SOURCE_DIR="path/to/pytorch"  # path to torch_npu source code (where "third_party" locates)
...
```

Then, the `flash_attention_demo` can be built with:

```bash
bash build.sh
```

## Test

```bash
./build/flash_attention_demo
```
The output should be:
```
init successful!
Loaded libop.so from: ./lib/libop.so
Test Passed!
```

## Basic Usage

Call the integrated operator in C++ with tensors on the NPU device:

```cpp
torch::Tensor output = flash_attention_fwd(q, k, v);
```

See [`flash_attention.cpp`](./flash_attention.cpp) for details.

## Directory Structure

```
demo_libtorch/
├── build.sh             # Script for building the demo
├── prepare.py           # Script for preparing compiled TileLang-Ascend operators in lib/ (Using compile_tl_op/)
├── CMakeLists.txt       # CMake configuration
├── flash_attention.cpp  # Main code of the demo
├── README.md            # This document
└── lib/                 # Directory for compiled operators (.so)

compile_tl_op/           # Utilities for compiling operators
```