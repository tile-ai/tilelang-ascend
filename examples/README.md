# tilelang-npu Guide

## Installation && Environment Configuration

To install tilelang-npu, execute the following command:

```bash
bash install_ascend.sh
```

Alternatively, you can inspect the `install_ascend.sh` script to manually build the dynamic library through a custom installation process.

After the above process, configure tilelang development enviroment like this:

```bash
source set_env.sh
```

## NPU Examples Directory (`npu_example/`)

The `npu_example` directory contains resources for generating and testing NPU code:

### 1. Pre-generated AscendC Example (`examples/`)
- **What's included**: Ready-to-use AscendC code
- **How to use**:
  1. Compile the example:
     ```bash
     cd examples
     bash build.sh
     ```
  2. Test the generated dynamic library:
     ```bash
     python test.py
     ```

### 2. Matrix Multiplication Code Generator (`mm_basic.py`)
- **Functionality**: Generates C++ code for matrix multiplication kernels
- **How to use**:
  ```bash
  python mm_basic.py # a @ b
  python mm_trans.py # a @ b.T
  ```
  This will output corresponding AscendC implementation code.



