# Example of intergrating TileLang-Ascend operators in PyTorch (as a Python C binding)

This directory provides an example of intergrating TileLang-Ascend operators into PyTorch in the form of a Python C binding. Specifically, A python package named `torch_tl_ascend` is shipped inside.

At present, the example operator in [`examples/flash_attention/flash_attn_bhsd`](../flash_attention/flash_attn_bhsd.py) is supported.

## Build & Install

With [tilelang-ascend installed](../../README.md#tilelang-ascend-installation), the `torch_tl_ascend` python package can be built and installed with:

```bash
python setup.py install
```

For building the `.whl` file only:

```bash
python setup.py bdist_wheel
```

## Test

To test the PyTorch operator integration (i.e. `torch.ops.tl_ascend.flash_attention`):

```bash
python test_torch.py
```

To test the packaged source code of the integrated TileLang-Ascend operator (i.e. `torch_tl_ascend.op_source.flash_attn_bhsd.flash_attention_fwd`):

```bash
python test_source.py
```

## Basic Usage

To call integrated operators in PyTorch, Please refer to [test_torch.py](./test_torch.py) 

```python
import torch
import torch_tl_ascend
...
output = torch.ops.tl_ascend.flash_attention(q, k, v)
```

Source code of integrated operators are also packaged. To call them, please refer to [test_source.py](./test_source.py)

```python
from torch_tl_ascend.op_source.flash_attn_bhsd import flash_attention_fwd
...
kernel = flash_attention_fwd(B, S, H, D)
output = kernel(q, k, v)
```

## Directory Structure

```
torch_tl_ascend/
├── compile_tl_op        # Utilities for compiling TileLang-Ascend operators and assembling the package
├── src
│   ├── torch_tl_ascend  # The torch_tl_ascend package
│   └── _inner.cpp       # The Python C module of the package for wrapping operators and registering to PyTorch
├── README.md            # This document
├── setup.py             # Script for building and installing the package
├── test_source.py       # Script for testing the PyTorch operator integrations
└── test_torch.py        # Script for testing the packaged source code of integrated operators
```