#!/bin/bash

TL_ROOT=$(readlink -f "${BASH_SOURCE[0]}")
export TL_ROOT=$(dirname "$TL_ROOT")
export PYTHONPATH=${TL_ROOT}:$PYTHONPATH

# disable the import of tvm when using torch_npu
export ACL_OP_INIT_MODE=1