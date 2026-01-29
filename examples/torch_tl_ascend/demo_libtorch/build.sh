#!/bin/bash
set -x

python prepare.py || { echo "Failed to prepare libop.so"; exit 1; }  # compile tilelang operator and get libop.so

CMAKE_PREFIX_TORCH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
CMAKE_PREFIX_TORCH_NPU="/mnt/workspace/torch_npu/libtorch_npu"  # root path to compiled libtorch_npu
TORCH_NPU_SOURCE_DIR="${TORCH_NPU_SOURCE_DIR:-/mnt/workspace/torch_npu}"  # source path to torch_npu (where "third_party" locates), default /mnt/workspace/torch_npu

mkdir -p build && cd build

cmake -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_TORCH};${CMAKE_PREFIX_TORCH_NPU}" \
      -DTORCH_NPU_SOURCE_DIR="${TORCH_NPU_SOURCE_DIR}" \
      ..

cmake --build . --config Release