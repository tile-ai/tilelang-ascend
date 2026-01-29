#!/bin/bash
set -x

python prepare.py || { echo "Failed to prepare libop.so"; exit 1; }  # 编译 tilelang 算子，获取 libop.so

CMAKE_PREFIX_TORCH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
CMAKE_PREFIX_TORCH_NPU="/mnt/workspace/torch_npu/libtorch_npu"  # 编译好的 libtorch_npu 的根目录
TORCH_NPU_SOURCE_DIR="${TORCH_NPU_SOURCE_DIR:-/mnt/workspace/torch_npu}"  # torch_npu 源码目录（third_party 目录所在的目录），此处值为 /mnt/workspace/torch_npu

mkdir -p build && cd build

cmake -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_TORCH};${CMAKE_PREFIX_TORCH_NPU}" \
      -DTORCH_NPU_SOURCE_DIR="${TORCH_NPU_SOURCE_DIR}" \
      ..

cmake --build . --config Release