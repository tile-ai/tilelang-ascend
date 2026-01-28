mkdir build && cd build
CMAKE_PREFIX_TORCH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
CMAKE_PREFIX_TORCH_NPU="path/to/libtorch_npu"
cmake -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_TORCH};${CMAKE_PREFIX_TORCH_NPU}" ..
cmake --build . --config Release