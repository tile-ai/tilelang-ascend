#!/bin/bash
# package_only.sh
set -e

# 1. Ensure compilation is completed via install_npuir.sh
if [ ! -f "build/libtilelang.so" ]; then
    echo "Error: Please run ./install_npuir.sh first to complete compilation"
    exit 1
fi

# 2. Skip CMake build and package directly
export TILELANG_SKIP_BUILD=1
export USE_NPUIR=true
export BISHENGIR_PATH=$(pwd)/3rdparty/AscendNPU-IR/build/install

# 3. Clean old build artifacts and generate wheel
rm -rf build/lib build/bdist* dist *.egg-info
python setup.py bdist_wheel

echo "Packaging completed, wheel is located in dist/ directory"