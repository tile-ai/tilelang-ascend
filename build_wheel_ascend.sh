#!/bin/bash

# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

# Build wheel package for Ascend platform

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Parse command line arguments
USE_LLVM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-llvm)
            USE_LLVM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enable-llvm]"
            exit 1
            ;;
    esac
done

# Check ASCEND_HOME_PATH
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Error: ASCEND_HOME_PATH is not set."
    echo "Please set ASCEND_HOME_PATH before building, e.g.:"
    echo "  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"
    exit 1
fi

echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "LLVM enabled: $USE_LLVM"

# Export environment variables for setup.py
export USE_ASCEND=true
export USE_LLVM=$USE_LLVM

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
IFS='.' read -r major minor <<< "$python_version"
if (( major >= 3 && minor >= 10 )); then
    echo "Python version $python_version >= 3.10, pass"
else
    echo "[ERROR] Python version $python_version < 3.10, please upgrade it."
    exit 1
fi

# Install build requirements
echo "Installing build requirements..."
pip install -r requirements-build.txt
pip install -r requirements.txt

# Update git submodules
echo "Updating git submodules..."
git submodule update --init --recursive

# Clean previous build
if [ -d dist ]; then
    rm -r dist
fi
if [ -d build ]; then
    rm -r build
fi
if [ -d tilelang.egg-info ]; then
    rm -r tilelang.egg-info
fi

# Build wheel
echo "Building wheel package for Ascend..."
python setup.py bdist_wheel

if [ $? -ne 0 ]; then
    echo "Error: Failed to build the wheel."
    exit 1
fi

echo "Wheel built successfully."
echo "Wheel files:"
ls -la dist/

# Print installation instructions
echo ""
echo "=============================================="
echo "Build completed successfully!"
echo "=============================================="
echo ""
echo "To install the wheel package:"
echo "  pip install dist/tilelang-*.whl"
echo ""
echo "Usage in Python:"
echo "  import tilelang as tla"
echo ""
echo "Note: Make sure ASCEND_HOME_PATH is set before running your program:"
echo "  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"
echo ""