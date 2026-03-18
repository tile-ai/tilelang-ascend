#!/bin/bash

# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

PYTHON="$(command -v python3 2>/dev/null)" || PYTHON="$(command -v python 2>/dev/null)"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "Error: No python3/python found in PATH. Activate your venv/conda and re-run." >&2
    exit 1
fi
PYTHON_DIR="$(dirname "$PYTHON")"
export PATH="${PYTHON_DIR}:$PATH"
echo "Using Python (current env): $PYTHON"
$PYTHON --version

# Add command line option parsing
USE_LLVM=false
BISHENGIR_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-llvm)
            USE_LLVM=true
            shift
            ;;
        --bishengir-path=*)
            BISHENGIR_PATH="${1#*=}"
            shift
            ;;
        --bishengir-path)
            if [ -n "$2" ]; then
                BISHENGIR_PATH="$2"
                shift 2
            else
                echo "err: --bishengir-path needs to be specified with bishengir-compile install path" >&2
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enable-llvm] [--bishengir-path=DIR]"
            exit 1
            ;;
    esac
done

echo "Starting installation script..."
echo "LLVM enabled: $USE_LLVM"

# Step 1: Install Python requirements
echo "Installing Python requirements from requirements.txt..."
"$PYTHON" -m pip install -r requirements-build.txt
"$PYTHON" -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi

# Step 2: Define LLVM version and architecture
if $USE_LLVM; then
    LLVM_VERSION="10.0.1"
    IS_AARCH64=false
    EXTRACT_PATH="3rdparty"
    echo "LLVM version set to ${LLVM_VERSION}."
    echo "Is AARCH64 architecture: $IS_AARCH64"

    # Step 3: Determine the correct Ubuntu version based on LLVM version
    UBUNTU_VERSION="16.04"
    if [[ "$LLVM_VERSION" > "17.0.0" ]]; then
        UBUNTU_VERSION="22.04"
    elif [[ "$LLVM_VERSION" > "16.0.0" ]]; then
        UBUNTU_VERSION="20.04"
    elif [[ "$LLVM_VERSION" > "13.0.0" ]]; then
        UBUNTU_VERSION="18.04"
    fi
    echo "Ubuntu version for LLVM set to ${UBUNTU_VERSION}."

    # Step 4: Set download URL and file name for LLVM
    BASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
    if $IS_AARCH64; then
        FILE_NAME="clang+llvm-${LLVM_VERSION}-aarch64-linux-gnu.tar.xz"
    else
        FILE_NAME="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-${UBUNTU_VERSION}.tar.xz"
    fi
    DOWNLOAD_URL="${BASE_URL}/${FILE_NAME}"
    echo "Download URL for LLVM: ${DOWNLOAD_URL}"

    # Step 5: Create extraction directory
    echo "Creating extraction directory at ${EXTRACT_PATH}..."
    mkdir -p "$EXTRACT_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create extraction directory."
        exit 1
    else
        echo "Extraction directory created successfully."
    fi

    # Step 6: Download LLVM
    echo "Downloading $FILE_NAME from $DOWNLOAD_URL..."
    curl -L -o "${EXTRACT_PATH}/${FILE_NAME}" "$DOWNLOAD_URL"
    if [ $? -ne 0 ]; then
        echo "Error: Download failed!"
        exit 1
    else
        echo "Download completed successfully."
    fi

    # Step 7: Extract LLVM
    echo "Extracting $FILE_NAME to $EXTRACT_PATH..."
    tar -xJf "${EXTRACT_PATH}/${FILE_NAME}" -C "$EXTRACT_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Extraction failed!"
        exit 1
    else
        echo "Extraction completed successfully."
    fi

    # Step 8: Determine LLVM config path
    LLVM_CONFIG_PATH="$(realpath ${EXTRACT_PATH}/$(basename ${FILE_NAME} .tar.xz)/bin/llvm-config)"
    echo "LLVM config path determined as: $LLVM_CONFIG_PATH"
fi

# Step 9: Clone and build TVM
echo "Cloning TVM repository and initializing submodules..."
# clone and build tvm
git submodule update --init --recursive 3rdparty/catlass 3rdparty/composable_kernel 3rdparty/cutlass 3rdparty/tvm

if [ -z "$BISHENGIR_PATH" ]; then
    echo "warring: no --bishengir-path set, bishengir path will be found in environment variable PATH"
    # build bishengir in 3rdparty
    echo "build bishengir in 3rdparty"
    git submodule update --init --recursive 3rdparty/AscendNPU-IR
    pushd 3rdparty/AscendNPU-IR
    bash ./build-tools/apply_patches.sh
    rm -rf ./build
    ./build-tools/build.sh -o ./build --python-binding --c-compiler=clang --cxx-compiler=clang++ \
    --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" --apply-patches --bishengir-publish=off
    BISHENGIR_PATH="./3rdparty/AscendNPU-IR/build/install"
    popd
fi

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build

echo "set(USE_NPUIR ON)" >> config.cmake
echo "set(BISHENGIR_ROOT_PATH $BISHENGIR_PATH)" >> config.cmake

echo "Running CMake for TileLang..."
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE="$PYTHON" ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."

# Calculate 75% of available CPU cores
# Other wise, make will use all available cores
# and it may cause the system to be unresponsive
CORES=$(nproc)
MAKE_JOBS=$(( CORES * 75 / 100 ))
make -j${MAKE_JOBS}

if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..

# Step 11: Set environment variables
TILELANG_PATH="$(pwd)"
echo "TileLang path set to: $TILELANG_PATH"
echo "Configuring environment variables for TVM..."

TILELANG_EXPORT_COMMAND="export PYTHONPATH=${TILELANG_PATH}:\$PYTHONPATH"
if ! grep -Fxq "$TILELANG_EXPORT_COMMAND" ~/.bashrc; then
    echo "$TILELANG_EXPORT_COMMAND" >> ~/.bashrc
    echo "$TILELANG_EXPORT_COMMAND updated in ~/.bashrc"
else
    echo "$TILELANG_EXPORT_COMMAND already exists in ~/.bashrc"
fi

# NPUIR runtime: require AscendNPU-IR python_packages (mlir_core + bishengir) and add to PYTHONPATH.
BISHENGIR_ABS="$(cd "$(dirname "$BISHENGIR_PATH")" 2>/dev/null && pwd)/$(basename "$BISHENGIR_PATH")"
if [ ! -d "$BISHENGIR_ABS" ]; then
    BISHENGIR_ABS="$(realpath "$BISHENGIR_PATH" 2>/dev/null)" || BISHENGIR_ABS="$BISHENGIR_PATH"
fi
BISHENGIR_PY_PKGS="${BISHENGIR_ABS}/python_packages"
if [ ! -d "${BISHENGIR_PY_PKGS}/mlir_core" ]; then
    echo "Error: NPUIR requires python_packages/mlir_core; not found under ${BISHENGIR_ABS}" >&2
    exit 1
fi
if [ ! -d "${BISHENGIR_PY_PKGS}/bishengir" ]; then
    echo "Error: NPUIR requires python_packages/bishengir; not found under ${BISHENGIR_ABS}" >&2
    exit 1
fi
BISHENGIR_PYTHON="${BISHENGIR_PY_PKGS}/mlir_core:${BISHENGIR_PY_PKGS}/bishengir"
if ! grep -Fq "${BISHENGIR_PY_PKGS}/mlir_core" ~/.bashrc; then
    echo "export PYTHONPATH=${BISHENGIR_PYTHON}:\$PYTHONPATH" >> ~/.bashrc
    echo "Added AscendNPU-IR python_packages (mlir_core + bishengir) to PYTHONPATH for NPUIR."
fi

echo "NOTE: Please run \"source ~/.bashrc\" or relaunch the terminal to apply the environment changes"

echo "Installation script completed successfully."
