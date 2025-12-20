#!/bin/bash

# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

# Add command line option parsing
USE_LLVM=false
USE_SHMEM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-llvm)
            USE_LLVM=true
            shift
            ;;
        --enable-shmem)
            USE_SHMEM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enable-llvm] [--enable-shmem]"
            exit 1
            ;;
    esac
done

echo "Starting installation script..."
echo "LLVM enabled: $USE_LLVM"
echo "SHMEM enabled: $USE_SHMEM"

# Step 1: Install Python requirements
echo "Installing Python requirements from requirements.txt..."
pip install -r requirements-build.txt
pip install -r requirements.txt
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
git submodule update --init --recursive

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build

echo "set(USE_ASCEND ON)" >> config.cmake

echo "Running CMake for TileLang..."
cmake ..
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
echo "export PYTHONPATH=${TILELANG_PATH}:\$PYTHONPATH" >> ~/.bashrc

# compile and install aclshmem package
if $USE_SHMEM; then
    echo "Starting installation aclshmem..."
    cd 3rdparty/aclshmem-dev
    bash scripts/build.sh --package --python_extension
    ACLSHMEM_INSTALL_PATH=$(pwd)/install
    arch=$(uname -m)
    cd package/$arch/
    chmod +x ACLSHMEM*.run
    ./ACLSHMEM*.run --check
    ./ACLSHMEM*.run --install --install-path=$ACLSHMEM_INSTALL_PATH
    if [ $? -ne 0 ]; then
        echo "Error: ACLSHMEM C++ pkg install failed."
        exit 1
    else
        echo "ACLSHMEM C++ pkg install success in $ACLSHMEM_INSTALL_PATH."
    fi
    cd ../../src/python/dist
    python -m pip install aclshmem*.whl --force-reinstall
    if [ $? -ne 0 ]; then
        echo "python -m pip install failed, try pip3 install ..."
        pip3 install aclshmem*.whl --force-reinstall
        if [ $? -ne 0 ]; then
            echo "Error: aclshmem-xxx.whl install failed."
            exit 1
        else
            echo "aclshmem-xxx.whl install success."
        fi
    else
        echo "aclshmem-xxx.whl install success."
    fi
    source ../../../install/aclshmem/latest/set_env.sh
    if [ $? -ne 0 ]; then
        echo "Error: set aclshmem env failed."
        exit 1
    fi
    # back to path tilelang-ascend/
    cd ../../../../..
    echo "Install aclshmem all success."
fi

# Step 12: Source .bashrc to apply changes
echo "Applying environment changes by sourcing .bashrc..."
source ~/.bashrc
if [ $? -ne 0 ]; then
    echo "Error: Failed to source .bashrc."
    exit 1
else
    echo "Environment configured successfully."
fi

echo "Installation script completed successfully."
exec bash

