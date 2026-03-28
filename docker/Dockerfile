FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ARG CHIP_TYPE=A3
ARG CANN_VERSION=8.5.0
ARG REGION=ChinaMainland
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV DEBIAN_FRONTEND=noninteractive \
    TZ="Asia/shanghai" \
    CHIP_TYPE=${CHIP_TYPE} \
    CANN_VERSION=${CANN_VERSION} \
    REGION=${REGION} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg \
    make \
    sudo \
    unzip \
    vim \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776" \
    | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa.gpg  && \
    chmod 644 /usr/share/keyrings/deadsnakes-ppa.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-ppa.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    ccache \
    clang-15 \
    clang-format \
    cmake \
    lld-15 \
    python3-dev \
    python3-venv \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-15   100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-15 && \
    update-alternatives --install /usr/bin/lld     lld     /usr/bin/lld-15     100 && \
    update-alternatives --install /usr/bin/ld.lld  ld.lld  /usr/bin/ld.lld-15  100

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    --slave /usr/bin/python3 python3 /usr/bin/python3.9 \
    --slave /usr/bin/pip pip /usr/local/bin/pip3.9 \
    --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.9 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 \
    --slave /usr/bin/python3 python3 /usr/bin/python3.10 \
    --slave /usr/bin/pip pip /usr/local/bin/pip3.10 \
    --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3 \
    --slave /usr/bin/python3 python3 /usr/bin/python3.11 \
    --slave /usr/bin/pip pip /usr/local/bin/pip3.11 \
    --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 && \
    update-alternatives --set python /usr/bin/python3.10

# Add user tilelang (optional, but main build runs as root)
RUN useradd tilelang \
    --create-home \
    --shell=/bin/bash \
    --uid=1000 \
    --user-group && \
    echo 'Defaults env_keep += "HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy"' >> /etc/sudoers.d/env_keep

WORKDIR /home/tilelang

# update cmake
RUN apt remove --purge --auto-remove --yes cmake && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget gpg ca-certificates gnupg2 && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" | tee /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/* && \
    echo cmake version: $(cmake --version | head -n 1)

# install libzstd-dev and patch
RUN apt install --yes --no-install-recommends --no-install-suggests \
    libzstd-dev \
    patch

# Install tilelang requirements
COPY Makefile requirements.txt requirements-dev.txt build/

# Install CANN and tilelang requirements
RUN cd build && \
    make install_cann && \
    for PY in python3.9 python3.10 python3.11; do \
    ${PY} -m pip install "ninja>=1.12.0" && \
    ${PY} -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu && \
    ${PY} -m pip install "torch_npu==2.7.1" && \
    make install_dev_reqs PYTHON=${PY} && \
    make install_rt_reqs PYTHON=${PY} && \
    rm -f .req_rt_installed .req_dev_installed && \
    rm -rf ~/.cache/pip; \
    done && \
    cd .. && \
    rm -rf build


# Install TileLang for 3.9/3.10/3.11 python dev
COPY . tilelang-ascend/
RUN cd tilelang-ascend && \
    if [ "$REGION" = "ChinaMainland" ]; then \
    find . -name ".gitmodules" -type f -exec sed -i 's|https://github.com/llvm/llvm-project.git|https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git|g' {} + ; \
    fi

RUN cd tilelang-ascend && \
    git submodule sync && \
    git submodule update --init

RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    if [ "$REGION" = "ChinaMainland" ]; then \
    find . -name ".gitmodules" -type f -exec sed -i 's|https://github.com/llvm/llvm-project.git|https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git|g' {} + ; \
    fi

RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    git submodule sync && \
    git submodule update --init

RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    if [ "$REGION" = "ChinaMainland" ]; then \
    find . -name ".gitmodules" -type f -exec sed -i 's|https://github.com/llvm/llvm-project.git|https://mirrors.tuna.tsinghua.edu.cn/git/llvm-project.git|g' {} + ; \
    fi

RUN cd tilelang-ascend/3rdparty/AscendNPU-IR && \
    bash ./build-tools/apply_patches.sh && \
    rm -rf ./build && \
    ./build-tools/build.sh -o ./build --python-binding --c-compiler=clang --cxx-compiler=clang++ --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" --apply-patches --bishengir-publish=off

# Copy compiled build files and clean up 3rdparty to save space
RUN mkdir -p /build/AscendNPU-IR/build && \
    cp -r tilelang-ascend/3rdparty/AscendNPU-IR/build/* /build/AscendNPU-IR/build/

RUN cd tilelang-ascend && \
    bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install

RUN rm -rf tilelang-ascend/3rdparty/AscendNPU-IR

# Set environment variables for CANN
RUN echo "source /usr/local/Ascend/cann/set_env.sh" >> ~/.bashrc

# Add welcome message for container usage
RUN echo "Welcome to TileLang AscendNPU IR Docker container!" > /etc/motd && \
    echo "" >> /etc/motd && \
    echo "TileLang URL: https://github.com/tile-ai/tilelang-ascend/tree/npuir" >> /etc/motd && \
    echo "" >> /etc/motd && \
    echo "Pre-compiled AscendNPU-IR binaries are available at /build/AscendNPU-IR/build/" >> /etc/motd && \
    echo "To reinstall or update NPU IR with pre-compiled binaries (faster), use:" >> /etc/motd && \
    echo "  bash install_npuir.sh --bishengir-path=/build/AscendNPU-IR/build/install" >> /etc/motd && \
    echo "" >> /etc/motd

# Show welcome message when entering an interactive bash shell via docker exec.
RUN grep -q "TileLang MOTD" /etc/bash.bashrc || \
    printf '\n# TileLang MOTD\nif [[ $- == *i* ]] && [[ -f /etc/motd ]]; then\n  cat /etc/motd\nfi\n' >> /etc/bash.bashrc
