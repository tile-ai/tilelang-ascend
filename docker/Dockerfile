FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

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

RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776 && \
    apt-get update

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    clang \
    clang-format \
    cmake \
    lld \
    ninja-build \
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
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
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

# Add user tilelang
RUN useradd tilelang \
    --create-home \
    --shell=/bin/bash \
    --uid=1000 \
    --user-group && \
    echo "tilelang ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers.d/nopasswd
	
USER tilelang

WORKDIR /home/tilelang

# update cmake
RUN sudo apt remove --purge --auto-remove --yes \
    cmake && \
    sudo apt update && \
    sudo apt install ca-certificates gpg wget

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
    
RUN sudo apt update

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg

RUN sudo apt install kitware-archive-keyring

RUN sudo apt update && \
    sudo apt install --yes --no-install-recommends --no-install-suggests \
    cmake

# update ninja-build
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip && \
    unzip ninja-linux.zip -d ninja-linux && \
    export PATH=/home/tilelang/ninja-linux:$PATH && \
    echo "PATH=/home/tilelang/ninja-linux:$PATH" >> /home/tilelang/.bashrc && \
    sudo rm ninja-linux.zip

# install libzstd-dev and patch
RUN sudo apt update && \
    sudo apt install --yes --no-install-recommends --no-install-suggests \
    libzstd-dev \
    patch