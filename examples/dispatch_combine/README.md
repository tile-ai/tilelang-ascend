# 概述 

本文档介绍如何运行 Tilelang 里的 Dispatch 算子和 Combine 算子。 

# 环境准备 ## 

## 硬件要求 

- 产品型号：Atlas系列：800I A2/A3、800T A2/A3
- 操作系统：Linux ARM

## 软件依赖 

- 驱动固件：Ascend HDK 25.0.RC1.1。下载链接：[驱动固件资源](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=All&driver=Ascend+HDK+25.0.RC1) 

- CANN版本（根据功能选择其一）：
  - 8.2.RC1.alpha003 及以上（社区版）支持 D2D 功能。下载链接：[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) 
  - 8.5.0 及以上（尝鲜版）支持 D2D、D2H、H2D 功能。下载链接：[尝鲜版链接](https://ascend.devcloud.huaweicloud.com/cann/run/software/)
- 工具链：
  - cmake ≥ 3.19  
  - GLIBC ≥ 2.28

## 软件包安装

1. 安装驱动与固件，安装 CANN toolkit 包，安装 CANN ops 包。（用户可自定义安装路径，也可采用默认安装路径）

   ```
   # 安装驱动包（先安装firmware包，再安装driver包）
   bash Atlas-A3-hdk-npu-firmware_7.7.0.3.228.run --upgrade
   bash Atlas-A3-hdk-npu-driver_25.0.rc1.3_linux-aarch64.run --upgrade
   # 重启
   reboot
   # 安装CANN包
   ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install --install-path=${install_path}  ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install --install-path=${install_path}
   ```

2. 获取 Tielalng源码

   ```
   git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
   ```

## 设置环境变量

1. 设置 CANN 包的环境变量

   ```
   # 用户自定义安装路径   
   source ${install_path}/ascend-toolkit/latest/bin/setenv.bash
   ```

2. 设置 tilelang 的环境变量

   ```
   cd tilelang-ascend   
   bash install_ascend.sh --enable-shmem   
   source set_env.sh
   ```

# 运行算子

1. 配置IP

   ```
   vim tilelang-ascend/examples/dispatch_combine/dispatch_combine_shmem.py 
   # G_IP_PORT设置为本机IP(100.102.280.145改为本机IP) 
   G_IP_PORT = "tcp://100.102.180.145:8666" 
   ```

2. 运行代码

   ```
   python tilelang-ascend/examples/dispatch_combine/dispatch_combine_shmem.py
   ```