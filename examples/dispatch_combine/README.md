# overview 

This document describes how to run the Dispatch operator and the Combine operator in Tilelang.

# Environment Preparation ## 

## Hardware Requirements 

- Product Model: Atlas Series: 800I A2/A3、800T A2/A3
- Operating System: Linux ARM

## Software Dependencies 

- Driver and Firmware: Ascend HDK 25.0.RC1.1。Download Link: [Driver and Firmware Resources](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=All&driver=Ascend+HDK+25.0.RC1) 

- CANN Version (Select one according to the function):
  - 8.2.RC1.alpha003 and above (Community Edition) support the D2D function. Download Link: [Community Edition Resources](https://www.hiascend.com/developer/download/community/result?module=cann) 
  - 8.5.0 and above (Early Access Edition) support D2D, D2H and H2D functions. Download Link: [Early Access Edition Link](https://ascend.devcloud.huaweicloud.com/cann/run/software/)
- Toolchain:
  - cmake ≥ 3.19  
  - GLIBC ≥ 2.28

## Software Package Installation

1. Install the driver and firmware, install the CANN toolkit package, and install the CANN ops package. (Users can customize the installation paths or use the default ones.)

   ```
   # Install the driver and firmware packages 
   # Install the firmware package first, then the driver package
   bash Atlas-A3-hdk-npu-firmware_7.7.0.3.228.run --upgrade
   bash Atlas-A3-hdk-npu-driver_25.0.rc1.3_linux-aarch64.run --upgrade
   # Restart
   reboot
   # Install the CANN package
   ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install --install-path=${install_path}  ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install --install-path=${install_path}
   ```

2. Obtain the Tielang source code

   ```
   git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
   ```

## Set the environment variables

1. Set the environment variables for the CANN packages

   ```
   # User-defined installation path  
   source ${install_path}/ascend-toolkit/latest/bin/setenv.bash
   ```

2. Set the environment variables for TileLang and Shmem

   ```
   cd tilelang-ascend   
   bash install_ascend.sh --enable-shmem   
   source set_env.sh
   source 3rdparty/shmem/install/set_env.sh
   ```

# Run the operator

1. Configure the IP

   ```
   vim tilelang-ascend/examples/dispatch_combine/dispatch_combine_shmem.py 
   # Set G_IP_PORT to the local IP address
   # Replace xxx.xxx.xxx.xxx with the local IP address and xxxx with the port number 
   G_IP_PORT = "tcp://xxx.xxx.xxx.xxx:xxxx" 
   ```

2. Run the code

   ```
   python tilelang-ascend/examples/dispatch_combine/dispatch_combine_shmem.py
   ```