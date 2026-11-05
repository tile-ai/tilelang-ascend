# overview 

This document describes how to run the Dispatch operator and the Combine operator in Tilelang.

# Environment Preparation ## 

## Hardware Requirements 

- Product Model: Atlas Series: 800I A3、800T A3
- Operating System: Linux ARM

## Software Dependencies 

- Driver and Firmware（*.run）: Ascend HDK 25.3.RC1.2。Download Link: [Driver and Firmware Resources](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=34&cann=8.3.RC1.alpha003&driver=Ascend+HDK+25.3.RC1.2) 
   ```
   # example
   Atlas-A3-hdk-npu-firmware_7.8.0.2.212.run
   Atlas-A3-hdk-npu-driver_25.3.rc1.2_linux-aarch64.run
   ```
- CANN Version (Select one according to the function):
  - 8.2.RC1.alpha003 and above (Community Edition) support the D2D function. Download Link: [Community Edition Resources](https://www.hiascend.com/developer/download/community/result?module=cann). Select "AArch64" for CPU architecture, select "run" for software package format, and download the .run package with "A3-ops" and "toolkit" in the name
  ```
   # example
   Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
   Ascend-cann-toolkit_8.5.0_linux-aarch64.run
   ```
- Toolchain:
  - cmake ≥ 3.19  
  - GLIBC ≥ 2.28

## Software Package Installation

1. Install the driver and firmware, install the CANN toolkit package, and install the CANN ops package. (Users can customize the installation paths or use the default ones.)

   ```
   # Install the driver and firmware packages 
   # Install the firmware package first, then the driver package
   chmod +x *.run
   bash Atlas-A3-hdk-npu-firmware_7.8.0.2.212.run --upgrade
   bash Atlas-A3-hdk-npu-driver_25.3.rc1.2_linux-aarch64.run --upgrade
   # Restart
   reboot
   # Set the install path and all its parent directories to permission 755
   chmod +x *.run
   chmod 755 ${install_path}
   chmod 755 ${parent directories path}
   # Install the CANN package
   ./Ascend-cann-A3-ops_8.5.0_linux-aarch64.run --install --install-path=${install_path}  
   ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install --install-path=${install_path}
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

# Cross-machine Adaptation

1. When initializing the cross-machine shmem, when configuring the object created by "aclshmem_module.InitAttr()", its "ip_port" attribute should be set to the IP address and port of the master node:

   ```
   For example: IP_1 is the IP address plus port of Machine 1 (the master node)
   # Machine 1 (Master Node):：
   attributes = aclshmem_module.InitAttr()
   attributes.ip_port = IP_1
   # Machine 2 (Slave Node):
   attributes = aclshmem_module.InitAttr()
   attributes.ip_port = IP_1
   ```

2. The "n_ranks of" the shmem object needs to be set to the total number of ranks (for example, if there are 2 machines with 16 ranks per machine, then the "n_ranks" of the shmem object on both machines should be set to 32):

   ```
   attributes.n_ranks = 32
   ```

3. When creating a tensor using the "aclshmem_module", the "device_id" needs to be taken the remainder of the number of ranks on a single machine (assuming the number of enabled ranks is consistent across all machines):

   ```
   device_id=rank % 16
   ```

# Whole Network Adaptation Strategy:

- Please note that the Tilelang operator may temporarily conflict with the TBE operator. If TBE operators appear in the whole network, they need to be bypassed.
- Please note that the currently developed dispatch & combine operators are in non-quantization mode. Quantization should be disabled for the entire network. On this basis, features such as quantization can be further developed to meet specific requirements.
- For the cross-machine adaptation of the entire network, it should also be carried out in accordance with the aforementioned cross-machine adaptation ideas, including the setting of the attributes of the shmem object, as well as the cross-machine IP and total number of ranks in the entire network, all of which are common adaptation points.
- The shmem component should avoid repeated initialization and free operations. It is recommended to define a dedicated function (e.g., init_shmem_once()) to initialize shmem exactly once.
- Graph mode is not currently supported.