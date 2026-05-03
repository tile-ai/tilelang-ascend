# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import Literal, Union
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

AVALIABLE_TARGETS = {
    "auto",
    "cuda",
    "hip",
    "webgpu",
    "c",  # represent c source backend
    "llvm",
}


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system by locating the CUDA path.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        nvcc.find_cuda_path()
        return True
    except Exception:
        return False


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def check_npu_availability() -> bool:
    """
    Check if NPU (Ascend) is available on the system by checking torch.npu.
    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        import torch
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False


def check_tensorpulse_availability() -> bool:
    """
    Check if TensorPulse hardware/toolchain is available on the system.

    Detection priority:
      1. TENSORPULSE_HOME / TENSORPULSE_HOME_PATH env var set;
      2. torch.tensorpulse module present and reporting is_available().
    """
    import os
    if os.environ.get("TENSORPULSE_HOME") or os.environ.get("TENSORPULSE_HOME_PATH"):
        return True
    try:
        import torch
        return hasattr(torch, 'tensorpulse') and torch.tensorpulse.is_available()
    except Exception:
        return False


def determine_target(target: Union[str, Target, Literal["auto"]] = "auto",
                     return_object: bool = False) -> Union[str, Target]:
    """
    Determine the appropriate target for compilation
    (CUDA, HIP, NPU/Ascend, TensorPulse, or manual selection).

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will auto-detect available hardware.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target.

    Raises:
        ValueError: If no supported hardware is available and the target is "auto".
        AssertionError: If the target is invalid.
    """

    return_var: Union[str, Target] = target

    if target == "auto":
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()
        is_npu_available = check_npu_availability()
        is_tensorpulse_available = check_tensorpulse_availability()

        if is_cuda_available:
            return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        elif is_npu_available:
            return_var = "llvm --keys=ascend"
        elif is_tensorpulse_available:
            return_var = "llvm --keys=tensorpulse"
        else:
            raise ValueError("No CUDA, HIP, NPU, or TensorPulse available on this system.")
    elif target in ["ascendc", "pto"]:
        return_var = "llvm --keys=ascend"
    elif target == "tensorpulse":
        return_var = "llvm --keys=tensorpulse"
    else:
        assert isinstance(
            target, Target) or target in AVALIABLE_TARGETS, f"Target {target} is not supported"
        return_var = target

    if return_object:
        return Target(return_var)
    return return_var

def determine_platform(platform: str = "auto") -> str:
    """
    Determine the appropriate platform for compilation (e.g., "A3", "A2").

    Args:
        platform (str): User-specified platform.
            - If "auto", the system will automatically detect the platform based on the device properties.
            - If a string, it is directly validated.

    Returns:
        str: The selected platform ("A3", "A2", etc.).
    """
    if platform != "auto":
        return platform

    # Detect platform based on NPU device properties
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            props = torch.npu.get_device_properties(torch.npu.current_device())
            name = props.name.upper()

            if "910B" in name:
                return "A2"
            elif "910_93" in name:
                return "A3"
            elif "910C" in name:
                return "A3"
            elif "950" in name:
                return "A5"
            elif "910_95" in name:
                return "A5"
            elif "910" in name:  # Covers 910A
                return "A2"
            else:
                pass
    except Exception:
        pass

    # Default fallback if detection fails
    return "A3"