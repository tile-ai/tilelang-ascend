# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
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


def determine_target(target: Union[str, Target, Literal["auto"]] = "auto",
                     return_object: bool = False) -> Union[str, Target]:
    """
    Determine the appropriate target for compilation (CUDA, HIP, or manual selection).

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will automatically detect whether CUDA or HIP is available.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target ("cuda", "hip", or a valid Target object).

    Raises:
        ValueError: If no CUDA or HIP is available and the target is "auto".
        AssertionError: If the target is invalid.
    """

    return_var: Union[str, Target] = target

    if target == "auto":
        # Check for CUDA and HIP availability
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()
        is_npu_available = check_npu_availability()

        # Determine the target based on availability
        if is_cuda_available:
            return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        elif is_npu_available:
            # NPU (Ascend) is available, use llvm as the TVM target
            # tilelang will handle Ascend-specific compilation internally
            return_var = "llvm --keys=ascend"
        else:
            raise ValueError("No CUDA, HIP, or NPU available on this system.")
    elif target in ["ascendc", "pto"]:
        return_var = "llvm --keys=ascend"
    else:
        # Validate the target if it's not "auto"
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
            - If "auto", the system will first check TL_PLATFORM env var,
              then automatically detect the platform based on the device properties.
            - If a string, it is directly validated.

    Returns:
        str: The selected platform ("A3", "A2", etc.).
    """
    if platform != "auto":
        return platform

    # Allow explicit platform override via environment variable (useful for sim mode)
    env_platform = os.environ.get("TL_PLATFORM")
    if env_platform:
        return env_platform

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