# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import os
from typing import Literal
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

ASCEND_PLATFORM_MCPU = {
    "A2": "dav-2201",
    "A3": "dav-2201",
    "A5": "dav-3510",
}


def ascend_mcpu_for_platform(platform: str) -> str:
    try:
        return ASCEND_PLATFORM_MCPU[platform]
    except KeyError as err:
        raise ValueError(f"Unsupported Ascend platform {platform!r}; expected one of {sorted(ASCEND_PLATFORM_MCPU)}") from err


def ascend_platform_from_mcpu(mcpu: str | None) -> str | None:
    if not mcpu:
        return None
    normalized = mcpu.lower()
    if "3510" in normalized or "950" in normalized or "910_95" in normalized:
        return "A5"
    if "2201" in normalized:
        return "A2/A3"
    if "910c" in normalized or "910_93" in normalized:
        return "A3"
    if "910b" in normalized or "310p" in normalized or "910" in normalized:
        return "A2"
    return None


def ascend_platform_from_device_name(name: str) -> str | None:
    normalized = name.upper()
    if "950" in normalized or "910_95" in normalized or "3510" in normalized:
        return "A5"
    if "910_93" in normalized or "910C" in normalized:
        return "A3"
    if "910B" in normalized or "310P" in normalized or "910" in normalized:
        return "A2"
    return None


def validate_ascend_platform_device(platform: str, device_name: str) -> None:
    runtime_platform = ascend_platform_from_device_name(device_name)
    if runtime_platform is not None and runtime_platform != platform:
        raise RuntimeError(f"Ascend runtime device {device_name!r} is {runtime_platform}, but the compiled target platform is {platform}")


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

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def determine_target(target: str | Target | Literal["auto"] = "auto", return_object: bool = False) -> str | Target:
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

    return_var: str | Target = target

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
        assert isinstance(target, Target) or target in AVALIABLE_TARGETS, f"Target {target} is not supported"
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

    name = ""
    try:
        import acl

        name = acl.get_soc_name()
    except ImportError:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            name = torch.npu.get_device_name()

    name = name.upper()

    detected_platform = ascend_platform_from_device_name(name)
    if detected_platform is not None:
        return detected_platform

    # Default fallback if detection fails
    return "A3"
