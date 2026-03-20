# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import logging


class AscendArch:
    """Base class for Ascend architecture capabilities."""

    def __init__(self, name: str):
        self.name = name

    @property
    def supports_native_bf16(self) -> bool:
        # Currently, all supported chips require legalization for BF16.
        return False


class AscendArch910B(AscendArch):
    """Specific properties for Ascend 910B series."""

    pass


class AscendArch910_95(AscendArch):
    """Specific properties for Ascend 910_95 series."""

    pass


# Map device name prefixes to their corresponding architecture classes.
# Order matters if prefixes overlap.
ARCH_MAP = {
    "Ascend910B": AscendArch910B,
    "Ascend910_95": AscendArch910_95,
}


def get_arch_obj(device_name: str) -> AscendArch:
    """Identify the architecture type based on the device name prefix."""
    for prefix, arch_cls in ARCH_MAP.items():
        if device_name.startswith(prefix):
            return arch_cls(device_name)
    # Default fallback for unknown architectures.
    return AscendArch(device_name)


def get_ascend_device_name() -> str:
    # 1. Highest priority: User-specified environment variable
    #    Useful for cross-compilation or overriding runtime detection.
    device_name = os.environ.get("TILELANG_ASCEND_DEVICE_NAME")
    if device_name:
        return device_name.strip()

    # 2. Secondary priority: Runtime capability detection
    try:
        from tilelang.utils import NPUUtils

        return NPUUtils.get().get_arch()
    except Exception as e:
        # We don't want to crash on non-Ascend machines, but silent pass is bad for debugging
        logging.getLogger(__name__).warning(
            f"Failed to get Ascend arch from NPUUtils: {e}. "
            "Please set TILELANG_ASCEND_DEVICE_NAME environment variable."
            "Otherwise we will fallback to Ascend910B."
        )

    # 3. Fallback to Ascend910B if runtime detection fails
    return "Ascend910B"


def supports_native_bf16(device_name: str) -> bool:
    """Check if the given device natively supports BF16 instructions."""
    arch = get_arch_obj(device_name)
    return arch.supports_native_bf16
