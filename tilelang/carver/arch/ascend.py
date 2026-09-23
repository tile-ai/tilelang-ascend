from __future__ import annotations

from tvm.target import Target
from .arch_base import TileDevice
from tilelang.utils.target import ascend_platform_from_device_name, ascend_platform_from_mcpu
import logging

# check if torch_npu is available, if not, degrade using CHIP_SPECS only
try:
    import torch
    import torch_npu  # noqa: F401  # Registers the torch.npu device backend.

    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_ascend_arch(arch: TileDevice) -> bool:
    return isinstance(arch, Ascend)


# Chip specifications (sizes in bytes)
CHIP_SPECS = {
    "Ascend950": {
        # DAV_3510 core counts vary by physical SKU and vNPU profile.  They
        # must come from the active device instead of a model-name default.
        "cores": None,
        "UB": 248 * 1024,
        "L1": 512 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 256 * 1024,
        "L2": None,
        "cube": [16, 16, 16],
    },
    "Ascend910A": {
        "cores": 32,
        "UB": 256 * 1024,
        "L1": 1024 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 256 * 1024,
        "L2": 16 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
    "Ascend910B": {
        "cores": 30,  # Default, can vary
        "UB": 192 * 1024,  # 910B UB size
        "L1": 1024 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 512 * 1024,
        "L2": 16 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
    "Ascend310P": {
        "cores": 8,
        "UB": 256 * 1024,
        "L1": 512 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 256 * 1024,
        "L2": 8 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
}
DEFAULT_CHIP = "Ascend910A"


class CubeInstruction:
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class Ascend(TileDevice):
    def __init__(self, target: Target | str = "llvm -keys=ascend", chip_name: str | None = None):
        if isinstance(target, str):
            target = Target(target)
        self.target = target
        self.platform = "ascend"

        target_platform = ascend_platform_from_mcpu(getattr(target, "mcpu", None))
        target_chip_name = None
        if target_platform == "A5":
            target_chip_name = "Ascend950"
        elif target_platform == "A3":
            target_chip_name = "Ascend910A"
        elif target_platform == "A2":
            mcpu = str(getattr(target, "mcpu", "")).lower()
            target_chip_name = "Ascend910B" if "910b" in mcpu else "Ascend910A"

        requested_chip_name = chip_name or target_chip_name

        # Query properties whenever possible.  In particular, an explicit A5
        # profile still cannot supply a SKU-independent core count.
        detected_cores = None
        L2_cache_size_bytes = None
        runtime_chip_name = None
        runtime_platform = None
        if _TORCH_NPU_AVAILABLE:
            try:
                if torch.npu.is_available() and torch.npu.device_count() > 0:
                    # Get the current NPU device properties
                    props = torch.npu.get_device_properties(torch.npu.current_device())
                    npu_name = props.name.upper()  # e.g., "ASCEND910B"

                    runtime_platform = ascend_platform_from_device_name(npu_name)
                    if runtime_platform == "A5":
                        runtime_chip_name = "Ascend950"
                    elif "910B" in npu_name:
                        runtime_chip_name = "Ascend910B"
                    elif "310P" in npu_name:
                        runtime_chip_name = "Ascend310P"
                    elif runtime_platform in {"A2", "A3"}:
                        runtime_chip_name = "Ascend910A"

                    if hasattr(props, "cube_core_num"):
                        detected_cores = props.cube_core_num
                    # torch_npu exposes this with a capital ``L2``.  Keep the
                    # lowercase fallback for older vendor builds, but prefer
                    # the public property spelling used by current CANN.
                    if hasattr(props, "L2_cache_size"):
                        L2_cache_size_bytes = props.L2_cache_size
                    elif hasattr(props, "l2_cache_size"):
                        L2_cache_size_bytes = props.l2_cache_size

                    logger.debug(f"Detected Ascend NPU: {npu_name}, runtime profile: {runtime_chip_name}")
            except Exception as e:
                logger.warning(f"Failed to detect Ascend NPU properties from torch_npu: {e}")

        requested_platform = target_platform
        if requested_platform is None and requested_chip_name is not None:
            if requested_chip_name == "Ascend950":
                requested_platform = "A5"
            elif requested_chip_name in {"Ascend910B", "Ascend310P"}:
                requested_platform = "A2"
            elif requested_chip_name == "Ascend910A":
                requested_platform = "A3"
        if requested_platform is not None and runtime_platform is not None and runtime_platform not in requested_platform.split("/"):
            raise RuntimeError(f"Target profile {requested_chip_name} does not match runtime device profile {runtime_chip_name}")

        # Explicit chip_name, then target mcpu, then runtime detection, then fallback.
        chip_name = requested_chip_name or runtime_chip_name or DEFAULT_CHIP

        self.chip_name = chip_name
        spec = CHIP_SPECS.get(chip_name, CHIP_SPECS[DEFAULT_CHIP]).copy()

        # AI Core size
        if detected_cores is not None and detected_cores > 0:
            self.compute_max_core = detected_cores
        elif spec["cores"] is None:
            raise RuntimeError(
                f"{chip_name} core count is device/SKU-specific; "
                "query torch.npu.get_device_properties(...).cube_core_num on the target device"
            )
        else:
            self.compute_max_core = spec["cores"]

        # Memory unit sizes
        self.ub_cap = spec["UB"]
        self.l1_cap = spec["L1"]
        self.l0a_cap = spec["L0A"]
        self.l0b_cap = spec["L0B"]
        self.l0c_cap = spec["L0C"]
        if L2_cache_size_bytes is not None and L2_cache_size_bytes > 0:
            self.l2_cache_size_bytes = L2_cache_size_bytes
        elif spec["L2"] is None:
            raise RuntimeError(
                f"{chip_name} L2 size is device/SKU-specific; query torch.npu.get_device_properties(...).L2_cache_size on the target device"
            )
        else:
            self.l2_cache_size_bytes = spec["L2"]

        self.cube_spec = spec.get("cube", [16, 16, 16])

        # Map to generic TileDevice properties
        # For Ascend, UB is the primary "shared" memory constraint for tiling, all L0X units are transported through UB
        self.smem_cap = self.ub_cap
        self.max_smem_usage = self.smem_cap

        # Register capacity, Ascend does not expose register file size in the same way
        self.reg_cap = 0

        # Transfer parameters
        self.transaction_size = [32, 32]  # 32 bytes alignment for DMA
        self.bandwidth = [900000, 900000]  # Example values

        # NPU specific parameters
        self.warp_size = 1  # Ascend executes one thread at a time (conceptually)
        self.sm_partition = 1

    @property
    def cube_dim(self) -> int:
        """
        Return the dimension of the CUBE-k size.
        """
        return self.cube_spec[-1]

    @property
    def cube_shape(self) -> list[int]:
        """
        Return the full dimensions of the CUBE unit [M, N, K].
        """
        return self.cube_spec

    @property
    def fractal_shape(self) -> tuple[int, int]:
        return (self.cube_spec[0], self.cube_spec[1])

    def get_avaliable_tensorintrin_shapes(self):
        self.available_cube_instructions = (CubeInstruction("Davich", [16, 16]),)
        return [t.shape for t in self.available_cube_instructions]


# TODO: consider the dtype of the input a and b seperately
# As the tensorcore may supports e4m3_float8 * e5m2
def is_cube_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if not isinstance(arch, Ascend):
        return False
    if arch.chip_name in {"Ascend910A", "Ascend910B", "Ascend310P", "Ascend950"}:
        # Ascend NPU supports float16 and bfloat16 tensor core operations
        return in_dtype in ["float16", "bfloat16"] and accum_dtype in [
            "float16",
            "bfloat16",
            "float32",
        ]
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


__all__ = ["Ascend", "is_ascend_arch", "is_cube_supported_precision"]
