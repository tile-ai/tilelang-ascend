from __future__ import annotations

from tvm.target import Target
from .arch_base import TileDevice
from .ascend_specs import A3_SKUS, a3_resources, a3_sku
import logging

# check if torch_npu is available, if not, degrade using CHIP_SPECS only
try:
    import torch
    import torch_npu  # noqa: F401 - Registers torch.npu.

    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_ascend_arch(arch: TileDevice) -> bool:
    return isinstance(arch, Ascend)


# Chip specifications (sizes in bytes)
CHIP_SPECS = {
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
    def __init__(self, target: Target | str = "llvm -keys=ascend", chip_name: str | None = None, platform_config: str | None = None):
        """Describe Carver budgets; A3 requires its own selected SDK INI.

        Cross compilation can name an exact chip and platform_config path.
        SDK raw capacities remain in raw_memory_capacities; *_cap fields
        respect the smaller arenas emitted by the current Ascend backend.
        """
        if isinstance(target, str):
            target = Target(target)
        self.target = target
        self.platform = "ascend"

        # detect npu properties if chip_name is not provided
        detected_cores = None
        detected_vector_cores = None
        L2_cache_size_bytes = None
        if chip_name is None and _TORCH_NPU_AVAILABLE:
            try:
                if torch.npu.is_available() and torch.npu.device_count() > 0:
                    # Get the current NPU device properties
                    props = torch.npu.get_device_properties(torch.npu.current_device())
                    npu_name = props.name.upper()  # e.g., "ASCEND910B"

                    if hasattr(props, "cube_core_num"):
                        detected_cores = props.cube_core_num
                    if hasattr(props, "l2_cache_size"):
                        L2_cache_size_bytes = props.l2_cache_size
                    elif hasattr(props, "L2_cache_size"):
                        L2_cache_size_bytes = props.L2_cache_size
                    if hasattr(props, "vector_core_num"):
                        detected_vector_cores = props.vector_core_num

                    if a3_sku(npu_name):
                        chip_name = a3_sku(npu_name)
                    elif "910B" in npu_name:
                        chip_name = "Ascend910B"
                    elif "310P" in npu_name:
                        chip_name = "Ascend310P"
                    elif "910" in npu_name:
                        chip_name = "Ascend910A"

                    logger.debug(f"Detected Ascend NPU: {npu_name}, using chip profile: {chip_name}")
            except Exception as e:
                logger.warning(f"Failed to detect Ascend NPU properties from torch_npu: {e}")

        # Else determine chip properties from CHIP_SPEC if not chip name provided
        if chip_name is None:
            mcpu = getattr(target, "mcpu", None)
            if mcpu:
                mcpu = mcpu.lower()
                if a3_sku(mcpu):
                    chip_name = a3_sku(mcpu)
                elif "910b" in mcpu:
                    chip_name = "Ascend910B"
                elif "310p" in mcpu:
                    chip_name = "Ascend310P"
                elif "910" in mcpu:
                    chip_name = "Ascend910A"
                else:
                    chip_name = DEFAULT_CHIP
            else:
                chip_name = DEFAULT_CHIP

        chip_name = a3_sku(chip_name) or chip_name
        self.chip_name = chip_name
        if chip_name in A3_SKUS:
            self.sdk_resources = a3_resources(chip_name, platform_config)
            self.raw_memory_capacities = dict(self.sdk_resources["raw_memory_capacities"])
            spec = {
                **self.sdk_resources["usable_memory_capacities"],
                "cores": self.sdk_resources["cores"],
                "L2": self.sdk_resources["L2"],
                "cube": self.sdk_resources["cube"],
            }
            self.compute_max_vector_core = (
                detected_vector_cores
                if detected_vector_cores is not None and detected_vector_cores > 0
                else self.sdk_resources["vector_cores"]
            )
        else:
            spec = CHIP_SPECS.get(chip_name, CHIP_SPECS[DEFAULT_CHIP]).copy()

        # AI Core size
        if detected_cores is not None and detected_cores > 0:
            self.compute_max_core = detected_cores
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


# TODO: consider the dtype of the input a and b separately
# As the tensorcore may supports e4m3_float8 * e5m2
def is_cube_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if not isinstance(arch, Ascend):
        return False
    if arch.chip_name in A3_SKUS:
        input_types = ["float16"]
        accumulator_types = ["float16", "float32"]
        if arch.sdk_resources["supports_bf16"]:
            input_types.append("bfloat16")
            accumulator_types.append("bfloat16")
        return in_dtype in input_types and accum_dtype in accumulator_types
    if arch.chip_name in CHIP_SPECS:
        # Ascend NPU supports float16 and bfloat16 tensor core operations
        return in_dtype in ["float16", "bfloat16"] and accum_dtype in ["float16", "bfloat16", "float32"]
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


__all__ = ["Ascend", "is_ascend_arch", "is_cube_supported_precision"]
