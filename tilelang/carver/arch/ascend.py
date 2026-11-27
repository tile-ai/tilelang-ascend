from typing import Optional, List
from tvm.target import Target
from .arch_base import TileDevice
import logging

# check if torch_npu is available, if not, degrade using CHIP_SPECS only
try:
    import torch
    import torch_npu

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

# Chip names that share a memory profile with another entry.  Atlas A3
# (Ascend910_93) and Ascend910C are DAV_2201 parts, i.e. the same core
# architecture as 910B, so they reuse its UB/L1/L0A/L0B/L0C sizes instead of
# duplicating the numbers.  Without this mapping the name falls through the
# "910" bucket below and silently selects the 910A profile (256 KB UB,
# 256 KB L0C), which overstates UB and understates L0C on A3.
CHIP_PROFILE_ALIASES = {
    "Ascend910C": "Ascend910B",
}

# Device-name fragments, longest/most specific first: the generic "910"
# bucket must never shadow 910B / 910C / 910_93. A ``None`` target marks a
# family that is recognised but has no carver profile yet: those names stay
# unresolved instead of being silently claimed by the "910" bucket below.
_CHIP_NAME_PATTERNS = (
    ("910_95", None),  # A5 family (910_95xx / 950xx): no profile yet
    ("950", None),
    ("910_93", "Ascend910C"),
    ("910c", "Ascend910C"),
    ("910b", "Ascend910B"),
    ("310p", "Ascend310P"),
    ("910a", "Ascend910A"),
    ("910", "Ascend910A"),
)


# Profiles that expose the Ascend cube (tensor core) at all.
CUBE_PRECISION_PROFILES = ("Ascend910A", "Ascend910B", "Ascend310P")


def resolve_chip_name(device_name: str) -> Optional[str]:
    """Resolve a device name or target ``mcpu`` string to a chip profile.

    Returns ``None`` for names outside the known families so callers can
    decide between a loud fallback and an error.
    """
    name = (device_name or "").lower()
    for fragment, chip_name in _CHIP_NAME_PATTERNS:
        if fragment in name:
            return chip_name
    return None


class CubeInstruction(object):
    def __init__(self, name: str, shape: List[int]):
        self.name = name
        self.shape = shape

class Ascend(TileDevice):
    def __init__(self, target: Target | str = "llvm -keys=ascend", chip_name: Optional[str] = None):
        if isinstance(target, str):
            target = Target(target)
        self.target = target
        self.platform = "ascend"

        # detect npu properties if chip_name is not provided
        detected_cores = None
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

                    chip_name = resolve_chip_name(npu_name)
                    if chip_name is None:
                        logger.warning(
                            f"Unrecognized Ascend device '{npu_name}'; "
                            "falling back to the target mcpu / default profile."
                        )

                    logger.debug(
                        f"Detected Ascend NPU: {npu_name}, using chip profile: {chip_name}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to detect Ascend NPU properties from torch_npu: {e}"
                )

        # Else determine chip properties from CHIP_SPEC if not chip name provided
        if chip_name is None:
            mcpu = getattr(target, "mcpu", None)
            chip_name = resolve_chip_name(mcpu) if mcpu else None
            if chip_name is None:
                chip_name = DEFAULT_CHIP

        # Canonicalise names passed in by the caller (``chip_name="Ascend910_93"``,
        # lowercase spellings, ...) so the alias table and the capability checks
        # see the same spelling as detected devices. Unknown names are kept
        # verbatim so a caller-supplied model still fails loudly later.
        chip_name = resolve_chip_name(chip_name) or chip_name

        self.chip_name = chip_name
        # Alias-resolved profile (e.g. Ascend910C -> Ascend910B). Profile
        # capability checks must use this instead of enumerating raw chip
        # names, otherwise every new alias has to be added to each list.
        self.chip_profile = CHIP_PROFILE_ALIASES.get(chip_name, chip_name)
        spec = CHIP_SPECS.get(self.chip_profile, CHIP_SPECS[DEFAULT_CHIP]).copy()

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
    def cube_shape(self) -> List[int]:
        """
        Return the full dimensions of the CUBE unit [M, N, K].
        """
        return self.cube_spec

    @property
    def fractal_shape(self) -> tuple[int, int]:
        return (self.cube_spec[0], self.cube_spec[1])

    def get_avaliable_tensorintrin_shapes(self):
        self.available_cube_instructions = (
            CubeInstruction("Davich", [16, 16]),
        )
        return [t.shape for t in self.available_cube_instructions]

# TODO: consider the dtype of the input a and b seperately
# As the tensorcore may supports e4m3_float8 * e5m2
def is_cube_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if not isinstance(arch, Ascend):
        return False
    # Compare the alias-resolved profile, not the raw chip name: Ascend910C
    # (Atlas A3) shares the 16x16x16 cube and the memory profile of 910B (see
    # CHIP_PROFILE_ALIASES), so it must take the same branch instead of
    # raising and silently degrading cube hints to the default policy.
    profile = arch.chip_profile
    if profile not in CUBE_PRECISION_PROFILES:
        raise ValueError(f"Unsupported architecture: chip={arch.chip_name}, profile={profile}")
    # Ascend NPU supports float16 and bfloat16 tensor core operations
    return in_dtype in ["float16", "bfloat16"] and accum_dtype in [
        "float16",
        "bfloat16",
        "float32",
    ]


__all__ = ["Ascend", "is_ascend_arch", "is_cube_supported_precision"]
