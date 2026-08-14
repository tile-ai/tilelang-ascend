"""Shared config for cann_bench ForeachNorm package."""

__version__ = "1.0.0"

import torch
import tilelang


# Developer mode: AUTO_SYNC ON (compiler auto-sync), MEMORY_PLANNING ON,
# AUTO_CV_COMBINE ON (fuses cast + compute to cut UB round-trips for the
# upcast fp16/bf16 -> fp32 path used by every partial-reduction kernel).
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"

DEFAULT_BLOCK_N = 8192
CORE_NUM = 24  # Ascend910B3 physical AI Core count

SUPPORTED_DTYPES = {"float16", "float32", "bfloat16"}


def torch_dtype_to_tl(dtype):
    """Map torch dtype -> tilelang dtype string."""
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
