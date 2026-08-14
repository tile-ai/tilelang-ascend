"""Shared config for cann_bench Sigmoid package."""

__version__ = "1.0.0"

import torch
import tilelang
from tilelang import language as T


# AUTO_CV_COMBINE intentionally OFF: sigmoid is pure Vector (element-wise),
# the pass was adding an idle AIC core. AUTO_SYNC + MEMORY_PLANNING kept on
# (Developer mode auto-sync + UB memory planning).
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


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
