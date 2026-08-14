"""Shared config for cann_bench Mish package."""

__version__ = "1.0.0"

import torch
import tilelang


# Developer mode: AUTO_SYNC ON (compiler auto-sync), MEMORY_PLANNING ON.
# AUTO_CV_COMBINE OFF (Stage 3 [#1]: mish is pure Vector 12-step element-wise,
# the pass was adding an idle AIC core; closing it is a反模式修复 per
# performance-antipatterns.md, bench end-to-end unchanged but code quality improved).
#
# Expert double buffer (AUTO_SYNC OFF) was evaluated and rejected — mish's 12-step
# compute + 6 fp32 buffers (5 compute + 1 cast bridge) exceed UB budget under
# stages=2 double buffer. Fixed Core mode (launch min(block_num,24) + T.serial)
# was also rejected — large shapes (8192,8192) regressed +25-36% because T.serial
# loop overhead (171 tiles/core) outweighs launch-count reduction for heavy
# 12-step compute per tile (unlike sigmoid's 1-step where Fixed Core gave -25.5%).
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
