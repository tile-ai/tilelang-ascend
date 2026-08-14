"""Shared config for cann_bench Mish package.

Developer mode: AUTO_SYNC ON (compiler auto-sync), MEMORY_PLANNING ON.
AUTO_CV_COMBINE OFF — mish is pure Vector 12-step element-wise; enabling it
would spawn an idle AIC core paying launch + buffer init cost (Stage 3 finding,
see custom/mish/perf_tuning/perf_report.md).
"""

__version__ = "1.0.0"

import torch
import tilelang


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Intermediate compute dtype: float32 for precision + bf16 CANN intrinsic gap.
ACC_DTYPE = "float32"

# Intra-core vector parallelism (vid dimension). block_M must be a multiple.
VEC_NUM = 2

# Cast modes for T.tile.cast at GM<->UB boundary.
CAST_MODE_LOW2HIGH = "CAST_NONE"  # fp16/bf16 -> fp32 (lossless)
CAST_MODE_HIGH2LOW = "CAST_RINT"  # fp32 -> fp16/bf16 (round to nearest)

# UB budget (Ascend A2/A3 UB = 196352 B). Kernel allocates 5 fp32 compute
# buffers + 1 orig-dtype cast-bridge buffer. See mish.py docstring for detail.
UB_BUDGET = 196352

# Bytes per element (worst-case, all buffers live):
#   cast path (fp16/bf16): 5 fp32 (20B) + 1 orig (2B) = 22B
#   direct path (fp32):    5 fp32 (20B), tmp_orig dead -> MEMORY_PLANNING reuses
BYTES_PER_ELEM = {
    "float16": 22,
    "bfloat16": 22,
    "float32": 20,
}


def torch_dtype_to_tl(dtype):
    """Map torch dtype -> tilelang dtype string."""
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
