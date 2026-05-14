"""Ascend NPU adapted quant kernels for TileKernels.

All GPU quant kernels adapted to run on Ascend A3/A5 NPU
via tilelang-ascend with NPU-specific pass configs.
"""

from .cast_back_kernel import cast_back, per_token_cast_back
from .cast_back_e5m6_kernel import cast_back_e5m6