"""NPU-adapted common utilities for quantization kernels.

Adapted from tile_kernels/quant/common.py for Ascend NPU.
Removes GPU-specific dependencies (nvcc, TMA, etc.) and provides
NPU-compatible helpers.
"""

import os

import torch
import tilelang
from tilelang import language as T

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# NPU适配: 替代 torch.cuda.get_device_properties 获取核心数
def _get_npu_num_sms(default=80):
    """Get NPU core count, with fallback to env var or default."""
    try:
        import torch_npu
        return torch_npu.npu.get_device_properties(0).multi_processor_count
    except Exception:
        return int(os.environ.get("ASCEND_NUM_SMS", default))


_npu_num_sms = 0


def set_npu_num_sms(num_sms: int) -> None:
    global _npu_num_sms
    _npu_num_sms = num_sms


def get_npu_num_sms() -> int:
    global _npu_num_sms
    if _npu_num_sms == 0:
        return _get_npu_num_sms()
    return _npu_num_sms


# NPU适配: Ascend pass configs
NPU_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def align_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


QuantTensor = tuple[torch.Tensor, torch.Tensor]
