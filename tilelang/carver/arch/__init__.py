# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from .arch_base import TileDevice
from .cuda import CUDA
from .cpu import CPU
from .cdna import CDNA
from .ascend import Ascend, is_cube_supported_precision
from typing import Union
from tvm.target import Target
import torch


def get_arch(target: Union[str, Target] = "cuda") -> TileDevice:
    if isinstance(target, str):
        target = Target("llvm -keys=ascend") if target == "ascend" else Target(target)

    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        if "ascend" in target.keys:
            return Ascend(target)
        return CPU(target)
    elif target.kind.name == "hip":
        return CDNA(target)
    elif target.kind.name == "ascend":
        return Ascend(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")


def auto_infer_current_arch() -> TileDevice:
    # TODO(lei): This is a temporary solution to infer the current architecture
    # Can be replaced by a more sophisticated method in the future
    if torch.version.hip is not None:
        return get_arch("hip")
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return get_arch("ascend")
    if torch.cuda.is_available():
        return get_arch("cuda")
    else:
        return get_arch("llvm")


from .cpu import is_cpu_arch  # noqa: F401
from .cuda import (
    is_cuda_arch,  # noqa: F401
    is_volta_arch,  # noqa: F401
    is_ampere_arch,  # noqa: F401
    is_ada_arch,  # noqa: F401
    is_hopper_arch,  # noqa: F401
    is_tensorcore_supported_precision,  # noqa: F401
    has_mma_support,  # noqa: F401
)
from .cdna import is_cdna_arch  # noqa: F401
