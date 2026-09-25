# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tilelang.engine.param import KernelParam


class BaseKernelAdapter(ABC):
    func: Callable | None = None

    def __init__(
        self,
        mod,
        params: list[KernelParam],
        result_idx: list[int],
        workspace_idx: list[int],
    ) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_auto_memory_idx(result_idx, "result_idx")
        self.workspace_idx = self._legalize_auto_memory_idx(workspace_idx, "workspace_idx")
        self._post_init()

    @staticmethod
    def _legalize_memory_idx(
        memory_idx: list[int] | int | None,
        num_params: int,
        memory_name: str = "auto_memory_idx",
    ) -> list[int]:
        if memory_idx is None:
            indices = []
        elif isinstance(memory_idx, int):
            indices = [memory_idx]
        elif isinstance(memory_idx, list):
            indices = list(memory_idx)
        else:
            raise ValueError(f"{memory_name} should be a list of integers")

        for i, idx in enumerate(indices):
            if idx >= num_params or idx < -num_params:
                raise ValueError(f"{memory_name} should be an integer between {-num_params} and {num_params - 1}")
            if idx < 0:
                indices[i] = num_params + idx
        return indices

    def _legalize_auto_memory_idx(
        self,
        memory_idx: list[int] | int | None = None,
        memory_name: str = "auto_memory_idx",
    ) -> list[int]:
        return self._legalize_memory_idx(memory_idx, len(self.params), memory_name)

    @abstractmethod
    def _convert_torch_func(self) -> Callable:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()

    def _post_init(self):
        self.func = self._convert_torch_func()
