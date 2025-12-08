# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from abc import ABC, abstractmethod
from typing import Any, List, Callable, Optional, Union
from tilelang.engine.param import KernelParam


class BaseKernelAdapter(ABC):

    func: Optional[Callable] = None

    def __init__(self, mod, params: List[KernelParam], 
                 result_idx: List[int], workspace_idx: List[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_auto_memory_idx(result_idx, "result_idx")
        self.workspace_idx = self._legalize_auto_memory_idx(workspace_idx, "workspace_idx")
        self._post_init()

    def _legalize_auto_memory_idx(self, memory_idx: Union[List[int], int, None] = None, memory_name = "auto_memory_idx") -> List[int]:
        params = self.params
        if memory_idx is None:
            memory_idx = []

        elif isinstance(memory_idx, int):
            if memory_idx >= len(params) or memory_idx < -len(params):
                raise ValueError(
                    f"{memory_name} should be an integer between {-len(params)} and {len(params) - 1}") 
            if memory_idx < 0:
                memory_idx = len(params) + memory_idx
            memory_idx = [memory_idx]

        elif isinstance(memory_idx, list):
            for i, idx in enumerate(memory_idx):
                if idx >= len(params) or idx < -len(params):
                    raise ValueError(
                        f"{memory_name} should be an integer between {-len(params)} and {len(params) - 1}")
                if idx < 0:
                    memory_idx[i] = len(params) + idx

        else:
            raise ValueError(f"{memory_name} should be a list of integers")

        return memory_idx
    

    @abstractmethod
    def _convert_torch_func(self) -> callable:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()

    def _post_init(self):
        self.func = self._convert_torch_func()
