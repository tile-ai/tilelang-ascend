from dataclasses import dataclass
from typing import List, Callable, Any
import numpy as np


class Config:
    kwargs: list = None

    def __init__(self, kwargs, value):
        self.kwargs = kwargs
        self.value = value


@dataclass
class PolyConstraints:
    polynomial: List[float]

    other_value: List[float]

    def calc_results(self, tile):
        value = 1.0

        if len(tile) != len(self.polynomial):
            raise ValueError("tile len should be same as polynomial constraints!")

        for x, y in zip(tile, self.polynomial, strict=True):
            value *= pow(x * 16, y)

        for x in self.other_value:
            value *= x

        return value


class FuncConstraints:
    using_tile: List[int]

    other_value: List[float]

    func: Callable[..., Any] = None

    def __init__(self, using_tile, other_value, func, **kwargs):
        self.using_tile = using_tile
        self.other_value = other_value
        self.func = func
        self.kwargs = kwargs

    def calc_results(self, tile):
        if not callable(self.func):
            raise TypeError("func must be callable")

        if len(tile) < max(self.using_tile):
            raise ValueError("using_tile position should be less then tile len!")

        m = len(self.using_tile)
        idx = [0] * m
        for i in range(m):
            if self.using_tile[i] > 0:
                idx[i] = tile[self.using_tile[i] - 1] * 16

        value = self.func(idx, **self.kwargs)

        for x in self.other_value:
            value *= x

        return value


@dataclass
class PolyConstraintsSet:
    poly_add: List[PolyConstraints]
    poly_mul: List[PolyConstraints]
    func_add: List[FuncConstraints]
    func_mul: List[FuncConstraints]

    def calc_results(self, tile):
        value = 0.0
        for poly in self.poly_add:
            value += poly.calc_results(tile)
        for func in self.func_add:
            value += func.calc_results(tile)
        for poly in self.poly_mul:
            value *= poly.calc_results(tile)
        for func in self.func_mul:
            value += func.calc_results(tile)

        return value


@dataclass
class CustomTemplate:
    # Operation-related configuration parameters

    init_low_tile: List[int]
    init_tile: List[int]
    only_factor: List[bool]
    mem_caps: List[PolyConstraintsSet]
    batch_size: int = 1
    is_causal: bool = False


byte_per_numel = {
    "float32": 4,  # torch.float32 or torch.float
    "float64": 8,  # torch.float64 or torch.double
    "float16": 2,  # torch.float16 or torch.half
    "bfloat16": 2,  # torch.bfloat16
    "int32": 4,  # torch.int32 or torch.int
    "int64": 8,  # torch.int64 or torch.long
    "int16": 2,  # torch.int16 or torch.short
    "int8": 1,  # torch.int8
    "uint8": 1,  # torch.uint8
    "bool": 1,  # torch.bool
    "complex32": 4,  # torch.complex32 (not yet available in PyTorch as of the latest stable release)
    "complex64": 8,  # torch.complex64
    "complex128": 16,  # torch.complex128
}


def get_byte_per_numel(dtype: str = "None") -> int:
    return -1 if dtype == "None" else byte_per_numel[dtype]


def get_all_factors(n: int) -> List[int]:
    # Calculate the square root of n and round it up to the nearest integer
    n0 = int(np.ceil(np.sqrt(n)))

    # Find all divisors of n that are less than n0
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1

    # If n is a perfect square, add the square root to the list of factors
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]

    # Combine the factors and their corresponding larger pair factors
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]
