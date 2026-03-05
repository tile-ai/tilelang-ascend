import contextlib
import os
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import tilelang


TorchDTypeLike = Union[str, torch.dtype]

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

DEFAULT_TOLERANCE = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (2e-2, 2e-2),
    "float32": (1e-4, 1e-4),
    "int8": (0.0, 0.0),
    "int16": (0.0, 0.0),
    "int32": (0.0, 0.0),
    "int64": (0.0, 0.0),
    "bool": (0.0, 0.0),
}


def resolve_dtype(dtype: TorchDTypeLike) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_MAP[dtype]


def dtype_name(dtype: TorchDTypeLike) -> str:
    if isinstance(dtype, str):
        return dtype
    for name, torch_dtype in DTYPE_MAP.items():
        if torch_dtype == dtype:
            return name
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def set_npu_device(device_id: int) -> None:
    torch.npu.set_device(device_id)


def clear_tilelang_cache() -> None:
    tilelang.cache.clear_cache()


def gen_tensor(
    shape: Union[Sequence[int], torch.Size],
    dtype: TorchDTypeLike,
    *,
    kind: str = "randn",
    clear: bool = False,
    low: Optional[float] = None,
    high: Optional[float] = None,
    nonzero: bool = False,
    device: str = "npu",
) -> torch.Tensor:
    torch_dtype = resolve_dtype(dtype)

    if clear or kind == "zeros":
        out = torch.zeros(shape, dtype=torch_dtype)
    elif kind == "ones":
        out = torch.ones(shape, dtype=torch_dtype)
    elif kind == "randn":
        out = torch.randn(shape, dtype=torch_dtype)
    elif kind == "rand":
        out = torch.rand(shape, dtype=torch_dtype)
    elif kind == "randint":
        if torch_dtype == torch.bool:
            out = torch.randint(low=0, high=2, size=tuple(shape)).bool()
        else:
            int_low = 0 if low is None else int(low)
            int_high = 10 if high is None else int(high)
            out = torch.randint(low=int_low, high=int_high, size=tuple(shape), dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported tensor kind: {kind}")

    if low is not None and high is not None and kind in ("rand", "randn"):
        out = out * (high - low) + low

    if nonzero:
        out = torch.where(out == 0, torch.ones_like(out), out)

    return out.to(device=device)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: Optional[TorchDTypeLike] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
) -> None:
    if dtype is None:
        dtype = actual.dtype
    name = dtype_name(dtype)
    default_rtol, default_atol = DEFAULT_TOLERANCE[name]
    torch.testing.assert_close(
        actual,
        expected,
        rtol=default_rtol if rtol is None else rtol,
        atol=default_atol if atol is None else atol,
        equal_nan=equal_nan,
    )


def build_dtype_param_combos(
    *dtype_lists: Sequence[str],
    names: Optional[Sequence[str]] = None,
    accum_dtypes: Optional[Sequence[str]] = None,
):
    import pytest

    if accum_dtypes is not None:
        dtype_lists = (*dtype_lists, accum_dtypes)

    if not dtype_lists:
        raise ValueError("build_dtype_param_combos requires at least one dtype list.")

    for i, dtype_list in enumerate(dtype_lists):
        if len(dtype_list) == 0:
            raise ValueError(f"dtype list at index {i} must not be empty.")

    if names is None:
        default_names = ["in", "out", "acc"]
        names = [default_names[i] if i < len(default_names) else f"arg{i}" for i in range(len(dtype_lists))]
    else:
        names = list(names)
        if len(names) != len(dtype_lists):
            raise ValueError("names length must match the number of dtype lists.")

    combos = []
    for item in product(*dtype_lists):
        seen_dtypes = set()
        marks = []
        for dtype in item:
            if dtype not in seen_dtypes:
                marks.append(pytest.mark.dtype(dtype))
                seen_dtypes.add(dtype)

        param_id = "_".join(f"{name}_{dtype}" for name, dtype in zip(names, item))
        combos.append(pytest.param(*item, marks=marks, id=param_id))

    return combos


@contextlib.contextmanager
def ascend_mode(mode: str):
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev
