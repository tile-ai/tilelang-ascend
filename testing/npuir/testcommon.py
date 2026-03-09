import contextlib
import os
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import tilelang
from tilelang.engine import lower


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
    """设置 NPU 设备；无 torch_npu 时静默跳过（无卡环境）。"""
    try:
        torch.npu.set_device(device_id)
    except (AttributeError, RuntimeError):
        pass


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


def codegen_lower(func, mode: str = "Developer"):
    """
    无卡环境下对 PrimFunc 执行 lower，返回 NPU IR (MLIR) 字符串。
    不依赖 torch_npu，不需要 NPU 硬件。
    可通过 TILELANG_DUMP_IR=TRUE 查看中间 TVM IR 和 MLIR。
    """
    with ascend_mode(mode):
        result = lower(func, target="npuir")
    assert isinstance(result, str), "lower(..., target='npuir') 应返回 str"
    assert len(result) > 0, "NPU IR 输出不应为空"
    return result


def compile_to_kernel_o_if_available(mlir_content: str) -> Optional[bytes]:
    """
    如果 bishengir-compile 可用，将 NPU IR 编译为 kernel.o 的二进制内容；
    否则返回 None。用于无卡环境下的可选编译验证。
    """
    npu_compiler = _find_bishengir_compile()
    if npu_compiler is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        npuir_path = Path(tmpdir) / "kernel.npuir"
        npuir_path.write_text(mlir_content)
        out_path = Path(tmpdir) / "kernel"
        cmd = [
            npu_compiler,
            str(npuir_path),
            "--enable-auto-multi-buffer=true",
            "--enable-triton-kernel-compile=true",
            "--enable-hivm-compile=true",
            "-o",
            str(out_path),
        ]
        mode = os.environ.get("TILELANG_ASCEND_MODE")
        if mode is None or str(mode).lower().strip() in ("expert", "exp", "e"):
            cmd.insert(-2, "--disable-hivm-tensor-compile=true")
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True, cwd=tmpdir)
            for name in ("kernel.o", "kernel"):
                o_path = Path(tmpdir) / name
                if o_path.exists():
                    return o_path.read_bytes()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    return None


def _find_bishengir_compile() -> Optional[str]:
    """返回 bishengir-compile 可执行路径，若未找到则返回 None。"""
    import shutil

    path = shutil.which("bishengir-compile")
    if path:
        return path
    root = os.getenv("TRITON_NPU_COMPILER_PATH", "")
    if root:
        p = os.path.join(root, "npuc")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None
