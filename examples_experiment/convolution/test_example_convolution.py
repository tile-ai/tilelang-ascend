import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


CONVOLUTION_CASES = [
    pytest.param("perfect_alignment", 2, 2, 15, 15, 128, 8, 8, 1, 0, id="perfect_alignment"),
    pytest.param("m_padding", 1, 2, 32, 32, 50, 3, 3, 1, 0, id="m_padding"),
    pytest.param("n_padding", 1, 4, 17, 17, 128, 3, 3, 1, 0, id="n_padding"),
    pytest.param("k_padding", 2, 3, 28, 28, 128, 3, 3, 2, 1, id="k_padding"),
    pytest.param("all_dim_padding", 1, 3, 17, 17, 64, 3, 3, 1, 0, id="all_dim_padding"),
    pytest.param("multi_block", 4, 8, 28, 28, 256, 5, 5, 1, 0, id="multi_block"),
]


def _load_convolution_example() -> ModuleType:
    source = Path(__file__).with_name("example_convolution.py")
    spec = importlib.util.spec_from_file_location("_convolution_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module


@pytest.mark.parametrize(
    ("name", "batch", "channels", "height", "width", "out_channels", "kernel_h", "kernel_w", "stride", "padding"),
    CONVOLUTION_CASES,
)
def test_convolution_accuracy(
    name: str,
    batch: int,
    channels: int,
    height: int,
    width: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    padding: int,
) -> None:
    example = _load_convolution_example()

    example.run_test(
        name,
        batch,
        channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
    )
