"""Pytest wrapper for example_gemm.py.

原 Example 实现算子语义 C = A @ B，使用 Expert 模式（手动内存层级
+手动Scope）。原文件含 argparse / 顶层 NPU 初始化 / 顶层编译等副作用，
通过 importlib + sys.argv 隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b
"""

import importlib.util
import sys
from pathlib import Path

import pytest

BLOCK_M = 128
BLOCK_N = 256
K_L1 = 64


def _run_example(m: int, n: int, k: int) -> None:
    source = Path(__file__).with_name("example_gemm.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source), "--m", str(m), "--n", str(n), "--k", str(k)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1024, 1024, 1024),
        (512, 512, 512),
        (2048, 1024, 512),
        (1024, 2048, 1024),
        (1024, 1024, 2048),
    ],
    ids=[
        "default_1024x1024x1024",
        "small_512x512x512",
        "rect_2048x1024x512",
        "rect_1024x2048x1024",
        "rect_1024x1024x2048",
    ],
)
def test_example_gemm_precision(m: int, n: int, k: int):
    """运行 example_gemm.py，验证 C = A @ B 精度。"""
    _run_example(m, n, k)
