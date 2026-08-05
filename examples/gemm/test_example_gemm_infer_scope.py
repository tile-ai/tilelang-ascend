"""Pytest wrapper for example_gemm_infer_scope.py.

原 Example 实现算子语义 C = A @ B，使用 Developer 模式（alloc_shared /
alloc_fragment + pass_configs，编译器自动推断 L1/UB/L0）。原文件含
argparse / 顶层 NPU 初始化 / 顶层编译等副作用，用 importlib + sys.argv 隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - pass_configs: AUTO_CV_COMBINE + AUTO_SYNC
  - Golden: ref_c = a @ b
"""

import importlib.util
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int) -> None:
    source = Path(__file__).with_name("example_gemm_infer_scope.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_infer_scope_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")


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
def test_example_gemm_infer_scope_precision(m: int, n: int, k: int):
    """运行 example_gemm_infer_scope.py，验证 C = A @ B 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example(m, n, k)
