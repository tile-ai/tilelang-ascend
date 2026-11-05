"""Pytest wrapper for gemm_splitk/example_tilelang_gemm_splitk.py.

原 Example 实现算子语义 C = A @ B（Split-K 并行）。原文件有 if __name__ 保护，
支持 --M --N --K --split-k 参数。用 importlib + sys.argv 隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=128, block_K=32
  - dtype=float16, accum_dtype=float
  - seed=42, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b
  - 默认 case: (128,128,128,2) + (1024,1024,1024,4)
"""

import importlib.util
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int, split_k: int) -> None:
    source = Path(__file__).with_name("example_tilelang_gemm_splitk.py")
    spec = importlib.util.spec_from_file_location("_example_tilelang_gemm_splitk_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")


@pytest.mark.parametrize(
    ("m", "n", "k", "split_k"),
    [
        (128, 128, 128, 2),
        (1024, 1024, 1024, 4),
        (512, 512, 512, 2),
        (1024, 1024, 1024, 2),
        (2048, 1024, 512, 4),
    ],
    ids=[
        "small_128_split2",
        "default_1024_split4",
        "mid_512_split2",
        "rect_1024_split2",
        "rect_2048x1024x512_split4",
    ],
)
def test_tilelang_gemm_splitk_precision(m: int, n: int, k: int, split_k: int):
    """运行 example_tilelang_gemm_splitk.py，验证 Split-K GEMM 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example(m, n, k, split_k)
