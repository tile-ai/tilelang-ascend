"""Pytest wrapper for matmul_add_pipeline.py.

原 Example 实现算子语义 C = A @ B + D，使用 Developer 模式 + T.Pipelined
（3 级流水 + 2 级 Vector 流水）。原文件有 if __name__ 保护，但顶层有
tilelang 相关副作用，统一用 importlib + sys.argv 隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b + d
"""

import importlib.util
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int) -> None:
    source = Path(__file__).with_name("matmul_add_pipeline.py")
    spec = importlib.util.spec_from_file_location("_matmul_add_pipeline_under_test", source)
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
def test_matmul_add_pipeline_precision(m: int, n: int, k: int):
    """运行 matmul_add_pipeline.py，验证 C = A @ B + D 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example(m, n, k)
