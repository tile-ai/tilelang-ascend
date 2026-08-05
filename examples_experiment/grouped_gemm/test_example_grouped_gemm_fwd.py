"""Pytest wrapper for grouped_gemm/example_grouped_gemm_fwd.py.

原 Example 实现 Grouped GEMM forward（metadata 驱动）。原文件有 if __name__
保护，有 test_grouped_gemm() 内置 5 组 case。用 subprocess 传 --batch_sizes
等参数运行。

原 Example 关键参数（保持不变）：
  - block_M=64, block_N=128, block_K=64
  - dtype=float16, accum_dtype=float32
  - 默认 K=8192, M=8192
  - Golden: torch_gmm
  - 容差: rtol=0.01, atol=0.01 (torch.allclose)
"""

import importlib.util
from pathlib import Path

import pytest


def _run_example() -> None:
    source = Path(__file__).with_name("example_grouped_gemm_fwd.py")
    spec = importlib.util.spec_from_file_location("_example_grouped_gemm_fwd_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")


@pytest.mark.parametrize(
    ("batch_sizes", "k", "m"),
    [
        ("64", 8192, 8192),
        ("64,128,256", 8192, 8192),
        ("63", 8192, 8192),
        ("100,200,300,400", 8192, 8192),
        ("63,77,111,280", 4096, 4096),
    ],
    ids=[
        "single_64",
        "multi_64_128_256",
        "tail_63",
        "multi_100_200_300_400",
        "tail_multi_63_77_111_280_small",
    ],
)
def test_grouped_gemm_fwd_precision(batch_sizes, k, m):
    """运行 example_grouped_gemm_fwd.py，验证 Grouped GEMM forward 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。"""
    _run_example(batch_sizes, k, m)
