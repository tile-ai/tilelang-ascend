"""Pytest wrapper for example_gemm_fp8_pto.py.

原 Example 实现算子语义 C = A @ B（FP8 矩阵乘），target="pto"。FP8 TMATMUL
需要 A5 Cube 核心，非 A5 平台原文件会自动 skip。通过 importlib + sys.argv
隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=fp8(e4m3/e5m2), output=float32
  - seed=0, rtol=1e-2, atol=1e-2
  - target="pto"
  - Golden: ref_c = a_fp8.float() @ b_fp8.float()
  - --fp8 参数：e4m3 或 e5m2
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int, fp8: str) -> None:
    source = Path(__file__).with_name("example_gemm_fp8_pto.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_fp8_pto_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source), "--m", str(m), "--n", str(n), "--k", str(k), "--fp8", fp8]
        spec.loader.exec_module(module)
    except SystemExit as e:
        if e.code != 0:
            raise
    finally:
        sys.argv = original_argv


@pytest.mark.parametrize(
    ("m", "n", "k", "fp8"),
    [
        (1024, 1024, 1024, "e4m3"),
        (1024, 1024, 1024, "e5m2"),
        (512, 512, 512, "e4m3"),
        (2048, 1024, 512, "e5m2"),
    ],
    ids=[
        "default_1024_e4m3",
        "default_1024_e5m2",
        "small_512_e4m3",
        "rect_2048x1024x512_e5m2",
    ],
)
def test_example_gemm_fp8_pto_precision(m: int, n: int, k: int, fp8: str):
    """运行 example_gemm_fp8_pto.py，验证 FP8 C = A @ B 精度。"""
    _run_example(m, n, k, fp8)
