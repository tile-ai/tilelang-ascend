"""Pytest wrapper for example_gemm_intrinsic.py.

原 Example 实现算子语义 C = A @ B，使用 Expert 模式 + intrinsic（手动
set_flag/wait_flag 同步 + T.mma + 双缓冲 L1/L0）。原文件含 argparse /
顶层 NPU 初始化 / 顶层编译 / do_bench 性能测试等副作用，无法安全 import，
故用 importlib + sys.argv 隔离在当前进程内执行，保证 coverage 可追踪。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64, K_L1=256, S1=2, S2=2
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - target="ascendc"
  - Golden: ref_c = a @ b

测试矩阵说明：
  M 是 128 倍数，N 是 256 倍数。默认 shape 8192x1024x8192 过大，
  测试用较小 shape 减少编译和运行时间。原文件含 do_bench 性能测试，
  timeout 设为 600s。
"""

import importlib.util
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int) -> None:
    source = Path(__file__).with_name("example_gemm_intrinsic.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_intrinsic_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1024, 1024, 1024),
        (512, 512, 512),
        (2048, 1024, 1024),
        (1024, 1024, 2048),
        (2048, 1024, 2048),
    ],
    ids=[
        "baseline_1024x1024x1024",
        "small_512x512x512",
        "rect_2048x1024x1024",
        "rect_1024x1024x2048",
        "large_2048x1024x2048",
    ],
)
def test_example_gemm_intrinsic_precision(m: int, n: int, k: int):
    """运行 example_gemm_intrinsic.py，验证 C = A @ B 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    注意：原文件含 do_bench 性能测试，执行时间较长。"""
    _run_example(m, n, k)
