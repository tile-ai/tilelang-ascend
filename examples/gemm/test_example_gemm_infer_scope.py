"""Pytest wrapper for example_gemm_infer_scope.py.

原 Example 实现算子语义 C = A @ B，使用 Developer 模式（alloc_shared /
alloc_fragment + pass_configs，编译器自动推断 L1/UB/L0）。原文件含
argparse / 顶层 NPU 初始化 / 顶层编译等副作用，用 subprocess.run 包装。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - pass_configs: AUTO_CV_COMBINE + AUTO_SYNC
  - Golden: ref_c = a @ b
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_gemm_infer_scope.py")

BLOCK_M = 128
BLOCK_N = 256
K_L1 = 64


def _run_example(m: int, n: int, k: int, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT, "--m", str(m), "--n", str(n), "--k", str(k)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLE_DIR,
    )


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

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(m, n, k)
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
