"""Pytest wrapper for gemm_splitk/example_tilelang_gemm_splitk.py.

原 Example 实现算子语义 C = A @ B（Split-K 并行）。原文件有 if __name__ 保护，
支持 --M --N --K --split-k 参数。用 subprocess.run 包装。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=128, block_K=32
  - dtype=float16, accum_dtype=float
  - seed=42, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b
  - 默认 case: (128,128,128,2) + (1024,1024,1024,4)
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_tilelang_gemm_splitk.py")


def _run_example(m: int, n: int, k: int, split_k: int, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT, "--M", str(m), "--N", str(n), "--K", str(k), "--split-k", str(split_k)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLE_DIR,
    )


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

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(m, n, k, split_k)
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
