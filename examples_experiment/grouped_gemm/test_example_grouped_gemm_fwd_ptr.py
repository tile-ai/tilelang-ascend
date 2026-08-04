"""Pytest wrapper for grouped_gemm/example_grouped_gemm_fwd_ptr.py.

原 Example 实现 Grouped GEMM forward（ptr 版，padding 支持）。原文件有
if __name__ 保护，有 test_grouped_gemm_fwd_ptr() 内置 4 组 case。
用 subprocess 传参数运行。

原 Example 关键参数（保持不变）：
  - block_M=64, block_N=128, block_K=64
  - dtype=float16, accum_dtype=float32
  - 默认 K=4096, N=4096
  - Golden: torch_grouped_gemm
  - 容差: atol=1e-3, rtol=1e-3
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_grouped_gemm_fwd_ptr.py")


def _run_example(batch_sizes, k, n, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT, "--batch_sizes", batch_sizes, "--K", str(k), "--N", str(n)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLE_DIR,
    )


@pytest.mark.parametrize(
    ("batch_sizes", "k", "n"),
    [
        ("16,33,96", 128, 96),
        ("16,64,128", 128, 96),
        ("29,57,101", 128, 96),
        ("100,200,300", 128, 96),
        ("64,128,256", 4096, 4096),
    ],
    ids=[
        "tail_16_33_96",
        "baseline_16_64_128",
        "tail_29_57_101",
        "multi_100_200_300",
        "large_64_128_256",
    ],
)
def test_grouped_gemm_fwd_ptr_precision(batch_sizes, k, n):
    """运行 example_grouped_gemm_fwd_ptr.py，验证 Grouped GEMM forward (ptr) 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(batch_sizes, k, n)
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
