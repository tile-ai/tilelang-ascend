"""Pytest wrapper for grouped_gemm/example_grouped_gemm_bwd.py.

原 Example 实现 Grouped GEMM backward。原文件有 if __name__ 保护。
用 subprocess 传 --batch_sizes / --M / --N 等参数运行。

原 Example 关键参数（保持不变）：
  - block_M=64, block_N=64, block_K=64
  - dtype=float16, accum_dtype=float32
  - 默认 M=512, N=512
  - Golden: torch.mm(A_i.T, B_i) per group
  - 容差: rtol=1e-2, atol=1e-2 (torch.allclose)
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_grouped_gemm_bwd.py")


def _run_example(batch_sizes, m, n, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT, "--batch_sizes", batch_sizes, "--M", str(m), "--N", str(n)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLE_DIR,
    )


@pytest.mark.parametrize(
    ("batch_sizes", "m", "n"),
    [
        ("64,128", 512, 512),
        ("128,256", 512, 512),
        ("64,128,256", 256, 256),
        ("128", 512, 512),
    ],
    ids=[
        "baseline_64_128",
        "multi_128_256",
        "small_64_128_256",
        "single_128",
    ],
)
def test_grouped_gemm_bwd_precision(batch_sizes, m, n):
    """运行 example_grouped_gemm_bwd.py，验证 Grouped GEMM backward 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(batch_sizes, m, n)
    assert result.returncode == 0, (
        f"脚本执行失败 (exit={result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Kernel Output Match!" in result.stdout, (
        f"精度校验未通过\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
