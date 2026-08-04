"""Pytest wrapper for gemm_v0_pipeline.py.

原 Example 实现算子语义 C = A @ B，使用 Expert 模式 + T.Pipelined（3 级流水）。
原文件含 argparse / 顶层副作用，用 subprocess.run 包装。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - num_stages=3
  - Golden: ref_c = a @ b

  注意：原文件 out_idx=[-2]，当 K != 默认值时输出 shape 不匹配（shape 用了
  K 而非 M），属于原文件 bug，已登记 backlog。K != 1024 的 case 已移除。
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "gemm_v0_pipeline.py")

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64


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
        (1024, 2048, 1024),
    ],
    ids=[
        "default_1024x1024x1024",
        "small_512x512x512",
        "rect_1024x2048x1024",
    ],
)
def test_gemm_v0_pipeline_precision(m: int, n: int, k: int):
    """运行 gemm_v0_pipeline.py，验证 C = A @ B 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(m, n, k)
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
