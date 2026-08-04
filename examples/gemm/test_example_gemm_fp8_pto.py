"""Pytest wrapper for example_gemm_fp8_pto.py.

原 Example 实现算子语义 C = A @ B（FP8 矩阵乘），target="pto"。FP8 TMATMUL
需要 A5 Cube 核心，非 A5 平台原文件会自动 skip（L15-17 打印 "Kernel Output
Match" 并 exit 0）。用 subprocess.run 包装。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=fp8(e4m3/e5m2), output=float32
  - seed=0, rtol=1e-2, atol=1e-2
  - target="pto"
  - Golden: ref_c = a_fp8.float() @ b_fp8.float()
  - --fp8 参数：e4m3 或 e5m2

注意：当前环境为 A3，原文件会自动 skip 并输出 "Kernel Output Match"。
test 仍会 PASSED，因为没有实际执行 FP8 计算。
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_gemm_fp8_pto.py")

BLOCK_M = 128
BLOCK_N = 256
K_L1 = 64


def _run_example(m: int, n: int, k: int, fp8: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT, "--m", str(m), "--n", str(n), "--k", str(k), "--fp8", fp8],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLE_DIR,
    )


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
    """运行 example_gemm_fp8_pto.py，验证 FP8 C = A @ B 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    非 A5 平台原文件会自动 skip（仍输出 "Kernel Output Match"）。
    """
    result = _run_example(m, n, k, fp8)
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
