"""Pytest wrapper for example_gemm.py.

原 Example 实现算子语义 C = A @ B，使用 Expert 模式（手动内存层级
+手动Scope）。原文件含 argparse / 顶层 NPU 初始化 / 顶层编译等副作用，
无法安全 import，故按阶段一规范用 subprocess.run 包装。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, K_L1=64
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_c = a @ b 

测试矩阵说明：
  所有 M/N/K 取值均为 block 大小的整数倍（block_M=128, block_N=256,
  block_K=64），因为原 Kernel 用 M // block_M 做整数切分，非整除会截断
  导致结果错误——这是原 Example 的固有限制，不是 test 引入的。
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_gemm.py")

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
        (2048,1024, 512),
        (1024, 2048, 1024),
        (1024,1024,2048),
    ],
    ids=[
        "default_1024x1024x1024",
        "small_512x512x512",
        "rect_2048x1024x512",
        "rect_1024x2048x1024",
        "rect_1024x1024x2048",
    ],
)
def test_example_gemm_precision(m: int, n: int, k: int):
    """运行 example_gemm.py，验证 C = A @ B精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = _run_example(m, n, k)
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
