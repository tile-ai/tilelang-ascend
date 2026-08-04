"""Pytest wrapper for activation/gelu_grad.py.

原 Example 实现 GELU 梯度计算。原文件无 argparse、无 if __name__，
自带 6 组 test_configs 顶层循环执行。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - 6 组 test_configs（原文件 L79-86）
  - dtype=float (float32)
  - seed=0, rtol=1e-3, atol=1e-2
  - Golden: ref_grad_input = torch_npu.npu_gelu_backward(dy, x, approximate="none")
"""

import os
import subprocess
import sys


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "gelu_grad.py")


def test_gelu_grad_precision():
    """运行 gelu_grad.py，验证 GELU 梯度精度。

    原脚本自带 6 组 test_configs，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_DIR,
    )
    assert result.returncode == 0, f"脚本执行失败 (exit={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Kernel Output Match!" in result.stdout, f"精度校验未通过\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
