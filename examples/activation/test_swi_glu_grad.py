"""Pytest wrapper for activation/swi_glu_grad.py.

原 Example 实现 SwiGLU backward。原文件无 argparse、无 if __name__，
自带 7 组 test_configs 顶层循环执行。用 subprocess 直接运行原脚本。

原 Example 关键参数（保持不变）：
  - 7 组 test_configs（原文件 L149-157）
  - dtype=bfloat16/float
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ascend_grad = torch.ops.npu.npu_swiglu_backward(dy, a)
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "swi_glu_grad.py")


def test_swi_glu_grad_precision():
    """运行 swi_glu_grad.py，验证 SwiGLU backward 精度。

    原脚本自带 7 组 test_configs，直接运行即可覆盖。
    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    """
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=EXAMPLE_DIR,
    )
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
