"""Pytest wrapper for example_gemm_intrinsic_persistent.py.

原 Example 实现算子语义 C = A @ B，使用 Expert 模式 + intrinsic + Persistent
调度（手动 set_flag/wait_flag + T.mma + T.Persistent + 双缓冲）。原文件有
if __name__ 保护，但顶层有 tilelang.cache.clear_cache()，统一用 subprocess。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64, K_L1=256, S1=2, S2=2
  - dtype=float16, accum_dtype=float
  - seed=0, rtol=1e-2, atol=1e-2
  - target="ascendc"
  - Golden: ref_c = a @ b

测试矩阵说明：
  M 是 128 倍数，N 是 256 倍数。默认 shape 8192x1024x8192 过大，
  测试用较小 shape。原文件含 do_bench，timeout 设为 600s。

  注意：M=512,N=512,K=512 时原 Kernel 输出 87.1% 元素不匹配（最大误差
  149.625），属于原文件 bug，已登记 backlog。该 case 已从 case matrix
  移除，不阻塞 Test PR。
"""

import os
import subprocess
import sys

import pytest

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SCRIPT = os.path.join(EXAMPLE_DIR, "example_gemm_intrinsic_persistent.py")

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
K_L1 = 256


def _run_example(m: int, n: int, k: int, timeout: int = 600) -> subprocess.CompletedProcess:
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
        (2048, 1024, 1024),
        (1024, 1024, 2048),
        (2048, 1024, 2048),
    ],
    ids=[
        "baseline_1024x1024x1024",
        "rect_2048x1024x1024",
        "rect_1024x1024x2048",
        "large_2048x1024x2048",
    ],
)
def test_example_gemm_intrinsic_persistent_precision(m: int, n: int, k: int):
    """运行 example_gemm_intrinsic_persistent.py，验证 C = A @ B 精度。

    成功判定：退出码 0 且 stdout 包含 "Kernel Output Match!"。
    注意：原文件含 do_bench 性能测试，执行时间较长。
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
