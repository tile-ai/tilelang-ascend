"""Pytest wrapper for example_quant_matmul.py.

原 Example 实现量化矩阵乘法：C = (A @ B) * scale
- A/B 为 int8 矩阵，相乘后转浮点再乘缩放系数
- 支持 per-tensor (scale_size="1") 和 per-channel (scale_size="N") 两种缩放
- 支持 float16/bfloat16 输出

原文件已有 check_case() + main() 函数式结构，且有 if __name__ 保护，
可安全 import，故采用 import 模式直接调用 check_case()。

原 Example 关键参数（保持不变）：
  - block_M=128, block_N=256, block_K=64
  - in_dtype=int8, accum_dtype=int32, scale_dtype=float32
  - seed=0, rtol=1e-2, atol=1e-2
  - Golden: ref_program 中 C = A.to(int32) @ B.to(int32); (C.to(float32) * scale).to(out_dtype)
"""

import pytest
import torch
import tilelang as tl

from example_quant_matmul import check_case


@pytest.mark.parametrize(
    ("m", "n", "k", "scale_size", "out_dtype"),
    [
        (1024, 1024, 1024, "1", "bfloat16"),
        (1024, 1024, 1024, "N", "float16"),
        (512, 512, 512, "1", "float16"),
        (2048, 1024, 512, "N", "bfloat16"),
        (1024, 2048, 1024, "1", "bfloat16"),
    ],
    ids=[
        "baseline_1024_scale1_bf16",
        "baseline_1024_scaleN_fp16",
        "small_512_scale1_fp16",
        "rect_2048x1024x512_scaleN_bf16",
        "rect_1024x2048x1024_scale1_bf16",
    ],
)
def test_quant_matmul_precision(m, n, k, scale_size, out_dtype):
    """运行量化矩阵乘法，验证 C = (A @ B) * scale 精度。

    直接调用原文件的 check_case()，它内部会：
    1. 构造 int8 输入 A/B 和 float32 scale
    2. 编译并运行 Kernel
    3. 用 ref_program 计算 Golden
    4. torch.testing.assert_close(rtol=1e-2, atol=1e-2)
    """
    torch.manual_seed(0)
    tl.cache.clear_cache()
    check_case(
        m,
        n,
        k,
        scale_size=scale_size,
        block_M=128,
        block_N=256,
        block_K=64,
        out_dtype=out_dtype,
    )
