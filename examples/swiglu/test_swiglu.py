"""Test for SwiGLU v3 kernel.

Precision: compares against torch golden (silu(x0) * x1, upcast to fp32).
cann-bench evaluation (pure NPU kernel time, 20 cases): avg speedup 0.7751x
vs torch_npu npu_swiglu baseline. See PR description for the full report.
"""

import torch
import torch.nn.functional as F

from swiglu import swi_glu


def golden(input, dim=-1):
    """Torch golden: output = silu(x0) * x1, upcast fp16/bf16 to fp32."""
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)


torch.manual_seed(0)

# Tests: (shape, dim, dtype)
test_configs = [
    ((1024, 2048), -1, torch.float16),
    ((2048, 4096), -1, torch.float32),
    ((4096, 8192), -1, torch.bfloat16),
    ((1024, 2048), 0, torch.float16),
    ((1009, 2016), -1, torch.float16),
    ((256, 256), -1, torch.float16),
    ((256, 258), -1, torch.bfloat16),
    ((4, 8, 256), -1, torch.float16),
    ((320, 256), 0, torch.float32),
]

for shape, dim, dtype in test_configs:
    print(f"Testing SwiGLU shape={shape}, dim={dim}, dtype={dtype}")
    a = torch.randn(shape, dtype=dtype)
    b = swi_glu(a, dim=dim)
    ref = golden(a, dim=dim)
    torch.testing.assert_close(b.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
    print("  Test passed!")

print("Kernel Output Match!")
