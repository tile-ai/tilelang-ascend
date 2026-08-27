"""SwiGLU test: golden reference + mixed-precision tolerance checks."""

import torch
import torch.nn.functional as F

from swiglu import swi_glu


def _golden(input, dim=-1):
    """Torch golden: output = silu(x0) * x1, upcast fp16/bf16 to fp32."""
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)


# Mixed tolerance per dtype: |actual - golden| <= atol + rtol * |golden|
# Required matched_ratio >= 0.99
_TOL_MAP = {
    torch.float16: (6.10e-5, 1.95e-3),
    torch.bfloat16: (9.77e-4, 1.56e-2),
    torch.float32: (1.53e-5, 9.77e-4),
}


def _check_precision(actual, golden, dtype):
    """Mixed tolerance check: per-dtype atol/rtol + matched_ratio >= 0.99."""
    atol, rtol = _TOL_MAP.get(dtype, (1e-2, 1e-2))
    diff = (actual - golden).abs()
    threshold = atol + rtol * golden.abs()
    matched_ratio = (diff <= threshold).float().mean().item()
    max_diff = diff.max().item()
    return matched_ratio >= 0.99, matched_ratio, max_diff


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
    a = torch.randn(shape, dtype=dtype, device="cpu").npu()
    b = swi_glu(a, dim=dim)
    ref = _golden(a, dim=dim)
    passed, ratio, max_diff = _check_precision(b.cpu(), ref.cpu(), dtype)
    assert passed, (
        f"precision fail: shape={shape} dtype={dtype} "
        f"matched_ratio={ratio:.4f} max_diff={max_diff:.3e}"
    )
    print(f"  PASS  matched_ratio={ratio:.4f}  max_diff={max_diff:.3e}")

print("Kernel Output Match!")
