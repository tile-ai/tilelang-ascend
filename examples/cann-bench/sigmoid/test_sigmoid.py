"""Sigmoid activation test: golden reference + precision verification.

Usage:
    python test_sigmoid.py

Imports the kernel + adapter from sigmoid.py, runs representative test configs
(L0 aligned + L1 non-aligned prime), and verifies precision via mixed tolerance
(cann-bench-derived thresholds).
"""

import torch

from sigmoid import sigmoid


def golden_sigmoid(x):
    """Sigmoid golden: y = 1 / (1 + exp(-x)). Uses torch.sigmoid."""
    return torch.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Representative configs: L0 (aligned) + L1 (non-aligned prime shape)
    test_configs = [
        ((1024, 1024), torch.float16),  # L0: basic fp16, block-aligned
        ((1009, 1021), torch.float16),  # L1: non-aligned prime shape
        ((2048, 2048), torch.float32),  # L0: basic fp32
        ((4096, 4096), torch.bfloat16),  # L0: basic bf16
        ((8192, 8192), torch.float16),  # L0: large fp16
    ]

    for shape, dtype in test_configs:
        print(f"Testing sigmoid shape={shape}, dtype={dtype}")
        x = torch.randn(shape, dtype=dtype, device="cpu").npu()
        y = sigmoid(x)
        ref = golden_sigmoid(x)
        # Precision check (mixed tolerance by dtype)
        y_cpu, ref_cpu = y.detach().cpu().float(), ref.detach().cpu().float()
        m = torch.isfinite(ref_cpu) & torch.isfinite(y_cpu)
        abs_err = (y_cpu[m] - ref_cpu[m]).abs()
        if dtype == torch.float16:
            atol, rtol, max_abs_limit, required_ratio = 6.10e-5, 9.77e-4, 1e2, 0.99
        elif dtype == torch.bfloat16:
            atol, rtol, max_abs_limit, required_ratio = 4.88e-4, 7.81e-3, 1e3, 0.99
        else:
            atol, rtol, max_abs_limit, required_ratio = 7.63e-6, 1.22e-4, 1e0, 0.99
        ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
        max_abs = abs_err.max().item()
        assert ratio >= required_ratio and max_abs <= max_abs_limit, f"precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
        print(f"  Test pass! matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

    print("Kernel Output Match!")
