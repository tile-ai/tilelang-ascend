"""Test suite for ForeachNorm operator."""

import torch
from foreach_norm_kernel import foreach_norm, golden_foreach_norm


def main():
    import tilelang

    tilelang.disable_cache()
    torch.manual_seed(0)
    x_list = [torch.randn(8192, dtype=torch.float16, device="npu") for _ in range(4)]
    output = foreach_norm(x_list, 2.0)
    golden = golden_foreach_norm(x_list, 2.0)
    all_match = True
    for i, (out, gold) in enumerate(zip(output, golden)):
        diff = abs(out.item() - gold.item())
        if diff > 1e-3:
            all_match = False
            print(f"[PRECISION_FAIL] tensor {i}: output={out.item()}, golden={gold.item()}, diff={diff:.3e}")
        else:
            print(f"[PRECISION_PASS] tensor {i}: output={out.item():.4f}, diff={diff:.3e}")
    if all_match:
        print("KERNEL OUTPUT MATCH")
        print("TEST PASSED!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
