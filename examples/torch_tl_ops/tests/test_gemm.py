"""
Dynamic Shape GEMM Operator Test
"""

import torch
import torch_npu
import tl_ascend_ops


def test_gemm():
    """Test dynamic shape GEMM operator"""
    print("=" * 60)
    print("Testing dynamic shape GEMM operator")
    print("=" * 60)
    
    test_cases = [
        (1024, 512, 2048),
        (512, 1024, 512),
        (256, 256, 256),
    ]
    
    for M, N, K in test_cases:
        print(f"\nTesting shape: M={M}, N={N}, K={K}")
        
        a = torch.randn(M, K, dtype=torch.float16, device="npu:0")
        b = torch.randn(K, N, dtype=torch.float16, device="npu:0")
        ref = a @ b
        
        # Method 1: Call via package
        c = tl_ascend_ops.gemm(a, b)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ tl_ascend_ops.gemm verification passed")
        
        # Method 2: Call via torch_npu
        c2 = torch_npu.gemm(a, b)
        torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch_npu.gemm verification passed")
        
        # Method 3: Call via torch.ops
        c3 = torch.zeros(M, N, dtype=torch.float16, device="npu:0")
        torch.ops.tl_ascend_ops.gemm(a, b, c3)
        torch.testing.assert_close(c3, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch.ops.tl_ascend_ops.gemm verification passed")


if __name__ == "__main__":
    test_gemm()
