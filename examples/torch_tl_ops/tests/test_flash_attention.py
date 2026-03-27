"""
Flash Attention Operator Test
"""

import torch
import torch_npu
import tl_ascend_ops


def test_flash_attention():
    """Test Flash Attention operator"""
    print("=" * 60)
    print("Testing Flash Attention operator")
    print("=" * 60)
    
    seq_len = 512
    dim = 128
    
    print(f"\nTesting shape: seq_len={seq_len}, dim={dim}")
    
    q = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    k = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    v = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    
    scale = (1.0 / dim) ** 0.5
    
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    # Method 1: Call via package
    output = tl_ascend_ops.flash_attention(q, k, v)
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ tl_ascend_ops.flash_attention verification passed")
    
    # Method 2: Call via torch_npu
    output2 = torch_npu.flash_attention(q, k, v)
    torch.testing.assert_close(output2, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch_npu.flash_attention verification passed")
    
    # Method 3: Call via torch.ops
    output3 = torch.ops.tl_ascend_ops.flash_attention(q, k, v)
    torch.testing.assert_close(output3, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch.ops.tl_ascend_ops.flash_attention verification passed")


if __name__ == "__main__":
    test_flash_attention()
