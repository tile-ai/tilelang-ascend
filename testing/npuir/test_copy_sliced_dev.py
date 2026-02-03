# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

# ==========================================
# 2D Kernel & Test
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def test_slice_copy_2d(block_M, block_N, idx, idx2, dtype="float16"):
    @T.prim_func
    def main(
        In_ones: T.Tensor((block_M, block_N), dtype),
        In_zeros: T.Tensor((block_M, block_N), dtype),
        Out1: T.Tensor((block_M, block_N), dtype),
        Out2: T.Tensor((block_M, block_N), dtype),
        Out3: T.Tensor(block_N, dtype),
        Out4: T.Tensor(block_N, dtype),
        Out5: T.Tensor((block_M, block_N), dtype),
        Out6: T.Tensor((block_M, block_N), dtype),
        Out7: T.Tensor((block_M, block_N), dtype),
        Out8: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((block_M, block_N), dtype)
            B_frag = T.alloc_fragment((block_M, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), dtype)
            D_frag = T.alloc_fragment((block_M, block_N), dtype)

            A_slice = T.alloc_fragment(block_N, dtype)
            B_slice = T.alloc_fragment(block_N, dtype)
            
            # 1. Initialize
            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag)
            T.copy(In_zeros, D_frag)
            
            # 2. Test Local Slice Copy (Frag -> Frag Slice)
            # T.copy(A_frag[idx] -> A_slice -> B_frag[idx])
            T.copy(A_frag[idx, :], A_slice)
            T.copy(A_slice, B_frag[idx, :])

            # 3. Test Global Slice Copy
            T.copy(In_ones[idx, :], B_slice)
            T.copy(B_slice, Out5[idx, :])
            
            # 4. Test Cross-Location Copy (GM -> UB -> GM)
            T.copy(In_ones[idx, :], C_frag[idx2, :])
            T.copy(C_frag[idx2, :], Out6[idx, :])

            # 5. Test Direct Fragment to Fragment Slice Copy with Offset
            T.copy(A_frag[idx, :], D_frag[idx2, :])

            # Outputs
            T.copy(C_frag, Out7)  # Dump C_frag
            T.copy(D_frag, Out8)  # Dump D_frag (Verify Frag->Frag copy)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)
    return main

def test_2d():
    print("="*30 + " Running 2D Test " + "="*30)
    block_M, block_N = 32, 128
    idx, idx2 = 8, 16
    
    kernel = test_slice_copy_2d(block_M, block_N, idx, idx2)
    
    input_ones = torch.ones(block_M, block_N).npu().half()
    input_zeros = torch.zeros(block_M, block_N).npu().half()
    
    # Initialize outputs (Out1, 2, 5, 6, 7, 8)
    outs = [torch.zeros(block_M, block_N).npu().half() for _ in range(6)]
    out3 = torch.zeros(block_N).npu().half()
    out4 = torch.zeros(block_N).npu().half()
    
    # Launch
    kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])
    
    expected_full = torch.ones(block_M, block_N).npu().half()
    expected_row = torch.ones(block_N).npu().half()
    
    # Basic Checks
    torch.testing.assert_close(outs[0], expected_full, rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        torch.testing.assert_close(out[idx], expected_row, rtol=1e-5, atol=1e-5)
        out[idx] = 0
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out3, expected_row, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out4, expected_row, rtol=1e-5, atol=1e-5)
    
    # Check Out6 (Cross Location)
    torch.testing.assert_close(outs[3][idx], expected_row, rtol=1e-5, atol=1e-5)
    outs[3][idx] = 0
    torch.testing.assert_close(outs[3], torch.zeros_like(outs[3]), rtol=1e-5, atol=1e-5)
    
    # Check Out7 (C_frag dump)
    torch.testing.assert_close(outs[4][idx2], expected_row, rtol=1e-5, atol=1e-5)
    outs[4][idx2] = 0
    torch.testing.assert_close(outs[4], torch.zeros_like(outs[4]), rtol=1e-5, atol=1e-5)

    # Check Out8 (D_frag dump): Should have 1s at idx2 (copied from A_frag[idx])
    torch.testing.assert_close(outs[5][idx2], expected_row, rtol=1e-5, atol=1e-5)
    outs[5][idx2] = 0
    torch.testing.assert_close(outs[5], torch.zeros_like(outs[5]), rtol=1e-5, atol=1e-5)
    
    print("2D Test Passed!")

# ==========================================
# 3D Kernel & Test
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def test_slice_copy_3d(B, M, N, idx, idx2, dtype="float16"):
    @T.prim_func
    def main(
        In_ones: T.Tensor((B, M, N), dtype),
        In_zeros: T.Tensor((B, M, N), dtype),
        Out1: T.Tensor((B, M, N), dtype),
        Out2: T.Tensor((B, M, N), dtype),
        Out3: T.Tensor((M, N), dtype),
        Out4: T.Tensor((M, N), dtype),
        Out5: T.Tensor((B, M, N), dtype),
        Out6: T.Tensor((B, M, N), dtype),
        Out7: T.Tensor((B, M, N), dtype),
        Out8: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((B, M, N), dtype)
            B_frag = T.alloc_fragment((B, M, N), dtype)
            C_frag = T.alloc_fragment((B, M, N), dtype)
            D_frag = T.alloc_fragment((B, M, N), dtype)

            A_slice = T.alloc_fragment((M, N), dtype)
            B_slice = T.alloc_fragment((M, N), dtype)
            
            # Initialize
            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag)
            T.copy(In_zeros, D_frag)
            
            # Normal Slice Tests
            T.copy(A_frag[idx, :, :], A_slice)
            T.copy(A_slice, B_frag[idx, :, :])
            T.copy(In_ones[idx, :, :], B_slice)
            T.copy(B_slice, Out5[idx, :, :])
            T.copy(In_ones[idx, :, :], C_frag[idx2, :, :])
            T.copy(C_frag[idx2, :, :], Out6[idx, :, :])
            
            # Frag to Frag Slice Copy with Offset
            # A_frag[idx] -> D_frag[idx2]
            T.copy(A_frag[idx, :, :], D_frag[idx2, :, :])

            T.copy(C_frag, Out7)
            T.copy(D_frag, Out8)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)
    return main

def test_3d():
    print("\n" + "="*30 + " Running 3D Test " + "="*30)
    B, M, N = 4, 32, 128
    idx, idx2 = 1, 3
    
    kernel = test_slice_copy_3d(B, M, N, idx, idx2)
    
    input_ones = torch.ones(B, M, N).npu().half()
    input_zeros = torch.zeros(B, M, N).npu().half()
    
    outs = [torch.zeros(B, M, N).npu().half() for _ in range(6)]
    out3 = torch.zeros(M, N).npu().half()
    out4 = torch.zeros(M, N).npu().half()
    
    kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])
    
    expected_full = torch.ones(B, M, N).npu().half()
    expected_slice = torch.ones(M, N).npu().half()
    
    # Basic Checks
    torch.testing.assert_close(outs[0], expected_full, rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        torch.testing.assert_close(out[idx], expected_slice, rtol=1e-5, atol=1e-5)
        out[idx] = 0
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out3, expected_slice, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out4, expected_slice, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[3][idx], expected_slice, rtol=1e-5, atol=1e-5)
    outs[3][idx] = 0
    torch.testing.assert_close(outs[3], torch.zeros_like(outs[3]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[4][idx2], expected_slice, rtol=1e-5, atol=1e-5)
    outs[4][idx2] = 0
    torch.testing.assert_close(outs[4], torch.zeros_like(outs[4]), rtol=1e-5, atol=1e-5)

    print(f"Verifying Frag->Frag Copy (A_frag[{idx}] -> D_frag[{idx2}])...")
    torch.testing.assert_close(outs[5][idx2], expected_slice, rtol=1e-5, atol=1e-5)
    outs[5][idx2] = 0
    torch.testing.assert_close(outs[5], torch.zeros_like(outs[5]), rtol=1e-5, atol=1e-5)

    print("3D Test Passed!")

# ==========================================
# 4D Kernel & Test
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def test_slice_copy_4d(B, H, M, N, idx_b, idx_h, idx_b2, idx_h2, dtype="float16"):
    @T.prim_func
    def main(
        In_ones: T.Tensor((B, H, M, N), dtype),
        In_zeros: T.Tensor((B, H, M, N), dtype),
        Out1: T.Tensor((B, H, M, N), dtype),
        Out2: T.Tensor((B, H, M, N), dtype),
        Out3: T.Tensor((M, N), dtype),
        Out4: T.Tensor((M, N), dtype),
        Out5: T.Tensor((B, H, M, N), dtype),
        Out6: T.Tensor((B, H, M, N), dtype),
        Out7: T.Tensor((B, H, M, N), dtype),
        Out8: T.Tensor((B, H, M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            A_frag = T.alloc_fragment((B, H, M, N), dtype)
            B_frag = T.alloc_fragment((B, H, M, N), dtype)
            C_frag = T.alloc_fragment((B, H, M, N), dtype)
            D_frag = T.alloc_fragment((B, H, M, N), dtype)

            A_slice = T.alloc_fragment((M, N), dtype)
            B_slice = T.alloc_fragment((M, N), dtype)
            
            T.copy(In_ones, A_frag)
            T.copy(In_zeros, B_frag)
            T.copy(In_zeros, C_frag) 
            T.copy(In_zeros, D_frag)
            
            # Normal Tests
            T.copy(A_frag[idx_b, idx_h, :, :], A_slice)
            T.copy(A_slice, B_frag[idx_b, idx_h, :, :])
            T.copy(In_ones[idx_b, idx_h, :, :], B_slice)
            T.copy(B_slice, Out5[idx_b, idx_h, :, :])
            T.copy(In_ones[idx_b, idx_h, :, :], C_frag[idx_b2, idx_h2, :, :])
            T.copy(C_frag[idx_b2, idx_h2, :, :], Out6[idx_b, idx_h, :, :])
            
            # Frag to Frag Slice Copy with Offset (4D)
            # A_frag[b, h] -> D_frag[b2, h2]
            T.copy(A_frag[idx_b, idx_h, :, :], D_frag[idx_b2, idx_h2, :, :])

            T.copy(C_frag, Out7)
            T.copy(D_frag, Out8)

            T.copy(A_frag, Out1)
            T.copy(B_frag, Out2)
            T.copy(A_slice, Out3)
            T.copy(B_slice, Out4)
    return main

def test_4d():
    print("\n" + "="*30 + " Running 4D Test " + "="*30)
    B, H, M, N = 2, 4, 32, 64
    idx_b, idx_h = 0, 1
    idx_b2, idx_h2 = 1, 3
    kernel = test_slice_copy_4d(B, H, M, N, idx_b, idx_h, idx_b2, idx_h2)
    
    input_ones = torch.ones(B, H, M, N).npu().half()
    input_zeros = torch.zeros(B, H, M, N).npu().half()
    
    outs = [torch.zeros(B, H, M, N).npu().half() for _ in range(6)] 
    out3 = torch.zeros(M, N).npu().half()
    out4 = torch.zeros(M, N).npu().half()
    
    kernel(input_ones, input_zeros, outs[0], outs[1], out3, out4, outs[2], outs[3], outs[4], outs[5])
    
    expected_full = torch.ones(B, H, M, N).npu().half()
    expected_slice = torch.ones(M, N).npu().half()
    
    # Basic Checks
    torch.testing.assert_close(outs[0], expected_full, rtol=1e-5, atol=1e-5)
    for out in [outs[1], outs[2]]:
        torch.testing.assert_close(out[idx_b, idx_h], expected_slice, rtol=1e-5, atol=1e-5)
        out[idx_b, idx_h] = 0
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out3, expected_slice, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out4, expected_slice, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[3][idx_b, idx_h], expected_slice, rtol=1e-5, atol=1e-5)
    outs[3][idx_b, idx_h] = 0
    torch.testing.assert_close(outs[3], torch.zeros_like(outs[3]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[4][idx_b2, idx_h2], expected_slice, rtol=1e-5, atol=1e-5)
    outs[4][idx_b2, idx_h2] = 0
    torch.testing.assert_close(outs[4], torch.zeros_like(outs[4]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs[5][idx_b2, idx_h2], expected_slice, rtol=1e-5, atol=1e-5)
    outs[5][idx_b2, idx_h2] = 0
    torch.testing.assert_close(outs[5], torch.zeros_like(outs[5]), rtol=1e-5, atol=1e-5)

    print("4D Test Passed!")

def main():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_2d()
    test_3d()
    test_4d()

if __name__ == "__main__":
    main()