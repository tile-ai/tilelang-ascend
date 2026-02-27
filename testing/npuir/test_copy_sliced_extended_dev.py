# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

# ==========================================
# 4D Strided Kernel & Test (FIXED)
# Pattern: [Scalar, Slice, Scalar, Slice]
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def test_strided_copy_4d_kernel(
    B, S, H, D,        # Shape (Constants)
    S_blk, D_blk,      # Block Sizes (Constants)
    dtype="float16"
):
    @T.prim_func
    def main(
        In: T.Tensor((B, S, H, D), dtype),
        Out: T.Tensor((B, S, H, D), dtype),
        Debug_Frag: T.Tensor((S_blk, D_blk), dtype),
        idx_b: T.int32,
        idx_h: T.int32,
        off_s: T.int32,
        off_d: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            frag = T.alloc_fragment((S_blk, D_blk), dtype)
            
            # 1. Strided Read (GM -> UB)
            T.copy(
                In[
                    idx_b,
                    off_s : off_s + S_blk,
                    idx_h,
                    off_d : off_d + D_blk
                ],
                frag
            )
            
            # 2. Dump
            T.copy(frag, Debug_Frag)
            
            # 3. Strided Write (UB -> GM)
            T.copy(
                frag,
                Out[
                    idx_b,
                    off_s : off_s + S_blk,
                    idx_h,
                    off_d : off_d + D_blk
                ]
            )
    return main

def test_4d_strided():
    print("\n" + "="*30 + " Running 4D Strided Test (Middle Scalar) " + "="*30)
    # Params
    B, S, H, D = 2, 64, 8, 128
    S_blk, D_blk = 32, 64
    idx_b, idx_h = 1, 3
    off_s, off_d = 16, 32
    
    kernel = test_strided_copy_4d_kernel(B, S, H, D, S_blk, D_blk)
    
    # Data
    inp = torch.randn(B, S, H, D).npu().half()
    out = torch.zeros(B, S, H, D).npu().half()
    debug = torch.zeros(S_blk, D_blk).npu().half()
    
    kernel(inp, out, debug, idx_b, idx_h, off_s, off_d)
    
    # Verification
    expected_slice = inp[idx_b, off_s:off_s+S_blk, idx_h, off_d:off_d+D_blk]
    
    torch.testing.assert_close(debug, expected_slice, rtol=1e-5, atol=1e-5)
    
    expected_out = torch.zeros_like(out)
    expected_out[idx_b, off_s:off_s+S_blk, idx_h, off_d:off_d+D_blk] = expected_slice
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    
    print(">> 4D Strided Test Passed!")

# ==========================================
# 5D Interleaved Kernel & Test (FIXED: Arg Merging)
# Pattern: [Scalar, Scalar, Slice, Scalar, Slice]
# Optimization: Merge idx_0 and idx_1 to avoid 16-arg limit
# ==========================================
@tilelang.jit(out_idx=[-1], target="npuir")
def test_strided_copy_5d_kernel(
    D0, D1, D2, D3, D4, # Shape (Constants)
    Blk_2, Blk_4,       # Block Sizes (Constants)
    dtype="float16"
):
    @T.prim_func
    def main(
        In: T.Tensor((D0, D1, D2, D3, D4), dtype),
        Out: T.Tensor((D0, D1, D2, D3, D4), dtype),
        Debug_Frag: T.Tensor((Blk_2, Blk_4), dtype),
        # Merged Index to save argument slots (Max args ~16)
        idx_01: T.int32,  # Encodes idx_0 * D1 + idx_1
        idx_3: T.int32,
        off_2: T.int32,
        off_4: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (idx_, _):
            frag = T.alloc_fragment((Blk_2, Blk_4), dtype)
            
            # Unpack merged index
            # idx_0 = idx_01 // D1
            # idx_1 = idx_01 % D1
            # Since D1 is constant, this is efficient
            r_idx_0 = idx_01 // D1
            r_idx_1 = idx_01 % D1
            
            # 1. Deep Interleaved Read
            T.copy(
                In[
                    r_idx_0,
                    r_idx_1,
                    off_2 : off_2 + Blk_2,
                    idx_3,
                    off_4 : off_4 + Blk_4
                ],
                frag
            )
            
            # 2. Check
            T.copy(frag, Debug_Frag)
            
            # 3. Write
            T.copy(
                frag,
                Out[
                    r_idx_0,
                    r_idx_1,
                    off_2 : off_2 + Blk_2,
                    idx_3,
                    off_4 : off_4 + Blk_4
                ]
            )
    return main

def test_5d_strided():
    print("\n" + "="*30 + " Running 5D Interleaved Test " + "="*30)
    dims = (2, 2, 64, 4, 128) # D0, D1, ...
    D1 = dims[1]
    blk_2, blk_4 = 16, 32
    
    idx_0, idx_1, idx_3 = 1, 0, 2
    off_2, off_4 = 32, 64
    
    # Pre-calculate merged index
    idx_01_merged = idx_0 * D1 + idx_1
    
    kernel = test_strided_copy_5d_kernel(*dims, blk_2, blk_4)
    
    inp = torch.randn(*dims).npu().half()
    out = torch.zeros(*dims).npu().half()
    debug = torch.zeros(blk_2, blk_4).npu().half()
    
    # Pass merged index: 3 Tensors + 4 Scalars = 16 Args (Safe!)
    kernel(inp, out, debug, idx_01_merged, idx_3, off_2, off_4)
    
    # Verification
    expected_slice = inp[idx_0, idx_1, off_2:off_2+blk_2, idx_3, off_4:off_4+blk_4]
    
    torch.testing.assert_close(debug, expected_slice, rtol=1e-5, atol=1e-5)
    
    expected_out = torch.zeros_like(out)
    expected_out[idx_0, idx_1, off_2:off_2+blk_2, idx_3, off_4:off_4+blk_4] = expected_slice
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    
    print(">> 5D Interleaved Test Passed!")

def main():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_4d_strided()
    test_5d_strided()

if __name__ == "__main__":
    main()