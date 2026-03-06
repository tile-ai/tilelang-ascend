import torch
import tilelang
import tilelang.language as T


def simple_4d_atomic_add_kernel(B, S, H, D, dtype="float32"):
    
    @T.prim_func
    def atomic_add_4d(
        A: T.Tensor((B, S, H, D), dtype),
        B_tensor: T.Tensor((B, S, H, D), dtype),
        shape_B: T.int32,
        shape_H: T.int32,
    ):
        with T.Kernel(B * H, is_npu=True) as (cid, _):
            b_idx = cid // shape_H
            h_idx = cid % shape_H
            
            tile = T.alloc_shared((S, D), dtype)
            
            T.copy(A[b_idx, 0:S, h_idx, 0:D], tile)

            T.npuir_atomic_add(B_tensor[b_idx, 0:S, h_idx, 0:D], tile)
    
    return atomic_add_4d


def test_atomic_add():
    torch.npu.set_device(0)
    
    B, S, H, D = 2, 64, 4, 128
    
    A = torch.randn(B, S, H, D, dtype=torch.float32).npu()
    B_tensor = torch.randn(B, S, H, D, dtype=torch.float32).npu()
    
    expected = A + B_tensor
    
    func = simple_4d_atomic_add_kernel(B, S, H, D)
    compiled_kernel = tilelang.compile(func, target="npuir")
    
    print("Running 4D atomic add...")
    compiled_kernel(A, B_tensor, B, H)
    
    print(f"Expected: First few elements: {expected[0:64]}")
    print(f"Actual: First few elements of B: {B_tensor[0:64]}")
    print(f"All elements equal to expected: {torch.allclose(B_tensor, expected)}")
    
    return B_tensor


if __name__ == "__main__":
    import os
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    test_atomic_add()