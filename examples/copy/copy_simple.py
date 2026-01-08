# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1], target="npuir")
def simple_copy_1d(L, block_L, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        In: T.Tensor((L,), dtype),
        A: T.Tensor((L,), dtype),
        B: T.Tensor((L,), dtype),
        C: T.Tensor((L,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(L, block_L), is_npu=True) as (cid, _):
            start_idx = cid * block_L
            
            A_frag = T.alloc_fragment((block_L,), dtype)
            B_frag = T.alloc_fragment((block_L,), dtype)
            C_frag = T.alloc_fragment((block_L,), accum_dtype)
            
            T.copy(In[start_idx], A_frag)

            T.copy(A_frag, B_frag)  # ub to ub
            T.npuir_add(A_frag, B_frag, B_frag)
            T.copy(B_frag, C_frag)  # ub to ub with cast

            T.copy(A_frag, A[start_idx])
            T.copy(B_frag, B[start_idx])
            T.copy(C_frag, C[start_idx])

    return main

def test_1d():
    print("Testing 1d copy...")
    kernel = simple_copy_1d(1024, 256)
    
    input = torch.ones(1024).npu().half()
    a = torch.zeros(1024).npu().half()
    b = torch.zeros(1024).npu().half()
    c = torch.zeros(1024).npu()
    
    kernel(input, a, b, c)
    print("Input:\n", input)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    torch.testing.assert_close(a, input, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b, a * 2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c, b.to(torch.float32), rtol=1e-5, atol=1e-5)
    print("Test success!")

def main():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    test_1d()

if __name__ == "__main__":
    main()