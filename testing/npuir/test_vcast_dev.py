# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

dtype = "float32"
dtype2 = "float16"
M = 512
N = 512

def vec_cast(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, N), src_dtype),
            B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), src_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            T.copy(A[bx, by], A_VEC)
            T.npuir_cast(A_VEC, B_VEC, round_mode="rint")
            T.copy(B_VEC, B[bx, by])

    return main

def test_vec_cast():
    # torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    func = vec_cast(M, N, 128, 256)
    compiled_kernel = tilelang.compile(func, target="npuir")
    

    v1 = torch.randn(size=[M, N], dtype=eval("torch." + dtype)).npu()
    v2 = torch.randn(size=[M, N], dtype=eval("torch." + dtype2)).npu()

    compiled_kernel(v1, v2)

    print(v1)
    print(v2)


if __name__ == "__main__":
    test_vec_cast()