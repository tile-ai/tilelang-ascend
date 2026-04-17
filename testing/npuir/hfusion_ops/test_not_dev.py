import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()
dtype = "int32"
M = 2
K = 64
N = 32


def vec_not(M, K, N, dtype="int32"):
    @T.prim_func
    def main(A: T.Tensor((M, K, N), dtype),
             B: T.Tensor((M, K, N), dtype),
             ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, K, N), dtype)
            s = T.alloc_shared((M, K, N), dtype)

            T.copy(A, a)
            T.vnot(a, s)

            T.copy(s, B)

    return main


def test_not():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vec_not(M, K, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randint(low=-10, high=10, size=[M, K, N], dtype=getattr(torch, dtype)).npu()
    v2 = torch.empty_like(v1)

    v_ref = ~v1
    compiled_kernel(v1, v2)

    torch.testing.assert_close(v_ref, v2, rtol=1e-2, atol=1e-2)
    print("Not Pass!")


if __name__ == "__main__":
    test_not()