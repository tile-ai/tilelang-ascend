import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()
dtype = "float32"
M = 2
K = 64
N = 32


def vec_sqrt(M, K, N, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((M, K, N), dtype),
             B: T.Tensor((M, K, N), dtype),
             ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, K, N), dtype)
            s = T.alloc_shared((M, K, N), dtype)

            T.copy(A, a)
            T.vsqrt(a, s)

            T.copy(s, B)

    return main


def test_sqrt():
    torch.npu.set_device(0)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

    func = vec_sqrt(M, K, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).abs().npu()
    v2 = torch.randn(size=[M, K, N], dtype=eval("torch." + dtype)).npu()

    v_ref = torch.sqrt(v1)
    compiled_kernel(v1, v2)
    print(v_ref)

    torch.testing.assert_close(v_ref, v2, rtol=1e-2, atol=1e-2)
    print("Sqrt Pass!")


if __name__ == "__main__":
    test_sqrt()