import os
import argparse
import torch
import tilelang
import tilelang.language as T

torch.npu.set_device(1)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=3, help="")
parser.add_argument("--N", type=int, default=4, help="")
parser.add_argument("--block_M", type=int, default=32, help="")

dtype = "float16"

def reshape_demo(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype),
        C: T.Tensor((N, M), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            tmp = T.alloc_shared((N, M), dtype)
            b = T.alloc_shared((N, M), dtype)
            c = T.alloc_shared((N, M), dtype)
            T.copy(A, a)
            T.copy(B, b)

            T.npuir_reshape(a, tmp)
            T.npuir_add(b, tmp, c)
            T.copy(c, C)
    return main

def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def main(main_args):
    os.environ["TILELANG_ASCEND_MODE"] = "dev"
    M = main_args.M
    N = main_args.N
    func = reshape_demo(M, N)
    kernel = tilelang.compile(func, target="npuir")

    shape1 = (M, N)
    shape2 = (N, M)
    A = generate_tensor(shape1, dtype).npu()
    B = generate_tensor(shape2, dtype).npu()
    C = generate_tensor(shape2, dtype).npu()

    kernel(A, B, C)
    print(A)
    print(C)

    res = A.reshape(N, M)
    res += B

    print(res)

    torch.testing.assert_close(
        C, res, rtol=1e-3, atol=1e-3
    )

    print("\033[92mReshape demo passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)