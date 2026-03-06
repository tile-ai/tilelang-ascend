import os
import argparse
import torch
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
# parser.add_argument("--M", type=int, default=8, help="")
# parser.add_argument("--N", type=int, default=16, help="")

dtype = "float16"

def reshape_dev(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((N, M), dtype)
            T.copy(A,a)
            T.npuir_reshape(a,b)
            T.npuir_exp(b,b)
            T.copy(b,B)
    return main

def reshape_exp(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype)
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), dtype)
            b = T.alloc_ub((N, M), dtype)
            T.copy(A,a)
            T.npuir_reshape(a,b)
            T.npuir_exp(b,b)
            T.copy(b,B)
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
    M = torch.randint(0, 64, (1,)).item()
    N = torch.randint(0, 64, (1,)).item()
    if os.environ["TILELANG_ASCEND_MODE"] == "Dev":
        func = reshape_dev(M, N)
    else:
        func = reshape_exp(M, N)
    kernel = tilelang.compile(func, target="npuir")

    shape1 = (M, N)
    shape2 = (N, M)
    A = generate_tensor(shape1, dtype).npu()
    B = generate_tensor(shape2, dtype).npu()
    kernel(A, B)

    res = A.reshape(N, M)
    res = torch.exp(res)
    torch.testing.assert_close(
        C, res, rtol=1e-3, atol=1e-3
    )

    print("\033[92mReshape demo passed!\033[0m")


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    main(args)
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    main(args)
