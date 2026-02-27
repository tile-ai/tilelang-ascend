import os
import argparse
import torch
import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=128, help="")
parser.add_argument("--N", type=int, default=128, help="")

dtype = "float16"

def gather_dev(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_VEC = T.alloc_shared((M, N), dtype)
            B_VEC = T.alloc_shared((M, N), dtype)
            indices = T.alloc_shared((M, N), "int32")
            value_one = 1
            T.npuir_brc(value_one, indices)
            T.copy(A, A_VEC)
            T.vgather(A_VEC, B_VEC, indices)
            T.copy(B_VEC, B)

    return main

def gather_exp(M, N, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            A_VEC = T.alloc_ub((M, N), dtype)
            B_VEC = T.alloc_ub((M, N), dtype)
            indices = T.alloc_ub((M, N), "int32")
            value_one = 1
            T.npuir_brc(value_one, indices)
            T.copy(A, A_VEC)
            T.vgather(A_VEC, B_VEC, indices)
            T.copy(B_VEC, B)

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
    if os.environ['TILELANG_ASCEND_MODE'] == 'Expert':
        func = gather_exp(
            main_args.M,
            main_args.N
        )
    else:
        func = gather_dev(
            main_args.M,
            main_args.N
        )

    compiled_kernel = tilelang.compile(func, target='npuir')


    shape = (main_args.M, main_args.N)
    A = generate_tensor(shape, dtype).npu()
    B = generate_tensor(shape, dtype).npu()

    compiled_kernel(A, B)
    print("Actual Result:")
    print(B)
    idx = torch.ones(shape, dtype=torch.int32).npu()
    res = torch.gather(A, dim=1, index=idx)
    print(res)
    torch.testing.assert_close(B, res, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    main(args)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main(args)