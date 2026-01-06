import torch
import argparse
import tilelang
import tilelang.language as T

# A[M, N] -> O[M], O[i] = sum_j A[i, j]
torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=2, help="")
parser.add_argument("--N", type=int, default=2, help="")
parser.add_argument("--block_M", type=int, default=32, help="")

dtype = "float16"
accum_dtype = "float16"

def row_reduce_sum(M, N, block_M):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
                O: T.Tensor((M,1), accum_dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):

            x = T.alloc_ub((M,N), dtype)
            s = T.alloc_ub((M,1), accum_dtype)

            T.copy(A, x)

            # T.reduce_max(x, s)
            # T.reduce_min(x, s)
            # T.reduce_sum(x, s)
            T.reduce(x, s, dims=1, reduce_mode="sum")

            T.copy(s, O)

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
    func = row_reduce_sum(
        main_args.M,
        main_args.N,
        main_args.block_M
    )

    compiled_kernel = tilelang.compile(func, target='npuir')

    # torch.manual_seed(0)

    shape = (main_args.M, main_args.N)
    shape2 = (main_args.M,1)
    A = generate_tensor(shape, dtype).npu()
    O = generate_tensor(shape2, accum_dtype, True).npu()

    compiled_kernel(A, O)
    # res = torch.max(A, dim=1, keepdim=True).values
    # res = torch.min(A, dim=1, keepdim=True).values
    res = torch.sum(A, dim=1, keepdim=True)
    print(A)
    print("Actual Result:")
    print(O)
    print("Expected Result:")
    print(res)

    torch.testing.assert_close(O, res, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
