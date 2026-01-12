import torch
import argparse
import tilelang
import tilelang.language as T
import os

torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=16, help="")
parser.add_argument("--N", type=int, default=16, help="")
parser.add_argument("--block_M", type=int, default=32, help="")

dtype = "float16"
accum_dtype = "float16"

def row_reduce_sum_exp(M, N, block_M):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
                C: T.Tensor((M, N), dtype),
                D: T.Tensor((M, N), dtype),
                O: T.Tensor((M,1), accum_dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_ub((M,N), dtype)
            b = T.alloc_ub((M,N), dtype)
            c = T.alloc_ub((M,N), dtype)
            d = T.alloc_ub((M,N), dtype)
            s = T.alloc_ub((M,1), accum_dtype)

            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, c)
            T.copy(D, d)
            
            T.reduce_abssum(a, s)
            T.reduce_max(b, s, clear = False)
            T.reduce_min(c, s, clear = False)
            T.reduce(d, s, dims=1, reduce_mode="sum", clear = False)

            T.copy(s, O)

    return main

def row_reduce_sum_dev(M, N, block_M):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
                C: T.Tensor((M, N), dtype),
                D: T.Tensor((M, N), dtype),
                O: T.Tensor((M,1), accum_dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            a = T.alloc_shared((M,N), dtype)
            b = T.alloc_shared((M,N), dtype)
            c = T.alloc_shared((M,N), dtype)
            d = T.alloc_shared((M,N), dtype)
            s = T.alloc_shared((M,1), accum_dtype)

            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, c)
            T.copy(D, d)
            
            T.reduce_abssum(a, s)
            T.reduce_max(b, s, clear = False)
            T.reduce_min(c, s, clear = False)
            T.reduce(d, s, dims=1, reduce_mode="sum", clear = False)

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
    if os.environ['TILELANG_ASCEND_MODE'] == 'Expert':
        func = row_reduce_sum_exp(
            main_args.M,
            main_args.N,
            main_args.block_M
        )
    else:
        func = row_reduce_sum_dev(
            main_args.M,
            main_args.N,
            main_args.block_M
        )

    compiled_kernel = tilelang.compile(func, target='npuir')

    # torch.manual_seed(0)

    shape = (main_args.M, main_args.N)
    shape2 = (main_args.M,1)
    A = generate_tensor(shape, dtype).npu()
    B = generate_tensor(shape, dtype).npu()
    C = generate_tensor(shape, dtype).npu()
    D = generate_tensor(shape, dtype).npu()
    O = generate_tensor(shape2, accum_dtype, True).npu()

    compiled_kernel(A, B, C, D, O)
    res1 = torch.sum(torch.abs(A), dim=1, keepdim=True)
    res2 = torch.max(B, dim=1, keepdim=True).values
    res3 = torch.maximum(res1, res2)
    res4 = torch.min(C, dim=1, keepdim=True).values
    res5 = torch.minimum(res3, res4)
    res = res5 + torch.sum(D, dim=1, keepdim=True)
    print("Actual Result:")
    print(O)
    print("Expected Result:")
    print(res)

    torch.testing.assert_close(O, res, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    main(args)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main(args)
