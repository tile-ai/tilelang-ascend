import torch
import argparse
import tilelang
import tilelang.language as T
import os
import filecmp

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

def test_reduce():
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main_args = parser.parse_args([])
    func = row_reduce_sum_dev(
        main_args.M,
        main_args.N,
        main_args.block_M
    )
    kernel = tilelang.engine.lower(func, target='npuir')
    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    # Export to .mlir file
    output_file = './output/' + curr_name
    with open(output_file, 'w') as f:
        f.write(kernel)
    
    ref_file = "./mlir_files/" + curr_name
    # filecmp.cmp returns True if files are identical, False otherwise
    are_identical = filecmp.cmp(output_file, ref_file , shallow=False)
    # assertion for pytest
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"

if __name__ == "__main__":
    test_reduce()
