import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T

# Set the NPU device
torch.npu.set_device(0)
tilelang.cache.clear_cache()

# Argument parser
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4, help="Size of dimension M")
parser.add_argument("--N", type=int, default=4, help="Size of dimension N")
parser.add_argument("--dim", type=int, default=0, help="Dimension to perform cumulative sum on")
parser.add_argument("--reverse", action="store_true", help="Perform reverse cumulative sum")

dtype = "float16"
accum_dtype = "float16"

def cumsum_kernel(M, N, dim, reverse):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB memory
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)
            
            # Copy data from GM to UB
            T.copy(src, src_ub)
            
            # Perform cumulative sum
            T.cumsum(src_ub, dst_ub, dim=dim, reverse=reverse)
            
            # Copy results back to GM
            T.copy(dst_ub, dst)
    
    return main

def generate_tensor(shape, dtype, clear=False):
    """Generate a tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=10, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError(f'Invalid dtype parameter: {dtype}')

def main(main_args):
    # Compile the cumsum kernel
    func = cumsum_kernel(
        main_args.M,
        main_args.N,
        main_args.dim,
        main_args.reverse
    )
    
    compiled_kernel = tilelang.compile(func, target='npuir')
    
    # Create input and output tensors
    shape = (main_args.M, main_args.N)
    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, accum_dtype, clear=True).npu()
    
    print("Input tensor:")
    print(src.cpu())
    
    # Run the kernel
    compiled_kernel(src, dst)
    
    print("\nNPU cumsum result:")
    print(dst.cpu())
    
    # Compute reference result using PyTorch
    if main_args.reverse:
        ref = torch.flip(torch.cumsum(torch.flip(src.cpu(), dims=[main_args.dim]), 
                                     dim=main_args.dim), dims=[main_args.dim])
    else:
        ref = torch.cumsum(src.cpu(), dim=main_args.dim)
    
    print("\nPyTorch reference result:")
    print(ref)
    
    # Validate results
    if torch.allclose(dst.cpu(), ref, rtol=1e-3, atol=1e-3):
        print("\n\033[92mResults match!\033[0m")
    else:
        print("\n\033[91mResults do NOT match!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
