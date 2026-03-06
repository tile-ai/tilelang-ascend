import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T
import os

torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Sigmoid Kernel Test (Fixed)")
parser.add_argument("--M", type=int, default=4, help="Size of dimension M")
parser.add_argument("--N", type=int, default=4, help="Size of dimension N")
parser.add_argument("--dtype", type=str, default="float16", help="Data type")

def sigmoid_kernel(M, N, dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
 
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), dtype)
            
            T.copy(src, src_ub)
            T.npuir_sigmoid(src_ub, dst_ub)
            T.copy(dst_ub, dst)
    
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
    print("=" * 60)
    print(f"Testing Sigmoid Function")
    print("=" * 60)
    
    func = sigmoid_kernel(
            main_args.M,
            main_args.N,
            main_args.dtype
        )

    compiled_kernel = tilelang.compile(func, target='npuir')

    # Create input and output tensors
    shape = (main_args.M, main_args.N)
    src = generate_tensor(shape, main_args.dtype).npu()
    dst = generate_tensor(shape, main_args.dtype, clear=True).npu()
    
    print(f"\nInput Tensor:")
    print(src.cpu())
    
    # Compute reference result using PyTorch
    ref = torch.sigmoid(src.cpu())

    try:
        # Execute the kernel function
        compiled_kernel(src, dst)
        print("Kernel execution successful")
    except Exception as e:
        print(f"Kernel execution failed: {e}")
        return
    
    print("Actual Result:")
    print(dst.cpu())
    print("Expected Result:")
    print(ref)
    
    # Verify the results
    if torch.allclose(dst.cpu(), ref, rtol=1e-3, atol=1e-3):
        print("\033[92mAll check passed!\033[0m")
    else:
        print("\n\033[91mResults do NOT match!\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    main(args)
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main(args)