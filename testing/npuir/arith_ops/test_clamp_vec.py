import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T
import os

# -------------------------
# NPU Configuration
# -------------------------
torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="TileLang Clamp Kernel Test")
parser.add_argument("--M", type=int, default=4, help="Size of dimension M")
parser.add_argument("--N", type=int, default=4, help="Size of dimension N")

dtype = "float16"
accum_dtype = "float16"
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0

# -------------------------
# TileLang clamp kernel (support min/max as tensors)
# -------------------------
def clamp_kernel(M, N, min_tensor, max_tensor):
    BLOCK_SIZE = 1

    @T.prim_func
    def clampVecExpKernel(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype),
             min_val: T.Tensor((M, N), dtype),
             max_val: T.Tensor((M, N), dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB memory
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)
            min_ub = T.alloc_ub((M, N), dtype)
            max_ub = T.alloc_ub((M, N), dtype)

            # Copy from GM to UB
            T.copy(src, src_ub)

            # Clamp operation
            
            T.copy(min_val, min_ub)
            T.copy(max_val, max_ub)
            T.vclamp(src_ub, dst_ub, min_ub, max_ub)
        

            # Copy back from UB to GM
            T.copy(dst_ub, dst)

    return clampVecExpKernel



# -------------------------
# Generate test tensor
# -------------------------
def generate_tensor(shape, dtype, clear=False):
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return (torch.rand(size=shape, dtype=eval("torch." + dtype)) * 200.0) - 50.0
    raise ValueError(f'Unsupported dtype: {dtype}')

# -------------------------
# Main function
# -------------------------
def main_func(main_args):
    shape = (main_args.M, main_args.N)
    
    # Create input tensor
    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, accum_dtype, clear=True).npu()
    
    # Generate min_tensor and max_tensor such that min < max
    min_tensor = generate_tensor(shape, dtype).npu()
    # Ensure max_tensor > min_tensor by adding a positive offset
    positive_offset = torch.rand(shape, dtype=min_tensor.cpu().dtype) * 50.0 + 1.0
    max_tensor = (min_tensor.cpu() + positive_offset).npu()

    print("Input tensor:")
    print(src.cpu())
    print("Min tensor:")
    print(min_tensor.cpu())
    print("Max tensor:")
    print(max_tensor.cpu())

    # Compile kernel
    func = clamp_kernel(main_args.M, main_args.N, min_tensor=min_tensor, max_tensor=max_tensor)
    compiled_kernel = tilelang.compile(func, target='npuir')

    # Execute kernel
    compiled_kernel(src, dst, min_tensor, max_tensor)

    print("\nNPU clamp result:")
    print(dst.cpu())

    # PyTorch reference
    ref = torch.clamp(src.cpu(), min=min_tensor.cpu(), max=max_tensor.cpu())
    print("\nPyTorch reference result:")
    print(ref)

    # Verification
    if torch.allclose(dst.cpu(), ref, rtol=1e-3, atol=1e-3):
        print("\n\033[92mResults match!\033[0m")
    else:
        print("\n\033[91mResults do not match!\033[0m")
        diff = torch.abs(dst.cpu() - ref)
        print(f"Max difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    print("<<<<< Expert Mode >>>>>")
    os.environ['TILELANG_ASCEND_MODE'] = 'Expert'
    main_func(args)
