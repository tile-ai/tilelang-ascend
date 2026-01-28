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
# TileLang clamp kernel
# -------------------------
def clamp_kernel(M, N):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype)):

        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # Allocate UB (Unified Buffer) memory
            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_fragment((M, N), accum_dtype)

            # Copy from GM (Global Memory) to UB
            T.copy(src, src_ub)

            # Clamp operation
            T.vclamp(src_ub, dst_ub, CLAMP_MIN, CLAMP_MAX)

            # Copy back from UB to GM
            T.copy(dst_ub, dst)

    return main

# -------------------------
# Generate test data
# -------------------------
def generate_tensor(shape, dtype, clear=False):
    """Generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        # Input range [-50, 150] to ensure clamp boundaries are triggered
        return (torch.rand(size=shape, dtype=eval("torch." + dtype)) * 200.0) - 50.0
    raise ValueError(f'Unsupported dtype: {dtype}')

# -------------------------
# Main function
# -------------------------
def main_func(main_args):
    # Compile the kernel
    func = clamp_kernel(main_args.M, main_args.N)
    compiled_kernel = tilelang.compile(func, target='npuir')

    # Create input and output tensors
    shape = (main_args.M, main_args.N)
    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, accum_dtype, clear=True).npu()

    print("Input tensor:")
    print(src.cpu())

    # Execute the kernel
    compiled_kernel(src, dst)

    print("\nNPU clamp result:")
    print(dst.cpu())

    # PyTorch reference result
    ref = torch.clamp(src.cpu(), min=CLAMP_MIN, max=CLAMP_MAX)
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
    print("<<<<< Developer Mode >>>>>")
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    main_func(args)

