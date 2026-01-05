import torch
import torch_npu
import argparse
import tilelang
import tilelang.language as T

# cumsum示例：对输入张量沿指定维度进行累积求和
torch.npu.set_device(0)
tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4, help="维度M的大小")
parser.add_argument("--N", type=int, default=4, help="维度N的大小")
parser.add_argument("--dim", type=int, default=0, help="累积求和的维度")
parser.add_argument("--reverse", action="store_true", help="是否反向累积")

dtype = "float16"
accum_dtype = "float16"

def cumsum_kernel(M, N, dim, reverse):
    BLOCK_SIZE = 1

    @T.prim_func
    def main(src: T.Tensor((M, N), dtype),
             dst: T.Tensor((M, N), accum_dtype)):
        
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            # 分配UB内存
            src_ub = T.alloc_ub((M, N), dtype)
            dst_ub = T.alloc_ub((M, N), accum_dtype)
            
            # 从GM拷贝到UB
            T.copy(src, src_ub)
            
            T.cumsum(src_ub, dst_ub, dim=dim, reverse=reverse)
            
            # 将结果拷贝回GM
            T.copy(dst_ub, dst)
    
    return main

def generate_tensor(shape, dtype, clear=False):
    """生成张量"""
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
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def main(main_args):
    # 编译cumsum内核
    func = cumsum_kernel(
        main_args.M,
        main_args.N,
        main_args.dim,
        main_args.reverse
    )
    
    compiled_kernel = tilelang.compile(func, target='npuir')
    
    # 创建输入输出张量
    shape = (main_args.M, main_args.N)
    src = generate_tensor(shape, dtype).npu()
    dst = generate_tensor(shape, accum_dtype, clear=True).npu()
    
    print("输入张量:")
    print(src.cpu())
    
    # 执行核函数
    compiled_kernel(src, dst)
    
    print("\nNPU cumsum结果:")
    print(dst.cpu())
    
    # 使用PyTorch计算参考结果
    if main_args.reverse:
        # 反向cumsum
        ref = torch.flip(torch.cumsum(torch.flip(src.cpu(), dims=[main_args.dim]), 
                                     dim=main_args.dim), dims=[main_args.dim])
    else:
        # 正向cumsum
        ref = torch.cumsum(src.cpu(), dim=main_args.dim)
    
    print("\nPyTorch参考结果:")
    print(ref)
    
    # 验证结果
    if torch.allclose(dst.cpu(), ref, rtol=1e-3, atol=1e-3):
        print("\n\033[92m结果匹配！\033[0m")
    else:
        print("\n\033[91m结果不匹配！\033[0m")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
