import tilelang
from tilelang import language as T
import torch
import random

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def clamp_kernel(size, max_val, min_val, dtype="float16"):
    block_size = 64 * 1024
    loop_num = (size + block_size - 1) // block_size
    
    VEC_NUM = 2
    
    @T.prim_func
    def main(
        input: T.Tensor([size], dtype),
        output: T.Tensor([size], dtype),
    ):
        with T.Kernel(loop_num, is_npu=True) as (cid, vid):
            idx = cid
            
            in_ub = T.alloc_ub((block_size // VEC_NUM), dtype)
            tmp_ub = T.alloc_ub((block_size // VEC_NUM), "uint8")
            
            with T.Scope("V"):
                T.copy(input[idx * block_size // VEC_NUM], in_ub)
                for i in range(size):
                    T.barrier_all()
                    # T.tile.clamp_min(in_ub, in_ub, tmp_ub, min_val, block_size // VEC_NUM)
                    # T.tile.clamp_max(in_ub, in_ub, tmp_ub, max_val, block_size // VEC_NUM)
                    T.tile.clamp(in_ub, in_ub, tmp_ub, min_val, max_val, block_size // VEC_NUM)
                    T.barrier_all()
                    
                T.copy(in_ub, output[idx * block_size // VEC_NUM])
                
    return main

size = 10
thresh = 10000
max_v = random.uniform(-1 * thresh, thresh)
min_v = random.uniform(-1 * thresh, thresh)

if min_v > max_v:
    max_v, min_v = min_v, max_v
    
func = clamp_kernel(size, max_v, min_v, "float16")
print("init successful!")

input_data = (torch.rand([size]) - 0.5) * 2 * thresh
input_data = input_data.half().npu()

output_data = func(input_data)
ref_output_data = torch.clamp(input_data, min_v, max_v)

result = (output_data == ref_output_data).all()
print(f"output_data == ref_output_data all is : {result}")
if result:
    print("Kernel Output Match!")