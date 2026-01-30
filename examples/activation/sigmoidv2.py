import tilelang
from tilelang import language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(42)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoidv2():
    dtype = "float"
    
    @T.prim_func
    def main(input: T.Tensor([4, 8], dtype),
             output: T.Tensor([4, 8], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            input_shared = T.alloc_ub((4, 8), dtype)
            output_shared = T.alloc_ub((4, 8), dtype)
            tmp_shared = T.alloc_ub((4, 8), "uint8")
            
            T.copy(input, input_shared)
            T.tile.sigmoid(output_shared, input_shared, tmp_shared)
            T.copy(output_shared, output)
            
    return main

dtype = torch.float
input = torch.randn([4, 8], dtype=dtype)
func = sigmoidv2()
print("init successful!")
output = func(input)

torch.npu.synchronize()

ref_output = torch.sigmoid(input)

torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")