import tilelang
from tilelang import DataType, language as T
import torch

torch.set_default_device('npu')
torch.manual_seed(42)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], target="pto", pass_configs=pass_configs)
def reduce_max_slice_buffer():
    dtype = "float"

    @T.prim_func
    def main(Input: T.Tensor([4, 8], dtype),
             Output: T.Tensor([1, 8], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_shared = T.alloc_ub((4, 8), dtype)
            out_shared = T.alloc_ub((1, 8), dtype=dtype)
            reduce_tmp_shared = T.alloc_shared([3 * DataType(dtype).bits // 8 * 4 * 8], "float")

            if vid == 0:
                T.copy(Input, in_shared)
                T.tile.reduce_max(out_shared, in_shared, reduce_tmp_shared, dim=-1, real_shape=[4, 4])
                T.copy(out_shared, Output)

    return main


func = reduce_max_slice_buffer()
print("init successful!")

dtype = torch.float
input = torch.randn((4, 8), dtype=dtype)
output = torch.empty((1, 8), dtype=dtype)
torch.npu.synchronize()

output = func(input)
torch.npu.synchronize()

ref_output = torch.max(input[:, :4], dim =-1, keepdim=True).values.T
torch.npu.synchronize()

# print(f"ref_output: {ref_output}")
# print(f"input: {input}")
# print(f"output: {output[:, :4]}")

torch.testing.assert_close(ref_output, output[:, :4], rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")

# kernel = tilelang.engine.lower(func,target="pto")
# print(kernel.kernel_source)