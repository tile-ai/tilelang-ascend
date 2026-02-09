import argparse

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--n", type=int, default=1024, help="Vector N dimension")
args = parser.parse_args()

N = args.n

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def generate_arithmetic_progression(N, block_size, dtype="int32"):
    num_blocks = N // block_size

    @T.prim_func
    def main(
            output: T.Tensor((N,), dtype), 
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            start_idx = cid * block_size

            seq_ub = T.alloc_shared((block_size,), dtype)

            T.tile.arith_progression(seq_ub, start_idx, 1, block_size)

            T.copy(seq_ub, output[start_idx])
            
    return main

func = generate_arithmetic_progression(N, 64)

output = torch.zeros(N, dtype=torch.int32).npu()

torch.npu.synchronize()
print("init successful!")

result = func(output)

ref_result = torch.arange(0, N, dtype=torch.int32).npu()

torch.testing.assert_close(result, ref_result, rtol=0, atol=0)
print("Kernel Output Match!")