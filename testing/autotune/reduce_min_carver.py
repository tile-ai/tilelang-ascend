import argparse
import itertools

import tilelang
import tilelang.language as T
import torch
import os
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

tilelang.cache.clear_cache()

os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=64, help="Matrix N dimension")
args = parser.parse_args()

M = args.m
N = args.n

def get_config() -> list[dict]:
    arch = Ascend()
    carver_template = carver.GeneralReductionTemplate(
        structure="SR",
        shape=[M, N],
        dtype="float16",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        print(hint)
        config = {
            "block_M": hint.block[0],
        }
        configs.append(config)
    
    return configs

def ref_prog(a):
    return torch.min(a, dim=-1).values

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, N).half().npu(),
    ]

@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="npuir")
def reduce_min(M, N, block_M, dtype="float16", accum_dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            O: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            a = T.alloc_ub((block_M, N), dtype)
            s = T.alloc_ub((block_M, 1), dtype)
            
            offset = cid * block_M

            T.copy(A[offset, 0], a, size=[block_M, N])

            T.reduce_min(a, s, clear=True)

            T.copy(s, O[offset, 0], size=[block_M, 1])

    return main

func = reduce_min(M, N)

print("Best Config:", func.get_tuner_result())
print("Test passed!")
