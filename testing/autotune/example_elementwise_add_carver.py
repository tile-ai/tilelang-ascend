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
parser.add_argument("--m", type=int, default=1024)
parser.add_argument("--n", type=int, default=1024)
args = parser.parse_args()

M = args.m
N = args.n

def ref_prog(x, y):
    return x + y

def get_config() -> list[dict]:
    arch = Ascend()
    carver_template = carver.ElementwiseFixTemplate(
        shape = [M, N],
        dtype="float32",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        print(hint)
        config = {
            "block_M": hint.block[0],
            "block_N": hint.block[1],
        }
        configs.append(config)
    
    return configs

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, N).npu(),
        torch.randn(M, N).npu(),
    ]

@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="npuir")
def elementwise_add(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
    @T.prim_func
    def elemAdd(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return elemAdd

func = elementwise_add(M, N)

print("Best Config:", func.get_tuner_result())
print("Test passed!")

