import argparse
import itertools

import tilelang
import tilelang.language as T
import torch
import os

from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

N = args.n
K = args.k

def get_config() -> list[dict]:
    arch = Ascend()
    carver_template = carver.GEMVTemplate(
        N = N,
        K = K,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        print(hint)
        config = {
            #"BLOCK_M": hint.block[0] = 16
            "BLOCK_N": hint.block[1],
            "BLOCK_K": hint.rstep[0],
        }
        configs.append(config)
    
    return configs

def ref_prog(A, B):
    return A @ B.T

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(K,).half().npu(),
        torch.randn(N, K).half().npu(),
    ]

@tilelang.autotune(
    configs=get_config(), # get_config_combination is also ok
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="npuir")
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (bn, _):
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_shared((1,), accum_dtype)
            for tn in T.serial(BLOCK_N):
                T.clear(C_reg)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
                C[bn * BLOCK_N + tn] = C_reg[0]

    return main

os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

# To trigger auto-tuning, we should not provide the tunable parameters (block_M, block_N, K_L1)
# If provided, the auto-tuner will skip the tuning process and use the provided values.
func = naive_gemv(N, K)

print("Best Config:", func.get_tuner_result())

print("GEMV passed!")
