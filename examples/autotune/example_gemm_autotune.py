import argparse
import itertools

import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

# config method 1: directly defining search space in get_config function
def get_config():
    return [
        {"block_M": 128, "block_N": 128, "K_L1": 64},
        {"block_M": 256, "block_N": 128, "K_L1": 64},
        {"block_M": 128, "block_N": 256, "K_L1": 64},
    ]

# config method 2: using itertools to generate combinations
def get_config_combination():
    block_M_options = [64, 128, 256]
    block_N_options = [64, 128, 256]
    K_L1_options = [64, 128]
    
    _config = list(itertools.product(block_M_options, block_N_options, K_L1_options))
    config = [{"block_M": c[0], "block_N": c[1], "K_L1": c[2]} for c in _config]
    return config

def ref_prog(A, B):
    return A @ B

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, K).half().npu(),
        torch.randn(K, N).half().npu()
    ]

@tilelang.autotune(
    configs=get_config(), # get_config_combination is also ok
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# To trigger auto-tuning, we should not provide the tunable parameters (block_M, block_N, K_L1)
# If provided, the auto-tuner will skip the tuning process and use the provided values.
func = matmul(M, N, K)

print("Best Config:", func.get_tuner_result())
print("Test passed!")
