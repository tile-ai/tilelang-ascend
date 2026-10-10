import argparse

import tilelang
import tilelang.language as T
import torch

from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def get_config() -> list[dict]:
    arch = Ascend()
    carver_template = carver.MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        config = {
            "block_M": hint.block[0],
            "block_N": hint.block[1],
            "K_L1": hint.rstep[0],
        }
        configs.append(config)

    return configs


def ref_prog(A, B):
    return A @ B


def supply_prog(params):
    torch.manual_seed(0)
    return [torch.randn(M, K).half().npu(), torch.randn(K, N).half().npu()]


def manual_check_prog(lib_outs, ref_outs):
    actual, golden = lib_outs[0].detach().cpu().float(), ref_outs[0].detach().cpu().float()
    if actual.shape != golden.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {golden.shape}")
    atol, rtol, limit = 2**-14, 2**-9, 1e-1
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    ):
        raise AssertionError("NaN/Inf structure mismatch")
    finite = torch.isfinite(golden)
    if not finite.any():
        return
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    assert ratio >= 0.99 and max_abs <= limit, f"matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"


@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    manual_check_prog=manual_check_prog,
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
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

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)

            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# To trigger auto-tuning, we should not provide the tunable parameters (block_M, block_N, K_L1)
# If provided, the auto-tuner will skip the tuning process and use the provided values.
func = matmul(M, N, K)

print("Best Config:", func.get_tuner_result())
print("Test passed!")
