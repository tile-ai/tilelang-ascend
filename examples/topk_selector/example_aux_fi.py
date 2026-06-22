import argparse
import os
from typing import Literal
import torch
import tilelang as tl
import tilelang.language as T

INDEX_DTYPE = "int64"
FLOAT_DTYPE = "float32"
INT32_DTYPE = "int32"

pass_configs = {
    tl.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tl.jit(pass_configs=pass_configs)
def get_aux_fi_kernel(B_val: int, N_val: int, top_k: int, block_N: int, topk_sum: int, dtype: Literal["float32"] = FLOAT_DTYPE):
    """
    Generate NPU kernel.
    """
    num_tokens = B_val * N_val
    num_aligned_topk = (top_k + 31) // 32 * 32

    @T.prim_func
    def aux_fi_kernel(
        topk_idx: T.Tensor[(num_tokens, num_aligned_topk), INDEX_DTYPE],
        out: T.Tensor[(topk_sum,), dtype],
        num_aux_topk: T.int32,
    ):
        with T.Kernel(topk_sum, is_npu=True) as (cid, _):
            topk_idx_ub = T.alloc_ub((num_aligned_topk,), INDEX_DTYPE)
            count_ub = T.alloc_ub((1,), dtype)

            count_ub[0] = 0.0

            for t in T.serial(num_tokens):
                T.copy(topk_idx[t, 0:num_aligned_topk], topk_idx_ub[0:num_aligned_topk])

                for j in T.serial(top_k):
                    idx_int32 = T.cast(topk_idx_ub[j], INT32_DTYPE)
                    if idx_int32 >= 0 and idx_int32 == cid:
                        count_ub[0] = count_ub[0] + 1.0

            denom = T.cast(num_tokens, dtype) * T.cast(num_aux_topk, dtype)
            count_ub[0] = count_ub[0] * T.cast(topk_sum, dtype) / denom

            T.copy(count_ub[0:1], out[cid : cid + 1])

    return aux_fi_kernel


def aux_fi(
    B: int,
    N: int,
    top_k: int,
    block_N: int,
    topk_sum: int,
    topk_idx: torch.Tensor,
    num_aux_topk: int,
    dtype: Literal["float32"] = FLOAT_DTYPE,
) -> torch.Tensor:
    """
    Upper-level interface: flatten and pad input tensor, then launch kernel.
    """
    num_tokens = B * N

    out = torch.zeros(topk_sum, dtype=torch.float32, device=topk_idx.device)
    if num_tokens == 0:
        return out

    assert topk_idx.is_contiguous()

    if topk_idx.dim() == 3:
        assert topk_idx.shape[0] == B and topk_idx.shape[1] == N and topk_idx.shape[2] == top_k
        topk_idx_flat = topk_idx.view(-1, top_k)
    else:
        assert topk_idx.shape[0] == num_tokens and topk_idx.shape[1] == top_k
        topk_idx_flat = topk_idx

    num_aligned_topk = (top_k + 31) // 32 * 32

    if top_k < num_aligned_topk:
        pad_len = num_aligned_topk - top_k
        topk_idx_padded = torch.nn.functional.pad(topk_idx_flat, (0, pad_len), value=-1).contiguous()
    else:
        topk_idx_padded = topk_idx_flat

    kernel = get_aux_fi_kernel(B_val=B, N_val=N, top_k=top_k, block_N=block_N, topk_sum=topk_sum, dtype=dtype)

    if int(os.getenv("TK_PRINT_KERNEL_SOURCE", 0)):
        print(kernel.get_kernel_source())

    kernel(topk_idx_padded, out, num_aux_topk)

    return out


def ref_aux_fi(topk_idx: torch.Tensor, num_experts: int, num_aux_topk: int) -> torch.Tensor:
    """Compute auxiliary load-balancing frequency indicator f_i for each expert."""
    num_tokens, num_topk = topk_idx.shape
    if num_tokens == 0:
        return torch.zeros(num_experts, dtype=torch.float32, device=topk_idx.device)
    valid_idx = topk_idx[topk_idx >= 0]
    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_idx.device)
    ones_value = torch.ones(valid_idx.shape, dtype=torch.int64, device=topk_idx.device)
    counts.scatter_add_(0, valid_idx, ones_value)
    return counts.float() * num_experts / (num_tokens * num_aux_topk)


def check_case(B: int, N: int, top_k: int, block_N: int, topk_sum: int, num_aux_topk: int):
    """Single test case execution and verification."""
    print(f"\n Testing Case: B={B}, N={N}, top_k={top_k}, topk_sum(experts)={topk_sum} ")

    topk_idx = torch.randint(0, topk_sum, (B, N, top_k), dtype=torch.int64, device="npu")

    mask = torch.rand((B, N, top_k), device="npu") < 0.05
    topk_idx[mask] = -1

    res_out = aux_fi(
        B=B, N=N, top_k=top_k, block_N=block_N, topk_sum=topk_sum, topk_idx=topk_idx, num_aux_topk=num_aux_topk, dtype=FLOAT_DTYPE
    )

    topk_idx_flat = topk_idx.view(-1, top_k)
    ref_out = ref_aux_fi(topk_idx_flat, num_experts=topk_sum, num_aux_topk=num_aux_topk)

    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-5)
    print("[Forward] Accuracy Check Passed! (PyTorch Reference Match!)")


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="topk_sum_and_topk_group_idx_kernel Test Example")
    parser.add_argument("--b", type=int, default=16, help="Matrix dimension B")
    parser.add_argument("--n", type=int, default=64, help="Matrix dimension N")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    B, N = args.b, args.n

    top_k = 4
    block_N = 16
    topk_sum = 32
    num_aux_topk = 4

    torch.manual_seed(0)
    tl.cache.clear_cache()

    check_case(B, N, top_k, block_N, topk_sum, num_aux_topk)

    check_case(B=18672, N=1, top_k=6, block_N=1, topk_sum=8, num_aux_topk=1)
    check_case(B=32, N=128, top_k=8, block_N=32, topk_sum=64, num_aux_topk=8)
    check_case(B=4, N=2048, top_k=16, block_N=64, topk_sum=128, num_aux_topk=16)
    check_case(B=1, N=1, top_k=4, block_N=16, topk_sum=32, num_aux_topk=4)
    check_case(B=0, N=64, top_k=4, block_N=16, topk_sum=32, num_aux_topk=4)

    print("\ntopk_sum_and_topk_group_idx_kernel example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
