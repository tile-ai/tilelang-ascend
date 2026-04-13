import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

M = 1
N = 131
K = 10  # number of top elements to extract
DTYPE = "float"

# sort32 requires 32-element aligned, DMA requires 32-byte aligned
ub_N = ((N + 31) // 32) * 32

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def topk_kernel(M, N, K, block_M, block_N, dtype=DTYPE):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 1

    ub_N = ((block_N + 31) // 32) * 32

    @T.prim_func
    def main(
        x: T.Tensor((M, N), dtype),
        out: T.Tensor((M, 2 * K), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            src_ub = T.alloc_ub((block_M // VEC_NUM, ub_N), dtype)
            # dst: interleaved (value, index) pairs for top K, needs 2*K size
            dst_ub = T.alloc_ub((block_M // VEC_NUM, 2 * K), dtype)
            # tmp for topk_new: needs space for sort destination (2*ub_N)
            # plus Sort's own workspace (2*ub_N for float, 8*ub_N for half)
            # Total: 4*ub_N for float, 10*ub_N for half
            tmp_ub = T.alloc_ub((block_M // VEC_NUM, ub_N * 4), dtype)

            T.copy(x[bx * block_M + vid * block_M // VEC_NUM, by * ub_N], src_ub)

            # TopK: sort then extract top K (value, index) pairs into dst
            T.tile.topk(dst_ub, src_ub, tmp_ub, K, block_N)

            T.copy(dst_ub, out[bx * block_M + vid * block_M // VEC_NUM, 0])

    return main


block_M = 1
block_N = N

func = topk_kernel(M, N, K, block_M, block_N)
# print(func.get_kernel_source())
torch.manual_seed(0)

torch_dtype = torch.float16 if DTYPE in ("half", "float16") else torch.float32

x = torch.randn(M, N, dtype=torch_dtype).npu()

torch.npu.synchronize()
torch.npu.synchronize()

out = func(x)
torch.npu.synchronize()

out_cpu = out.cpu().float().reshape(-1)

# Output is interleaved (value, index, value, index, ...), take 2*K elements
out_values = out_cpu[0::2][:K]   # even positions: values
out_indices = out_cpu[1::2][:K]  # odd positions: indices

# Reference: full descending sort on original N elements, take top K
x_cpu = x.cpu().float().reshape(-1)
sorted_vals, sorted_indices = torch.sort(x_cpu, descending=True)
ref_values = sorted_vals[:K]
ref_indices = sorted_indices[:K].float()

print("out_values:", out_values)
print("out_indices:", out_indices)
print("ref_values:", ref_values)
print("ref_indices:", ref_indices)

# Verify values are sorted descending
assert torch.all(out_values[:-1] >= out_values[1:]), "Values are not sorted descending!"

# Verify top-K values match reference
torch.testing.assert_close(out_values, ref_values, rtol=1e-3, atol=1e-3)

# Verify indices match reference
torch.testing.assert_close(out_indices, ref_indices, rtol=1e-3, atol=1e-3)
print(f"TopKNew test passed! (M={M}, N={N}, K={K}, DTYPE={DTYPE})")
