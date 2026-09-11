"""Minimal tail-block elementwise-add example for the AscendC backend.

M and N are deliberately *not* multiples of the block, so the last row/column
blocks are tails. With the opt-in tail-mask scheme (TL_ASCEND_TAIL_MASK)
``T.tile.add`` is rewritten to a tail-aware helper that computes only over the
valid rectangle; the unused UB gap is pad-filled and never written back.
"""

import tilelang
import tilelang.language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {"float16": (2**-14, 2**-9, 0.1), "bfloat16": (2**-10, 2**-6, 1.0), "float32": (2**-16, 2**-10, 0.01)}
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    atol, rtol, limit = table.get(str(dtype).replace("torch.", ""), table["float16"])
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    # Opt in to the tail-block valid-region scheme (default off).
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def tail_add(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N, block_M, block_N, dtype in [
        (34, 130, 32, 32, "float"),
        (34, 130, 32, 32, "float16"),
        (100, 200, 64, 128, "float"),
    ]:
        print(f"tail_add M={M} N={N} block=({block_M},{block_N}) dtype={dtype}")
        func = tail_add(M, N, block_M, block_N, dtype=dtype)
        torch_dtype = torch.float32 if dtype == "float" else torch.float16
        a = torch.randn(M, N, dtype=torch_dtype).npu()
        b = torch.randn(M, N, dtype=torch_dtype).npu()
        c = func(a, b)
        passed, ratio, max_abs = _check_precision(c, a + b, c.dtype)
        assert passed, f"dtype={c.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
        print("  pass")
    print("Kernel Output Match!")
