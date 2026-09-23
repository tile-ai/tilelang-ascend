import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden):
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if a.shape != g.shape:
        raise AssertionError("shape mismatch")
    if not (a.is_floating_point() or g.is_floating_point()):
        if not torch.equal(a, g):
            raise AssertionError("integer mismatch")
        return
    table = {
        "torch.float16": (2**-14, 2**-9, 1e-1),
        "torch.bfloat16": (2**-10, 2**-6, 1e0),
        "torch.float32": (2**-16, 2**-10, 1e-2),
        "hifloat32": (2**-16, 2**-10, 1e-2),
        "float8_e4m3": (2**-4, 2**-2, 1e0),
        "float8_e5m2": (2**-3, 2**-1, 1e-1),
    }
    name = str(g.dtype)
    key = "float8_e4m3" if "float8_e4m3" in name else "float8_e5m2" if "float8_e5m2" in name else name
    atol, rtol, limit = table.get(key, table["torch.float16"])
    a, g = a.float(), g.float()
    if not (torch.equal(torch.isnan(a), torch.isnan(g)) and torch.equal(torch.isinf(a), torch.isinf(g))):
        raise AssertionError("NaN/Inf structure mismatch")
    finite = torch.isfinite(g)
    if not finite.any():
        return
    error = (a[finite] - g[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    if (error <= atol + rtol * g[finite].abs()).float().mean().item() < 0.99 or error.max().item() > limit:
        raise AssertionError("precision mismatch")


"""
Functionality:
Chunkwisely calculate the prefix sum
"""

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, CC=8, accum_dtype="float"):
    chunk_num = T.ceildiv(L, C * CC)
    VEC_NUM = 2

    @T.prim_func
    def main(
        G: T.Tensor([B, H, L], accum_dtype),
        S: T.Tensor([B, H, L], accum_dtype),
    ):
        with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
            bz = (cid // chunk_num) // (H // VEC_NUM)

            g_ub = T.alloc_ub(
                [
                    C * CC,
                ],
                accum_dtype,
            )
            s_ub = T.alloc_ub(
                [
                    C * CC,
                ],
                accum_dtype,
            )  # Process CC chunks at a time

            with T.Scope("V"):
                T.tile.fill(s_ub, 0.0)
                T.copy(G[bz, by, bx * C * CC], g_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
                for ii in range(CC):  # For each chunk
                    ofs = ii * C

                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)

                    s_ub[ofs + 0] = g_ub[ofs + 0]
                    for i in range(1, C):
                        tmp2 = s_ub[ofs + i - 1] + g_ub[ofs + i]
                        s_ub[ofs + i] = tmp2  # Calculate prefix sum
                        # Must use variable tmp2 due to some compiler issue
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(s_ub, S[bz, by, bx * C * CC])

    return main


def chunk_cumsum(g, C):
    B, H, L = g.shape
    ker = cumsum_ker(B, H, L, C)
    g_sum = ker(g)
    return g_sum


def ref_chunk_cumsum(g, C):
    B, H, L = g.shape
    chunk_num = (L + C - 1) // C
    g = g.view(B, H, chunk_num, C)
    g_sum = torch.cumsum(g, dim=-1)
    g_sum = g_sum.view(B, H, L)
    return g_sum


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    test_configs = [
        (2, 16, 16384, 128),
    ]

    for B, H, L, C in test_configs:  # Ensure that L % (C * CC) = 0
        print(f"Testing cumsum with B={B}, H={H}, L={L}, C={C}")
        g = torch.randn((B, H, L)).npu().to(torch.float)
        g_sum = chunk_cumsum(g, C)
        ref_g_sum = ref_chunk_cumsum(g, C)
        _check_precision(g_sum.cpu(), ref_g_sum.cpu())
        print("Test passed!")

    print("Kernel Output Match!")
