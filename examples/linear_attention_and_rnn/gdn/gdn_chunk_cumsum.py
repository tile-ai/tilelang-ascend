import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    if not (actual.is_floating_point() or expected.is_floating_point()):
        mismatches = (actual != expected).sum().item()
        if mismatches:
            raise AssertionError(f"integer mismatch: {mismatches} elements")
        return
    table = {
        "float16": (2**-14, 2**-9, 1e-1),
        "bfloat16": (2**-10, 2**-6, 1e0),
        "float32": (2**-16, 2**-10, 1e-2),
        "hifloat32": (2**-16, 2**-10, 1e-2),
        "float8_e4m3": (2**-4, 2**-2, 1e0),
        "float8_e4m3fn": (2**-4, 2**-2, 1e0),
        "float8_e5m2": (2**-3, 2**-1, 1e-1),
    }
    atol, rtol, cap = table.get(str(expected.dtype).removeprefix("torch."), table["float16"])
    actual, expected = actual.float(), expected.float()
    special = ~torch.isfinite(expected)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(expected[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(expected[special]))
    ):
        raise AssertionError("NaN/Inf structure mismatch")
    finite = torch.isfinite(expected)
    if not finite.any():
        return
    diff = (actual[finite] - expected[finite]).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    ratio = (diff <= atol + rtol * expected[finite].abs()).float().mean().item()
    max_err = diff.max().item()
    if ratio < 0.99 or max_err > cap:
        raise AssertionError(f"precision mismatch: ratio={ratio:.6f}, max_abs={max_err:.6g}")


"""
Functionality:
Chunkwisely calculate the prefix sum
"""

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def cumsum_ker(B, H, L, C, accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
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
                    C,
                ],
                accum_dtype,
            )
            s_ub = T.alloc_ub(
                [
                    C,
                ],
                accum_dtype,
            )

            with T.Scope("V"):
                T.tile.fill(s_ub, 0.0)
                T.copy(G[bz, by, bx * C], g_ub)
                for i in range(C):
                    if i > 0:
                        s_ub[i] = s_ub[i - 1]
                    tmp = s_ub[i] + g_ub[i]
                    s_ub[i] = tmp
                T.copy(s_ub, S[bz, by, bx * C])

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
        (2, 32, 256, 32),
    ]

    for B, H, L, C in test_configs:
        print(f"Testing cumsum with B={B}, H={H}, L={L}, C={C}")
        g = torch.randn((B, H, L)).npu().to(torch.float)
        g_sum = chunk_cumsum(g, C)
        ref_g_sum = ref_chunk_cumsum(g, C)
        _check_precision(g_sum.cpu(), ref_g_sum.cpu())
        print("Test passed!")

    print("Kernel Output Match!")
