import tilelang
from tilelang import DataType, language as T
import torch


def _check_precision(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape:
        raise AssertionError("shape mismatch")
    if not (actual.is_floating_point() or expected.is_floating_point()):
        if not torch.equal(actual, expected):
            raise AssertionError("integer mismatch")
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
    ratio, maximum = (diff <= atol + rtol * expected[finite].abs()).float().mean().item(), diff.max().item()
    if ratio < 0.99 or maximum > cap:
        raise AssertionError(f"precision mismatch: ratio={ratio:.6f}, max_abs={maximum:.6g}")


import torch.nn.functional as F

"""
Functionality:
O = (I + A)^{-1}
"""

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


def current_jit_target(jit_func):
    return jit_func.__jit_impl__.target


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def solve_tril_ker(B, H, L, C, dtype="float16", accum_dtype="float"):
    chunk_num = T.ceildiv(L, C)
    VEC_NUM = 2
    is_pto = current_jit_target(solve_tril_ker) == "pto"

    def alloc_temp():
        if is_pto:
            return T.alloc_ub([C, C], accum_dtype)
        else:
            return T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * C * C // VEC_NUM], "uint8")

    @T.prim_func
    def main(
        A: T.Tensor([B, H, L, C], dtype),
        I: T.Tensor([C, C], accum_dtype),
        O: T.Tensor([B, H, L, C], dtype),
    ):
        with T.Kernel(B * (H // VEC_NUM) * chunk_num, is_npu=True) as (cid, vid):
            bx = cid % chunk_num
            by = (cid // chunk_num) % (H // VEC_NUM) * 2 + vid
            bz = (cid // chunk_num) // (H // VEC_NUM)

            o_ub = T.alloc_ub([C, C], accum_dtype)
            i_ub = T.alloc_ub([C, C], accum_dtype)
            mul_ub = T.alloc_ub([C, C], accum_dtype)
            red_ub = T.alloc_ub(
                [
                    C,
                ],
                accum_dtype,
            )
            o_ub_half = T.alloc_ub([C, C], dtype)

            with T.Scope("V"):
                T.copy(A[bz, by, bx * C, 0], o_ub_half)
                T.copy(I[0, 0], i_ub)
                T.tile.fill(mul_ub, 0.0)
                T.copy(o_ub_half, o_ub)
                for i in range(2, C):
                    T.tile.fill(red_ub, 0.0)
                    for j, k in T.Parallel(C, C):
                        mul_ub[j, k] = o_ub[j, k] * o_ub[i, j]
                    T.reduce_sum(mul_ub, red_ub, dim=0)
                    for j in T.Parallel(C):
                        o_ub[i, j] = o_ub[i, j] - red_ub[j]
                for i, j in T.Parallel(C, C):
                    o_ub[i, j] = i_ub[i, j] - o_ub[i, j]
                T.copy(o_ub, o_ub_half)
                T.copy(o_ub_half, O[bz, by, bx * C, 0])

    return main


def solve_tril(a):
    B, H, L, C = a.shape
    idt = torch.eye(C).npu().to(torch.float)
    ker = solve_tril_ker(B, H, L, C)
    b = ker(a, idt)
    return b


def solve_triangular(a):
    B, H, C = a.shape[0:-1]
    idt = torch.eye(C).npu().to(torch.float).view(1, 1, C, C)
    for i in range(C):
        mul = torch.zeros((B, H, C, C)).npu().to(torch.float)
        for j in range(C):
            mul[:, :, j] = a[:, :, j] * a[:, :, i, j].unsqueeze(-1)
        mul = mul.sum(axis=-2)
        a[:, :, i] -= mul
    a = idt - a
    return a


def ref_solve_tril(a):
    B, H, L, C = a.shape
    chunk_num = (L + C - 1) // C
    o = torch.zeros((B, H, L, C)).npu().to(torch.float)
    a = a.float()
    for i in range(chunk_num):
        a_c = a[:, :, i * C : (i + 1) * C, :]
        o_c = solve_triangular(a_c)
        o[:, :, i * C : (i + 1) * C, :] = o_c
    return o.to(torch.float16)


def ref_chunk_cumsum(g, C):
    B, H, L = g.shape
    chunk_num = (L + C - 1) // C
    g = g.view(B, H, chunk_num, C)
    g_sum = torch.cumsum(g, dim=-1)
    g_sum = g_sum.view(B, H, L)
    return g_sum


def ref_kkt(k, beta, g, C):
    B, H, L, DK = k.shape
    chunk_num = (L + C - 1) // C
    a = torch.zeros((B, H, L, C)).npu().to(torch.float)
    beta = beta.float()

    for i in range(chunk_num):
        k_c = k[:, :, i * C : (i + 1) * C, :]
        beta_c = beta[:, :, i * C : (i + 1) * C]
        g_c = g[:, :, i * C : (i + 1) * C]
        kkt = torch.einsum("bhid,bhjd->bhij", k_c, k_c).float()
        gamma = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        gamma = torch.exp(gamma)
        a_c = (kkt * beta_c.unsqueeze(-1) * gamma).tril(-1)
        a[:, :, i * C : (i + 1) * C, :] = a_c

    return a.to(torch.float16)


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=float("inf"), sci_mode=True)

    test_configs = [
        (2, 32, 256, 64, 32),
    ]

    for B, H, L, DK, C in test_configs:
        print(f"Testing Solve Tril with B={B}, H={H}, L={L}, C={C}")
        chunk_num = (L + C - 1) // C
        k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
        beta = torch.rand((B, H, L)).npu().to(torch.float16)
        g = torch.randn((B, H, L)).npu().to(torch.float)
        k = F.normalize(k, dim=-1, p=2)
        g = F.logsigmoid(g)
        g = ref_chunk_cumsum(g, C)
        a = ref_kkt(k, beta, g, C)

        o = solve_tril(a)
        ref_o = ref_solve_tril(a)
        _check_precision(o.cpu(), ref_o.cpu())
        print("Test passed!")

    print("Kernel Output Match!")
