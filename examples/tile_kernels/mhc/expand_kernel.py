import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2

_PASS_CONFIGS_FWD = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_PASS_CONFIGS_BWD = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS_FWD)
def expand_to_mhc_fwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.symbolic("num_tokens")
    dbtype = "bfloat16"
    h = hidden
    mhc = mhc_mult

    blk_n = 64
    sub_blk_n = blk_n // VEC_NUM
    blk_h = hidden

    m_blocks = T.ceildiv(n, blk_n)

    @T.prim_func
    def expand_to_mhc_fwd_kernel(
        x: T.Tensor[(n, h), dbtype],
        o: T.Tensor[(n, mhc, h), dbtype],
    ) -> None:
        with T.Kernel(m_blocks, is_npu=True) as (cid, vid):
            if n > 0:
                row_start = cid * blk_n + vid * sub_blk_n

                xl_ub = T.alloc_ub((sub_blk_n, blk_h), dbtype)

                T.copy(x[row_start : row_start + sub_blk_n, 0:blk_h], xl_ub)
                for m in T.serial(mhc):
                    T.copy(xl_ub, o[row_start : row_start + sub_blk_n, m, 0:blk_h])

    return expand_to_mhc_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS_BWD)
def expand_to_mhc_bwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden
    mhc = mhc_mult

    blk_n = 8
    sub_blk_n = blk_n // VEC_NUM
    blk_h = hidden

    m_blocks = T.ceildiv(n, blk_n)

    @T.prim_func
    def expand_to_mhc_bwd_kernel(
        o_grad: T.Tensor[(n, mhc, h), dbtype],
        x_grad: T.Tensor[(n, h), dbtype],
    ) -> None:
        with T.Kernel(m_blocks, is_npu=True) as (cid, vid):
            if n > 0:
                row_start = cid * blk_n + vid * sub_blk_n

                ogl_bf16_db = T.alloc_ub((2, sub_blk_n, blk_h), dbtype)
                ogl_f32 = T.alloc_ub((sub_blk_n, blk_h), dtype)
                xgl_f32 = T.alloc_ub((sub_blk_n, blk_h), dtype)

                T.tile.fill(xgl_f32, 0.0)

                T.copy(o_grad[row_start : row_start + sub_blk_n, 0, 0:blk_h], ogl_bf16_db[0, :, :])
                T.set_flag("mte2", "v", 0)

                for m in T.serial(mhc):
                    cur = m % 2

                    T.wait_flag("mte2", "v", cur)

                    if m + 1 < mhc:
                        nxt = (m + 1) % 2
                        if m >= 1:
                            T.wait_flag("v", "mte2", 2 + nxt)
                        T.copy(o_grad[row_start : row_start + sub_blk_n, m + 1, 0:blk_h], ogl_bf16_db[nxt, :, :])
                        T.set_flag("mte2", "v", nxt)

                    T.tile.cast(ogl_f32, ogl_bf16_db[cur, :, :], "CAST_NONE", sub_blk_n * blk_h)
                    T.tile.add(xgl_f32, xgl_f32, ogl_f32)

                    if m + 2 < mhc:
                        T.set_flag("v", "mte2", 2 + cur)

                T.tile.cast(ogl_bf16_db[0, :, :], xgl_f32, "CAST_ROUND", sub_blk_n * blk_h)

                T.set_flag("v", "mte3", 4)
                T.wait_flag("v", "mte3", 4)
                T.copy(ogl_bf16_db[0, :, :], x_grad[row_start : row_start + sub_blk_n, 0:blk_h])

    return expand_to_mhc_bwd_kernel


def expand_to_mhc_ref(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()


def test_fwd():
    n = 8192
    mhc_mult = 4
    h = 1280

    device = "npu"

    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)

    out_ref = expand_to_mhc_ref(x, mhc_mult)
    out_tl = torch.zeros((n, mhc_mult, h), dtype=torch.bfloat16, device=device)

    fwd_func = expand_to_mhc_fwd_tl(h, mhc_mult)
    fwd_func(x, out_tl)

    torch.testing.assert_close(out_tl, out_ref)
    print("Kernel Output Match!")


def test_bwd():
    n = 8192
    mhc_mult = 4
    h = 1280

    device = "npu"

    torch.manual_seed(42)
    o_grad = torch.randn((n, mhc_mult, h), dtype=torch.bfloat16, device=device)

    x_grad_tl = torch.zeros((n, h), dtype=torch.bfloat16, device=device)

    bwd_func = expand_to_mhc_bwd_tl(h, mhc_mult)
    bwd_func(o_grad, x_grad_tl)

    x_grad_ref = o_grad.sum(dim=1).to(torch.bfloat16)

    torch.testing.assert_close(x_grad_tl, x_grad_ref, atol=4e-2, rtol=1e-2)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
