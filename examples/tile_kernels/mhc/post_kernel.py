import math

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_PASS_CONFIGS_BWD = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


VEC_NUM = 2

TOKEN_BLOCK_SIZE_FWD = 64
TOKEN_BLOCK_SIZE_BWD = 64

MAX_FWD_H_BLK = 2048
MAX_BWD_H_BLK = 2048


def _compute_safe_h_blk(hidden: int, max_h_blk: int) -> int:
    if hidden <= max_h_blk:
        return hidden
    return math.gcd(hidden, max_h_blk)


def _compute_pad_mhc(mhc: int) -> int:
    return ((mhc + 7) // 8) * 8


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_post_fwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 0) -> tilelang.JITKernel:
    n = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden

    if h_blk == 0:
        h_blk = _compute_safe_h_blk(hidden, MAX_FWD_H_BLK)

    assert h % h_blk == 0, f"hidden={h} must be divisible by h_blk={h_blk}"

    pad_h_blk = T.ceildiv(h_blk, 16) * 16
    pad_mhc = _compute_pad_mhc(mhc)
    sub_blk_n = TOKEN_BLOCK_SIZE_FWD // VEC_NUM

    @T.prim_func
    def _mhc_post_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), dtype],
        b: T.Tensor[(n, mhc, h), dbtype],
        c: T.Tensor[(n, mhc), dtype],
        d: T.Tensor[(n, h), dbtype],
        x: T.Tensor[(n, mhc, h), dbtype],
    ) -> None:
        num_cores = T.ceildiv(n, TOKEN_BLOCK_SIZE_FWD)
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((pad_mhc, pad_mhc), dtype)
            c_ub = T.alloc_ub((pad_mhc,), dtype)

            b_bf16_ub = T.alloc_ub((pad_mhc, pad_h_blk), dbtype)
            b_f32_ub = T.alloc_ub((pad_mhc, pad_h_blk), dtype)

            d_bf16_ub = T.alloc_ub((pad_h_blk,), dbtype)
            d_f32_ub = T.alloc_ub((pad_h_blk,), dtype)

            x_f32_ub = T.alloc_ub((pad_mhc, pad_h_blk), dtype)
            x_bf16_ub = T.alloc_ub((pad_mhc, pad_h_blk), dbtype)

            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)

            for i_token in T.serial(sub_blk_n):
                pid_n = cid * TOKEN_BLOCK_SIZE_FWD + vid * sub_blk_n + i_token

                if pid_n < n:
                    with T.Scope("V"):
                        T.set_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 0)

                        T.copy(a[pid_n, 0:mhc, 0:mhc], a_ub[0:mhc, 0:mhc])
                        T.copy(c[pid_n, 0:mhc], c_ub[0:mhc])

                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)

                    for i0_h in T.serial(h // h_blk):
                        h_offset = i0_h * h_blk

                        with T.Scope("V"):
                            T.set_flag("mte3", "mte2", 0)
                            T.wait_flag("mte3", "mte2", 0)

                            T.tile.fill(b_bf16_ub, T.cast(0.0, "bfloat16"))
                            T.tile.fill(d_bf16_ub, T.cast(0.0, "bfloat16"))

                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)

                            for i_mhc_cp in T.serial(mhc):
                                T.copy(b[pid_n, i_mhc_cp, h_offset : h_offset + h_blk], b_bf16_ub[i_mhc_cp, 0:h_blk])
                            T.copy(d[pid_n, h_offset : h_offset + h_blk], d_bf16_ub[0:h_blk])

                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)

                        T.tile.cast(b_f32_ub, b_bf16_ub, "CAST_NONE", pad_mhc * pad_h_blk)
                        T.tile.cast(d_f32_ub, d_bf16_ub, "CAST_NONE", pad_h_blk)

                        for i_mhco in T.serial(mhc):
                            T.tile.mul(x_f32_ub[i_mhco, :], d_f32_ub, c_ub[i_mhco])
                            for i_mhci in T.serial(mhc):
                                T.tile.axpy(x_f32_ub[i_mhco, :], b_f32_ub[i_mhci, :], a_ub[i_mhci, i_mhco])

                        T.tile.cast(x_bf16_ub, x_f32_ub, "CAST_ROUND", pad_mhc * pad_h_blk)

                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)

                        for i_mhc_wp in T.serial(mhc):
                            T.copy(x_bf16_ub[i_mhc_wp, 0:h_blk], x[pid_n, i_mhc_wp, h_offset : h_offset + h_blk])

    return _mhc_post_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS_BWD)
def _mhc_post_bwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 0) -> tilelang.JITKernel:
    n = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden

    if h_blk == 0:
        h_blk = _compute_safe_h_blk(hidden, MAX_BWD_H_BLK)

    assert h % h_blk == 0, f"hidden={h} must be divisible by h_blk={h_blk}"

    pad_h_blk = T.ceildiv(h_blk, 16) * 16
    pad_mhc = _compute_pad_mhc(mhc)
    sub_blk_n = TOKEN_BLOCK_SIZE_BWD // VEC_NUM

    @T.prim_func
    def _mhc_post_bwd_kernel(
        dx: T.Tensor[(n, mhc, h), dbtype],
        a: T.Tensor[(n, mhc, mhc), dtype],
        b: T.Tensor[(n, mhc, h), dbtype],
        c: T.Tensor[(n, mhc), dtype],
        d: T.Tensor[(n, h), dbtype],
        da: T.Tensor[(n, mhc, mhc), dtype],
        db: T.Tensor[(n, mhc, h), dbtype],
        dc: T.Tensor[(n, mhc), dtype],
        dd: T.Tensor[(n, h), dbtype],
    ) -> None:
        num_cores = T.ceildiv(n, TOKEN_BLOCK_SIZE_BWD)
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((pad_mhc, pad_mhc), dtype)
            c_ub = T.alloc_ub((pad_mhc,), dtype)

            da_ub = T.alloc_ub((pad_mhc, pad_mhc), dtype)
            dc_ub = T.alloc_ub((pad_mhc,), dtype)

            dx_bf16_ub = T.alloc_ub((pad_mhc, pad_h_blk), dbtype)
            dx_f32_ub = T.alloc_ub((pad_mhc, pad_h_blk), dtype)

            b_bf16_ub = T.alloc_ub((pad_mhc, pad_h_blk), dbtype)
            b_f32_ub = T.alloc_ub((pad_mhc, pad_h_blk), dtype)

            d_bf16_ub = T.alloc_ub((pad_h_blk,), dbtype)
            d_f32_ub = T.alloc_ub((pad_h_blk,), dtype)

            db_f32_ub = T.alloc_ub((pad_mhc, pad_h_blk), dtype)
            db_bf16_ub = T.alloc_ub((pad_mhc, pad_h_blk), dbtype)

            dd_f32_ub = T.alloc_ub((pad_h_blk,), dtype)
            dd_bf16_ub = T.alloc_ub((pad_h_blk,), dbtype)

            prod_ub = T.alloc_ub((pad_h_blk,), dtype)
            reduce_elem_ub = T.alloc_ub((1,), dtype)

            for i_token in T.serial(sub_blk_n):
                pid_n = cid * TOKEN_BLOCK_SIZE_BWD + vid * sub_blk_n + i_token

                if pid_n < n:
                    with T.Scope("V"):
                        T.tile.fill(da_ub, 0.0)
                        T.tile.fill(dc_ub, 0.0)

                        T.copy(a[pid_n, 0:mhc, 0:mhc], a_ub[0:mhc, 0:mhc])
                        T.copy(c[pid_n, 0:mhc], c_ub[0:mhc])

                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)

                    for i0_h in T.serial(h // h_blk):
                        h_offset = i0_h * h_blk

                        with T.Scope("V"):
                            T.tile.fill(dx_bf16_ub, T.cast(0.0, "bfloat16"))
                            T.tile.fill(b_bf16_ub, T.cast(0.0, "bfloat16"))
                            T.tile.fill(d_bf16_ub, T.cast(0.0, "bfloat16"))
                            T.tile.fill(db_f32_ub, 0.0)
                            T.tile.fill(dd_f32_ub, 0.0)

                            for i_mhc_cp in T.serial(mhc):
                                T.copy(dx[pid_n, i_mhc_cp, h_offset : h_offset + h_blk], dx_bf16_ub[i_mhc_cp, 0:h_blk])
                                T.copy(b[pid_n, i_mhc_cp, h_offset : h_offset + h_blk], b_bf16_ub[i_mhc_cp, 0:h_blk])
                            T.copy(d[pid_n, h_offset : h_offset + h_blk], d_bf16_ub[0:h_blk])

                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)

                        T.tile.cast(dx_f32_ub, dx_bf16_ub, "CAST_NONE", pad_mhc * pad_h_blk)
                        T.tile.cast(b_f32_ub, b_bf16_ub, "CAST_NONE", pad_mhc * pad_h_blk)
                        T.tile.cast(d_f32_ub, d_bf16_ub, "CAST_NONE", pad_h_blk)

                        for i_mhci in T.serial(mhc):
                            for i_mhco in T.serial(mhc):
                                T.tile.axpy(db_f32_ub[i_mhci, :], dx_f32_ub[i_mhco, :], a_ub[i_mhci, i_mhco])

                        for i_mhci in T.serial(mhc):
                            for i_mhco in T.serial(mhc):
                                T.tile.mul(prod_ub, b_f32_ub[i_mhci, :], dx_f32_ub[i_mhco, :])
                                T.reduce_sum(prod_ub, reduce_elem_ub, dim=-1, clear=True)
                                da_ub[i_mhci, i_mhco] += reduce_elem_ub[0]

                        T.tile.mul(dd_f32_ub, dx_f32_ub[0, :], c_ub[0])
                        T.tile.mul(prod_ub, d_f32_ub, dx_f32_ub[0, :])
                        T.reduce_sum(prod_ub, reduce_elem_ub, dim=-1, clear=True)
                        dc_ub[0] += reduce_elem_ub[0]

                        for i_mhc_inner in T.serial(mhc - 1):
                            i_mhc = i_mhc_inner + 1
                            T.tile.axpy(dd_f32_ub, dx_f32_ub[i_mhc, :], c_ub[i_mhc])

                            T.tile.mul(prod_ub, d_f32_ub, dx_f32_ub[i_mhc, :])
                            T.reduce_sum(prod_ub, reduce_elem_ub, dim=-1, clear=True)
                            dc_ub[i_mhc] += reduce_elem_ub[0]

                        T.tile.cast(db_bf16_ub, db_f32_ub, "CAST_ROUND", pad_mhc * pad_h_blk)
                        T.tile.cast(dd_bf16_ub, dd_f32_ub, "CAST_ROUND", pad_h_blk)

                        T.set_flag("v", "mte3", 2)
                        T.wait_flag("v", "mte3", 2)

                        for i_mhc_wp in T.serial(mhc):
                            T.copy(db_bf16_ub[i_mhc_wp, 0:h_blk], db[pid_n, i_mhc_wp, h_offset : h_offset + h_blk])
                        T.copy(dd_bf16_ub[0:h_blk], dd[pid_n, h_offset : h_offset + h_blk])

                        T.set_flag("mte3", "v", 2)
                        T.wait_flag("mte3", "v", 2)

                    T.set_flag("v", "mte3", 3)
                    T.wait_flag("v", "mte3", 3)

                    for i_da_row in T.serial(mhc):
                        T.copy(da_ub[i_da_row, 0:mhc], da[pid_n, i_da_row, 0:mhc])
                    T.copy(dc_ub[0:mhc], dc[pid_n, 0:mhc])

                    T.set_flag("mte3", "v", 3)
                    T.wait_flag("mte3", "v", 3)

    return _mhc_post_bwd_kernel


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
    h_blk: int = 0,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f"{x.dtype=}"
    assert residual.dtype == torch.bfloat16, f"{residual.dtype=}"
    assert post_layer_mix.dtype == torch.float32, f"{post_layer_mix.dtype=}"
    assert comb_res_mix.dtype == torch.float32, f"{comb_res_mix.dtype=}"
    assert x.shape == (num_seqs, num_tokens, hidden), f"{x.shape=}"
    assert post_layer_mix.shape == (num_seqs, num_tokens, mhc, 1), f"{post_layer_mix.shape=}"
    assert comb_res_mix.shape == (num_seqs, num_tokens, mhc, mhc), f"{comb_res_mix.shape=}"
    assert mhc == 4

    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    residual = residual.contiguous()
    if out is None:
        out = torch.empty_like(residual)

    fwd_h_blk = h_blk if h_blk > 0 else _compute_safe_h_blk(hidden, MAX_FWD_H_BLK)

    kernel = _mhc_post_fwd(mhc, hidden, h_blk=fwd_h_blk)
    kernel(
        comb_res_mix.flatten(0, 1),
        residual.flatten(0, 1),
        post_layer_mix.flatten(0, 1).squeeze(-1),
        x.flatten(0, 1),
        out.flatten(0, 1),
    )
    return out


def mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
    fuse_grad_acc: bool = True,
    h_blk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]

    d_comb_res_mix_3d = torch.empty((n, mhc, mhc), dtype=comb_res_mix.dtype, device=comb_res_mix.device)
    d_residual_3d = torch.empty((n, mhc, h), dtype=residual.dtype, device=residual.device)
    d_post_layer_mix_2d = torch.empty((n, mhc), dtype=post_layer_mix.dtype, device=post_layer_mix.device)
    d_x_2d = torch.empty((n, h), dtype=x.dtype, device=x.device)

    bwd_h_blk = h_blk if h_blk > 0 else _compute_safe_h_blk(h, MAX_BWD_H_BLK)

    bwd_kernel = _mhc_post_bwd(mhc, h, h_blk=bwd_h_blk)

    bwd_kernel(
        d_o.contiguous().view(n, mhc, h),
        comb_res_mix.contiguous().view(n, mhc, mhc),
        residual.contiguous().view(n, mhc, h),
        post_layer_mix.contiguous().view(n, mhc),
        x.contiguous().view(n, h),
        d_comb_res_mix_3d,
        d_residual_3d,
        d_post_layer_mix_2d,
        d_x_2d,
    )

    d_residual = d_residual_3d.view_as(residual)
    if fuse_grad_acc:
        residual.grad_from_mhc_post = d_residual

    return (
        d_x_2d.view_as(x),
        d_residual,
        d_post_layer_mix_2d.view_as(post_layer_mix),
        d_comb_res_mix_3d.view_as(comb_res_mix),
    )


def mhc_post_fwd_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = residual.shape
    residual_f = residual.float()
    x_f = x.float()
    c = post_layer_mix.squeeze(-1)
    a = comb_res_mix
    out = c.unsqueeze(-1) * x_f.unsqueeze(2) + torch.matmul(a.transpose(-1, -2), residual_f)
    return out.to(torch.bfloat16)


def mhc_post_bwd_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]

    d_o_f = d_o.float().view(n, mhc, h)
    a = comb_res_mix.view(n, mhc, mhc)
    b = residual.view(n, mhc, h).float()
    c = post_layer_mix.view(n, mhc)
    d = x.view(n, h).float()
    dx = d_o_f

    da = torch.bmm(b, dx.transpose(-1, -2))
    db = torch.bmm(a, dx).to(torch.bfloat16)
    dc = torch.bmm(d.unsqueeze(1), dx.transpose(-1, -2)).squeeze(1)
    dd = (c.unsqueeze(-1) * dx).sum(dim=1).to(torch.bfloat16)

    return dd.view_as(x), db.view_as(residual), dc.view_as(post_layer_mix.squeeze(-1)), da.view_as(comb_res_mix)


def test_fwd():
    device = "npu"
    configs = [
        (8192, 7168, 4),
    ]

    for num_tokens, hidden_size, mhc_mult in configs:
        torch.manual_seed(42)
        x = torch.randn((1, num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        residual = torch.randn((1, num_tokens, mhc_mult, hidden_size), dtype=torch.bfloat16, device=device)
        post_layer_mix = torch.randn((1, num_tokens, mhc_mult, 1), dtype=torch.float32, device=device)
        comb_res_mix = torch.randn((1, num_tokens, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

        out_ref = mhc_post_fwd_ref(x, residual, post_layer_mix, comb_res_mix)
        out_tl = mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix)

        torch.testing.assert_close(out_tl, out_ref, atol=4e-2, rtol=1e-2)
        print("Kernel Output Match!")


def test_bwd():
    device = "npu"
    configs = [
        (8192, 7168, 4),
    ]

    for num_tokens, hidden_size, mhc_mult in configs:
        torch.manual_seed(42)
        x = torch.randn((1, num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        residual = torch.randn((1, num_tokens, mhc_mult, hidden_size), dtype=torch.bfloat16, device=device)
        post_layer_mix = torch.randn((1, num_tokens, mhc_mult, 1), dtype=torch.float32, device=device)
        comb_res_mix = torch.randn((1, num_tokens, mhc_mult, mhc_mult), dtype=torch.float32, device=device)
        d_o = torch.randn((1, num_tokens, mhc_mult, hidden_size), dtype=torch.bfloat16, device=device)

        dd_ref, db_ref, dc_ref, da_ref = mhc_post_bwd_ref(x, residual, post_layer_mix, comb_res_mix, d_o)
        dd_tl, db_tl, dc_tl, da_tl = mhc_post_bwd(x, residual, post_layer_mix, comb_res_mix, d_o, fuse_grad_acc=False)

        torch.testing.assert_close(da_tl, da_ref, atol=4e-2, rtol=1e-2)
        torch.testing.assert_close(db_tl, db_ref, atol=4e-2, rtol=1e-2)
        torch.testing.assert_close(dc_tl.squeeze(-1), dc_ref, atol=4e-2, rtol=1e-2)
        torch.testing.assert_close(dd_tl, dd_ref, atol=4e-2, rtol=1e-2)
        print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
