import math
import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_split_mixes_fwd(
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
) -> tilelang.JITKernel:
    num_tokens = T.symbolic('num_tokens')
    dtype = "float32"
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    sub_mhc = mhc_mult // VEC_NUM
    sub_mhc2 = mhc_mult2 // VEC_NUM

    assert mhc_mult % VEC_NUM == 0
    assert mhc_mult2 % VEC_NUM == 0

    pad_block = T.ceildiv(token_block_size, 8) * 8
    pad_sub_mhc = T.ceildiv(sub_mhc, 8) * 8
    pad_sub_mhc2 = T.ceildiv(sub_mhc2, 8) * 8

    @T.prim_func
    def mhc_pre_split_mixes_fwd_kernel(
        input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype],
        mhc_scale: T.Tensor[(3,), dtype],
        mhc_base: T.Tensor[mhc_mult3, dtype],
        pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size), is_npu=True) as (cid, vid):
            row_start = cid * token_block_size
            cur_block_tokens = T.min(token_block_size, num_tokens - row_start)

            mhc_start = vid * sub_mhc
            mhc2_start = vid * sub_mhc2

            inSlice0_ub = T.alloc_ub((pad_block, pad_sub_mhc), dtype)
            inSlice1_ub = T.alloc_ub((pad_block, pad_sub_mhc), dtype)
            inSlice2_ub = T.alloc_ub((pad_block, pad_sub_mhc2), dtype)

            base_slice0_ub = T.alloc_ub((pad_sub_mhc,), dtype)
            base_slice1_ub = T.alloc_ub((pad_sub_mhc,), dtype)
            base_slice2_ub = T.alloc_ub((pad_sub_mhc2,), dtype)

            pre_out_ub = T.alloc_ub((pad_block, pad_sub_mhc), dtype)
            post_out_ub = T.alloc_ub((pad_block, pad_sub_mhc), dtype)
            comb_out_ub = T.alloc_ub((pad_block, pad_sub_mhc2), dtype)

            scale_ub = T.alloc_ub((8,), dtype)

            T.tile.fill(inSlice0_ub, 0.0)
            T.tile.fill(inSlice1_ub, 0.0)
            T.tile.fill(inSlice2_ub, 0.0)
            T.tile.fill(base_slice0_ub, 0.0)
            T.tile.fill(base_slice1_ub, 0.0)
            T.tile.fill(base_slice2_ub, 0.0)

            T.copy(mhc_scale[0:3], scale_ub[0:3])
            T.copy(mhc_base[mhc_start : mhc_start + sub_mhc], base_slice0_ub[0:sub_mhc])
            T.copy(mhc_base[mhc_mult + mhc_start : mhc_mult + mhc_start + sub_mhc], base_slice1_ub[0:sub_mhc])
            T.copy(mhc_base[mhc_mult * 2 + mhc2_start : mhc_mult * 2 + mhc2_start + sub_mhc2], base_slice2_ub[0:sub_mhc2])
            T.copy(input_mixes[row_start : row_start + cur_block_tokens, mhc_start : mhc_start + sub_mhc], inSlice0_ub[0:cur_block_tokens, 0:sub_mhc])
            T.copy(input_mixes[row_start : row_start + cur_block_tokens, mhc_mult + mhc_start : mhc_mult + mhc_start + sub_mhc], inSlice1_ub[0:cur_block_tokens, 0:sub_mhc])
            T.copy(input_mixes[row_start : row_start + cur_block_tokens, mhc_mult * 2 + mhc2_start : mhc_mult * 2 + mhc2_start + sub_mhc2], inSlice2_ub[0:cur_block_tokens, 0:sub_mhc2])

            T.tile.broadcast(pre_out_ub, base_slice0_ub)
            T.tile.axpy(pre_out_ub, inSlice0_ub, scale_ub[0])
            T.tile.sigmoid(pre_out_ub, pre_out_ub)
            T.tile.add(pre_out_ub, pre_out_ub, mhc_pre_eps)

            T.tile.broadcast(post_out_ub, base_slice1_ub)
            T.tile.axpy(post_out_ub, inSlice1_ub, scale_ub[1])
            T.tile.sigmoid(post_out_ub, post_out_ub)
            T.tile.mul(post_out_ub, post_out_ub, mhc_post_mult_value)

            T.tile.broadcast(comb_out_ub, base_slice2_ub)
            T.tile.axpy(comb_out_ub, inSlice2_ub, scale_ub[2])

            T.copy(pre_out_ub[0:cur_block_tokens, 0:sub_mhc], pre_layer_mix[row_start : row_start + cur_block_tokens, mhc_start : mhc_start + sub_mhc])
            T.copy(post_out_ub[0:cur_block_tokens, 0:sub_mhc], post_layer_mix[row_start : row_start + cur_block_tokens, mhc_start : mhc_start + sub_mhc])
            T.copy(comb_out_ub[0:cur_block_tokens, 0:sub_mhc2], comb_res_mix[row_start : row_start + cur_block_tokens, mhc2_start : mhc2_start + sub_mhc2])

    return mhc_pre_split_mixes_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_split_mixes_bwd(
    mhc_mult: int,
    mhc_post_mult_value: float,
    token_block_size: int,
    partial_size: int,
) -> tilelang.JITKernel:
    num_tokens = T.symbolic('num_tokens')
    dtype = "float32"
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    sub_block_tokens = token_block_size // VEC_NUM
    pad_sub_block = T.ceildiv(sub_block_tokens, 8) * 8
    pad_mhc = T.ceildiv(mhc_mult, 8) * 8
    pad_mhc2 = T.ceildiv(mhc_mult2, 8) * 8
    pad_mhc3 = T.ceildiv(mhc_mult3, 8) * 8

    inv_post_mult = 1.0 / mhc_post_mult_value
    grid_size = T.ceildiv(num_tokens, token_block_size)

    @T.prim_func
    def mhc_pre_split_mixes_bwd_kernel(
        pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
        post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
        comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), dtype],
        input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype],
        post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        mhc_scale: T.Tensor[(3,), dtype],
        mhc_base: T.Tensor[mhc_mult3, dtype],
        input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), dtype],
        mhc_scale_grad_partial: T.Tensor[(partial_size, 3), dtype],
        mhc_base_grad_partial: T.Tensor[(partial_size, mhc_mult3), dtype],
    ) -> None:
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            partial_idx = cid * VEC_NUM + vid
            row_start = cid * token_block_size + vid * sub_block_tokens
            cur_tokens = T.min(sub_block_tokens, num_tokens - row_start)

            grad_in_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            fwd_out_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            x_in_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)

            comb_grad_ub = T.alloc_ub((pad_sub_block, pad_mhc2), dtype)
            in_slice2_ub = T.alloc_ub((pad_sub_block, pad_mhc2), dtype)

            sig_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            one_minus_sig_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            grad_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            tmp_scale_ub = T.alloc_ub((pad_sub_block, pad_mhc), dtype)
            tmp_scale2_ub = T.alloc_ub((pad_sub_block, pad_mhc2), dtype)

            scale_ub = T.alloc_ub((8,), dtype)
            base_slice0_ub = T.alloc_ub((pad_mhc,), dtype)
            base_slice1_ub = T.alloc_ub((pad_mhc,), dtype)
            base_slice2_ub = T.alloc_ub((pad_mhc2,), dtype)

            scale_grad_accum = T.alloc_ub((8,), dtype)
            base_grad0_accum = T.alloc_ub((pad_mhc,), dtype)
            base_grad1_accum = T.alloc_ub((pad_mhc,), dtype)
            base_grad2_accum = T.alloc_ub((pad_mhc2,), dtype)

            base_grad_out_mhc = T.alloc_ub((pad_mhc,), dtype)
            base_grad_out_mhc2 = T.alloc_ub((pad_mhc2,), dtype)
            reduce_scale_mhc = T.alloc_ub((pad_mhc,), dtype)
            reduce_scale_mhc2 = T.alloc_ub((pad_mhc2,), dtype)

            T.tile.fill(scale_grad_accum, 0.0)
            T.tile.fill(base_grad0_accum, 0.0)
            T.tile.fill(base_grad1_accum, 0.0)
            T.tile.fill(base_grad2_accum, 0.0)
            T.tile.fill(scale_ub, 0.0)
            T.tile.fill(base_slice0_ub, 0.0)
            T.tile.fill(base_slice1_ub, 0.0)
            T.tile.fill(base_slice2_ub, 0.0)

            T.copy(mhc_scale[0:3], scale_ub[0:3])
            T.copy(mhc_base[0:mhc_mult], base_slice0_ub[0:mhc_mult])
            T.copy(mhc_base[mhc_mult : mhc_mult * 2], base_slice1_ub[0:mhc_mult])
            T.copy(mhc_base[mhc_mult * 2 : mhc_mult3], base_slice2_ub[0:mhc_mult2])

            T.tile.fill(grad_in_ub, 0.0)
            T.tile.fill(fwd_out_ub, 0.0)

            T.copy(pre_layer_mix_grad[row_start : row_start + cur_tokens, 0:mhc_mult], grad_in_ub[0:cur_tokens, 0:mhc_mult])
            T.copy(input_mixes[row_start : row_start + cur_tokens, 0:mhc_mult], fwd_out_ub[0:cur_tokens, 0:mhc_mult])

            T.tile.broadcast(sig_ub, base_slice0_ub)
            T.tile.axpy(sig_ub, fwd_out_ub, scale_ub[0])
            T.tile.sigmoid(sig_ub, sig_ub)
            T.tile.fill(one_minus_sig_ub, 1.0)
            T.tile.sub(one_minus_sig_ub, one_minus_sig_ub, sig_ub)
            T.tile.mul(grad_ub, grad_in_ub, sig_ub)
            T.tile.mul(grad_ub, grad_ub, one_minus_sig_ub)
            T.reduce_sum(grad_ub, base_grad_out_mhc, dim=0)
            T.tile.add(base_grad0_accum, base_grad0_accum, base_grad_out_mhc)
            T.tile.mul(tmp_scale_ub, grad_ub, fwd_out_ub)
            T.reduce_sum(tmp_scale_ub, reduce_scale_mhc, dim=0)
            T.tile.mul(fwd_out_ub, grad_ub, scale_ub[0])

            for j in T.serial(mhc_mult):
                scale_grad_accum[0] += reduce_scale_mhc[j]

            T.copy(fwd_out_ub[0:cur_tokens, 0:mhc_mult], input_mixes_grad[row_start : row_start + cur_tokens, 0:mhc_mult])

            T.tile.fill(grad_in_ub, 0.0)
            T.tile.fill(fwd_out_ub, 0.0)
            T.tile.fill(x_in_ub, 0.0)

            T.copy(post_layer_mix_grad[row_start : row_start + cur_tokens, 0:mhc_mult], grad_in_ub[0:cur_tokens, 0:mhc_mult])
            T.copy(post_layer_mix[row_start : row_start + cur_tokens, 0:mhc_mult], fwd_out_ub[0:cur_tokens, 0:mhc_mult])
            T.copy(input_mixes[row_start : row_start + cur_tokens, mhc_mult : mhc_mult * 2], x_in_ub[0:cur_tokens, 0:mhc_mult])

            T.tile.mul(sig_ub, fwd_out_ub, inv_post_mult)
            T.tile.fill(one_minus_sig_ub, 1.0)
            T.tile.sub(one_minus_sig_ub, one_minus_sig_ub, sig_ub)
            T.tile.mul(grad_ub, grad_in_ub, fwd_out_ub)
            T.tile.mul(grad_ub, grad_ub, one_minus_sig_ub)
            T.reduce_sum(grad_ub, base_grad_out_mhc, dim=0)
            T.tile.add(base_grad1_accum, base_grad1_accum, base_grad_out_mhc)
            T.tile.mul(tmp_scale_ub, grad_ub, x_in_ub)
            T.reduce_sum(tmp_scale_ub, reduce_scale_mhc, dim=0)
            T.tile.mul(x_in_ub, grad_ub, scale_ub[1])

            for j in T.serial(mhc_mult):
                scale_grad_accum[1] += reduce_scale_mhc[j]

            T.copy(x_in_ub[0:cur_tokens, 0:mhc_mult], input_mixes_grad[row_start : row_start + cur_tokens, mhc_mult : mhc_mult * 2])

            T.tile.fill(comb_grad_ub, 0.0)
            T.tile.fill(in_slice2_ub, 0.0)

            T.copy(comb_res_mix_grad[row_start : row_start + cur_tokens, 0:mhc_mult2], comb_grad_ub[0:cur_tokens, 0:mhc_mult2])
            T.copy(input_mixes[row_start : row_start + cur_tokens, mhc_mult * 2 : mhc_mult3], in_slice2_ub[0:cur_tokens, 0:mhc_mult2])

            T.reduce_sum(comb_grad_ub, base_grad_out_mhc2, dim=0)
            T.tile.add(base_grad2_accum, base_grad2_accum, base_grad_out_mhc2)
            T.tile.mul(tmp_scale2_ub, comb_grad_ub, in_slice2_ub)
            T.reduce_sum(tmp_scale2_ub, reduce_scale_mhc2, dim=0)
            T.tile.mul(comb_grad_ub, comb_grad_ub, scale_ub[2])

            for j in T.serial(mhc_mult2):
                scale_grad_accum[2] += reduce_scale_mhc2[j]

            T.copy(comb_grad_ub[0:cur_tokens, 0:mhc_mult2], input_mixes_grad[row_start : row_start + cur_tokens, mhc_mult * 2 : mhc_mult3])
            T.copy(base_grad0_accum[0:mhc_mult], mhc_base_grad_partial[partial_idx, 0:mhc_mult])
            T.copy(base_grad1_accum[0:mhc_mult], mhc_base_grad_partial[partial_idx, mhc_mult : mhc_mult * 2])
            T.copy(base_grad2_accum[0:mhc_mult2], mhc_base_grad_partial[partial_idx, mhc_mult * 2 : mhc_mult3])
            T.copy(scale_grad_accum[0:3], mhc_scale_grad_partial[partial_idx, 0:3])

    return mhc_pre_split_mixes_bwd_kernel


def mhc_pre_split_mixes_fwd_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ],
    )
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def test_fwd():
    device = "npu"
    configs = [
        (1, 8192, 4),
    ]

    for n0, n1, mhc_mult in configs:
        mhc_mult2 = mhc_mult * mhc_mult
        mhc_mult3 = mhc_mult * 2 + mhc_mult2

        torch.manual_seed(42)
        input_mixes = torch.randn(n0, n1, mhc_mult3, dtype=torch.float32, device=device)
        mhc_scale = torch.randn(3, dtype=torch.float32, device=device)
        mhc_base = torch.randn(mhc_mult3, dtype=torch.float32, device=device)

        pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_fwd_ref(
            input_mixes, mhc_scale, mhc_base, mhc_mult, 2.0, 1e-2
        )

        pre_layer_mix = torch.empty(n0 * n1, mhc_mult, dtype=torch.float32, device=device)
        post_layer_mix = torch.empty(n0 * n1, mhc_mult, dtype=torch.float32, device=device)
        comb_res_mix = torch.empty(n0 * n1, mhc_mult2, dtype=torch.float32, device=device)

        fwd_kernel = _mhc_pre_split_mixes_fwd(mhc_mult, 2.0, 1e-2, token_block_size=128)
        fwd_kernel(
            input_mixes.view(-1, mhc_mult3),
            mhc_scale,
            mhc_base,
            pre_layer_mix,
            post_layer_mix,
            comb_res_mix,
        )

        pre_tl = pre_layer_mix.view(n0, n1, mhc_mult, 1)
        post_tl = post_layer_mix.view(n0, n1, mhc_mult, 1)
        comb_tl = comb_res_mix.view(n0, n1, mhc_mult, mhc_mult)

        torch.testing.assert_close(pre_tl, pre_ref, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(post_tl, post_ref, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(comb_tl, comb_ref, rtol=1e-5, atol=2e-5)
        print("Kernel Output Match!")


def test_bwd():
    device = "npu"
    mhc_mult = 4
    n0 = 1
    n1 = 8192
    mhc_post_mult_value = 2.0
    mhc_pre_eps = 1e-2
    token_block_size = 128

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    num_tokens = n0 * n1

    bwd_grid_size = math.ceil(num_tokens / token_block_size)
    partial_size = bwd_grid_size * VEC_NUM

    torch.manual_seed(42)
    input_mixes_arg = torch.randn(n0, n1, mhc_mult3, dtype=torch.float32, device=device, requires_grad=True)
    mhc_scale_arg = torch.randn(3, dtype=torch.float32, device=device, requires_grad=True)
    mhc_base_arg = torch.randn(mhc_mult3, dtype=torch.float32, device=device, requires_grad=True)

    pre_layer_mix_grad_arg = torch.randn(n0, n1, mhc_mult, 1, dtype=torch.float32, device=device)
    post_layer_mix_grad_arg = torch.randn(n0, n1, mhc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix_grad_arg = torch.randn(n0, n1, mhc_mult, mhc_mult, dtype=torch.float32, device=device)

    pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_fwd_ref(
        input_mixes_arg, mhc_scale_arg, mhc_base_arg, mhc_mult, mhc_post_mult_value, mhc_pre_eps
    )
    torch.autograd.backward(
        [pre_ref, post_ref, comb_ref],
        [pre_layer_mix_grad_arg, post_layer_mix_grad_arg, comb_res_mix_grad_arg],
    )
    input_mixes_grad_ref = input_mixes_arg.grad.clone()
    mhc_scale_grad_ref = mhc_scale_arg.grad.clone()
    mhc_base_grad_ref = mhc_base_arg.grad.clone()

    input_mixes_arg.grad = None
    mhc_scale_arg.grad = None
    mhc_base_arg.grad = None

    input_mixes_flat = input_mixes_arg.detach().view(-1, mhc_mult3).contiguous()
    mhc_scale_flat = mhc_scale_arg.detach().contiguous()
    mhc_base_flat = mhc_base_arg.detach().contiguous()

    pre_layer_mix_out = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=device)
    post_layer_mix_out = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=device)
    comb_res_mix_out = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=device)

    fwd_kernel = _mhc_pre_split_mixes_fwd(mhc_mult, mhc_post_mult_value, mhc_pre_eps, token_block_size=token_block_size)
    fwd_kernel(input_mixes_flat, mhc_scale_flat, mhc_base_flat, pre_layer_mix_out, post_layer_mix_out, comb_res_mix_out)

    pre_grad_flat = pre_layer_mix_grad_arg.view(-1, mhc_mult).contiguous()
    post_grad_flat = post_layer_mix_grad_arg.view(-1, mhc_mult).contiguous()
    comb_grad_flat = comb_res_mix_grad_arg.view(-1, mhc_mult2).contiguous()

    input_mixes_grad = torch.empty(num_tokens, mhc_mult3, dtype=torch.float32, device=device)
    mhc_scale_grad_partial = torch.empty(partial_size, 3, dtype=torch.float32, device=device)
    mhc_base_grad_partial = torch.empty(partial_size, mhc_mult3, dtype=torch.float32, device=device)

    bwd_kernel = _mhc_pre_split_mixes_bwd(mhc_mult, mhc_post_mult_value, token_block_size=token_block_size, partial_size=partial_size)
    bwd_kernel(
        pre_grad_flat, post_grad_flat, comb_grad_flat,
        input_mixes_flat, post_layer_mix_out, mhc_scale_flat, mhc_base_flat,
        input_mixes_grad, mhc_scale_grad_partial, mhc_base_grad_partial,
    )

    mhc_scale_grad_tl = mhc_scale_grad_partial.sum(0)
    mhc_base_grad_tl = mhc_base_grad_partial.sum(0)

    torch.testing.assert_close(input_mixes_grad, input_mixes_grad_ref.view(-1, mhc_mult3), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_scale_grad_tl, mhc_scale_grad_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_base_grad_tl, mhc_base_grad_ref, rtol=1e-5, atol=2e-5)
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
