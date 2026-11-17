import math
import tilelang
import torch


def _check_precision(actual, golden, dtype="float32"):
    configs = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1e0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual, golden = actual.detach().cpu(), golden.detach().cpu()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatch = (actual != golden).sum().item()
        total = max(actual.numel(), 1)
        return mismatch == 0, 1.0 - mismatch / total, 0.0 if mismatch == 0 else float("inf")
    atol, rtol, max_abs_limit, required_ratio = configs[dtype]
    actual, golden = actual.float(), golden.float()
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
    ratio = (error <= atol + rtol * golden[finite].abs()).float().mean().item()
    maximum = error.max().item()
    return ratio >= required_ratio and maximum <= max_abs_limit, ratio, maximum


from tilelang import language as T

_FWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_BWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

VEC_NUM = 2


@tilelang.jit(pass_configs=_FWD_PASS_CONFIGS)
def _mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    reshape_factor: int = 1,
    token_block_size: int = 512,
) -> tilelang.JITKernel:
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    grid_size = T.ceildiv(num_tokens, token_block_size)
    pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
    sub_block_tokens = token_block_size // VEC_NUM
    orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

    @T.prim_func
    def mhc_head_compute_mix_fwd_kernel(
        input_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        mhc_scale: T.Tensor[(1,), dtype],
        mhc_base: T.Tensor[(orig_mhc_mult,), dtype],
        output_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
    ) -> None:
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            row_start = cid * token_block_size + vid * sub_block_tokens
            in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            out_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            scale_ub = T.alloc_ub((1,), dtype)
            base_ub = T.alloc_ub((pad_mhc_mult,), dtype)

            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)
            T.copy(input_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult], in_ub[0:sub_block_tokens, 0:mhc_mult])
            T.copy(mhc_scale[0:1], scale_ub)
            T.copy(mhc_base[0:orig_mhc_mult], base_ub[0:orig_mhc_mult])
            T.set_flag("mte2", "v", 0)

            T.wait_flag("mte2", "v", 0)

            for r in range(1, reshape_factor):
                for j in range(orig_mhc_mult):
                    base_ub[r * orig_mhc_mult + j] = base_ub[j]

            T.tile.broadcast(bcast_ub, base_ub, axis=0)
            T.tile.axpy(bcast_ub, in_ub, scale_ub[0])

            # sigmoid computation (in bcast_ub instead of in_ub)
            T.tile.sigmoid(out_ub, bcast_ub)
            T.tile.add(out_ub, out_ub, mhc_pre_eps)

            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(out_ub[0:sub_block_tokens, 0:mhc_mult], output_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult])
            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)

    return mhc_head_compute_mix_fwd_kernel


@tilelang.jit(pass_configs=_BWD_PASS_CONFIGS)
def _mhc_head_compute_mix_bwd(
    mhc_mult: int,
    reshape_factor: int = 1,
    token_block_size: int = 128,
    partial_size: int = 128,
) -> tilelang.JITKernel:
    num_tokens = T.symbolic("num_tokens")
    dtype = "float32"
    pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
    sub_block_tokens = token_block_size // VEC_NUM
    grid_size = T.ceildiv(num_tokens, token_block_size)
    orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

    @T.prim_func
    def mhc_head_compute_mix_bwd_kernel(
        output_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
        input_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
        mhc_scale: T.Tensor[(1,), dtype],
        mhc_base: T.Tensor[(orig_mhc_mult,), dtype],
        input_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
        mhc_scale_grad_partial: T.Tensor[(partial_size, mhc_mult), dtype],
        mhc_base_grad_partial: T.Tensor[(partial_size, mhc_mult), dtype],
    ) -> None:
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            row_start = cid * token_block_size + vid * sub_block_tokens

            base_grad_ub = T.alloc_ub((1, pad_mhc_mult), dtype)
            reduce_col_ub = T.alloc_ub((1, pad_mhc_mult), dtype)
            scale_grad_ub = T.alloc_ub((1, pad_mhc_mult), dtype)

            in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            buf_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            val_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            sig_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
            scale_ub = T.alloc_ub((1,), dtype)
            base_ub = T.alloc_ub((pad_mhc_mult,), dtype)

            T.tile.fill(base_grad_ub, 0.0)
            T.tile.fill(scale_grad_ub, 0.0)

            T.set_flag("mte3", "mte2", 0)
            T.wait_flag("mte3", "mte2", 0)
            T.copy(input_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult], in_ub[0:sub_block_tokens, 0:mhc_mult])
            T.copy(output_mix_grad[row_start : row_start + sub_block_tokens, 0:mhc_mult], buf_ub[0:sub_block_tokens, 0:mhc_mult])
            T.copy(mhc_scale[0:1], scale_ub)
            T.copy(mhc_base[0:orig_mhc_mult], base_ub[0:orig_mhc_mult])

            T.set_flag("mte2", "v", 0)

            T.wait_flag("mte2", "v", 0)

            for r in range(1, reshape_factor):
                for j in range(orig_mhc_mult):
                    base_ub[r * orig_mhc_mult + j] = base_ub[j]

            T.tile.broadcast(bcast_ub, base_ub, axis=0)
            T.tile.axpy(bcast_ub, in_ub, scale_ub[0])

            T.tile.sigmoid(sig_ub, bcast_ub)

            T.tile.fill(val_ub, 1.0)
            T.tile.sub(val_ub, val_ub, sig_ub)
            T.tile.mul(sig_ub, sig_ub, val_ub)

            T.tile.mul(sig_ub, sig_ub, buf_ub)

            T.reduce_sum(sig_ub, reduce_col_ub, dim=0)
            T.tile.add(base_grad_ub, base_grad_ub, reduce_col_ub)

            T.tile.mul(buf_ub, sig_ub, scale_ub[0])

            T.tile.mul(bcast_ub, sig_ub, in_ub)
            T.reduce_sum(bcast_ub, reduce_col_ub, dim=0)
            T.tile.add(scale_grad_ub, scale_grad_ub, reduce_col_ub)

            T.set_flag("v", "mte3", 0)

            T.wait_flag("v", "mte3", 0)
            partial_idx = cid * VEC_NUM + vid
            T.copy(base_grad_ub[0, 0:mhc_mult], mhc_base_grad_partial[partial_idx, 0:mhc_mult])
            T.copy(scale_grad_ub[0, 0:mhc_mult], mhc_scale_grad_partial[partial_idx, 0:mhc_mult])
            T.copy(buf_ub[0:sub_block_tokens, 0:mhc_mult], input_mix_grad[row_start : row_start + sub_block_tokens, 0:mhc_mult])
            T.set_flag("mte3", "mte2", 0)

            T.wait_flag("mte3", "mte2", 0)

    return mhc_head_compute_mix_bwd_kernel


def mhc_head_compute_mix_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


_RESHAPE_FACTOR = 4


def test_fwd():
    n = 8192
    mhc_mult = 4
    reshape_mhc_mult = mhc_mult * _RESHAPE_FACTOR
    mhc_pre_eps = 1e-2
    fwd_token_block_size = 512 // _RESHAPE_FACTOR

    device = "npu"
    print(f"[Reshape] Running forward test on device: {device}, n={n}, reshape_mhc_mult={reshape_mhc_mult}")

    torch.manual_seed(42)
    input_mix = torch.randn((n, mhc_mult), dtype=torch.float32, device=device)
    mhc_scale = torch.randn((1,), dtype=torch.float32, device=device)
    mhc_base = torch.randn((mhc_mult,), dtype=torch.float32, device=device)

    input_reshaped = input_mix.reshape(-1, reshape_mhc_mult)
    output_reshaped = torch.empty_like(input_reshaped)

    fwd_func = _mhc_head_compute_mix_fwd(
        reshape_mhc_mult, mhc_pre_eps, reshape_factor=_RESHAPE_FACTOR, token_block_size=fwd_token_block_size
    )
    fwd_func(input_reshaped, mhc_scale, mhc_base, output_reshaped)

    output_tl = output_reshaped.reshape(-1, mhc_mult)
    ref_output = mhc_head_compute_mix_ref(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
    passed, ratio, max_abs = _check_precision(output_tl, ref_output)
    assert passed, f"forward: matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"

    print("Kernel Output Match!")


def test_bwd():
    n = 8192
    mhc_mult = 4
    reshape_mhc_mult = mhc_mult * _RESHAPE_FACTOR
    token_block_size = 128

    device = "npu"
    print(f"[Reshape] Running backward test on device: {device}, n={n}, reshape_mhc_mult={reshape_mhc_mult}")

    torch.manual_seed(42)
    input_mix = torch.randn((n, mhc_mult), dtype=torch.float32, device=device, requires_grad=True)
    mhc_scale = torch.randn((1,), dtype=torch.float32, device=device, requires_grad=True)
    mhc_base = torch.randn((mhc_mult,), dtype=torch.float32, device=device, requires_grad=True)
    output_mix_grad = torch.randn((n, mhc_mult), dtype=torch.float32, device=device)

    input_reshaped_ref = input_mix.reshape(-1, reshape_mhc_mult)
    base_bcast = mhc_base.repeat(_RESHAPE_FACTOR)
    ref_forward = torch.sigmoid((base_bcast + input_reshaped_ref * mhc_scale))
    ref_forward.backward(output_mix_grad.reshape_as(ref_forward))
    input_reshaped = input_mix.detach().reshape(-1, reshape_mhc_mult)
    output_grad_reshaped = output_mix_grad.reshape(-1, reshape_mhc_mult)

    reshaped_num_tokens = n // _RESHAPE_FACTOR
    grid_size = math.ceil(reshaped_num_tokens / token_block_size)
    partial_size = grid_size * VEC_NUM

    input_grad_reshaped = torch.empty_like(input_reshaped)
    mhc_scale_grad_partial = torch.empty((partial_size, reshape_mhc_mult), dtype=torch.float32, device=device)
    mhc_base_grad_partial = torch.empty((partial_size, reshape_mhc_mult), dtype=torch.float32, device=device)

    bwd_func = _mhc_head_compute_mix_bwd(
        reshape_mhc_mult, reshape_factor=_RESHAPE_FACTOR, token_block_size=token_block_size, partial_size=partial_size
    )
    bwd_func(output_grad_reshaped, input_reshaped, mhc_scale, mhc_base, input_grad_reshaped, mhc_scale_grad_partial, mhc_base_grad_partial)
    input_grad_tl_result = input_grad_reshaped.reshape(-1, mhc_mult)

    base_grad_partial_clean = mhc_base_grad_partial[:, 0:reshape_mhc_mult]
    scale_grad_partial_clean = mhc_scale_grad_partial[:, 0:reshape_mhc_mult]

    scale_grad_tl_result = scale_grad_partial_clean.sum().reshape(1)
    base_grad_tl_result = base_grad_partial_clean.sum(dim=0).reshape(_RESHAPE_FACTOR, mhc_mult).sum(dim=0)
    for name, actual, golden in (
        ("input_grad", input_grad_tl_result, input_mix.grad),
        ("scale_grad", scale_grad_tl_result, mhc_scale.grad),
        ("base_grad", base_grad_tl_result, mhc_base.grad),
    ):
        passed, ratio, max_abs = _check_precision(actual, golden)
        assert passed, f"{name}: matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Kernel Output Match!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
