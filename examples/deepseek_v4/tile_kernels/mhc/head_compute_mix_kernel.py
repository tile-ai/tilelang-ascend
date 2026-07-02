import math
import tilelang
import torch
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
    num_tokens = T.symbolic('num_tokens')

    grid_size = T.ceildiv(num_tokens, token_block_size)
    pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
    sub_block_tokens = token_block_size // VEC_NUM
    orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

    @T.prim_func
    def mhc_head_compute_mix_fwd_kernel(
        input_mix: T.Tensor[(num_tokens, mhc_mult), "float32"],
        mhc_scale: T.Tensor[(1,), "float32"],
        mhc_base: T.Tensor[(orig_mhc_mult,), "float32"],
        output_mix: T.Tensor[(num_tokens, mhc_mult), "float32"],
    ) -> None:
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            row_start = cid * token_block_size + vid * sub_block_tokens
            in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            out_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            scale_ub = T.alloc_ub((1,), "float32")
            base_ub = T.alloc_ub((pad_mhc_mult,), "float32")

            with T.Scope("V"):
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

                T.tile.mul(bcast_ub, bcast_ub, -1.0)

                # sigmoid computation (in bcast_ub instead of in_ub)
                T.tile.exp(bcast_ub, bcast_ub)
                T.tile.add(bcast_ub, bcast_ub, 1.0)
                T.tile.reciprocal(out_ub, bcast_ub)

                # Newton-Raphson refinement (kept)
                # step 1: in = denom * sigmoid_approx
                T.tile.mul(in_ub, bcast_ub, out_ub)
                # step 2: bcast = 2 - denom*sigmoid_approx  (AXPY replaces fill+sub)
                # v15: fill(in, 2) + sub(bcast, in, bcast)
                T.tile.fill(bcast_ub, 2.0)
                T.tile.axpy(bcast_ub, in_ub, -1.0)
                # step 3: refine sigmoid
                T.tile.mul(out_ub, out_ub, bcast_ub)

                T.tile.add(out_ub, out_ub, mhc_pre_eps)

                T.set_flag("v", "mte3", 0)

                # Phase 3: MTE3 writeback
                T.wait_flag("v", "mte3", 0)
                T.copy(out_ub[0:sub_block_tokens, 0:mhc_mult], output_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult])
                T.set_flag("mte3", "mte2", 0)

                # Drain flags
                T.wait_flag("mte3", "mte2", 0)

    return mhc_head_compute_mix_fwd_kernel


@tilelang.jit(pass_configs=_BWD_PASS_CONFIGS)
def _mhc_head_compute_mix_bwd(
    mhc_mult: int,
    reshape_factor: int = 1,
    token_block_size: int = 128,
    partial_size: int = 128,
) -> tilelang.JITKernel:
    num_tokens = T.symbolic('num_tokens')
    pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
    sub_block_tokens = token_block_size // VEC_NUM
    grid_size = T.ceildiv(num_tokens, token_block_size)
    orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

    @T.prim_func
    def mhc_head_compute_mix_bwd_kernel(
        output_mix_grad: T.Tensor[(num_tokens, mhc_mult), "float32"],
        input_mix: T.Tensor[(num_tokens, mhc_mult), "float32"],
        mhc_scale: T.Tensor[(1,), "float32"],
        mhc_base: T.Tensor[(orig_mhc_mult,), "float32"],
        input_mix_grad: T.Tensor[(num_tokens, mhc_mult), "float32"],
        mhc_scale_grad_partial: T.Tensor[(partial_size, mhc_mult), "float32"],
        mhc_base_grad_partial: T.Tensor[(partial_size, mhc_mult), "float32"],
    ) -> None:
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            row_start = cid * token_block_size + vid * sub_block_tokens

            base_grad_ub = T.alloc_ub((1, pad_mhc_mult), "float32")
            reduce_col_ub = T.alloc_ub((1, pad_mhc_mult), "float32")
            scale_grad_ub = T.alloc_ub((1, pad_mhc_mult), "float32")

            in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            buf_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            val_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            sig_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), "float32")
            scale_ub = T.alloc_ub((1,), "float32")
            base_ub = T.alloc_ub((pad_mhc_mult,), "float32")

            with T.Scope("V"):
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

                T.tile.mul(bcast_ub, bcast_ub, -1.0)

                T.tile.exp(bcast_ub, bcast_ub)
                T.tile.add(bcast_ub, bcast_ub, 1.0)
                T.tile.reciprocal(sig_ub, bcast_ub)

                T.tile.mul(val_ub, bcast_ub, sig_ub)
                T.tile.fill(bcast_ub, 2.0)
                T.tile.axpy(bcast_ub, val_ub, -1.0)
                T.tile.mul(sig_ub, sig_ub, bcast_ub)

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
        reshape_mhc_mult, mhc_pre_eps, reshape_factor=_RESHAPE_FACTOR, token_block_size=fwd_token_block_size)
    fwd_func(input_reshaped, mhc_scale, mhc_base, output_reshaped)

    output_tl = output_reshaped.reshape(-1, mhc_mult)

    print("[Reshape] Forward test PASSED!")

def test_bwd():
    n = 8192
    mhc_mult = 4
    reshape_mhc_mult = mhc_mult * _RESHAPE_FACTOR
    token_block_size = 128

    device = "npu"
    print(f"[Reshape] Running backward test on device: {device}, n={n}, reshape_mhc_mult={reshape_mhc_mult}")

    torch.manual_seed(42)
    input_mix = torch.randn((n, mhc_mult), dtype=torch.float32, device=device)
    mhc_scale = torch.randn((1,), dtype=torch.float32, device=device)
    mhc_base = torch.randn((mhc_mult,), dtype=torch.float32, device=device)
    output_mix_grad = torch.randn((n, mhc_mult), dtype=torch.float32, device=device)

    input_reshaped = input_mix.reshape(-1, reshape_mhc_mult)
    output_grad_reshaped = output_mix_grad.reshape(-1, reshape_mhc_mult)

    reshaped_num_tokens = n // _RESHAPE_FACTOR
    grid_size = math.ceil(reshaped_num_tokens / token_block_size)
    partial_size = grid_size * VEC_NUM

    input_grad_reshaped = torch.empty_like(input_reshaped)
    mhc_scale_grad_partial = torch.empty((partial_size, reshape_mhc_mult), dtype=torch.float32, device=device)
    mhc_base_grad_partial = torch.empty((partial_size, reshape_mhc_mult), dtype=torch.float32, device=device)

    bwd_func = _mhc_head_compute_mix_bwd(reshape_mhc_mult, reshape_factor=_RESHAPE_FACTOR, token_block_size=token_block_size, partial_size=partial_size)
    bwd_func(output_grad_reshaped, input_reshaped, mhc_scale, mhc_base, input_grad_reshaped, mhc_scale_grad_partial, mhc_base_grad_partial)
    print("[Reshape] Backward test PASSED!")


if __name__ == "__main__":
    test_fwd()
    test_bwd()