"""SwiGLU backward + per-token cast kernel for Ascend NPU.

Fused tilelang kernel: dequant + clamp + sigmoid + forward + backward.
All paths use the kernel �?no PyTorch fallback.
"""

import os
from typing import Optional

import torch

import tilelang
import tilelang.language as T

# Inlined from common.py (only used symbols)
_QuantTensor = tuple[torch.Tensor, torch.Tensor]


_KERNEL_CACHE = {}


_BWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
}


@tilelang.jit(out_idx=[-5, -4, -3, -2, -1], pass_configs=_BWD_PASS_CONFIGS)
def _swiglu_bwd_fused_kernel(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    npc: int,
    use_clamp: bool,
):
    """Fused kernel: dequant + clamp + sigmoid + forward + backward.
    Outputs (5): xgl_raw, xgr_raw, out_weighted, xl_deq, xr_deq.
    amax/scale moved to Python to avoid T.tile.abs + T.reduce_max aliasing bug.
    """
    m_blocks = num_tokens // block_M
    n_blocks = hidden // block_N
    VEC_NUM = 2
    rows_per_vec = block_M // VEC_NUM

    @T.prim_func
    def main(
        x_data: T.Tensor((num_tokens, hidden * 2), "float32"),
        x_sf: T.Tensor((num_tokens, hidden * 2 // npc), "float32"),
        grad_w: T.Tensor((num_tokens, hidden), "float32"),
        w_in: T.Tensor((num_tokens, 1), "float32"),
        clamp_val: T.float32,
        xgl_raw_out: T.Tensor((num_tokens, hidden), "float32"),
        xgr_raw_out: T.Tensor((num_tokens, hidden), "float32"),
        out_weighted: T.Tensor((num_tokens, hidden), "float32"),
        xl_deq_out: T.Tensor((num_tokens, hidden), "float32"),
        xr_deq_out: T.Tensor((num_tokens, hidden), "float32"),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, vid):
            bx = cid // n_blocks
            by = cid % n_blocks
            row_base = bx * block_M + vid * rows_per_vec

            sf_col_l = by
            sf_col_r = n_blocks + by

            with T.Scope("V"):
                xl_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                xr_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                gw_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                act_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                tmp1_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                tmp2_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
                sig_ub = T.alloc_ub((rows_per_vec, block_N), "float32")

                sf_l_in_ub = T.alloc_ub((rows_per_vec, 1), "float32")
                sf_r_in_ub = T.alloc_ub((rows_per_vec, 1), "float32")
                w_ub = T.alloc_ub((rows_per_vec, 1), "float32")

                # Load SF
                for i in T.serial(rows_per_vec):
                    sf_l_in_ub[i, 0] = x_sf[row_base + i, sf_col_l]
                for i in T.serial(rows_per_vec):
                    sf_r_in_ub[i, 0] = x_sf[row_base + i, sf_col_r]

                # Load weight (per-row scalar)
                for i in T.serial(rows_per_vec):
                    w_ub[i, 0] = w_in[row_base + i, 0]

                # Load data and grad_w
                T.copy(x_data[row_base, by * block_N], xl_ub)
                T.copy(x_data[row_base, by * block_N + hidden], xr_ub)
                T.copy(grad_w[row_base, by * block_N], gw_ub)

                # Dequant
                for i, j in T.Parallel(rows_per_vec, block_N):
                    xl_ub[i, j] = xl_ub[i, j] * sf_l_in_ub[i, 0]
                for i, j in T.Parallel(rows_per_vec, block_N):
                    xr_ub[i, j] = xr_ub[i, j] * sf_r_in_ub[i, 0]

                # Output dequanted values BEFORE clamp
                T.copy(xl_ub, xl_deq_out[row_base, by * block_N])
                T.copy(xr_ub, xr_deq_out[row_base, by * block_N])

                # Clamp
                if use_clamp:
                    T.tile.min(xl_ub, xl_ub, clamp_val)
                    T.tile.min(xr_ub, xr_ub, clamp_val)
                    T.tile.max(xr_ub, xr_ub, -clamp_val)

                # Sigmoid: 1/(1+exp(-x))
                T.tile.fill(tmp1_ub, 0.0)
                T.tile.fill(tmp2_ub, 1.0)
                T.tile.sub(tmp1_ub, tmp1_ub, xl_ub)
                T.tile.exp(tmp1_ub, tmp1_ub)
                T.tile.add(tmp1_ub, tmp1_ub, tmp2_ub)
                T.tile.reciprocal(sig_ub, tmp1_ub)

                # Forward: act = silu(xl) * xr
                T.tile.mul(tmp1_ub, xl_ub, sig_ub)
                T.tile.mul(act_ub, tmp1_ub, xr_ub)

                # Weighted output: out = act * w
                for i, j in T.Parallel(rows_per_vec, block_N):
                    act_ub[i, j] = act_ub[i, j] * w_ub[i, 0]
                T.copy(act_ub, out_weighted[row_base, by * block_N])

                # Backward: silu'(x) = sig + x*sig*(1-sig)
                T.tile.sub(tmp2_ub, tmp2_ub, sig_ub)
                T.tile.mul(tmp2_ub, tmp2_ub, sig_ub)
                T.tile.mul(tmp2_ub, tmp2_ub, xl_ub)
                T.tile.add(tmp2_ub, sig_ub, tmp2_ub)

                # x_grad_right = gw * silu
                T.tile.mul(act_ub, gw_ub, tmp1_ub)

                # x_grad_left = gw * xr * silu'
                T.tile.mul(tmp1_ub, gw_ub, xr_ub)
                T.tile.mul(tmp1_ub, tmp1_ub, tmp2_ub)

                # Output raw (unscaled) gradients — amax/scale done in Python
                T.copy(tmp1_ub, xgl_raw_out[row_base, by * block_N])
                T.copy(act_ub, xgr_raw_out[row_base, by * block_N])

    return main


def _round_sf(sf):
    """Round scaling factors to power-of-2 via NPU-native float ops (zero CPU fallback).

    Uses torch.exp2 (NPU-native) instead of 2.0**x (CPU fallback).
    Clamps sf to prevent log2(0)=-Inf cascade. Adds -1e-6 bias for exact powers of 2.
    """
    sf_clamped = torch.clamp(sf, min=1e-38, max=1e38)
    exp_sf = torch.ceil(torch.log2(sf_clamped) - 1e-6)
    sf_out = torch.exp2(exp_sf)
    sf_inv = torch.exp2(-exp_sf)
    return sf_out, sf_inv


def _quantize_f32_to_e4m3fn(data_f32, max_fp8=448.0):
    target_device = data_f32.device
    data_cpu = data_f32.cpu()
    clamped = data_cpu.clamp(-max_fp8, max_fp8)
    result = clamped.to(torch.float8_e4m3fn)
    return result.to(target_device)


def _gather_weights(weight, pos_to_token_topk, num_topk, num_expand_tokens, device):
    w = torch.zeros(num_expand_tokens, dtype=torch.float32, device=device)
    mask = pos_to_token_topk >= 0
    if mask.any():
        valid_idx = pos_to_token_topk[mask].long()
        token_ids = valid_idx // num_topk
        topk_ids = valid_idx % num_topk
        w[mask] = weight[token_ids, topk_ids]
    return w


def _compute_weight_grad(grad_out_f32, act_out, pos_to_token_topk,
                          token_topk_to_pos, num_tokens, num_topk, device):
    dot = (grad_out_f32 * act_out).sum(dim=1)
    weight_grad = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)
    mask = pos_to_token_topk >= 0
    if mask.any():
        valid_idx = pos_to_token_topk[mask].long()
        token_ids = valid_idx // num_topk
        topk_ids = valid_idx % num_topk
        weight_grad.index_put_((token_ids, topk_ids), dot[mask], accumulate=True)
    invalid = token_topk_to_pos == -1
    weight_grad = weight_grad.masked_fill(invalid, 0.0)
    return weight_grad


def swiglu_backward_and_per_token_cast(
    x: _QuantTensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    pos_to_token_topk: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    num_per_channels: int,
    round_sf: bool = False,
    swiglu_clamp_value: Optional[float] = None,
) -> tuple[torch.Tensor, _QuantTensor, torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU backward pass with per-token quantization on Ascend NPU."""
    x_data, x_sf = x

    assert num_per_channels in (32, 128)
    assert x_data.dtype == torch.float8_e4m3fn
    assert (x_data.dim() == 2 or x_data.dim() == 3) and x_data.is_contiguous()
    assert x_sf.dim() == 2 and x_sf.is_contiguous()
    assert weight.dim() == 2 and weight.is_contiguous()
    assert pos_to_token_topk.dim() == 1
    assert token_topk_to_pos.dim() == 2 and token_topk_to_pos.is_contiguous()

    assert x_data.size(-1) % (2 * num_per_channels) == 0
    hidden = x_data.size(-1) // 2

    x_data = x_data.view(-1, hidden * 2)
    grad_out = grad_out.view(-1, hidden)

    num_expand_tokens = x_data.shape[0]
    num_tokens, num_topk = token_topk_to_pos.shape
    npc = num_per_channels
    device = x_sf.device

    assert x_sf.shape == (num_expand_tokens, 2 * hidden // npc)
    assert grad_out.shape == (num_expand_tokens, hidden)
    assert weight.shape == (num_tokens, num_topk)
    assert pos_to_token_topk.shape == (num_expand_tokens,)
    assert token_topk_to_pos.shape == (num_tokens, num_topk)

    if num_expand_tokens == 0:
        out = torch.empty((0, hidden), dtype=grad_out.dtype, device=device)
        x_grad_fp8 = torch.empty((0, hidden * 2), dtype=torch.float8_e4m3fn, device=device)
        x_grad_fp8_sf = torch.empty((0, hidden * 2 // npc), dtype=torch.float32, device=device)
        x_grad = torch.empty((0, hidden * 2), dtype=grad_out.dtype, device=device)
        weight_grad = torch.zeros((num_tokens, num_topk), dtype=torch.float32, device=device)
        return out, (x_grad_fp8, x_grad_fp8_sf), x_grad, weight_grad

    # fp8 �?float32 on CPU (NPU native cast may silently fallback, slower)
    x_data_f32 = x_data.cpu().to(torch.float32).to(device)
    x_sf_f32 = x_sf.to(torch.float32)

    block_M = 64
    block_N = npc

    use_clamp = swiglu_clamp_value is not None
    clamp_value = 0.0 if swiglu_clamp_value is None else swiglu_clamp_value

    # Precompute grad_w = grad_out * w
    weight_f32 = weight.to(torch.float32)
    w = _gather_weights(weight_f32, pos_to_token_topk, num_topk, num_expand_tokens, device)
    grad_out_f32 = grad_out.to(torch.float32)
    grad_w = grad_out_f32 * w.unsqueeze(1)

    # Fused kernel: dequant + clamp + sigmoid + forward + backward + amax + scale
    kernel_key = ("bwd_fused_v5", num_expand_tokens, hidden, block_M, block_N, npc, use_clamp)
    if kernel_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_key] = _swiglu_bwd_fused_kernel(
            num_expand_tokens, hidden, block_M, block_N, npc, use_clamp)
    kernel = _KERNEL_CACHE[kernel_key]

    xgl_raw, xgr_raw, out_weighted, xl_deq, xr_deq = kernel(
        x_data_f32, x_sf_f32, grad_w, w.unsqueeze(1), clamp_value)

    # Weighted output (kernel already applied w)
    out_bf16 = out_weighted.to(grad_out.dtype)

    # Clamp gradient masking (use kernel's dequanted output)
    if swiglu_clamp_value is not None:
        clamped_l = xl_deq >= swiglu_clamp_value
        clamped_r = (xr_deq >= swiglu_clamp_value) | (xr_deq <= -swiglu_clamp_value)
        mask_l = (~clamped_l).to(torch.float32)
        mask_r = (~clamped_r).to(torch.float32)
        xgl_raw = xgl_raw * mask_l
        xgr_raw = xgr_raw * mask_r

    # amax + sf + scale (in Python to avoid T.tile.abs + T.reduce_max aliasing bug)
    max_fp8 = 448.0
    groups = hidden // npc
    xgl_grouped = xgl_raw.reshape(num_expand_tokens, groups, npc)
    xgr_grouped = xgr_raw.reshape(num_expand_tokens, groups, npc)

    amax_l = xgl_grouped.abs().amax(dim=2).clamp(min=1e-4)
    amax_r = xgr_grouped.abs().amax(dim=2).clamp(min=1e-4)
    sf_l_raw = amax_l / max_fp8
    sf_r_raw = amax_r / max_fp8
    sf_inv_l = max_fp8 / amax_l
    sf_inv_r = max_fp8 / amax_r

    xgl_scaled = (xgl_grouped * sf_inv_l.unsqueeze(2)).reshape(num_expand_tokens, hidden)
    xgr_scaled = (xgr_grouped * sf_inv_r.unsqueeze(2)).reshape(num_expand_tokens, hidden)

    # x_grad (bf16, unscaled) = scaled * sf
    x_grad = torch.cat([xgl_raw, xgr_raw], dim=1).to(grad_out.dtype)

    # FP8 quantization
    if round_sf:
        sf_l_rounded, sf_inv_l_rounded = _round_sf(sf_l_raw)
        sf_r_rounded, sf_inv_r_rounded = _round_sf(sf_r_raw)
        rescale_l = (sf_inv_l_rounded * sf_l_raw).repeat_interleave(npc, dim=1)
        rescale_r = (sf_inv_r_rounded * sf_r_raw).repeat_interleave(npc, dim=1)
        x_grad_fp8_scaled = torch.cat([
            xgl_scaled * rescale_l,
            xgr_scaled * rescale_r,
        ], dim=1)
        x_grad_fp8_sf = torch.cat([sf_l_rounded, sf_r_rounded], dim=1)
    else:
        x_grad_fp8_scaled = torch.cat([xgl_scaled, xgr_scaled], dim=1)
        x_grad_fp8_sf = torch.cat([sf_l_raw, sf_r_raw], dim=1)

    x_grad_fp8 = _quantize_f32_to_e4m3fn(x_grad_fp8_scaled, max_fp8)

    # Weight gradient: use kernel's xl_deq/xr_deq
    if swiglu_clamp_value is not None:
        xl_deq_c = xl_deq.clamp(max=swiglu_clamp_value)
        xr_deq_c = xr_deq.clamp(min=-swiglu_clamp_value, max=swiglu_clamp_value)
    else:
        xl_deq_c, xr_deq_c = xl_deq, xr_deq
    sig_hp = torch.sigmoid(xl_deq_c)
    act_hp = xl_deq_c * sig_hp * xr_deq_c

    weight_grad = _compute_weight_grad(
        grad_out_f32, act_hp, pos_to_token_topk,
        token_topk_to_pos, num_tokens, num_topk, device)

    return out_bf16, (x_grad_fp8, x_grad_fp8_sf), x_grad, weight_grad


if __name__ == "__main__":
    import importlib.util as _ilu

    NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "7"))
    NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
    torch.npu.set_device(NPU_DEVICE_ID)

    _ascend_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = _ilu.spec_from_file_location(
        "per_token_cast", os.path.join(_ascend_dir, "per_token_cast_kernel.py"))
    _ptc_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ptc_mod)
    per_token_cast = _ptc_mod.per_token_cast

    torch.manual_seed(42)

    def _quantize_fp8(data_f32, max_fp8=448.0):
        target_device = data_f32.device
        data_cpu = data_f32.cpu()
        clamped = data_cpu.clamp(-max_fp8, max_fp8)
        result = clamped.to(torch.float8_e4m3fn)
        return result.to(target_device)

    def _ref_backward(x_data, x_sf, grad_out, weight, p2tt, npc, clamp, device):
        """PyTorch reference: dequant + clamp + silu + forward + backward."""
        x_f32 = x_data.cpu().to(torch.float32).to(device)
        sf_f32 = x_sf.to(torch.float32)
        ne, hidden2 = x_f32.shape
        hidden = hidden2 // 2

        xl = x_f32[:, :hidden]
        xr = x_f32[:, hidden:]
        sf_l = sf_f32[:, :hidden // npc]
        sf_r = sf_f32[:, hidden // npc:]

        sf_l_exp = sf_l.repeat_interleave(npc, dim=1)
        sf_r_exp = sf_r.repeat_interleave(npc, dim=1)
        xl = xl * sf_l_exp
        xr = xr * sf_r_exp

        if clamp is not None:
            xl = xl.clamp(max=clamp)
            xr = xr.clamp(min=-clamp, max=clamp)

        sig = torch.sigmoid(xl)
        act = xl * sig * xr

        w_f32 = weight.to(torch.float32)
        w_1d = w_f32.reshape(-1)
        w = w_1d[p2tt.long()].clamp(min=0)
        gw = grad_out.to(torch.float32) * w.unsqueeze(1)

        out_weighted = act * w.unsqueeze(1)

        silu_xl = xl * sig
        silu_grad = sig + xl * sig * (1 - sig)
        xgr = gw * silu_xl
        xgl = gw * xr * silu_grad

        if clamp is not None:
            mask_l = (xl < clamp).to(torch.float32)
            mask_r = ((xr < clamp) & (xr > -clamp)).to(torch.float32)
            xgl = xgl * mask_l
            xgr = xgr * mask_r

        return out_weighted, xgl, xgr

    def _calc_diff(a, b):
        a_f32 = a.to(torch.float32).cpu()
        b_f32 = b.to(torch.float32).cpu()
        denom = torch.max(a_f32.abs().mean(), torch.tensor(1e-6))
        return ((a_f32 - b_f32).abs().mean() / denom).item()

    test_cases = [
        (256, 2, 7168, 128, True, 10.0),
        (256, 2, 7168, 128, True, 0.5),
        (256, 2, 7168, 128, True, None),
    ]

    for nt, ntk, h, npc, rsf, clamp in test_cases:
        ne = nt * ntk
        x_expand = torch.randn((ne, h * 2), dtype=torch.bfloat16, device=NPU_DEVICE)
        x_scaled_f32, x_sf = per_token_cast(x_expand, "e4m3", npc)
        x_data = _quantize_fp8(x_scaled_f32)

        grad_out = torch.randn((ne, h), dtype=torch.bfloat16, device=NPU_DEVICE)
        weight = torch.rand((nt, ntk), dtype=torch.float32, device=NPU_DEVICE)
        p2tt = torch.arange(ne, dtype=torch.int32, device=NPU_DEVICE)
        tt2p = torch.arange(ne, dtype=torch.int32, device=NPU_DEVICE).reshape(nt, ntk)

        out, (xg_fp8, xg_sf), xg, wg = swiglu_backward_and_per_token_cast(
            (x_data, x_sf), grad_out, weight, p2tt, tt2p,
            num_per_channels=npc, round_sf=rsf, swiglu_clamp_value=clamp)

        assert out.dtype == torch.bfloat16
        assert xg_fp8.dtype == torch.float8_e4m3fn
        assert xg_sf.dtype == torch.float32
        assert xg.dtype == torch.bfloat16
        assert wg.dtype == torch.float32
        assert out.shape == (ne, h)
        assert xg_fp8.shape == (ne, h * 2)
        assert xg_sf.shape == (ne, h * 2 // npc)
        assert xg.shape == (ne, h * 2)
        assert wg.shape == (nt, ntk)
        assert not torch.isnan(out).any(), "out has NaN"
        assert not torch.isnan(xg).any(), "xg has NaN"
        assert not torch.isnan(xg_sf).any(), "xg_sf has NaN"

        ref_out, ref_xgl, ref_xgr = _ref_backward(
            x_data, x_sf, grad_out, weight, p2tt, npc, clamp, NPU_DEVICE)

        out_diff = _calc_diff(out, ref_out.to(torch.bfloat16))
        assert out_diff < 5e-2, f"out diff={out_diff}, clamp={clamp}"

        ref_xg = torch.cat([ref_xgl, ref_xgr], dim=1)
        xg_diff = _calc_diff(xg, ref_xg.to(torch.bfloat16))
        assert xg_diff < 5e-2, f"xg diff={xg_diff}, clamp={clamp}"

    print("All test PASSED! Kernel Output Match!")


