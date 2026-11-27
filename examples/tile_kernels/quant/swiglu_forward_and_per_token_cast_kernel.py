"""SwiGLU forward + per-token cast kernel for Ascend NPU (V3 �� fully aligned with GPU).

V3 changes vs V2:
- Output dtype: float8_e4m3fn for e4m3 (CPU cast), E5M6 packed uint8 for e5m6,
  E2M1 packed int8 for e2m1 �� matching GPU output formats
- Input dtype: supports bfloat16, float16, float32 (parameterized in_dtype_str)
- SF format: supports float32 (default) and packed UE8M0 (int32)
- SF layout: supports row-major (default) and col-major (use_tma_aligned_col_major_sf)
- Add GPU-matching assertions: fmt in ('e4m3','e5m6','e2m1'), hidden%128==0
- All GPU parameter combos fully supported (round_sf, use_packed_ue8m0, sf_clamp_min, etc.)
- clamped_count computed on Python side (matching GPU atomic_add semantics)
"""

import os
from typing import Optional

import torch

import tilelang
import tilelang.language as T

_NPU_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_QuantTensor = tuple[torch.Tensor, torch.Tensor]


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


_FMT_MAX_VALUE = {
    "e4m3": 448.0,
    "e5m6": 65024.0,
    "e2m1": 6.0,
}

_FMT_CLAMP_MIN = {
    "e4m3": 1e-4,
    "e5m6": 1e-4,
    "e2m1": 6.0 * (2.0**-126),
}

_IN_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _round_sf(sf):
    """Round scaling factors to power-of-2 on NPU via float arithmetic (no CPU roundtrip)."""
    safe_sf = torch.where(sf == 0, 1e-8, sf)
    exp_sf = torch.ceil(torch.log2(safe_sf))
    sf_out = 2.0**exp_sf
    sf_inv = 2.0 ** (-exp_sf)
    sf_out = torch.where(sf == 0, 0.0, sf_out)
    sf_inv = torch.where(sf == 0, 0.0, sf_inv)
    return sf_out, sf_inv


def _pack_ue8m0_sf(sf_f32, num_tokens, groups, device):
    """Pack float32 SF to UE8M0 (int32) on NPU via float arithmetic (no CPU roundtrip)."""
    safe_sf = torch.where(sf_f32 == 0, 1e-8, sf_f32)
    exp_sf = torch.ceil(torch.log2(safe_sf))
    ue8m0 = (exp_sf + 127).clamp(0, 255).to(torch.uint8)

    padded_groups = _align_up(groups, 4)
    if padded_groups != groups:
        pad = torch.zeros(num_tokens, padded_groups - groups, dtype=torch.uint8, device=ue8m0.device)
        ue8m0 = torch.cat([ue8m0, pad], dim=1)

    packed = ue8m0.reshape(num_tokens, padded_groups // 4, 4).to(torch.int64)
    packed_int64 = packed[:, :, 0] + packed[:, :, 1] * 256 + packed[:, :, 2] * 65536 + packed[:, :, 3] * 16777216
    packed_int32 = torch.where(packed_int64 >= 2147483648, packed_int64 - 4294967296, packed_int64).to(torch.int32)
    return packed_int32.contiguous()


def _float_to_e5m6_pack(data):
    """Pack float32 data into E5M6 format (uint8) on CPU.

    E5M6: 12-bit truncated half-precision. 8 values -> 96 bits -> 12 bytes.
    NPU: bit operations on CPU (NPU doesn't support uint32 bitwise shift).
    """
    num_tokens, hidden = data.shape
    assert hidden % 8 == 0
    groups = hidden // 8
    target_device = data.device

    data_cpu = data.cpu()
    f32_bits = data_cpu.view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    sign = (f32_bits >> 31) & 1
    exp32 = (f32_bits >> 23) & 0xFF

    exp16 = exp32 - 112
    mant16 = (f32_bits >> 13) & 0x3FF

    u16_rz = (sign << 15) | (torch.clamp(exp16, 0, 30) << 10) | mant16
    u16_rz = torch.where(exp32 == 0, torch.zeros_like(u16_rz), u16_rz)
    u16_rz = torch.where(exp16 < 1, sign << 15, u16_rz)

    remain_bits = f32_bits & 0x1FFFF
    u12 = (u16_rz >> 4) & 0xFFF
    lsb = u12 & 1
    round_up = ((lsb + remain_bits) > 0x10000).to(torch.int64)
    u12 = u12 + round_up
    u12 = torch.clamp(u12, max=0xFFF)

    vals = u12.reshape(num_tokens, groups, 8)

    w0 = (vals[:, :, 0] << 20) | (vals[:, :, 1] << 8) | (vals[:, :, 2] >> 4)
    w1 = (vals[:, :, 2] << 28) | (vals[:, :, 3] << 16) | (vals[:, :, 4] << 4) | (vals[:, :, 5] >> 8)
    w2 = (vals[:, :, 5] << 24) | (vals[:, :, 6] << 12) | vals[:, :, 7]

    packed = torch.stack([w0, w1, w2], dim=2)
    packed = (packed & 0xFFFFFFFF).to(torch.int32)
    packed = packed.reshape(num_tokens, groups * 3)

    return packed.view(torch.uint8).to(target_device)


_E2M1_LUT = None


def _get_e2m1_lut():
    """Build FP4 (E2M1) LUT: index 0-15 -> float32 value."""
    global _E2M1_LUT
    if _E2M1_LUT is None:
        lut = torch.zeros(16, dtype=torch.float32)
        for i in range(16):
            s = (i >> 3) & 1
            e = (i >> 1) & 0x3
            m = i & 0x1
            sign = -1.0 if s else 1.0
            if e == 0:
                val = sign * 0.5 * m
            else:
                val = sign * (2.0 ** (e - 1)) * (1.0 + 0.5 * m)
            lut[i] = val
        _E2M1_LUT = lut
    return _E2M1_LUT


def _float_to_e2m1_pack(data):
    """Pack float32 data to E2M1 (FP4) format (int8, 2 values per byte) on CPU."""
    num_tokens, hidden = data.shape
    assert hidden % 2 == 0
    target_device = data.device
    data_cpu = data.cpu().reshape(-1, 1)
    lut = _get_e2m1_lut().unsqueeze(0)
    diff = (data_cpu - lut).abs()
    codes = diff.argmin(dim=1).to(torch.uint8)
    codes = codes.reshape(num_tokens, hidden // 2, 2)
    packed = (codes[:, :, 0] & 0x0F) | ((codes[:, :, 1] & 0x0F) << 4)
    return packed.to(torch.int8).reshape(num_tokens, hidden // 2).to(target_device)


def _float_to_e4m3_pack_npu(data):
    """Pack float32 into E4M3 (uint8) on NPU via pure float arithmetic."""
    sign = (data < 0).to(torch.int32)
    a = data.abs()
    safe_a = torch.where(a == 0, 1e-8, a)
    log2_a = torch.log2(safe_a)
    exp = torch.floor(log2_a).to(torch.int32) + 7
    is_subnormal = exp < 1
    exp_normal = exp.clamp(1, 15)
    scale_normal = 2.0 ** (exp_normal - 7)
    mant_raw_normal = (safe_a / scale_normal - 1.0) * 8.0
    mant_raw_subnormal = safe_a * 512.0
    mant_raw = torch.where(is_subnormal, mant_raw_subnormal, mant_raw_normal)
    biased_exp = torch.where(is_subnormal, 0, exp_normal)
    mant = torch.round(mant_raw).to(torch.int32)
    carry_normal = (mant == 8) & (~is_subnormal)
    biased_exp = torch.where(carry_normal, biased_exp + 1, biased_exp)
    mant = torch.where(carry_normal, 0, mant)
    carry_subnormal = (mant == 8) & is_subnormal
    biased_exp = torch.where(carry_subnormal, 1, biased_exp)
    mant = torch.where(carry_subnormal, 0, mant)
    is_overflow = (biased_exp > 15) | ((biased_exp == 15) & (mant >= 7))
    biased_exp = torch.where(is_overflow, 15, biased_exp)
    mant = torch.where(is_overflow, 6, mant)
    biased_exp = torch.where(a == 0, 0, biased_exp)
    mant = torch.where(a == 0, 0, mant)
    u8_val = sign * 128 + biased_exp * 8 + mant
    return u8_val.to(torch.uint8)


def _cast_output(out_f32, fmt, device):
    """Cast float32 scaled values to target quantized format."""
    if fmt == "e4m3":
        max_val = _FMT_MAX_VALUE["e4m3"]
        out_clamped = out_f32.clamp(-max_val, max_val)
        return out_clamped.cpu().to(torch.float8_e4m3fn).to(device)
    elif fmt == "e5m6":
        return _float_to_e5m6_pack(out_f32)
    elif fmt == "e2m1":
        return _float_to_e2m1_pack(out_f32)
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")


def _format_sf(sf_f32, num_tokens, groups, use_packed_ue8m0, use_tma_aligned_col_major_sf, device):
    """Format SF tensor: pack to UE8M0 if needed, transpose for col-major if needed.

    After GPU epilogue, the final SF is always (num_tokens, groups) or
    (num_tokens, _ceil_div(groups,4)). Col-major only affects intermediate
    kernel storage; after epilogue transpose, values are identical.
    On NPU we produce row-major directly (equivalent values).
    """
    if use_packed_ue8m0:
        sf_out = _pack_ue8m0_sf(sf_f32, num_tokens, groups, device)
    else:
        sf_out = sf_f32
    return sf_out


_KERNEL_CACHE = {}


@tilelang.jit(out_idx=[-1], pass_configs=_NPU_PASS_CONFIGS)
def _swiglu_fwd_kernel(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    in_dtype_str: str,
):
    """SwiGLU + clamp kernel. All UB buffers. Input can be bf16/f16/f32."""
    m_blocks = num_tokens // block_M
    n_blocks = hidden // block_N

    @T.prim_func
    def main(
        x_in: T.Tensor((num_tokens, hidden * 2), in_dtype_str),
        clamp_val: T.float32,
        neg_clamp_val: T.float32,
        act_out: T.Tensor((num_tokens, hidden), "float32"),
    ):
        with T.Kernel(m_blocks * n_blocks, threads=2, is_npu=True) as (cid):
            bx = cid // n_blocks
            by = cid % n_blocks

            xl_ub = T.alloc_ub((block_M, block_N), "float32")
            xr_ub = T.alloc_ub((block_M, block_N), "float32")
            act_ub = T.alloc_ub((block_M, block_N), "float32")

            if in_dtype_str == "float32":
                T.copy(x_in[bx * block_M, by * block_N], xl_ub)
                T.copy(x_in[bx * block_M, by * block_N + hidden], xr_ub)
            else:
                x_raw_ub = T.alloc_ub((block_M, block_N), in_dtype_str)
                T.copy(x_in[bx * block_M, by * block_N], x_raw_ub)
                T.tile.cast(xl_ub, x_raw_ub, "CAST_NONE", block_M * block_N)
                T.copy(x_in[bx * block_M, by * block_N + hidden], x_raw_ub)
                T.tile.cast(xr_ub, x_raw_ub, "CAST_NONE", block_M * block_N)

            T.tile.min(xl_ub, xl_ub, clamp_val)
            T.tile.min(xr_ub, xr_ub, clamp_val)
            T.tile.max(xr_ub, xr_ub, neg_clamp_val)

            T.tile.silu(act_ub, xl_ub)
            T.tile.mul(act_ub, act_ub, xr_ub)

            T.copy(act_ub, act_out[bx * block_M, by * block_N])

    return main


@tilelang.jit(out_idx=[-1, -2], pass_configs=_NPU_PASS_CONFIGS)
def _swiglu_amax_scale_kernel(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    in_dtype_str: str,
):
    """SwiGLU + clamp + amax + scale fused kernel.

    Only usable when block_N == npc (one block per scaling group).
    Uses fixed max_value=448.0 and clamp_min=1e-4 (e4m3 defaults).
    sf_clamp_min handled in Python after kernel.
    """
    m_blocks = num_tokens // block_M
    n_blocks = hidden // block_N

    @T.prim_func
    def main(
        x_in: T.Tensor((num_tokens, hidden * 2), in_dtype_str),
        clamp_val: T.float32,
        neg_clamp_val: T.float32,
        sf_out: T.Tensor((num_tokens, n_blocks), "float32"),
        act_out: T.Tensor((num_tokens, hidden), "float32"),
    ):
        with T.Kernel(m_blocks * n_blocks, threads=2, is_npu=True) as (cid):
            bx = cid // n_blocks
            by = cid % n_blocks

            xl_ub = T.alloc_ub((block_M, block_N), "float32")
            xr_ub = T.alloc_ub((block_M, block_N), "float32")
            act_ub = T.alloc_ub((block_M, block_N), "float32")
            act_abs_ub = T.alloc_ub((block_M, block_N), "float32")
            amax_ub = T.alloc_ub((block_M, 1), "float32")
            sf_inv_ub = T.alloc_ub((block_M, 1), "float32")
            sf_ub = T.alloc_ub((block_M, 1), "float32")

            if in_dtype_str == "float32":
                T.copy(x_in[bx * block_M, by * block_N], xl_ub)
                T.copy(x_in[bx * block_M, by * block_N + hidden], xr_ub)
            else:
                x_raw_ub = T.alloc_ub((block_M, block_N), in_dtype_str)
                T.copy(x_in[bx * block_M, by * block_N], x_raw_ub)
                T.tile.cast(xl_ub, x_raw_ub, "CAST_NONE", block_M * block_N)
                T.copy(x_in[bx * block_M, by * block_N + hidden], x_raw_ub)
                T.tile.cast(xr_ub, x_raw_ub, "CAST_NONE", block_M * block_N)

            T.tile.min(xl_ub, xl_ub, clamp_val)
            T.tile.min(xr_ub, xr_ub, clamp_val)
            T.tile.max(xr_ub, xr_ub, neg_clamp_val)

            T.tile.silu(act_ub, xl_ub)
            T.tile.mul(act_ub, act_ub, xr_ub)

            for i, j in T.Parallel(block_M, block_N):
                act_abs_ub[i, j] = T.abs(act_ub[i, j])
            T.reduce_max(act_abs_ub, amax_ub, dim=1)

            for i in T.serial(block_M):
                clamped = T.max(amax_ub[i, 0], 1e-4)
                sf_ub[i, 0] = clamped / 448.0
                sf_inv_ub[i, 0] = 448.0 / clamped

            T.copy(sf_ub, sf_out[bx * block_M, by : by + 1])

            for i, j in T.Parallel(block_M, block_N):
                act_ub[i, j] = act_ub[i, j] * sf_inv_ub[i, 0]

            T.copy(act_ub, act_out[bx * block_M, by * block_N])

    return main


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    pos_to_expert: Optional[torch.Tensor] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
    sf_clamp_min: Optional[float] = None,
) -> _QuantTensor:
    assert x.dim() == 2 and x.is_contiguous()
    num_tokens, full_hidden = x.shape
    hidden = full_hidden // 2

    assert hidden % 128 == 0
    assert num_per_channels == 128 or num_per_channels == hidden
    assert fmt in ("e4m3", "e5m6", "e2m1")
    assert num_per_channels == 128 or (not use_tma_aligned_col_major_sf)

    if pos_to_token_topk is not None:
        assert pos_to_token_topk.dim() == 1
        assert x.shape[0] == num_tokens
        assert topk_weights is not None
        assert topk_weights.dim() == 2

    if pos_to_expert is not None:
        assert pos_to_expert.dim() == 1
        assert pos_to_expert.shape[0] == num_tokens

    if clamped_count is not None:
        assert swiglu_clamp_value is not None
        assert clamped_count.dim() == 1
        assert clamped_count.shape[0] == 3

    max_value = _FMT_MAX_VALUE[fmt]
    clamp_min = _FMT_CLAMP_MIN[fmt]
    if sf_clamp_min is not None:
        clamp_min = sf_clamp_min

    if num_tokens == 0:
        sf_cols = _ceil_div(hidden, num_per_channels)
        out_dtype = torch.float8_e4m3fn if fmt == "e4m3" else (torch.uint8 if fmt == "e5m6" else torch.int8)
        sf_dtype = torch.int32 if use_packed_ue8m0 else torch.float32
        return (torch.empty((0, hidden), dtype=out_dtype, device=x.device), torch.empty((0, sf_cols), dtype=sf_dtype, device=x.device))

    npc = num_per_channels
    block_M = 64
    block_N = min(npc, 128)

    clamp_val = swiglu_clamp_value if swiglu_clamp_value is not None else 1e10
    neg_clamp_val = -clamp_val

    in_dtype_str = _IN_DTYPE_STR.get(x.dtype, "bfloat16")
    groups = hidden // npc

    kernel_key = ("basic", num_tokens, hidden, block_M, block_N, in_dtype_str)
    if kernel_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_key] = _swiglu_fwd_kernel(num_tokens, hidden, block_M, block_N, in_dtype_str)
    swiglu_kernel = _KERNEL_CACHE[kernel_key]
    act = swiglu_kernel(x, clamp_val, neg_clamp_val)

    if clamped_count is not None and swiglu_clamp_value is not None:
        x_f32 = x.to(torch.float32)
        xl_chk = x_f32[:, :hidden]
        xr_chk = x_f32[:, hidden:]
        clamped_count[0] += (xl_chk > swiglu_clamp_value).sum()
        clamped_count[1] += (xr_chk > swiglu_clamp_value).sum()
        clamped_count[2] += (xr_chk < -swiglu_clamp_value).sum()

    if pos_to_token_topk is not None and topk_weights is not None:
        weights_1d = topk_weights.reshape(-1)
        w = weights_1d[pos_to_token_topk.long()].clamp(min=0).to(torch.float32)
        act = act * w.unsqueeze(1)

    if pos_to_expert is not None:
        mask = (pos_to_expert == -1).unsqueeze(1)
        act = act.masked_fill(mask, 0.0)

    groups = hidden // npc
    act_grouped = act.reshape(num_tokens, groups, npc)
    sf = act_grouped.abs().amax(dim=2).clamp(min=clamp_min) / max_value

    if round_sf:
        sf, sf_inv = _round_sf(sf)
        out_f32 = (act_grouped * sf_inv.unsqueeze(2)).reshape(num_tokens, hidden)
    else:
        out_f32 = (act_grouped / sf.unsqueeze(2)).reshape(num_tokens, hidden)

    out = _cast_output(out_f32, fmt, x.device)
    out_sf = _format_sf(sf, num_tokens, groups, use_packed_ue8m0, use_tma_aligned_col_major_sf, x.device)

    return out, out_sf


if __name__ == "__main__":
    NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
    torch.npu.set_device(NPU_DEVICE_ID)

    num_tokens = _align_up(4001, 64)
    dtype = torch.bfloat16
    torch.manual_seed(42)

    def _ref_swiglu_cast(x, npc, rsf, clamp):
        """PyTorch reference: SwiGLU + per-token cast."""
        x_f32 = x.to(torch.float32)
        hidden = x_f32.shape[1] // 2
        xl = x_f32[:, :hidden]
        xr = x_f32[:, hidden:]
        if clamp is not None:
            xl = xl.clamp(max=clamp)
            xr = xr.clamp(min=-clamp, max=clamp)
        act = xl * torch.sigmoid(xl) * xr
        groups = hidden // npc
        act_grouped = act.reshape(-1, groups, npc)
        amax = act_grouped.abs().amax(dim=2).clamp(min=1e-4)
        sf = amax / 448.0
        if rsf:
            sf_cpu = sf.cpu()
            bits = sf_cpu.view(torch.int32)
            exp_sf = ((bits - 1) >> 23) + 1 - 127
            sf = ((127 + exp_sf) << 23).view(torch.float32).to(x.device)
            sf_inv = ((127 - exp_sf) << 23).view(torch.float32).to(x.device)
        else:
            sf_inv = 448.0 / amax
        out_f32 = (act_grouped * sf_inv.unsqueeze(2)).reshape(-1, hidden)
        out_f32 = out_f32.clamp(-448.0, 448.0)
        out_fp8 = out_f32.cpu().to(torch.float8_e4m3fn).to(x.device)
        return out_fp8, sf

    def _calc_diff(a, b):
        a_f32 = a.to(torch.float32).cpu()
        b_f32 = b.to(torch.float32).cpu()
        denom = torch.max(a_f32.abs().mean(), torch.tensor(1e-6))
        return ((a_f32 - b_f32).abs().mean() / denom).item()

    test_cases = [
        (1024, 128, False, True, False, None),
        (1024, 128, True, True, True, 10.0),
        (2048, 128, False, False, False, 0.5),
        (2048, 2048, False, True, True, None),
        (3072, 128, False, True, False, 10.0),
        (3072, 3072, False, False, False, 0.5),
        (3584, 128, True, True, True, None),
        (3584, 3584, False, False, False, 10.0),
        (3584, 128, True, True, False, 0.5),
        (3584, 3584, False, True, True, None),
    ]

    for hidden, npc, tma, rsf, ue8m0, clamp in test_cases:
        block_M = 64
        block_N = min(128, hidden)
        in_dtype_str = "bfloat16"
        kkey = ("basic", num_tokens, hidden, block_M, block_N, in_dtype_str)
        if kkey not in _KERNEL_CACHE:
            _KERNEL_CACHE[kkey] = _swiglu_fwd_kernel(num_tokens, hidden, block_M, block_N, in_dtype_str)

        x = torch.randn((num_tokens, hidden * 2), dtype=dtype, device=NPU_DEVICE)
        out, sf = swiglu_forward_and_per_token_cast(
            x,
            "e4m3",
            npc,
            use_tma_aligned_col_major_sf=tma,
            round_sf=rsf,
            use_packed_ue8m0=ue8m0,
            swiglu_clamp_value=clamp,
        )

        out_f32 = out.float() if out.dtype != torch.float8_e4m3fn else out.cpu().float()
        assert not torch.isnan(out_f32).any(), "out has NaN"

        ref_out, ref_sf = _ref_swiglu_cast(x, npc, rsf, clamp)
        ref_out_f32 = ref_out.float() if ref_out.dtype != torch.float8_e4m3fn else ref_out.cpu().float()
        diff = _calc_diff(out_f32, ref_out_f32)
        assert diff < 2e-2, f"out diff={diff}, hidden={hidden}, npc={npc}, rsf={rsf}, clamp={clamp}"

        if not tma and not ue8m0:
            sf_f32 = sf.to(torch.float32).cpu()
            ref_sf_f32 = ref_sf.to(torch.float32).cpu()
            sf_diff = _calc_diff(sf_f32, ref_sf_f32)
            assert sf_diff < 1e-5, f"sf diff={sf_diff}, hidden={hidden}, npc={npc}, rsf={rsf}"

    print("All test PASSED! Kernel Output Match!")
