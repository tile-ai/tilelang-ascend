"""Cast-back from E5M6 kernel for Ascend NPU (tilelang v2 — fused kernel).

v2: Fused unpack + decode + dequant in one tilelang kernel.
    No CPU roundtrip — E5M6 bit unpack done via T.grid scalar ops in-kernel.

Also includes standalone per_token_cast_to_e5m6 (inlined, full GPU parity).
"""

import os

import tilelang
import tilelang.language as T
import torch


_QuantTensor = tuple[torch.Tensor, torch.Tensor]


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_KERNEL_CACHE = {}
_UE8M0_LUT_CPU = None
_E5M6_DECODE_LUT_CACHE = {}
_E5M6_FULL_LUT_CACHE = {}


def _get_ue8m0_lut_cpu() -> torch.Tensor:
    global _UE8M0_LUT_CPU
    if _UE8M0_LUT_CPU is None:
        bits = torch.arange(256, dtype=torch.int32)
        _UE8M0_LUT_CPU = torch.bitwise_left_shift(bits, 23).view(torch.float32)
    return _UE8M0_LUT_CPU


def _get_e5m6_full_lut(device: torch.device) -> torch.Tensor:
    """Build 4096-entry LUT: 12-bit E5M6 value -> float32.

    Replaces per-value exp/mant/sign extraction + 2 LUT lookups + arithmetic
    with a single LUT lookup.
    """
    global _E5M6_FULL_LUT_CACHE
    if device not in _E5M6_FULL_LUT_CACHE:
        lut = torch.zeros(4096, dtype=torch.float32)
        for u in range(4096):
            sign = -1.0 if (u >> 11) & 1 else 1.0
            exp = (u >> 6) & 0x1F
            mant = u & 0x3F
            if exp == 0:
                val = mant * (2.0**-20)
            elif exp == 31:
                val = 65504.0
            else:
                val = (2.0 ** (exp - 15)) * (1.0 + mant / 64.0)
            lut[u] = sign * val
        _E5M6_FULL_LUT_CACHE[device] = lut.to(device)
    return _E5M6_FULL_LUT_CACHE[device]


def _get_e5m6_decode_lut(device: torch.device):
    global _E5M6_DECODE_LUT_CACHE
    if device not in _E5M6_DECODE_LUT_CACHE:
        lut_a = torch.zeros(32, dtype=torch.float32)
        lut_b = torch.zeros(32, dtype=torch.float32)
        for i in range(32):
            if i == 0:
                lut_a[i] = 0.0
                lut_b[i] = 2.0**-20
            elif i == 31:
                lut_a[i] = 65504.0
                lut_b[i] = 0.0
            else:
                lut_a[i] = 2.0 ** (i - 15)
                lut_b[i] = 2.0 ** (i - 21)
        _E5M6_DECODE_LUT_CACHE[device] = (lut_a.to(device), lut_b.to(device))
    return _E5M6_DECODE_LUT_CACHE[device]


def _view_sf_2d(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if sf.dim() == 1:
        assert cols == 1, "1D sf only valid when num_per_channels == hidden"
        return sf[:rows].reshape(rows, 1)
    return sf


def _decode_sf_to_float32(sf: torch.Tensor, sf_rows: int, sf_cols: int) -> torch.Tensor:
    if sf_rows == 0 or sf_cols == 0:
        return torch.empty((sf_rows, sf_cols), dtype=torch.float32)
    sf_2d = _view_sf_2d(sf.cpu(), sf_rows, sf_cols)
    if sf_2d.dtype == torch.float32:
        return sf_2d[:sf_rows, :sf_cols].contiguous().to(torch.float32)
    assert sf_2d.dtype == torch.int32
    packed_cols = _ceil_div(sf_cols, 4)
    sf_cpu = sf_2d[:sf_rows, :packed_cols].contiguous()
    sf_u8 = sf_cpu.view(torch.uint8)[:, :sf_cols]
    sf_f32 = _get_ue8m0_lut_cpu()[sf_u8.to(torch.long)]
    return sf_f32.to(torch.float32)


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _e5m6_unpack_dequant_kernel(
    padded_m: int,
    padded_n_packed: int,
    padded_n: int,
    sf_rows_padded: int,
    sf_cols_padded: int,
    block_M: int,
    block_N: int,
    num_per_tokens: int,
    num_per_channels: int,
    out_dtype: str,
):
    m_blocks = padded_m // block_M
    n_blocks = padded_n // block_N
    block_N_packed = block_N * 3 // 2

    VEC_NUM = 2
    vec_M = block_M // VEC_NUM
    sf_vec_M = _ceil_div(vec_M + num_per_tokens - 1, num_per_tokens)
    sf_dim_N = _ceil_div(block_N, num_per_channels)
    sf_dim_N_padded = _align_up(max(sf_dim_N, 1), 16)

    @T.prim_func
    def main(
        x_packed: T.Tensor((padded_m, padded_n_packed), "uint8"),
        x_sf: T.Tensor((sf_rows_padded, sf_cols_padded), "float32"),
        full_lut: T.Tensor((4096,), "float32"),
        out: T.Tensor((padded_m, padded_n), out_dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, vid):
            bm = cid // n_blocks
            bn = cid % n_blocks

            row_start = bm * block_M + vid * vec_M
            col_packed = bn * block_N_packed
            sf_row_start = row_start // num_per_tokens
            sf_col_start = (bn * block_N) // num_per_channels

            packed_ub = T.alloc_ub((vec_M, block_N_packed), "uint8")
            sf_ub = T.alloc_ub((sf_vec_M, sf_dim_N_padded), "float32")
            out_ub = T.alloc_ub((vec_M, block_N), out_dtype)
            lut_ub = T.alloc_ub((4096,), "float32")

            T.copy(full_lut, lut_ub)
            T.copy(x_packed[row_start : row_start + vec_M, col_packed : col_packed + block_N_packed], packed_ub)
            T.copy(x_sf[sf_row_start : sf_row_start + sf_vec_M, sf_col_start : sf_col_start + sf_dim_N_padded], sf_ub)

            T.set_flag("MTE2", "V", 0)
            T.wait_flag("MTE2", "V", 0)
            with T.Scope("V"):
                for m, g in T.grid(vec_M, block_N // 8):
                    base = g * 12
                    b0 = T.cast(packed_ub[m, base + 0], "int32")
                    b1 = T.cast(packed_ub[m, base + 1], "int32")
                    b2 = T.cast(packed_ub[m, base + 2], "int32")
                    b3 = T.cast(packed_ub[m, base + 3], "int32")
                    b4 = T.cast(packed_ub[m, base + 4], "int32")
                    b5 = T.cast(packed_ub[m, base + 5], "int32")
                    b6 = T.cast(packed_ub[m, base + 6], "int32")
                    b7 = T.cast(packed_ub[m, base + 7], "int32")
                    b8 = T.cast(packed_ub[m, base + 8], "int32")
                    b9 = T.cast(packed_ub[m, base + 9], "int32")
                    b10 = T.cast(packed_ub[m, base + 10], "int32")
                    b11 = T.cast(packed_ub[m, base + 11], "int32")

                    u0 = (b3 << 4) | (b2 >> 4)
                    u1 = b1 | ((b2 & 0x0F) << 8)
                    u2 = (b0 << 4) | (b7 >> 4)
                    u3 = b6 | ((b7 & 0x0F) << 8)
                    u4 = (b5 << 4) | (b4 >> 4)
                    u5 = b11 | ((b4 & 0x0F) << 8)
                    u6 = (b10 << 4) | (b9 >> 4)
                    u7 = b8 | ((b9 & 0x0F) << 8)

                    sf_m = ((row_start + m) // num_per_tokens) - sf_row_start

                    n0 = g * 8 + 0
                    sf_n0 = ((bn * block_N + n0) // num_per_channels) - sf_col_start
                    out_ub[m, n0] = T.cast(lut_ub[u0] * sf_ub[sf_m, sf_n0], out_dtype)

                    n1 = g * 8 + 1
                    sf_n1 = ((bn * block_N + n1) // num_per_channels) - sf_col_start
                    out_ub[m, n1] = T.cast(lut_ub[u1] * sf_ub[sf_m, sf_n1], out_dtype)

                    n2 = g * 8 + 2
                    sf_n2 = ((bn * block_N + n2) // num_per_channels) - sf_col_start
                    out_ub[m, n2] = T.cast(lut_ub[u2] * sf_ub[sf_m, sf_n2], out_dtype)

                    n3 = g * 8 + 3
                    sf_n3 = ((bn * block_N + n3) // num_per_channels) - sf_col_start
                    out_ub[m, n3] = T.cast(lut_ub[u3] * sf_ub[sf_m, sf_n3], out_dtype)

                    n4 = g * 8 + 4
                    sf_n4 = ((bn * block_N + n4) // num_per_channels) - sf_col_start
                    out_ub[m, n4] = T.cast(lut_ub[u4] * sf_ub[sf_m, sf_n4], out_dtype)

                    n5 = g * 8 + 5
                    sf_n5 = ((bn * block_N + n5) // num_per_channels) - sf_col_start
                    out_ub[m, n5] = T.cast(lut_ub[u5] * sf_ub[sf_m, sf_n5], out_dtype)

                    n6 = g * 8 + 6
                    sf_n6 = ((bn * block_N + n6) // num_per_channels) - sf_col_start
                    out_ub[m, n6] = T.cast(lut_ub[u6] * sf_ub[sf_m, sf_n6], out_dtype)

                    n7 = g * 8 + 7
                    sf_n7 = ((bn * block_N + n7) // num_per_channels) - sf_col_start
                    out_ub[m, n7] = T.cast(lut_ub[u7] * sf_ub[sf_m, sf_n7], out_dtype)

            T.set_flag("V", "MTE3", 0)
            T.wait_flag("V", "MTE3", 0)
            T.copy(out_ub, out[row_start : row_start + vec_M, bn * block_N : bn * block_N + block_N])

    return main


def cast_back_e5m6(
    x: _QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize E5M6 packed tensor to float — fused in-kernel, no CPU roundtrip."""
    assert fmt in ("bf16", "fp32")
    out_torch_dtype = torch.bfloat16 if fmt == "bf16" else torch.float32
    kernel_out_dtype = "float32"

    x_data, x_sf = x
    assert x_data.dim() == 2
    assert x_data.dtype == torch.uint8
    assert x_sf.dim() in (1, 2)
    assert x_sf.dtype in (torch.float32, torch.int32)

    num_tokens = x_data.shape[0]
    packed_cols = x_data.shape[1]
    hidden = packed_cols * 2 // 3
    assert hidden % 8 == 0, f"hidden={hidden} must be divisible by 8"

    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=out_torch_dtype, device=x_data.device)

    num_per_tokens, num_per_channels = x_block_size
    assert num_per_tokens > 0 and num_per_channels > 0

    sf_rows = _ceil_div(num_tokens, num_per_tokens)
    sf_cols = _ceil_div(hidden, num_per_channels)

    # Handle col-major SF (from use_tma_aligned_col_major_sf in per_token_cast)
    if x_sf.dim() == 2 and x_sf.shape[0] == 1 and x_sf.shape[1] == num_tokens:
        x_sf = x_sf.t().contiguous()

    block_M = 128
    block_N = 128
    assert block_N % 8 == 0

    padded_m = _align_up(num_tokens, block_M)
    padded_n = _align_up(hidden, block_N)
    padded_n_packed = padded_n * 3 // 2

    vec_M = block_M // 2
    sf_dim_M_last = _ceil_div(vec_M + num_per_tokens - 1, num_per_tokens)
    sf_dim_N_last = _ceil_div(block_N, num_per_channels)
    sf_dim_N_padded = _align_up(max(sf_dim_N_last, 1), 16)
    last_sf_row = ((padded_m - block_M) + vec_M) // num_per_tokens
    last_sf_col = (padded_n - block_N) // num_per_channels

    sf_rows_padded = _align_up(max(sf_rows, last_sf_row + sf_dim_M_last), 16)
    sf_cols_padded = _align_up(max(sf_cols, last_sf_col + sf_dim_N_padded), 16)

    if padded_m == num_tokens and padded_n_packed == packed_cols:
        x_packed_npu = x_data.contiguous()
    else:
        x_packed_npu = torch.zeros((padded_m, padded_n_packed), dtype=torch.uint8, device=x_data.device)
        x_packed_npu[:num_tokens, :packed_cols].copy_(x_data.contiguous())

    if x_sf.dtype == torch.float32:
        sf_2d = _view_sf_2d(x_sf, sf_rows, sf_cols)
        sf_f32_npu = sf_2d[:sf_rows, :sf_cols].contiguous().to(torch.float32)
        if sf_rows_padded == sf_rows and sf_cols_padded == sf_cols:
            sf_padded_npu = sf_f32_npu
        else:
            sf_padded_npu = torch.ones((sf_rows_padded, sf_cols_padded), dtype=torch.float32, device=x_data.device)
            sf_padded_npu[:sf_rows, :sf_cols].copy_(sf_f32_npu)
    else:
        sf_f32_cpu = _decode_sf_to_float32(x_sf, sf_rows, sf_cols)
        sf_padded_cpu = torch.ones((sf_rows_padded, sf_cols_padded), dtype=torch.float32)
        sf_padded_cpu[:sf_rows, :sf_cols].copy_(sf_f32_cpu)
        sf_padded_npu = sf_padded_cpu.to(x_data.device)

    full_lut_npu = _get_e5m6_full_lut(x_data.device)

    kernel_key = (
        padded_m,
        padded_n_packed,
        padded_n,
        sf_rows_padded,
        sf_cols_padded,
        block_M,
        block_N,
        num_per_tokens,
        num_per_channels,
        kernel_out_dtype,
    )
    if kernel_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_key] = _e5m6_unpack_dequant_kernel(
            padded_m,
            padded_n_packed,
            padded_n,
            sf_rows_padded,
            sf_cols_padded,
            block_M,
            block_N,
            num_per_tokens,
            num_per_channels,
            out_dtype=kernel_out_dtype,
        )
    kernel = _KERNEL_CACHE[kernel_key]

    out_padded = kernel(x_packed_npu, sf_padded_npu, full_lut_npu)
    out = out_padded[:num_tokens, :hidden].contiguous()
    return out.to(out_torch_dtype) if out.dtype != out_torch_dtype else out


# ---------------------------------------------------------------------------
# Standalone per_token_cast_to_e5m6 (inlined — no external import needed)
# Full GPU parity: use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
# ---------------------------------------------------------------------------


def _float_to_e5m6_pack(data: torch.Tensor) -> torch.Tensor:
    """Pack float32 data into E5M6 format (uint8) on CPU.

    E5M6: 12-bit truncated half-precision. 8 values -> 96 bits -> 12 bytes.
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


def _pack_ue8m0_sf_npu(sf_f32: torch.Tensor) -> torch.Tensor:
    """Pack float32 SF to UE8M0 (int32) on NPU.

    Matches GPU: 4 UE8M0 bytes packed into 1 int32.
    UE8M0 = 8-bit exponent (exp_sf + 127), no mantissa.
    """
    num_tokens, groups = sf_f32.shape
    safe_sf = torch.where(sf_f32 == 0, 1e-8, sf_f32)
    exp_sf = torch.ceil(torch.log2(safe_sf))
    ue8m0 = (exp_sf + 127).clamp(0, 255).to(torch.uint8)

    padded_groups = _align_up(groups, 4)
    if padded_groups != groups:
        pad = torch.zeros(num_tokens, padded_groups - groups, dtype=torch.uint8, device=ue8m0.device)
        ue8m0 = torch.cat([ue8m0, pad], dim=1)

    packed = ue8m0.reshape(num_tokens, padded_groups // 4, 4).to(torch.int64)
    packed_int64 = packed[:, :, 0] + packed[:, :, 1] * 256 + packed[:, :, 2] * 65536 + packed[:, :, 3] * 16777216
    packed_int32 = torch.where(
        packed_int64 >= 2147483648,
        packed_int64 - 4294967296,
        packed_int64,
    ).to(torch.int32)
    return packed_int32.contiguous()


def per_token_cast_to_e5m6(
    x: torch.Tensor,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> _QuantTensor:
    """Cast a tensor to E5M6 with per-token scaling.

    Full GPU parity: use_tma_aligned_col_major_sf (transpose SF),
    round_sf (power-of-2 rounding), use_packed_ue8m0 (UE8M0 int32 packing).
    """
    assert x.dim() == 2
    num_tokens, hidden = x.shape
    assert num_per_channels == hidden

    if num_tokens == 0:
        out = torch.empty((0, hidden // 8 * 3), dtype=torch.int32, device=x.device).view(torch.uint8)
        if use_packed_ue8m0:
            sf = torch.empty((0, _ceil_div(1, 4)), dtype=torch.int32, device=x.device)
        else:
            sf = torch.empty((0, 1), dtype=torch.float32, device=x.device)
        return out, sf

    x_f32 = x.to(torch.float32).contiguous()
    max_e5m6 = 65024.0
    amax = x_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    sf = amax / max_e5m6
    sf_inv = max_e5m6 / amax

    if round_sf:
        target_device = sf.device
        sf_cpu = sf.cpu()
        bits = sf_cpu.view(torch.int32)
        exp_sf = ((bits - 1) >> 23) + 1 - 127
        sf = ((127 + exp_sf) << 23).view(torch.float32).to(target_device)
        sf_inv_bits = (127 - exp_sf).clamp(min=0) << 23
        sf_inv = sf_inv_bits.view(torch.float32).to(target_device)

    scaled = x_f32 * sf_inv
    packed = _float_to_e5m6_pack(scaled)

    if use_packed_ue8m0:
        sf = _pack_ue8m0_sf_npu(sf)

    if use_tma_aligned_col_major_sf:
        sf = sf.t().contiguous()

    return packed, sf


if __name__ == "__main__":
    import time

    NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
    torch.npu.set_device(NPU_DEVICE_ID)
    torch.manual_seed(42)

    def _calc_diff(a, b):
        a_f32 = a.to(torch.float32).cpu()
        b_f32 = b.to(torch.float32).cpu()
        return ((a_f32 - b_f32).abs().mean() / torch.max(a_f32.abs().mean(), torch.tensor(1e-6))).item()

    def _run_case(nt, h, fmt, rsf, ue8m0=False, tma=False):
        out_dtype = torch.bfloat16 if fmt == "bf16" else torch.float32
        x = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
        packed, sf = per_token_cast_to_e5m6(
            x,
            h,
            round_sf=rsf,
            use_packed_ue8m0=ue8m0,
            use_tma_aligned_col_major_sf=tma,
        )
        result = cast_back_e5m6((packed, sf), fmt, (1, h))
        assert result.dtype == out_dtype, f"dtype={result.dtype} != {out_dtype}"
        assert result.shape == (nt, h), f"shape={result.shape} != ({nt},{h})"
        diff = _calc_diff(result, x)
        assert diff < 5e-3, f"diff={diff} >= 5e-3 (nt={nt}, h={h}, fmt={fmt}, rsf={rsf})"
        print(f"  [PASS] {nt}x{h} fmt={fmt} rsf={rsf} ue8m0={ue8m0} tma={tma} -> diff={diff:.4e}", flush=True)

    test_cases = [
        (4001, 576, "fp32", True, False, False),
        (4001, 576, "fp32", False, False, False),
        (4001, 576, "bf16", True, False, False),
        (4001, 2048, "fp32", False, False, False),
        (8001, 2048, "bf16", True, False, False),
        (8001, 4096, "fp32", True, False, False),
        (8001, 4096, "bf16", False, False, False),
        (8001, 6144, "fp32", True, True, False),
        (8001, 7168, "fp32", False, False, False),
        (8001, 7168, "bf16", True, False, True),
    ]

    print("=== Correctness tests (10 cases) ===", flush=True)
    for nt, h, fmt, rsf, ue8m0, tma in test_cases:
        _run_case(nt, h, fmt, rsf, ue8m0, tma)

    print("\n=== Benchmark ===", flush=True)
    nt, h = 8001, 4096
    x = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
    packed, sf = per_token_cast_to_e5m6(x, h, round_sf=True)
    for _ in range(3):
        cast_back_e5m6((packed, sf), "fp32", (1, h))
    torch.npu.synchronize()

    repeat = 10
    t0 = time.perf_counter()
    for _ in range(repeat):
        cast_back_e5m6((packed, sf), "fp32", (1, h))
    torch.npu.synchronize()
    t_us = (time.perf_counter() - t0) / repeat * 1e6
    result = cast_back_e5m6((packed, sf), "fp32", (1, h))
    num_bytes = packed.nelement() * packed.element_size() + sf.nelement() * sf.element_size() + result.nelement() * result.element_size()
    bw = num_bytes / t_us / 1e3
    print(f"[cast_back_e5m6] {nt}x{h} fp32 -> {t_us:.1f}us, {bw:.1f} GB/s", flush=True)

    print("\nAll test PASSED! Kernel Output Match!", flush=True)
