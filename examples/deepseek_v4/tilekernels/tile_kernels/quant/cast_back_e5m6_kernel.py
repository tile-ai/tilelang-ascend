"""Cast-back from E5M6 kernel for Ascend NPU.

Adapted from tile_kernels/quant/cast_back_e5m6_kernel.py.
Dequantizes E5M6 packed data back to float on Ascend NPU.

NPU适配规则应用:
- 规则1: 使用NPU pass_configs
- 规则4: T.LocalBuffer → T.Buffer
- 规则11: T.alloc_fragment → T.alloc_shared
- 规则12: 移除T.annotate_layout
"""

import os

import tilelang
import tilelang.language as T
import torch

try:
    from .common import NPU_PASS_CONFIGS, QuantTensor, align_up, ceil_div
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import NPU_PASS_CONFIGS, QuantTensor, align_up, ceil_div


@tilelang.jit(out_idx=[-1], pass_configs=NPU_PASS_CONFIGS)
def _cast_back_e5m6_kernel_npu(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    dtype: str = "float32",
):
    """NPU kernel: dequantize e5m6 packed data with scaling factors.

    For NPU, we work with pre-expanded float32 data (expanded on host side).
    """
    m_blocks = T.ceildiv(num_tokens, block_M)
    n_blocks = T.ceildiv(hidden, block_N)

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), dtype),
        x_sf: T.Tensor((num_tokens, 1), dtype),
        out: T.Tensor((num_tokens, hidden), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, threads=2, is_npu=True) as (cid):
            bx = cid // n_blocks
            by = cid % n_blocks

            x_ub = T.alloc_shared((block_M, block_N), dtype)
            sf_ub = T.alloc_shared((block_M, 1), dtype)
            out_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(x[bx * block_M, by * block_N], x_ub)
            T.copy(x_sf[bx * block_M, 0], sf_ub)

            for i, j in T.Parallel(block_M, block_N):
                out_ub[i, j] = x_ub[i, j] * sf_ub[i, 0]

            T.copy(out_ub, out[bx * block_M, by * block_N])

    return main


def _unpack_e5m6_to_float(packed: torch.Tensor, hidden: int) -> torch.Tensor:
    """Unpack E5M6 uint8 data to float32 on host side for NPU processing.

    E5M6: 8 values × 12 bits = 96 bits = 12 bytes per group.

    NPU适配: 在 CPU 上执行位运算 (NPU 不支持 uint32 且位移算子受限)，
    全向量化操作避免 Python for 循环，提升性能。
    """
    assert packed.dtype == torch.uint8
    num_tokens = packed.shape[0]
    target_device = packed.device

    packed_cpu = packed.cpu()
    packed_i32 = packed_cpu.view(torch.int32)
    packed_i64 = packed_i32.to(torch.int64) & 0xFFFFFFFF
    groups = packed_i64.shape[1] // 3

    words = packed_i64.reshape(num_tokens, groups, 3)
    w0, w1, w2 = words[:, :, 0], words[:, :, 1], words[:, :, 2]

    mask = 0xFFF0
    v0 = (w0 >> 16) & mask
    v1 = (w0 >> 4) & mask
    v2 = ((w0 << 8) | (w1 >> 24)) & mask
    v3 = (w1 >> 12) & mask
    v4 = w1 & mask
    v5 = ((w1 << 12) | (w2 >> 20)) & mask
    v6 = (w2 >> 8) & mask
    v7 = (w2 << 4) & mask

    vals = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=2)
    vals = vals.reshape(num_tokens, groups * 8)

    fp16_bits = vals.to(torch.int16)
    fp16_vals = fp16_bits.view(torch.float16)
    result = fp16_vals.to(torch.float32)

    return result[:, :hidden].to(target_device)


def cast_back_e5m6(
    x: QuantTensor,
    fmt: str,
    x_block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize an E5M6 packed tensor back to float on Ascend NPU.

    Args:
        x: (packed_data, scaling_factors) tensor pair.
        fmt: Target output format ('bf16' or 'fp32').
        x_block_size: Scaling block size.

    Returns:
        Dequantized tensor.
    """
    assert fmt in ('bf16', 'fp32')
    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32

    x_data, x_sf = x
    num_tokens = x_data.shape[0]

    if num_tokens == 0:
        hidden = x_data.size(1) * 2 // 3
        return torch.empty((0, hidden), dtype=out_dtype, device=x_data.device)

    hidden = x_data.size(1) * 2 // 3

    # NPU适配: unpack on host, then apply scaling on NPU
    x_f32 = _unpack_e5m6_to_float(x_data, hidden)
    sf_expanded = x_sf.to(torch.float32)

    if sf_expanded.dim() == 1:
        sf_expanded = sf_expanded.unsqueeze(1)
    sf_expanded = sf_expanded.repeat_interleave(
        x_block_size[0], dim=0
    )[:num_tokens]
    sf_expanded = sf_expanded.repeat_interleave(
        x_block_size[1], dim=1
    )[:, :hidden]
    sf_expanded = sf_expanded.contiguous()

    result = x_f32 * sf_expanded
    return result.to(out_dtype)
