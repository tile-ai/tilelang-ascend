"""Cast-back (dequantization) kernel for Ascend NPU using tilelang-ascend.

Adapted from the GPU cast_back_kernel.py implementation.
Performs: out[m, k] = x_data[m, k] * scaling_factor[m // npt, k // npc]

Supports Ascend A3 / A5 NPU via tilelang Developer mode with auto-sync.
"""

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def _cast_back_kernel_npu(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    dtype: str = "float32",
):
    """Ascend NPU kernel: element-wise multiply for dequantization.

    Both ``x`` and ``x_sf`` are expected to be the same dtype (float32).
    The scaling-factor broadcast/expansion is handled by the Python wrapper
    before this kernel is invoked.
    """
    m_blocks = T.ceildiv(num_tokens, block_M)
    n_blocks = T.ceildiv(hidden, block_N)

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), dtype),
        x_sf: T.Tensor((num_tokens, hidden), dtype),
        out: T.Tensor((num_tokens, hidden), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, threads=2, is_npu=True) as (cid):
            bx = cid // n_blocks
            by = cid % n_blocks

            x_ub = T.alloc_shared((block_M, block_N), dtype)
            sf_ub = T.alloc_shared((block_M, block_N), dtype)
            out_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(x[bx * block_M, by * block_N], x_ub)
            T.copy(x_sf[bx * block_M, by * block_N], sf_ub)

            for i, j in T.Parallel(block_M, block_N):
                out_ub[i, j] = x_ub[i, j] * sf_ub[i, j]

            T.copy(out_ub, out[bx * block_M, by * block_N])

    return main


def _expand_sf(
    x_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    num_per_tokens: int,
    num_per_channels: int,
) -> torch.Tensor:
    """Expand compact scaling factors to full ``(num_tokens, hidden)`` shape."""
    sf = x_sf.to(torch.float32)
    sf = sf.repeat_interleave(num_per_tokens, dim=0)[:num_tokens]
    sf = sf.repeat_interleave(num_per_channels, dim=1)[:, :hidden]
    return sf.contiguous()


def cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    x_block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize a quantized tensor back to FP16/BF16/FP32 on Ascend NPU.

    This is the Ascend NPU adaptation of the GPU ``cast_back`` operation.

    Args:
        x: ``(quantized_data, scaling_factors)`` tensor pair.
        fmt: Target output format — ``'bf16'``, ``'fp16'``, or ``'fp32'``.
        x_block_size: ``(num_per_tokens, num_per_channels)`` block dimensions
            that describe how scaling factors map to data elements.

    Returns:
        Dequantized tensor in the requested format.
    """
    assert fmt in ("bf16", "fp16", "fp32"), f"Unsupported format: {fmt}"
    out_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    out_torch_dtype = out_dtype_map[fmt]

    x_data, x_sf = x
    num_tokens, hidden = x_data.shape
    num_per_tokens, num_per_channels = x_block_size

    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=out_torch_dtype, device=x_data.device)

    x_f32 = x_data.to(torch.float32).contiguous()
    sf_expanded = _expand_sf(x_sf, num_tokens, hidden, num_per_tokens, num_per_channels)

    block_M = 128
    block_N = 128
    padded_m = align_up(num_tokens, block_M)
    padded_n = align_up(hidden, block_N)

    need_pad = padded_m != num_tokens or padded_n != hidden
    if need_pad:
        x_padded = torch.zeros((padded_m, padded_n), dtype=torch.float32, device=x_f32.device)
        sf_padded = torch.ones((padded_m, padded_n), dtype=torch.float32, device=x_f32.device)
        x_padded[:num_tokens, :hidden] = x_f32
        sf_padded[:num_tokens, :hidden] = sf_expanded
        x_f32 = x_padded
        sf_expanded = sf_padded

    kernel = _cast_back_kernel_npu(padded_m, padded_n, block_M, block_N)
    out_f32 = kernel(x_f32, sf_expanded)

    if need_pad:
        out_f32 = out_f32[:num_tokens, :hidden]

    return out_f32.to(out_torch_dtype)


def per_token_cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
) -> torch.Tensor:
    """Dequantize with per-token scaling on Ascend NPU.

    Convenience wrapper around :func:`cast_back` with ``num_per_tokens=1``.

    Args:
        x: ``(quantized_data, scaling_factors)`` tensor pair.
        fmt: Target output format — ``'bf16'``, ``'fp16'``, or ``'fp32'``.
        num_per_channels: Number of channels per scaling block.

    Returns:
        Dequantized tensor in the requested format.
    """
    return cast_back(x, fmt, (1, num_per_channels))
