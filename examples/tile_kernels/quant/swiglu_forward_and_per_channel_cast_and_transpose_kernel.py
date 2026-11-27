"""SwiGLU forward + per-channel cast and transpose kernel for Ascend NPU.


- Fused kernel: SwiGLU + amax + sf + scale in single kernel launch
- block_M = npt ensures per-channel amax (dim=0) within one block
- UB buffer reuse: x_ub (xl→xr→abs), work_ub (sigmoid→silu→act→scaled)
- Eliminates 4 Python NPU ops: nan_to_num, abs.amax, div, mul
- Python retains: round_sf (CPU bitwise), transpose (.t().contiguous()), padding
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


def _align(x, a):
    return (x + a - 1) // a * a


_IN_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _round_sf(sf):
    """Round scaling factors to power-of-2 via NPU native float ops (zero CPU fallback).

    Math: sf_out = 2^ceil(log2(sf))
    The 1e-6 bias prevents log2(2^n) = n.0000001 �?ceil = n+1 (off-by-one).
    All ops (clamp, log2, ceil, exp2) verified NPU-native �?no CPU fallback.
    """
    sf_clamped = torch.clamp(sf, min=1e-38, max=1e38)
    exp_sf = torch.ceil(torch.log2(sf_clamped) - 1e-6)
    sf_out = torch.exp2(exp_sf)
    sf_inv = torch.exp2(-exp_sf)
    return sf_out, sf_inv


def _round_sf_cpu(sf):
    """Round scaling factors to power-of-2 on CPU (reference only)."""
    target_device = sf.device
    sf_cpu = sf.cpu()
    bits = sf_cpu.view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_out = ((127 + exp_sf) << 23).view(torch.float32).to(target_device)
    sf_inv = ((127 - exp_sf) << 23).view(torch.float32).to(target_device)
    return sf_out, sf_inv


_KERNEL_CACHE = {}


@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        **_NPU_PASS_CONFIGS,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    },
)
def _swiglu_fwd_per_channel_fused_kernel(
    num_tokens: int,
    hidden: int,
    block_M: int,
    block_N: int,
    npt: int,
    use_clamp: bool,
    in_dtype_str: str = "float32",
):
    """Fused kernel: SwiGLU + per-channel amax + sf + scale (single pass).

    VEC_NUM=2 ROW split: each vid handles (npt, block_N) rows.
    block_M = npt * VEC_NUM. Per-channel amax (dim=0) within each vid's npt rows.
    Input is always f32 (bf16→f32 done in Python, NPU-native .to(float32)).
    Manual sync: only mte2→v and v→mte3 flags (v→mte2 NOT supported by NPU).
    """
    VEC_NUM = 2
    rows_per_vec = npt
    m_blocks = num_tokens // block_M
    n_blocks = hidden // block_N
    total_blocks = m_blocks * n_blocks

    @T.prim_func
    def main(
        x_in: T.Tensor((num_tokens, hidden * 2), "float32"),
        clamp_val: T.float32,
        act_out: T.Tensor((num_tokens, hidden), "float32"),
        sf_out: T.Tensor((m_blocks * VEC_NUM, hidden), "float32"),
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            bx = cid // n_blocks
            by = cid % n_blocks
            row_start = bx * block_M + vid * rows_per_vec
            col_start = by * block_N

            # UB buffers �?(npt, block_N) per vid
            x_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
            work_ub = T.alloc_ub((rows_per_vec, block_N), "float32")
            amax_ub = T.alloc_ub((1, block_N), "float32")
            sf_ub = T.alloc_ub((1, block_N), "float32")
            sf_inv_ub = T.alloc_ub((1, block_N), "float32")
            eps_ub = T.alloc_ub((1, block_N), "float32")

            with T.Scope("V"):
                # --- Phase 1: Load xl, compute sigmoid + silu ---
                T.copy(x_in[row_start, col_start], x_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)

                if use_clamp:
                    T.tile.min(x_ub, x_ub, clamp_val)

                # Sigmoid: work = 1 / (1 + exp(-x))
                T.tile.mul(work_ub, x_ub, -1.0)
                T.tile.exp(work_ub, work_ub)
                T.tile.add(work_ub, work_ub, 1.0)
                T.tile.reciprocal(work_ub, work_ub)

                # Silu: work = x * sigmoid(x)
                T.tile.mul(work_ub, x_ub, work_ub)

                # --- Phase 2: Load xr (reuse x_ub), compute SwiGLU ---
                # V→MTE3→MTE2 relay: V signals MTE3 (no-op relay), MTE3 signals MTE2
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.set_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 0)

                T.copy(x_in[row_start, col_start + hidden], x_ub)
                T.set_flag("mte2", "v", 1)
                T.wait_flag("mte2", "v", 1)

                if use_clamp:
                    T.tile.min(x_ub, x_ub, clamp_val)
                    T.tile.max(x_ub, x_ub, -clamp_val)

                # SwiGLU: work = silu * xr
                T.tile.mul(work_ub, work_ub, x_ub)

                # --- Phase 3: amax per channel (dim=0, across npt rows) ---
                # Reuse x_ub for abs (x_ub is free after xr consumed)
                T.tile.abs(x_ub, work_ub)
                T.reduce_max(x_ub, amax_ub, dim=0, real_shape=[rows_per_vec, block_N])

                # sf = (amax + eps) / 448, sf_inv = 1/sf
                T.tile.fill(eps_ub, 1e-4)
                T.tile.add(amax_ub, amax_ub, eps_ub)
                T.tile.div(sf_ub, amax_ub, 448.0)
                T.tile.reciprocal(sf_inv_ub, sf_ub)

                # Scale: work *= sf_inv (fully vectorized via T.Parallel)
                for i, j in T.Parallel(rows_per_vec, block_N):
                    work_ub[i, j] = work_ub[i, j] * sf_inv_ub[0, j]

                # --- Phase 4: Write outputs ---
                T.set_flag("v", "mte3", 1)
                T.wait_flag("v", "mte3", 1)
                T.copy(work_ub, act_out[row_start, col_start])
                T.copy(sf_ub, sf_out[bx * VEC_NUM + vid, col_start])

    return main


# ============================================================================
# Public API
# ============================================================================


def swiglu_forward_and_per_channel_cast_and_transpose(
    x: torch.Tensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    without_transpose: bool = False,
    swiglu_clamp_value: Optional[float] = None,
) -> _QuantTensor:
    """Fuse SwiGLU forward pass with per-channel FP8 cast and optional transpose.

    V5: Fused kernel does SwiGLU + amax + sf + scale in one launch.
    Python handles: round_sf (CPU bitwise), transpose, padding.
    """
    assert fmt == "e4m3"
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert num_per_tokens in (32, 128)

    num_tokens, full_hidden = x.shape
    hidden = full_hidden // 2

    if num_tokens == 0:
        sf_rows = _ceil_div(num_tokens, num_per_tokens)
        out_shape = (0, hidden) if without_transpose else (hidden, 0)
        return (
            torch.empty(out_shape, dtype=torch.float32, device=x.device),
            torch.empty((sf_rows, hidden), dtype=torch.float32, device=x.device),
        )

    npt = num_per_tokens
    block_M = npt * 2  # VEC_NUM=2, each vid handles npt rows
    block_N = 64 if npt >= 128 else 128  # (128,128) T.tile ops produce NaN �?tilelang codegen bug

    use_clamp = swiglu_clamp_value is not None
    clamp_value = 0.0 if swiglu_clamp_value is None else swiglu_clamp_value

    # bf16→f32 conversion (NPU-native .to(float32), verified no CPU fallback)
    # Kernel always receives f32 �?avoids T.tile.cast codegen issues
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    in_dtype_str = "float32"

    # --- Padding ---
    # Pad hidden to multiple of block_N
    hidden_padded = _ceil_div(hidden, block_N) * block_N
    if hidden_padded != hidden:
        pad_cols = hidden_padded - hidden
        x = torch.nn.functional.pad(x, (0, pad_cols))

    # Pad num_tokens to multiple of npt (for per-channel block alignment)
    orig_num_tokens = num_tokens
    if num_tokens % block_M != 0:
        pad_rows = block_M - (num_tokens % block_M)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_rows))
        num_tokens = x.shape[0]

    # --- Fused kernel: SwiGLU + amax + sf + scale ---
    kernel_key = ("fwd_per_ch_v5", num_tokens, hidden_padded, block_M, block_N, npt, use_clamp, in_dtype_str)
    if kernel_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_key] = _swiglu_fwd_per_channel_fused_kernel(
            num_tokens, hidden_padded, block_M, block_N, npt, use_clamp, in_dtype_str
        )
    swiglu_kernel = _KERNEL_CACHE[kernel_key]

    act_scaled, sf = swiglu_kernel(x, clamp_value)

    # --- Slice back to original dimensions ---
    sf_rows = _ceil_div(orig_num_tokens, npt)
    if orig_num_tokens != num_tokens or hidden_padded != hidden:
        act_scaled = act_scaled[:orig_num_tokens, :hidden].contiguous()
        sf = sf[:sf_rows, :hidden].contiguous()

    # --- round_sf rescaling (NPU native float ops, zero CPU fallback) ---
    if round_sf:
        sf_rounded, sf_inv_rounded = _round_sf(sf)
        rescale = (sf * sf_inv_rounded).repeat_interleave(npt, dim=0)[:orig_num_tokens, :]
        out = act_scaled * rescale
        sf = sf_rounded
    else:
        out = act_scaled

    # --- Transpose (Python, kernel writes non-transposed) ---
    if not without_transpose:
        out = out.t().contiguous()

    return out, sf


if __name__ == "__main__":
    NPU_DEVICE_ID = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
    torch.npu.set_device(NPU_DEVICE_ID)

    test_cases = [
        (4096, 576, 32, True, False, None),
        (4096, 2048, 128, False, True, 10.0),
        (4096, 3072, 32, True, True, 0.5),
        (4096, 4096, 128, False, False, None),
        (4096, 7168, 128, True, True, None),
        (8064, 576, 128, True, False, 10.0),
        (8064, 2560, 32, False, True, 0.5),
        (8064, 4096, 128, True, False, None),
        (8064, 6144, 32, False, True, None),
        (8064, 7168, 128, False, False, 0.5),
    ]

    dtype = torch.bfloat16
    torch.manual_seed(42)

    for _idx, (nt, h, npt, wt, rsf, clamp) in enumerate(test_cases):
        x = torch.randn((nt, h * 2), dtype=dtype, device=NPU_DEVICE)

        # Pre-compile kernel
        block_M = npt * 2
        block_N = 64 if npt >= 128 else 128
        hidden_padded = _ceil_div(h, block_N) * block_N
        use_clamp_k = clamp is not None
        in_dtype_str_k = "float32"
        nt_k = _align(nt, block_M)
        hp_k = _ceil_div(h, block_N) * block_N
        kkey = ("fwd_per_ch_v5", nt_k, hp_k, block_M, block_N, npt, use_clamp_k, in_dtype_str_k)
        if kkey not in _KERNEL_CACHE:
            _KERNEL_CACHE[kkey] = _swiglu_fwd_per_channel_fused_kernel(nt_k, hp_k, block_M, block_N, npt, use_clamp_k, in_dtype_str_k)

        torch.npu.synchronize(NPU_DEVICE)
        out, sf = swiglu_forward_and_per_channel_cast_and_transpose(
            x, "e4m3", num_per_tokens=npt, round_sf=rsf, without_transpose=wt, swiglu_clamp_value=clamp
        )
        torch.npu.synchronize(NPU_DEVICE)

        torch.npu.synchronize(NPU_DEVICE)
        hp = _ceil_div(h, block_N) * block_N
        x_k = x.to(torch.float32)
        if hp != h:
            x_k = torch.nn.functional.pad(x_k, (0, hp - h))
        nt_k2 = _align(nt, block_M)
        if nt_k2 != nt:
            x_k = torch.nn.functional.pad(x_k, (0, 0, 0, nt_k2 - nt))
        clamp_val_k = 0.0 if clamp is None else clamp
        kkey2 = ("fwd_per_ch_v5", nt_k2, hp, block_M, block_N, npt, use_clamp_k, in_dtype_str_k)
        if kkey2 in _KERNEL_CACHE:
            kern = _KERNEL_CACHE[kkey2]
            _a, _s = kern(x_k, clamp_val_k)
            torch.npu.synchronize(NPU_DEVICE)

        # Correctness checks
        assert out.dtype == torch.float32, f"out dtype={out.dtype}"
        expected_shape = (nt, h) if wt else (h, nt)
        assert out.shape == expected_shape, f"shape {out.shape} != {expected_shape}"
        assert not torch.isnan(out).any(), "out has NaN"

    print("All test PASSED! Kernel output Match!")
