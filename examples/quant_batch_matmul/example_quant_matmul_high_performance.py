"""High-performance int8 QuantMatmul with explicit MMA pipelines.

The Cube path overlaps GM-to-L1 and L1-to-L0 transfers with MMA by using
double-buffered L1 and L0A/L0B storage. The Vector path dequantizes the int32
accumulator and applies the optional offset, per-token scale, and bias.
"""

from typing import Literal, Optional

import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2
CAST_MODE = "CAST_RINT"
NUM_CORES = 20  # 910B3 has 20 Cube Cores (verified by torch.npu.get_device_properties)

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 256
DEFAULT_BLOCK_K = 256  # GM→L1 K tile
K_L0 = 128  # L1→L0 / MMA K tile

# Cross-CV events. Mode-2 waits complete after both Vector subcores signal.
C2V_EVENT_BASE = 0
V2C_EVENT_BASE = 2

ROWS_PER_STEP = 16

# Cube pipeline events.
L1_BUFFER_NUM = 2
L0_BUFFER_NUM = 2
L1_C0_ELEMS = 32  # 32-byte C0 / sizeof(int8)
L1_EVENT_BASE = 0
L0_EVENT_BASE = 2
L0C_EVENT = 4
SMALL_INPUT_EVENT = 2

MANUAL_SYNC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(
    out_idx=[-2],
    workspace_idx=[-1],
    pass_configs=MANUAL_SYNC_PASS_CONFIGS,
)
def quant_matmul_kernel(
    Batch: int,
    M: int,
    N: int,
    K: int,
    N_scale: int,
    scale_size: Literal["1", "N"],
    has_int32_bias: Literal[True, False],
    has_float_bias: Literal[True, False],
    has_pertoken: Literal[True, False],
    has_offset: Literal[True, False],
    scale_dtype: Literal["float32", "bfloat16"],
    out_dtype: Literal["float16", "bfloat16"],
    bias_dtype: Literal["int32", "bfloat16", "float16", "float32"],
    block_M: int = DEFAULT_BLOCK_M,
    block_N: int = DEFAULT_BLOCK_N,
    block_K: int = DEFAULT_BLOCK_K,
):
    """Compile-time parameterized QuantMatmul kernel (Expert CV fusion)."""
    in_dtype = "int8"
    accum_dtype = "int32"
    block_M_2 = block_M // VEC_NUM
    v_steps = block_M_2 // ROWS_PER_STEP
    loop_kk = block_K // K_L0
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)
    # Pad M up to a whole number of block_M tiles so that the Cube core can
    # always write a full [block_M, block_N] tile into the workspace without
    # out-of-bounds, and the Vector core (vid=1) can read its half safely even
    # when M < block_M (e.g. M=1 decode). Tail rows are zero-padded by T.copy
    # dynamic slicing and never written back to the (exact-M) output.
    M_padded = m_num * block_M

    # Balance contiguous output tiles across the fixed physical Cube cores.
    total_tiles = Batch * m_num * n_num
    q_tasks = total_tiles // NUM_CORES
    r_tasks = total_tiles % NUM_CORES

    @T.prim_func
    def main(
        A: T.Tensor([Batch, M, K], in_dtype),  # type: ignore
        B: T.Tensor([Batch, K, N], in_dtype),  # type: ignore
        scale: T.Tensor([N_scale], scale_dtype),  # type: ignore
        offset: T.Tensor([N_scale], "float32"),  # type: ignore
        pertoken_scale: T.Tensor([M_padded], "float32"),  # type: ignore
        bias: T.Tensor([N], bias_dtype),  # type: ignore
        C: T.Tensor([Batch, M, N], out_dtype),  # type: ignore
        workspace_1: T.Tensor([NUM_CORES, 2, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # Cores 0..r-1 process q+1 tiles; the remaining cores process q.
            # start = [0, q+1, 2*(q+1), ..., r*(q+1), r*(q+1)+q, ...]
            my_start_r = cid * (q_tasks + 1)
            my_start_other = r_tasks * (q_tasks + 1) + (cid - r_tasks) * q_tasks
            my_start = T.if_then_else(cid < r_tasks, my_start_r, my_start_other)
            my_count = T.if_then_else(cid < r_tasks, q_tasks + 1, q_tasks)

            # Cube domain: GM -> L1 -> L0A/L0B -> MMA -> L0C -> workspace.
            A_L1 = T.alloc_L1([L1_BUFFER_NUM, block_M, block_K], in_dtype)
            B_L1 = T.alloc_L1([L1_BUFFER_NUM, block_K, block_N], in_dtype)
            A_L0 = T.alloc_L0A([L0_BUFFER_NUM, block_M, K_L0], in_dtype)
            B_L0 = T.alloc_L0B([L0_BUFFER_NUM, K_L0, block_N], in_dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            with T.Scope("C"):
                for event_offset in T.unroll(L1_BUFFER_NUM):
                    T.set_flag("MTE1", "MTE2", L1_EVENT_BASE + event_offset)
                for event_offset in T.unroll(L0_BUFFER_NUM):
                    T.set_flag("M", "MTE1", L0_EVENT_BASE + event_offset)
                T.set_flag("FIX", "M", L0C_EVENT)

                for t in T.serial(my_count):
                    tile_idx = my_start + t
                    bb = tile_idx // (m_num * n_num)
                    bm = (tile_idx % (m_num * n_num)) // n_num
                    bn = tile_idx % n_num
                    slot = t % 2

                    # The first two tasks use distinct empty slots. Afterwards,
                    # wait until both Vector subcores have copied the old slot.
                    if t >= 2:
                        T.wait_cross_flag(V2C_EVENT_BASE + slot)

                    # Prologue: first GM→L1 tile.
                    T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE)
                    T.copy(A[bb, bm * block_M, 0], A_L1[0, :, :])
                    T.copy(B[bb, 0, bn * block_N], B_L1[0, :, :])
                    T.set_flag("MTE2", "MTE1", L1_EVENT_BASE)
                    T.wait_flag("FIX", "M", L0C_EVENT)

                    for bk in T.serial(k_num):
                        l1_side = bk % L1_BUFFER_NUM

                        # Prefetch the next K_L1 tile to the other L1 side.
                        if bk < k_num - 1:
                            next_l1_side = (bk + 1) % L1_BUFFER_NUM
                            T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE + next_l1_side)
                            T.copy(A[bb, bm * block_M, (bk + 1) * block_K], A_L1[next_l1_side, :, :])
                            T.copy(B[bb, (bk + 1) * block_K, bn * block_N], B_L1[next_l1_side, :, :])
                            T.set_flag("MTE2", "MTE1", L1_EVENT_BASE + next_l1_side)

                        for kk in T.serial(loop_kk):
                            l0_side = kk % L0_BUFFER_NUM
                            if kk == 0:
                                T.wait_flag("MTE2", "MTE1", L1_EVENT_BASE + l1_side)
                            T.wait_flag("M", "MTE1", L0_EVENT_BASE + l0_side)
                            # The int8 zZ/zN layouts require physical K-tile
                            # offsets instead of ordinary row-major offsets.
                            T.copy(
                                A_L1[l1_side, kk * block_M * K_L0 // block_K, 0],
                                A_L0[l0_side, :, :],
                            )
                            T.copy(
                                B_L1[l1_side, 0, kk * L1_C0_ELEMS * K_L0],
                                B_L0[l0_side, :, :],
                            )
                            if kk == loop_kk - 1:
                                T.set_flag("MTE1", "MTE2", L1_EVENT_BASE + l1_side)
                            T.set_flag("MTE1", "M", L0_EVENT_BASE + l0_side)
                            T.wait_flag("MTE1", "M", L0_EVENT_BASE + l0_side)
                            T.mma(
                                A_L0[l0_side, :, :],
                                B_L0[l0_side, :, :],
                                C_L0,
                                init=T.And(bk == 0, kk == 0),
                            )
                            T.set_flag("M", "MTE1", L0_EVENT_BASE + l0_side)

                    T.set_flag("M", "FIX", L0C_EVENT)
                    T.wait_flag("M", "FIX", L0C_EVENT)
                    T.copy(C_L0, workspace_1[cid, slot, 0, 0])
                    T.set_cross_flag("FIX", C2V_EVENT_BASE + slot)
                    T.set_flag("FIX", "M", L0C_EVENT)

                for event_offset in T.unroll(L1_BUFFER_NUM):
                    T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE + event_offset)
                for event_offset in T.unroll(L0_BUFFER_NUM):
                    T.wait_flag("M", "MTE1", L0_EVENT_BASE + event_offset)
                T.wait_flag("FIX", "M", L0C_EVENT)

            # Vector domain: dequantization and post-processing.
            with T.Scope("V"):
                # MTE2/V/MTE3 double-buffered pipeline.
                c_ub = T.alloc_ub([2, ROWS_PER_STEP, block_N], accum_dtype)
                c_scale = T.alloc_ub([ROWS_PER_STEP, block_N], "float32")
                c_out = T.alloc_ub([2, ROWS_PER_STEP, block_N], out_dtype)
                # Small inputs are reused across all row slices of one output tile.
                scale_ub = T.alloc_ub([block_N], "float32")
                scale_in_ub = T.alloc_ub([block_N], scale_dtype)
                bias_int32_ub = T.alloc_ub([block_N], "int32")
                bias_float_ub = T.alloc_ub([block_N], "float32")
                bias_in_ub = T.alloc_ub([block_N], bias_dtype)
                offset_ub = T.alloc_ub([block_N], "float32")
                pertoken_ub = T.alloc_ub([block_M_2], "float32")

                for t in T.serial(my_count):
                    tile_idx = my_start + t
                    bb = tile_idx // (m_num * n_num)
                    bm = (tile_idx % (m_num * n_num)) // n_num
                    bn = tile_idx % n_num
                    slot = t % 2

                    # Wait until the Cube core has written this workspace slot.
                    T.wait_cross_flag(C2V_EVENT_BASE + slot)

                    # Load small inputs once per output tile.
                    if scale_dtype == "float32":
                        if scale_size == "N":
                            T.copy(scale[bn * block_N], scale_ub)
                        else:
                            T.copy(scale[0], scale_ub)
                    else:
                        if scale_size == "N":
                            T.copy(scale[bn * block_N], scale_in_ub)
                        else:
                            T.copy(scale[0], scale_in_ub)

                    if has_int32_bias:
                        T.copy(bias[bn * block_N], bias_int32_ub)
                    if has_float_bias:
                        if bias_dtype != "float32":
                            T.copy(bias[bn * block_N], bias_in_ub)
                        else:
                            T.copy(bias[bn * block_N], bias_float_ub)
                    if has_offset:
                        if scale_size == "N":
                            T.copy(offset[bn * block_N], offset_ub)
                        else:
                            T.copy(offset[0], offset_ub)
                    if has_pertoken:
                        T.copy(pertoken_scale[bm * block_M + vid * block_M_2], pertoken_ub)

                    # All small inputs arrive through MTE2. Manual-sync mode
                    # must make them visible to V before bf16 casts/fills or
                    # any later post-processing reads them.
                    T.set_flag("MTE2", "V", SMALL_INPUT_EVENT)
                    T.wait_flag("MTE2", "V", SMALL_INPUT_EVENT)

                    if scale_dtype != "float32":
                        for j in T.Parallel(block_N):
                            scale_ub[j] = T.cast(scale_in_ub[j], "float32")

                    if scale_size != "N":
                        T.tile.fill(scale_ub, scale_ub[0])
                    T.pipe_barrier("V")

                    # Initialize both MTE3-to-MTE2 buffer-availability events.
                    T.set_flag("MTE3", "MTE2", 0)
                    T.set_flag("MTE3", "MTE2", 1)

                    # Prefetch the first row slice.
                    T.wait_flag("MTE3", "MTE2", 0)
                    T.copy(workspace_1[cid, slot, vid * block_M_2, 0], c_ub[0, :, :])
                    T.set_flag("MTE2", "V", 0)

                    # Pipeline the remaining row slices.
                    for sub_t in T.serial(v_steps - 1):
                        cur = sub_t % 2
                        nxt = (sub_t + 1) % 2
                        ws_row = vid * block_M_2 + (sub_t + 1) * ROWS_PER_STEP

                        # MTE2: load the next row slice.
                        T.wait_flag("MTE3", "MTE2", nxt)
                        T.copy(workspace_1[cid, slot, ws_row, 0], c_ub[nxt, :, :])
                        T.set_flag("MTE2", "V", nxt)

                        # V: process the current row slice.
                        T.wait_flag("MTE2", "V", cur)

                        # Cast int32 to float32 before post-processing.
                        T.tile.cast(c_scale, c_ub[cur, :, :], mode=CAST_MODE, count=ROWS_PER_STEP * block_N)
                        T.pipe_barrier("V")

                        # Integer bias is applied before dequantization.
                        if has_int32_bias:
                            for j in T.Parallel(block_N):
                                bias_float_ub[j] = T.cast(bias_int32_ub[j], "float32")
                            T.pipe_barrier("V")
                            for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                c_scale[i, j] = c_scale[i, j] + bias_float_ub[j]
                            T.pipe_barrier("V")

                        # Per-channel or per-tensor dequantization scale.
                        for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                            c_scale[i, j] = c_scale[i, j] * scale_ub[j]
                        T.pipe_barrier("V")

                        # Optional dequantization offset.
                        if has_offset:
                            for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                c_scale[i, j] = c_scale[i, j] + offset_ub[j]
                            T.pipe_barrier("V")

                        # Optional per-token scale.
                        if has_pertoken:
                            for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                c_scale[i, j] = c_scale[i, j] * pertoken_ub[sub_t * ROWS_PER_STEP + i]
                            T.pipe_barrier("V")

                        # Floating-point bias is applied after dequantization.
                        if has_float_bias:
                            if bias_dtype != "float32":
                                for j in T.Parallel(block_N):
                                    bias_float_ub[j] = T.cast(bias_in_ub[j], "float32")
                                T.pipe_barrier("V")
                                for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                    c_scale[i, j] = c_scale[i, j] + bias_float_ub[j]
                            else:
                                for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                    c_scale[i, j] = c_scale[i, j] + bias_in_ub[j]
                            T.pipe_barrier("V")

                        # Cast to the requested output dtype.
                        T.tile.cast(c_out[cur, :, :], c_scale, mode=CAST_MODE, count=ROWS_PER_STEP * block_N)
                        T.set_flag("V", "MTE3", cur)

                        # MTE3: store the current row slice.
                        T.wait_flag("V", "MTE3", cur)
                        out_row = bm * block_M + vid * block_M_2 + sub_t * ROWS_PER_STEP
                        if out_row < M:
                            T.copy(c_out[cur, :, :], C[bb, out_row, bn * block_N])
                        T.set_flag("MTE3", "MTE2", cur)

                    # All four workspace slices have now been issued on MTE2.
                    # Both Vector subcores signal; the Cube wait completes only
                    # after their reads are ordered before this event.
                    T.set_cross_flag("MTE2", V2C_EVENT_BASE + slot)

                    # Drain the last row slice.
                    last = (v_steps - 1) % 2
                    T.wait_flag("MTE2", "V", last)

                    # Cast int32 to float32 before post-processing.
                    T.tile.cast(c_scale, c_ub[last, :, :], mode=CAST_MODE, count=ROWS_PER_STEP * block_N)
                    T.pipe_barrier("V")

                    # Integer bias is applied before dequantization.
                    if has_int32_bias:
                        for j in T.Parallel(block_N):
                            bias_float_ub[j] = T.cast(bias_int32_ub[j], "float32")
                        T.pipe_barrier("V")
                        for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                            c_scale[i, j] = c_scale[i, j] + bias_float_ub[j]
                        T.pipe_barrier("V")

                    # Per-channel or per-tensor dequantization scale.
                    for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                        c_scale[i, j] = c_scale[i, j] * scale_ub[j]
                    T.pipe_barrier("V")

                    # Optional dequantization offset.
                    if has_offset:
                        for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                            c_scale[i, j] = c_scale[i, j] + offset_ub[j]
                        T.pipe_barrier("V")

                    # Optional per-token scale.
                    if has_pertoken:
                        for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                            c_scale[i, j] = c_scale[i, j] * pertoken_ub[(v_steps - 1) * ROWS_PER_STEP + i]
                        T.pipe_barrier("V")

                    # Floating-point bias is applied after dequantization.
                    if has_float_bias:
                        if bias_dtype != "float32":
                            for j in T.Parallel(block_N):
                                bias_float_ub[j] = T.cast(bias_in_ub[j], "float32")
                            T.pipe_barrier("V")
                            for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                c_scale[i, j] = c_scale[i, j] + bias_float_ub[j]
                        else:
                            for i, j in T.Parallel(ROWS_PER_STEP, block_N):
                                c_scale[i, j] = c_scale[i, j] + bias_in_ub[j]
                        T.pipe_barrier("V")

                    # Cast to the requested output dtype.
                    T.tile.cast(c_out[last, :, :], c_scale, mode=CAST_MODE, count=ROWS_PER_STEP * block_N)
                    T.set_flag("V", "MTE3", last)

                    # MTE3: store the last row slice.
                    T.wait_flag("V", "MTE3", last)
                    last_row = bm * block_M + vid * block_M_2 + (v_steps - 1) * ROWS_PER_STEP
                    if last_row < M:
                        T.copy(c_out[last, :, :], C[bb, last_row, bn * block_N])
                    T.set_flag("MTE3", "MTE2", last)

                    # Drain the two local pipeline events.
                    T.wait_flag("MTE3", "MTE2", 0)
                    T.wait_flag("MTE3", "MTE2", 1)

    return main


_kernel_cache: dict = {}

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
}


def _tl_dtype(torch_dtype: torch.dtype) -> str:
    for name, dt in _TORCH_DTYPE_MAP.items():
        if dt == torch_dtype:
            return name
    raise ValueError(f"unsupported dtype: {torch_dtype}")


def _reshape_bias(bias: torch.Tensor, N: int) -> torch.Tensor:
    """Normalize bias ([n] / [1,n] / [batch,1,n]) to a 1D [N] tensor."""
    if bias.dim() == 1:
        return bias.reshape(N)
    if bias.dim() == 2:
        return bias.reshape(N)
    # [batch, 1, n] -> the bias is shared across batch; take batch 0.
    return bias[0, 0, :].reshape(N)


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
) -> torch.Tensor:
    """Run QuantMatmul for a 2D or batched 3D input.

    out[..., m, n] = dequant(x1[...,m,k] @ x2[...,k,n]) with optional
    offset / pertoken_scale / bias (int32 pre-scale or float post-scale).
    """
    orig_ndim = x1.ndim
    if orig_ndim == 2:
        x1_3d = x1.unsqueeze(0)
        x2_3d = x2.unsqueeze(0)
    elif orig_ndim == 3:
        x1_3d = x1
        x2_3d = x2
    else:
        raise ValueError(f"unsupported x1.ndim={orig_ndim}, expect 2 or 3")

    Batch, M, K = x1_3d.shape
    N = x2_3d.shape[-1]
    if x2_3d.shape[-2] != K:
        raise ValueError(f"K mismatch: x1 K={K} but x2 K={x2_3d.shape[-2]}")

    out_dtype = output_dtype or "float16"
    scale_dtype = "bfloat16" if scale.dtype == torch.bfloat16 else "float32"
    scale_size: Literal["1", "N"] = "N" if scale.shape[0] == N else "1"
    N_scale = N if scale_size == "N" else 1
    has_int32_bias = bias is not None and bias.dtype == torch.int32
    has_float_bias = bias is not None and bias.dtype != torch.int32
    has_pertoken = pertoken_scale is not None
    has_offset = offset is not None
    bias_dtype = _tl_dtype(bias.dtype) if bias is not None else "int32"  # dummy

    block_M, block_N, block_K = DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K
    m_num = (M + block_M - 1) // block_M
    M_padded = m_num * block_M

    # K-tail zero-padding: T.copy does not zero-pad the K reduction tail in
    # Expert mode, so a non-block_K-aligned K would let stale/OOB L1 data
    # contribute to the GEMM accumulator. Pad K up to a whole number of
    # block_K tiles with zeros (zeros contribute nothing to the matmul, so the
    # result is unchanged). M-tail and N-tail are safe because their garbage
    # tiles are discarded by clamped T.copy on the output path.
    k_num = (K + block_K - 1) // block_K
    K_padded = k_num * block_K
    if K_padded > K:
        pad_n = K_padded - K
        x1_3d = torch.cat([x1_3d, torch.zeros(Batch, M, pad_n, dtype=torch.int8, device=x1_3d.device)], dim=-1)
        x2_3d = torch.cat([x2_3d, torch.zeros(Batch, pad_n, N, dtype=torch.int8, device=x2_3d.device)], dim=-2)
        K = K_padded
        # print(x1_3d.shape)
        # print(x2_3d.shape)
        # print(K)

    key = (
        Batch,
        M,
        N,
        K,
        N_scale,
        scale_size,
        has_int32_bias,
        has_float_bias,
        has_pertoken,
        has_offset,
        scale_dtype,
        out_dtype,
        bias_dtype,
    )
    if key not in _kernel_cache:
        _kernel_cache[key] = quant_matmul_kernel(
            Batch,
            M,
            N,
            K,
            N_scale,
            scale_size,
            has_int32_bias,
            has_float_bias,
            has_pertoken,
            has_offset,
            scale_dtype,
            out_dtype,
            bias_dtype,
            block_M,
            block_N,
            block_K,
        )
    kernel = _kernel_cache[key]

    # Move real inputs to NPU.
    x1_3d = x1_3d.npu()
    x2_3d = x2_3d.npu()
    scale = scale.npu()
    dev = x1_3d.device

    # Empty placeholders avoid launching a fill kernel for unused inputs.
    if offset is not None:
        offset_t = offset.npu()
    else:
        offset_t = torch.empty(N_scale, dtype=torch.float32, device=dev)

    if pertoken_scale is not None:
        pertoken_t = pertoken_scale.npu()
        if pertoken_t.shape[0] < M_padded:
            pad = torch.zeros(M_padded - pertoken_t.shape[0], dtype=torch.float32, device=dev)
            pertoken_t = torch.cat([pertoken_t, pad])
    else:
        pertoken_t = torch.empty(M_padded, dtype=torch.float32, device=dev)

    if bias is not None:
        bias_t = _reshape_bias(bias, N).contiguous().npu()
    else:
        bias_t = torch.empty(N, dtype=_TORCH_DTYPE_MAP[bias_dtype], device=dev)

    out = kernel(x1_3d, x2_3d, scale, offset_t, pertoken_t, bias_t)
    if orig_ndim == 2:
        return out.squeeze(0)
    return out


def quant_matmul_golden(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
) -> torch.Tensor:
    """Compute a high-precision PyTorch reference for QuantMatmul."""
    if x1.dtype in (torch.int8, torch.int32) and x2.dtype in (torch.int8, torch.int32):
        mm = torch.matmul(x1.double(), x2.double())
    else:
        mm = torch.matmul(x1.float(), x2.float()).double()

    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.double()

    y = mm * scale.double()

    if offset is not None:
        y = y + offset.double()

    if pertoken_scale is not None:
        y = y * pertoken_scale.double().unsqueeze(-1)

    if bias is not None and bias.dtype != torch.int32:
        y = y + bias.double()

    if output_dtype is None or output_dtype == "float16":
        return y.to(torch.float16)
    elif output_dtype == "bfloat16":
        return y.to(torch.bfloat16)
    raise ValueError(f"unsupported output_dtype: {output_dtype}")


def _run_example(
    x1_shape: tuple[int, ...],
    x2_shape: tuple[int, ...],
    scale_dtype: torch.dtype,
    output_dtype: str,
) -> None:
    n = x2_shape[-1]
    x1 = torch.randint(-128, 128, x1_shape, dtype=torch.int8)
    x2 = torch.randint(-128, 128, x2_shape, dtype=torch.int8)
    scale = torch.empty(n, dtype=scale_dtype).uniform_(0.001, 0.01)

    actual = quant_matmul(x1, x2, scale, output_dtype=output_dtype)
    expected = quant_matmul_golden(x1, x2, scale, output_dtype=output_dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    _run_example((128, 512), (512, 512), torch.float32, "float16")
    _run_example((2, 127, 513), (2, 513, 255), torch.bfloat16, "bfloat16")
    print("Kernel Output Match!")
