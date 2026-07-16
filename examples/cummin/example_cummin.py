"""cummin operator for TileLang-Ascend (Developer mode, pure Vector).

Inclusive prefix-min scan along the last dim, single input -> dual output:
  - values : running minimum, same shape/dtype as input
  - indices: argmin position (first occurrence on ties), int64 (kernel int32 + host cast)

Design: custom/cummin/DESIGN.md
"""

import argparse
import math
import sys

import tilelang
from tilelang import language as T
import torch


# ========== golden ==========
def golden_cummin(x: torch.Tensor, dim: int):
    """Reference via torch.cummin. Returns (values, indices[int64])."""
    values, indices = torch.cummin(x, dim=dim)
    return values, indices


# ========== vectorized operator implementation ==========

_VEC_ALIGN = 64  # 256B compare alignment (fp32: 64 * 4 = 256B)
_CORE_NUM = 24  # physical AIV cores on A2/A3
_VEC_BYTES_PER_ELEM = {"float16": 28, "bfloat16": 28, "float32": 34, "int32": 34}
_VEC_UB_BUDGET = 172 * 1024  # actual UB 192KB, headroom for stack


def _find_vec_N(Rows, dtype):
    """Reverse-derive vector width N: fill cores first, then fill UB."""
    bpe = _VEC_BYTES_PER_ELEM[dtype]
    n_ub = (_VEC_UB_BUDGET // bpe // _VEC_ALIGN) * _VEC_ALIGN
    n_core = (Rows + _CORE_NUM - 1) // _CORE_NUM
    n_core = ((n_core + _VEC_ALIGN - 1) // _VEC_ALIGN) * _VEC_ALIGN
    n = min(n_ub, max(_VEC_ALIGN, n_core))
    return max(_VEC_ALIGN, (n // _VEC_ALIGN) * _VEC_ALIGN)


_TILE_BYTES_PER_ELEM = {"float16": 16, "bfloat16": 16, "float32": 20, "int32": 20}
_RESIDENT_BYTES_PER_ELEM = {"float16": 18, "bfloat16": 18, "float32": 18, "int32": 18}


def _find_block_L(N, L, dtype):
    """Reverse-derive scan-axis block size block_L to fill remaining UB."""
    tile_bpe = _TILE_BYTES_PER_ELEM[dtype]
    resident = _RESIDENT_BYTES_PER_ELEM[dtype] * N
    avail = _VEC_UB_BUDGET - resident
    block_L = avail // (N * tile_bpe)
    block_L = min(block_L, L)
    return max(1, block_L)


@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def cummin_vec_ker(Rows, L, core_num, single_core_load, N, block_L, has_nan=True, dtype="float16"):
    """Vectorized prefix-min scan. Data layout: [L, Rows] (transposed).

    All non-fp32 dtypes (fp16/bf16/int32) are cast to fp32 for scan.
    """
    low_prec = dtype in ("float16", "bfloat16", "int32")
    cal_dtype = "float32"
    sel_mode = "VSEL_CMPMASK_SPR" if (N <= _VEC_ALIGN and dtype != "int32") else "VSEL_TENSOR_TENSOR_MODE"
    num_chunks = T.ceildiv(Rows, N)
    n_full = L // block_L  # full blocks
    partial = L % block_L  # tail rows

    @T.prim_func
    def main(
        A: T.Tensor([L, Rows], dtype),
        Values: T.Tensor([L, Rows], dtype),
        Indices: T.Tensor([L, Rows], "float32"),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            in_tile = T.alloc_shared([block_L, N], dtype)
            val_tile = T.alloc_shared([block_L, N], dtype)
            idx_tile = T.alloc_shared([block_L, N], "float32")
            in_cal_tile = T.alloc_shared([block_L, N], cal_dtype)
            val_cal_tile = T.alloc_shared([block_L, N], cal_dtype)
            run_min_cal_ub = T.alloc_shared([1, N], cal_dtype)
            run_idx_ub = T.alloc_shared([1, N], "float32")
            idx_curr_ub = T.alloc_shared([1, N], "float32")
            mask_le_ub = T.alloc_shared([1, N], "uint8")
            mask_nan_ub = T.alloc_shared([1, N], "uint8")

            with T.Scope("V"):
                for inner in T.serial(single_core_load):
                    c = cid * single_core_load + inner
                    if c < num_chunks:
                        col_base = c * N

                        for bi in T.serial(n_full):
                            l_base = bi * block_L

                            T.copy(A[l_base : l_base + block_L, col_base : col_base + N], in_tile)
                            T.barrier_all()
                            if low_prec:
                                T.tile.cast(in_cal_tile, in_tile, "CAST_NONE", block_L * N)
                            else:
                                T.copy(in_tile, in_cal_tile)

                            if bi == 0:
                                T.copy(in_cal_tile[0, :], run_min_cal_ub[0, :])
                                T.tile.fill(run_idx_ub, 0.0)

                            for jj in range(block_L):
                                T.tile.fill(idx_curr_ub, T.cast(l_base + jj, "float32"))
                                if has_nan:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.compare(mask_nan_ub, in_cal_tile[jj, :], in_cal_tile[jj, :], "EQ")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.select(run_idx_ub, mask_nan_ub, run_idx_ub, idx_curr_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                else:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                T.copy(run_min_cal_ub[0, :], val_cal_tile[jj, :])
                                T.copy(run_idx_ub[0, :], idx_tile[jj, :])

                            if low_prec:
                                T.tile.cast(val_tile, val_cal_tile, "CAST_RINT", block_L * N)
                            else:
                                T.copy(val_cal_tile, val_tile)
                            T.barrier_all()
                            T.copy(val_tile, Values[l_base : l_base + block_L, col_base : col_base + N])
                            T.copy(idx_tile, Indices[l_base : l_base + block_L, col_base : col_base + N])
                            T.barrier_all()

                        if partial > 0:
                            tb = n_full * block_L
                            T.copy(A[tb : tb + partial, col_base : col_base + N], in_tile)
                            T.barrier_all()
                            if low_prec:
                                T.tile.cast(in_cal_tile, in_tile, "CAST_NONE", partial * N)
                            else:
                                T.copy(in_tile[0:partial, :], in_cal_tile[0:partial, :])
                            for jj in range(partial):
                                T.tile.fill(idx_curr_ub, T.cast(tb + jj, "float32"))
                                if has_nan:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.compare(mask_nan_ub, in_cal_tile[jj, :], in_cal_tile[jj, :], "EQ")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.select(run_idx_ub, mask_nan_ub, run_idx_ub, idx_curr_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                else:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                T.copy(run_min_cal_ub[0, :], val_cal_tile[jj, :])
                                T.copy(run_idx_ub[0, :], idx_tile[jj, :])
                            if low_prec:
                                T.tile.cast(val_tile, val_cal_tile, "CAST_RINT", partial * N)
                            else:
                                T.copy(val_cal_tile[0:partial, :], val_tile[0:partial, :])
                            T.barrier_all()
                            T.copy(val_tile[0:partial, :], Values[tb : tb + partial, col_base : col_base + N])
                            T.copy(idx_tile[0:partial, :], Indices[tb : tb + partial, col_base : col_base + N])
                            T.barrier_all()

    return main


# ========== vectorized operator implementation v2 (no-transpose) ==========


def _find_vec_N_v2(M, N, dtype):
    """Reverse-derive vector width block_N for [M, R, N] layout."""
    bpe = _VEC_BYTES_PER_ELEM[dtype]
    n_ub = (_VEC_UB_BUDGET // bpe // _VEC_ALIGN) * _VEC_ALIGN
    chunks_needed = max(1, (_CORE_NUM + M - 1) // M)
    n_core = (N + chunks_needed - 1) // chunks_needed
    n_core = ((n_core + _VEC_ALIGN - 1) // _VEC_ALIGN) * _VEC_ALIGN
    n = min(n_ub, max(_VEC_ALIGN, n_core))
    return max(_VEC_ALIGN, (n // _VEC_ALIGN) * _VEC_ALIGN)


def _find_block_R(block_N, R, dtype):
    """Reverse-derive scan-axis block size block_R."""
    tile_bpe = _TILE_BYTES_PER_ELEM[dtype]
    resident = _RESIDENT_BYTES_PER_ELEM[dtype] * block_N
    avail = _VEC_UB_BUDGET - resident
    block_R = avail // (block_N * tile_bpe)
    block_R = min(block_R, R)
    return max(1, block_R)


@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def cummin_vec_ker_v2(M, R, N, core_num, single_core_load, block_N, block_R, has_nan=True, dtype="float16"):
    """Vectorized prefix-min scan on [M*R, N] physical layout (no transpose)."""
    low_prec = dtype in ("float16", "bfloat16", "int32")
    cal_dtype = "float32"
    sel_mode = "VSEL_CMPMASK_SPR" if (block_N <= _VEC_ALIGN and dtype != "int32") else "VSEL_TENSOR_TENSOR_MODE"
    num_chunks = T.ceildiv(N, block_N)
    total_tasks = M * num_chunks
    n_full = R // block_R
    partial = R % block_R
    MR = M * R

    @T.prim_func
    def main(
        A: T.Tensor([MR, N], dtype),
        Values: T.Tensor([MR, N], dtype),
        Indices: T.Tensor([MR, N], "float32"),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            in_tile = T.alloc_shared([block_R, block_N], dtype)
            val_tile = T.alloc_shared([block_R, block_N], dtype)
            idx_tile = T.alloc_shared([block_R, block_N], "float32")
            in_cal_tile = T.alloc_shared([block_R, block_N], cal_dtype)
            val_cal_tile = T.alloc_shared([block_R, block_N], cal_dtype)
            run_min_cal_ub = T.alloc_shared([1, block_N], cal_dtype)
            run_idx_ub = T.alloc_shared([1, block_N], "float32")
            idx_curr_ub = T.alloc_shared([1, block_N], "float32")
            mask_le_ub = T.alloc_shared([1, block_N], "uint8")
            mask_nan_ub = T.alloc_shared([1, block_N], "uint8")

            with T.Scope("V"):
                for inner in T.serial(single_core_load):
                    c = cid * single_core_load + inner
                    if c < total_tasks:
                        m = c // num_chunks
                        n_chunk = c % num_chunks
                        col_base = n_chunk * block_N
                        row_base = m * R

                        for bi in T.serial(n_full):
                            l_base = row_base + bi * block_R

                            T.copy(A[l_base : l_base + block_R, col_base : col_base + block_N], in_tile)
                            T.barrier_all()
                            if low_prec:
                                T.tile.cast(in_cal_tile, in_tile, "CAST_NONE", block_R * block_N)
                            else:
                                T.copy(in_tile, in_cal_tile)

                            if bi == 0:
                                T.copy(in_cal_tile[0, :], run_min_cal_ub[0, :])
                                T.tile.fill(run_idx_ub, 0.0)

                            for jj in range(block_R):
                                T.tile.fill(idx_curr_ub, T.cast(bi * block_R + jj, "float32"))
                                if has_nan:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.compare(mask_nan_ub, in_cal_tile[jj, :], in_cal_tile[jj, :], "EQ")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.select(run_idx_ub, mask_nan_ub, run_idx_ub, idx_curr_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                else:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                T.copy(run_min_cal_ub[0, :], val_cal_tile[jj, :])
                                T.copy(run_idx_ub[0, :], idx_tile[jj, :])

                            if low_prec:
                                T.tile.cast(val_tile, val_cal_tile, "CAST_RINT", block_R * block_N)
                            else:
                                T.copy(val_cal_tile, val_tile)
                            T.barrier_all()
                            T.copy(val_tile, Values[l_base : l_base + block_R, col_base : col_base + block_N])
                            T.copy(idx_tile, Indices[l_base : l_base + block_R, col_base : col_base + block_N])
                            T.barrier_all()

                        if partial > 0:
                            tb = row_base + n_full * block_R
                            T.copy(A[tb : tb + partial, col_base : col_base + block_N], in_tile)
                            T.barrier_all()
                            if low_prec:
                                T.tile.cast(in_cal_tile, in_tile, "CAST_NONE", partial * block_N)
                            else:
                                T.copy(in_tile[0:partial, :], in_cal_tile[0:partial, :])
                            for jj in range(partial):
                                T.tile.fill(idx_curr_ub, T.cast(n_full * block_R + jj, "float32"))
                                if has_nan:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.compare(mask_nan_ub, in_cal_tile[jj, :], in_cal_tile[jj, :], "EQ")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.select(run_idx_ub, mask_nan_ub, run_idx_ub, idx_curr_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                else:
                                    T.tile.compare(mask_le_ub, in_cal_tile[jj, :], run_min_cal_ub, "LE")
                                    T.tile.select(run_idx_ub, mask_le_ub, idx_curr_ub, run_idx_ub, sel_mode)
                                    T.tile.min(run_min_cal_ub, run_min_cal_ub, in_cal_tile[jj, :])
                                T.copy(run_min_cal_ub[0, :], val_cal_tile[jj, :])
                                T.copy(run_idx_ub[0, :], idx_tile[jj, :])
                            if low_prec:
                                T.tile.cast(val_tile, val_cal_tile, "CAST_RINT", partial * block_N)
                            else:
                                T.copy(val_cal_tile[0:partial, :], val_tile[0:partial, :])
                            T.barrier_all()
                            T.copy(val_tile[0:partial, :], Values[tb : tb + partial, col_base : col_base + block_N])
                            T.copy(idx_tile[0:partial, :], Indices[tb : tb + partial, col_base : col_base + block_N])
                            T.barrier_all()

    return main


def _run_vec_kernel_v2(x_npu, M, R, N, dtype, has_nan=True):
    """Run vectorized kernel v2 on [M, R, N] physical layout. No transpose."""
    block_N = _find_vec_N_v2(M, N, dtype)
    block_R = _find_block_R(block_N, R, dtype)

    num_chunks = (N + block_N - 1) // block_N
    total_tasks = M * num_chunks
    core_num = min(total_tasks, _CORE_NUM)
    single_core_load = (total_tasks + core_num - 1) // core_num

    x_2d = x_npu.reshape(M * R, N)

    ker = cummin_vec_ker_v2(M, R, N, core_num, single_core_load, block_N, block_R, has_nan, dtype=dtype)
    out_v_t, out_i32_t = ker(x_2d)

    out_v = out_v_t.reshape(M, R, N)
    out_i32 = out_i32_t.to(torch.int32).reshape(M, R, N)
    return out_v, out_i32


_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}

# per-dtype value tolerance
_TOL = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-5),
    "bfloat16": (1e-2, 5e-3),
    "int32": (0.0, 0.0),
}


def _make_input(shape, dtype):
    tdt = _TORCH_DTYPE[dtype]
    if dtype == "int32":
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == "bfloat16":
        return torch.randn(shape).to(torch.bfloat16)
    return torch.randn(shape, dtype=tdt)


def _run_vec_kernel(x_LR, Rows, L, dtype, has_nan=True):
    """Run vectorized kernel on [L, Rows] layout."""
    N = _find_vec_N(Rows, dtype)
    block_L = _find_block_L(N, L, dtype)

    num_chunks = (Rows + N - 1) // N
    core_num = min(num_chunks, _CORE_NUM)
    single_core_load = (num_chunks + core_num - 1) // core_num

    ker = cummin_vec_ker(Rows, L, core_num, single_core_load, N, block_L, has_nan, dtype=dtype)
    out_v_t, out_i32_t = ker(x_LR)  # [L, Rows]

    out_v = out_v_t.contiguous()
    out_i32 = out_i32_t.contiguous().to(torch.int32)
    return out_v, out_i32


def cummin_run_npu(x_npu: torch.Tensor, dim: int):
    """All-NPU host wrapper. Returns (values[same shape/dtype], indices[int64])."""
    dtype = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
    }[x_npu.dtype]
    ndim = x_npu.ndim
    d = dim % ndim

    # v2 no-transpose path: operate directly on [M, R, N] physical layout.
    # When N < _VEC_ALIGN, fall back to v1 movedim+vec path.
    shape = x_npu.shape
    M = 1
    for i in range(d):
        M *= shape[i]
    R = shape[d]
    N = 1
    for i in range(d + 1, ndim):
        N *= shape[i]

    if N >= _VEC_ALIGN:
        x_contig = x_npu.contiguous()  # no-op if already contiguous
        if dtype == "int32":
            has_nan = False
        else:
            has_nan = bool(torch.isnan(x_contig).any())
        out_v, out_i32 = _run_vec_kernel_v2(x_contig, M, R, N, dtype, has_nan)
        out_v = out_v.reshape(shape)
        out_i = out_i32.to(torch.int64).reshape(shape)
        return out_v, out_i
    # N too small (e.g. dim=last, N=1): v2 vec can't parallelize.
    # Use v1 movedim+vec path (transpose cost << vectorization gain).
    xm = x_npu.movedim(d, 0)
    moved_shape = xm.shape
    L = moved_shape[0]
    Rows = xm.numel() // L if L > 0 else 0
    x_LR = xm.reshape(L, Rows).contiguous()

    if dtype == "int32":
        has_nan = False
    else:
        has_nan = bool(torch.isnan(x_LR).any())

    out_v_LR, out_i32_LR = _run_vec_kernel(x_LR, Rows, L, dtype, has_nan)
    out_v = out_v_LR.reshape(moved_shape).movedim(0, d).contiguous()
    out_i = out_i32_LR.to(torch.int64).reshape(moved_shape).movedim(0, d).contiguous()
    return out_v, out_i


def cummin_run(x: torch.Tensor, dim: int):
    """General host wrapper: accepts CPU or NPU tensor."""
    if x.is_npu:
        return cummin_run_npu(x, dim)
    # CPU input: transfer to NPU, run, transfer back
    out_v_npu, out_i_npu = cummin_run_npu(x.npu(), dim)
    torch.npu.synchronize()
    return out_v_npu.cpu(), out_i_npu.cpu()


def _rel_stats(out_v: torch.Tensor, ref_v: torch.Tensor):
    """MERE (max element relative error) and MARE (mean) in fp32 domain."""
    o = out_v.float()
    r = ref_v.float()
    denom = r.abs() + 1e-12
    rel = (o - r).abs() / denom
    return rel.max().item(), rel.mean().item()


def _compare(out_v_cpu, out_i_cpu, ref_v, ref_i, dtype):
    """Compare values (by dtype tol) + indices (exact). Returns (ok, msg).

    Handles inf/nan: uses equal_nan=True for allclose, and skips max_diff
    computation when inf/nan is present.
    """
    atol, rtol = _TOL[dtype]
    if dtype == "int32":
        v_ok = torch.equal(out_v_cpu, ref_v)
        max_diff = 0.0 if v_ok else (out_v_cpu - ref_v).abs().max().item()
        vmsg = "values_exact" if v_ok else "values_MISMATCH"
    else:
        v_ok = torch.allclose(out_v_cpu.float(), ref_v.float(), atol=atol, rtol=rtol, equal_nan=True)
        has_special = torch.isinf(out_v_cpu).any() or torch.isnan(out_v_cpu).any()
        if has_special:
            max_diff = float("nan")
            mere = float("nan")
            mare = float("nan")
        else:
            max_diff = (out_v_cpu.float() - ref_v.float()).abs().max().item()
            mere, mare = _rel_stats(out_v_cpu, ref_v)
        vmsg = f"values max_diff={max_diff:.3e} MERE={mere:.3e} MARE={mare:.3e}"

    i_ok = torch.equal(out_i_cpu, ref_i)
    idx_mismatch = int((out_i_cpu != ref_i).sum().item())
    imsg = "indices_exact" if i_ok else f"indices_MISMATCH n={idx_mismatch}"
    return (v_ok and i_ok), f"{vmsg} | {imsg} | max_diff={max_diff:.3e}"


def _make_case_input(shape, dtype, value_range):
    """Generate input tensor with given value range (cann-bench compatible).

    Special value ranges:
      [-inf, inf]: random values with some +/-inf injected
      [nan, nan]:  random values with some nan injected
    """
    tdt = _TORCH_DTYPE[dtype]
    lo, hi = value_range
    if dtype == "int32":
        return torch.randint(int(lo), int(hi) + 1, shape, dtype=torch.int32)
    if lo == hi:
        return torch.full(shape, lo, dtype=tdt)
    if math.isinf(lo) or math.isinf(hi):
        x = torch.randn(shape, dtype=tdt) * 10
        mask = torch.rand(shape) < 0.1
        x[mask] = float("inf")
        mask = torch.rand(shape) < 0.1
        x[mask] = float("-inf")
        return x
    if math.isnan(lo) or math.isnan(hi):
        x = torch.randn(shape, dtype=tdt) * 10
        mask = torch.rand(shape) < 0.2
        x[mask] = float("nan")
        return x
    x = torch.rand(shape, dtype=tdt) * (hi - lo) + lo
    return x


# (case_id, shape, dtype, dim, value_range, note)
_CASES = [
    (1, [1024, 1024], "float16", -1, [-1, 1], "S-float16-1M-aligned-dim=-1"),
    (2, [2048, 2048], "float32", -1, [-2, 2], "M-float32-4M-aligned-dim=-1"),
    (3, [4096, 4096], "bfloat16", -1, [-3, 3], "M-bfloat16-16M-aligned-dim=-1"),
    (4, [8192, 8192], "int32", -1, [-10000, 10000], "L-int32-67M-aligned-dim=-1"),
    (5, [16384, 16384], "float16", 0, [-100, 100], "L-float16-268M-aligned-dim=0"),
    (6, [8192, 8192], "float32", 1, [-1000, 1000], "L-float32-1G-aligned-dim=1"),
    (7, [1023, 1023], "bfloat16", -1, [-0.1, 0.1], "S-bfloat16-1M-unaligned-dim=-1"),
    (8, [1009, 1021], "float16", 0, [-1, 2], "S-float16-1M-prime-unaligned-dim=0"),
    (9, [1537, 769], "float32", -1, [-5, 10], "S-float32-1M-unaligned-dim=-1"),
    (10, [363, 367, 373], "bfloat16", 1, [-50, 100], "M-bfloat16-50M-3D-dim=1"),
    (11, [2049, 513], "float16", -1, [-65504, 65504], "S-float16-fp16-extreme-dim=-1"),
    (12, [3, 7, 13, 4001], "float32", -1, [-88, 88], "S-float32-4D-dim=-1"),
    (13, [1000003], "bfloat16", -1, [-float("inf"), float("inf")], "S-bfloat16-inf-1D-dim=-1"),
    (14, [11, 13, 17, 67, 67], "float16", 2, [float("nan"), float("nan")], "M-float16-nan-5D-dim=2"),
    (15, [3, 7, 11, 13, 1013], "int32", -1, [0, 0], "M-int32-zero-5D-dim=-1"),
    (16, [512, 2049], "float32", -1, [-0.5, 0.5], "S-float32-unaligned-dim=-1"),
    (17, [255, 8193], "bfloat16", 0, [-1, 3], "S-bfloat16-unaligned-dim=0"),
    (18, [4097, 511], "float16", -1, [-1000, 1000], "S-float16-unaligned-dim=-1"),
    (19, [2, 511, 2049], "float32", 1, [-0.2, 0.2], "S-float32-3D-dim=1"),
    (20, [4, 255, 2049], "bfloat16", -1, [-3, 6], "S-bfloat16-3D-dim=-1"),
]


def _run_case_wrapped(case_id, shape, dtype, dim, value_range, note):
    """Run a single case via cummin_run_npu. Returns (ok, msg)."""
    x_cpu = _make_case_input(shape, dtype, value_range)
    x_npu = x_cpu.npu()
    out_v_npu, out_i_npu = cummin_run_npu(x_npu, dim)
    torch.npu.synchronize()
    out_v = out_v_npu.cpu()
    out_i = out_i_npu.cpu()
    ref_v, ref_i = golden_cummin(x_cpu, dim)
    return _compare(out_v, out_i, ref_v, ref_i, dtype)


def test_cummin_all():
    """Run all 20 cann-bench cases."""
    ok = True
    for case_id, shape, dtype, dim, value_range, note in _CASES:
        try:
            case_ok, msg = _run_case_wrapped(
                case_id,
                shape,
                dtype,
                dim,
                value_range,
                note,
            )
            if case_ok:
                print(f"[PRECISION_PASS] case_{case_id} {note}: {msg}")
            else:
                print(f"[PRECISION_FAIL] case_{case_id} {note}: {msg}")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"[PRECISION_FAIL] case_{case_id} {note}: {e}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="cummin NPU kernel test")
    parser.add_argument("--level", default="all", choices=["all"])
    parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = test_cummin_all()

    if ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
