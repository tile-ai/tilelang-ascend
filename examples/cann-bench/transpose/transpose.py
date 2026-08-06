"""Transpose operator: permute tensor dimensions via 3-path dispatch.

Path 1: 3D transpose (T.tile.transpose) — perm normalizable to (batch, M, N) -> (batch, N, M).
Path 2: record aggregation (T.copy 4D) — shared contiguous suffix record.
Path 3: stride fallback (T.copy 1D dynamic) — full reversal etc.

dtype specialization:
  - float16/int16/float32/int32: hardware T.tile.transpose
  - bfloat16: host-side view(int16) zero-copy reinterpret
  - int8: kernel-internal T.tile.cast to float16
  - int64: T.tile.transpose scalar fallback (correct but slower)
"""

import tilelang
from tilelang import language as T
import torch

# A2/A3 physical AIV core count
_CORE_NUM = 24

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
}

_kernel_cache = {}


def _torch_dtype_to_str(dtype):
    return _DTYPE_MAP[dtype]


_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def _dtype_elem_bytes(dtype):
    """Return element size in bytes for a dtype."""
    if isinstance(dtype, torch.dtype):
        dtype = _torch_dtype_to_str(dtype)
    return _DTYPE_BYTES.get(dtype, 4)


# ---------------------------------------------------------------------------
# Path 1: 3D transpose kernel (T.tile.transpose)
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _kernel_2d_transpose(batch, M, N, block_M, block_N, dtype, use_int8_cast):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    total_tasks = batch * m_num * n_num
    VEC_NUM = 2
    block_num = T.ceildiv(total_tasks, VEC_NUM)

    @T.prim_func
    def kernel(
        x: T.Tensor((batch, M, N), dtype),  # type: ignore
        y: T.Tensor((batch, N, M), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task = cid * VEC_NUM + vid

            if task < total_tasks:
                b = task // (m_num * n_num)
                rem = task % (m_num * n_num)
                bm = rem // n_num
                bn = rem % n_num

                with T.Scope("V"):
                    a_ub = T.alloc_ub((block_M, block_N), dtype)

                    if use_int8_cast:
                        a_cal = T.alloc_ub((block_M, block_N), "float16")
                        b_cal = T.alloc_ub((block_N, block_M), "float16")
                        b_ub = T.alloc_ub((block_N, block_M), dtype)

                        T.copy(x[b, bm * block_M, bn * block_N], a_ub)
                        T.tile.cast(a_cal, a_ub, "CAST_NONE", block_M * block_N)
                        T.tile.transpose(b_cal, a_cal)
                        T.tile.cast(b_ub, b_cal, "CAST_RINT", block_N * block_M)
                        T.copy(b_ub, y[b, bn * block_N, bm * block_M])
                    else:
                        b_ub = T.alloc_ub((block_N, block_M), dtype)
                        T.copy(x[b, bm * block_M, bn * block_N], a_ub)
                        T.tile.transpose(b_ub, a_ub)
                        T.copy(b_ub, y[b, bn * block_N, bm * block_M])

    return kernel


# ---------------------------------------------------------------------------
# Path 2: record aggregation kernel (T.copy 2D, row-by-row)
# Input is viewed as 3D (batch*M, N, record_len) so a (block_N, record_len)
# slice is genuinely 2D-contiguous (rows adjacent, stride=record_len) and can
# be loaded into a 2D UB buffer in a single DMA. Output is 2D (batch*N, M*record_len);
# the UB->GM write is a single 2D strided copy (block_N rows, record_len each,
# dstGap = y_cols - record_len), replacing the former block_N per-row DMAs.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _kernel_record_swap(batch, M, N, record_len, block_M, block_N, dtype):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    total_tasks = batch * m_num * n_num
    VEC_NUM = 2
    block_num = T.ceildiv(total_tasks, VEC_NUM)
    x_rows = batch * M
    y_rows = batch * N
    y_cols = M * record_len

    @T.prim_func
    def kernel(
        x_3d: T.Tensor((x_rows, N, record_len), dtype),  # type: ignore
        y_2d: T.Tensor((y_rows, y_cols), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task = cid * VEC_NUM + vid

            ub_2d = T.alloc_ub((block_N, record_len), dtype)

            if task < total_tasks:
                b = task // (m_num * n_num)
                rem = task % (m_num * n_num)
                bm = rem // n_num
                bn = rem % n_num

                with T.Scope("V"):
                    for mi in T.serial(block_M):
                        m_idx = bm * block_M + mi
                        if m_idx < M:
                            x_row = b * M + m_idx
                            y_row_start = b * N + bn * block_N
                            # GM -> UB: 2D contiguous load (block_N rows, record_len each)
                            T.copy(x_3d[x_row, bn * block_N : bn * block_N + block_N, :], ub_2d)
                            # UB -> GM: single 2D strided write (block_N rows, record_len each)
                            T.copy(
                                ub_2d,
                                y_2d[
                                    y_row_start : y_row_start + block_N,
                                    m_idx * record_len : (m_idx + 1) * record_len,
                                ],
                            )

    return kernel


# ---------------------------------------------------------------------------
# Path 3: stride fallback kernel (T.copy 1D dynamic slice)
# ---------------------------------------------------------------------------
def _build_offset_expr(task, n_dims, shape_list, stride_list):
    """Build a tir offset expression via Python static unroll.

    task is a tir expression (runtime value).
    shape_list and stride_list are Python tuples (compile-time constants).
    Returns a tir expression for the linear offset.
    """
    off = 0
    idx = task
    for d in range(n_dims):
        off = off + (idx % shape_list[d]) * stride_list[d]
        idx = idx // shape_list[d]
    return off


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _kernel_stride(total_numel, record_len, block_num, single_core_load, dtype, n_dims, out_shape, src_strides_mapped, dst_strides):
    total_records = total_numel // record_len
    VEC_NUM = 2

    @T.prim_func
    def kernel(
        x_flat: T.Tensor((total_numel,), dtype),  # type: ignore
        y_flat: T.Tensor((total_numel,), dtype),  # type: ignore
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub((record_len,), dtype)

            for t in T.serial(single_core_load):
                task = (cid * VEC_NUM + vid) * single_core_load + t
                if task < total_records:
                    src_off = _build_offset_expr(task, n_dims, out_shape, src_strides_mapped)
                    dst_off = _build_offset_expr(task, n_dims, out_shape, dst_strides)

                    T.copy(x_flat[src_off : src_off + record_len], row_ub)
                    T.copy(row_ub, y_flat[dst_off : dst_off + record_len])

    return kernel


# ---------------------------------------------------------------------------
# Perm topology analysis (host-side Python integer arithmetic)
# ---------------------------------------------------------------------------
def _classify_perm(perm, shape):
    """Analyze perm topology, return (path, params).

    Returns:
        ("identity", None)
        ("path1", (batch, M, N))           — 3D transpose, record_len=1
        ("path2", (batch, M, N, record_len)) — record aggregation
        ("fallback", (record_len,))        — stride fallback (with shared record_len)
    """
    n = len(perm)

    # 1. Strip invariant suffix -> record_len
    suffix_start = n
    while suffix_start > 0 and perm[suffix_start - 1] == suffix_start - 1:
        suffix_start -= 1

    record_len = 1
    for i in range(suffix_start, n):
        record_len *= shape[i]

    # 2. Strip invariant prefix -> batch
    prefix_end = 0
    while prefix_end < suffix_start and perm[prefix_end] == prefix_end:
        prefix_end += 1

    batch = 1
    for i in range(prefix_end):
        batch *= shape[i]

    # 3. Check if remaining [prefix_end:suffix_start] is A/B swap
    if prefix_end >= suffix_start:
        return "identity", None

    # Convert to relative indices (relative to prefix_end)
    rem_perm = [perm[i] - prefix_end for i in range(prefix_end, suffix_start)]
    rem_shape = shape[prefix_end:suffix_start]
    rem_n = len(rem_perm)

    # A/B swap: rem_perm = [B_group] + [A_group]
    # B_group starts at b_start_rel, A_group starts at 0
    b_start_rel = rem_perm[0]
    if b_start_rel <= 0:
        return "fallback", (record_len,)

    a_len = b_start_rel
    b_len = rem_n - a_len
    if b_len <= 0:
        return "fallback", (record_len,)

    # Verify B group: rem_perm[0:b_len] = [b_start_rel, ..., b_start_rel+b_len-1]
    for i in range(b_len):
        if rem_perm[i] != b_start_rel + i:
            return "fallback", (record_len,)

    # Verify A group: rem_perm[b_len:b_len+a_len] = [0, ..., a_len-1]
    for i in range(a_len):
        if rem_perm[b_len + i] != i:
            return "fallback", (record_len,)

    # Compute M (A group product) and N (B group product)
    M = 1
    for i in range(a_len):
        M *= rem_shape[i]
    N = 1
    for i in range(a_len, a_len + b_len):
        N *= rem_shape[i]

    if record_len > 1:
        return "path2", (batch, M, N, record_len)
    else:
        return "path1", (batch, M, N)


def _compute_strides(shape):
    """Compute row-major strides for a shape."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


# ---------------------------------------------------------------------------
# Host entry
# ---------------------------------------------------------------------------
def transpose(x: torch.Tensor, perm: list) -> torch.Tensor:
    """Transpose tensor dimensions (equivalent to torch.permute).

    Args:
        x: Input tensor (must be contiguous), 2D~8D.
        perm: Dimension permutation list.

    Returns:
        Output tensor with dimensions permuted.
    """
    # Validate perm
    ndim = x.ndim
    if len(perm) != ndim:
        raise ValueError(f"perm length {len(perm)} != input ndim {ndim}")
    if sorted(perm) != list(range(ndim)):
        raise ValueError(f"perm {perm} is not a valid permutation of [0, {ndim})")

    # bfloat16: zero-copy reinterpret as int16
    if x.dtype == torch.bfloat16:
        y_int16 = transpose(x.view(torch.int16), perm)
        return y_int16.view(torch.bfloat16)

    shape = list(x.shape)
    dtype_str = _torch_dtype_to_str(x.dtype)
    n = len(perm)

    path, params = _classify_perm(perm, shape)

    if path == "identity":
        return x.clone()

    out_shape = [shape[perm[i]] for i in range(n)]

    # Determine record_len early (needed for path 2/3)
    if path == "path2":
        record_len = params[3]
    elif path == "fallback":
        record_len = params[0] if params else 1
    else:
        record_len = 1

    # Check if path 2 needs to fall back to path 3 due to UB 32B alignment
    if path == "path2":
        elem_bytes = _dtype_elem_bytes(x.dtype)
        if record_len * elem_bytes % 32 != 0:
            path = "fallback"

    if path == "path1":
        batch, M, N = params
        use_int8_cast = x.dtype == torch.int8
        elem_bytes = _dtype_elem_bytes(x.dtype)

        # Determine block size based on dtype to fit UB budget (192KB).
        # Non-int8: 2 buffers (a_ub + b_ub), each block_M*block_N*elem_bytes.
        # int8: 4 buffers (a_ub + a_cal + b_cal + b_ub) = 6*block_M*block_N bytes.
        if x.dtype == torch.int64:
            # 2 * 64*64*8 = 64KB < 192KB; 64*8=512B is 32B-aligned
            block_M = 64
            block_N = 64
        elif elem_bytes == 2 and not use_int8_cast:
            # fp16/bf16/int16: widen block_N to 256 for larger GM load bursts
            # (load was BW-bound at small 256B bursts). 2 * 128*256*2 = 128KB < 192KB.
            # H=128, W=256 both 16-aligned for T.tile.transpose.
            block_M = 128
            block_N = 256
        else:
            block_M = 128
            block_N = 128

        x_3d = x.reshape(batch, M, N)

        cache_key = ("path1", batch, M, N, block_M, block_N, dtype_str, use_int8_cast)
        if cache_key not in _kernel_cache:
            _kernel_cache[cache_key] = _kernel_2d_transpose(batch, M, N, block_M, block_N, dtype_str, use_int8_cast)
        kernel = _kernel_cache[cache_key]

        y_3d = kernel(x_3d)
        return y_3d.reshape(out_shape)

    elif path == "path2":
        # UB alignment check already done above; if we reach here, path is still "path2"
        pass

    if path == "path2":
        batch, M, N = params[0], params[1], params[2]
        # record_len already determined above
        block_M = min(128, M)
        block_N = min(128, N)

        # Reshape to 3D (batch*M, N, record_len): the (block_N, record_len) tile is
        # genuinely 2D-contiguous, enabling a single DMA load into a 2D UB buffer.
        x_3d = x.reshape(batch * M, N, record_len)

        cache_key = ("path2", batch, M, N, record_len, block_M, block_N, dtype_str)
        if cache_key not in _kernel_cache:
            _kernel_cache[cache_key] = _kernel_record_swap(batch, M, N, record_len, block_M, block_N, dtype_str)
        kernel = _kernel_cache[cache_key]

        y_2d = kernel(x_3d)
        return y_2d.reshape(out_shape)

    else:  # fallback (path 3: stride fallback)
        # record_len already determined above (from _classify_perm or path 2 fallback)
        total_numel = x.numel()

        x_flat = x.reshape(-1)

        in_strides = _compute_strides(shape)
        out_strides = _compute_strides(out_shape)
        # Map input strides through perm: for output dim d, input dim is perm[d]
        src_strides_mapped = [in_strides[perm[d]] for d in range(n)]

        total_records = total_numel // record_len
        VEC_NUM = 2
        block_num = min(total_records, _CORE_NUM)
        eff_cores = block_num * VEC_NUM
        single_core_load = (total_records + eff_cores - 1) // eff_cores

        cache_key = (
            "path3",
            total_numel,
            record_len,
            block_num,
            single_core_load,
            dtype_str,
            n,
            tuple(out_shape),
            tuple(src_strides_mapped),
            tuple(out_strides),
        )
        if cache_key not in _kernel_cache:
            _kernel_cache[cache_key] = _kernel_stride(
                total_numel,
                record_len,
                block_num,
                single_core_load,
                dtype_str,
                n,
                tuple(out_shape),
                tuple(src_strides_mapped),
                tuple(out_strides),
            )
        kernel = _kernel_cache[cache_key]

        y_flat = kernel(x_flat)
        return y_flat.reshape(out_shape)


# ---------------------------------------------------------------------------
# Standalone run entry (smoke test, picked from test_transpose.py L0 case)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (1024, 1024)
    perm = [1, 0]
    x = (torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0).to(torch.float16).npu()
    y = transpose(x, perm)
    ref = torch.permute(x.cpu(), perm)
    torch.testing.assert_close(y.cpu().float(), ref.float(), rtol=1e-3, atol=1e-3)
    print("Kernel Output Match!")
