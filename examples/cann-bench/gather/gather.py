import math
import torch
import tilelang
from tilelang import language as T

tilelang.disable_cache()

# ---------------------------------------------------------------------------
# Pass configs (from cann_bench/_common.py + gather.py)
# ---------------------------------------------------------------------------
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

GATHER_PASS_CONFIGS = {
    **PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

ELEM_SIZE_MAP = {
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "float": 4,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}

ALIGN_K = 32
UB_SIZE_LIMIT = 256 * 1024

_kernel_cache = {}


def _torch_dtype_to_str(dtype):
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1], pass_configs=GATHER_PASS_CONFIGS)
def _gather_kernel_standard(outer_size, M, K, TILE_OUTER, dtype, idx_dtype, elem_size, M_orig):
    # M is the UB buffer aligned size for x; M_orig is the original GM size for x.
    # x (data, large) is no longer host-side F.pad; instead kernel-internal T.copy(pad_value=0) fills the tail.
    # index/output are still passed with aligned size K (K is typically much smaller than x, and int64 index
    # non-aligned copy would trigger UB overflow, so host-side alignment is retained).
    block_num = outer_size // TILE_OUTER
    zero = T.cast(0, dtype)

    @T.prim_func
    def kernel(
        x_2d: T.Tensor([outer_size, M_orig], dtype),
        index_2d: T.Tensor([outer_size, K], idx_dtype),
        output_2d: T.Tensor([outer_size, K], dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            tile_start = cid * TILE_OUTER
            x_row = T.alloc_ub([1, M], dtype)
            idx_row = T.alloc_ub([1, K], idx_dtype)
            off_row_i32 = T.alloc_ub([1, K], "int32")
            off_row_u32 = T.alloc_ub([1, K], "uint32")
            out_row = T.alloc_ub([1, K], dtype)

            for i in T.serial(0, TILE_OUTER):
                T.copy(x_2d[tile_start + i, 0:M_orig], x_row[0, 0:M_orig], pad_value=zero)
                T.copy(index_2d[tile_start + i, 0], idx_row[0, :])

                if idx_dtype == "int32":
                    T.tile.mul(off_row_i32, idx_row, elem_size)
                else:
                    idx_i32 = T.alloc_ub([1, K], "int32")
                    T.tile.cast(idx_i32, idx_row, "CAST_NONE", K)
                    T.tile.mul(off_row_i32, idx_i32, elem_size)

                T.reinterpretcast(off_row_u32, off_row_i32, "uint32_t")
                T.tile.gather(out_row, x_row, off_row_u32, 0)
                T.copy(out_row[0, :], output_2d[tile_start + i, 0])

    return kernel


@tilelang.jit(out_idx=[-1], pass_configs=GATHER_PASS_CONFIGS)
def _gather_kernel_int64(outer_size, M2, K, TILE_OUTER, idx_dtype, word_off):
    # word_off = 0 -> gather low int32 word (byte offset 8k)
    # word_off = 1 -> gather high int32 word (byte offset 8k + 4)
    block_num = outer_size // TILE_OUTER
    byte_off = word_off * 4

    @T.prim_func
    def kernel(
        x_int32: T.Tensor([outer_size, M2], "int32"),
        index_2d: T.Tensor([outer_size, K], idx_dtype),
        out_int32: T.Tensor([outer_size, K], "int32"),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            tile_start = cid * TILE_OUTER
            x_row = T.alloc_ub([1, M2], "int32")
            idx_row = T.alloc_ub([1, K], idx_dtype)
            idx_doubled = T.alloc_ub([1, K], "int32")
            off_i32 = T.alloc_ub([1, K], "int32")
            off_u32 = T.alloc_ub([1, K], "uint32")
            out_row = T.alloc_ub([1, K], "int32")

            for i in T.serial(0, TILE_OUTER):
                T.copy(x_int32[tile_start + i, 0], x_row[0, :])
                T.copy(index_2d[tile_start + i, 0], idx_row[0, :])

                if idx_dtype != "int32":
                    idx_tmp = T.alloc_ub([1, K], "int32")
                    T.tile.cast(idx_tmp, idx_row, "CAST_NONE", K)
                    T.tile.mul(idx_doubled, idx_tmp, 2)
                else:
                    T.tile.mul(idx_doubled, idx_row, 2)

                # byte offset = idx * 8 (+ 4 for the high word)
                T.tile.mul(off_i32, idx_doubled, 4)
                if byte_off != 0:
                    T.tile.add(off_i32, off_i32, byte_off)
                T.reinterpretcast(off_u32, off_i32, "uint32_t")
                T.tile.gather(out_row, x_row, off_u32, 0)
                T.copy(out_row[0, :], out_int32[tile_start + i, 0])

    return kernel


def _get_kernel(kernel_fn, *args):
    key = (kernel_fn.__name__,) + args
    if key not in _kernel_cache:
        _kernel_cache[key] = kernel_fn(*args)
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Host entry (torch.gather semantics)
# ---------------------------------------------------------------------------
def gather(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # int8 hardware gather does not support 1-byte granularity fetch (output is always 0);
    # promote to int32 (4-byte aligned) gather, then cast back to int8.
    if x.dtype == torch.int8:
        y_i32 = gather(x.to(torch.int32), index, dim=dim)
        return y_i32.to(torch.int8)

    x_shape = list(x.shape)
    idx_shape = list(index.shape)
    ndim = len(x_shape)
    dtype_str = _torch_dtype_to_str(x.dtype)
    idx_dtype_str = _torch_dtype_to_str(index.dtype)
    elem_size = ELEM_SIZE_MAP.get(dtype_str, 4)

    M = x_shape[dim]
    row_bytes = M * elem_size
    if row_bytes > UB_SIZE_LIMIT:
        # Gather-axis single row exceeds UB capacity: tile along the gather axis.
        # Each chunk performs an independent gather (in-chunk index = global index - chunk start),
        # then a mask selects results whose index actually falls within the chunk and merges them.
        # Budget is half of UB, leaving headroom for idx/offset/out and other buffers.
        max_chunk_M = (UB_SIZE_LIMIT // 2) // elem_size
        max_chunk_M = max((max_chunk_M // ALIGN_K) * ALIGN_K, ALIGN_K)

        result = None
        start = 0
        while start < M:
            size = min(max_chunk_M, M - start)
            local_x = x.narrow(dim, start, size).contiguous()
            local_index = index - start
            # Clamp to valid range; out-of-bounds in-chunk values will be discarded by mask
            local_index_clamped = local_index.clamp_(0, size - 1)
            out_chunk = gather(local_x, local_index_clamped, dim=dim)
            mask = (index >= start) & (index < start + size)
            if result is None:
                result = out_chunk
            else:
                result = torch.where(mask, out_chunk, result)
            start += size
        return result

    if dtype_str == "float":
        dtype_str = "float32"
        elem_size = 4

    perm = list(range(ndim))
    perm.remove(dim)
    perm.append(dim)

    perm_inv = [0] * ndim
    for i, p in enumerate(perm):
        perm_inv[p] = i

    x_t = x.permute(*perm).contiguous()
    index_t = index.permute(*perm).contiguous()

    K = idx_shape[dim]

    outer_size = 1
    for i in range(ndim):
        if i != dim:
            outer_size *= idx_shape[i]

    # Under torch.gather semantics, non-dim dims satisfy index.shape[i] <= x.shape[i].
    # After permute, dim is moved to the end; the leading dims are non-gather dims.
    # x_t's non-dim dims may be larger than index_t; crop to index_t's outer shape first,
    # otherwise flattening to 2D would misalign row indices with index_2d.
    idx_outer_shape = list(index_t.shape[:-1])
    if list(x_t.shape[:-1]) != idx_outer_shape:
        slices = tuple(slice(0, s) for s in idx_outer_shape) + (slice(None),)
        x_t = x_t[slices].contiguous()

    x_2d = x_t.reshape(-1, M).contiguous()
    index_2d = index_t.reshape(outer_size, K).contiguous()

    TILE_OUTER = 16
    padded_outer = math.ceil(outer_size / TILE_OUTER) * TILE_OUTER

    if padded_outer > outer_size:
        x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, padded_outer - x_2d.shape[0]))
        index_2d = torch.nn.functional.pad(index_2d, (0, 0, 0, padded_outer - outer_size))

    if dtype_str == "int64":
        x_int32 = x_2d.view(torch.int32)
        M2 = 2 * M

        kernel_lo = _get_kernel(_gather_kernel_int64, padded_outer, M2, K, TILE_OUTER, idx_dtype_str, 0)
        out_lo = kernel_lo(x_int32, index_2d)
        out_lo = out_lo[:outer_size, :].clone()

        kernel_hi = _get_kernel(_gather_kernel_int64, padded_outer, M2, K, TILE_OUTER, idx_dtype_str, 1)
        out_hi = kernel_hi(x_int32, index_2d)
        out_hi = out_hi[:outer_size, :].clone()

        out_int32 = torch.stack([out_lo, out_hi], dim=-1).contiguous()
        output_2d = out_int32.view(torch.int64).squeeze(-1)

    else:
        M_ALIGNED = ((M + ALIGN_K - 1) // ALIGN_K) * ALIGN_K
        # K only needs ALIGN_K alignment; no need to align with M.
        # gather source (M) and target (K) can differ in length (int64 kernel too);
        # the old code max(..., M_ALIGNED) would inflate K when M is large, causing UB overflow/segfault.
        K_PADDED = ((K + ALIGN_K - 1) // ALIGN_K) * ALIGN_K
        K_ORIG = K
        M_ORIG = M

        # x (data, large) is no longer host-side F.pad; kernel-internal T.copy(pad_value=0) fills the tail.
        # index/output remain aligned to K_PADDED (K is much smaller than x; int64 index non-aligned copy causes UB overflow).
        if K_PADDED > K:
            index_2d = torch.nn.functional.pad(index_2d, (0, K_PADDED - K), value=0)
            K = K_PADDED

        kernel = _get_kernel(
            _gather_kernel_standard,
            padded_outer,
            M_ALIGNED,
            K,
            TILE_OUTER,
            dtype_str,
            idx_dtype_str,
            elem_size,
            M_ORIG,
        )
        output_2d = kernel(x_2d, index_2d)
        output_2d = output_2d[:outer_size, :K_ORIG]

    out_t_shape = [idx_shape[p] for p in perm]
    output_t = output_2d.reshape(out_t_shape)
    output = output_t.permute(*perm_inv).contiguous()

    return output


# ---------------------------------------------------------------------------
# Golden reference (torch.gather semantics)
# ---------------------------------------------------------------------------
def torch_gather(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.gather(x, dim, index)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
}

# Precision thresholds (float types use relative/absolute error; integers require exact match)
RTOL_MAP = {"float16": 1e-3, "bfloat16": 8e-3, "float32": 1e-4}
ATOL_MAP = {"float16": 1e-3, "bfloat16": 8e-3, "float32": 1e-5}


def _make_x(x_shape, x_dtype_str, value_range):
    torch_dtype = DTYPE_MAP[x_dtype_str]
    lo, hi = value_range

    def _sf(v):
        if isinstance(v, str):
            return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[v]
        return v

    lo, hi = _sf(lo), _sf(hi)

    if x_dtype_str in ("float16", "float32", "bfloat16"):
        if isinstance(lo, float) and math.isnan(lo):
            return torch.full(x_shape, float("nan"), dtype=torch_dtype, device="npu")
        if isinstance(lo, float) and (math.isinf(lo) or math.isinf(hi)):
            base = torch.randn(x_shape, dtype=torch_dtype, device="npu")
            base.view(-1)[0] = float("inf")
            if base.numel() > 1:
                base.view(-1)[1] = float("-inf")
            return base
        if lo == hi:
            return torch.full(x_shape, float(lo), dtype=torch_dtype, device="npu")
        return (torch.rand(x_shape, dtype=torch.float32, device="npu") * (hi - lo) + lo).to(torch_dtype)
    else:
        return torch.randint(int(lo), int(hi) + 1, x_shape, dtype=torch_dtype, device="npu")


def run_gather(case_id, x_shape, idx_shape, x_dtype_str, idx_dtype_str, dim, value_range):
    idx_torch_dtype = DTYPE_MAP[idx_dtype_str]

    x = _make_x(x_shape, x_dtype_str, value_range[0])

    # Index elements must fall within [0, x.shape[dim])
    dim_size = x_shape[dim]
    index = torch.randint(0, dim_size, idx_shape, dtype=idx_torch_dtype, device="npu")

    y = gather(x, index, dim=dim)
    ref = torch_gather(x, index, dim=dim)

    if x_dtype_str in ("int8", "int32", "int64"):
        ok = torch.equal(y.cpu(), ref.cpu())
        if not ok:
            raise AssertionError("integer gather mismatch")
    else:
        rtol = RTOL_MAP.get(x_dtype_str, 1e-2)
        atol = ATOL_MAP.get(x_dtype_str, 1e-2)
        y_c = y.cpu().float()
        ref_c = ref.cpu().float()
        # NaN positions only need to match (NaN != NaN)
        nan_mask = torch.isnan(ref_c)
        if nan_mask.any():
            if not torch.equal(torch.isnan(y_c), nan_mask):
                raise AssertionError("NaN pattern mismatch")
            y_c = torch.where(nan_mask, torch.zeros_like(y_c), y_c)
            ref_c = torch.where(nan_mask, torch.zeros_like(ref_c), ref_c)
        torch.testing.assert_close(y_c, ref_c, rtol=rtol, atol=atol, equal_nan=True)

    print(f"Case {case_id}: PASSED  (x={x_shape}, idx={idx_shape}, x_dtype={x_dtype_str}, idx_dtype={idx_dtype_str}, dim={dim})")


if __name__ == "__main__":
    torch.manual_seed(42)

    # (case_id, x_shape, idx_shape, x_dtype, idx_dtype, dim, [x_range, idx_range])
    test_cases = [
        (1, [1024, 1024], [512, 1024], "float16", "int32", 0, [[-1, 1], [0, 1023]]),
        (2, [2048, 2048], [1024, 2048], "float32", "int64", 0, [[-2, 2], [0, 2047]]),
        (3, [4096, 4096], [2048, 4096], "bfloat16", "int32", 0, [[-3, 3], [0, 4095]]),
        (4, [8192, 8192], [4096, 8192], "int32", "int32", 0, [[-1000, 1000], [0, 8191]]),
        (5, [4096, 4096], [4096, 4096], "int64", "int64", 0, [[-100000, 100000], [0, 4095]]),
        (6, [16384, 8192], [8192, 8192], "int8", "int32", 0, [[-128, 127], [0, 16383]]),
        (7, [1023, 1023], [511, 1023], "float16", "int64", 1, [[-0.1, 0.1], [0, 1022]]),
        (8, [1009, 1021], [505, 1021], "float32", "int32", 0, [[-1, 2], [0, 1008]]),
        (9, [1537, 769], [769, 769], "bfloat16", "int64", 0, [[-5, 10], [0, 1536]]),
        (10, [363, 367, 373], [181, 367, 373], "float16", "int32", 1, [[-50, 100], [0, 362]]),
        (11, [2049, 513], [1024, 513], "float32", "int64", 0, [[-65504, 65504], [0, 2048]]),
        (12, [3, 7, 13, 4001], [2, 7, 13, 4001], "bfloat16", "int32", 1, [[-88, 88], [0, 2]]),
        (13, [1000003], [1000], "float32", "int64", 0, [["-inf", "inf"], [0, 1000002]]),
        (14, [11, 13, 17, 67, 67], [7, 13, 17, 67, 67], "float16", "int32", 1, [["nan", "nan"], [0, 10]]),
        (15, [3, 7, 11, 13, 1013], [2, 7, 11, 13, 1013], "float32", "int64", 1, [[0, 0], [0, 2]]),
        (16, [512, 2049], [256, 2049], "bfloat16", "int32", 0, [[-0.5, 0.5], [0, 511]]),
        (17, [255, 8193], [127, 8193], "float16", "int64", 0, [[-1, 3], [0, 254]]),
        (18, [4097, 511], [2048, 511], "float32", "int32", 0, [[-1000, 1000], [0, 4096]]),
        (19, [2, 511, 2049], [2, 255, 2049], "bfloat16", "int64", 1, [[-0.2, 0.2], [0, 510]]),
        (20, [4, 255, 2049], [4, 127, 2049], "float32", "int32", 2, [[-3, 6], [0, 254]]),
    ]

    print("=" * 70)
    print("Gather TileLang-Ascend 测试 (torch.gather 语义)")
    print(f"共 {len(test_cases)} 个测试用例")
    print("=" * 70)

    passed = 0
    failed = 0
    for case_id, x_shape, idx_shape, x_dtype, idx_dtype, dim, value_range in test_cases:
        try:
            run_gather(case_id, x_shape, idx_shape, x_dtype, idx_dtype, dim, value_range)
            passed += 1
        except Exception as e:
            print(f"Case {case_id}: FAILED - {e}")
            failed += 1

    print("=" * 70)
    print(f"测试完成: {passed} passed, {failed} failed")
    if failed == 0:
        print("Test Passed!")
