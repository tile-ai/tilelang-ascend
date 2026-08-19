import torch
import tilelang
from tilelang import language as T

CAST_MODE_LOW2HIGH = "CAST_NONE"

_TORCH_TO_TL_DTYPE = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}


def torch_dtype_to_tl(dtype):
    return _TORCH_TO_TL_DTYPE[dtype]


_kernel_cache = {}

_ALIGN = 8
_MAX_BLOCK_N = 2048
_VEC_NUM = 2
_BLOCK_M = 32
_SUB_BLOCK_M = _BLOCK_M // _VEC_NUM

_NO_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _argmax_wholereduce_kernel(M, N, block_N, in_dtype="float16"):
    cal_dtype = "float32"
    use_fp32_compute = in_dtype in ["float16", "bfloat16"]
    pad_val = -T.infinity(cal_dtype)
    m_num = T.ceildiv(M, _BLOCK_M)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], in_dtype),
        Out: T.Tensor([M, 8], "int64"),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_tile = T.alloc_ub([block_N], in_dtype)
            a_cal = T.alloc_ub([block_N], cal_dtype)
            idx_buf = T.alloc_ub([block_N], cal_dtype)
            mask = T.alloc_ub([block_N], cal_dtype)
            sel = T.alloc_ub([block_N], cal_dtype)
            max_val = T.alloc_ub([1], cal_dtype)
            min_idx = T.alloc_ub([1], cal_dtype)
            idx_out_i64 = T.alloc_ub([8], "int64")

            T.tile.createvecindex(idx_buf, 0)

            for ri in T.serial(_SUB_BLOCK_M):
                row = cid * _BLOCK_M + vid * _SUB_BLOCK_M + ri
                if row < M:
                    T.copy(A[row, 0:block_N], a_tile, pad_value=pad_val)

                    if use_fp32_compute:
                        T.tile.cast(a_cal, a_tile, CAST_MODE_LOW2HIGH, block_N)
                    else:
                        T.copy(a_tile, a_cal)

                    T.reduce_max(a_cal, max_val, dim=-1, clear=True)
                    T.tile.compare(mask, a_cal, max_val[0], "EQ")
                    T.tile.select(sel, mask, idx_buf, 999999.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_min(sel, min_idx, dim=-1, clear=True)
                    T.tile.cast(idx_out_i64, min_idx, "CAST_RINT", 8)
                    T.copy(idx_out_i64, Out[row, 0:8])
    return main


@tilelang.jit(out_idx=[1], pass_configs=_SYNC)
def _argmax_sort_kernel(M, N, block_N, in_dtype="float16"):
    cal_dtype = "float32"
    use_fp32_compute = in_dtype in ["float16", "bfloat16"]
    pad_val = -T.infinity(cal_dtype)
    m_num = T.ceildiv(M, _BLOCK_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], in_dtype),
        Out: T.Tensor([M, 8], "int64"),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            running_max = T.alloc_ub([8], cal_dtype)
            running_idx_i32 = T.alloc_ub([8], "int32")
            running_idx_i64 = T.alloc_ub([8], "int64")
            a_tile = T.alloc_ub([block_N], in_dtype)
            a_cal = T.alloc_ub([block_N], cal_dtype)
            tile_max = T.alloc_ub([1], cal_dtype)
            sort_dst = T.alloc_ub([block_N * 2], cal_dtype)
            tile_max_f = T.alloc_ub([8], cal_dtype)
            tile_idx_f = T.alloc_ub([8], cal_dtype)
            best_tile = T.alloc_ub([1], "int32")

            for ri in T.serial(_SUB_BLOCK_M):
                row = cid * _BLOCK_M + vid * _SUB_BLOCK_M + ri
                if row < M:
                    T.tile.fill(running_max, -T.infinity(cal_dtype))
                    T.tile.fill(running_idx_i32, 0)
                    T.tile.fill(best_tile, 0)

                    for bn in T.serial(n_num):
                        col_start = bn * block_N
                        T.copy(A[row, col_start : col_start + block_N], a_tile,
                               pad_value=pad_val)
                        if use_fp32_compute:
                            T.tile.cast(a_cal, a_tile, CAST_MODE_LOW2HIGH, block_N)
                        else:
                            T.copy(a_tile, a_cal)
                        T.reduce_max(a_cal, tile_max, dim=-1, clear=True)
                        if tile_max[0] > running_max[0]:
                            running_max[0] = tile_max[0]
                            best_tile[0] = bn

                    col_start = best_tile[0] * block_N
                    T.copy(A[row, col_start : col_start + block_N], a_tile,
                           pad_value=pad_val)
                    if use_fp32_compute:
                        T.tile.cast(a_cal, a_tile, CAST_MODE_LOW2HIGH, block_N)
                    else:
                        T.copy(a_tile, a_cal)
                    T.tile.sort(sort_dst, a_cal, block_N)
                    T.tile.gather_mask(tile_max_f, sort_dst, "P0101")
                    T.tile.gather_mask(tile_idx_f, sort_dst, "P1010")
                    running_idx_i32[0] = col_start + T.cast(tile_idx_f[0], "int32")
                    T.tile.cast(running_idx_i64, running_idx_i32, "CAST_NONE", 8)
                    T.copy(running_idx_i64, Out[row, 0:8])
    return main


def _get_kernel(M, N, tl_dtype):
    key = (M, N, tl_dtype)
    if key not in _kernel_cache:
        block_N = min(((N + 63) // 64) * 64, _MAX_BLOCK_N)
        if block_N < 64:
            block_N = 64
        n_num = (N + block_N - 1) // block_N
        if n_num <= 1:
            _kernel_cache[key] = _argmax_wholereduce_kernel(M, N, block_N, in_dtype=tl_dtype)
        else:
            _kernel_cache[key] = _argmax_sort_kernel(M, N, block_N, in_dtype=tl_dtype)
    return _kernel_cache[key]


def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    ndim = input.ndim
    if ndim == 0:
        return torch.tensor(0, dtype=torch.int64, device=input.device)
    dim = dim % ndim
    original_shape = list(input.shape)
    is_int = input.dtype in (torch.int32, torch.int64)

    if dim != ndim - 1:
        if is_int:
            input = input.to(torch.float32)
        reduce_size = input.shape[dim]
        non_reduce_shape = [input.shape[i] for i in range(ndim) if i != dim]
        outer_size = 1
        for s in non_reduce_shape:
            outer_size *= s
        perm_dims = [i for i in range(ndim) if i != dim] + [dim]
        x = input.permute(perm_dims).contiguous()
        x = x.reshape(outer_size, reduce_size)
    else:
        x = input if input.is_contiguous() else input.contiguous()
        if is_int:
            x = x.to(torch.float32)

    if is_int:
        tl_dtype = "float"
    elif input.dtype in (torch.float16, torch.bfloat16):
        tl_dtype = torch_dtype_to_tl(input.dtype)
        if torch.isnan(x).any():
            x = x.nan_to_num(nan=float("inf"))
    else:
        tl_dtype = torch_dtype_to_tl(input.dtype)

    N = x.shape[-1]
    outer = 1
    for s in x.shape[:-1]:
        outer *= s
    x_2d = x.reshape(outer, N)

    kernel = _get_kernel(outer, N, tl_dtype)
    out_2d = kernel(x_2d)[:, 0]

    if keepdim:
        if dim != ndim - 1:
            out_shape = non_reduce_shape + [1]
            out = out_2d.reshape(out_shape)
            full_keepdim = list(original_shape)
            full_keepdim[dim] = 1
            out = out.reshape(full_keepdim)
        else:
            transposed_keepdim_shape = list(x.shape[:-1]) + [1]
            out = out_2d.reshape(transposed_keepdim_shape)
            final_shape = list(original_shape)
            final_shape[dim] = 1
            out = out.reshape(final_shape)
    else:
        if dim != ndim - 1:
            out = out_2d.reshape(non_reduce_shape)
        else:
            out = out_2d.reshape(x.shape[:-1])

    return out


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _make_x(shape, dtype_str, value_range):
    torch_dtype = DTYPE_MAP[dtype_str]
    lo, hi = value_range
    if dtype_str in ("float16", "float32", "bfloat16"):
        x = (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo).to(torch_dtype)
    else:
        x = torch.randint(int(lo), int(hi) + 1, shape, dtype=torch_dtype)
    return x.npu()


def run_arg_max(case_id, shape, dtype_str, dim, keepdim, value_range):
    x = _make_x(shape, dtype_str, value_range)
    y = arg_max(x, dim=dim, keepdim=keepdim)
    ref = torch.argmax(x, dim=dim, keepdim=keepdim)

    if not torch.equal(y.cpu(), ref.cpu()):
        max_diff = (y.cpu() != ref.cpu()).sum().item()
        raise AssertionError(f"argmax mismatch: {max_diff} elements differ")

    print(
        f"Case {case_id}: PASSED  "
        f"(shape={shape}, dtype={dtype_str}, dim={dim}, keepdim={keepdim})"
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    # (case_id, shape, dtype, dim, keepdim, value_range)
    test_cases = [
        (1, [1024, 1024], "float16", -1, False, [-1, 1]),
        (2, [2048, 2048], "float32", -1, False, [-2, 2]),
        (3, [4096, 4096], "bfloat16", -1, False, [-3, 3]),
        (4, [8192, 8192], "int32", -1, False, [-1000, 1000]),
        (5, [4096, 4096], "int64", -1, False, [-100000, 100000]),
        (6, [1024, 1024], "float16", 0, False, [-1, 1]),
        (7, [2048, 2048], "float32", 0, True, [-2, 2]),
        (8, [4096, 4096], "bfloat16", 1, True, [-3, 3]),
        (9, [1023, 1023], "float16", -1, False, [-1, 1]),
        (10, [1009, 1021], "float32", 0, False, [-1, 2]),
        (11, [1537, 769], "bfloat16", 1, False, [-5, 10]),
        (12, [363, 367, 373], "float16", 1, False, [-50, 100]),
        (13, [363, 367, 373], "float16", 2, True, [-50, 100]),
        (14, [363, 367, 373], "float32", 0, False, [-1, 1]),
        (15, [3, 7, 13, 4001], "bfloat16", 1, False, [-88, 88]),
        (16, [3, 7, 13, 4001], "float32", 3, True, [-1, 1]),
        (17, [3, 7, 13, 4001], "float16", 0, False, [-1, 1]),
        (18, [1000003], "float32", 0, False, [-100, 100]),
        (19, [1000003], "float16", 0, False, [-1, 1]),
        (20, [11, 13, 17, 67, 67], "bfloat16", 1, True, [-88, 88]),
        (21, [512, 2049], "bfloat16", -1, False, [-0.5, 0.5]),
        (22, [255, 8193], "float16", 0, False, [-1, 3]),
        (23, [4097, 511], "float32", -1, False, [-1000, 1000]),
        (24, [2, 511, 2049], "bfloat16", 1, True, [-0.2, 0.2]),
        (25, [4, 255, 2049], "float32", 2, False, [-3, 6]),
        (26, [32, 64], "int32", -1, False, [-100, 100]),
        (27, [32, 64], "int64", 0, True, [-100, 100]),
        (28, [1, 1024], "float16", -1, False, [-1, 1]),
        (29, [1, 1024], "float32", 0, True, [-1, 1]),
        (30, [1024, 1], "bfloat16", -1, False, [-1, 1]),
    ]

    print("=" * 70)
    print("ArgMax TileLang-Ascend 测试 (torch.argmax 语义)")
    print(f"共 {len(test_cases)} 个测试用例")
    print("=" * 70)

    passed = 0
    failed = 0
    for case_id, shape, dtype, dim, keepdim, value_range in test_cases:
        try:
            run_arg_max(case_id, shape, dtype, dim, keepdim, value_range)
            passed += 1
        except Exception as e:
            print(f"Case {case_id}: FAILED - {e}")
            failed += 1

    print("=" * 70)
    print(f"测试完成: {passed} passed, {failed} failed")
    if failed == 0:
        print("Test Passed!")
