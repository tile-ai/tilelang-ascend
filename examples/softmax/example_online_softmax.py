import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 0.1),
        "bfloat16": (2**-10, 2**-6, 1.0),
        "float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    dtype_name = str(dtype).replace("torch.", "")
    dtype_name = "float8_e4m3" if "float8_e4m3" in dtype_name else "float8_e5m2" if "float8_e5m2" in dtype_name else dtype_name
    atol, rtol, limit = table.get(dtype_name, table["float16"])
    if not (torch.equal(torch.isnan(actual), torch.isnan(golden)) and torch.equal(torch.isinf(actual), torch.isinf(golden))):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    """
    Safe softmax with online normalizer.
    Supports float, float16, and bfloat16.

    Algorithm:
        m_0 = -inf
        s_0 = 0
        for j in [1, N]:
            m_j = max(m_{j-1}, x_j)
            s_j = s_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
        for j in [1, N]:
            y_j = exp(x_j - m_N) / s_N
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        # One core process one block row
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)
            # First pass: compute max and sum
            for by in T.serial(n_num):
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )  # Load input
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)  # Cast to compute dtype if needed
                T.reduce_max(a_cal, tile_max, dim=-1)  # Compute tile max
                T.tile.max(tile_max, prev_max, tile_max)  # m_j = max(m_{j-1}, x_j)
                T.tile.sub(tmp_exp, prev_max, tile_max)  # m_{j-1} - m_j
                T.tile.exp(tmp_exp, tmp_exp)  # exp(m_{j-1} - m_j)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)  # s_{j-1} * exp(m_{j-1} - m_j)
                T.tile.broadcast(tile_max_2d, tile_max)  # Broadcast tile_max for sub operation
                T.tile.sub(a_cal, a_cal, tile_max_2d)  # x_j - m_j
                T.tile.exp(a_cal, a_cal)  # exp(x_j - m_j)
                T.reduce_sum(a_cal, tile_sum, dim=-1)  # sum_j exp(x_j - m_j)
                T.tile.add(prev_sum, tile_sum, tmp_exp)  # s_j = s_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
                T.copy(tile_max, prev_max)

            # Second pass: compute final output
            # After first pass, prev_max holds m_N, prev_sum holds s_N
            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
                )  # Load input
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)  # Cast to compute dtype if needed
                T.tile.sub(a_cal, a_cal, prev_max_2d)  # x_j - m_N
                T.tile.exp(a_cal, a_cal)  # exp(x_j - m_N)
                T.tile.div(a_cal, a_cal, prev_sum_2d)  # y_j = exp(x_j - m_N) / s_N
                cast_or_copy(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)  # Cast back if needed
                T.copy(
                    a, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
                )  # Store output

    return main


torch.manual_seed(0)
test_configs = [
    (34, 130, 32, 32, "float"),
    (34, 130, 32, 32, "float16"),
    (34, 130, 32, 32, "bfloat16"),
    (1024, 51200, 128, 128, "float"),
    (1024, 51200, 128, 128, "float16"),
    (1024, 51200, 128, 128, "bfloat16"),
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing online_softmax with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = online_softmax(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    a = torch.randn(M, N, dtype=getattr(torch, dtype) if dtype != "float" else torch.float32).npu()
    b = func(a)
    ref_b = torch.nn.functional.softmax(a, dim=1)
    passed, ratio, max_abs = _check_precision(b, ref_b, b.dtype)
    assert passed, f"dtype={b.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Test pass!")

print("\nKernel Output Match!")
