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
    actual_cpu, golden_cpu = actual.detach().cpu(), golden.detach().cpu()
    if actual_cpu.shape != golden_cpu.shape:
        return False, 0.0, float("inf")
    name = str(dtype).replace("torch.", "")
    if name.startswith("float8_e4m3"):
        name = "float8_e4m3"
    if name.startswith("float8_e5m2"):
        name = "float8_e5m2"
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatch = (actual_cpu != golden_cpu).sum().item()
        return mismatch == 0, 1.0 - mismatch / max(actual_cpu.numel(), 1), 0.0 if mismatch == 0 else float("inf")
    atol, rtol, limit = table.get(name, table["float16"])
    actual_cpu, golden_cpu = actual_cpu.float(), golden_cpu.float()
    special = ~torch.isfinite(golden_cpu)
    if special.any() and (
        not torch.equal(torch.isnan(actual_cpu[special]), torch.isnan(golden_cpu[special]))
        or not torch.equal(torch.isinf(actual_cpu[special]), torch.isinf(golden_cpu[special]))
        or not torch.equal(actual_cpu[special][torch.isinf(golden_cpu[special])], golden_cpu[special][torch.isinf(golden_cpu[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden_cpu)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual_cpu[finite] - golden_cpu[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, max_abs = (error <= atol + rtol * golden_cpu[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and max_abs <= limit, ratio, max_abs


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def layer_norm(M, N, block_M, block_N, eps=1e-5, dtype="float"):
    """
    Layer Norm
    """

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            a_ub = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            sum_i = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            sum_square_i = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            sum_ub = T.alloc_ub([sub_block_M], cal_dtype)
            sum_square_ub = T.alloc_ub([sub_block_M], cal_dtype)
            mean_ub = T.alloc_ub([sub_block_M, 1], cal_dtype)
            mean_square_ub = T.alloc_ub([sub_block_M, 1], cal_dtype)

            with T.Scope("V"):
                # Initialize
                T.tile.fill(sum_i, 0.0)
                T.tile.fill(sum_square_i, 0.0)
                T.tile.fill(sum_ub, 0.0)
                T.tile.fill(sum_square_ub, 0.0)
                T.tile.fill(mean_ub, N)
                T.tile.fill(mean_square_ub, N)

                # Accumulation
                for by in T.serial(n_num):
                    T.copy(
                        A[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_ub,
                    )
                    cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                    T.tile.add(sum_i, sum_i, a_cal)
                    T.tile.mul(a_cal, a_cal, a_cal)
                    T.tile.add(sum_square_i, sum_square_i, a_cal)

                # Reduce
                T.reduce_sum(sum_i, sum_ub, dim=-1)
                T.reduce_sum(sum_square_i, sum_square_ub, dim=-1)

                # Compute mean and variance
                T.tile.div(mean_ub, sum_ub, mean_ub)
                T.tile.div(mean_square_ub, sum_square_ub, mean_square_ub)
                T.tile.mul(sum_ub, mean_ub, mean_ub)
                T.tile.sub(mean_square_ub, mean_square_ub, sum_ub)
                T.tile.fill(sum_ub, eps)
                T.tile.add(mean_square_ub, mean_square_ub, sum_ub)
                T.tile.sqrt(mean_square_ub, mean_square_ub)

                T.tile.broadcast(sum_i, mean_ub)
                T.tile.broadcast(sum_square_i, mean_square_ub)

                # Normalize
                for by in T.serial(n_num):
                    T.copy(
                        A[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_ub,
                    )
                    cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                    T.tile.sub(a_cal, a_cal, sum_i)
                    T.tile.div(a_cal, a_cal, sum_square_i)

                    cast_or_copy(a_ub, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                    T.copy(
                        a_ub,
                        B[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )

    return main


torch.manual_seed(0)
test_configs = [
    (34, 34, 32, 32, "float"),
    (34, 34, 32, 32, "float16"),
    (34, 34, 32, 32, "bfloat16"),
    (270, 270, 64, 64, "float"),
    (1030, 1030, 128, 128, "float16"),
    (1024, 51200, 128, 128, "bfloat16"),
]

for M, N, block_M, block_N, dtype in test_configs:
    print(f"Testing layer_norm with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = layer_norm(M, N, block_M, block_N, dtype=dtype)
    print("Init successful!")
    torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
    a = torch.randn(M, N, dtype=torch_dtype).npu()
    b = func(a)
    ref_b = torch.layer_norm(a, normalized_shape=[N])
    passed, ratio, max_abs = _check_precision(b, ref_b, b.dtype)
    assert passed, f"dtype={b.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
    print("Test pass!")

print("Kernel Output Match!")
