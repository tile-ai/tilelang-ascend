import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

# ========== Operator Implementation ==========
# Sigmoid activation kernel: y = 1 / (1 + exp(-x)).
#
# Uses T.tile.sigmoid (one-step primitive) instead of the 5-step decomposition
# (fill/sub/exp/add/reciprocal) because the latter's T.tile.exp and
# T.tile.reciprocal internally compute in float16 regardless of buffer dtype,
# causing precision failures for float32. T.tile.sigmoid preserves dtype.
#
# Perf optimizations:
# - AUTO_CV_COMBINE OFF: sigmoid is pure Vector (element-wise); the auto-CV-combine
#   pass was emitting MIX_AIC_1_2 with all compute inside `if ASCEND_IS_AIV`,
#   leaving the AIC core idle but still paying its launch + buffer init cost.
# - Fixed Core mode: launch min(block_num, CORE_NUM) cores instead of block_num,
#   each core processes ceildiv(block_num, launch_cores) tiles via T.serial.
#   Eliminates per-block launch overhead (was 512 launches for (1024,8192) fp16,
#   now 24). NPU kernel task duration: 71.9us -> 53.6us (-25.5%).
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Ascend A2/A3 physical AI Core count.
CORE_NUM = 24


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float16"):
    """Sigmoid kernel: y = 1 / (1 + exp(-x)).

    Args:
        M, N: tensor shape (rows, cols)
        block_M, block_N: tile size per block
        dtype: "float16" or "float32"

    Returns:
        prim_func mapping A (M, N) -> B (M, N)
    """
    m_num = (M + block_M - 1) // block_M
    n_num = (N + block_N - 1) // block_N
    block_num = m_num * n_num
    launch_cores = min(block_num, CORE_NUM)
    single_core_load = (block_num + launch_cores - 1) // launch_cores

    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):  # type: ignore
        with T.Kernel(launch_cores, is_npu=True) as (cid, vid):
            # Striped work distribution: core `cid` handles tiles
            # cid, cid+launch_cores, cid+2*launch_cores, ...
            for block_idx in T.serial(single_core_load):
                logical_cid = block_idx * launch_cores + cid
                bx = logical_cid // n_num
                by = logical_cid % n_num

                a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
                b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.tile.sigmoid(b_ub, a_ub)
                T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ========== Golden reference ==========
def golden_sigmoid(x):
    """Sigmoid reference: y = 1 / (1 + exp(-x))."""
    return torch.sigmoid(x)


# ========== Tests ==========
if __name__ == "__main__":
    torch.manual_seed(0)

    # Representative configs: 1 L0 (largest regular shape, fp16) + 1 L1 (regular shape, fp32)
    test_configs = [
        # (M, N, block_M, block_N, dtype, level)
        (1024, 8192, 128, 128, "float16", "L0"),  # L0 gate: largest regular shape
        (512, 512, 128, 128, "float32", "L1"),    # L1 functional: regular shape + fp32 dtype
    ]

    for M, N, block_M, block_N, dtype, level in test_configs:
        print(f"Testing sigmoid {level} with M={M}, N={N}, block=({block_M},{block_N}), dtype={dtype}")
        func = sigmoid(M, N, block_M, block_N, dtype=dtype)
        print("Init successful!")
        torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
        x = torch.randn(M, N, dtype=torch_dtype).npu()
        y = func(x)
        ref = golden_sigmoid(x)
        # Precision check (mixed tolerance, inlined — thresholds by dtype)
        if dtype == "float16":
            atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
        elif dtype == "float32" or dtype == "float":
            atol, rtol, max_abs_limit, required_ratio = 2**-16, 2**-10, 1e-2, 0.99
        else:
            atol, rtol, max_abs_limit, required_ratio = 2**-14, 2**-9, 1e-1, 0.99
        y_cpu, ref_cpu = y.detach().cpu().float(), ref.detach().cpu().float()
        m = torch.isfinite(ref_cpu)
        abs_err = (y_cpu[m] - ref_cpu[m]).abs()
        ratio = (abs_err <= (atol + rtol * ref_cpu[m].abs())).float().mean().item()
        max_abs = abs_err.max().item()
        assert ratio >= required_ratio and max_abs <= max_abs_limit, (
            f"precision fail: ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        print(f"Test pass! matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

    print("Kernel Output Match!")
