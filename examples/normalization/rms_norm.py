import logging

import torch
import tilelang
from tilelang import language as T

logger = logging.getLogger(__name__)

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


def _get_optimized_tiling(M, N, block_M_in, block_N_in, vec_num):
    budget = block_M_in * block_N_in
    ideal_n = budget // 16

    block_N = min(N // 2, ideal_n)
    if block_N < 128:
        block_N = 128 if N >= 256 else N

    while N % block_N != 0:
        block_N -= 1
        if block_N <= 0:
            block_N = 1
            break

    block_M = budget // block_N
    if M % block_M != 0:
        block_M = block_M_in if M % block_M_in == 0 else vec_num
    return block_M, block_N


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def _rms_norm(M, N, block_M_in, block_N_in, eps=1e-5, dtype="float"):
    VEC_NUM = 2
    block_M, block_N = _get_optimized_tiling(M, N, block_M_in, block_N_in, VEC_NUM)
    m_num = M // block_M
    n_num = N // block_N
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N

    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def tilelang_rms_norm(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # Use a single accumulator to save massive UB space and prevent OOM.
            a_ub_0 = T.alloc_ub([ROWS, block_N], dtype)
            a_ub_1 = T.alloc_ub([ROWS, block_N], dtype)
            a_ub_cast_0 = T.alloc_ub([ROWS, block_N], acc_dtype)
            a_ub_cast_1 = T.alloc_ub([ROWS, block_N], acc_dtype)

            # SINGLE accumulator: No need for double buffering here!
            sum_sq_acc = T.alloc_ub([ROWS, block_N], acc_dtype)
            sum_sq_row = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            inv_rms_tile = T.alloc_ub([ROWS, block_N], acc_dtype)

            row_start = cid * block_M + vid * ROWS

            # Initialize single accumulator
            T.tile.fill(sum_sq_acc, 0.0)

            # Step 1: Accumulate squares with Ping-Pong Input Pipeline
            for by in T.serial(n_num // 2):
                # Buffer 0 (Even Blocks)
                col_off_0 = (by * 2) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_cast_0)

                T.tile.mul(a_ub_cast_0, a_ub_cast_0, a_ub_cast_0)
                # Accumulate directly into single buffer
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_0)

                # Buffer 1 (Odd Blocks)
                col_off_1 = (by * 2 + 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_1)
                    T.tile.cast(a_ub_cast_1, a_ub_1, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_cast_1)

                T.tile.mul(a_ub_cast_1, a_ub_cast_1, a_ub_cast_1)
                # Accumulate directly into single buffer
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_1)

            # Handle odd remainder
            if n_num % 2 != 0:
                col_off_rem = (n_num - 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_cast_0)
                T.tile.mul(a_ub_cast_0, a_ub_cast_0, a_ub_cast_0)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_cast_0)

            # Step 2: Reduction
            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)

            # Step 3: Compute inv_rms
            inv_n = T.cast(1.0 / N, acc_dtype)
            eps_val = T.cast(eps, acc_dtype)
            T.tile.mul(sum_sq_row, sum_sq_row, inv_n)
            T.tile.add(sum_sq_row, sum_sq_row, eps_val)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            # Step 4: Normalize and write back with Ping-Pong Pipeline
            for by in T.serial(n_num // 2):
                # Buffer 0
                col_off_0 = (by * 2) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N], a_ub_cast_0)

                T.tile.mul(a_ub_cast_0, a_ub_cast_0, inv_rms_tile)

                if need_cast:
                    T.tile.cast(a_ub_0, a_ub_cast_0, "CAST_RINT", tile_elements)
                    T.copy(a_ub_0, B[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N])
                else:
                    T.copy(a_ub_cast_0, B[row_start : row_start + ROWS, col_off_0 : col_off_0 + block_N])

                # Buffer 1
                col_off_1 = (by * 2 + 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_1)
                    T.tile.cast(a_ub_cast_1, a_ub_1, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N], a_ub_cast_1)

                T.tile.mul(a_ub_cast_1, a_ub_cast_1, inv_rms_tile)

                if need_cast:
                    T.tile.cast(a_ub_1, a_ub_cast_1, "CAST_RINT", tile_elements)
                    T.copy(a_ub_1, B[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N])
                else:
                    T.copy(a_ub_cast_1, B[row_start : row_start + ROWS, col_off_1 : col_off_1 + block_N])

            # Handle odd remainder
            if n_num % 2 != 0:
                col_off_rem = (n_num - 1) * block_N
                if need_cast:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_0)
                    T.tile.cast(a_ub_cast_0, a_ub_0, "CAST_NONE", tile_elements)
                else:
                    T.copy(A[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N], a_ub_cast_0)

                T.tile.mul(a_ub_cast_0, a_ub_cast_0, inv_rms_tile)

                if need_cast:
                    T.tile.cast(a_ub_0, a_ub_cast_0, "CAST_RINT", tile_elements)
                    T.copy(a_ub_0, B[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N])
                else:
                    T.copy(a_ub_cast_0, B[row_start : row_start + ROWS, col_off_rem : col_off_rem + block_N])

    return tilelang_rms_norm


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def _rms_norm_grad(M, N, block_M_in, block_N_in, eps=1e-5, dtype="float"):
    VEC_NUM = 2
    block_M, block_N = _get_optimized_tiling(M, N, block_M_in, block_N_in, VEC_NUM)
    m_num = M // block_M
    n_num = N // block_N
    ROWS = block_M // VEC_NUM
    tile_elements = ROWS * block_N

    need_cast = dtype not in ("float", "float32")
    acc_dtype = "float32" if need_cast else dtype

    @T.prim_func
    def tilelang_rms_norm_grad(dY: T.Tensor((M, N), dtype), X: T.Tensor((M, N), dtype), dX: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            # Buffers with ping pong dimension size 2
            x_ub = T.alloc_ub([2, ROWS, block_N], dtype)
            dy_ub = T.alloc_ub([2, ROWS, block_N], dtype)

            x_cast = T.alloc_ub([2, ROWS, block_N], acc_dtype)
            # dy_cast is reused as a product buffer in first pass (dy * x),
            # then reloaded from original dY before second pass.
            dy_cast = T.alloc_ub([2, ROWS, block_N], acc_dtype)

            # Adjusted to ROWS 1 to match broadcast dimension requirements
            rms_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            dot_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            scalar_ub = T.alloc_ub([ROWS, 1], acc_dtype)
            temp_vec_1 = T.alloc_ub([ROWS, 1], acc_dtype)
            temp_vec_2 = T.alloc_ub([ROWS, 1], acc_dtype)

            math_mat = T.alloc_ub([ROWS, block_N], acc_dtype)

            T.tile.fill(rms_ub, 0.0)
            T.tile.fill(dot_ub, 0.0)

            # First pass Stats accumulation using ping pong buffers
            for by in T.serial(n_num):
                pid = by % 2
                base_m = cid * block_M + vid * ROWS
                base_n = by * block_N

                if need_cast:
                    T.copy(X[base_m : base_m + ROWS, base_n : base_n + block_N], x_ub[pid, :, :])
                    T.copy(dY[base_m : base_m + ROWS, base_n : base_n + block_N], dy_ub[pid, :, :])
                    T.tile.cast(x_cast[pid, :, :], x_ub[pid, :, :], "CAST_NONE", tile_elements)
                    T.tile.cast(dy_cast[pid, :, :], dy_ub[pid, :, :], "CAST_NONE", tile_elements)
                else:
                    T.copy(X[base_m : base_m + ROWS, base_n : base_n + block_N], x_cast[pid, :, :])
                    T.copy(dY[base_m : base_m + ROWS, base_n : base_n + block_N], dy_cast[pid, :, :])

                T.tile.mul(dy_cast[pid, :, :], dy_cast[pid, :, :], x_cast[pid, :, :])
                T.reduce_sum(dy_cast[pid, :, :], temp_vec_1, dim=-1)
                T.tile.add(dot_ub, dot_ub, temp_vec_1)

                T.tile.mul(x_cast[pid, :, :], x_cast[pid, :, :], x_cast[pid, :, :])
                T.reduce_sum(x_cast[pid, :, :], temp_vec_2, dim=-1)
                T.tile.add(rms_ub, rms_ub, temp_vec_2)

            T.tile.fill(scalar_ub, float(N))
            T.tile.div(rms_ub, rms_ub, scalar_ub)
            T.tile.fill(temp_vec_1, float(eps))
            T.tile.add(rms_ub, rms_ub, temp_vec_1)
            T.tile.sqrt(rms_ub, rms_ub)

            T.tile.div(dot_ub, dot_ub, scalar_ub)
            T.tile.mul(scalar_ub, rms_ub, rms_ub)
            T.tile.div(dot_ub, dot_ub, scalar_ub)

            # Broadcast ROWS 1 to ROWS block N
            T.tile.fill(scalar_ub, 1.0)
            T.tile.div(rms_ub, scalar_ub, rms_ub)

            # Second pass Output computation using ping pong buffers
            for by in T.serial(n_num):
                pid = by % 2
                base_m = cid * block_M + vid * ROWS
                base_n = by * block_N

                if need_cast:
                    T.copy(X[base_m : base_m + ROWS, base_n : base_n + block_N], x_ub[pid, :, :])
                    T.copy(dY[base_m : base_m + ROWS, base_n : base_n + block_N], dy_ub[pid, :, :])
                    T.tile.cast(x_cast[pid, :, :], x_ub[pid, :, :], "CAST_NONE", tile_elements)
                    T.tile.cast(dy_cast[pid, :, :], dy_ub[pid, :, :], "CAST_NONE", tile_elements)
                else:
                    T.copy(X[base_m : base_m + ROWS, base_n : base_n + block_N], x_cast[pid, :, :])
                    T.copy(dY[base_m : base_m + ROWS, base_n : base_n + block_N], dy_cast[pid, :, :])

                # vectorized matrix operations
                T.tile.broadcast(math_mat, dot_ub)
                T.tile.mul(x_cast[pid, :, :], x_cast[pid, :, :], math_mat)
                T.tile.sub(dy_cast[pid, :, :], dy_cast[pid, :, :], x_cast[pid, :, :])

                T.tile.broadcast(math_mat, rms_ub)
                T.tile.mul(dy_cast[pid, :, :], dy_cast[pid, :, :], math_mat)

                if need_cast:
                    T.tile.cast(dy_ub[pid, :, :], dy_cast[pid, :, :], "CAST_RINT", tile_elements)
                    T.copy(dy_ub[pid, :, :], dX[base_m : base_m + ROWS, base_n : base_n + block_N])
                else:
                    T.copy(dy_cast[pid, :, :], dX[base_m : base_m + ROWS, base_n : base_n + block_N])

    return tilelang_rms_norm_grad


torch.manual_seed(0)

test_configs = [
    (256, 256, 64, 64, "float"),
    (1024, 1024, 128, 128, "float"),
    (1024, 51200, 128, 128, "float"),
    (16384, 7168, 64, 128, "bfloat16"),
    (16384, 1536, 64, 128, "bfloat16"),
    (16384, 512, 64, 128, "bfloat16"),
]


for M, N, block_M, block_N, dtype in test_configs:
    print(f"\nTesting rms_norm with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")
    func = _rms_norm(M, N, block_M, block_N, dtype=dtype)
    print("  Init successful!")

    if dtype == "bfloat16":
        a = torch.randn(M, N, device="npu", dtype=torch.bfloat16)
    else:
        a = torch.randn(M, N, device="npu", dtype=torch.float32)

    b = func(a)

    ref_b = torch.rms_norm(a.float(), normalized_shape=[N]).to(a.dtype)
    torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
    print("  Test passed!")

print("All rms_norm tests passed! Kernel Output Match!")

for M, N, block_M, block_N, dtype in test_configs:
    print(f"\nTesting rms_norm_grad with M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")

    func_grad = _rms_norm_grad(M, N, block_M, block_N, dtype=dtype)
    print("  Init successful!")

    if dtype == "bfloat16":
        x = torch.randn(M, N, device="npu", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(M, N, device="npu", dtype=torch.bfloat16)
    else:
        x = torch.randn(M, N, device="npu", dtype=torch.float32, requires_grad=True)
        dy = torch.randn(M, N, device="npu", dtype=torch.float32)

    y_ref = torch.rms_norm(x.float(), normalized_shape=[N]).to(x.dtype)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()

    dx = func_grad(dy, x.detach())

    torch.testing.assert_close(dx.cpu(), dx_ref.cpu(), rtol=1e-2, atol=1e-2)
    print("  Test passed!")

print("All rms_norm_grad tests passed! Kernel Output Match!")
