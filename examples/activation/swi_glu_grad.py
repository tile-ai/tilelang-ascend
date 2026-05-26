import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()
torch.set_default_device("npu")

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    # tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def swi_glu_backward_db(block_M, block_N, split_dim, dtype="float"):
    M = T.symbolic("M")
    N = T.symbolic("N")

    block_M = min(block_M, 64)
    block_N = min(block_N, 64)

    need_cast = dtype not in ("float", "float32")
    ACC_DTYPE = "float32"

    m_div = 1
    n_div = 2
    if split_dim == 0 or split_dim == -2:
        m_div = 2
        n_div = 1

    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)

    VEC_NUM = 2
    num_blocks = m_num * n_num

    rows_per_vec = block_M // VEC_NUM
    elem_num = rows_per_vec * block_N

    NUM_CORES = 48
    stages = 2
    num_iters = T.ceildiv(num_blocks, NUM_CORES)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        dY: T.Tensor((M // m_div, N // n_div), dtype),
        dA: T.Tensor((M, N), dtype),
    ):
        m_offset = 0
        n_offset = N // 2
        if split_dim == 0 or split_dim == -2:
            m_offset = M // 2
            n_offset = 0

        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            x1_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            x2_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            dy_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            tmp_bf16_1 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_2 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_3 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_4 = T.alloc_ub((rows_per_vec, block_N), dtype)
            tmp_bf16_5 = T.alloc_ub((rows_per_vec, block_N), dtype)

            sigmoid_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            dx1_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            dx2_ub = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            temp1 = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)
            temp2 = T.alloc_ub((stages, rows_per_vec, block_N), ACC_DTYPE)

            zero_ub = T.alloc_ub((rows_per_vec, block_N), ACC_DTYPE)
            one_ub = T.alloc_ub((rows_per_vec, block_N), ACC_DTYPE)

            T.tile.fill(zero_ub, 0.0)
            T.tile.fill(one_ub, 1.0)
            for s in T.serial(stages):
                T.set_flag("v", "mte2", s)

            for i in T.serial(num_iters):
                cur = i % stages

                block_id = cid + i * NUM_CORES

                if block_id < num_blocks:
                    bx = block_id // n_num
                    by = block_id % n_num

                    row_start = bx * block_M + vid * rows_per_vec
                    col_start = by * block_N

                    T.wait_flag("v", "mte2", cur)

                    ## load
                    T.set_flag("v", "mte2", 3)
                    T.wait_flag("v", "mte2", 3)
                    if need_cast:
                        T.copy(A[row_start, col_start], tmp_bf16_1)
                        T.copy(A[row_start + m_offset, col_start + n_offset], tmp_bf16_2)
                        T.copy(dY[row_start, col_start], tmp_bf16_3)
                        T.set_flag("mte2", "v", cur)
                        T.wait_flag("mte2", "v", cur)
                        T.tile.cast(x1_ub[cur, :, :], tmp_bf16_1, "CAST_NONE", elem_num)
                        T.tile.cast(x2_ub[cur, :, :], tmp_bf16_2, "CAST_NONE", elem_num)
                        T.tile.cast(dy_ub[cur, :, :], tmp_bf16_3, "CAST_NONE", elem_num)
                    else:
                        T.copy(A[row_start, col_start], x1_ub[cur, :, :])
                        T.copy(A[row_start + m_offset, col_start + n_offset], x2_ub[cur, :, :])
                        T.copy(dY[row_start, col_start], dy_ub[cur, :, :])

                    T.pipe_barrier("mte2")

                    T.set_flag("mte2", "v", cur)

                    T.wait_flag("mte2", "v", cur)
                    ## sigmoid(x1)
                    T.tile.sub(temp1[cur, :, :], zero_ub, x1_ub[cur, :, :])
                    T.tile.exp(temp1[cur, :, :], temp1[cur, :, :])
                    T.tile.add(temp1[cur, :, :], temp1[cur, :, :], 1.0)
                    T.tile.div(sigmoid_ub[cur, :, :], one_ub, temp1[cur, :, :])

                    ## silu(x1) = x1 * sigmoid(x1)
                    T.tile.mul(temp1[cur, :, :], x1_ub[cur, :, :], sigmoid_ub[cur, :, :])

                    ## dx2 = dy * silu(x1)
                    T.tile.mul(dx2_ub[cur, :, :], dy_ub[cur, :, :], temp1[cur, :, :])

                    # temp1 = (1 - sigmoid(x1))
                    T.tile.sub(temp2[cur, :, :], one_ub, sigmoid_ub[cur, :, :])

                    # temp2 = x1 * sigmoid(x1) * (1-sigmoid(x1))
                    T.tile.mul(temp2[cur, :, :], temp2[cur, :, :], sigmoid_ub[cur, :, :])
                    T.tile.mul(temp2[cur, :, :], x1_ub[cur, :, :], temp2[cur, :, :])

                    # temp2 = sigmoid(x1) + x1*sigmoid(x1)*(1-sigmoid(x1))
                    T.tile.add(temp2[cur, :, :], sigmoid_ub[cur, :, :], temp2[cur, :, :])

                    ## dx1 = dy * x2 * derivative
                    T.tile.mul(dx1_ub[cur, :, :], dy_ub[cur, :, :], x2_ub[cur, :, :])
                    T.set_flag("mte3", "v", cur)
                    T.wait_flag("mte3", "v", cur)
                    T.tile.mul(dx1_ub[cur, :, :], dx1_ub[cur, :, :], temp2[cur, :, :])

                    T.set_flag("v", "mte3", cur)
                    T.set_flag("v", "mte2", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.pipe_barrier("mte3")

                    ## store
                    if row_start + m_offset < M and col_start + n_offset < N:
                        if need_cast:
                            T.set_flag("mte3", "v", cur)
                            T.wait_flag("mte3", "v", cur)
                            T.tile.cast(tmp_bf16_4, dx2_ub[cur, :, :], "CAST_RINT", elem_num)
                            T.tile.cast(tmp_bf16_5, dx1_ub[cur, :, :], "CAST_RINT", elem_num)
                            T.copy(tmp_bf16_4, dA[row_start + m_offset, col_start + n_offset])
                            T.copy(tmp_bf16_5, dA[row_start, col_start])
                        else:
                            T.copy(dx2_ub[cur, :, :], dA[row_start + m_offset, col_start + n_offset])
                            T.copy(dx1_ub[cur, :, :], dA[row_start, col_start])

            T.wait_flag("v", "mte2", 0)
            T.wait_flag("v", "mte2", 1)

    return main


test_configs = [
    (1024, 18432, 64, 128, 1, "bfloat16"),
    (1024, 18432, 64, 128, 1, "float"),
    (1024, 1024, 64, 128, 1, "bfloat16"),
    (1024, 1024, 64, 128, 1, "float"),
    (256, 256, 64, 128, 1, "bfloat16"),
    (256, 256, 64, 128, 1, "float"),
    (128, 128, 16, 16, 1, "float"),
]
for i in range(1):
    for M, N, block_M, block_N, split_dim, dtype in test_configs:
        print(f"\nRun {i + 1}/10")
        print(f"Testing SwiGLU backward with M={M}, N={N}, block_M={block_M}, block_N={block_N}, split_dim={split_dim}, dtype={dtype}")
        func = swi_glu_backward_db(block_M, block_N, split_dim, dtype=dtype)
        print("Init successful!")
        # print(func.get_kernel_source())
        if dtype == "bfloat16":
            a = torch.randn(M, N, dtype=torch.bfloat16).npu()
            dy = torch.randn(M, N, dtype=torch.bfloat16).npu()
        else:
            a = torch.randn(M, N, dtype=torch.float32).npu()
            dy = torch.randn(M, N, dtype=torch.float32).npu()

        tile_grad = func(a, dy)

        # ascendC ref
        ascend_grad = torch.ops.npu.npu_swiglu_backward(dy, a)

        torch.testing.assert_close(ascend_grad.cpu(), tile_grad.cpu(), rtol=1e-2, atol=1e-2)
        print("Backward Test Passed")

print("Kernel Output Match!")
