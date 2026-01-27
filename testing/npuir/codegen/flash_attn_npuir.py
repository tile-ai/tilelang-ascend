# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import tilelang
import tilelang.language as T

from utils import assert_compile_to_kernel_o_success



def flashattn(dtype, accum_dtype, seq_len, dim, block_m, block_n, block_k):
    scale = (1.0 / dim)**0.5
    shape = [seq_len, dim]
    shape2 = [seq_len, seq_len]

    block_m_half = (block_m + 1) // 2
    block_share = max(block_n, block_k)

    num_blocks = (seq_len - 1) // block_n + 1
    shape3 = [seq_len, dim * num_blocks]

    @T.prim_func
    def main(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            workspace_1: T.Tensor(shape2, dtype),
            workspace_2: T.Tensor(shape2, dtype),
            workspace_3: T.Tensor(shape3, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, subid):
            tail_size_m = T.min(block_m, seq_len - cid * block_m)
            l1_a = T.alloc_L1([block_m, block_share], dtype)
            l1_b = T.alloc_L1([block_m, block_k], dtype)

            l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

            logsum = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_max = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_max_prev = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_scale = T.alloc_ub([block_m_half, 1], accum_dtype)

            scores_sum = T.alloc_ub([block_m_half, 1], accum_dtype)
            scales = T.alloc_ub([T.ceildiv(seq_len, block_n)*block_m_half, 1], accum_dtype)


            cross_kernel_f16_dim = T.alloc_ub([block_m_half, dim], dtype)
            cross_kernel_f16_N = T.alloc_ub([block_m_half, block_n], dtype)
            cross_kernel_f32_dim = T.alloc_ub([block_m_half, dim], accum_dtype)
            cross_kernel_f32_N = T.alloc_ub([block_m_half, block_n], accum_dtype)
            acc_o = T.alloc_ub([block_m_half, dim], accum_dtype)

            acc_c_scale = scale

            with T.Scope("Cube"):
                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    tail_size_n = T.min(block_n, seq_len - i * block_n)
                    for k in T.serial(T.ceildiv(dim, block_k)):
                        tail_size_k = T.min(block_k, dim - k * block_k)
                        T.npuir_load_nd2nz(Q[cid * block_m, k * block_k], l1_a,
                                           [tail_size_m, tail_size_k])
                        T.npuir_load_nd2nz(K[i * block_n, k * block_k], l1_b,
                                           [tail_size_n, tail_size_k])
                        if k == 0:
                            T.npuir_dot(
                                l1_a,
                                l1_b,
                                l0_c,
                                initC=True,
                                b_transpose=True,
                                size=[tail_size_m, tail_size_k, tail_size_n])
                        else:
                            T.npuir_dot(
                                l1_a,
                                l1_b,
                                l0_c,
                                initC=False,
                                b_transpose=True,
                                size=[tail_size_m, tail_size_k, tail_size_n])

                    with T.rs("PIPE_FIX"):
                        T.npuir_store_fixpipe(
                            l0_c,
                            workspace_1[cid * block_m, i * block_n],
                            size=[tail_size_m, tail_size_n],
                            enable_nz2nd=True)
                        T.sync_block_set(i)

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    tail_size_n = T.min(block_n, seq_len - i * block_n)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.npuir_load_nd2nz(
                            workspace_2[cid * block_m, i * block_n],
                            l1_a,
                            size=[tail_size_m, tail_size_n])

                    for k in T.serial(T.ceildiv(dim, block_k)):
                        tail_size_k = T.min(block_k, dim - k * block_k)
                        by1 = i * dim
                        by2 = k * block_k
                        T.npuir_load_nd2nz(V[i * block_n, k * block_k], l1_b,
                                           [tail_size_n, tail_size_k])
                        T.npuir_dot(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=True,
                            size=[tail_size_m, tail_size_n, tail_size_k])
                        T.npuir_store_fixpipe(
                            l0_c,
                            workspace_3[cid * block_m, by1 + by2],
                            size=[tail_size_m, tail_size_k],
                            enable_nz2nd=True)

                    with T.rs("PIPE_FIX"):
                        T.sync_block_set(i)

            with T.Scope("Vector"):
                value_zero = 0
                value_min = -T.infinity("float32")
                T.npuir_brc(value_zero, logsum)
                T.npuir_brc(value_zero, acc_o)
                T.npuir_brc(value_zero, scores_scale)
                T.npuir_brc(value_zero, scales)
                T.npuir_brc(value_min, scores_max)

                real_m = (tail_size_m + 1) // 2
                bx = cid * block_m + subid * real_m
                real_m = real_m - (tail_size_m % 2) * subid

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    tail_size_n = T.min(block_n, seq_len - i * block_n)
                    T.copy(scores_max, scores_max_prev)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.copy(
                            workspace_1[bx, i * block_n],
                            cross_kernel_f16_N,
                            size=[real_m, tail_size_n])
                        T.npuir_cast(cross_kernel_f16_N, cross_kernel_f32_N, round_mode="rint")

                    T.npuir_mul(cross_kernel_f32_N, acc_c_scale, cross_kernel_f32_N)
                    T.npuir_reduce(cross_kernel_f32_N, scores_max, dims=[1], reduce_mode="max")
                    if i != 0:
                        T.npuir_max(scores_max_prev, scores_max, scores_max)
                        T.npuir_sub(scores_max_prev, scores_max, scores_scale)
                        T.npuir_exp(scores_scale, scores_scale)

                        T.copy(scores_scale, scales[i * block_m_half, 0], size=[block_m_half, 1])

                    T.npuir_sub(cross_kernel_f32_N, scores_max, cross_kernel_f32_N)
                    T.npuir_exp(cross_kernel_f32_N, cross_kernel_f32_N)
                    T.npuir_cast(cross_kernel_f32_N, cross_kernel_f16_N, round_mode="rint")

                    with T.rs("PIPE_MTE3"):
                        T.copy(
                            cross_kernel_f16_N,
                            workspace_2[bx, i * block_n],
                            size=[real_m, tail_size_n])
                        T.sync_block_set(i)

                    T.npuir_reduce(cross_kernel_f32_N, scores_sum, dims=[1], reduce_mode="sum")
                    T.npuir_mul(logsum, scores_scale, logsum)
                    T.npuir_add(logsum, scores_sum, logsum)


                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.copy(workspace_3[bx, i * dim], cross_kernel_f16_dim, size=[real_m, dim])
                    T.npuir_cast(cross_kernel_f16_dim, cross_kernel_f32_dim, round_mode="rint")
                    if i != 0:
                        T.copy(scales[i*block_m_half, 0], scores_scale, size=[block_m_half, 1])
                    T.npuir_mul(acc_o, scores_scale, acc_o)
                    T.npuir_add(acc_o, cross_kernel_f32_dim, acc_o)

                T.npuir_div(acc_o, logsum, acc_o)
                T.npuir_cast(acc_o, cross_kernel_f16_dim, round_mode="rint")
                T.copy(cross_kernel_f16_dim, Output[bx, 0], size=[real_m, dim])

    return main


def test_flash_attention_compile():
    """pytest 用例：flash attn 仅编译出 kernel.o，不依赖 torch_npu。"""
    func = flashattn(
        dtype="float16",
        accum_dtype="float32",
        seq_len=4096,
        dim=128,
        block_m=128,
        block_n=128,
        block_k=128,
    )
    o_bytes = assert_compile_to_kernel_o_success(func)
    assert o_bytes is not None and len(o_bytes) > 0


if __name__ == "__main__":
    test_flash_attention_compile()