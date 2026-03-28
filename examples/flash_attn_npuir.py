# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import argparse
import torch

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

parser.add_argument(
    "--dtype",
    type=str,
    default="float16",
    help="Data type for matrix operations (e.g., float16)",
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float32",
    help="Data type for accumulation and vector operations (higher precision for numerical stability)",
)
parser.add_argument(
    "--seq_len", type=int, default=4096, help="Sequence length of input tensors"
)
parser.add_argument(
    "--dim", type=int, default=128, help="Feature dimension size (hidden dimension)"
)
parser.add_argument(
    "--block_m",
    type=int,
    default=96,
    help="Block size for the sequence length dimension in tiling",
)
parser.add_argument(
    "--block_n",
    type=int,
    default=256,
    help="Block size for the key/value sequence length dimension in tiling",
)
parser.add_argument(
    "--block_k",
    type=int,
    default=128,
    help="Block size for the feature dimension in tiling",
)


@tilelang.jit(target="npuir", out_idx=[3])
def flash_attn_kernel(dtype, accum_dtype, seq_len, dim, block_m, block_n, block_k):
    scale = (1.0 / dim) ** 0.5
    shape = [seq_len, dim]
    shape2 = [seq_len, seq_len]

    block_m_half = (block_m + 1) // 2
    block_share = max(block_n, block_k)

    num_blocks = (seq_len - 1) // block_n + 1
    shape3 = [seq_len, dim * num_blocks]

    @T.prim_func
    def FlashAttnExp(
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
            acc_c_scale = scale
            offset_m = cid * block_m

            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, block_share], dtype)
                l1_b = T.alloc_L1([block_n, block_k], dtype)

                l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    offset_n = i * block_n
                    tail_size_n = T.min(block_n, seq_len - offset_n)
                    for k in T.serial(T.ceildiv(dim, block_k)):
                        offset_k = k * block_k
                        tail_size_k = T.min(block_k, dim - offset_k)

                        T.copy(
                            Q[
                                offset_m : offset_m + tail_size_m,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_a[:tail_size_m, :tail_size_k],
                        )
                        T.copy(
                            K[
                                offset_n : offset_n + tail_size_n,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_b[:tail_size_n, :tail_size_k],
                        )
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=(k == 0),
                            b_transpose=True,
                            size=[tail_size_m, tail_size_k, tail_size_n],
                        )

                    with T.rs("PIPE_FIX"):
                        T.copy(
                            l0_c[:tail_size_m, :tail_size_n],
                            workspace_1[
                                offset_m : offset_m + tail_size_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                        )
                        T.sync_block_set(i)

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    offset_n = i * block_n
                    tail_size_n = T.min(block_n, seq_len - offset_n)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.copy(
                            workspace_2[
                                offset_m : offset_m + tail_size_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                            l1_a[:tail_size_m, :tail_size_n],
                        )

                    for k in T.serial(T.ceildiv(dim, block_k)):
                        offset_k = k * block_k
                        tail_size_k = T.min(block_k, dim - offset_k)
                        offset_w3 = i * dim + offset_k
                        T.copy(
                            V[
                                offset_n : offset_n + tail_size_n,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_b[:tail_size_n, :tail_size_k],
                        )
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=True,
                            size=[tail_size_m, tail_size_n, tail_size_k],
                        )
                        T.copy(
                            l0_c[:tail_size_m, :tail_size_k],
                            workspace_3[
                                offset_m : offset_m + tail_size_m,
                                offset_w3 : offset_w3 + tail_size_k,
                            ],
                        )

                    with T.rs("PIPE_FIX"):
                        T.sync_block_set(i)

            with T.Scope("Vector"):
                logsum = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max_prev = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_scale = T.alloc_ub([block_m_half, 1], accum_dtype)

                scores_sum = T.alloc_ub([block_m_half, 1], accum_dtype)
                scales = T.alloc_ub(
                    [T.ceildiv(seq_len, block_n) * block_m_half, 1], accum_dtype
                )

                cross_kernel_f16_dim = T.alloc_ub([block_m_half, dim], dtype)
                cross_kernel_f16_N = T.alloc_ub([block_m_half, block_n], dtype)
                cross_kernel_f32_dim = T.alloc_ub([block_m_half, dim], accum_dtype)
                cross_kernel_f32_N = T.alloc_ub([block_m_half, block_n], accum_dtype)
                acc_o = T.alloc_ub([block_m_half, dim], accum_dtype)

                value_zero = 0
                value_min = -T.infinity("float32")
                T.vbrc(value_zero, logsum)
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, scores_scale)
                T.vbrc(value_zero, scales)
                T.vbrc(value_min, scores_max)

                real_m = (tail_size_m + 1) // 2
                bx = cid * block_m + subid * real_m
                real_m = real_m - (tail_size_m % 2) * subid

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    offset_n = i * block_n
                    tail_size_n = T.min(block_n, seq_len - offset_n)
                    T.copy(scores_max, scores_max_prev)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.copy(
                            workspace_1[
                                bx : bx + real_m, offset_n : offset_n + tail_size_n
                            ],
                            cross_kernel_f16_N[0:real_m, 0:tail_size_n],
                        )
                        T.vcast(
                            cross_kernel_f16_N, cross_kernel_f32_N, round_mode="rint"
                        )

                    T.vmul(cross_kernel_f32_N, acc_c_scale, cross_kernel_f32_N)
                    T.reduce(
                        cross_kernel_f32_N, scores_max, dims=[1], reduce_mode="max"
                    )
                    if i != 0:
                        T.vmax(scores_max_prev, scores_max, scores_max)
                        T.vsub(scores_max_prev, scores_max, scores_scale)
                        T.vexp(scores_scale, scores_scale)

                        T.copy(
                            scores_scale,
                            scales[
                                i * block_m_half : i * block_m_half + block_m_half, :
                            ],
                        )

                    T.vsub(cross_kernel_f32_N, scores_max, cross_kernel_f32_N)
                    T.vexp(cross_kernel_f32_N, cross_kernel_f32_N)
                    T.vcast(cross_kernel_f32_N, cross_kernel_f16_N, round_mode="rint")

                    with T.rs("PIPE_MTE3"):
                        T.copy(
                            cross_kernel_f16_N[0:real_m, 0:tail_size_n],
                            workspace_2[
                                bx : bx + real_m, offset_n : offset_n + tail_size_n
                            ],
                        )
                        T.sync_block_set(i)

                    T.reduce(
                        cross_kernel_f32_N, scores_sum, dims=[1], reduce_mode="sum"
                    )
                    T.vmul(logsum, scores_scale, logsum)
                    T.vadd(logsum, scores_sum, logsum)

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(i)
                        T.copy(
                            workspace_3[bx : bx + real_m, i * dim : i * dim + dim],
                            cross_kernel_f16_dim[0:real_m, 0:dim],
                        )
                    T.vcast(
                        cross_kernel_f16_dim, cross_kernel_f32_dim, round_mode="rint"
                    )
                    if i != 0:
                        T.copy(
                            scales[
                                i * block_m_half : i * block_m_half + block_m_half, 0:1
                            ],
                            scores_scale,
                        )
                    T.vmul(acc_o, scores_scale, acc_o)
                    T.vadd(acc_o, cross_kernel_f32_dim, acc_o)

                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, cross_kernel_f16_dim, round_mode="rint")
                T.copy(
                    cross_kernel_f16_dim[0:real_m, 0:dim],
                    Output[bx : bx + real_m, 0:dim],
                )

    return FlashAttnExp


def run_test(main_args):
    kernel = flash_attn_kernel(
        main_args.dtype,
        main_args.accum_dtype,
        main_args.seq_len,
        main_args.dim,
        main_args.block_m,
        main_args.block_n,
        main_args.block_k,
    )

    num_blocks = (main_args.seq_len - 1) // main_args.block_n + 1
    shape = [main_args.seq_len, main_args.dim]
    shape2 = [main_args.seq_len, main_args.seq_len]
    shape3 = [main_args.seq_len, main_args.dim * num_blocks]

    torch.manual_seed(88888888)  # set the random seed for torch

    # Get torch dtype from string
    dtype = getattr(torch, main_args.dtype)

    # Generate random tensors for Q, K, V
    q = torch.randn(shape, dtype=dtype).npu()
    k = torch.randn(shape, dtype=dtype).npu()
    v = torch.randn(shape, dtype=dtype).npu()

    # Generate empty workspace tensors
    w1 = torch.empty(shape2, dtype=dtype).npu()
    w2 = torch.empty(shape2, dtype=dtype).npu()
    w3 = torch.empty(shape3, dtype=dtype).npu()

    # Reference computation using PyTorch
    scale = (1.0 / main_args.dim) ** 0.5
    ref_output = (
        torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(
            dtype
        )
        @ v
    )

    # Run the compiled kernel
    o = kernel(q, k, v, w1, w2, w3)

    torch.testing.assert_close(o, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()
    args = parser.parse_args()
    run_test(args)
