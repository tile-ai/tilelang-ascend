# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch

torch.npu.set_device(0)

import tilelang
import tilelang.language as T
from tilelang.jit import compiler_npu

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

parser.add_argument(
    "--dtype", type=str, default="float16", help="Data type for matrix operations (e.g., float16)")
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float32",
    help="Data type for accumulation and vector operations (higher precision for numerical stability)"
)
parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length of input tensors")
parser.add_argument(
    "--dim", type=int, default=128, help="Feature dimension size (hidden dimension)")
parser.add_argument(
    "--block_m",
    type=int,
    default=128,
    help="Block size for the sequence length dimension in tiling")
parser.add_argument(
    "--block_n",
    type=int,
    default=128,
    help="Block size for the key/value sequence length dimension in tiling")
parser.add_argument(
    "--block_k", type=int, default=128, help="Block size for the feature dimension in tiling")


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
            tail_size_m = cid * block_m
            tail_size_m = seq_len - tail_size_m
            tail_size_m = T.min(block_m, tail_size_m)

            l1_a = T.alloc_L1([block_m, block_share], dtype)
            l1_b = T.alloc_L1([block_m, block_k], dtype)

            l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

            logsum = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_max = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_max_prev = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_scale = T.alloc_ub([block_m_half, 1], accum_dtype)
            scores_sum = T.alloc_ub([block_m_half, 1], accum_dtype)

            cross_kernel_f16_dim = T.alloc_ub([block_m_half, dim], dtype)
            cross_kernel_f16_N = T.alloc_ub([block_m_half, block_n], dtype)
            cross_kernel_f32_dim = T.alloc_ub([block_m_half, dim], accum_dtype)
            cross_kernel_f32_N = T.alloc_ub([block_m_half, block_n], accum_dtype)
            acc_o = T.alloc_ub([block_m_half, dim], accum_dtype)

            acc_c_scale = scale

            with T.Scope("Cube"):
                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    tail_size_n = i * block_n
                    tail_size_n = seq_len - tail_size_n
                    tail_size_n = T.min(block_n, tail_size_n)
                    for k in T.serial(T.ceildiv(dim, block_k)):
                        tail_size_k = k * block_k
                        tail_size_k = dim - tail_size_k
                        tail_size_k = T.min(block_k, tail_size_k)
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
                        T.sync_block_set(0)

                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(0)
                        T.npuir_load_nd2nz(
                            workspace_2[cid * block_m, i * block_n],
                            l1_a,
                            size=[tail_size_m, tail_size_n])

                    for k in T.serial(T.ceildiv(dim, block_k)):
                        tail_size_k = k * block_k
                        tail_size_k = dim - tail_size_k
                        tail_size_k = T.min(block_k, tail_size_k)
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
                        T.sync_block_set(0)

            with T.Scope("Vector"):
                value_zero = 0
                value_min = -T.infinity("float32")
                T.npuir_brc(value_zero, logsum)
                T.npuir_brc(value_zero, acc_o)
                T.npuir_brc(value_zero, scores_scale)
                T.npuir_brc(value_min, scores_max)

                real_m = (tail_size_m + 1)
                real_m = real_m // 2

                bx = cid * block_m
                subblock_M = subid * real_m
                bx = bx + subblock_M

                tail_size_m_mod2 = tail_size_m % 2
                incomplete_block_m = tail_size_m_mod2 * subid
                real_m = real_m - incomplete_block_m

                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    tail_size_n = i * block_n
                    tail_size_n = seq_len - tail_size_n
                    tail_size_n = T.min(block_n, tail_size_n)
                    T.copy(scores_max, scores_max_prev)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(0)
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
                    T.npuir_sub(cross_kernel_f32_N, scores_max, cross_kernel_f32_N)
                    T.npuir_exp(cross_kernel_f32_N, cross_kernel_f32_N)

                    T.npuir_reduce(cross_kernel_f32_N, scores_sum, dims=[1], reduce_mode="sum")

                    T.npuir_mul(logsum, scores_scale, logsum)
                    T.npuir_add(logsum, scores_sum, logsum)
                    T.npuir_mul(acc_o, scores_scale, acc_o)
                    T.npuir_cast(cross_kernel_f32_N, cross_kernel_f16_N, round_mode="rint")

                    with T.rs("PIPE_MTE3"):
                        T.copy(
                            cross_kernel_f16_N,
                            workspace_2[bx, i * block_n],
                            size=[real_m, tail_size_n])
                        T.sync_block_set(0)

                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(0)
                        T.copy(workspace_3[bx, i * dim], cross_kernel_f16_dim, size=[real_m, dim])
                    T.npuir_cast(cross_kernel_f16_dim, cross_kernel_f32_dim, round_mode="rint")
                    T.npuir_add(acc_o, cross_kernel_f32_dim, acc_o)

                T.npuir_div(acc_o, logsum, acc_o)
                T.npuir_cast(acc_o, cross_kernel_f16_dim, round_mode="rint")
                T.copy(cross_kernel_f16_dim, Output[bx, 0], size=[real_m, dim])

    return main


def generate_tensor(shape, dtype, clear=False):
    """generate tensor"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=2000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def run_test(main_args):
    func = flashattn(
        main_args.dtype,
        main_args.accum_dtype,
        main_args.seq_len,
        main_args.dim,
        main_args.block_m,
        main_args.block_n,
        main_args.block_k,
    )

    compiled_kernel = tilelang.compile(func, target='npuir')

    num_blocks = (main_args.seq_len - 1) // main_args.block_n + 1
    shape = [main_args.seq_len, main_args.dim]
    shape2 = [main_args.seq_len, main_args.seq_len]
    shape3 = [main_args.seq_len, main_args.dim * num_blocks]

    torch.manual_seed(88888888)  # set the random seed for torch

    q = generate_tensor(shape, main_args.dtype).npu()
    k = generate_tensor(shape, main_args.dtype).npu()
    v = generate_tensor(shape, main_args.dtype).npu()
    o = generate_tensor(shape, main_args.dtype, clear=True).npu()
    w1 = generate_tensor(shape2, main_args.dtype, clear=True).npu()
    w2 = generate_tensor(shape2, main_args.dtype, clear=True).npu()
    w3 = generate_tensor(shape3, main_args.dtype, clear=True).npu()

    scale = (1.0 / main_args.dim)**0.5
    ref_output = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v

    compiled_kernel(q, k, v, o, w1, w2, w3)
    torch.set_printoptions(sci_mode=False)
    print("Actual Result:")
    print(o)
    print("Expected Result:")
    print(ref_output)
    torch.testing.assert_close(o, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    args = parser.parse_args()

    run_test(args)
