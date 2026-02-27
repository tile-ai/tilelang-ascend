# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import sys
import os
import argparse
import torch
import torch.nn as nn


import tilelang
import tilelang.language as T

torch.npu.set_device(0)
tilelang.cache.clear_cache()


parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--b", type=int, default=2, help="Matrix B dimension")
parser.add_argument("--m", type=int, default=2048, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=4096, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=64, help="Matrix K dimension")
parser.add_argument("--h", type=int, default=32, help="Matrix H dimension")
parser.add_argument("--bs", type=int, default=64, help="Matrix bs dimension")
parser.add_argument("--nums_kernel", type=int, default=24, help="number of enabled cores")
args = parser.parse_args()

B = args.b
M = args.m
N = args.n
K = args.k
H = args.h

BLOCK_SIZE_N = args.bs
nums_kernel = args.nums_kernel
def fp8_lighting_indexer(B,
                    H, M, N, K,
                    BLOCK_SIZE_N,nums_kernel):
    m_num = M
    n_num = N // BLOCK_SIZE_N
    num_logic_kernels = m_num * n_num
    dtype = "float16"
    accum_dtype = "float32"
    FFTS_FLAG_THRESHOLD = 15

    @T.prim_func
    def main(
            q_ptr: T.Tensor((B*M*H, K), dtype),
            k_ptr: T.Tensor((B*N*1, K), dtype),
            o_ptr: T.Tensor((B*M, N), dtype),
            q_s_ptr: T.Tensor((M * H, 1), dtype),
            k_s_ptr: T.Tensor((1, N), dtype),
            workspace: T.Tensor((B*M *H, N), dtype),
            mask: T.Tensor((B*M, N), dtype),
    ):

        with T.Kernel(nums_kernel, is_npu=True) as (kernel_id, subid):

            with T.Scope("Cube"):
                for task_id in T.serial(T.ceildiv(m_num*n_num, nums_kernel)):
                    cid = task_id * nums_kernel+ kernel_id
                    if cid < num_logic_kernels:
                        mi = cid // n_num
                        ni = cid % (n_num)
                        offset_n = ni * BLOCK_SIZE_N

                        for bi in T.serial(B):
                            q = T.alloc_L1((H, K), dtype)
                            k = T.alloc_L1((BLOCK_SIZE_N, K), dtype)
                            l0_c = T.alloc_L0C((H, BLOCK_SIZE_N), accum_dtype)

                            offset = (bi * M + mi) * H
                            T.npuir_load_nd2nz(q_ptr[offset, 0], q, [H, K])

                            offset = bi * N+ offset_n
                            T.npuir_load_nd2nz(k_ptr[offset, 0], k, [BLOCK_SIZE_N, K])
                            T.npuir_dot(q, k, l0_c, initC=True, b_transpose=True,size=[H, K, BLOCK_SIZE_N])

                            with T.rs("PIPE_FIX"):
                                offset = (bi * M + mi) * H
                                T.npuir_store_fixpipe(l0_c, workspace[offset, offset_n],size=[H, BLOCK_SIZE_N], enable_nz2nd=True, pre_relu_mode="relu") 
                                T.sync_block_set(0)

                            flag = (task_id * B + bi) % FFTS_FLAG_THRESHOLD
                            temp = FFTS_FLAG_THRESHOLD - 1
                            if flag == temp:
                                with T.rs("PIPE_FIX"):
                                    T.sync_block_wait(0)


            with T.Scope("Vector"):
                for task_id in T.serial(T.ceildiv(m_num * n_num, nums_kernel)):
                    cid = task_id * nums_kernel + kernel_id
                    if cid < num_logic_kernels:
                        mi = cid // n_num
                        ni = cid % (n_num)
                        offset_n = ni * BLOCK_SIZE_N

                        for bi in T.serial(B):

                            q_s = T.alloc_ub((H, 1), dtype)
                            q_s_32 = T.alloc_ub((H, 1), accum_dtype)
                            k_s = T.alloc_ub((1, BLOCK_SIZE_N), dtype)
                            k_s_32 = T.alloc_ub((1, BLOCK_SIZE_N), accum_dtype)
                            msk = T.alloc_ub((1, BLOCK_SIZE_N), dtype)
                            msk_32 = T.alloc_ub((1, BLOCK_SIZE_N), accum_dtype)

                            logits = T.alloc_ub((1, BLOCK_SIZE_N), dtype)
                            logits_32 = T.alloc_ub((1, BLOCK_SIZE_N), accum_dtype)
                            o_temp = T.alloc_ub((1, BLOCK_SIZE_N), accum_dtype)
                            scores_sum = T.alloc_ub((1, BLOCK_SIZE_N), accum_dtype)  
                            ub_relu = T.alloc_ub((H, BLOCK_SIZE_N), dtype)
                            ub_relu_32 = T.alloc_ub((H, BLOCK_SIZE_N), accum_dtype)
                            temp = T.alloc_ub((H, BLOCK_SIZE_N), accum_dtype)

                            with T.rs("PIPE_MTE2"):

                                T.sync_block_wait(0)
                                offset = (bi * M + mi) * H
                                T.copy(workspace[offset : offset + H, offset_n : offset_n + BLOCK_SIZE_N], ub_relu)
                                T.npuir_cast(ub_relu, ub_relu_32, round_mode="rint")

                                T.copy(q_s_ptr[mi*H : (mi+1)*H, 0 : 1], q_s)
                                T.npuir_cast(q_s, q_s_32, round_mode="rint")

                                T.copy(k_s_ptr[0 : 1, offset_n : offset_n + BLOCK_SIZE_N], k_s)
                                T.npuir_cast(k_s, k_s_32, round_mode="rint")

                                T.npuir_mul(ub_relu_32, q_s_32, temp)

                                T.npuir_reduce(temp, scores_sum, dims=[0], reduce_mode="sum")
                                T.npuir_mul(scores_sum, k_s_32, o_temp)

                            flag = (task_id * B + bi) % FFTS_FLAG_THRESHOLD
                            temp = FFTS_FLAG_THRESHOLD - 1
                            if flag == temp:
                                with T.rs("PIPE_MTE2"):
                                    T.sync_block_set(0)

                            offset = bi * M + mi
                            T.copy(mask[offset : offset + 1, offset_n : offset_n + BLOCK_SIZE_N], msk)

                            T.npuir_cast(msk, msk_32, round_mode="rint")
                            T.npuir_add(msk_32, o_temp, logits_32)

                            T.npuir_cast(logits_32, logits, round_mode="rint")
                            T.copy(logits, o_ptr[offset : offset + 1, offset_n : offset_n + BLOCK_SIZE_N])


    return main


def ref_fp8_index(q_ptr: torch.Tensor, q_s_ptr: torch.Tensor, k_ptr: torch.Tensor, k_s_ptr: torch.Tensor, mask: torch.Tensor):

    # q_ptr: (B, M, H, K)
    # q_s_ptr: (M*H)
    # k_ptr: (B, N, K)
    # k_s_ptr: (N)
    q_s = q_s_ptr.view(M, H)
    
    k_ptr = k_ptr.transpose(1, 2)

    q_reshaped = q_ptr.view(B, M * H, K)
    q_reshaped.to(dtype=torch.float16)
    temp = torch.matmul(q_reshaped, k_ptr)  # (B, M * H, N)
    temp = temp.view(B, M, H, N)
    temp_relu = torch.relu(temp)
    temp_relu = temp_relu.to(dtype=torch.float32)

    temp_temp = temp_relu * q_s[None, :, :, None] # q_s维度扩展为(1, M, H, 1)

    o_temp = torch.sum(temp_temp, dim=2) # (B, M, N)
    o_temp_temp = o_temp * k_s_ptr[None, None, :] # k_s_ptr 维度扩展为 (1, 1, N)

    o_temp_temp += mask
    o_temp_temp = o_temp_temp.to(dtype=torch.float16)
    return o_temp_temp



if __name__ == '__main__':
    func = fp8_lighting_indexer(args.b, args.h, args.m, args.n, args.k, args.bs, args.nums_kernel)
    compiled_kernel = tilelang.compile(func, target='npuir')
    torch.manual_seed(0)

    B = 2
    M = 2048
    H = 32
    K = 64
    N = 4096 
    BLOCK_SIZE_N = 64
    q = torch.randn((B*M * H, K), dtype=torch.float16)
    k = torch.randn((B*N, K), dtype=torch.float16)
    o = torch.zeros((B*M, N), dtype=torch.float16).npu()
    q_s = torch.randn((M * H, 1), dtype=torch.float16)
    k_s = torch.randn((1, N), dtype=torch.float16)
    workspace = torch.zeros((B* M *H, N), dtype=torch.float16).npu()

    mask = torch.full((M, N), float("-inf")).triu_(1).to(dtype=torch.float16) if M > 1 else None
    mask = torch.stack([mask for _ in range(B)], dim=0) #(B, M, N)

    compiled_kernel(q.npu(), k.npu(), o, q_s.npu(), k_s.npu(), workspace, mask.reshape(B * M, N).npu())
    print(o.cpu().reshape(B, M, N))

    o_torch = ref_fp8_index(q.reshape(B, M, H, K), q_s.reshape(M * H), k.reshape(B, N, K), k_s.reshape(N), mask)
    print("torch:", o_torch.cpu())

    torch.testing.assert_close(o.cpu().reshape(B, M, N), o_torch, rtol=1e-2, atol=1e-2)