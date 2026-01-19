# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 128

torch.npu.set_device(0)

@tilelang.jit(out_idx=[-1], target="npuir")
def online_flash_attention(block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    shape_work = [seq_len, seq_len]
    block_m = block_M
    block_n = block_N
    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
            offset = cid * block_m
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])

            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            scores_cast = T.alloc_fragment([block_m, block_n], "float16")
            correction = T.alloc_fragment([block_m,1], "float32")
            local_max = T.alloc_fragment([block_m,1], "float32")
            local_sum = T.alloc_fragment([block_m,1], "float32")
            acc_m = T.alloc_fragment([block_m, 1], "float32")
            acc_l = T.alloc_fragment([block_m, 1], "float32")
            acc_o = T.alloc_fragment([block_m, dim], "float32")
            tmp = T.alloc_fragment([block_m, block_n], "float32")
            tmp1 = T.alloc_fragment([block_m,1], "float32")
            new_max = T.alloc_fragment([block_m,1], "float32")
            scales = T.alloc_fragment([block_m, block_n], "float32")

            value_zero = 0
            scale = (1.0 / dim)**0.5
            value_min = -T.infinity("float32")
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):

                # cube
                T.copy(K[k * block_n, 0], K_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                # vec
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max ,tmp1)
                T.exp(tmp1, correction)
                #scores for current loop
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                #copy new_max to acc_m
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)

                # cube
                T.copy(V[k * block_n, 0], V_shared)
                T.gemm(scores_cast, V_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m, seq_len - cid * block_m)
            T.copy(O_cast, Output[cid * block_m : (cid+1) * block_m, :],size=[real_m, dim])

    return flash_attention

def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_MODE'] = 'Developer'
    kernel = online_flash_attention(64, 64, 32)

    q = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    output = torch.randn((seq_len, dim), dtype=torch.float16).npu()

    kernel(q, k, v, output)

    scale = (1.0 / dim)**0.5
    ref_output = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    print("All check passed.")

if __name__ == "__main__":
    main()