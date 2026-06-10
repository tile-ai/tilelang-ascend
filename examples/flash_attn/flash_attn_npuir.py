# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T


seq_len = 512
dim = 128

torch.npu.set_device(0)


@tilelang.jit(out_idx=[-2, -1], target="npuir")
def online_flash_attention(
    block_M, block_N, block_K, dtype="float16", accum_dtype="float32"
):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    shape_lse = [seq_len]
    block_m = block_M
    block_n = block_N

    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        LSE: T.Tensor(shape_lse, accum_dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
            offset = cid * block_m
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            real_m = T.min(block_m, seq_len - cid * block_m)
            T.copy(Q[offset : offset + real_m, 0:dim], Q_shared)

            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            scores_cast = T.alloc_fragment([block_m, block_n], dtype)
            correction = T.alloc_fragment([block_m, 1], accum_dtype)
            local_max = T.alloc_fragment([block_m, 1], accum_dtype)
            local_sum = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )
            acc_m = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )
            acc_l = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )
            acc_o = T.alloc_fragment(
                [block_m, dim], accum_dtype
            )
            tmp = T.alloc_fragment([block_m, block_n], accum_dtype)
            tmp1 = T.alloc_fragment(
                [block_m, 1], accum_dtype
            )
            new_max = T.alloc_fragment([block_m, 1], accum_dtype)
            scales = T.alloc_fragment([block_m, block_n], accum_dtype)

            value_zero = 0
            scale = (1.0 / dim) ** 0.5
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):
                # cube
                real_n = T.min(block_n, seq_len - k * block_n)
                offset_n = k * block_n
                T.copy(K[offset_n : offset_n + real_n, 0:dim], K_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                # vec
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max, tmp1)
                T.vexp(tmp1, correction)
                # scores for current loop
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(
                    acc_l, correction, acc_l
                )
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                # copy new_max to acc_m
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)

                # cube
                T.copy(V[offset_n : offset_n + real_n, 0:dim], V_shared)
                T.gemm(scores_cast, V_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")

            T.copy(O_cast, Output[offset : offset + real_m, 0:dim])
            # lse
            lse_cast = T.alloc_shared([block_m, 1], accum_dtype)
            lse_reshape = T.alloc_shared([block_m], accum_dtype)
            T.vln(acc_l, acc_l)
            T.vadd(acc_l, acc_m, lse_cast)
            T.reshape(lse_cast, lse_reshape)
            T.copy(lse_reshape, LSE[offset : offset + real_m])

    return flash_attention


@tilelang.jit(out_idx=[-1], target="npuir")
def flash_attention_bwd_preprocess(
    block_M, block_N, dtype="float16", accum_dtype="float32"
):
    shape_o = [seq_len, dim]
    shape_do = [seq_len, dim]
    shape_delta = [seq_len]
    block_m = block_M
    block_n = block_N

    @T.prim_func
    def flash_attention_bwd_preprocess_kernel(
        Output: T.Tensor(shape_o, dtype),
        do: T.Tensor(shape_do, dtype),
        delta: T.Tensor(shape_delta, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), is_npu=True) as (cid, _):
            O_shared = T.alloc_shared([block_m, block_n], dtype)
            do_shared = T.alloc_shared([block_m, block_n], dtype)
            O_cast = T.alloc_shared([block_m, block_n], accum_dtype)
            do_cast = T.alloc_shared([block_m, block_n], accum_dtype)
            acc = T.alloc_fragment([block_m, block_n], accum_dtype)
            delta_shared = T.alloc_shared([block_m, 1], accum_dtype)

            offset_m = cid * block_m
            real_m = T.min(block_m, seq_len - cid * block_m)
            T.clear(acc)

            for k in T.Pipelined(T.ceildiv(dim, block_n), num_stages=2):
                offset_n = k * block_n
                real_n = T.min(block_n, dim - k * block_n)
                T.copy(
                    Output[
                        offset_m : offset_m + real_m,
                        offset_n : offset_n + real_n,
                    ],
                    O_shared,
                )
                T.vcast(O_shared, O_cast)
                T.copy(
                    do[
                        offset_m : offset_m + real_m,
                        offset_n : offset_n + real_n,
                    ],
                    do_shared,
                )
                T.vcast(do_shared, do_cast)
                for i, j in T.Parallel(block_m, block_n):
                    acc[i, j] += O_cast[i, j] * do_cast[i, j]
            T.reduce_sum(acc, delta_shared, dim=1)
            T.copy(delta_shared[:, 0], delta[offset_m : offset_m + real_m])

    return flash_attention_bwd_preprocess_kernel


@tilelang.jit(out_idx=[-1], target="npuir")
def flash_attention_bwd_dq(block_M, block_N, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_do = [seq_len, dim]
    shape_delta = [seq_len]
    shape_lse = [seq_len]
    block_m = block_M
    block_n = block_N

    @T.prim_func
    def flash_attention_bwd_dq_kernel(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        delta: T.Tensor(shape_delta, accum_dtype),
        LSE: T.Tensor(shape_lse, accum_dtype),
        do: T.Tensor(shape_do, dtype),
        dq: T.Tensor(shape_q, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), is_npu=True) as (cid, _):

            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_shared([block_m, block_n], accum_dtype)
            LSE_shared = T.alloc_shared([block_m], accum_dtype)
            acc_p = T.alloc_fragment([block_m, block_n], accum_dtype)
            dp_shared = T.alloc_fragment([block_m, block_n], accum_dtype)
            do_shared = T.alloc_shared([block_m, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            delta_shared = T.alloc_fragment([block_m], accum_dtype)
            ds_shared = T.alloc_shared([block_m, block_n], accum_dtype)
            dq_shared = T.alloc_fragment([block_m, dim], accum_dtype)
            acc_dq = T.alloc_fragment([block_m, dim], accum_dtype)
            K_shared_fp32 = T.alloc_shared([block_n, dim], accum_dtype)

            offset_m = cid * block_m
            real_m = T.min(block_m, seq_len - cid * block_m)
            T.copy(Q[offset_m : offset_m + real_m, 0:dim], Q_shared)
            T.copy(LSE[offset_m : offset_m + real_m], LSE_shared)
            T.copy(do[offset_m : offset_m + real_m, 0:dim], do_shared)
            T.copy(delta[offset_m : offset_m + real_m], delta_shared)

            scale = (1.0 / dim) ** 0.5
            value_zero = 0
            T.vbrc(value_zero, acc_dq)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):
                # 1.p
                offset_n = k * block_n
                real_n = T.min(block_n, seq_len - k * block_n)
                T.copy(K[offset_n : offset_n + real_n, 0:dim], K_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)
                T.vmul(scores, scale, scores)
                for i, j in T.Parallel(block_m, block_n):
                    acc_p[i, j] = scores[i, j] - LSE_shared[i]
                T.vexp(acc_p, acc_p)
                # 2.dp
                T.copy(V[offset_n : offset_n + real_n, 0:dim], V_shared)
                T.gemm(do_shared, V_shared, dp_shared, initC=True, b_transpose=True)
                # 3.ds
                for i, j in T.Parallel(block_m, block_n):
                    ds_shared[i, j] = acc_p[i, j] * (dp_shared[i, j] - delta_shared[i])
                # 4.dq
                T.vcast(K_shared, K_shared_fp32)
                T.gemm(ds_shared, K_shared_fp32, dq_shared, initC=True)
                T.vmul(dq_shared, scale, dq_shared)
                T.vadd(acc_dq, dq_shared, acc_dq)
            T.copy(acc_dq, dq[offset_m : offset_m + real_m, 0:dim])

    return flash_attention_bwd_dq_kernel


@tilelang.jit(out_idx=[-2, -1], target="npuir")
def flash_attention_bwd_dkdv(block_M, block_N, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_do = [seq_len, dim]
    shape_delta = [seq_len]
    shape_lse = [seq_len]
    block_m = block_M
    block_n = block_N

    @T.prim_func
    def flash_attention_bwd_dkdv(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        delta: T.Tensor(shape_delta, accum_dtype),
        LSE: T.Tensor(shape_lse, accum_dtype),
        do: T.Tensor(shape_do, dtype),
        dk: T.Tensor(shape_k, dtype),
        dv: T.Tensor(shape_v, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_n), is_npu=True) as (cid, _):
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_shared([block_m, block_n], accum_dtype)
            LSE_shared = T.alloc_shared([block_m], accum_dtype)
            acc_p = T.alloc_fragment([block_m, block_n], accum_dtype)
            dp_shared = T.alloc_fragment([block_m, block_n], accum_dtype)
            do_shared = T.alloc_shared([block_m, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            delta_shared = T.alloc_fragment([block_m], accum_dtype)
            ds_shared = T.alloc_shared([block_m, block_n], accum_dtype)
            dv_shared = T.alloc_fragment([block_n, dim], accum_dtype)
            acc_dv = T.alloc_fragment([block_n, dim], accum_dtype)
            dk_shared = T.alloc_fragment([block_n, dim], accum_dtype)
            acc_dk = T.alloc_fragment([block_n, dim], accum_dtype)
            Q_shared_fp32 = T.alloc_shared([block_m, dim], accum_dtype)
            do_shared_fp32 = T.alloc_shared([block_m, dim], accum_dtype)

            offset_n = cid * block_n
            real_row = T.min(block_n, seq_len - cid * block_n)
            T.copy(K[offset_n : offset_n + real_row, 0:dim], K_shared)
            T.copy(V[offset_n : offset_n + real_row, 0:dim], V_shared)
            scale = (1.0 / dim) ** 0.5
            value_zero = 0
            T.vbrc(value_zero, acc_dv)
            T.vbrc(value_zero, acc_dk)

            for k in T.Pipelined(T.ceildiv(seq_len, block_m), num_stages=2):
                # 1.p
                real_col = T.min(block_m, seq_len - k * block_m)
                offset_m = k * block_m
                T.copy(Q[offset_m : offset_m + real_col, 0:dim], Q_shared)
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)
                T.vmul(scores, scale, scores)
                T.copy(LSE[offset_m : offset_m + real_col], LSE_shared)
                for i, j in T.Parallel(block_m, block_n):
                    acc_p[i, j] = scores[i, j] - LSE_shared[i]
                T.vexp(acc_p, acc_p)
                # 2.dp
                T.copy(do[offset_m : offset_m + real_col, 0:dim], do_shared)
                T.gemm(do_shared, V_shared, dp_shared, initC=True, b_transpose=True)
                # 3.ds
                T.copy(delta[offset_m : offset_m + real_col], delta_shared)
                for i, j in T.Parallel(block_m, block_n):
                    ds_shared[i, j] = acc_p[i, j] * (dp_shared[i, j] - delta_shared[i])
                # 4.dk,dv
                T.vcast(do_shared, do_shared_fp32)
                T.gemm(acc_p, do_shared_fp32, dv_shared, initC=True, a_transpose=True)
                T.vadd(acc_dv, dv_shared, acc_dv)

                T.vcast(Q_shared, Q_shared_fp32)
                T.gemm(
                    ds_shared, Q_shared_fp32, dk_shared, initC=True, a_transpose=True
                )
                T.vmul(dk_shared, scale, dk_shared)
                T.vadd(acc_dk, dk_shared, acc_dk)

            T.copy(acc_dv, dv[offset_n : offset_n + real_row, 0:dim])
            T.copy(acc_dk, dk[offset_n : offset_n + real_row, 0:dim])

    return flash_attention_bwd_dkdv


def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.manual_seed(2026)
    kernel = online_flash_attention(64, 64, 32)

    q = torch.randn((seq_len, dim), dtype=torch.float16).npu().requires_grad_()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu().requires_grad_()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu().requires_grad_()

    LSE, output = kernel(q, k, v)

    scale = (1.0 / dim) ** 0.5
    ref_output = (
        torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(
            torch.float16
        )
        @ v
    )
    ref_lse = torch.logsumexp((q @ k.T).to(torch.float32) * scale, dim=-1)
    print("output:")
    print(output)
    print("ref_output:")
    print(ref_output)
    print("lse:", LSE)
    print("ref_lse:", ref_lse)
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_lse, LSE, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    print("bwd begin:")

    do = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    ref_output.backward(do, retain_graph=True)
    # torch
    dq_ref = q.grad.detach().clone()
    dk_ref = k.grad.detach().clone()
    dv_ref = v.grad.detach().clone()

    # tilelang
    # compile
    preprocess_kernel = flash_attention_bwd_preprocess(64, 64)
    bwd_dkdv_kernel = flash_attention_bwd_dkdv(64, 64)
    bwd_dq_kernel = flash_attention_bwd_dq(64, 64)
    # compute
    delta = preprocess_kernel(output, do)
    dk, dv = bwd_dkdv_kernel(q, k, v, delta, LSE, do)
    dq = bwd_dq_kernel(q, k, v, delta, LSE, do)

    print("dq_ref:", dq_ref)
    print("dq:", dq)
    torch.testing.assert_close(dq_ref, dq, rtol=1e-2, atol=1e-2)

    print("dk_ref:", dk_ref)
    print("dk:", dk)
    torch.testing.assert_close(dk_ref, dk, rtol=1e-2, atol=1e-2)

    print("dv_ref:", dv_ref)
    print("dv", dv)
    torch.testing.assert_close(dv_ref, dv, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
