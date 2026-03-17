import torch
import math
import tilelang as tl
import tilelang.language as T
import os

torch.npu.set_device(6)

dtype = "float16"
accum_dtype = "float32"
indices_dtype = "int32"


@tl.jit(target="npuir")
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_I=64,
    num_stages=2,
):
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]

    H = head_kv
    padded_H = max(tl.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tl.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H * batch * kv_group, is_npu=True) as (
            cid,
            _,
        ):
            tmp = cid
            bx = tmp % (seq_len * REPLICATE_H)
            tmp = tmp // (seq_len * REPLICATE_H)
            by = tmp % batch
            bz = tmp // batch

            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)

            acc_o = T.alloc_shared([H_per_block, D], accum_dtype)
            acc_s = T.alloc_shared([H_per_block, BI], accum_dtype)
            acc_s_tmp = T.alloc_shared([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_shared([H_per_block, 1], accum_dtype)
            sumexp_tmp = T.alloc_shared([H_per_block, 1], accum_dtype)
            sumexp_i = T.alloc_shared([H_per_block, 1], accum_dtype)
            alpha = T.alloc_shared([H_per_block, 1], accum_dtype)
            alpha_tmp = T.alloc_shared([H_per_block, 1], accum_dtype)
            m_i = T.alloc_shared([H_per_block, 1], accum_dtype)
            m_i_prev = T.alloc_shared([H_per_block, 1], accum_dtype)

            T.clear(acc_o)
            T.clear(sumexp)
            val = -(2**30)
            T.fill(m_i, val)  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                    ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    if Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i:
                        acc_s[h_i, bi_i] = 0
                    else:
                        acc_s[h_i, bi_i] = -T.infinity(accum_dtype)
                    # acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, KV_shared, acc_s, b_transpose=True)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, b_transpose=True)
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                T.vmax(m_i, m_i_prev, m_i)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i, 0] = m_i_prev[h_i, 0] - m_i[h_i, 0]
                T.vexp2(alpha, alpha, alpha_tmp)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = (
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i, 0] * sm_scale
                    )
                T.vexp2(acc_s, acc_s, acc_s_tmp)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i, 0] = sumexp[h_i, 0] * alpha[h_i, 0] + sumexp_i[h_i, 0]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i, 0]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i, 0]
            T.vlog2(sumexp, sumexp, sumexp_tmp)
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i, 0] = sumexp[h_i, 0] + m_i[h_i, 0] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0], size=[1, H_per_block])

    return main


def sparse_mla_fwd_interface(
    q, kv, indices, out, lse, sm_scale=None, d_v=512, block_I=64, num_stages=2
):
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    dim = d_v

    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices.shape
    kernel = sparse_mla_fwd(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
    )
    kernel(q, kv, indices, out, lse)
    return out, lse


def sparse_mla_fwd_torch(
    Q,
    KV,
    Indices,
    dim,
    tail_dim,
    kv_group=1,
    sm_scale=None,
):
    B, Sq, H, total_dim = Q.shape
    _, Skv, G, _ = KV.shape
    _, _, _, topk = Indices.shape

    assert total_dim == dim + tail_dim
    assert H % kv_group == 0
    assert kv_group == G

    head_kv = H // kv_group

    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    scale_e = sm_scale * math.log(2.0)

    q_dtype = Q.dtype
    acc_dtype = torch.float32

    Qf = Q.to(acc_dtype)
    KVf = KV.to(acc_dtype)
    Indices = Indices.long()

    O = torch.zeros((B, Sq, H, dim), device=Q.device, dtype=q_dtype)
    Lse = torch.zeros((B, Sq, H), device=Q.device, dtype=acc_dtype)

    for b in range(B):
        for s in range(Sq):
            for g in range(kv_group):
                h0 = g * head_kv
                h1 = (g + 1) * head_kv

                q = Qf[b, s, h0:h1, :]
                idx = Indices[b, s, g, :]
                k = KVf[b, idx, g, :]
                v = KVf[b, idx, g, :dim]

                logits = q @ k.transpose(0, 1)

                valid = (idx <= s) & (idx >= 0) & (idx < Skv)
                logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))

                any_valid = valid.any()
                if not any_valid:
                    O[b, s, h0:h1, :] = 0
                    Lse[b, s, h0:h1] = float("-inf")
                    continue

                logits_e = logits * scale_e

                probs = torch.softmax(logits_e, dim=-1)  # [Hg, topk]

                probs_cast = probs.to(q_dtype)
                v_cast = v.to(q_dtype)

                out = torch.matmul(probs_cast.to(acc_dtype), v_cast.to(acc_dtype))

                lse = torch.logsumexp(logits_e, dim=-1) / math.log(2.0)

                O[b, s, h0:h1, :] = out.to(q_dtype)
                Lse[b, s, h0:h1] = lse

    return O, Lse


def generate_tensor(shape, dtype, clear=False):
    """Generate tensor with specified shape and data type"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(
            low=0, high=10000, size=shape, dtype=eval("torch." + dtype)
        )
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def run_test():
    compiled_kernel = sparse_mla_fwd(
        1,  # batch
        32,  # seq_len
        32,  # seq_len_kv
        16,  # heads
        128,  # dim
        64,  # tail_dim
        16,  # topk
        1,  # kv_group
        None,  # sm_scale
        16,  # block_I
        2,  # num_stages
    )
    print("compile finished!")

    Q_shape = [1, 32, 16, 192]
    KV_shape = [1, 32, 1, 192]
    Indices_shape = [1, 32, 1, 16]
    Output_shape = [1, 32, 16, 128]
    Lse_shape = [1, 32, 16]

    Q = generate_tensor(Q_shape, dtype).npu()
    KV = generate_tensor(KV_shape, dtype).npu()

    Indices_cpu = torch.zeros(Indices_shape, dtype=torch.int32)
    for b in range(1):
        for s in range(32):
            for g in range(1):
                Indices_cpu[b, s, g, :] = torch.randint(
                    0, s + 1, (16,), dtype=torch.int32
                )
    Indices = Indices_cpu.npu()

    O = torch.zeros(Output_shape, dtype=torch.float16).npu()
    Lse = torch.zeros(Lse_shape, dtype=torch.float32).npu()

    compiled_kernel(Q, KV, Indices, O, Lse)

    O_ref, Lse_ref = sparse_mla_fwd_torch(
        Q=Q,
        KV=KV,
        Indices=Indices,
        dim=128,
        tail_dim=64,
        kv_group=1,
        sm_scale=None,
    )
    torch.testing.assert_close(O, O_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(Lse, Lse_ref, rtol=1e-2, atol=1e-2)

    print("\033[92mDemo check passed!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
