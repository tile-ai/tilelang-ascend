import os
import torch
import tilelang as tl
import tilelang.language as T
from sparse_mla_fwd import sparse_mla_fwd, sparse_mla_fwd_torch

dtype = "float16"
accum_dtype = "float32"


@tl.jit(target="npuir")
def preprocess(B, S, H, D, block_ND=32, num_stages=5):
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H * T.ceildiv(S, block_ND) * B, is_npu=True) as (cid, _):
            SB = T.ceildiv(S, block_ND)
            bz = cid // (H * SB)
            rem = cid % (H * SB)
            bx = rem // SB
            by = rem % SB
            o_tmp = T.alloc_shared([block_ND, block_ND], dtype)
            do_tmp = T.alloc_shared([block_ND, block_ND], dtype)
            o = T.alloc_shared([block_ND, block_ND], accum_dtype)
            do = T.alloc_shared([block_ND, block_ND], accum_dtype)
            delta = T.alloc_shared([block_ND, 1], accum_dtype)
            acc = T.alloc_shared([block_ND, block_ND], accum_dtype)

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o_tmp,
                )
                T.vcast(o_tmp, o)
                T.copy(
                    dO[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do_tmp,
                )
                T.vcast(do_tmp, do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]

            T.reduce_sum(acc, delta, dim=1)
            T.copy(delta[:, 0], Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tl.jit(target="npuir")
def postprocess(B, S_kv, D, D_tail, kv_group=1, block_N=64):
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N) * kv_group * B, is_npu=True) as (cid, _):
            SB = T.ceildiv(S_kv, block_N)
            bz = cid // (SB * kv_group)
            rem = cid % (SB * kv_group)

            by = rem // SB
            bx = rem % SB

            buf = T.alloc_shared([block_N, D + D_tail], accum_dtype)

            T.copy(dKV[bz, bx * block_N : (bx + 1) * block_N, by, :], buf)
            T.copy(buf, dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :])

    return postprocess_kernel


@tl.jit(target="npuir")
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    indices_dtype="int32",
):
    block_size = max(block_size, 16)
    block_size = (block_size + 15) // 16 * 16

    assert topk % block_size == 0, (
        f"topk({topk}) must be divisible by block_size({block_size})"
    )
    assert is_causal is True, "non-casual is not supported now"
    assert topk % block_size == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    H = H_kv
    padded_H = max(tl.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tl.cdiv(topk, block_size)

    split_store = 2
    if split_store > BS or BS % split_store != 0:
        split_store = 1

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S * B * (kv_group * NH), is_npu=True) as (cid, _):
            bz = cid % (kv_group * NH)
            tmp = cid // (kv_group * NH)
            by = tmp % B
            s_i = tmp // B

            Q_shared = T.alloc_shared([block_H, D], dtype)
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

            acc_p = T.alloc_shared([block_H, BS], accum_dtype)
            tmp = T.alloc_shared([block_H, BS], accum_dtype)
            acc_dp = T.alloc_shared([block_H, BS], accum_dtype)
            acc_dq = T.alloc_shared([block_H, D], accum_dtype)
            acc_dq_tail = T.alloc_shared([block_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_shared([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_shared([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared(
                [BS // split_store, D_tail], accum_dtype
            )

            max_kv_i = s_i

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            Lse_shared = T.alloc_shared([B, S, H], accum_dtype)
            T.copy(Lse, Lse_shared)
            acc_p_reshape = T.alloc_shared([1, 1, block_H, BS], accum_dtype)
            Lse_reshape = T.alloc_shared([B, S, H, 1], accum_dtype)

            Delta_shared = T.alloc_shared([B, S, H], accum_dtype)
            T.copy(Delta, Delta_shared)
            acc_dp_reshape = T.alloc_shared([1, 1, block_H, BS], accum_dtype)
            Delta_reshape = T.alloc_shared([B, S, H, 1], accum_dtype)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Compute attention scores
                for h_i, bi_i in T.Parallel(block_H, BS):
                    if Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i:
                        acc_p[h_i, bi_i] = 0
                    else:
                        acc_p[h_i, bi_i] = -T.infinity(accum_dtype)

                # Load KV for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i
                    ]

                T.gemm(Q_shared, KV_shared, acc_p, b_transpose=True)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[
                        by,
                        Indices[by, s_i, bz // NH, i_i * BS + bi_i],
                        bz // NH,
                        D + d_i,
                    ]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, b_transpose=True)

                T.reshape(acc_p, acc_p_reshape)
                T.reshape(Lse_shared, Lse_reshape)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p_reshape[0, 0, h_i, bi_i] = (
                        acc_p_reshape[0, 0, h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse_reshape[by, s_i, bz * block_H + h_i, 0]
                    )
                T.reshape(acc_p_reshape, acc_p)
                T.vexp2(acc_p, acc_p, tmp)

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, b_transpose=True, initC=True)

                T.reshape(acc_dp, acc_dp_reshape)
                T.reshape(Delta_shared, Delta_reshape)
                T.reshape(acc_p, acc_p_reshape)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp_reshape[0, 0, h_i, bi_i] = (
                        acc_p_reshape[0, 0, h_i, bi_i]
                        * (
                            acc_dp_reshape[0, 0, h_i, bi_i]
                            - Delta_reshape[by, s_i, bz * block_H + h_i, 0]
                        )
                        * sm_scale
                    )
                T.reshape(acc_dp_reshape, acc_dp)

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, a_transpose=True, initC=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, a_transpose=True)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, a_transpose=True)

                for s in range(split_store):
                    T.copy(
                        acc_dkv[
                            s * (BS // split_store) : (s + 1) * (BS // split_store), :
                        ],
                        acc_dkv_shared,
                    )
                    T.copy(
                        acc_dkv_tail[
                            s * (BS // split_store) : (s + 1) * (BS // split_store), :
                        ],
                        acc_dkv_tail_shared,
                    )

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        if (
                            Indices[
                                by,
                                s_i,
                                bz // NH,
                                i_i * BS + bi_i + s * (BS // split_store),
                            ]
                            <= max_kv_i
                        ):
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    Indices[
                                        by,
                                        s_i,
                                        bz // NH,
                                        i_i * BS + bi_i + s * (BS // split_store),
                                    ],
                                    bz // NH,
                                    d_i * 4,
                                ],
                                acc_dkv_shared[bi_i, d_i * 4],
                                [1, 4],
                            )

                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        if (
                            Indices[
                                by,
                                s_i,
                                bz // NH,
                                i_i * BS + bi_i + s * (BS // split_store),
                            ]
                            <= max_kv_i
                        ):
                            T.atomic_addx4(
                                dKV[
                                    by,
                                    Indices[
                                        by,
                                        s_i,
                                        bz // NH,
                                        i_i * BS + bi_i + s * (BS // split_store),
                                    ],
                                    bz // NH,
                                    D + d_i * 4,
                                ],
                                acc_dkv_tail_shared[bi_i, d_i * 4],
                                [1, 4],
                            )

            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


def generate_tensor(shape, dtype, clear=False):
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
    raise ValueError(f'Invalid parameter "dtype" is found : {dtype}')


def build_indices_no_conflict(B, S, kv_group, topk, mode="diag", S_kv=None):
    Indices_cpu = torch.zeros([B, S, kv_group, topk], dtype=torch.int32)
    pad_val = S_kv if S_kv is not None else S

    for b in range(B):
        for g in range(kv_group):
            for s in range(S):
                if mode == "diag":
                    assert topk == 1
                    Indices_cpu[b, s, g, 0] = s
                elif mode == "unique":
                    valid_n = min(topk, s + 1)
                    vals = torch.randperm(s + 1)[:valid_n].to(torch.int32)
                    pad = torch.full((topk - valid_n,), pad_val, dtype=torch.int32)
                    vals = torch.cat([vals, pad], dim=0)
                    Indices_cpu[b, s, g, :] = vals
                elif mode == "no_conflict":
                    Indices_cpu[b, s, g, 0] = s
                    Indices_cpu[b, s, g, 1:] = pad_val
                else:
                    raise ValueError(f"unknown mode: {mode}")

    return Indices_cpu


def ref_delta_torch(o, do):
    # delta[b, s, h] = sum_d o * do
    return (o.float() * do.float()).sum(dim=-1)


def ref_sparse_mla_bwd_torch(
    Q,
    KV,
    dO,
    Indices,
    Lse,
    Delta,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=1,
):
    assert is_causal is True, "non-causal is not supported now"

    B, S, H, dim_plus_tail = Q.shape
    _, S_kv, kv_group_kv, kv_dim = KV.shape
    assert kv_group_kv == kv_group
    assert kv_dim == dim_plus_tail
    assert D + D_tail == dim_plus_tail
    assert dO.shape == (B, S, H, D)
    assert Indices.shape == (B, S, kv_group, topk)
    assert Lse.shape == (B, S, H)
    assert Delta.shape == (B, S, H)

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)

    H_kv = H // kv_group
    dq = torch.zeros_like(Q, dtype=torch.float32)
    dkv = torch.zeros_like(KV, dtype=torch.float32)

    Qf = Q.float()
    KVf = KV.float()
    dOf = dO.float()
    Lsef = Lse.float()
    Deltaf = Delta.float()

    for b in range(B):
        for s in range(S):
            for h in range(H):
                g = h // H_kv

                q_full = Qf[b, s, h, :]  # [D + D_tail]
                q_main = q_full[:D]  # [D]
                q_tail = q_full[D:]  # [D_tail]
                do_main = dOf[b, s, h, :]  # [D]

                lse_val = Lsef[b, s, h]
                delta_val = Deltaf[b, s, h]

                for t in range(topk):
                    idx = int(Indices[b, s, g, t].item())
                    if idx > s:
                        continue
                    kv_full = KVf[b, idx, g, :]  # [D + D_tail]
                    kv_main = kv_full[:D]
                    kv_tail = kv_full[D:]

                    # acc_p after gemm + scale/log2e - lse, then exp2
                    score = torch.dot(q_full, kv_full)
                    p = torch.exp2(score * (sm_scale * 1.44269504) - lse_val)

                    # acc_dp only uses dO and KV[:D]
                    acc_dp = torch.dot(do_main, kv_main)

                    # dP
                    dP = p * (acc_dp - delta_val) * sm_scale

                    # dQ
                    dq[b, s, h, :D] += dP * kv_main
                    if D_tail > 0:
                        dq[b, s, h, D:] += dP * kv_tail

                    # dKV main
                    dkv[b, idx, g, :D] += dP * q_main + p * do_main

                    # dKV tail
                    if D_tail > 0:
                        dkv[b, idx, g, D:] += dP * q_tail

    return dq.to(Q.dtype), dkv


def sparse_mla_bwd(
    q,
    kv,
    o,
    do,
    indices,
    lse,
    sm_scale=None,
    is_casual=True,
    return_kernel=False,
    delta=None,
    D=None,
    block_size=None,
    num_stages=0,
):

    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, kv_dim = kv.shape

    topk = indices.shape[-1]

    is_causal = is_casual

    if D is None:
        D = o.shape[-1]

    assert dim_plus_tail_dim >= D, (
        f"D({D}) cannot be larger than q.shape[-1]({dim_plus_tail_dim})"
    )
    D_tail = dim_plus_tail_dim - D

    if block_size is None:
        block_size = topk

    assert topk % block_size == 0, (
        f"topk({topk}) must be divisible by block_size({block_size})"
    )

    # compile kernels
    preprocess_kernel = preprocess(B, S, H, D)
    bwd_kernel = bwd(
        B,
        S,
        S_kv,
        H,
        D,
        D_tail,
        topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_causal,
        block_size=block_size,
        num_stages=num_stages,
        indices_dtype="int32",
    )
    postprocess_kernel = postprocess(
        B,
        S_kv,
        D,
        D_tail,
        kv_group=kv_group,
    )

    # delta
    if delta is None:
        delta = torch.zeros((B, S, H), dtype=torch.float32, device=o.device)
        preprocess_kernel(o, do, delta)
    else:
        assert delta.shape == (B, S, H)
        assert delta.is_contiguous()

    # bwd
    dq = torch.zeros_like(q, dtype=q.dtype)
    dkv_fp32 = torch.zeros(
        (B, S_kv, kv_group, D + D_tail), dtype=torch.float32, device=kv.device
    )

    bwd_kernel(q, kv, do, indices, lse, delta, dq, dkv_fp32)

    # postprocess: fp32 -> dtype
    dkv = torch.zeros_like(kv, dtype=kv.dtype)
    postprocess_kernel(dkv_fp32, dkv)

    if return_kernel:
        return dq, dkv, (preprocess_kernel, bwd_kernel, postprocess_kernel)

    return dq, dkv


def run_test():
    B = 1
    S = 8
    S_kv = 8
    H = 16
    D = 32
    D_tail = 4
    topk = 16
    kv_group = 1
    block_size = 16

    q_shape = [B, S, H, D + D_tail]
    kv_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    lse_shape = [B, S, H]

    Q = generate_tensor(q_shape, dtype).npu().contiguous()
    KV = generate_tensor(kv_shape, dtype).npu().contiguous()
    dO = generate_tensor(o_shape, dtype).npu().contiguous()
    Indices_cpu = build_indices_no_conflict(
        B, S, kv_group, topk, mode="no_conflict", S_kv=S_kv
    )
    Indices = Indices_cpu.npu().contiguous()

    O = generate_tensor(o_shape, dtype).npu().contiguous()
    Lse = generate_tensor(lse_shape, accum_dtype).npu().contiguous()

    sparse_mla_fwd_kernel = sparse_mla_fwd(
        B,
        S,
        S_kv,
        H,
        D,
        D_tail,
        topk,
        kv_group,
        None,
        block_size,
        2,
    )
    sparse_mla_fwd_kernel(Q, KV, Indices, O, Lse)

    Q_ref_in = Q.detach().clone()
    KV_ref_in = KV.detach().clone()
    dO_ref_in = dO.detach().clone()
    Indices_ref_in = Indices.detach().clone()

    dQ, dKV = sparse_mla_bwd(
        q=Q,
        kv=KV,
        o=O,
        do=dO,
        indices=Indices,
        lse=Lse,
        sm_scale=None,
        is_casual=True,
        return_kernel=False,
        delta=None,
        D=D,
        block_size=block_size,
        num_stages=0,
    )

    # torch reference
    with torch.no_grad():
        O_ref_in, Lse_ref_in = sparse_mla_fwd_torch(
            Q,
            KV,
            Indices,
            D,
            D_tail,
            kv_group,
            sm_scale=None,
        )
        Delta_ref = ref_delta_torch(O_ref_in, dO_ref_in)

        dQ_ref, dKV_ref = ref_sparse_mla_bwd_torch(
            Q=Q_ref_in,
            KV=KV_ref_in,
            dO=dO_ref_in,
            Indices=Indices_ref_in,
            Lse=Lse_ref_in,
            Delta=Delta_ref,
            D=D,
            D_tail=D_tail,
            topk=topk,
            kv_group=kv_group,
            sm_scale=None,
            is_causal=True,
            block_size=block_size,
        )

    dKV_ref_cast = dKV_ref.to(KV.dtype)

    dQ_cpu = dQ.detach().float().cpu()
    dKV_cpu = dKV.detach().float().cpu()
    dQ_ref_cpu = dQ_ref.detach().float().cpu()
    dKV_ref_cpu = dKV_ref_cast.detach().float().cpu()

    torch.testing.assert_close(dQ_cpu, dQ_ref_cpu, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dKV_cpu, dKV_ref_cpu, atol=1e-2, rtol=1e-2)

    print("\033[92mCheck passed!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
