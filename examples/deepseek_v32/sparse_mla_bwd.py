import argparse
import torch
import tilelang as tl
import tilelang.language as T
# from utils import assert_tensors_similar
import os
torch.npu.set_device(6)
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--B", type=int, default=1, help="")
parser.add_argument("--S", type=int, default=128, help="")
parser.add_argument("--SKV", type=int, default=128, help="")
parser.add_argument("--H", type=int, default=32, help="")
parser.add_argument("--HKV", type=int, default=512, help="")
parser.add_argument("--DQK", type=int, default=64, help="")
parser.add_argument("--DV", type=int, default=64, help="")
parser.add_argument("--topk", type=int, default=32, help="")
parser.add_argument("--block_I", type=int, default=32, help="")
parser.add_argument("--num_stages", type=int, default=1, help="")

dtype = "float16"
accum_dtype = "float32"
@tl.jit(target="npuir")
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5
):
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
            o = T.alloc_shared([block_ND, block_ND], accum_dtype)
            do = T.alloc_shared([block_ND, block_ND], accum_dtype)
            delta = T.alloc_shared([block_ND, 1], accum_dtype)
            acc = T.alloc_shared([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND),num_stages=num_stages):
                T.copy(O[bz,by * block_ND : (by + 1) * block_ND,bx,k * block_ND : (k + 1) * block_ND],o)

                T.copy(dO[bz,by * block_ND : (by + 1) * block_ND,bx,k * block_ND : (k + 1) * block_ND],do)

                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]

            T.reduce_sum(acc, delta, dim=1)
            T.copy(delta[:, 0],Delta[bz,by * block_ND : (by + 1) * block_ND,bx])
    return preprocess_kernel

@tl.jit(target="npuir")
def postprocess(
    B,
    S_kv,
    D,
    D_tail,
    kv_group=1,
    block_N=64
):
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

            T.copy(dKV[bz, bx * block_N:(bx + 1) * block_N,by, :], buf)
            T.copy(buf, dKV_out[bz, bx * block_N:(bx + 1) * block_N, by, :])

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
    assert is_causal == True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"

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
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

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

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, b_transpose=True)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, b_transpose=True)

                T.reshape(acc_p,acc_p_reshape)
                T.reshape(Lse_shared,Lse_reshape)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p_reshape[0, 0, h_i, bi_i] = acc_p_reshape[0, 0, h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse_reshape[by, s_i, bz * block_H + h_i, 0]
                T.reshape(acc_p_reshape,acc_p)
                T.vexp2(acc_p,acc_p,tmp)

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, b_transpose=True, initC=True)

                T.reshape(acc_dp, acc_dp_reshape)
                T.reshape(Delta_shared, Delta_reshape)
                T.reshape(acc_p,acc_p_reshape)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp_reshape[0, 0, h_i, bi_i] = acc_p_reshape[0, 0, h_i, bi_i] * (acc_dp_reshape[0, 0, h_i, bi_i] - Delta_reshape[by, s_i, bz * block_H + h_i, 0]) * sm_scale
                T.reshape(acc_dp_reshape, acc_dp)

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, a_transpose=True, initC=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, a_transpose=True)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, a_transpose=True)
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, D + d_i * 4],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


def generate_tensor(shape, dtype, clear=False):
    """Generate tensor with specified shape and data type"""
    if clear:
        return torch.zeros(shape, dtype=eval("torch." + dtype))
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(size=shape, dtype=eval("torch." + dtype))
    if dtype in ("int32", "int64", "int16"):
        return torch.randint(low=0, high=10000, size=shape, dtype=eval("torch." + dtype))
    if dtype == "int8":
        return torch.randint(low=0, high=127, size=shape, dtype=eval("torch." + dtype))
    if dtype == "bool":
        return torch.randint(low=0, high=2, size=shape).bool()
    raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

import torch


def ref_sparse_mla_bwd_torch(
    Q,          # [B, S, H, D + D_tail]
    KV,         # [B, S_kv, kv_group, D + D_tail]
    dO,         # [B, S, H, D]
    Indices,    # [B, S, kv_group, topk]
    Lse,        # [B, S, H]
    Delta,      # [B, S, H]
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
):
    assert is_causal is True
    assert topk % block_size == 0

    B, S, H_total, DH = Q.shape
    _, S_kv, kv_group2, DH2 = KV.shape
    _, _, H2, D_out = dO.shape

    assert kv_group2 == kv_group
    assert H_total == H2
    assert DH == DH2 == D + D_tail
    assert D_out == D

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    H_kv = H_total // kv_group
    padded_H = max(1 << (H_kv - 1).bit_length(), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = (topk + BS - 1) // BS

    accum_dtype = Lse.dtype
    device = Q.device

    Q_main = Q[..., :D].to(accum_dtype)
    Q_tail = Q[..., D:].to(accum_dtype)
    KV_main = KV[..., :D].to(accum_dtype)
    KV_tail = KV[..., D:].to(accum_dtype)
    dO_acc = dO.to(accum_dtype)
    Lse_acc = Lse.to(accum_dtype)
    Delta_acc = Delta.to(accum_dtype)

    dQ_ref = torch.zeros_like(Q, dtype=accum_dtype, device=device)
    dKV_ref = torch.zeros((B, S_kv, kv_group, D + D_tail), dtype=accum_dtype, device=device)

    for by in range(B):
        for s_i in range(S):
            max_kv_i = s_i

            for bz in range(kv_group * NH):
                g = bz // NH
                head_start = bz * block_H
                head_end = min((bz + 1) * block_H, H_total)
                if head_start >= H_total:
                    continue

                cur_block_H = head_end - head_start

                q_main_blk = Q_main[by, s_i, head_start:head_end, :]   # [cur_block_H, D]
                q_tail_blk = Q_tail[by, s_i, head_start:head_end, :]   # [cur_block_H, D_tail]
                do_blk = dO_acc[by, s_i, head_start:head_end, :]       # [cur_block_H, D]
                lse_blk = Lse_acc[by, s_i, head_start:head_end]        # [cur_block_H]
                delta_blk = Delta_acc[by, s_i, head_start:head_end]    # [cur_block_H]

                acc_dq_main = torch.zeros((cur_block_H, D), dtype=accum_dtype, device=device)
                acc_dq_tail = torch.zeros((cur_block_H, D_tail), dtype=accum_dtype, device=device)

                for i_i in range(NS):
                    idx_block = Indices[by, s_i, g, i_i * BS:(i_i + 1) * BS].long()
                    assert idx_block.numel() == BS

                    valid = idx_block <= max_kv_i
                    idx_safe = idx_block.clamp(min=0, max=S_kv - 1)

                    k_main_blk = KV_main[by, idx_safe, g, :]   # [BS, D]
                    k_tail_blk = KV_tail[by, idx_safe, g, :]   # [BS, D_tail]

                    acc_p = torch.full(
                        (cur_block_H, BS),
                        fill_value=-torch.inf,
                        dtype=accum_dtype,
                        device=device,
                    )
                    acc_p[:, valid] = 0

                    acc_p = acc_p + q_main_blk @ k_main_blk.transpose(0, 1)
                    acc_p = acc_p + q_tail_blk @ k_tail_blk.transpose(0, 1)

                    acc_p = acc_p * sm_scale_mul_reciprocal_log2 - lse_blk[:, None]
                    p = torch.pow(torch.tensor(2.0, dtype=accum_dtype, device=device), acc_p)

                    acc_dp = do_blk @ k_main_blk.transpose(0, 1)
                    acc_dp = p * (acc_dp - delta_blk[:, None]) * sm_scale

                    acc_dq_main = acc_dq_main + acc_dp @ k_main_blk
                    acc_dq_tail = acc_dq_tail + acc_dp @ k_tail_blk

                    dkv_main_blk = acc_dp.transpose(0, 1) @ q_main_blk + p.transpose(0, 1) @ do_blk
                    dkv_tail_blk = acc_dp.transpose(0, 1) @ q_tail_blk

                    for bi in range(BS):
                        if valid[bi]:
                            tgt = idx_safe[bi].item()
                            dKV_ref[by, tgt, g, :D] += dkv_main_blk[bi]
                            dKV_ref[by, tgt, g, D:] += dkv_tail_blk[bi]

                dQ_ref[by, s_i, head_start:head_end, :D] = acc_dq_main
                dQ_ref[by, s_i, head_start:head_end, D:] = acc_dq_tail

    return dQ_ref.to(Q.dtype), dKV_ref

def run_test():
    B = 1
    S = 32
    S_kv = 32
    H = 16
    D = 28
    D_tail = 4
    topk = 32
    kv_group = 1

    compiled_kernel = bwd(B, S, S_kv, H, D, D_tail, topk, kv_group)
    print("compile finished!")

    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    Q = generate_tensor(q_shape, dtype).npu()
    KV = generate_tensor(k_shape, dtype).npu()
    dO = generate_tensor(o_shape, dtype).npu()

    Indices_cpu = torch.zeros(indices_shape, dtype=torch.int32)
    for s in range(S):
        Indices_cpu[0, s, 0, :] = torch.randint(0, s + 1, (topk,), dtype=torch.int32)
    Indices = Indices_cpu.npu()

    Lse = generate_tensor(lse_shape, accum_dtype).npu()
    Delta = generate_tensor(delta_shape, accum_dtype).npu()

    # 输出建议清零，避免旧垃圾值干扰
    dQ = generate_tensor(q_shape, dtype, clear=True).npu()
    dKV = generate_tensor(k_shape, accum_dtype, clear=True).npu()

    # 保存一份输入给 torch reference 用
    Q_ref_in = Q.detach().clone()
    KV_ref_in = KV.detach().clone()
    dO_ref_in = dO.detach().clone()
    Indices_ref_in = Indices.detach().clone()
    Lse_ref_in = Lse.detach().clone()
    Delta_ref_in = Delta.detach().clone()

    # 1) 先跑 kernel
    compiled_kernel(Q, KV, dO, Indices, Lse, Delta, dQ, dKV)
    print("kernel finished!")

    # # 2) 再跑 torch reference
    # with torch.no_grad():
    #     dQ_ref, dKV_ref = ref_sparse_mla_bwd_torch(
    #         Q=Q_ref_in,
    #         KV=KV_ref_in,
    #         dO=dO_ref_in,
    #         Indices=Indices_ref_in,
    #         Lse=Lse_ref_in,
    #         Delta=Delta_ref_in,
    #         D=D,
    #         D_tail=D_tail,
    #         topk=topk,
    #         kv_group=kv_group,
    #         sm_scale=None,
    #         is_causal=True,
    #         block_size=32,
    #     )
    # print("torch reference finished!")

    # # 3) 比较
    # dQ_cpu = dQ.detach().float().cpu()
    # dKV_cpu = dKV.detach().float().cpu()
    # dQ_ref_cpu = dQ_ref.detach().float().cpu()
    # dKV_ref_cpu = dKV_ref.detach().float().cpu()

    # dq_max_diff = (dQ_cpu - dQ_ref_cpu).abs().max().item()
    # dkv_max_diff = (dKV_cpu - dKV_ref_cpu).abs().max().item()

    # print(f"dQ  max abs diff:  {dq_max_diff}")
    # print(f"dKV max abs diff:  {dkv_max_diff}")

    # torch.testing.assert_close(
    #     dQ_cpu,
    #     dQ_ref_cpu,
    #     atol=1e-2,
    #     rtol=1e-2,
    #     msg=f"dQ mismatch, max abs diff = {dq_max_diff}",
    # )

    # torch.testing.assert_close(
    #     dKV_cpu,
    #     dKV_ref_cpu,
    #     atol=1e-2,
    #     rtol=1e-2,
    #     msg=f"dKV mismatch, max abs diff = {dkv_max_diff}",
    # )

    # print("\033[92mTorch reference check passed!\033[0m")
if __name__ == "__main__":
    main_args = parser.parse_args()
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()