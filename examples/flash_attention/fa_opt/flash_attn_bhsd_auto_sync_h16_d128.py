import argparse
import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

import torch

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()

L0_MAX_SIZE = 64 * 1024  # 64KB
NUM_CORES = 24  # 910B has 24 AI Cores

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
    num_stages=8,  # 固定为 4，避免编译器 bug
):
    assert heads_q % heads_kv == 0, "heads_q must be a multiple of heads_kv"
    block_M, block_N = 128, 128
    assert dim == 128, "dim must be 128"
    assert seq_len % block_N == 0, f"seq_len ({seq_len}) must be divisible by block_N ({block_N})"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape_q = [batch, heads_q, seq_len, dim]
    shape_kv = [batch, heads_kv, seq_len, dim]

    num_seq_blocks = seq_len // block_M
    block_num = num_seq_blocks * heads_q * batch
    num_iters = T.ceildiv(seq_len, block_N)

    # 确保 num_iters 能被 num_stages 整除
    assert num_iters % num_stages == 0, f"num_iters ({num_iters}) must be divisible by num_stages ({num_stages})"

    num_outer = num_iters // num_stages

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    SEM_WS1_C2V = 0
    SEM_WS1_V2C = 1
    SEM_WS2_V2C = 2
    SEM_WS2_C2V = 3
    SEM_WS3_C2V = 4
    SEM_WS3_V2C = 5

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_q, dtype),
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            half_M = block_M // 2

            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)

            io_buf = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            tmp_ub = T.alloc_ub([DataType(accum_dtype).bits // 8 * block_M // 2 * 128], "uint8")
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_C2V)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)
                    kv_by = by // (heads_q // heads_kv)

                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.barrier_all()

                    for k in T.serial(num_outer):
                        # 由于整除，batch_iters 固定为 num_stages
                        T.wait_cross_flag(SEM_WS1_V2C)

                        # Process pairs: num_stages // 2 = 2 iterations
                        for p in T.serial(num_stages // 2):
                            base_i = p * 2
                            base_idx = k * num_stages + base_i

                            T.copy(K[bz, kv_by, base_idx * block_N : (base_idx + 1) * block_N, :], k_l1)
                            T.copy(q_l1, l0a[0, :, :])
                            T.copy(k_l1, l0b[0, :, :])
                            T.mma(l0a[0, :, :], l0b[0, :, :], l0c[0, :, :], init=True)
                            T.copy(l0c[0, :, :], workspace_1[cid, base_i, :, :])

                            T.copy(K[bz, kv_by, (base_idx + 1) * block_N : (base_idx + 2) * block_N, :], k_l1)
                            T.copy(q_l1, l0a[1, :, :])
                            T.copy(k_l1, l0b[1, :, :])
                            T.mma(l0a[1, :, :], l0b[1, :, :], l0c[1, :, :], init=True)
                            T.copy(l0c[1, :, :], workspace_1[cid, base_i + 1, :, :])

                            T.set_cross_flag("FIX", SEM_WS1_C2V)

                        T.wait_cross_flag(SEM_WS3_V2C)

                        for p in T.serial(num_stages // 2):
                            base_i = p * 2
                            base_idx = k * num_stages + base_i

                            T.copy(V[bz, kv_by, base_idx * block_N : (base_idx + 1) * block_N, :], v_l1)
                            T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, base_i, :, :], p_l1)
                            T.copy(v_l1, l0b[0, :, :])
                            T.copy(p_l1, l0a[0, :, :])
                            T.mma(l0a[0, :, :], l0b[0, :, :], l0c[0, :, :], init=True)
                            T.copy(l0c[0, :, :], workspace_3[cid, base_i, :, :])

                            T.copy(V[bz, kv_by, (base_idx + 1) * block_N : (base_idx + 2) * block_N, :], v_l1)
                            T.copy(workspace_2[cid, base_i + 1, :, :], p_l1)
                            T.copy(v_l1, l0b[1, :, :])
                            T.copy(p_l1, l0a[1, :, :])
                            T.mma(l0a[1, :, :], l0b[1, :, :], l0c[1, :, :], init=True)
                            T.copy(l0c[1, :, :], workspace_3[cid, base_i + 1, :, :])

                            T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    for k in T.serial(num_outer):
                        T.wait_cross_flag(SEM_WS2_C2V)

                        # === Softmax loop with cross-pair prefetch ===
                        # Prologue: wait for first pair, prefetch stage 0
                        T.wait_cross_flag(SEM_WS1_C2V)
                        T.copy(workspace_1[cid, 0, vid * half_M : vid * half_M + half_M, :], io_buf)

                        # Main loop: pairs 0..num_stages//2-2
                        for p in T.serial(num_stages // 2 - 1):
                            base_i = p * 2

                            # --- Stage base_i (even): io_buf already prefetched ---
                            T.copy(io_buf, work_ub)
                            T.copy(workspace_1[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.reduce_max(work_ub, neg_sm[0, :, :], tmp_ub, dim=-1)
                            T.tile.mul(neg_sm[0, :, :], neg_sm[0, :, :], -sm_scale)
                            T.tile.min(neg_sm[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[0, :, :], tmp_ub)
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)
                            T.copy(work_ub, acc_s_half)
                            T.copy(acc_s_half, workspace_2[cid, base_i, vid * half_M : vid * half_M + half_M, :])
                            T.reduce_sum(work_ub, sumexp_is[base_i, :, :], tmp_ub, dim=-1)
                            T.tile.sub(r_factors[base_i, :, :], neg_sm[0, :, :], neg_sm[1, :, :])

                            # --- Stage base_i+1 (odd): io_buf already prefetched ---
                            T.copy(io_buf, work_ub)
                            T.wait_cross_flag(SEM_WS1_C2V)  # wait for NEXT pair
                            T.copy(
                                workspace_1[cid, base_i + 2, vid * half_M : vid * half_M + half_M, :], io_buf
                            )  # prefetch first of next pair
                            T.reduce_max(work_ub, neg_sm[1, :, :], tmp_ub, dim=-1)
                            T.tile.mul(neg_sm[1, :, :], neg_sm[1, :, :], -sm_scale)
                            T.tile.min(neg_sm[1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[1, :, :], tmp_ub)
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)
                            T.copy(work_ub, acc_s_half)
                            T.copy(acc_s_half, workspace_2[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :])
                            T.reduce_sum(work_ub, sumexp_is[base_i + 1, :, :], tmp_ub, dim=-1)
                            T.tile.sub(r_factors[base_i + 1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])

                            T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        # Epilogue: last pair (no next pair to prefetch)
                        base_i = (num_stages // 2 - 1) * 2

                        T.copy(io_buf, work_ub)
                        T.copy(workspace_1[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                        T.reduce_max(work_ub, neg_sm[0, :, :], tmp_ub, dim=-1)
                        T.tile.mul(neg_sm[0, :, :], neg_sm[0, :, :], -sm_scale)
                        T.tile.min(neg_sm[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])
                        T.tile.broadcast(buf_2d, neg_sm[0, :, :], tmp_ub)
                        T.tile.axpy(buf_2d, work_ub, sm_scale)
                        T.tile.exp(work_ub, buf_2d)
                        T.copy(work_ub, acc_s_half)
                        T.copy(acc_s_half, workspace_2[cid, base_i, vid * half_M : vid * half_M + half_M, :])
                        T.reduce_sum(work_ub, sumexp_is[base_i, :, :], tmp_ub, dim=-1)
                        T.tile.sub(r_factors[base_i, :, :], neg_sm[0, :, :], neg_sm[1, :, :])

                        T.copy(io_buf, work_ub)
                        T.reduce_max(work_ub, neg_sm[1, :, :], tmp_ub, dim=-1)
                        T.tile.mul(neg_sm[1, :, :], neg_sm[1, :, :], -sm_scale)
                        T.tile.min(neg_sm[1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])
                        T.tile.broadcast(buf_2d, neg_sm[1, :, :], tmp_ub)
                        T.tile.axpy(buf_2d, work_ub, sm_scale)
                        T.tile.exp(work_ub, buf_2d)
                        T.copy(work_ub, acc_s_half)
                        T.copy(acc_s_half, workspace_2[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :])
                        T.reduce_sum(work_ub, sumexp_is[base_i + 1, :, :], tmp_ub, dim=-1)
                        T.tile.sub(r_factors[base_i + 1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])

                        T.set_cross_flag("MTE3", SEM_WS2_V2C)

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # === Acc_o loop with cross-pair prefetch ===
                        # Prologue: wait for first pair, prefetch stage 0
                        T.wait_cross_flag(SEM_WS3_C2V)
                        T.copy(workspace_3[cid, 0, vid * half_M : vid * half_M + half_M, :], io_buf)

                        # Main loop: pairs 0..num_stages//2-2
                        for p in T.serial(num_stages // 2 - 1):
                            base_i = p * 2

                            # --- Stage base_i: io_buf already prefetched ---
                            T.tile.exp(r_factors[base_i, :, :], r_factors[base_i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[base_i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[base_i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[base_i, :, :], tmp_ub)
                            T.tile.mul(acc_o, acc_o, buf_2d)
                            T.copy(io_buf, work_ub)
                            T.copy(workspace_3[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.tile.add(acc_o, acc_o, work_ub)

                            # --- Stage base_i+1: io_buf already prefetched ---
                            T.tile.exp(r_factors[base_i + 1, :, :], r_factors[base_i + 1, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[base_i + 1, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[base_i + 1, :, :])
                            T.tile.broadcast(buf_2d, r_factors[base_i + 1, :, :], tmp_ub)
                            T.tile.mul(acc_o, acc_o, buf_2d)
                            T.copy(io_buf, work_ub)
                            T.wait_cross_flag(SEM_WS3_C2V)  # wait for NEXT pair
                            T.copy(workspace_3[cid, base_i + 2, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.tile.add(acc_o, acc_o, work_ub)

                        # Epilogue: last pair (no next pair to prefetch)
                        base_i = (num_stages // 2 - 1) * 2

                        T.tile.exp(r_factors[base_i, :, :], r_factors[base_i, :, :])
                        T.tile.mul(sumexp, sumexp, r_factors[base_i, :, :])
                        T.tile.add(sumexp, sumexp, sumexp_is[base_i, :, :])
                        T.tile.broadcast(buf_2d, r_factors[base_i, :, :], tmp_ub)
                        T.tile.mul(acc_o, acc_o, buf_2d)
                        T.copy(io_buf, work_ub)
                        T.copy(workspace_3[cid, base_i + 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                        T.tile.add(acc_o, acc_o, work_ub)

                        T.copy(io_buf, work_ub)
                        T.tile.exp(r_factors[base_i + 1, :, :], r_factors[base_i + 1, :, :])
                        T.tile.mul(sumexp, sumexp, r_factors[base_i + 1, :, :])
                        T.tile.add(sumexp, sumexp, sumexp_is[base_i + 1, :, :])
                        T.tile.broadcast(buf_2d, r_factors[base_i + 1, :, :], tmp_ub)
                        T.tile.mul(acc_o, acc_o, buf_2d)
                        T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    T.tile.broadcast(buf_2d, sumexp, tmp_ub)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    T.copy(acc_o, acc_s_half)
                    T.barrier_all()
                    T.copy(
                        acc_s_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :]
                    )

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--S", type=int, default=4096, help="seq len")
    parser.add_argument("--H", type=int, default=16, help="attention head size")
    parser.add_argument("--q-heads", type=int, default=None, help="query head size")
    parser.add_argument("--kv-heads", type=int, default=None, help="kv head size")
    parser.add_argument("--D", type=int, default=128, help="hidden dim")
    parser.add_argument("--no-check", action="store_true", help="disable reference check")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    Q_H = args.q_heads or H
    KV_H = args.kv_heads or H

    func = flash_attention_fwd(
        batch=B,
        seq_len=S,
        heads_q=Q_H,
        heads_kv=KV_H,
        dim=D,
    )
    print(func.get_kernel_source())

    def ref_flash_attn(q, k, v):
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q = q.float()
        k = k.float()
        v = v.float()

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return output.to(torch.float16)

    q = torch.randn((B, Q_H, S, D), dtype=torch.float16)
    k = torch.randn((B, KV_H, S, D), dtype=torch.float16)
    v = torch.randn((B, KV_H, S, D), dtype=torch.float16)

    torch.npu.synchronize()
    print("init successful!")

    output = func(q, k, v)
    torch.npu.synchronize()

    if not args.no_check:
        ref_output = ref_flash_attn(q, k, v)
        torch.npu.synchronize()
        torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
        print("Test Passed!")
    else:
        print("Reference check skipped.")
