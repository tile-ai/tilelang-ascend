"""
autocvsync_pipelined.py — KNOWN BROKEN (UB out of bounds at runtime)

This file attempts to use T.Pipelined for the unified C+V flash attention body.
It compiles successfully but crashes at runtime because the compiler ring-buffers
ALL UB scratch buffers by num_stages, exceeding the 256KB UB limit.

Root cause analysis (from compiler source study):
  1. PlanAndUpdateBufferAllocationLocation moves scratch buffer allocs into the
     pipelined loop's block (since their LCA is the loop body).
  2. InjectSoftwarePipeline (inject_pipeline.cc) collects ALL alloc_buffers from
     the pipeline block and calls ComputeBufferVersions() on each. Any buffer
     written and read across different pipeline stages gets num_versions > 1.
  3. Scratch buffers (io_buf, work_ub, buf_2d, acc_s_half, tmp_ub) total ~159KB.
     Even ×2 ring-buffering (318KB) exceeds the 256KB UB limit per vector core.

No compiler annotation exists to opt specific buffers out of ring-buffering.
The only fixes would be:
  (a) Compiler change: add a no_pipeline_version annotation
  (b) Dramatically reduce UB scratch usage (different softmax implementation)
  (c) Use T.serial instead (see autocvsync.py — the working version)

Kept as reference for future compiler improvements.
"""

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
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
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
    num_stages=8,
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

    assert num_iters % num_stages == 0, f"num_iters ({num_iters}) must be divisible by num_stages ({num_stages})"

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    half_M = block_M // 2

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
        # Workspace dims use 2 — compiler auto-inserts num_stages // 2 dimension
        workspace_1: T.Tensor([NUM_CORES, 2, block_M, block_N], dtype),
        workspace_2: T.Tensor([NUM_CORES, 2, block_M, block_N], dtype),
        workspace_3: T.Tensor([NUM_CORES, 2, block_M, dim], dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # --- L1 buffers (cube core) ---
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

            # --- L0 buffers (cube core) ---
            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # --- UB buffers (vector core) ---
            acc_o = T.alloc_ub([half_M, dim], accum_dtype)

            # Dim 0 is 2 (even/odd within one pipeline step);
            # compiler auto-inserts the num_stages // 2 ring-buffer dimension.
            r_factors = T.alloc_ub([2, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([2, half_M, 1], accum_dtype)

            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)

            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            tmp_ub = T.alloc_ub([DataType(accum_dtype).bits // 8 * half_M * 128], "uint8")
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)

            my_start, my_count = task_range(cid)

            # ============================================================
            # Unified C+V body with T.Pipelined:
            #
            # The two-level loop (k over num_outer, p over num_stages//2)
            # is merged into a single pipelined loop over num_iters // 2
            # iterations with num_stages // 2 pipeline stages.
            #
            # Each iteration processes 2 KV tiles (even + odd):
            #   1. C: QK even  → ws1[0]        (cube writes ws1)
            #   2. C: QK odd   → ws1[1]        (cube writes ws1)
            #   3. V: softmax(ws1[0])           (vec reads ws1)
            #   4. V: softmax(ws1[1])           (vec reads ws1)
            #   5. V: P_even   → ws2[0]         (vec writes ws2)
            #   6. V: P_odd    → ws2[1]         (vec writes ws2)
            #   7. C: ws2[0]   → p_l1           (cube reads ws2)
            #   8. C: PV even  → ws3[0]         (cube writes ws3)
            #   9. C: ws2[1]   → p_l1           (cube reads ws2)
            #  10. C: PV odd   → ws3[1]         (cube writes ws3)
            #  11. V: acc_o += ws3[0]            (vec reads ws3)
            #  12. V: acc_o += ws3[1]            (vec reads ws3)
            # ============================================================

            for t in T.serial(my_count):
                task_id = my_start + t
                bx = task_id % num_seq_blocks
                by = (task_id // num_seq_blocks) % heads_q
                bz = task_id // (num_seq_blocks * heads_q)
                kv_by = by // (heads_q // heads_kv)

                # C: load Q tile
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)

                # V: init accumulators
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(neg_sm, 2**30)

                T.barrier_all()

                for k in T.Pipelined(num_iters // 2, num_stages=num_stages // 2):
                    # k ranges over [0, num_iters // 2), each step = 2 KV tiles
                    kv_offset = k * 2

                    # ---- C: QK matmul (even) → ws1[0] ----
                    T.copy(K[bz, kv_by, kv_offset * block_N : (kv_offset + 1) * block_N, :], k_l1)
                    T.copy(q_l1, l0a[0, :, :])
                    T.copy(k_l1, l0b[0, :, :])
                    T.mma(l0a[0, :, :], l0b[0, :, :], l0c[0, :, :], init=True)
                    T.copy(l0c[0, :, :], workspace_1[cid, 0, :, :])

                    # ---- C: QK matmul (odd) → ws1[1] ----
                    T.copy(K[bz, kv_by, (kv_offset + 1) * block_N : (kv_offset + 2) * block_N, :], k_l1)
                    T.copy(q_l1, l0a[1, :, :])
                    T.copy(k_l1, l0b[1, :, :])
                    T.mma(l0a[1, :, :], l0b[1, :, :], l0c[1, :, :], init=True)
                    T.copy(l0c[1, :, :], workspace_1[cid, 1, :, :])

                    # ---- V: softmax(ws1[0]) ----
                    T.copy(workspace_1[cid, 0, vid * half_M : vid * half_M + half_M, :], io_buf)
                    T.copy(io_buf, work_ub)
                    T.copy(workspace_1[cid, 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                    T.reduce_max(work_ub, neg_sm[0, :, :], tmp_ub, dim=-1)
                    T.tile.mul(neg_sm[0, :, :], neg_sm[0, :, :], -sm_scale)
                    T.tile.min(neg_sm[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])
                    T.tile.broadcast(buf_2d, neg_sm[0, :, :], tmp_ub)
                    T.tile.axpy(buf_2d, work_ub, sm_scale)
                    T.tile.exp(work_ub, buf_2d)
                    T.copy(work_ub, acc_s_half)
                    T.copy(acc_s_half, workspace_2[cid, 0, vid * half_M : vid * half_M + half_M, :])
                    T.reduce_sum(work_ub, sumexp_is[0, :, :], tmp_ub, dim=-1)
                    T.tile.sub(r_factors[0, :, :], neg_sm[0, :, :], neg_sm[1, :, :])

                    # ---- V: softmax(ws1[1]) ----
                    T.copy(io_buf, work_ub)
                    T.reduce_max(work_ub, neg_sm[1, :, :], tmp_ub, dim=-1)
                    T.tile.mul(neg_sm[1, :, :], neg_sm[1, :, :], -sm_scale)
                    T.tile.min(neg_sm[1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])
                    T.tile.broadcast(buf_2d, neg_sm[1, :, :], tmp_ub)
                    T.tile.axpy(buf_2d, work_ub, sm_scale)
                    T.tile.exp(work_ub, buf_2d)
                    T.copy(work_ub, acc_s_half)
                    T.copy(acc_s_half, workspace_2[cid, 1, vid * half_M : vid * half_M + half_M, :])
                    T.reduce_sum(work_ub, sumexp_is[1, :, :], tmp_ub, dim=-1)
                    T.tile.sub(r_factors[1, :, :], neg_sm[1, :, :], neg_sm[0, :, :])

                    # ---- C: PV matmul (even): p_l1 × V → ws3[0] ----
                    T.copy(workspace_2[cid, 0, :, :], p_l1)
                    T.copy(V[bz, kv_by, kv_offset * block_N : (kv_offset + 1) * block_N, :], v_l1)
                    T.copy(v_l1, l0b[0, :, :])
                    T.copy(p_l1, l0a[0, :, :])
                    T.mma(l0a[0, :, :], l0b[0, :, :], l0c[0, :, :], init=True)
                    T.copy(l0c[0, :, :], workspace_3[cid, 0, :, :])

                    # ---- C: PV matmul (odd): p_l1 × V → ws3[1] ----
                    T.copy(workspace_2[cid, 1, :, :], p_l1)
                    T.copy(V[bz, kv_by, (kv_offset + 1) * block_N : (kv_offset + 2) * block_N, :], v_l1)
                    T.copy(v_l1, l0b[1, :, :])
                    T.copy(p_l1, l0a[1, :, :])
                    T.mma(l0a[1, :, :], l0b[1, :, :], l0c[1, :, :], init=True)
                    T.copy(l0c[1, :, :], workspace_3[cid, 1, :, :])

                    # ---- V: accumulate ws3[0] into acc_o ----
                    T.tile.exp(r_factors[0, :, :], r_factors[0, :, :])
                    T.tile.mul(sumexp, sumexp, r_factors[0, :, :])
                    T.tile.add(sumexp, sumexp, sumexp_is[0, :, :])
                    T.tile.broadcast(buf_2d, r_factors[0, :, :], tmp_ub)
                    T.tile.mul(acc_o, acc_o, buf_2d)
                    T.copy(workspace_3[cid, 0, vid * half_M : vid * half_M + half_M, :], io_buf)
                    T.copy(io_buf, work_ub)
                    T.copy(workspace_3[cid, 1, vid * half_M : vid * half_M + half_M, :], io_buf)
                    T.tile.add(acc_o, acc_o, work_ub)

                    # ---- V: accumulate ws3[1] into acc_o ----
                    T.tile.exp(r_factors[1, :, :], r_factors[1, :, :])
                    T.tile.mul(sumexp, sumexp, r_factors[1, :, :])
                    T.tile.add(sumexp, sumexp, sumexp_is[1, :, :])
                    T.tile.broadcast(buf_2d, r_factors[1, :, :], tmp_ub)
                    T.tile.mul(acc_o, acc_o, buf_2d)
                    T.copy(io_buf, work_ub)
                    T.tile.add(acc_o, acc_o, work_ub)

                # V: normalize
                T.tile.broadcast(buf_2d, sumexp, tmp_ub)
                T.tile.div(acc_o, acc_o, buf_2d)

                # V: write output
                T.copy(acc_o, acc_s_half)
                T.barrier_all()
                T.copy(acc_s_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + half_M, :])

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
