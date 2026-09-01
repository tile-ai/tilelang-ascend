"""TND Shared-Prefix FlashAttention (Expert mode, persistent, 6-flag, batch sync).

Based on flash_attn_bhsd_expert_h16_d128.py reference implementation.
Adapted for: TND packed layout, shared+private KV segments, block_metadata,
GQA, variable-length mask.

Key architecture:
- Persistent scheduling: NUM_CORES blocks, each processes multiple tasks
- 6 cross-core flags (3 pairs C2V/V2C) with init + destroy
- Batch sync: num_stages tiles per batch, cross_interval for flag batching
- Double-buffer L0A/L0B/L0C
- T.mma (not T.gemm_v0) for Cube GEMM
"""

import argparse
import math
import os
import sys

import tilelang
import torch
from tilelang import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tnd_shared_prefix_fa_developer import (
    build_block_metadata,
    ref_tnd_shared_prefix_fa,
)

NEG_INF = -(2.0**30)
NUM_CORES = 24

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

# Cross-core semaphore IDs (6 flags, 3 pairs)
SEM_S_C2V = 0  # Cube → Vector: S (score) data ready
SEM_S_V2C = 1  # Vector → Cube: S workspace released
SEM_P_V2C = 2  # Vector → Cube: P (softmax) data ready
SEM_P_C2V = 3  # Cube → Vector: P workspace released
SEM_O_C2V = 4  # Cube → Vector: O (output) data ready
SEM_O_V2C = 5  # Vector → Cube: O workspace released

# Intra-core signal IDs (C Scope)
SIG_K_L1 = 0
SIG_P_L1 = 1
SIG_V_L1 = 2
SIG_L0AB = 3
SIG_L0C = 5

# Intra-core signal IDs (V Scope)
SIG_IO_UB = 0
SIG_S_HALF = 1


@tilelang.jit(out_idx=[6], workspace_idx=[7, 8, 9], pass_configs=pass_configs)
def tnd_shared_prefix_fa_expert(
    q_head,
    kv_head,
    head_dim,
    shared_prefix_len,
    max_private_kv_len,
    total_q,
    total_private_kv,
    total_q_blocks,
    block_M=128,
    block_N=64,
    sm_scale=None,
    dtype_str="float16",
    causal_mask=False,
    num_stages=2,
    cross_interval=2,
):
    assert q_head % kv_head == 0, "GQA constraint: q_head must be divisible by kv_head"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"
    sm_scale = 1.0 / (head_dim**0.5) if sm_scale is None else sm_scale
    dtype = dtype_str
    accum_dtype = "float32"
    group_size = q_head // kv_head
    half_M = block_M // 2

    max_shared_iters = (shared_prefix_len + block_N - 1) // block_N
    max_private_iters = (max_private_kv_len + block_N - 1) // block_N
    total_iters = max_shared_iters + max_private_iters
    num_outer = T.ceildiv(total_iters, num_stages)
    block_num = total_q_blocks * q_head

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, q_head, head_dim], dtype),  # type: ignore
        K_shared: T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),  # type: ignore
        V_shared: T.Tensor([shared_prefix_len, kv_head, head_dim], dtype),  # type: ignore
        K_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        V_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        block_metadata: T.Tensor([total_q_blocks, 4], "int32"),  # type: ignore
        Output: T.Tensor([q_head, total_q, head_dim], dtype),  # type: ignore
        workspace_s: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_p: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_o: T.Tensor([NUM_CORES, num_stages, block_M, head_dim], dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            q_l1 = T.alloc_L1([block_M, head_dim], dtype)
            k_l1 = T.alloc_L1([block_N, head_dim], dtype)
            v_l1 = T.alloc_L1([block_N, head_dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            l0a = T.alloc_L0A([2, block_M, head_dim], dtype)
            l0b = T.alloc_L0B([2, head_dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([2, block_M, head_dim], accum_dtype)

            acc_o = T.alloc_ub([half_M, head_dim], accum_dtype)
            sumexp = T.alloc_ub([half_M, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, half_M, 1], accum_dtype)

            r_factors = T.alloc_ub([num_stages, half_M, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, half_M, 1], accum_dtype)

            io_buf = T.alloc_ub([half_M, block_N], dtype)
            acc_s_half = T.alloc_ub([half_M, block_N], dtype)
            work_ub = T.alloc_ub([half_M, block_N], accum_dtype)
            buf_2d = T.alloc_ub([half_M, block_N], accum_dtype)
            acc_o_half = T.alloc_ub([half_M, head_dim], dtype)

            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_P_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    tile_id = task_id // q_head
                    h_q = task_id % q_head
                    h_kv = h_q // group_size

                    q_packed_start = block_metadata[tile_id, 0]
                    private_kv_start = block_metadata[tile_id, 2]
                    private_kv_len = block_metadata[tile_id, 3]

                    T.copy(Q[q_packed_start : q_packed_start + block_M, h_q, :], q_l1)
                    T.barrier_all()

                    for ko in T.serial(num_outer):
                        _remaining = total_iters - ko * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: produce S into ws_s ---
                        T.wait_cross_flag(SEM_S_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = ko * num_stages + i

                            if idx < max_shared_iters:
                                kv_start = idx * block_N
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K_shared[kv_start : kv_start + block_N, h_kv, :], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)
                            else:
                                priv_k = idx - max_shared_iters
                                kv_start = priv_k * block_N
                                private_offset = private_kv_start + kv_start
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                if kv_start < private_kv_len:
                                    T.copy(
                                        K_private[private_offset : private_offset + block_N, h_kv, :],
                                        k_l1,
                                    )
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_s[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_S_C2V)

                        # --- GEMM2 batch: consume P from ws_p, produce O into ws_o ---
                        T.wait_cross_flag(SEM_O_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = ko * num_stages + i

                            if idx < max_shared_iters:
                                kv_start = idx * block_N
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V_shared[kv_start : kv_start + block_N, h_kv, :], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)
                            else:
                                priv_k = idx - max_shared_iters
                                kv_start = priv_k * block_N
                                private_offset = private_kv_start + kv_start
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                if kv_start < private_kv_len:
                                    T.copy(
                                        V_private[private_offset : private_offset + block_N, h_kv, :],
                                        v_l1,
                                    )
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_P_V2C)
                            T.copy(workspace_p[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], acc_o_l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(acc_o_l0c[side, :, :], workspace_o[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_O_C2V)

                        T.set_cross_flag("MTE2", SEM_P_C2V)

                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_S_V2C)
                T.set_cross_flag("MTE2", SEM_O_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    tile_id = task_id // q_head
                    h_q = task_id % q_head

                    q_packed_start = block_metadata[tile_id, 0]
                    q_valid = block_metadata[tile_id, 1]
                    private_kv_len = block_metadata[tile_id, 3]

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    for ko in T.serial(num_outer):
                        _remaining = total_iters - ko * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- softmax batch ---
                        T.wait_cross_flag(SEM_P_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = ko * num_stages + i

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_S_C2V)
                            T.copy(workspace_s[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            # Apply mask: KV tail cols + empty iters + causal
                            T.tile.mul(work_ub, work_ub, 1.0)
                            if idx < max_shared_iters:
                                kv_start_s = idx * block_N
                                if kv_start_s >= shared_prefix_len:
                                    T.tile.fill(work_ub, NEG_INF)
                                else:
                                    kv_valid_s = shared_prefix_len - kv_start_s
                                    if kv_valid_s < block_N:
                                        for row in T.serial(half_M):
                                            for col in T.serial(block_N):
                                                if col >= kv_valid_s:
                                                    work_ub[row, col] = NEG_INF
                                if causal_mask:  # noqa: SIM102
                                    if q_packed_start < shared_prefix_len:  # noqa: SIM102
                                        if kv_start_s + block_N > q_packed_start:
                                            for row in T.serial(half_M):
                                                q_pos = q_packed_start + vid * half_M + row
                                                for col in T.serial(block_N):
                                                    kv_pos = kv_start_s + col
                                                    if kv_pos > q_pos:
                                                        work_ub[row, col] = NEG_INF
                            else:
                                priv_k = idx - max_shared_iters
                                kv_start_p = priv_k * block_N
                                if kv_start_p >= private_kv_len:
                                    T.tile.fill(work_ub, NEG_INF)
                                else:
                                    kv_valid_p = private_kv_len - kv_start_p
                                    if kv_valid_p < block_N:
                                        for row in T.serial(half_M):
                                            for col in T.serial(block_N):
                                                if col >= kv_valid_p:
                                                    work_ub[row, col] = NEG_INF
                                if causal_mask:  # noqa: SIM102
                                    if shared_prefix_len + kv_start_p + block_N > q_packed_start:
                                        for row in T.serial(half_M):
                                            q_pos = q_packed_start + vid * half_M + row
                                            for col in T.serial(block_N):
                                                kv_pos = shared_prefix_len + kv_start_p + col
                                                if kv_pos > q_pos:
                                                    work_ub[row, col] = NEG_INF

                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(
                                acc_s_half,
                                workspace_p[cid, i, vid * half_M : vid * half_M + half_M, :],
                            )
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_P_V2C)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_S_V2C)

                        # --- O accumulation batch ---
                        for i in T.serial(batch_iters):
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_O_C2V)
                            T.copy(workspace_o[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_O_V2C)

                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    T.copy(acc_o, acc_o_half)
                    T.barrier_all()
                    for row in T.serial(half_M):
                        q_row = vid * half_M + row
                        if q_row < q_valid:
                            T.copy(
                                acc_o_half[row, :],
                                Output[
                                    h_q,
                                    q_packed_start + q_row,
                                    :,
                                ],
                            )

                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


def run_test_case(
    batch,
    q_head,
    kv_head,
    head_dim,
    shared_prefix_len,
    private_q_lens,
    block_M,
    block_N,
    dtype_str,
    tag,
    causal_mask=False,
):
    try:
        total_q = shared_prefix_len + sum(private_q_lens)
        total_private_kv = sum(private_q_lens)
        max_private_kv_len = max(private_q_lens) if private_q_lens else 0
        total_q_blocks = math.ceil(shared_prefix_len / block_M) + sum(math.ceil(l / block_M) for l in private_q_lens)

        torch.manual_seed(0)
        npu_dtype = getattr(torch, dtype_str)

        Q_cpu = torch.randn(total_q, q_head, head_dim, dtype=npu_dtype)
        K_shared_cpu = (
            torch.randn(shared_prefix_len, kv_head, head_dim, dtype=npu_dtype)
            if shared_prefix_len > 0
            else torch.zeros(0, kv_head, head_dim, dtype=npu_dtype)
        )
        V_shared_cpu = (
            torch.randn(shared_prefix_len, kv_head, head_dim, dtype=npu_dtype)
            if shared_prefix_len > 0
            else torch.zeros(0, kv_head, head_dim, dtype=npu_dtype)
        )
        K_private_cpu = torch.randn(total_private_kv, kv_head, head_dim, dtype=npu_dtype)
        V_private_cpu = torch.randn(total_private_kv, kv_head, head_dim, dtype=npu_dtype)
        block_metadata = build_block_metadata(shared_prefix_len, private_q_lens, block_M, "cpu")

        sm_scale = 1.0 / (head_dim**0.5)

        kernel = tnd_shared_prefix_fa_expert(
            q_head=q_head,
            kv_head=kv_head,
            head_dim=head_dim,
            shared_prefix_len=shared_prefix_len,
            max_private_kv_len=max_private_kv_len,
            total_q=total_q,
            total_private_kv=total_private_kv,
            total_q_blocks=total_q_blocks,
            block_M=block_M,
            block_N=block_N,
            sm_scale=sm_scale,
            dtype_str=dtype_str,
            causal_mask=causal_mask,
        )

        Q = Q_cpu.npu()
        KS = K_shared_cpu.npu()
        VS = V_shared_cpu.npu()
        KP = K_private_cpu.npu()
        VP = V_private_cpu.npu()
        bm = block_metadata.npu()

        output = kernel(Q, KS, VS, KP, VP, bm)
        torch.npu.synchronize()
        output = output.permute(1, 0, 2)

        ref = ref_tnd_shared_prefix_fa(
            Q_cpu,
            K_shared_cpu,
            V_shared_cpu,
            K_private_cpu,
            V_private_cpu,
            shared_prefix_len,
            private_q_lens,
            q_head,
            kv_head,
            head_dim,
            sm_scale=sm_scale,
            causal_mask=causal_mask,
        )

        torch.testing.assert_close(output.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
        print(f"[PASS] {tag} batch={batch} q_head={q_head} kv_head={kv_head} sp_len={shared_prefix_len} dtype={dtype_str}")
        return True
    except Exception as e:
        print(f"[FAIL] {tag}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True

    ok &= run_test_case(
        batch=2,
        q_head=4,
        kv_head=2,
        head_dim=64,
        shared_prefix_len=128,
        private_q_lens=[128, 128],
        block_M=128,
        block_N=64,
        dtype_str="float16",
        tag="l0_aligned",
    )

    if ok:
        ok &= run_test_case(
            batch=10,
            q_head=14,
            kv_head=2,
            head_dim=64,
            shared_prefix_len=24,
            private_q_lens=[150] * 10,
            block_M=128,
            block_N=64,
            dtype_str="float16",
            tag="l0_business",
        )

    if ok:
        ok &= run_test_case(
            batch=10,
            q_head=14,
            kv_head=2,
            head_dim=64,
            shared_prefix_len=24,
            private_q_lens=[218] * 10,
            block_M=128,
            block_N=64,
            dtype_str="float16",
            tag="l0_p99",
        )

    if ok:
        ok &= run_test_case(
            batch=10,
            q_head=14,
            kv_head=2,
            head_dim=64,
            shared_prefix_len=24,
            private_q_lens=[150] * 10,
            block_M=128,
            block_N=64,
            dtype_str="bfloat16",
            tag="l0_bf16",
        )

    if ok:
        ok &= run_test_case(
            batch=2,
            q_head=4,
            kv_head=2,
            head_dim=64,
            shared_prefix_len=64,
            private_q_lens=[128, 128],
            block_M=128,
            block_N=64,
            dtype_str="float16",
            tag="l0_causal",
            causal_mask=True,
        )

    if ok:
        print("Test Passed!")
        exit(0)
    exit(1)
