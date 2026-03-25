import argparse
import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

import torch

torch.set_default_device("npu")
torch.manual_seed(0)

tilelang.disable_cache()


class Lock:
    """Intra-core lock connecting two pipes via one or more consecutive signal IDs.

    Pure Python data class — all IR generation goes through @T.macro helpers.
    The producer is the first acquirer.

    When count > 1, sig_id is the base; use lock.at(offset) to get the actual
    signal ID for a specific slot (e.g. double-buffering LEFT=0, RIGHT=1).
    """

    __slots__ = ("sig_id", "producer", "consumer", "count")

    def __init__(self, sig_id, producer, consumer, count=1):
        self.sig_id = sig_id
        self.producer = producer
        self.consumer = consumer
        self.count = count

    def other(self, my_pipe):
        return self.consumer if my_pipe == self.producer else self.producer

    def at(self, offset):
        return self.sig_id + offset


def _resolve(entry):
    """Resolve a lock entry to (Lock, sig_id).

    entry can be:
      - Lock              → (lock, lock.sig_id)       plain lock
      - (Lock, offset)    → (lock, lock.at(offset))   indexed lock
    """
    if isinstance(entry, Lock):
        return entry, entry.sig_id
    lk, off = entry
    return lk, lk.at(off)


# ---------------------------------------------------------------------------
# Low-level @T.macro wrappers (only primitive types: str, int)
# ---------------------------------------------------------------------------


@T.macro(hygienic=False)
def _set_flag(src, dst, sid):
    T.set_flag(src, dst, sid)


@T.macro(hygienic=False)
def _wait_flag(src, dst, sid):
    T.wait_flag(src, dst, sid)


# ---------------------------------------------------------------------------
# High-level Lock helpers — resolve Lock objects then call macros
# ---------------------------------------------------------------------------


def init_lock(lk):
    """Bootstrap an intra-core Lock: pretend consumer already released.
    Initialises all count slots."""
    for i in range(lk.count):
        _set_flag(lk.consumer, lk.producer, lk.at(i))


def acquire(pipe, locks):
    """Acquire intra-core locks.
    Each element is either a Lock or (Lock, offset)."""
    for entry in locks:
        lk, sid = _resolve(entry)
        _wait_flag(lk.other(pipe), pipe, sid)


def release(pipe, locks):
    """Release intra-core locks.
    Each element is either a Lock or (Lock, offset)."""
    for entry in locks:
        lk, sid = _resolve(entry)
        _set_flag(pipe, lk.other(pipe), sid)


def destroy_lock(lk):
    """Consume outstanding init-direction flags so they return to 0."""
    for i in range(lk.count):
        _wait_flag(lk.consumer, lk.producer, lk.at(i))


L0_MAX_SIZE = 64 * 1024  # 64KB
NUM_CORES = 24  # 910B has 24 AI Cores

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads_q,
    heads_kv,
    dim,
    num_stages=14,
    cross_interval=2,
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

    # Total number of output tiles (logical tasks)
    num_seq_blocks = seq_len // block_M
    block_num = num_seq_blocks * heads_q * batch
    num_iters = T.ceildiv(seq_len, block_N)
    num_outer = T.ceildiv(num_iters, num_stages)  # may have a partial tail batch

    # ---------------------------------------------------------------------------
    # Static task distribution: evenly split block_num across NUM_CORES.
    #   - Cores 0 .. r-1       get (q+1) tasks each
    #   - Cores r .. NUM_CORES-1 get  q   tasks each
    # where q = block_num // NUM_CORES, r = block_num % NUM_CORES
    # ---------------------------------------------------------------------------
    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs (each direction gets its own ID)
    SEM_WS1_C2V = 0  # C → V: S[i] ready in ws1
    SEM_WS1_V2C = 1  # V → C: all S consumed from ws1
    SEM_WS2_V2C = 2  # V → C: P[i] ready in ws2
    SEM_WS2_C2V = 3  # C → V: all P consumed from ws2
    SEM_WS3_C2V = 4  # C → V: O[i] ready in ws3
    SEM_WS3_V2C = 5  # V → C: all O consumed from ws3

    # Intra-core locks (C Scope)
    LK_K_L1 = Lock(0, "MTE2", "MTE1")  # K in L1
    LK_P_L1 = Lock(1, "MTE2", "MTE1")  # P in L1
    LK_V_L1 = Lock(2, "MTE2", "MTE1")  # V in L1
    LK_L0AB = Lock(3, "MTE1", "M", count=2)  # L0A/L0B double-buffer (sig 3,4)
    LK_L0C = Lock(5, "M", "FIX", count=2)  # L0C double-buffer    (sig 5,6)

    # Intra-core locks (V Scope)
    LK_IO_UB = Lock(0, "MTE2", "V")  # io_buf (shared for S and O)
    LK_S_HALF = Lock(1, "V", "MTE3")

    def task_range(cid_val):
        """Return (start, count) for core cid_val."""
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
        Q: T.Tensor(shape_q, dtype),  # type: ignore
        K: T.Tensor(shape_kv, dtype),  # type: ignore
        V: T.Tensor(shape_kv, dtype),  # type: ignore
        Output: T.Tensor(shape_q, dtype),  # type: ignore
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype),
    ):
        # Launch exactly NUM_CORES blocks — one per physical AI Core
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

            # Shared L0 buffers — double-buffered (GEMM1 and GEMM2 reuse the same physical L0)
            # dim == block_N == 128, so shapes are compatible across both GEMMs
            l0a = T.alloc_L0A([2, block_M, dim], dtype)  # 2×32KB = 64KB
            l0b = T.alloc_L0B([2, dim, block_N], dtype)  # 2×32KB = 64KB
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)  # 2×64KB = 128KB

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)

            half_M = block_M // 2

            # Per-stage rescale factor buffer (V-pipe only)
            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)  # -scale * row_max

            io_buf = T.alloc_ub([block_M // 2, block_N], dtype)  # reused for S and O (dim == block_N)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            tmp_ub = T.alloc_ub([DataType(accum_dtype).bits // 8 * block_M // 2 * 128], "uint8")
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)

            # Compute this core's task range
            my_start, my_count = task_range(cid)

            # =================================================================
            # Cross-core semaphore protocol (set = +1, wait = -1):
            #
            #   enter loop:  wait  ws you PRODUCE  (consumer finished last batch)
            #   each iter:   wait  ws you CONSUME  (producer wrote this slot)
            #                set   ws you PRODUCE  (this slot ready)
            #   exit loop:   set   ws you CONSUME  (all slots consumed)
            # =================================================================

            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                init_lock(LK_K_L1)
                init_lock(LK_P_L1)
                init_lock(LK_V_L1)
                init_lock(LK_L0AB)
                init_lock(LK_L0C)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)
                    kv_by = by // (heads_q // heads_kv)

                    # Q: GM → L1 (once per task, stays resident)
                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.barrier_all()

                    for k in T.serial(num_outer):
                        _remaining = num_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1 batch: produce S into ws1 (double-buffered L0) ---
                        T.wait_cross_flag(SEM_WS1_V2C)  # V consumed all S
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            acquire("MTE2", [LK_K_L1])
                            T.copy(K[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], k_l1)
                            release("MTE2", [LK_K_L1])

                            acquire("MTE1", [(LK_L0AB, side)])
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            acquire("MTE1", [LK_K_L1])
                            T.copy(k_l1, l0b[side, :, :])
                            release("MTE1", [LK_K_L1, (LK_L0AB, side)])

                            acquire("M", [(LK_L0AB, side), (LK_L0C, side)])
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            release("M", [(LK_L0AB, side), (LK_L0C, side)])

                            acquire("FIX", [(LK_L0C, side)])
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            release("FIX", [(LK_L0C, side)])
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2 batch: consume P from ws2, produce O into ws3 (double-buffered L0) ---
                        T.wait_cross_flag(SEM_WS3_V2C)  # V consumed all O
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            acquire("MTE2", [LK_V_L1])
                            T.copy(V[bz, kv_by, idx * block_N : (idx + 1) * block_N, :], v_l1)
                            release("MTE2", [LK_V_L1])

                            acquire("MTE2", [LK_P_L1])
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            release("MTE2", [LK_P_L1])

                            # MTE1: v first (acquire L0AB), then p (release L0AB)
                            acquire("MTE1", [LK_V_L1, (LK_L0AB, side)])
                            T.copy(v_l1, l0b[side, :, :])
                            release("MTE1", [LK_V_L1])

                            acquire("MTE1", [LK_P_L1])
                            T.copy(p_l1, l0a[side, :, :])
                            release("MTE1", [LK_P_L1, (LK_L0AB, side)])

                            acquire("M", [(LK_L0AB, side), (LK_L0C, side)])
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            release("M", [(LK_L0AB, side), (LK_L0C, side)])

                            acquire("FIX", [(LK_L0C, side)])
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            release("FIX", [(LK_L0C, side)])
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)  # all P consumed

                destroy_lock(LK_K_L1)
                destroy_lock(LK_P_L1)
                destroy_lock(LK_V_L1)
                destroy_lock(LK_L0AB)
                destroy_lock(LK_L0C)

            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                init_lock(LK_IO_UB)
                init_lock(LK_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)

                    # Reset per-task accumulators
                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)

                    for k in T.serial(num_outer):
                        _remaining = num_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- softmax batch ---
                        T.wait_cross_flag(SEM_WS2_C2V)  # C consumed all P
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur

                            acquire("MTE2", [LK_IO_UB])
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            release("MTE2", [LK_IO_UB])

                            acquire("V", [LK_IO_UB])
                            T.copy(io_buf, work_ub)
                            release("V", [LK_IO_UB])

                            T.reduce_max(work_ub, neg_sm[cur, :, :], tmp_ub, dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :], tmp_ub)
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            acquire("V", [LK_S_HALF])
                            T.copy(work_ub, acc_s_half)
                            release("V", [LK_S_HALF])

                            acquire("MTE3", [LK_S_HALF])
                            T.copy(acc_s_half, workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :])
                            release("MTE3", [LK_S_HALF])
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            T.reduce_sum(work_ub, sumexp_is[i, :, :], tmp_ub, dim=-1)

                            # r[i] = -(m_new) - -(m_prev) = m_prev - m_new
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)  # all S consumed

                        # --- O accumulation batch ---
                        for i in T.serial(batch_iters):
                            # Deferred: exp(r_factors[i]) and sumexp update
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            # T.tile.fused_mul_add(sumexp, r_factors[i, :, :], sumexp_is[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            # Rescale acc_o by r[i] and accumulate O[i]
                            T.tile.broadcast(buf_2d, r_factors[i, :, :], tmp_ub)
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            # Load O[i] directly into work_ub (ws3 is fp32, work_ub is fp32)
                            acquire("MTE2", [LK_IO_UB])
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :], io_buf)
                            release("MTE2", [LK_IO_UB])

                            acquire("V", [LK_IO_UB])
                            T.copy(io_buf, work_ub)
                            release("V", [LK_IO_UB])

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)  # all O consumed

                    # === final normalization ===
                    T.tile.broadcast(buf_2d, sumexp, tmp_ub)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    T.copy(acc_o, acc_s_half)
                    T.barrier_all()
                    T.copy(
                        acc_s_half, Output[bz, by, bx * block_M + vid * block_M // 2 : bx * block_M + vid * block_M // 2 + block_M // 2, :]
                    )

                destroy_lock(LK_IO_UB)
                destroy_lock(LK_S_HALF)

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
    parser.add_argument("--cross-interval", type=int, default=2, help="cross-core signal interval")
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
        cross_interval=args.cross_interval,
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
