"""TND Shared-Prefix FlashAttention (Developer mode, direct TND packed tensors).

Workarounds for Developer mode compiler issues:
1. CombineCV sync mismatch: GEMM + L0C→UB transfer outside if/else.
   Only T.copy(GM→L1) and T.tile.fill(L0C mask) are branched.
2. AIV cid mapping bug (threads=2 + MIX_AIC_1_2): ALL block_metadata reads
   and masking on Cube side only (AIC cid is correct). Vector side has
   NO block_metadata read.
3. V-core vid offset bug for 3D output: Output = [q_head, total_q, head_dim]
   so row stride = head_dim. Host does permute(1,0,2).
"""

import argparse
import math

import tilelang
import torch
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

NEG_INF = -(2.0**30)


@tilelang.jit(out_idx=[6], pass_configs=pass_configs)
def tnd_shared_prefix_fa_developer(
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
    threads=2,
):
    assert q_head % kv_head == 0, "GQA constraint: q_head must be divisible by kv_head"
    sm_scale = 1.0 / (head_dim**0.5) if sm_scale is None else sm_scale
    dtype = dtype_str
    accum_dtype = "float32"
    group_size = q_head // kv_head

    total_q = T.symbolic("total_q")
    total_private_kv = T.symbolic("total_private_kv")
    shared_prefix_len_sym = T.symbolic("shared_prefix_len")
    total_q_blocks_sym = T.symbolic("total_q_blocks")

    max_shared_iters = (shared_prefix_len + block_N - 1) // block_N
    max_private_iters = (max_private_kv_len + block_N - 1) // block_N
    total_kv_iters = max_shared_iters + max_private_iters
    block_num = total_q_blocks * q_head

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, q_head, head_dim], dtype),  # type: ignore
        K_shared: T.Tensor([shared_prefix_len_sym, kv_head, head_dim], dtype),  # type: ignore
        V_shared: T.Tensor([shared_prefix_len_sym, kv_head, head_dim], dtype),  # type: ignore
        K_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        V_private: T.Tensor([total_private_kv, kv_head, head_dim], dtype),  # type: ignore
        block_metadata: T.Tensor([total_q_blocks_sym, 4], "int32"),  # type: ignore
        Output: T.Tensor([q_head, total_q, head_dim], dtype),  # type: ignore
    ):
        with T.Kernel(block_num, threads=threads, is_npu=True) as (cid):
            tile_id = cid // q_head
            h_q = cid % q_head
            h_kv = h_q // group_size

            q_packed_start = block_metadata[tile_id, 0]
            q_valid = block_metadata[tile_id, 1]
            private_kv_start = block_metadata[tile_id, 2]
            private_kv_len = block_metadata[tile_id, 3]

            q_l1 = T.alloc_shared([block_M, head_dim], dtype)
            k_l1 = T.alloc_shared([block_N, head_dim], dtype)
            v_l1 = T.alloc_shared([block_N, head_dim], dtype)
            acc_s_l1 = T.alloc_shared([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_fragment([block_M, head_dim], accum_dtype)

            acc_o = T.alloc_shared([block_M, head_dim], accum_dtype)
            sumexp = T.alloc_shared([block_M], accum_dtype)
            m_i = T.alloc_shared([block_M], accum_dtype)

            acc_s_ub = T.alloc_shared([block_M, block_N], accum_dtype)
            m_i_prev = T.alloc_shared([block_M], accum_dtype)
            acc_s_ub_ = T.alloc_shared([block_M, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_shared([block_M], accum_dtype)
            acc_s_half = T.alloc_shared([block_M, block_N], dtype)
            acc_o_ub = T.alloc_shared([block_M, head_dim], accum_dtype)
            acc_o_half = T.alloc_shared([block_M, head_dim], dtype)

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, NEG_INF)

            T.copy(Q[q_packed_start : q_packed_start + block_M, h_q, :], q_l1)

            for k in T.serial(total_kv_iters):
                kv_start_shared = k * block_N
                priv_k = k - max_shared_iters
                kv_start_priv = priv_k * block_N
                private_offset = private_kv_start + kv_start_priv

                if k < max_shared_iters:
                    if kv_start_shared < shared_prefix_len:
                        T.copy(
                            K_shared[
                                kv_start_shared : kv_start_shared + block_N,
                                h_kv,
                                :,
                            ],
                            k_l1,
                        )
                else:
                    if kv_start_priv < private_kv_len:
                        T.copy(
                            K_private[
                                private_offset : private_offset + block_N,
                                h_kv,
                                :,
                            ],
                            k_l1,
                        )

                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)

                T.copy(acc_s_l0c, acc_s_ub_)

                T.tile.fill(acc_s_ub, 0.0)
                T.copy(m_i, m_i_prev)
                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)

                if k < max_shared_iters:
                    if kv_start_shared >= shared_prefix_len:
                        T.tile.fill(acc_s_ub, NEG_INF)
                    else:
                        kv_valid = shared_prefix_len - kv_start_shared
                        if kv_valid < block_N:
                            for row in T.serial(block_M):
                                for col in T.serial(block_N):
                                    if col >= kv_valid:
                                        acc_s_ub[row, col] = NEG_INF
                        if causal_mask:  # noqa: SIM102
                            if q_packed_start < shared_prefix_len:  # noqa: SIM102
                                if kv_start_shared + block_N > q_packed_start:
                                    for row in T.serial(block_M):
                                        for col in T.serial(block_N):
                                            q_pos = q_packed_start + row
                                            kv_pos = kv_start_shared + col
                                            if kv_pos > q_pos and col < kv_valid:
                                                acc_s_ub[row, col] = NEG_INF
                else:
                    if kv_start_priv >= private_kv_len:
                        T.tile.fill(acc_s_ub, NEG_INF)
                    else:
                        kv_valid = private_kv_len - kv_start_priv
                        if kv_valid < block_N:
                            for row in T.serial(block_M):
                                for col in T.serial(block_N):
                                    if col >= kv_valid:
                                        acc_s_ub[row, col] = NEG_INF
                        if causal_mask:  # noqa: SIM102
                            if shared_prefix_len + kv_start_priv + block_N > q_packed_start:
                                for row in T.serial(block_M):
                                    for col in T.serial(block_N):
                                        q_pos = q_packed_start + row
                                        kv_pos = shared_prefix_len + kv_start_priv + col
                                        if kv_pos > q_pos and col < kv_valid:
                                            acc_s_ub[row, col] = NEG_INF

                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                T.reduce_max(acc_s_ub, m_i, dim=-1)
                T.tile.max(m_i, m_i, m_i_prev)
                T.tile.sub(m_i_prev, m_i_prev, m_i)
                T.tile.exp(m_i_prev, m_i_prev)

                for h_i in range(block_M):
                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                T.tile.exp(acc_s_ub, acc_s_ub)
                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                T.tile.mul(sumexp, sumexp, m_i_prev)
                T.tile.add(sumexp, sumexp, sumexp_i_ub)

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, acc_s_l1)

                if k < max_shared_iters:
                    if kv_start_shared < shared_prefix_len:
                        T.copy(
                            V_shared[
                                kv_start_shared : kv_start_shared + block_N,
                                h_kv,
                                :,
                            ],
                            v_l1,
                        )
                else:
                    if kv_start_priv < private_kv_len:
                        T.copy(
                            V_private[
                                private_offset : private_offset + block_N,
                                h_kv,
                                :,
                            ],
                            v_l1,
                        )

                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, acc_o_ub)

                for h_i in range(block_M):
                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                T.tile.add(acc_o, acc_o, acc_o_ub)

            for h_i in range(block_M):
                T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

            T.copy(acc_o, acc_o_half)
            if threads == 1:
                T.copy(
                    acc_o_half,
                    Output[
                        h_q,
                        q_packed_start : q_packed_start + q_valid,
                        :,
                    ],
                )
            else:
                T.copy(
                    acc_o_half,
                    Output[
                        h_q,
                        q_packed_start : q_packed_start + block_M,
                        :,
                    ],
                )

    return main


# ========== block_metadata construction (design.md §4.6) ==========
def build_block_metadata(shared_prefix_len, private_q_lens, block_M, device):
    metadata_list = []
    if shared_prefix_len > 0:
        for i in range(math.ceil(shared_prefix_len / block_M)):
            q_start = i * block_M
            q_valid = min(block_M, shared_prefix_len - q_start)
            metadata_list.append([q_start, q_valid, 0, 0])
    priv_offset = 0
    for _b, priv_len in enumerate(private_q_lens):
        if priv_len == 0:
            continue
        q_packed_offset = shared_prefix_len + priv_offset
        for i in range(math.ceil(priv_len / block_M)):
            q_start = q_packed_offset + i * block_M
            q_valid = min(block_M, priv_len - i * block_M)
            metadata_list.append([q_start, q_valid, priv_offset, priv_len])
        priv_offset += priv_len
    return torch.tensor(metadata_list, dtype=torch.int32, device=device)


# ========== Golden reference (CPU computation) ==========
def ref_tnd_shared_prefix_fa(
    Q,
    K_shared,
    V_shared,
    K_private,
    V_private,
    shared_prefix_len,
    private_q_lens,
    q_head,
    kv_head,
    head_dim,
    sm_scale=None,
    causal_mask=False,
):
    sm_scale = 1.0 / (head_dim**0.5) if sm_scale is None else sm_scale
    group_size = q_head // kv_head
    total_q = Q.shape[0]
    dtype = Q.dtype
    Q = Q.float()
    K_shared = K_shared.float()
    V_shared = V_shared.float()
    K_private = K_private.float()
    V_private = V_private.float()

    O = torch.zeros((total_q, q_head, head_dim), dtype=torch.float32)

    if shared_prefix_len > 0:
        q_seg = Q[:shared_prefix_len]
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]
            k = K_shared[:, h_kv, :]
            v = V_shared[:, h_kv, :]
            scores = torch.matmul(q, k.T) * sm_scale
            if causal_mask:
                mask = torch.triu(torch.ones(shared_prefix_len, shared_prefix_len), diagonal=1).bool()
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            O[:shared_prefix_len, h_q, :] = torch.matmul(attn, v)

    priv_offset = 0
    for _b, priv_len in enumerate(private_q_lens):
        if priv_len == 0:
            continue
        q_start = shared_prefix_len + priv_offset
        q_seg = Q[q_start : q_start + priv_len]
        k_priv = K_private[priv_offset : priv_offset + priv_len]
        v_priv = V_private[priv_offset : priv_offset + priv_len]
        for h_q in range(q_head):
            h_kv = h_q // group_size
            q = q_seg[:, h_q, :]
            if shared_prefix_len > 0:
                k = torch.cat([K_shared[:, h_kv, :], k_priv[:, h_kv, :]], dim=0)
                v = torch.cat([V_shared[:, h_kv, :], v_priv[:, h_kv, :]], dim=0)
            else:
                k = k_priv[:, h_kv, :]
                v = v_priv[:, h_kv, :]
            scores = torch.matmul(q, k.T) * sm_scale
            if causal_mask:
                total_kv = shared_prefix_len + priv_len
                q_pos = torch.arange(q_start, q_start + priv_len).unsqueeze(1)
                kv_pos = torch.arange(total_kv).unsqueeze(0)
                mask = kv_pos > q_pos
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            O[q_start : q_start + priv_len, h_q, :] = torch.matmul(attn, v)
        priv_offset += priv_len

    return O.to(dtype)


# ========== Precision standard ==========
def get_precision(dtype):
    fp_table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1e0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    m = torch.isfinite(g)
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== Test runner ==========
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
    threads=2,
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

        kernel = tnd_shared_prefix_fa_developer(
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
            threads=threads,
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

        passed, ratio, max_abs = check_precision(output, ref, dtype_str)
        status = "PASS" if passed else "FAIL"
        print(
            f"[PRECISION_{status}] {tag} "
            f"batch={batch} q_head={q_head} kv_head={kv_head} "
            f"sp_len={shared_prefix_len} priv_lens={private_q_lens} "
            f"dtype={dtype_str} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}"
        )
        return passed
    except Exception as e:
        print(f"[PRECISION_FAIL] {tag}: {e}")
        import traceback

        traceback.print_exc()
        return False


# ========== L0 tests (design.md §9.2) ==========
def test_l0():
    ok = True

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
        threads=1,
    )
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
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0", choices=["l0", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.manual_seed(0)

    ok = True
    if args.level in ("l0", "all"):
        ok &= test_l0()

    if ok:
        print("Test Passed!")
        exit(0)
    exit(1)
