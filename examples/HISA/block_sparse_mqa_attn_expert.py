"""
Block Sparse MQA Attention Kernel (Expert Mode)

Manual CV scope, manual sync, explicit L0A/L0B/MMA.
Four auto passes disabled: no auto CV combine, no auto sync,
no memory planning, no auto CV sync.

Follows flash_attn_bhsd_expert_h16_d128.py flag conventions:
  - MTE1↔MTE2 for L1 buffer management
  - M↔MTE1 for L0A/L0B management
  - M↔FIX for L0C management
  - V↔MTE2 for UB buffer management
  - V↔MTE3 for output buffer management
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    },
    out_idx=[3],
    workspace_idx=[-1],
    target="pto",
)
def block_sparse_mqa_attn_return_logits(
    seq_len: int,
    seq_len_kv: int,
    kv_block_size: int,
    topk: int,
    heads: int,
    index_dim: int,
    block_N: int = 8,
    num_stages: int = 2,
    threads: int = 2,
    num_pairs: int = 20,
):
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    num_tokens_per_kernel = 2 * num_pairs
    grid_size = T.ceildiv(seq_len, num_tokens_per_kernel)

    index_q_shape = [seq_len, heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, topk, kv_block_size]

    H_per_block = heads
    assert kv_block_size % block_N == 0, "block_N must divide kv_block_size"

    # ---------- Signal IDs ----------
    # C scope
    SIG_Q_L1 = 0    # Q L1 buffer: MTE1↔MTE2
    SIG_K_L1 = 1    # K L1 buffer: MTE1↔MTE2
    SIG_L0AB = 2    # L0A/L0B: M↔MTE1
    SIG_L0C = 3     # L0C: FIX↔M
    # V scope
    SIG_S_UB = 4    # s_ub: V↔MTE2
    SIG_W_UB = 5    # weights_ub: V↔MTE2
    SIG_LOGITS = 6  # logits: V↔MTE3
    SIG_VS = 7      # V↔S: scalar/vector sync for mask
    # Cross-scope
    CROSS_FLAG_C2V = 0

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        workspace_1: T.Tensor([seq_len, topk, H_per_block, kv_block_size], accum_dtype),
    ):
        with T.Kernel(grid_size, is_npu=True) as (bx, by):
            # ---- V scope: UB allocations ----
            s_ub = T.alloc_ub([H_per_block, kv_block_size], accum_dtype)
            logits = T.alloc_ub([1, kv_block_size], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)

            # ---- C scope: L1 / L0A / L0B / L0C allocations ----
            q_l1 = T.alloc_L1([H_per_block, index_dim], dtype)
            k_l1 = T.alloc_L1([kv_block_size, index_dim], dtype)
            l0a = T.alloc_L0A([H_per_block, index_dim], dtype)
            l0b = T.alloc_L0B([index_dim, kv_block_size], dtype)
            l0c = T.alloc_L0C([H_per_block, kv_block_size], accum_dtype)

            # ================================================================
            # C scope: DMA → L1 → L0A/L0B → MMA → L0C → workspace
            # ================================================================
            with T.Scope("C"):
                # Init: all buffers start free
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)   # Q L1 free
                T.set_flag("MTE1", "MTE2", SIG_K_L1)   # K L1 free
                T.set_flag("M", "MTE1", SIG_L0AB)       # L0A/L0B free
                T.set_flag("FIX", "M", SIG_L0C)         # L0C free

                for pair_i in T.serial(num_pairs):
                    t_a = pair_i * 2
                    token_a = bx * num_tokens_per_kernel + t_a
                    if token_a < seq_len:
                        for n_i in T.serial(topk):
                            # -- DMA K → L1 --
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_a, n_i]
                                    * kv_block_size : TopKBlockIndex[token_a, n_i]
                                    * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            # -- Stage to L0A/L0B --
                            # On first iteration: DMA Q→L1 and stage Q→L0A too.
                            # On subsequent iterations: Q already in L0A, only stage K→L0B.
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB)
                            if n_i == 0:
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(
                                    IndexQ[token_a, :, :],
                                    q_l1,
                                )
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.copy(q_l1, l0a)
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(k_l1, l0b, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB)

                            # -- MMA: S = Q @ K^T --
                            T.wait_flag("MTE1", "M", SIG_L0AB)
                            T.wait_flag("FIX", "M", SIG_L0C)
                            T.mma(l0a, l0b, l0c, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB)
                            T.set_flag("M", "FIX", SIG_L0C)

                            # -- Copy L0C → workspace --
                            T.wait_flag("M", "FIX", SIG_L0C)
                            T.copy(l0c, workspace_1[token_a, n_i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C)

                    t_b = pair_i * 2 + 1
                    token_b = bx * num_tokens_per_kernel + t_b
                    if token_b < seq_len:
                        for n_i in T.serial(topk):
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(
                                IndexK[
                                    TopKBlockIndex[token_b, n_i]
                                    * kv_block_size : TopKBlockIndex[token_b, n_i]
                                    * kv_block_size
                                    + kv_block_size,
                                    :,
                                ],
                                k_l1,
                            )
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB)
                            if n_i == 0:
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(
                                    IndexQ[token_b, :, :],
                                    q_l1,
                                )
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.copy(q_l1, l0a)
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                            T.copy(k_l1, l0b, transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB)

                            T.wait_flag("MTE1", "M", SIG_L0AB)
                            T.wait_flag("FIX", "M", SIG_L0C)
                            T.mma(l0a, l0b, l0c, init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB)
                            T.set_flag("M", "FIX", SIG_L0C)

                            T.wait_flag("M", "FIX", SIG_L0C)
                            T.copy(l0c, workspace_1[token_b, n_i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("FIX", "M", SIG_L0C)

                # Signal V scope
                T.set_cross_flag("FIX", CROSS_FLAG_C2V)

            # ================================================================
            # V scope: workspace → s_ub → ReLU → mul weights → reduce → Logits
            # ================================================================
            with T.Scope("V"):
                # Init: UB buffers start free
                T.set_flag("V", "MTE2", SIG_S_UB)
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS)
                T.set_flag("S", "V", SIG_VS)       # scalar pipe done (init)

                T.wait_cross_flag(CROSS_FLAG_C2V, "MTE2")

                for pair_i in T.serial(num_pairs):
                    t_a = pair_i * 2
                    for n_i in T.serial(topk):
                        token_idx = bx * num_tokens_per_kernel + t_a + by
                        if token_idx < seq_len:
                            # -- DMA workspace → s_ub --
                            T.wait_flag("V", "MTE2", SIG_S_UB)
                            T.copy(
                                workspace_1[token_idx, n_i, :, :],
                                s_ub,
                            )
                            T.set_flag("MTE2", "V", SIG_S_UB)

                            # -- DMA weights → weights_ub --
                            T.wait_flag("V", "MTE2", SIG_W_UB)
                            T.copy(
                                Weights[token_idx, :],
                                weights_ub,
                            )
                            T.set_flag("MTE2", "V", SIG_W_UB)

                            # -- Wait for both DMAs, then vector ops --
                            T.wait_flag("MTE2", "V", SIG_S_UB)
                            T.wait_flag("MTE2", "V", SIG_W_UB)

                            T.tile.relu(s_ub, s_ub)
                            # Convert weights half→float (reads weights_ub, writes weights)
                            T.copy(weights_ub, weights)
                            # weights_ub no longer needed
                            T.set_flag("V", "MTE2", SIG_W_UB)

                            for h_i in T.serial(heads):
                                T.tile.mul(
                                    s_ub[h_i, :],
                                    s_ub[h_i, :],
                                    weights[h_i],
                                )

                            # -- Reduce sum → logits --
                            T.wait_flag("MTE3", "V", SIG_LOGITS)
                            T.reduce_sum(s_ub, logits, dim=0, clear=True)
                            # s_ub consumed, release for next iteration
                            T.set_flag("V", "MTE2", SIG_S_UB)

                            # V→S: signal scalar pipe that reduce_sum is done, logits ready for mask
                            T.set_flag("V", "S", SIG_VS)

                            # -- Mask: set out-of-range positions to -inf --
                            T.wait_flag("V", "S", SIG_VS)   # S waits for V: ensure reduce_sum done
                            cu_k_s_min = CuSeqLenKS[token_idx]
                            cu_k_e_max = CuSeqLenKE[token_idx]
                            for i_i in T.serial(kv_block_size):
                                k_i = TopKBlockIndex[token_idx, n_i] * kv_block_size + i_i
                                if k_i < cu_k_s_min or k_i >= cu_k_e_max:
                                    logits[0, i_i] = -T.infinity(accum_dtype)
                            T.set_flag("S", "V", SIG_VS)   # S→V: mask done, logits finalized

                            # V waits for S: ensure mask complete before signaling MTE3
                            T.wait_flag("S", "V", SIG_VS)
                            T.set_flag("V", "MTE3", SIG_LOGITS)

                            # -- DMA logits → output --
                            T.wait_flag("V", "MTE3", SIG_LOGITS)
                            T.copy(
                                logits,
                                Logits[token_idx, n_i, :],
                            )
                            T.set_flag("MTE3", "V", SIG_LOGITS)

                # Destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_S_UB)
                T.wait_flag("V", "MTE2", SIG_W_UB)
                T.wait_flag("MTE3", "V", SIG_LOGITS)
                T.wait_flag("S", "V", SIG_VS)

    return kernel


def ref_block_sparse_mqa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_block_indices: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of sparse MQA attention using torch_npu vectorization."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = k.shape[0]
    topk = topk_block_indices.shape[1]

    q = q.float()
    k = k.float()
    weights = weights.float()

    block_indices = topk_block_indices.unsqueeze(-1) * kv_block_size + torch.arange(kv_block_size, device=q.device)
    block_indices = block_indices.long()

    k_gathered = k[block_indices.view(-1)]
    k_gathered = k_gathered.view(seq_len, topk, kv_block_size, index_dim)

    scores = torch.einsum("qhd,qkbd->qkbh", q, k_gathered)

    weights_expanded = weights.unsqueeze(1).unsqueeze(2)
    scores = scores.relu() * weights_expanded

    logits = scores.sum(dim=-1)

    pos_out_of_bounds = block_indices >= seq_len_kv
    cu_seqlen_ks_exp = cu_seqlen_ks.unsqueeze(1).unsqueeze(2)
    cu_seqlen_ke_exp = cu_seqlen_ke.unsqueeze(1).unsqueeze(2)
    pos_invalid = (block_indices < cu_seqlen_ks_exp) | (block_indices >= cu_seqlen_ke_exp)
    invalid_mask = pos_out_of_bounds | pos_invalid
    logits = logits.masked_fill(invalid_mask, float("-inf"))

    return logits


def test_block_sparse_mqa_attn(
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    index_dim: int,
    kv_block_size: int,
    topk: int,
    dtype: str = "float16",
    num_pairs: int = 64,
):
    """Test sparse MQA attention kernel with golden validation."""
    kernel = block_sparse_mqa_attn_return_logits(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        num_pairs=num_pairs,
    )
    print(kernel.get_kernel_source())

    q = torch.rand((seq_len, heads, index_dim), dtype=torch.float16, device="npu")
    k = torch.rand((seq_len_kv, index_dim), dtype=torch.float16, device="npu")
    weights = torch.randn((seq_len, heads), dtype=torch.float16, device="npu")

    cu_seqlen_ks = torch.zeros(seq_len, dtype=torch.int32, device="npu")
    cu_seqlen_ke = torch.arange(1, seq_len + 1, dtype=torch.int32, device="npu") * (seq_len_kv // seq_len)
    cu_seqlen_ke = cu_seqlen_ke.clamp(max=seq_len_kv)

    max_block_id = seq_len_kv // kv_block_size
    topk_block_indices = torch.randint(0, max_block_id, (seq_len, topk), dtype=torch.int32, device="npu")

    logits = torch.empty((seq_len, topk * kv_block_size), dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    logits = kernel(
        q,
        k,
        topk_block_indices,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    torch.npu.synchronize()
    ref_logits = ref_block_sparse_mqa_attn(
        q,
        k,
        topk_block_indices,
        kv_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(ref_logits, logits, rtol=1e-2, atol=1e-2)

    print(f"Test passed! seq_len={seq_len}, seq_len_kv={seq_len_kv}, heads={heads}, topk={topk}")
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  kv_block_size: {kv_block_size}")

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Sparse MQA Attention Kernel Test")
    parser.add_argument("--seq_len", type=int, default=1024, help="Query sequence length")
    parser.add_argument("--seq_len_kv", type=int, default=128 * 1024, help="KV sequence length")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--index_dim", type=int, default=128, help="Index dimension")
    parser.add_argument("--kv_block_size", type=int, default=128, help="KV block size")
    parser.add_argument("--topk", type=int, default=16, help="Number of top blocks")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of token pairs per kernel block")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    # assert args.seq_len % (2 * args.num_pairs) == 0, \
    #     f"seq_len ({args.seq_len}) must be divisible by 2*num_pairs ({2 * args.num_pairs})"

    print("=" * 60)
    print("Block Sparse MQA Attention Kernel Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  seq_len: {args.seq_len}")
    print(f"  seq_len_kv: {args.seq_len_kv}")
    print(f"  heads: {args.heads}")
    print(f"  index_dim: {args.index_dim}")
    print(f"  kv_block_size: {args.kv_block_size}")
    print(f"  topk: {args.topk}")
    print(f"  num_pairs: {args.num_pairs}")
    print(f"  kernel_blocks: {args.seq_len // (2 * args.num_pairs)}")
    print(f"  dtype: {args.dtype}")
    print()

    test_block_sparse_mqa_attn(
        seq_len=args.seq_len,
        seq_len_kv=args.seq_len_kv,
        heads=args.heads,
        index_dim=args.index_dim,
        kv_block_size=args.kv_block_size,
        topk=args.topk,
        num_pairs=args.num_pairs,
        dtype=args.dtype,
    )

    print()
    print("All tests passed!")
