"""
Paged Block Sparse MQA Attention Kernel — Decode, Flat Grid

Each kernel block processes: 1 token × 1 n_outer group (4 K blocks).
Grid: batch * topk_groups (1D) — bx // topk_groups → token, bx % topk_groups → n_outer.
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[3],
    workspace_idx=[-1],
    target="pto",
)
def paged_block_sparse_mqa_attn_return_logits(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    kv_block_size: int,
    topk: int,
    heads: int,
    index_dim: int,
    max_blocks: int,
    num_stages: int = 2,  # noqa: ARG001
    threads: int = 2,  # noqa: ARG001
):
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    assert topk % 4 == 0, "topk must be divisible by 4"
    topk_groups = topk // 4

    total_tokens = batch * seq_len  # = batch when seq_len=1
    grid_size = batch * topk_groups  # 1D: batch and n_outer merged

    # Flattened shapes
    index_q_shape = [total_tokens, heads, index_dim]
    topk_index_shape = [total_tokens, topk]
    logits_shape = [total_tokens, topk, kv_block_size]
    weights_shape = [total_tokens, heads]

    kv_cache_shape = [num_phys_blocks, kv_block_size, 1, index_dim]
    block_tables_shape = [batch, max_blocks]
    context_lens_shape = [batch]

    H_per_block = heads
    kv = kv_block_size

    # ---------- Signal IDs ----------
    SIG_Q_L1 = 0
    SIG_K_L1_0 = 1
    SIG_K_L1_1 = 2
    SIG_L0AB_0 = 0
    SIG_L0AB_1 = 1
    SIG_L0C_0 = 0
    SIG_L0C_1 = 1
    SIG_S_UB = 0
    SIG_W_UB = 1
    SIG_LOGITS = 0
    # Single cross-scope flag (no ping-pong — single-shot)
    CROSS_FLAG_C2V = 0

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor(topk_index_shape, index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        ContextLens: T.Tensor(context_lens_shape, index_dtype),  # type: ignore
        BlockTables: T.Tensor(block_tables_shape, index_dtype),  # type: ignore
        workspace_1: T.Tensor([total_tokens, topk, H_per_block, kv], accum_dtype),
    ):
        # Grid: batch * topk_groups (1D, batch and n_outer merged)
        #   token  = bx // topk_groups   (batch index)
        #   n_outer = bx - token * topk_groups
        with T.Kernel(grid_size, is_npu=True) as (bx, by):
            token = bx // topk_groups
            n_outer = bx - token * topk_groups
            n_i_base = n_outer * 4
            n_i0 = n_i_base + 0
            n_i1 = n_i_base + 1
            n_i2 = n_i_base + 2
            n_i3 = n_i_base + 3

            # ---- V scope: UB allocations (2-block per sub-block) ----
            s_ub_2x = T.alloc_ub([H_per_block, kv * 2], accum_dtype)
            logits_2x = T.alloc_ub([1, kv * 2], accum_dtype)
            weights_ub = T.alloc_ub([heads], dtype)
            weights = T.alloc_ub([heads], accum_dtype)
            # Mask buffers (2 blocks merged)
            kvpi_a = T.alloc_ub([kv], "int32")
            kvpi_b = T.alloc_ub([kv], "int32")
            kvpf_2x = T.alloc_ub([kv * 2], "float")
            mask1_ub = T.alloc_ub([kv * 2 // 8], "uint8")
            mask2_ub = T.alloc_ub([kv * 2 // 8], "uint8")

            # ---- C scope: L1 / L0A / L0B / L0C ----
            q_l1 = T.alloc_L1([H_per_block, index_dim], dtype)
            k_l1_0 = T.alloc_L1([kv, index_dim], dtype)
            k_l1_1 = T.alloc_L1([kv, index_dim], dtype)
            l0a = T.alloc_L0A([H_per_block, index_dim], dtype)
            l0b_0 = T.alloc_L0B([index_dim, kv], dtype)
            l0b_1 = T.alloc_L0B([index_dim, kv], dtype)
            l0c_0 = T.alloc_L0C([H_per_block, kv], accum_dtype)
            l0c_1 = T.alloc_L0C([H_per_block, kv], accum_dtype)

            # ================================================================
            # C scope: single-shot 4-block double-buffered pipeline
            # ================================================================
            with T.Scope("C"):
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                T.set_flag("M", "MTE1", SIG_L0AB_0)
                T.set_flag("M", "MTE1", SIG_L0AB_1)
                T.set_flag("FIX", "M", SIG_L0C_0)
                T.set_flag("FIX", "M", SIG_L0C_1)

                if token < total_tokens:
                    b = token // seq_len

                    # ---- Wave 0: DMA K[0] → k_l1_0 ----
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                    T.copy(
                        KvCache[
                            BlockTables[b, TopKBlockIndex[token, n_i0]],
                            :,
                            0,
                            :,
                        ],
                        k_l1_0,
                    )
                    T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                    # ---- Wave 1: DMA K[1]→k_l1_1 | Stage Q→l0a, K[0]→l0b_0 ----
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                    T.copy(
                        KvCache[
                            BlockTables[b, TopKBlockIndex[token, n_i1]],
                            :,
                            0,
                            :,
                        ],
                        k_l1_1,
                    )
                    T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                    T.wait_flag("M", "MTE1", SIG_L0AB_0)
                    T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(IndexQ[token, :, :], q_l1)
                    T.set_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                    T.copy(q_l1, l0a)
                    T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                    T.copy(k_l1_0, l0b_0, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                    T.set_flag("MTE1", "M", SIG_L0AB_0)

                    # ---- Wave 2: DMA K[2]→k_l1_0 | Stage K[1]→l0b_1 | MMA K[0]→l0c_0 ----
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                    T.copy(
                        KvCache[
                            BlockTables[b, TopKBlockIndex[token, n_i2]],
                            :,
                            0,
                            :,
                        ],
                        k_l1_0,
                    )
                    T.set_flag("MTE2", "MTE1", SIG_K_L1_0)

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                    T.wait_flag("M", "MTE1", SIG_L0AB_1)
                    T.copy(k_l1_1, l0b_1, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                    T.set_flag("MTE1", "M", SIG_L0AB_1)

                    T.wait_flag("MTE1", "M", SIG_L0AB_0)
                    T.wait_flag("FIX", "M", SIG_L0C_0)
                    T.mma(l0a, l0b_0, l0c_0, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_0)
                    T.set_flag("M", "FIX", SIG_L0C_0)

                    # ---- Wave 3: DMA K[3]→k_l1_1 | Stage K[2]→l0b_0 | MMA K[1]→l0c_1 | Copy l0c_0→ws ----
                    T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                    T.copy(
                        KvCache[
                            BlockTables[b, TopKBlockIndex[token, n_i3]],
                            :,
                            0,
                            :,
                        ],
                        k_l1_1,
                    )
                    T.set_flag("MTE2", "MTE1", SIG_K_L1_1)

                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_0)
                    T.wait_flag("M", "MTE1", SIG_L0AB_0)
                    T.copy(k_l1_0, l0b_0, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1_0)
                    T.set_flag("MTE1", "M", SIG_L0AB_0)

                    T.wait_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("FIX", "M", SIG_L0C_1)
                    T.mma(l0a, l0b_1, l0c_1, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_1)
                    T.set_flag("M", "FIX", SIG_L0C_1)

                    T.wait_flag("M", "FIX", SIG_L0C_0)
                    T.copy(l0c_0, workspace_1[token, n_i0, :, :])
                    T.set_flag("FIX", "M", SIG_L0C_0)

                    # ---- Wave 4: Stage K[3]→l0b_1 | MMA K[2]→l0c_0 | Copy l0c_1→ws ----
                    T.wait_flag("MTE2", "MTE1", SIG_K_L1_1)
                    T.wait_flag("M", "MTE1", SIG_L0AB_1)
                    T.copy(k_l1_1, l0b_1, transpose=True)
                    T.set_flag("MTE1", "MTE2", SIG_K_L1_1)
                    T.set_flag("MTE1", "M", SIG_L0AB_1)

                    T.wait_flag("MTE1", "M", SIG_L0AB_0)
                    T.wait_flag("FIX", "M", SIG_L0C_0)
                    T.mma(l0a, l0b_0, l0c_0, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_0)
                    T.set_flag("M", "FIX", SIG_L0C_0)

                    T.wait_flag("M", "FIX", SIG_L0C_1)
                    T.copy(l0c_1, workspace_1[token, n_i1, :, :])
                    T.set_flag("FIX", "M", SIG_L0C_1)

                    # ---- Wave 5: MMA K[3]→l0c_1 | Copy l0c_0→ws (drain) ----
                    T.wait_flag("MTE1", "M", SIG_L0AB_1)
                    T.wait_flag("FIX", "M", SIG_L0C_1)
                    T.mma(l0a, l0b_1, l0c_1, init=True)
                    T.set_flag("M", "MTE1", SIG_L0AB_1)
                    T.set_flag("M", "FIX", SIG_L0C_1)

                    T.wait_flag("M", "FIX", SIG_L0C_0)
                    T.copy(l0c_0, workspace_1[token, n_i2, :, :])
                    T.set_flag("FIX", "M", SIG_L0C_0)

                    # ---- Wave 6: Copy l0c_1→ws (drain) ----
                    T.wait_flag("M", "FIX", SIG_L0C_1)
                    T.copy(l0c_1, workspace_1[token, n_i3, :, :])
                    T.set_flag("FIX", "M", SIG_L0C_1)

                    # Signal V: 4 blocks ready
                    T.set_cross_flag("FIX", CROSS_FLAG_C2V)

                # Destroy
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_0)
                T.wait_flag("MTE1", "MTE2", SIG_K_L1_1)
                T.wait_flag("M", "MTE1", SIG_L0AB_0)
                T.wait_flag("M", "MTE1", SIG_L0AB_1)
                T.wait_flag("FIX", "M", SIG_L0C_0)
                T.wait_flag("FIX", "M", SIG_L0C_1)

            # ================================================================
            # V scope: single-shot 4-block merged processing
            # ================================================================
            with T.Scope("V"):
                T.set_flag("V", "MTE2", SIG_S_UB)
                T.set_flag("V", "MTE2", SIG_W_UB)
                T.set_flag("MTE3", "V", SIG_LOGITS)

                T.wait_cross_flag(CROSS_FLAG_C2V, "MTE2")

                if token < total_tokens:
                    b_v = token // seq_len

                    # Column split: by=0→n_i0,n_i1 (first 2*kv cols)
                    #              by=1→n_i2,n_i3 (last  2*kv cols)
                    start_n = by * 2  # block offset within the 4-block group

                    # DMA 2 workspace blocks → s_ub_2x
                    T.wait_flag("V", "MTE2", SIG_S_UB)
                    T.copy(
                        workspace_1[token, n_i_base + start_n + 0, :, :],
                        s_ub_2x[:, 0 * kv : 1 * kv],
                    )
                    T.copy(
                        workspace_1[token, n_i_base + start_n + 1, :, :],
                        s_ub_2x[:, 1 * kv : 2 * kv],
                    )
                    T.set_flag("MTE2", "V", SIG_S_UB)
                    T.wait_flag("MTE2", "V", SIG_S_UB)

                    # DMA weights once
                    T.wait_flag("V", "MTE2", SIG_W_UB)
                    T.copy(Weights[token, :], weights_ub)
                    T.set_flag("MTE2", "V", SIG_W_UB)
                    T.wait_flag("MTE2", "V", SIG_W_UB)
                    T.copy(weights_ub, weights)
                    T.pipe_barrier("v")
                    T.set_flag("V", "MTE2", SIG_W_UB)

                    # Vector ops on [H, 2*kv]
                    T.tile.relu(s_ub_2x, s_ub_2x)
                    T.tile.row_expand_mul(s_ub_2x, s_ub_2x, weights)

                    # Reduce: [H, 2*kv] → [2*kv]
                    T.wait_flag("MTE3", "V", SIG_LOGITS)
                    T.reduce_sum(s_ub_2x, logits_2x, dim=0, clear=True)
                    T.set_flag("V", "MTE2", SIG_S_UB)

                    # Mask: 2 blocks merged
                    T.tile.createvecindex(kvpi_a, TopKBlockIndex[token, n_i_base + start_n + 0] * kv)
                    T.tile.createvecindex(kvpi_b, TopKBlockIndex[token, n_i_base + start_n + 1] * kv)
                    T.copy(kvpi_a, kvpf_2x[0 * kv : 1 * kv])
                    T.copy(kvpi_b, kvpf_2x[1 * kv : 2 * kv])
                    T.pipe_barrier("v")

                    cu_k_e_max = ContextLens[b_v]
                    T.tile.compare(mask1_ub, kvpf_2x, T.float32(0), "GE")
                    T.tile.compare(mask2_ub, kvpf_2x, T.float32(cu_k_e_max), "LT")
                    T.pipe_barrier("v")
                    T.tile.bitwise_and(mask1_ub, mask1_ub, mask2_ub)

                    T.tile.select(logits_2x[0, :], mask1_ub, logits_2x[0, :], -T.infinity(accum_dtype), "VSEL_TENSOR_SCALAR_MODE")

                    T.set_flag("V", "MTE3", SIG_LOGITS)

                    # DMA output: 2 × [kv] slices
                    T.wait_flag("V", "MTE3", SIG_LOGITS)
                    T.copy(
                        logits_2x[0, 0 * kv : 1 * kv],
                        Logits[token, n_i_base + start_n + 0, :],
                    )
                    T.copy(
                        logits_2x[0, 1 * kv : 2 * kv],
                        Logits[token, n_i_base + start_n + 1, :],
                    )
                    T.set_flag("MTE3", "V", SIG_LOGITS)

                # Destroy
                T.wait_flag("V", "MTE2", SIG_S_UB)
                T.wait_flag("V", "MTE2", SIG_W_UB)
                T.wait_flag("MTE3", "V", SIG_LOGITS)

    return kernel


def ref_paged_block_sparse_mqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_block_indices: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Reference: paged sparse MQA attention."""
    batch, seq_len, heads, index_dim = q.shape
    topk = topk_block_indices.shape[2]

    q = q.float()
    kv_cache = kv_cache.float()
    weights = weights.float()

    logits_out = torch.zeros(
        (batch, seq_len, topk, kv_block_size),
        dtype=torch.float32,
        device=q.device,
    )

    for b in range(batch):
        ctx_len = context_lens[b].item()
        for s in range(seq_len):
            for k_i in range(topk):
                logical_block = topk_block_indices[b, s, k_i].item()
                phys_block = block_tables[b, logical_block].item()
                k_block = kv_cache[phys_block, :, 0, :]

                scores = q[b, s, :, :] @ k_block.T
                scores = scores.relu()
                scores = scores * weights[b, s, :].unsqueeze(1)
                logits_val = scores.sum(dim=0)

                block_start = logical_block * kv_block_size
                pos = torch.arange(kv_block_size, device=q.device) + block_start
                pos_out = pos >= ctx_len
                logits_val[pos_out] = float("-inf")

                logits_out[b, s, k_i, :] = logits_val

    return logits_out.view(batch, seq_len, topk * kv_block_size)


def test_paged_block_sparse_mqa_attn(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    heads: int,
    index_dim: int,
    kv_block_size: int,
    topk: int,
    max_blocks: int,
    dtype: str = "float16",
):
    """Test paged sparse MQA attention (decode, flat grid)."""
    kernel = paged_block_sparse_mqa_attn_return_logits(
        batch=batch,
        seq_len=seq_len,
        num_phys_blocks=num_phys_blocks,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
        max_blocks=max_blocks,
    )
    print(kernel.get_kernel_source())

    device = "npu"
    total_tokens = batch * seq_len
    topk_groups = topk // 4

    q = torch.rand((batch, seq_len, heads, index_dim), dtype=torch.float16).to(device)
    kv_cache = torch.rand((num_phys_blocks, kv_block_size, 1, index_dim), dtype=torch.float16).to(device)
    weights = torch.rand((batch, seq_len, heads), dtype=torch.float16).to(device)

    context_lens = torch.randint(kv_block_size, num_phys_blocks * kv_block_size + 1, (batch,), dtype=torch.int32).to(device)

    block_tables = torch.arange(max_blocks, dtype=torch.int32).unsqueeze(0).expand(batch, -1).contiguous().to(device)

    max_logical_block = (context_lens.max().item() + kv_block_size - 1) // kv_block_size
    max_logical_block = min(max_logical_block, max_blocks)
    topk_block_indices = torch.randint(0, max_logical_block, (batch, seq_len, topk), dtype=torch.int32).to(device)

    # Flatten batch×seq_len
    q_flat = q.reshape(total_tokens, heads, index_dim).contiguous()
    topk_flat = topk_block_indices.reshape(total_tokens, topk).contiguous()
    weights_flat = weights.reshape(total_tokens, heads).contiguous()

    torch.npu.synchronize()
    logits = kernel(
        q_flat,
        kv_cache,
        topk_flat,
        weights_flat,
        context_lens,
        block_tables,
    )
    torch.npu.synchronize()

    ref_logits = ref_paged_block_sparse_mqa_attn(
        q,
        kv_cache,
        topk_block_indices,
        kv_block_size,
        weights,
        context_lens,
        block_tables,
    )
    torch.npu.synchronize()

    logits_flat = logits.view(batch, seq_len, topk * kv_block_size)
    torch.testing.assert_close(ref_logits, logits_flat, rtol=1e-2, atol=1e-2)

    print(f"Test passed! batch={batch}, seq_len={seq_len}, heads={heads}, topk={topk}")
    print(f"  grid: [{batch}, {topk_groups}]  Q: {q.shape}  Logits: {logits_flat.shape}")

    return logits_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paged Sparse MQA Attention — Decode Flat Grid")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--num_phys_blocks", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--index_dim", type=int, default=128)
    parser.add_argument("--kv_block_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    assert args.topk % 4 == 0, "topk must be divisible by 4"

    print("=" * 60)
    print("Paged Sparse MQA Attention — Decode (Flat Grid)")
    print("=" * 60)
    print(f"  batch={args.batch}, seq_len={args.seq_len}")
    print(f"  num_phys_blocks={args.num_phys_blocks}, max_blocks={args.max_blocks}")
    print(f"  heads={args.heads}, index_dim={args.index_dim}")
    print(f"  kv_block_size={args.kv_block_size}, topk={args.topk}")
    print(f"  topk_groups={args.topk // 4}")
    print(f"  grid=[{args.batch}, {args.topk // 4}] = {args.batch * args.topk // 4} tasks")
    print()

    test_paged_block_sparse_mqa_attn(
        batch=args.batch,
        seq_len=args.seq_len,
        num_phys_blocks=args.num_phys_blocks,
        heads=args.heads,
        index_dim=args.index_dim,
        kv_block_size=args.kv_block_size,
        topk=args.topk,
        max_blocks=args.max_blocks,
    )
    print("Kernel Output Match!")
