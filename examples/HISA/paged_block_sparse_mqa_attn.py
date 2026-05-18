"""
Paged Block Sparse MQA Attention Kernel

This kernel computes sparse MQA attention on selected topk blocks with paged KV-cache.
Used in HISA for decode phase with paged attention support.

Reference: hisa_vllm_patch/custom_ops.py - paged_block_sparse_mqa_attn_return_logits
"""

import tilelang
from tilelang import language as T
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    },
    out_idx=[3],
    workspace_idx=[-1],
    target="pto",
)
def paged_block_sparse_mqa_attn_return_logits(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    max_blocks: int,
    paged_block_size: int,
    kv_block_size: int,
    topk: int,
    heads: int,
    index_dim: int,
    num_stages: int = 1,
    threads: int = 256,
):
    """Paged sparse MQA attention kernel on selected blocks.

    Computes attention scores on topk selected blocks with paged KV-cache access.

    Args:
        IndexQ: [batch, seq_len, heads, index_dim] - Query indices
        KvCache: [num_phys_blocks, paged_block_size, 1, index_dim] - Paged KV cache
        TopKBlockIndex: [batch, seq_len, topk] - Selected block indices (int32)
        Logits: [batch, seq_len, topk, kv_block_size] - Attention logits (float32)
        Weights: [batch, seq_len, heads] - Attention weights
        ContextLens: [batch] - Effective context length per batch (int32)
        BlockTables: [batch, max_blocks] - Logical to physical block mapping (int32)

    Returns:
        Logits tensor with sparse attention scores
    """
    dtype = "float16"
    accum_dtype = "float32"
    index_dtype = "int32"

    index_q_shape = [batch, seq_len, heads, index_dim]
    kv_cache_shape = [num_phys_blocks, paged_block_size, 1, index_dim]
    logits_shape = [batch, seq_len, topk, kv_block_size]
    weights_shape = [batch, seq_len, heads]

    H_per_block = heads
    block_N = paged_block_size
    assert kv_block_size % block_N == 0, "kv_block_size must divide paged_block_size"
    assert paged_block_size == block_N, "for simplicity we require paged_block_size == block_N"

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        KvCache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
        TopKBlockIndex: T.Tensor([batch, seq_len, topk], index_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor(weights_shape, dtype),  # type: ignore
        ContextLens: T.Tensor([batch], index_dtype),  # type: ignore
        BlockTables: T.Tensor([batch, max_blocks], index_dtype),  # type: ignore
        workspace_1: T.Tensor([batch, seq_len, topk, H_per_block, kv_block_size], accum_dtype),
    ):
        with T.Kernel(batch * seq_len, is_npu=True) as (bx, _):
            b = bx // seq_len
            seq_len_i = bx % seq_len

            index_q_shared = T.alloc_shared([H_per_block, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([H_per_block, block_N], accum_dtype)
            s_ub = T.alloc_shared([H_per_block, block_N], accum_dtype)
            logits = T.alloc_shared([1, block_N], accum_dtype)
            weights_ub = T.alloc_shared([heads], dtype)
            weights = T.alloc_shared([heads], accum_dtype)

            cu_k_e_max = ContextLens[b]

            T.copy(IndexQ[b, seq_len_i, :, :], index_q_shared)
            T.copy(Weights[b, seq_len_i, :], weights_ub)
            T.copy(weights_ub, weights)

            for n_i in T.serial(topk):
                topk_block_id = TopKBlockIndex[b, seq_len_i, n_i]
                block_s = topk_block_id * kv_block_size

                for b_i in T.serial(kv_block_size // block_N):
                    block_s_i = block_s + b_i * block_N

                    if block_s_i // paged_block_size >= 0 and block_s_i // paged_block_size < max_blocks:
                        phys = BlockTables[b, block_s_i // paged_block_size]
                        T.copy(KvCache[phys, :, 0, :], index_k_shared)

                    T.gemm_v0(index_q_shared, index_k_shared, s, transpose_A=False, transpose_B=True, init=True)

                    T.copy(s, workspace_1[b, seq_len_i, n_i, :, b_i * block_N : b_i * block_N + block_N])
                    T.copy(workspace_1[b, seq_len_i, n_i, :, b_i * block_N : b_i * block_N + block_N], s_ub)

                    T.tile.relu(s_ub, s_ub)
                    for h_i in T.serial(heads):
                        T.tile.mul(s_ub[h_i, :], s_ub[h_i, :], weights[h_i])

                    T.reduce_sum(s_ub, logits, dim=0, clear=True)

                    for i_i in T.serial(block_N):
                        k_i = block_s_i + i_i
                        p = k_i // paged_block_size
                        if (k_i < 0) or (k_i >= cu_k_e_max) or (p < 0) or (p >= max_blocks):
                            logits[0, i_i] = -T.infinity(accum_dtype)

                    T.copy(logits, Logits[b, seq_len_i, n_i, b_i * block_N : b_i * block_N + block_N])

    return kernel


def ref_paged_block_sparse_mqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_block_indices: torch.Tensor,
    kv_block_size: int,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    paged_block_size: int,
) -> torch.Tensor:
    """Reference implementation of paged sparse MQA attention using torch_npu vectorization.

    Args:
        q: [batch, seq_len, heads, index_dim] - Query tensor
        kv_cache: [num_phys_blocks, paged_block_size, 1, index_dim] - Paged KV cache
        topk_block_indices: [batch, seq_len, topk] - Selected block indices
        kv_block_size: Block size for computation
        weights: [batch, seq_len, heads] - Attention weights
        context_lens: [batch] - Effective context length per batch
        block_tables: [batch, max_blocks] - Logical to physical block mapping
        paged_block_size: Physical block size

    Returns:
        logits: [batch, seq_len, topk, kv_block_size] - Sparse attention logits
    """
    batch, seq_len, heads, index_dim = q.shape
    num_phys_blocks = kv_cache.shape[0]
    topk = topk_block_indices.shape[2]
    max_blocks = block_tables.shape[1]

    q = q.float()
    kv_cache = kv_cache.float()
    weights = weights.float()

    kv_cache_flat = kv_cache.view(num_phys_blocks, paged_block_size, index_dim)

    logical_positions = topk_block_indices.unsqueeze(-1) * kv_block_size + torch.arange(kv_block_size, device=q.device)
    logical_positions = logical_positions.long()

    paged_block_indices = logical_positions // paged_block_size

    valid_paged_blocks = paged_block_indices < max_blocks

    paged_block_indices_clamped = paged_block_indices.clamp(0, max_blocks - 1)

    batch_idx = torch.arange(batch, device=q.device).view(batch, 1, 1, 1)
    phys_block_indices = block_tables[batch_idx, paged_block_indices_clamped]

    pos_in_blocks = logical_positions % paged_block_size

    phys_block_indices_flat = phys_block_indices.view(-1)
    pos_in_blocks_flat = pos_in_blocks.view(-1)

    k_gathered = kv_cache_flat[phys_block_indices_flat, pos_in_blocks_flat]
    k_gathered = k_gathered.view(batch, seq_len, topk, kv_block_size, index_dim)

    scores = torch.einsum("bqhd,bqnkd->bqnkh", q, k_gathered)

    weights_expanded = weights.unsqueeze(2).unsqueeze(3)
    scores = scores.relu() * weights_expanded

    logits = scores.sum(dim=-1)

    context_lens_exp = context_lens.view(batch, 1, 1, 1)
    pos_out_of_context = logical_positions >= context_lens_exp

    invalid_paged_blocks = ~valid_paged_blocks

    invalid_mask = pos_out_of_context | invalid_paged_blocks
    logits = logits.masked_fill(invalid_mask, float("-inf"))

    return logits


def test_paged_block_sparse_mqa_attn(
    batch: int,
    seq_len: int,
    num_phys_blocks: int,
    max_blocks: int,
    heads: int,
    index_dim: int,
    paged_block_size: int,
    kv_block_size: int,
    topk: int,
    dtype: str = "float16",
):
    """Test paged sparse MQA attention kernel with golden validation."""

    kernel = paged_block_sparse_mqa_attn_return_logits(
        batch=batch,
        seq_len=seq_len,
        num_phys_blocks=num_phys_blocks,
        max_blocks=max_blocks,
        paged_block_size=paged_block_size,
        kv_block_size=kv_block_size,
        topk=topk,
        heads=heads,
        index_dim=index_dim,
    )

    q = torch.rand((batch, seq_len, heads, index_dim), dtype=torch.float16, device="npu")
    kv_cache = torch.rand((num_phys_blocks, paged_block_size, 1, index_dim), dtype=torch.float16, device="npu")
    weights = torch.randn((batch, seq_len, heads), dtype=torch.float16, device="npu")

    context_lens = torch.randint(
        max_blocks * paged_block_size // 2, max_blocks * paged_block_size, (batch,), dtype=torch.int32, device="npu"
    )

    block_tables = torch.randint(0, num_phys_blocks, (batch, max_blocks), dtype=torch.int32, device="npu")

    max_block_id = (context_lens.max().item() + kv_block_size - 1) // kv_block_size
    topk_block_indices = torch.randint(0, min(max_block_id, max_blocks), (batch, seq_len, topk), dtype=torch.int32, device="npu")

    torch.npu.synchronize()

    logits = kernel(
        q,
        kv_cache,
        topk_block_indices,
        weights,
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
        paged_block_size,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(ref_logits, logits, rtol=1e-2, atol=1e-2)

    print(f"Test passed! batch={batch}, seq_len={seq_len}, heads={heads}, topk={topk}")
    print(f"  Q shape: {q.shape}")
    print(f"  KvCache shape: {kv_cache.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  paged_block_size: {paged_block_size}, kv_block_size: {kv_block_size}")

    return logits


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Paged Block Sparse MQA Attention Kernel Test")
    # parser.add_argument("--batch", type=int, default=2, help="Batch size")
    # parser.add_argument("--seq_len", type=int, default=128, help="Query sequence length")
    # parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    # parser.add_argument("--index_dim", type=int, default=128, help="Index dimension")
    # parser.add_argument("--paged_block_size", type=int, default=16, help="Paged block size")
    # parser.add_argument("--kv_block_size", type=int, default=16, help="KV block size")
    # parser.add_argument("--topk", type=int, default=16, help="Number of top blocks")
    # parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Paged Block Sparse MQA Attention Kernel Test")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1, help="Query sequence length")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--index_dim", type=int, default=128, help="Index dimension")
    parser.add_argument("--paged_block_size", type=int, default=128, help="Paged block size")
    parser.add_argument("--kv_block_size", type=int, default=128, help="KV block size")
    parser.add_argument("--topk", type=int, default=16, help="Number of top blocks")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    args = parser.parse_args()

    num_phys_blocks = args.batch * 256
    max_blocks = 256

    torch.set_default_device("npu")
    torch.manual_seed(42)
    tilelang.disable_cache()

    print("=" * 60)
    print("Paged Block Sparse MQA Attention Kernel Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  batch: {args.batch}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  heads: {args.heads}")
    print(f"  index_dim: {args.index_dim}")
    print(f"  paged_block_size: {args.paged_block_size}")
    print(f"  kv_block_size: {args.kv_block_size}")
    print(f"  topk: {args.topk}")
    print(f"  dtype: {args.dtype}")
    print()

    test_paged_block_sparse_mqa_attn(
        batch=args.batch,
        seq_len=args.seq_len,
        num_phys_blocks=num_phys_blocks,
        max_blocks=max_blocks,
        heads=args.heads,
        index_dim=args.index_dim,
        paged_block_size=args.paged_block_size,
        kv_block_size=args.kv_block_size,
        topk=args.topk,
        dtype=args.dtype,
    )

    print()
    print("All tests passed!")
