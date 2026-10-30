"""GLM-5.3 paged FP8 pooled-history logits (fp8_mqa_logits) for Ascend NPU.

For every query row the kernel walks the row's valid pool range
``[pool_start, pool_end)``, gathers the pooled keys from a caller-owned paged
cache through ``pool_page_table``, dequantizes the E4M3 bytes with the same
bit-level scheme the compressor writes, scores the 32 indexer heads
(``keys @ query^T``), applies the cached per-pool scale, ReLU-gates the head
scores, weights them and reduces them to one FP32 logit per pool.  Pools
outside the range or backed by an invalid page emit ``-inf``.

Cache contract (Ascend port): the reference consumes FP8 tensors.  Ascend A2/A3
has no FP8 data path, so this port keeps the very same E4M3 byte patterns in a
``uint8`` ``k_cache`` (written by ``example_glm53_kpool_compress.py``) plus the
FP32 ``scale_cache``, and decodes them in-kernel.  ``query`` is BF16: the E4M3
query values load losslessly into BF16, and the caller folds the per-head query
scale into ``weights`` exactly like the reference.

Two kernels, replacing the reference's host-side ``torch.full(-inf)`` prefill:

  - ``..._mask_kernel`` builds a per-lane validity mask (``0`` inside the
    range with a valid page, ``-inf`` otherwise) from ``pool_starts`` /
    ``pool_ends`` / the page table.  It runs on a single AIV, where the
    64-lane compare/select and the index generator are exact.
  - ``..._kernel`` does gather -> E4M3 decode -> ``T.gemm_v0`` -> scale/ReLU/
    weights -> head reduce, then adds the mask and writes the tile.  The
    addition makes out-of-range lanes exact ``-inf`` without a per-lane
    compare/select on the vector tile.

Device-side notes:

  - Blocks are ``(row, pool-tile)`` tasks over one 64-pool cache page; each of
    the two AIVs owns a 32-pool half: one clamped page-table lookup plus one
    bulk half-page copy, a whole-half E4M3 decode, one whole-page GEMM, then
    its own half of the score post-processing and one static-extent write.
  - The Cube/Vector exchange runs through explicit ``workspace_keys`` /
    ``workspace_scores`` GM buffers (one producer/consumer pair each), which
    the auto cross-core sync can pair 1:1.  This follows
    ``examples/developer_mode/sparse_flash_attn_developer.py``; the
    ``threads=2`` vid-reduction path mis-splits this kernel's sub-region
    copies and small compare/broadcast/index ops (confirmed with probes).
    Workspace buffer names must contain ``workspace`` (the sync pass detects
    them by name).
  - The kernel writes whole static-extent tiles into the
    ``ceil(max_num_pools / 64) * 64`` padded output; the host adapter returns
    the ``[:, :max_num_pools]`` view (1D UB slices are not usable as copy
    operands on this target).

Mode: Developer pass_configs (auto CV combine/sync + auto sync + memory
planning) with ``alloc_L1``/``alloc_L0C``/``alloc_ub`` and explicit
``workspace_idx``, following
``examples/developer_mode/sparse_flash_attn_developer.py``.
"""

import torch

import tilelang
import tilelang.language as T

GLM53_HEAD_DIM = 128
GLM53_INDEX_HEADS = 32
GLM53_PAGE_SIZE = 64
E4M3_MAX = 448.0

# E4M3 lane layout: sign(1) | exp(4) | mantissa(3). Decoding rebuilds the FP32
# bit pattern from those fields, so every shift/bias below is named.
E4M3_SIGN_SHIFT = 7
E4M3_EXP_SHIFT = 3
E4M3_EXP_BITS = 4
E4M3_EXP_BIAS_GAP = 120  # fp32 exponent bias (127) - E4M3 exponent bias (7)
FP32_EXP_SHIFT = 23
E4M3_MANT_TO_FP32_SHIFT = 20  # fp32 mantissa bits (23) - E4M3 mantissa bits (3)
E4M3_SUBNORMAL_SCALE = 2.0**-9  # subnormal code step (2^-9)
SIGN_TO_FACTOR_STEP = -2  # sign bit s in {0, 1} -> multiplier 1 - 2s
SIGN_TO_FACTOR_OFFSET = 1

# One tile = one cache page (64 pools).
POOL_TILE = GLM53_PAGE_SIZE

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4], pass_configs=PASS_CONFIGS)
def glm53_kpool_fp8_mqa_logits_mask_kernel(
    num_rows: int,
    num_tiles: int,
    max_num_pools: int,
    num_blocks: int,
    num_page_rows: int,
    max_pages: int,
    tile_pools: int = POOL_TILE,
):
    """Per-lane validity mask: 0 inside ``[pool_start, pool_end)`` with a valid
    page, ``-inf`` otherwise.  Runs on a single AIV, where the 64-lane
    compare/select and the index generator are exact.
    """

    @T.prim_func
    def main(
        pool_page_table: T.Tensor((num_page_rows, max_pages), "int32"),
        page_table_rows: T.Tensor((num_rows,), "int32"),
        pool_starts: T.Tensor((num_rows,), "int32"),
        pool_ends: T.Tensor((num_rows,), "int32"),
        mmask: T.Tensor((num_rows * num_tiles, tile_pools), "float32"),
    ):
        with T.Kernel(num_rows * num_tiles, is_npu=True) as (cid, vid):
            if vid == 0:
                row = cid // num_tiles
                tile = cid % num_tiles
                t0 = tile * tile_pools
                neg_inf = -T.infinity("float32")

                buf = T.alloc_shared((tile_pools,), "float32")
                idx = T.alloc_shared((tile_pools,), "int32")
                idx_f = T.alloc_shared((tile_pools,), "float32")
                m_lo = T.alloc_shared((tile_pools,), "uint8")
                m_hi = T.alloc_shared((tile_pools,), "uint8")

                st = T.alloc_var("int32")
                st = pool_starts[row]
                en = T.alloc_var("int32")
                en = pool_ends[row]
                tr = T.alloc_var("int32")
                tr = page_table_rows[row]

                page_c = T.max(T.min(tile, max_pages - 1), 0)
                tr_c = T.max(T.min(tr, num_page_rows - 1), 0)
                entry = T.alloc_var("int32")
                entry = pool_page_table[tr_c, page_c]

                T.tile.createvecindex(idx, 0)
                T.tile.cast(idx_f, idx, "CAST_RINT", tile_pools)
                hi = T.min(en - t0, T.min(tile_pools, max_num_pools - t0))
                T.tile.fill(buf, 0.0)
                T.tile.compare(m_lo, idx_f, T.float32(st - t0), "GE")
                T.tile.select(buf, m_lo, buf, neg_inf, "VSEL_TENSOR_SCALAR_MODE")
                T.tile.compare(m_hi, idx_f, T.float32(hi), "LT")
                T.tile.select(buf, m_hi, buf, neg_inf, "VSEL_TENSOR_SCALAR_MODE")

                if tr < 0 or tr >= num_page_rows or tile >= max_pages or entry < 0 or entry >= num_blocks:
                    T.tile.fill(buf, neg_inf)

                T.copy(buf, mmask[row * num_tiles + tile, 0:tile_pools])

    return main


@tilelang.jit(out_idx=[9], workspace_idx=[10, 11], pass_configs=PASS_CONFIGS)
def glm53_kpool_fp8_mqa_logits_kernel(
    num_rows: int,
    max_num_pools: int,
    num_blocks: int,
    num_page_rows: int,
    max_pages: int,
    page_size: int = GLM53_PAGE_SIZE,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
):
    """Build the paged pooled-history logits kernel (workspace + vid fallback).

    Inputs:
      query           [num_rows, num_heads, head_dim] bf16
      k_cache         [num_blocks, page_size, head_dim] u8   (E4M3 bytes)
      scale_cache     [num_blocks, page_size] f32
      weights         [num_rows, num_heads] f32
      pool_page_table [num_page_rows, max_pages] i32
      page_table_rows [num_rows] i32
      pool_starts     [num_rows] i32
      pool_ends       [num_rows] i32
      mmask           [num_rows * num_tiles, tile_pools] f32 (0 / -inf)
    Output:
      logits          [num_rows, ceil(max_num_pools / POOL_TILE) * POOL_TILE] f32
    Workspaces (allocated by the adapter):
      workspace_keys         [num_rows * num_tiles, tile_pools, head_dim] bf16
      workspace_scores       [num_rows * num_tiles, tile_pools, num_heads] f32

    Each block owns one 64-pool page: the two AIVs decode 32-pool halves into
    ``workspace_keys``, the Cube reads the page back and writes the raw scores to
    ``workspace_scores``, and each AIV post-processes its own 32-pool half.  This
    mirrors ``examples/developer_mode/sparse_flash_attn_developer.py``: the
    explicit workspaces give the auto cross-core sync a 1:1 producer/consumer
    pair per workspace, which the ``threads=2`` vid-reduction path cannot
    express for this kernel's sub-region and broadcast patterns.
    """
    if page_size != GLM53_PAGE_SIZE:
        raise ValueError(f"GLM-5.3 k-pool requires page_size={GLM53_PAGE_SIZE}, got {page_size}")
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")

    tile_pools = POOL_TILE
    half = tile_pools // 2
    num_tiles = (max_num_pools + tile_pools - 1) // tile_pools
    padded_pools = num_tiles * tile_pools
    half_elems = half * head_dim

    @T.prim_func
    def main(
        query: T.Tensor((num_rows, num_heads, head_dim), "bfloat16"),
        k_cache: T.Tensor((num_blocks, page_size, head_dim), "uint8"),
        scale_cache: T.Tensor((num_blocks, page_size), "float32"),
        weights: T.Tensor((num_rows, num_heads), "float32"),
        pool_page_table: T.Tensor((num_page_rows, max_pages), "int32"),
        page_table_rows: T.Tensor((num_rows,), "int32"),
        pool_starts: T.Tensor((num_rows,), "int32"),
        pool_ends: T.Tensor((num_rows,), "int32"),
        mmask: T.Tensor((num_rows * num_tiles, tile_pools), "float32"),
        logits: T.Tensor((num_rows, padded_pools), "float32"),
        workspace_keys: T.Tensor((num_rows * num_tiles, tile_pools, head_dim), "bfloat16"),
        workspace_scores: T.Tensor((num_rows * num_tiles, tile_pools, num_heads), "float32"),
    ):
        with T.Kernel(num_rows * num_tiles, is_npu=True) as (cid, vid):
            row = cid // num_tiles
            tile = cid % num_tiles
            t0 = tile * tile_pools
            v0 = vid * half

            # ---- Cube side ----
            q_l1 = T.alloc_L1((num_heads, head_dim), "bfloat16")
            key_l1 = T.alloc_L1((tile_pools, head_dim), "bfloat16")
            scores_l0c = T.alloc_L0C((tile_pools, num_heads), "float32")

            # ---- Vector side (this AIV's 32-pool half) ----
            key_u8 = T.alloc_ub((half, head_dim), "uint8")
            key_ub = T.alloc_ub((half, head_dim), "bfloat16")
            sc_ub = T.alloc_ub((half,), "float32")
            w_ub = T.alloc_ub((num_heads,), "float32")
            s_ub = T.alloc_ub((half, num_heads), "float32")
            sc2d = T.alloc_ub((half, num_heads), "float32")
            w2d = T.alloc_ub((half, num_heads), "float32")
            v_ub = T.alloc_ub((half,), "float32")
            mask_ub = T.alloc_ub((half,), "float32")

            # E4M3 decode temporaries (this half).
            df16 = T.alloc_ub((half, head_dim), "float16")
            dbits = T.alloc_ub((half, head_dim), "int32")
            dsign = T.alloc_ub((half, head_dim), "int32")
            dtmp = T.alloc_ub((half, head_dim), "int32")
            de4 = T.alloc_ub((half, head_dim), "int32")
            dm3 = T.alloc_ub((half, head_dim), "int32")
            dmag = T.alloc_ub((half, head_dim), "int32")
            dfnorm = T.alloc_ub((half, head_dim), "float32")
            dfsub = T.alloc_ub((half, head_dim), "float32")
            dfmag = T.alloc_ub((half, head_dim), "float32")
            dmask = T.alloc_ub((half, head_dim), "uint8")
            dfsign = T.alloc_ub((half, head_dim), "float32")
            dfval = T.alloc_ub((half, head_dim), "float32")

            # 1. Gather this half of the page (clamped page-table lookup).
            page = tile
            page_c = T.max(T.min(page, max_pages - 1), 0)
            tr_c = T.max(T.min(page_table_rows[row], num_page_rows - 1), 0)
            blk = T.max(T.min(pool_page_table[tr_c, page_c], num_blocks - 1), 0)
            T.copy(k_cache[blk, v0 : v0 + half, 0:head_dim], key_u8)
            T.copy(scale_cache[blk, v0 : v0 + half], sc_ub)

            # 2. E4M3 bytes -> BF16 (same bit-level chain as the compressor).
            T.tile.cast(df16, key_u8, "CAST_NONE", half_elems)
            T.tile.cast(dbits, df16, "CAST_RINT", half_elems)
            T.tile.bitwise_rshift(dsign, dbits, E4M3_SIGN_SHIFT)
            T.tile.bitwise_rshift(dtmp, dbits, E4M3_EXP_SHIFT)
            T.tile.bitwise_lshift(de4, dsign, E4M3_EXP_BITS)
            T.tile.sub(de4, dtmp, de4)
            T.tile.bitwise_lshift(dtmp, dtmp, E4M3_EXP_SHIFT)
            T.tile.sub(dm3, dbits, dtmp)
            T.tile.add(dmag, de4, E4M3_EXP_BIAS_GAP)
            T.tile.bitwise_lshift(dmag, dmag, FP32_EXP_SHIFT)
            T.tile.bitwise_lshift(dtmp, dm3, E4M3_MANT_TO_FP32_SHIFT)
            T.tile.add(dmag, dmag, dtmp)
            T.reinterpretcast(dfnorm, dmag, "float")
            T.tile.cast(dfsub, dm3, "CAST_RINT", half_elems)
            T.tile.mul(dfsub, dfsub, E4M3_SUBNORMAL_SCALE)
            T.tile.compare(dmask, de4, T.int32(0), "EQ")
            T.tile.select(dfmag, dmask, dfsub, dfnorm, "VSEL_TENSOR_TENSOR_MODE")
            T.tile.mul(dsign, dsign, SIGN_TO_FACTOR_STEP)
            T.tile.add(dsign, dsign, SIGN_TO_FACTOR_OFFSET)
            T.tile.cast(dfsign, dsign, "CAST_RINT", half_elems)
            T.tile.mul(dfval, dfmag, dfsign)
            T.tile.cast(key_ub, dfval, "CAST_RINT", half_elems)
            T.copy(key_ub, workspace_keys[cid, v0 : v0 + half, 0:head_dim])

            # 3. Cube: scores = keys @ query^T for the whole page.
            T.copy(query[row, 0:num_heads, 0:head_dim], q_l1)
            T.copy(workspace_keys[cid, 0:tile_pools, 0:head_dim], key_l1)
            T.gemm_v0(key_l1, q_l1, scores_l0c, transpose_B=True, init=True)
            T.copy(scores_l0c, workspace_scores[cid, 0:tile_pools, 0:num_heads])

            # 4. Vector: scale, ReLU gate, per-head weights on this half.
            T.copy(workspace_scores[cid, v0 : v0 + half, 0:num_heads], s_ub)
            T.copy(weights[row, 0:num_heads], w_ub)
            T.tile.broadcast(sc2d, sc_ub, axis=1)  # per-row constant (pool scale)
            T.tile.mul(s_ub, s_ub, sc2d)
            T.tile.relu(s_ub, s_ub)
            T.tile.broadcast(w2d, w_ub, axis=0)  # per-column constant (head weight)
            T.tile.mul(s_ub, s_ub, w2d)
            T.reduce_sum(s_ub, v_ub, dim=-1)

            # 5. Validity mask (0 / -inf) -> out-of-range lanes become -inf.
            T.copy(mmask[cid, v0], mask_ub)
            T.tile.add(v_ub, v_ub, mask_ub)
            T.copy(v_ub, logits[row, t0 + v0 : t0 + v0 + half])

    return main


def _validate_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    max_num_pools: int,
    num_heads: int,
    head_dim: int,
) -> None:
    """Metadata-only validation (shapes/dtypes/device/contiguity)."""
    if num_heads != GLM53_INDEX_HEADS:
        raise ValueError(f"GLM-5.3 indexer requires num_heads={GLM53_INDEX_HEADS}, got {num_heads}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 indexer requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if max_num_pools < 0:
        raise ValueError(f"max_num_pools must be nonnegative, got {max_num_pools}")
    if query.ndim != 3:
        raise ValueError(f"query must have rank 3, got shape {tuple(query.shape)}")
    if tuple(query.shape[1:]) != (num_heads, head_dim):
        raise ValueError(f"query must have shape [num_rows, {num_heads}, {head_dim}], got {tuple(query.shape)}")
    if k_cache.ndim != 3 or k_cache.shape[2] != head_dim:
        raise ValueError(f"k_cache must have shape [num_blocks, page_size, {head_dim}], got {tuple(k_cache.shape)}")
    if k_cache.shape[1] != GLM53_PAGE_SIZE:
        raise ValueError(f"k_cache must use page_size={GLM53_PAGE_SIZE}, got {k_cache.shape[1]}")
    if tuple(scale_cache.shape) != tuple(k_cache.shape[:2]):
        raise ValueError(
            "scale_cache must match the first two k_cache dimensions; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    num_rows = query.shape[0]
    if tuple(weights.shape) != (num_rows, num_heads):
        raise ValueError(f"weights must have shape {(num_rows, num_heads)}, got {tuple(weights.shape)}")
    if pool_page_table.ndim != 2 or pool_page_table.shape[0] <= 0 or pool_page_table.shape[1] <= 0:
        raise ValueError(f"pool_page_table must be a nonempty rank-2 tensor, got {tuple(pool_page_table.shape)}")
    for name, tensor in (
        ("page_table_rows", page_table_rows),
        ("pool_starts", pool_starts),
        ("pool_ends", pool_ends),
    ):
        if tuple(tensor.shape) != (num_rows,):
            raise ValueError(f"{name} must have shape {(num_rows,)}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")

    if query.dtype != torch.bfloat16:
        raise TypeError(f"query must be torch.bfloat16 (E4M3 values load losslessly), got {query.dtype}")
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be torch.uint8 (E4M3 bytes), got {k_cache.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"weights must be torch.float32, got {weights.dtype}")
    if pool_page_table.dtype != torch.int32:
        raise TypeError(f"pool_page_table must be torch.int32, got {pool_page_table.dtype}")

    tensors = (query, k_cache, scale_cache, weights, pool_page_table, page_table_rows, pool_starts, pool_ends)
    if any(t.device.type != "npu" for t in tensors):
        raise ValueError("all tensors must be on an Ascend NPU device")
    if any(t.device != query.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")


def glm53_kpool_fp8_mqa_logits(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    *,
    max_num_pools: int,
    num_heads: int = GLM53_INDEX_HEADS,
    head_dim: int = GLM53_HEAD_DIM,
) -> torch.Tensor:
    """Compute FP32 pool logits from caller-owned paged FP8 caches.

    Returns a ``[num_rows, max_num_pools]`` FP32 tensor where pools outside
    ``[pool_starts[r], pool_ends[r])`` or backed by an invalid page are ``-inf``.
    ``max_num_pools`` is required (the reference derives it with a device
    reduction; this port keeps the host path free of device reads).  The
    returned tensor is a view of the padded kernel output, so it is contiguous
    exactly when ``max_num_pools`` is a multiple of the 64-pool tile.
    """
    _validate_inputs(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
        max_num_pools,
        num_heads,
        head_dim,
    )
    if query.shape[0] == 0 or max_num_pools == 0:
        return torch.empty((query.shape[0], max_num_pools), dtype=torch.float32, device=query.device)

    num_rows = query.shape[0]
    num_tiles = (max_num_pools + POOL_TILE - 1) // POOL_TILE

    mask_kernel = glm53_kpool_fp8_mqa_logits_mask_kernel(
        num_rows,
        num_tiles,
        max_num_pools,
        k_cache.shape[0],
        pool_page_table.shape[0],
        pool_page_table.shape[1],
    )
    mmask = mask_kernel(pool_page_table, page_table_rows, pool_starts, pool_ends)

    kernel = glm53_kpool_fp8_mqa_logits_kernel(
        num_rows,
        max_num_pools,
        k_cache.shape[0],
        pool_page_table.shape[0],
        pool_page_table.shape[1],
    )
    out = kernel(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
        mmask,
    )
    return out[:, :max_num_pools]


def glm53_kpool_fp8_mqa_logits_reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    weights: torch.Tensor,
    pool_page_table: torch.Tensor,
    page_table_rows: torch.Tensor,
    pool_starts: torch.Tensor,
    pool_ends: torch.Tensor,
    *,
    max_num_pools: int,
) -> torch.Tensor:
    """PyTorch reference (CPU): E4M3 bytes decode exactly, -inf outside the range."""
    output = torch.full(
        (query.shape[0], max_num_pools),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    page_size = k_cache.shape[1]
    for row in range(query.shape[0]):
        start = int(pool_starts[row])
        end = int(pool_ends[row])
        if start == end:
            continue
        logical = torch.arange(start, end, dtype=torch.long, device=query.device)
        blocks = pool_page_table[int(page_table_rows[row]), logical // page_size].long()
        offsets = logical % page_size
        # Index the uint8 view first (CPU advanced indexing has no FP8 kernel).
        keys = k_cache[blocks, offsets].view(torch.float8_e4m3fn).float()
        scales = scale_cache[blocks, offsets].float()
        per_head = torch.einsum("nd,hd->nh", keys, query[row].float())
        output[row, start:end] = (per_head.mul(scales[:, None]).clamp_min(0.0) * weights[row][None, :]).sum(dim=1)
    return output


def test() -> None:
    """Single small smoke case: 2 rows, 2-tile rows, full ranges."""
    torch.manual_seed(20261020)
    num_rows, max_num_pools = 2, 128
    num_blocks = 4
    query = (torch.randn(num_rows, GLM53_INDEX_HEADS, GLM53_HEAD_DIM) * 0.5).to(torch.bfloat16)
    weights = torch.rand(num_rows, GLM53_INDEX_HEADS) + 0.1
    # Cache bytes: E4M3-encoded random values in [0, 448), non-negative scales.
    raw = torch.rand(num_blocks, GLM53_PAGE_SIZE, GLM53_HEAD_DIM) * E4M3_MAX
    k_cache = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_cache = torch.rand(num_blocks, GLM53_PAGE_SIZE) + 0.5
    page_table_rows = torch.tensor([0, 1], dtype=torch.int32)
    pool_page_table = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32)
    pool_starts = torch.zeros(num_rows, dtype=torch.int32)
    pool_ends = torch.full((num_rows,), max_num_pools, dtype=torch.int32)

    expected = glm53_kpool_fp8_mqa_logits_reference(
        query,
        k_cache,
        scale_cache,
        weights,
        pool_page_table,
        page_table_rows,
        pool_starts,
        pool_ends,
        max_num_pools=max_num_pools,
    )

    actual = glm53_kpool_fp8_mqa_logits(
        query.npu(),
        k_cache.npu(),
        scale_cache.npu(),
        weights.npu(),
        pool_page_table.npu(),
        page_table_rows.npu(),
        pool_starts.npu(),
        pool_ends.npu(),
        max_num_pools=max_num_pools,
    )
    torch.npu.synchronize()

    actual_cpu = actual.cpu()
    assert torch.equal(torch.isinf(actual_cpu) & (actual_cpu < 0), torch.isinf(expected) & (expected < 0))
    torch.testing.assert_close(actual_cpu, expected, rtol=2**-10, atol=2**-16)
    print(f"fp8_mqa_logits: logits{tuple(actual.shape)} match reference (max diff {(actual_cpu - expected).abs().max().item():.3e})")
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
