"""GLM-5.3 k-pool compression (compress) for Ascend NPU using TileLang.

This operator is the fused compressor of the GLM-5.3-Flash sparse-attention
indexer. For every pool of `pool_size` tokens it applies a per-dimension
softmax over `slot_score + ape`, pools the BF16 K vectors with those
probabilities, rounds the pooled vector through BF16, applies a normalized
Hadamard-128 rotation, quantizes with one FP32 power-of-two absmax scale per
vector (optionally plain absmax scale) and scatters the result into a
caller-owned paged cache. The math matches the CUDA reference
(examples/kpool in tile-ai/tilelang) operation for operation.

Cache contract (Ascend port): the reference stores E4M3 values as FP8 tensors.
Ascend A2/A3 has no FP8 data path, so this port stores the very same E4M3 byte
patterns in a ``uint8`` ``k_cache`` and keeps ``scale_cache`` as FP32; the byte
layout is identical to a split of the production interleaved uint8 allocation.

Host-side preparation and write-back (numerically identical to the reference):

  - ``ape`` is folded into ``slot_score`` (FP32 add, same expression as the
    reference), and both tensors are repacked slot-major ([P, 4, D] -> [4P, D])
    so that every slot slice is a contiguous device copy;
  - rows are padded to a multiple of the 32-pool device tile; padded rows carry
    ``write_mask=0`` and sentinel locations and are never written;
  - the device kernel writes the E4M3 rows to the paged ``k_cache`` directly
    and emits the per-pool scales to a contiguous scratch; the host adapter
    scatters those scales into ``scale_cache`` with one device indexing op.
    4-byte scalar GM stores are not cache-line coherent across cores on this
    target (automatic DCCI insertion for scalar accesses is disabled by the
    toolchain), so a device-side scalar scatter can drop updates; the kernel
    only performs cache-line-sized (128-byte row) copies.

Device-side notes: E4M3 conversion is emulated with the Ascend cast whitelist
plus integer bit manipulation (round-half-even, subnormal handling, saturation
to 448); the Hadamard butterfly runs on contiguous half-slices of a dim-major
tile (bit-exact against the reference); scatter writes resolve the physical
page address from ``loc`` at run time.

Mode: Developer pass_configs (auto sync / memory planning) with ``T.alloc_ub``
buffers and ``T.tile`` vector primitives; pure Vector, one tile per block.
"""

import math

import tilelang
import tilelang.language as T
import torch

GLM53_HEAD_DIM = 128
GLM53_POOL_SIZE = 4
_HADAMARD128_SCALE = 1.0 / math.sqrt(GLM53_HEAD_DIM)
_FP8_MAX = 448.0

# Pools per device tile. 32 is the largest tile that fits the 192KB UB budget
# of this structure (P=64 fails memory planning), and 32 also keeps the
# Hadamard slicing butterfly inside its fast compile regime (P=256 takes >44min).
_POOL_TILE = 32

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def glm53_kpool_compress_kernel(
    num_tiles: int,
    num_blocks: int,
    page_size: int,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
    round_scale: bool = True,
):
    """Build the compress kernel for ``num_tiles`` device tiles of _POOL_TILE pools.

    Inputs (slot-major, padded):
      K     [pool_size * num_tiles * _POOL_TILE, head_dim] bf16
      S     [pool_size * num_tiles * _POOL_TILE, head_dim] f32  (scores + ape)
      LOC   [num_tiles * _POOL_TILE] i32   flat page-major physical slots
      MASK  [num_tiles * _POOL_TILE] u8    1 = write this row
    In/out cache and scale scratch:
      K_CACHE     [num_blocks, page_size, head_dim] u8   (E4M3 byte patterns)
      SC_SCRATCH  [num_tiles * _POOL_TILE] f32  per-pool scales, tile-contiguous;
                  the host adapter scatters them to `scale_cache` (see module
                  docstring: scalar GM stores are not cache-line coherent
                  across cores on this target, so the 4-byte scale scatter
                  cannot run on the device scalar pipe).
    """
    pools = num_tiles * _POOL_TILE
    rows = pool_size * pools
    dim = head_dim

    @T.prim_func
    def main(
        K: T.Tensor((rows, dim), "bfloat16"),
        S: T.Tensor((rows, dim), "float32"),
        LOC: T.Tensor((pools,), "int32"),
        MASK: T.Tensor((pools,), "uint8"),
        K_CACHE: T.Tensor((num_blocks, page_size, dim), "uint8"),
        SC_SCRATCH: T.Tensor((pools,), "float32"),
    ):
        with T.Kernel(num_tiles, is_npu=True) as (cid, vid):
            if vid == 0:
                base = cid * _POOL_TILE

                sf = T.alloc_ub((_POOL_TILE, dim), "float32")
                m = T.alloc_ub((_POOL_TILE, dim), "float32")
                denom = T.alloc_ub((_POOL_TILE, dim), "float32")
                pooled = T.alloc_ub((_POOL_TILE, dim), "float32")
                kf = T.alloc_ub((_POOL_TILE, dim), "float32")
                k16 = T.alloc_ub((_POOL_TILE, dim), "bfloat16")

                # 1. softmax pooling over slots (slot-order accumulation)
                T.copy(S[0 * pools + base : 0 * pools + base + _POOL_TILE, :], m)
                for s in range(1, pool_size):
                    T.copy(S[s * pools + base : s * pools + base + _POOL_TILE, :], sf)
                    T.tile.max(m, m, sf)

                T.tile.fill(denom, 0.0)
                T.tile.fill(pooled, 0.0)
                for s in range(pool_size):
                    T.copy(S[s * pools + base : s * pools + base + _POOL_TILE, :], sf)
                    T.tile.sub(sf, sf, m)
                    T.tile.exp(sf, sf)
                    T.tile.add(denom, denom, sf)
                    T.copy(K[s * pools + base : s * pools + base + _POOL_TILE, :], k16)
                    T.tile.cast(kf, k16, "CAST_NONE", _POOL_TILE * dim)
                    T.tile.mul(kf, kf, sf)
                    T.tile.add(pooled, pooled, kf)
                T.tile.div(pooled, pooled, denom)

                # 2. BF16 boundary of the pooled vector
                T.tile.cast(k16, pooled, "CAST_RINT", _POOL_TILE * dim)
                T.tile.cast(pooled, k16, "CAST_NONE", _POOL_TILE * dim)

                # 3. normalized Sylvester Hadamard-128 on a dim-major tile
                dm = T.alloc_ub((dim, _POOL_TILE), "float32")
                hdm = T.alloc_ub((dim, _POOL_TILE), "float32")
                T.tile.transpose(dm, pooled)

                for i in range(0, 64):
                    T.tile.add(hdm[i * 2 : i * 2 + 1, :], dm[i * 2 : i * 2 + 1, :], dm[i * 2 + 1 : i * 2 + 2, :])
                    T.tile.sub(hdm[i * 2 + 1 : i * 2 + 2, :], dm[i * 2 : i * 2 + 1, :], dm[i * 2 + 1 : i * 2 + 2, :])
                for i in range(0, 32):
                    T.tile.add(dm[i * 4 : i * 4 + 2, :], hdm[i * 4 : i * 4 + 2, :], hdm[i * 4 + 2 : i * 4 + 4, :])
                    T.tile.sub(dm[i * 4 + 2 : i * 4 + 4, :], hdm[i * 4 : i * 4 + 2, :], hdm[i * 4 + 2 : i * 4 + 4, :])
                for i in range(0, 16):
                    T.tile.add(hdm[i * 8 : i * 8 + 4, :], dm[i * 8 : i * 8 + 4, :], dm[i * 8 + 4 : i * 8 + 8, :])
                    T.tile.sub(hdm[i * 8 + 4 : i * 8 + 8, :], dm[i * 8 : i * 8 + 4, :], dm[i * 8 + 4 : i * 8 + 8, :])
                for i in range(0, 8):
                    T.tile.add(dm[i * 16 : i * 16 + 8, :], hdm[i * 16 : i * 16 + 8, :], hdm[i * 16 + 8 : i * 16 + 16, :])
                    T.tile.sub(dm[i * 16 + 8 : i * 16 + 16, :], hdm[i * 16 : i * 16 + 8, :], hdm[i * 16 + 8 : i * 16 + 16, :])
                for i in range(0, 4):
                    T.tile.add(hdm[i * 32 : i * 32 + 16, :], dm[i * 32 : i * 32 + 16, :], dm[i * 32 + 16 : i * 32 + 32, :])
                    T.tile.sub(hdm[i * 32 + 16 : i * 32 + 32, :], dm[i * 32 : i * 32 + 16, :], dm[i * 32 + 16 : i * 32 + 32, :])
                for i in range(0, 2):
                    T.tile.add(dm[i * 64 : i * 64 + 32, :], hdm[i * 64 : i * 64 + 32, :], hdm[i * 64 + 32 : i * 64 + 64, :])
                    T.tile.sub(dm[i * 64 + 32 : i * 64 + 64, :], hdm[i * 64 : i * 64 + 32, :], hdm[i * 64 + 32 : i * 64 + 64, :])
                T.tile.add(hdm[0:64, :], dm[0:64, :], dm[64:128, :])
                T.tile.sub(hdm[64:128, :], dm[0:64, :], dm[64:128, :])

                T.tile.mul(hdm, hdm, _HADAMARD128_SCALE)
                T.tile.cast(k16, hdm, "CAST_RINT", dim * _POOL_TILE)
                T.tile.cast(hdm, k16, "CAST_NONE", dim * _POOL_TILE)

                # 4. back to pool-major, absmax and scale
                rot = T.alloc_ub((_POOL_TILE, dim), "float32")
                absv = T.alloc_ub((_POOL_TILE, dim), "float32")
                T.tile.transpose(rot, hdm)
                T.tile.abs(absv, rot)
                amax = T.alloc_ub((_POOL_TILE,), "float32")
                T.reduce_max(absv, amax, dim=-1)
                T.tile.max(amax, amax, 1e-4)

                scale = T.alloc_ub((_POOL_TILE,), "float32")
                q = T.alloc_ub((_POOL_TILE, dim), "float32")
                if round_scale:
                    # scale = 2^ceil(log2(amax / 448)) through the fp32 bit field
                    z = T.alloc_ub((_POOL_TILE,), "float32")
                    T.tile.div(z, amax, _FP8_MAX)
                    zb = T.alloc_ub((_POOL_TILE,), "int32")
                    T.reinterpretcast(zb, z, "int32_t")
                    ef = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.bitwise_rshift(ef, zb, 23)
                    mant = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.bitwise_lshift(mant, ef, 23)
                    T.tile.sub(mant, zb, mant)
                    inc = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.add(inc, mant, 8388607)
                    T.tile.bitwise_rshift(inc, inc, 23)
                    ce = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.sub(ce, ef, 127)
                    T.tile.add(ce, ce, inc)
                    sb = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.add(sb, ce, 127)
                    T.tile.bitwise_lshift(sb, sb, 23)
                    T.reinterpretcast(scale, sb, "float")

                    # inv = 2^-ce (exact bit construction, no reciprocal error)
                    sbi = T.alloc_ub((_POOL_TILE,), "int32")
                    T.tile.bitwise_lshift(sbi, ce, 23)
                    T.tile.mul(sbi, sbi, -1)
                    T.tile.add(sbi, sbi, 1065353216)

                    inv_bits = T.alloc_ub((_POOL_TILE, dim), "int32")
                    inv_full = T.alloc_ub((_POOL_TILE, dim), "float32")
                    T.tile.broadcast(inv_bits, sbi)
                    T.reinterpretcast(inv_full, inv_bits, "float")
                    T.tile.mul(q, rot, inv_full)
                else:
                    T.tile.div(scale, amax, _FP8_MAX)
                    scale_full = T.alloc_ub((_POOL_TILE, dim), "float32")
                    T.tile.broadcast(scale_full, scale)
                    T.tile.div(q, rot, scale_full)

                # 5. clamp and E4M3 byte encoding (round-half-even, subnormals)
                absx = T.alloc_ub((_POOL_TILE, dim), "float32")
                T.tile.abs(absx, q)
                bits = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.reinterpretcast(bits, absx, "int32_t")
                y20 = T.alloc_ub((_POOL_TILE, dim), "int32")
                y21 = T.alloc_ub((_POOL_TILE, dim), "int32")
                bit20 = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.tile.bitwise_rshift(y20, bits, 20)
                T.tile.bitwise_rshift(y21, bits, 21)
                T.tile.bitwise_lshift(y21, y21, 1)
                T.tile.sub(bit20, y20, y21)
                nb = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.tile.add(nb, bits, 524287)
                T.tile.add(nb, nb, bit20)
                T.tile.bitwise_rshift(nb, nb, 20)
                T.tile.sub(nb, nb, 960)
                T.tile.min(nb, nb, 126)

                t = T.alloc_ub((_POOL_TILE, dim), "float32")
                sub = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.tile.mul(t, absx, 512.0)
                T.tile.cast(sub, t, "CAST_RINT", _POOL_TILE * dim)
                T.tile.max(sub, sub, 0)
                T.tile.min(sub, sub, 8)

                mask = T.alloc_ub((_POOL_TILE, dim), "uint8")
                nbf = T.alloc_ub((_POOL_TILE, dim), "float32")
                subf = T.alloc_ub((_POOL_TILE, dim), "float32")
                bm = T.alloc_ub((_POOL_TILE, dim), "float32")
                T.tile.compare(mask, absx, 0.015625, "LT")
                T.tile.cast(nbf, nb, "CAST_RINT", _POOL_TILE * dim)
                T.tile.cast(subf, sub, "CAST_RINT", _POOL_TILE * dim)
                T.tile.select(bm, mask, subf, nbf, "VSEL_TENSOR_TENSOR_MODE")

                qb = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.reinterpretcast(qb, q, "int32_t")
                sgn = T.alloc_ub((_POOL_TILE, dim), "int32")
                T.tile.bitwise_rshift(sgn, qb, 31)
                T.tile.mul(sgn, sgn, -1)
                sfx = T.alloc_ub((_POOL_TILE, dim), "float32")
                T.tile.cast(sfx, sgn, "CAST_RINT", _POOL_TILE * dim)
                T.tile.mul(sfx, sfx, 128.0)
                val = T.alloc_ub((_POOL_TILE, dim), "float32")
                T.tile.add(val, bm, sfx)

                h8 = T.alloc_ub((_POOL_TILE, dim), "float16")
                ob = T.alloc_ub((_POOL_TILE, dim), "uint8")
                T.tile.cast(h8, val, "CAST_NONE", _POOL_TILE * dim)
                T.tile.cast(ob, h8, "CAST_NONE", _POOL_TILE * dim)

                # 6. scatter the E4M3 rows into the paged cache (write_mask and
                # block-bounds guarded; rows are 128-byte cache-line sized and
                # exclusively owned, so the copies are cross-core coherent).
                with T.Scope("V"):
                    for i in T.serial(_POOL_TILE):
                        if MASK[base + i] != 0:
                            lv = T.alloc_var("int32")
                            lv = LOC[base + i]
                            blk = T.alloc_var("int32")
                            blk = lv // page_size
                            off = T.alloc_var("int32")
                            off = lv % page_size
                            if blk >= 0 and blk < num_blocks:
                                T.copy(ob[i : i + 1, :], K_CACHE[blk, off, 0:dim])

                # Per-pool scales to the contiguous scratch. Kept after the
                # scalar-guarded loop so the pipeline barriers it emits order
                # this copy after the UB writes.
                T.copy(scale, SC_SCRATCH[cid * _POOL_TILE : cid * _POOL_TILE + _POOL_TILE])

    return main


def _ordered_codes(values_u8: torch.Tensor) -> torch.Tensor:
    """Map E4M3 bytes to monotonic integer codes (ties at +-0 stay adjacent)."""
    bits = values_u8.to(torch.int16)
    magnitude = bits & 0x7F
    return torch.where((bits & 0x80) != 0, 0x80 - magnitude, 0x80 + magnitude)


def _assert_fp8_within_one_ulp(actual_u8: torch.Tensor, expected_u8: torch.Tensor) -> None:
    """Allow only adjacent finite E4M3 encodings between the two math paths."""
    assert actual_u8.shape == expected_u8.shape
    finite = (actual_u8 & 0x7F) != 0x7F
    assert finite.all() and ((expected_u8 & 0x7F) != 0x7F).all()
    ulp_distance = (_ordered_codes(actual_u8) - _ordered_codes(expected_u8)).abs()
    assert ulp_distance.max().item() <= 1, f"max E4M3 ULP distance was {int(ulp_distance.max().item())}"


def _validate_kpool_inputs(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Validate the public contract; returns the effective write mask."""
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if head_dim != GLM53_HEAD_DIM:
        raise ValueError(f"GLM-5.3 k-pool requires head_dim={GLM53_HEAD_DIM}, got {head_dim}")
    if slot_k.ndim != 3:
        raise ValueError(f"slot_k must have rank 3, got shape {tuple(slot_k.shape)}")
    expected_slot_shape = (slot_k.shape[0], pool_size, head_dim)
    if tuple(slot_k.shape) != expected_slot_shape:
        raise ValueError(f"slot_k must have shape {expected_slot_shape}, got {tuple(slot_k.shape)}")
    if tuple(slot_score.shape) != expected_slot_shape:
        raise ValueError(f"slot_score must have shape {expected_slot_shape}, got {tuple(slot_score.shape)}")
    if tuple(ape.shape) != (pool_size, head_dim):
        raise ValueError(f"ape must have shape {(pool_size, head_dim)}, got {tuple(ape.shape)}")
    if tuple(loc.shape) != (slot_k.shape[0],):
        raise ValueError(f"loc must have shape {(slot_k.shape[0],)}, got {tuple(loc.shape)}")

    if slot_k.dtype != torch.bfloat16:
        raise TypeError(f"slot_k must be torch.bfloat16, got {slot_k.dtype}")
    if slot_score.dtype != torch.bfloat16:
        raise TypeError(f"slot_score must be torch.bfloat16, got {slot_score.dtype}")
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be torch.float32, got {ape.dtype}")
    if loc.dtype != torch.int64:
        raise TypeError(f"loc must be torch.int64, got {loc.dtype}")
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be torch.uint8 (E4M3 bytes), got {k_cache.dtype}")
    if scale_cache.dtype != torch.float32:
        raise TypeError(f"scale_cache must be torch.float32, got {scale_cache.dtype}")

    if k_cache.ndim != 3:
        raise ValueError(f"k_cache must have rank 3, got shape {tuple(k_cache.shape)}")
    if scale_cache.ndim != 2:
        raise ValueError(f"scale_cache must have rank 2, got shape {tuple(scale_cache.shape)}")
    if tuple(k_cache.shape[1:]) != (scale_cache.shape[1], head_dim):
        raise ValueError(
            "k_cache must have shape [num_blocks, page_size, head_dim] and agree with scale_cache; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    if scale_cache.shape[0] != k_cache.shape[0]:
        raise ValueError(
            "scale_cache must have shape [num_blocks, page_size] and agree with k_cache; "
            f"got k_cache={tuple(k_cache.shape)}, scale_cache={tuple(scale_cache.shape)}"
        )
    if k_cache.shape[0] <= 0 or k_cache.shape[1] <= 0:
        raise ValueError(f"cache dimensions must be positive, got {tuple(k_cache.shape)}")

    tensors = (slot_k, slot_score, ape, loc, k_cache, scale_cache)
    if any(t.device.type != "npu" for t in tensors):
        raise ValueError("all tensors must be on an Ascend NPU device")
    if any(t.device != slot_k.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")

    if write_mask is None:
        write_mask = torch.ones((slot_k.shape[0],), dtype=torch.bool, device=slot_k.device)
    else:
        if tuple(write_mask.shape) != (slot_k.shape[0],):
            raise ValueError(f"write_mask must have shape {(slot_k.shape[0],)}, got {tuple(write_mask.shape)}")
        if write_mask.dtype != torch.bool:
            raise TypeError(f"write_mask must be torch.bool, got {write_mask.dtype}")
        if write_mask.device != slot_k.device:
            raise ValueError("write_mask must be on the same device as slot_k")
        if not write_mask.is_contiguous():
            raise ValueError("write_mask must be contiguous")

    active_locs = loc[write_mask]
    if active_locs.numel() != 0:
        cache_slots = k_cache.shape[0] * k_cache.shape[1]
        min_loc = int(active_locs.min().item())
        max_loc = int(active_locs.max().item())
        if min_loc < 0 or max_loc >= cache_slots:
            raise ValueError(f"active loc values must be in [0, {cache_slots}), got min={min_loc}, max={max_loc}")
        if torch.unique(active_locs).numel() != active_locs.numel():
            raise ValueError("active loc values must be unique to avoid concurrent cache writes")

    return write_mask


def glm53_kpool_compress_and_write_cache(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Compress pools and write caller-owned, separately typed caches in place.

    ``loc`` is a flat physical slot in page-major order. Masked rows may carry
    any sentinel location; only active locations are range- and uniqueness-
    checked, and masked rows are never dereferenced on device. Validation runs
    on the host before kernel launch. ``k_cache`` holds E4M3 byte patterns in a
    uint8 tensor (Ascend port of the reference FP8 contract).
    """
    write_mask = _validate_kpool_inputs(
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        k_cache,
        scale_cache,
        pool_size,
        head_dim,
    )
    num_pools = slot_k.shape[0]
    if num_pools == 0:
        return

    num_tiles = (num_pools + _POOL_TILE - 1) // _POOL_TILE
    padded_pools = num_tiles * _POOL_TILE

    # Host preparation: fold ape into the scores (fp32, reference-identical),
    # repack slot-major and pad each slot block to the device tile with
    # inactive rows (mirrors the kernel's [slot][padded_pool][dim] layout).
    scores = slot_score.float() + ape.unsqueeze(0)
    s2 = scores.permute(1, 0, 2)
    k2 = slot_k.permute(1, 0, 2)
    if padded_pools != num_pools:
        pad_pools = padded_pools - num_pools
        s2 = torch.cat([s2, s2.new_zeros(pool_size, pad_pools, head_dim)], dim=1)
        k2 = torch.cat([k2, k2.new_zeros(pool_size, pad_pools, head_dim)], dim=1)
    s2 = s2.reshape(pool_size * padded_pools, head_dim).contiguous()
    k2 = k2.reshape(pool_size * padded_pools, head_dim).contiguous()
    loc32 = loc.to(torch.int32)
    mask8 = write_mask.to(torch.uint8)
    if padded_pools != num_pools:
        pad_pools = padded_pools - num_pools
        loc32 = torch.cat([loc32, loc32.new_full((pad_pools,), -1)], dim=0)
        mask8 = torch.cat([mask8, mask8.new_zeros(pad_pools)], dim=0)

    kernel = glm53_kpool_compress_kernel(
        num_tiles,
        k_cache.shape[0],
        k_cache.shape[1],
        pool_size=pool_size,
        head_dim=head_dim,
        round_scale=round_scale,
    )
    sc_scratch = torch.empty(padded_pools, dtype=torch.float32, device=slot_k.device)
    kernel(k2, s2, loc32.contiguous(), mask8.contiguous(), k_cache, sc_scratch)

    # Scatter the per-pool scales to their physical slots. 4-byte scalar GM
    # stores are not cache-line coherent across cores on this target (the
    # compiler disables automatic DCCI for scalar accesses), so the scatter
    # runs as one device-side indexing op instead of kernel scalar stores.
    active = write_mask
    scale_cache.view(-1)[loc[active]] = sc_scratch[:num_pools][active]


def hadamard128_reference(x: torch.Tensor) -> torch.Tensor:
    """Matrix-free normalized Sylvester Hadamard-128 reference (same stage order)."""
    if x.shape[-1] != GLM53_HEAD_DIM:
        raise ValueError(f"expected last dimension {GLM53_HEAD_DIM}, got {x.shape[-1]}")
    result = x.float().clone()
    half = 1
    while half < GLM53_HEAD_DIM:
        pairs = result.reshape(*result.shape[:-1], -1, 2, half)
        left = pairs[..., 0, :].clone()
        right = pairs[..., 1, :].clone()
        pairs[..., 0, :] = left + right
        pairs[..., 1, :] = left - right
        half *= 2
    return result * _HADAMARD128_SCALE


def glm53_kpool_reference(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    *,
    round_scale: bool = True,
):
    """Reference compress returning E4M3 bytes (uint8) and FP32 scales (CPU)."""
    scores = slot_score.float() + ape.unsqueeze(0)
    max_score = scores.max(dim=1).values
    denom = torch.zeros_like(max_score)
    pooled = torch.zeros_like(max_score)
    for slot in range(slot_k.shape[1]):
        probability = torch.exp(scores[:, slot] - max_score)
        denom += probability
        pooled += slot_k[:, slot].float() * probability
    pooled = (pooled / denom).to(torch.bfloat16).float()
    rotated = hadamard128_reference(pooled).to(torch.bfloat16).float()

    absmax = rotated.abs().amax(dim=1).clamp(min=1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / _FP8_MAX)))
    else:
        scale = absmax / _FP8_MAX
    quantized = (rotated / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.view(torch.uint8), scale


def test() -> None:
    """Single small smoke case: 8 pools into a one-block, 64-slot paged cache."""
    torch.manual_seed(20260918)
    num_pools, num_blocks, page_size = 8, 1, 64
    slot_k = torch.randn(num_pools, GLM53_POOL_SIZE, GLM53_HEAD_DIM).to(torch.bfloat16)
    slot_score = torch.randn(num_pools, GLM53_POOL_SIZE, GLM53_HEAD_DIM).to(torch.bfloat16)
    ape = torch.randn(GLM53_POOL_SIZE, GLM53_HEAD_DIM)
    loc = torch.arange(num_pools, dtype=torch.int64)

    expected_k, expected_scale = glm53_kpool_reference(slot_k, slot_score, ape)

    k_cache = torch.full((num_blocks, page_size, GLM53_HEAD_DIM), 0x38, dtype=torch.uint8).npu()
    scale_cache = torch.full((num_blocks, page_size), -7.0, dtype=torch.float32).npu()
    glm53_kpool_compress_and_write_cache(
        slot_k.npu(),
        slot_score.npu(),
        ape.npu(),
        loc.npu(),
        k_cache,
        scale_cache,
    )
    torch.npu.synchronize()

    flat_k = k_cache.view(-1, GLM53_HEAD_DIM).cpu()
    flat_scale = scale_cache.view(-1).cpu()
    _assert_fp8_within_one_ulp(flat_k[loc], expected_k)
    torch.testing.assert_close(flat_scale[loc], expected_scale, rtol=2e-3, atol=1e-6)

    exact = int((flat_k[loc] == expected_k).all(dim=1).sum().item())
    print(f"compress: {exact}/{num_pools} rows byte-identical, all within one E4M3 ULP")
    print("Kernel Output Match!")


if __name__ == "__main__":
    test()
