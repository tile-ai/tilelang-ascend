"""GLM-5.3 k-pool compression (compress) for Ascend NPU using TileLang.

This operator is the fused compressor of the GLM-5.3-Flash sparse-attention
indexer. For every pool of `pool_size` tokens it applies a per-dimension
softmax over `slot_score + ape`, pools the BF16 K vectors with those
probabilities, rounds the pooled vector through BF16, applies a normalized
Hadamard-128 rotation, quantizes with one FP32 power-of-two absmax scale per
vector (optionally plain absmax scale) and scatters the result into a
caller-owned paged cache. The math matches the reference implementation
(examples/kpool in tile-ai/tilelang) operation for operation.

Cache contract (Ascend port): the reference stores E4M3 values as FP8 tensors.
A2/A3 has no FP8 data path, so this port computes the E4M3 codes with integer
bit arithmetic (round-half-even, subnormals, saturation to 448 -- no FP8 type
and no numeric approximation) and stores the resulting byte patterns in a
``uint8`` ``k_cache`` with a separate FP32 ``scale_cache``; the byte layout is
identical to a split of the production interleaved uint8 allocation.

Host-side preparation and write-back (numerically identical to the reference):

  - ``ape`` is folded into ``slot_score`` (FP32 add, same expression as the
    reference), and both tensors are repacked slot-major ([P, 4, D] -> [4P, D])
    so that every slot slice is a contiguous device copy;
  - rows are padded to a multiple of the 64-pool device block; padded rows are
    never scattered into the caches;
  - the device kernel writes the E4M3 rows and the per-pool scales to two
    contiguous scratches; the host adapter scatters both into the paged caches
    with one device indexing op each.  Two reasons: 4-byte scalar GM stores are
    not cache-line coherent across cores on this target (the toolchain disables
    automatic DCCI insertion for scalar accesses), and a device-side per-row
    scatter loop is serialized by conservative PIPE_ALL barriers emitted around
    every scalar access (measured ~30% of kernel time).

Device-side notes: E4M3 encoding is exact integer bit arithmetic on the fp32
bit pattern (round-half-even, subnormal handling, saturation to 448); the
Hadamard butterfly runs on contiguous half-slices of a dim-major tile
(bit-exact against the reference).

Mode: Developer pass_configs (auto sync / memory planning) with ``T.alloc_ub``
buffers and ``T.tile`` vector primitives; pure Vector, one 32-pool tile per
AIV subcore (each block covers 64 pools: subcore 0 handles the first 32, subcore
1 the second 32).
"""

import math

import tilelang
import tilelang.language as T
import torch

GLM53_HEAD_DIM = 128
GLM53_POOL_SIZE = 4
HADAMARD128_SCALE = 1.0 / math.sqrt(GLM53_HEAD_DIM)

# E4M3 encoding constants. fp32 is sign(1) | exp(8, bias 127) | mantissa(23);
# E4M3 is sign(1) | exp(4, bias 7) | mantissa(3) with subnormal codes 0..7
# covering magnitudes 2^-9 .. 2^-6 and code 8 = 2^-6.
E4M3_MAX = 448.0  # largest finite E4M3 magnitude
E4M3_MAX_CODE = 126  # 0x7E: largest finite code (saturation target)
E4M3_MIN_NORMAL_CODE = 8  # smallest normal code; magnitudes below 2^-6 are subnormal
E4M3_SUBNORMAL_SCALE = 2.0**9  # subnormal code = round(|x| * 2^9)
E4M3_RNE_BIAS = (1 << 19) - 1  # round the 20 discarded mantissa bits to nearest
E4M3_CODE_BASE = (127 - 7) << 3  # 960: (fp32 exp - E4M3 bias) << 3 mantissa bits
FP32_MANT_MASK = (1 << 23) - 1  # 0x7FFFFF
FP32_ONE_BITS = 0x3F800000  # 1.0f, base of the 2^-exp inverse construction

# Pools per device tile. 32 is the largest tile that fits the 192KB UB budget
# of this structure (P=64 fails memory planning), and 32 also keeps the
# Hadamard slicing butterfly inside its fast compile regime (P=256 takes >44min).
POOL_TILE = 32
# Pools per device block: one POOL_TILE per AIV subcore.
BLOCK_POOLS = 2 * POOL_TILE

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def glm53_kpool_compress_kernel(
    num_tiles: int,
    pool_size: int = GLM53_POOL_SIZE,
    head_dim: int = GLM53_HEAD_DIM,
    round_scale: bool = True,
):
    """Build the compress kernel for ``num_tiles`` device blocks of BLOCK_POOLS pools.

    Inputs (slot-major, padded):
      K     [pool_size * num_tiles * BLOCK_POOLS, head_dim] bf16
      S     [pool_size * num_tiles * BLOCK_POOLS, head_dim] f32  (scores + ape)
    Outputs (row-contiguous scratches; the host adapter scatters them into the
    paged caches, see module docstring):
      ROW_SCRATCH  [num_tiles * BLOCK_POOLS, head_dim] u8   (E4M3 byte patterns)
      SC_SCRATCH   [num_tiles * BLOCK_POOLS] f32            (per-pool scales)
    """
    pools = num_tiles * BLOCK_POOLS
    rows = pool_size * pools
    dim = head_dim

    @T.prim_func
    def main(
        K: T.Tensor((rows, dim), "bfloat16"),
        S: T.Tensor((rows, dim), "float32"),
        ROW_SCRATCH: T.Tensor((pools, dim), "uint8"),
        SC_SCRATCH: T.Tensor((pools,), "float32"),
    ):
        with T.Kernel(num_tiles, is_npu=True) as (cid, vid):
            base = cid * BLOCK_POOLS + vid * POOL_TILE

            sf = T.alloc_ub((POOL_TILE, dim), "float32")
            m = T.alloc_ub((POOL_TILE, dim), "float32")
            denom = T.alloc_ub((POOL_TILE, dim), "float32")
            pooled = T.alloc_ub((POOL_TILE, dim), "float32")
            kf = T.alloc_ub((POOL_TILE, dim), "float32")
            k16 = T.alloc_ub((POOL_TILE, dim), "bfloat16")

            # 1. softmax pooling over slots (slot-order accumulation)
            T.copy(S[0 * pools + base : 0 * pools + base + POOL_TILE, :], m)
            for s in range(1, pool_size):
                T.copy(S[s * pools + base : s * pools + base + POOL_TILE, :], sf)
                T.tile.max(m, m, sf)

            T.tile.fill(denom, 0.0)
            T.tile.fill(pooled, 0.0)
            for s in range(pool_size):
                T.copy(S[s * pools + base : s * pools + base + POOL_TILE, :], sf)
                T.tile.sub(sf, sf, m)
                T.tile.exp(sf, sf)
                T.tile.add(denom, denom, sf)
                T.copy(K[s * pools + base : s * pools + base + POOL_TILE, :], k16)
                T.tile.cast(kf, k16, "CAST_NONE", POOL_TILE * dim)
                T.tile.mul(kf, kf, sf)
                T.tile.add(pooled, pooled, kf)
            T.tile.div(pooled, pooled, denom)

            # 2. BF16 boundary of the pooled vector
            T.tile.cast(k16, pooled, "CAST_RINT", POOL_TILE * dim)
            T.tile.cast(pooled, k16, "CAST_NONE", POOL_TILE * dim)

            # 3. normalized Sylvester Hadamard-128 on a dim-major tile
            dm = T.alloc_ub((dim, POOL_TILE), "float32")
            hdm = T.alloc_ub((dim, POOL_TILE), "float32")
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

            T.tile.mul(hdm, hdm, HADAMARD128_SCALE)
            T.tile.cast(k16, hdm, "CAST_RINT", dim * POOL_TILE)
            T.tile.cast(hdm, k16, "CAST_NONE", dim * POOL_TILE)

            # 4. back to pool-major, absmax and scale
            rot = T.alloc_ub((POOL_TILE, dim), "float32")
            absv = T.alloc_ub((POOL_TILE, dim), "float32")
            T.tile.transpose(rot, hdm)
            T.tile.abs(absv, rot)
            amax = T.alloc_ub((POOL_TILE,), "float32")
            T.reduce_max(absv, amax, dim=-1)
            T.tile.max(amax, amax, 1e-4)

            scale = T.alloc_ub((POOL_TILE,), "float32")
            q = T.alloc_ub((POOL_TILE, dim), "float32")
            if round_scale:
                # scale = 2^ceil(log2(amax / 448)) through the fp32 bit field
                z = T.alloc_ub((POOL_TILE,), "float32")
                T.tile.div(z, amax, E4M3_MAX)
                zb = T.alloc_ub((POOL_TILE,), "int32")
                T.reinterpretcast(zb, z, "int32_t")
                ef = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.bitwise_rshift(ef, zb, 23)
                mant = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.bitwise_lshift(mant, ef, 23)
                T.tile.sub(mant, zb, mant)
                inc = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.add(inc, mant, FP32_MANT_MASK)
                T.tile.bitwise_rshift(inc, inc, 23)
                ce = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.sub(ce, ef, 127)
                T.tile.add(ce, ce, inc)
                sb = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.add(sb, ce, 127)
                T.tile.bitwise_lshift(sb, sb, 23)
                T.reinterpretcast(scale, sb, "float")

                # inv = 2^-ce (exact bit construction, no reciprocal error)
                sbi = T.alloc_ub((POOL_TILE,), "int32")
                T.tile.bitwise_lshift(sbi, ce, 23)
                T.tile.mul(sbi, sbi, -1)
                T.tile.add(sbi, sbi, FP32_ONE_BITS)

                inv_bits = T.alloc_ub((POOL_TILE, dim), "int32")
                inv_full = T.alloc_ub((POOL_TILE, dim), "float32")
                T.tile.broadcast(inv_bits, sbi)
                T.reinterpretcast(inv_full, inv_bits, "float")
                T.tile.mul(q, rot, inv_full)
            else:
                T.tile.div(scale, amax, E4M3_MAX)
                scale_full = T.alloc_ub((POOL_TILE, dim), "float32")
                T.tile.broadcast(scale_full, scale)
                T.tile.div(q, rot, scale_full)

            # 5. clamp and E4M3 byte encoding (round-half-even, subnormals).
            # This is exact integer bit arithmetic on the fp32 bit pattern; no
            # FP8 type and no numeric approximation is involved.
            absx = T.alloc_ub((POOL_TILE, dim), "float32")
            T.tile.abs(absx, q)
            bits = T.alloc_ub((POOL_TILE, dim), "int32")
            T.reinterpretcast(bits, absx, "int32_t")
            # normal-format code: round the 20 discarded mantissa bits to
            # nearest-even (E4M3_RNE_BIAS plus the bit-20 tie breaker), keep
            # exp(4) | mantissa(3) and saturate at the largest finite code.
            y20 = T.alloc_ub((POOL_TILE, dim), "int32")
            y21 = T.alloc_ub((POOL_TILE, dim), "int32")
            bit20 = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.bitwise_rshift(y20, bits, 20)
            T.tile.bitwise_rshift(y21, bits, 21)
            T.tile.bitwise_lshift(y21, y21, 1)
            T.tile.sub(bit20, y20, y21)
            nb = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.add(nb, bits, E4M3_RNE_BIAS)
            T.tile.add(nb, nb, bit20)
            T.tile.bitwise_rshift(nb, nb, 20)
            T.tile.sub(nb, nb, E4M3_CODE_BASE)
            T.tile.min(nb, nb, E4M3_MAX_CODE)

            # subnormal code: round(|x| * 2^9), clamped to the 0..8 code range
            t = T.alloc_ub((POOL_TILE, dim), "float32")
            sub = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.mul(t, absx, E4M3_SUBNORMAL_SCALE)
            T.tile.cast(sub, t, "CAST_RINT", POOL_TILE * dim)
            T.tile.max(sub, sub, 0)
            T.tile.min(sub, sub, E4M3_MIN_NORMAL_CODE)

            # |x| < 2^-6 iff the subnormal code exceeds the normal one (both
            # round-to-nearest, monotone), so the selection is the integer
            # identity code = nb + max(sub - nb, 0).
            delta = T.alloc_ub((POOL_TILE, dim), "int32")
            code = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.sub(delta, sub, nb)
            T.tile.max(delta, delta, 0)
            T.tile.add(code, nb, delta)

            # the sign bit lands in 0x80. The AscendC cast whitelist has no
            # i32->u8 route, so the byte (an exact 0..254 integer) is widened
            # to f32 and then carried through f16 to u8 losslessly. The direct
            # i32->f16 cast is deliberately avoided: it is non-deterministically
            # wrong on this toolchain (minimal repro: one compiled kernel, same
            # input, wrong bytes on some runs only).
            qb = T.alloc_ub((POOL_TILE, dim), "int32")
            T.reinterpretcast(qb, q, "int32_t")
            neg = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.bitwise_rshift(neg, qb, 31)
            hbit = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.mul(hbit, neg, -128)
            byte_i32 = T.alloc_ub((POOL_TILE, dim), "int32")
            T.tile.add(byte_i32, code, hbit)

            byte_f32 = T.alloc_ub((POOL_TILE, dim), "float32")
            h8 = T.alloc_ub((POOL_TILE, dim), "float16")
            ob = T.alloc_ub((POOL_TILE, dim), "uint8")
            T.tile.cast(byte_f32, byte_i32, "CAST_RINT", POOL_TILE * dim)
            T.tile.cast(h8, byte_f32, "CAST_NONE", POOL_TILE * dim)
            T.tile.cast(ob, h8, "CAST_NONE", POOL_TILE * dim)

            # 6. emit the E4M3 rows and the per-pool scales to the contiguous
            # scratches; the host adapter scatters both into the paged caches.
            # Kept adjacent so the single V_MTE3 dependency gate covers both
            # copies (MTE3 is in-order); see the module docstring for why the
            # scatter is not on the device.
            T.copy(ob, ROW_SCRATCH[base : base + POOL_TILE, :])
            T.copy(scale, SC_SCRATCH[base : base + POOL_TILE])

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
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Validate the public contract; returns the caller write mask (``None`` when
    implicit) and the active locations."""
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
        # Implicit all-true mask: keep ``None`` so the adapter can skip the
        # mask gathers on this hot path; every location is active.
        active_locs = loc
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

    return write_mask, active_locs


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
    checked, and masked rows are never scattered into the caches. Validation
    runs on the host before kernel launch. ``k_cache`` holds E4M3 byte patterns
    in a uint8 tensor (Ascend port of the reference FP8 contract).
    """
    write_mask, active_locs = _validate_kpool_inputs(
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

    num_tiles = (num_pools + BLOCK_POOLS - 1) // BLOCK_POOLS
    padded_pools = num_tiles * BLOCK_POOLS

    # Host preparation: fold ape into the scores (fp32, reference-identical),
    # repack slot-major and pad each slot block to the device block with zero
    # rows (mirrors the kernel's [slot][padded_pool][dim] layout; padded rows
    # only ever reach the scratch, never the caches).
    scores = slot_score.float() + ape.unsqueeze(0)
    s2 = scores.permute(1, 0, 2)
    k2 = slot_k.permute(1, 0, 2)
    if padded_pools != num_pools:
        pad_pools = padded_pools - num_pools
        s2 = torch.cat([s2, s2.new_zeros(pool_size, pad_pools, head_dim)], dim=1)
        k2 = torch.cat([k2, k2.new_zeros(pool_size, pad_pools, head_dim)], dim=1)
    s2 = s2.reshape(pool_size * padded_pools, head_dim).contiguous()
    k2 = k2.reshape(pool_size * padded_pools, head_dim).contiguous()

    kernel = glm53_kpool_compress_kernel(
        num_tiles,
        pool_size=pool_size,
        head_dim=head_dim,
        round_scale=round_scale,
    )
    row_scratch = torch.empty((padded_pools, head_dim), dtype=torch.uint8, device=slot_k.device)
    sc_scratch = torch.empty(padded_pools, dtype=torch.float32, device=slot_k.device)
    kernel(k2, s2, row_scratch, sc_scratch)

    # Scatter the E4M3 rows and the per-pool scales to their physical slots
    # with one device indexing op each. 4-byte scalar GM stores are not
    # cache-line coherent across cores on this target (the compiler disables
    # automatic DCCI for scalar accesses), and a device-side per-row scatter
    # loop is serialized by conservative PIPE_ALL barriers; both scratch
    # layouts keep the device work bulk and contiguous.
    if write_mask is None:
        k_cache.view(-1, head_dim)[active_locs] = row_scratch[:num_pools]
        scale_cache.view(-1)[active_locs] = sc_scratch[:num_pools]
    else:
        k_cache.view(-1, head_dim)[active_locs] = row_scratch[:num_pools][write_mask]
        scale_cache.view(-1)[active_locs] = sc_scratch[:num_pools][write_mask]


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
    return result * HADAMARD128_SCALE


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
        scale = torch.exp2(torch.ceil(torch.log2(absmax / E4M3_MAX)))
    else:
        scale = absmax / E4M3_MAX
    quantized = (rotated / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
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
