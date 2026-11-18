import pytest
import torch

import tilelang
import tilelang.language as T

"""
Regression test for issue #929: T.copy between a strided (non-last-dim) GM
region and a flat 1D UB buffer produced incorrect values.

Root cause
----------
A GM<->UB copy is emitted as a 2D MTE DMA: ``validRow`` bursts of ``validCol``
elements each. On the UB side every burst start (and therefore the per-row
pitch) must be 32-byte aligned -- the helper passes the UB-side inter-burst
gap in 32-byte units and the hardware starts each burst at a 32-byte boundary.
The issue's kernel loads ``beta_in[bz, s:s+32, by]`` (one element per row, 64
elements apart in GM) into a flat ``[32]`` UB buffer, which lowered to 32
bursts of 2 bytes each. Only burst 0 landed inside the buffer; bursts 1..31
scattered to 32-byte boundaries *past* the buffer, silently corrupting
neighbouring UB allocations (in the issue's kernel, the k tile loaded before
beta). Both directions were broken: a strided store from a flat 1D UB buffer
read out of bounds the same way.

The same 32-byte contract also covers a second latent shape: a 2D column
slice into a wider UB tile (e.g. ``a_ub[:, 0:32]`` of an ``[8, 64]`` tile)
whose rows the DMA packed at the *region* width instead of the tile's real
row pitch.

The fix (src/op/ascend.cc)
-------------------------
* Loads  (GM->UB): lowered to an elementwise scalar loop between the region's
  declared layouts, including the helper's padValue prefill semantics
  (MakeScalarGMCopyLoop).
* Stores (UB->GM): scalar GM stores are NOT a fallback on this uArch -- the
  S-pipe write-back cache drops most partial-line (2-byte) stores before they
  reach GM (the #1304 failure mode; a DataCacheCleanAndInvalid after the loop
  does not recover them). Instead the elements are staged into a 32-byte-row-
  aligned UB scratch tile and one single-burst DMA is emitted per row
  (MakeStagedUBToGMCopy). Single bursts have no UB-side pitch requirement and
  their staging-row source is 32-byte aligned by construction.
* ``copy_ub_to_gm``'s template column dim now prefers the source buffer's
  physical inner dim, so a column slice into a wider tile keeps a single DMA
  with the correct pitch when the gap is 32-byte expressible (matches what
  the PTO backend already does).
* ``T.tile.atomic_add`` with the same unrepresentable shapes now fails loudly
  (no correct scalar fallback exists on A2/A3) instead of emitting the broken
  DMA.
* The PTO backend keeps its own lowering for these shapes (its TLOAD model
  expresses sub-32B runs natively), so the fallbacks gate on the target model.

These tests target the AscendC backend (the fix is in the AscendC copy
lowering) and require an Ascend NPU.
"""

DTYPE = "float16"
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

# Issue geometry, scaled down (and the original L=8192 for the end-to-end
# case). All dims are concrete: buffer shapes must stay compile-time
# constants for the memory planner.
B, L, H, K, C = 1, 512, 64, 128, 64
HALF_C = C // 2


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.disable_cache()
    yield


def _issue_kernel():
    """The issue's kernel: strided k and beta loads, k * beta, strided k store.

    beta's load is the reported bug: ``beta_in[bz, s:s+32, by]`` into a flat
    ``[32]`` UB buffer is a 32-burst x 1-element (2-byte) DMA.
    """

    @T.prim_func
    def main(
        k_in: T.Tensor([B, L, H, K], DTYPE),  # type: ignore
        beta_in: T.Tensor([B, L, H], DTYPE),  # type: ignore
        k_out: T.Tensor([B, L, H, K], DTYPE),  # type: ignore
    ):
        with T.Kernel(B * H * (L // C), is_npu=True) as (cid, vid):
            bx = cid % (L // C)
            by = (cid // (L // C)) % H
            bz = (cid // (L // C)) // H
            k_buf = T.alloc_ub([HALF_C, K], DTYPE)
            beta_buf = T.alloc_ub([HALF_C], DTYPE)
            with T.Scope("V"):
                T.copy(
                    k_in[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                        0:K,
                    ],
                    k_buf,
                )
                T.copy(
                    beta_in[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                    ],
                    beta_buf,
                )

                for i, j in T.Parallel(HALF_C, K):
                    k_buf[i, j] = k_buf[i, j] * beta_buf[i]

                T.copy(
                    k_buf,
                    k_out[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                        0:K,
                    ],
                )

    return main


def test_issue_k_mul_beta_strided():
    """The issue's kernel at reduced scale."""
    kernel = tilelang.compile(_issue_kernel(), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    beta = torch.rand(B, L, H, dtype=torch.float16).npu()
    k = torch.randn(B, L, H, K, dtype=torch.float16).npu()
    out = kernel(k, beta)
    torch.npu.synchronize()
    ref = k * beta.unsqueeze(-1)
    # The kernel computes in fp16 exactly like the reference.
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)


def _strided_roundtrip_kernel():
    """Strided 1D load into a flat UB buffer + strided 1D store back.

    Both fallbacks end to end: the load goes through the scalar loop, the
    store through the staging tile + per-row single-burst DMAs.
    """

    @T.prim_func
    def main(
        beta_in: T.Tensor([B, L, H], DTYPE),  # type: ignore
        beta_out: T.Tensor([B, L, H], DTYPE),  # type: ignore
    ):
        with T.Kernel(B * H * (L // C), is_npu=True) as (cid, vid):
            bx = cid % (L // C)
            by = (cid // (L // C)) % H
            bz = (cid // (L // C)) // H
            beta_buf = T.alloc_ub([HALF_C], DTYPE)
            with T.Scope("V"):
                T.copy(
                    beta_in[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                    ],
                    beta_buf,
                )
                T.copy(
                    beta_buf,
                    beta_out[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                    ],
                )

    return main


def test_strided_1d_roundtrip():
    kernel = tilelang.compile(_strided_roundtrip_kernel(), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    beta = torch.randn(B, L, H, dtype=torch.float16).npu()
    out = kernel(beta)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), beta.cpu(), rtol=1e-2, atol=1e-2)


def _strided_compute_kernel():
    """Compute in UB, then strided-store the flat buffer to GM.

    The store direction alone: values are produced in UB (no GM load), so any
    corruption comes from the strided scatter.
    """

    @T.prim_func
    def main(
        beta_out: T.Tensor([B, L, H], DTYPE),  # type: ignore
    ):
        with T.Kernel(B * H * (L // C), is_npu=True) as (cid, vid):
            bx = cid % (L // C)
            by = (cid // (L // C)) % H
            bz = (cid // (L // C)) // H
            beta_buf = T.alloc_ub([HALF_C], DTYPE)
            with T.Scope("V"):
                for i in T.Parallel(HALF_C):
                    beta_buf[i] = (bx * C + vid * HALF_C + i) * 0.25 + by * 0.01
                T.copy(
                    beta_buf,
                    beta_out[
                        bz,
                        bx * C + vid * HALF_C : bx * C + (vid + 1) * HALF_C,
                        by,
                    ],
                )

    return main


def test_strided_1d_store_from_compute():
    kernel = tilelang.compile(_strided_compute_kernel(), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    out = kernel()
    torch.npu.synchronize()
    l_idx = torch.arange(L, dtype=torch.float32).unsqueeze(-1)
    h_idx = torch.arange(H, dtype=torch.float32).unsqueeze(0)
    ref = (l_idx * 0.25 + h_idx * 0.01).to(torch.float16).unsqueeze(0)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)


def _col_slice_kernel(tile_w: int):
    """A 2D column slice into a wider UB tile: ``a_ub[:, 0:32]`` of [8, tile_w].

    tile_w = 64: 128B row pitch, (64-32)*2 = 64B gap -- 32-byte expressible,
    the store keeps a single DMA with the corrected template; the load falls
    back to the scalar loop (the load template keeps the region width for its
    prefill semantics, so the fallback handles the pitch).
    tile_w = 40: 80B row pitch, (40-32)*2 = 16B gap -- not expressible, the
    store goes through the staging tile fallback.
    """

    @T.prim_func
    def main(
        A: T.Tensor([8, 32], DTYPE),  # type: ignore
        C_out: T.Tensor([8, tile_w], DTYPE),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub([8, tile_w], DTYPE)
            with T.Scope("V"):
                T.copy(A[0, 0], a_ub[:, 0:32])
                T.copy(a_ub[:, 0:32], C_out[0, 0])

    return main


@pytest.mark.parametrize("tile_w", [40, 64])
def test_column_slice_into_wider_tile(tile_w):
    kernel = tilelang.compile(_col_slice_kernel(tile_w), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    a = torch.randn(8, 32, dtype=torch.float16).npu()
    c = kernel(a)
    torch.npu.synchronize()
    # Only the copied columns [0:32) are defined; the rest of the tile keeps
    # its previous contents (no pad prefill for a clean column slice).
    torch.testing.assert_close(c.cpu()[:, :32], a.cpu(), rtol=1e-2, atol=1e-2)


def _strided_fp32_kernel():
    """fp32 variant: 1 element per row (4-byte bursts), stride H."""

    @T.prim_func
    def main(
        beta_in: T.Tensor([1, 64, 32], "float32"),  # type: ignore
        beta_out: T.Tensor([1, 64, 32], "float32"),  # type: ignore
    ):
        with T.Kernel(32, is_npu=True) as (cid, vid):
            by = cid
            beta_buf = T.alloc_ub([16], "float32")
            with T.Scope("V"):
                T.copy(beta_in[0, 0:16, by], beta_buf)
                T.copy(beta_buf, beta_out[0, 0:16, by])

    return main


def test_strided_1d_roundtrip_fp32():
    kernel = tilelang.compile(_strided_fp32_kernel(), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    beta = torch.randn(1, 64, 32, dtype=torch.float32).npu()
    out = kernel(beta)
    torch.npu.synchronize()
    # Only the l in [0:16) stripe is copied by the kernel.
    torch.testing.assert_close(out.cpu()[0, :16], beta.cpu()[0, :16], rtol=1e-4, atol=1e-4)


def test_atomic_add_strided_sub32b_fails_loudly():
    """T.tile.atomic_add with a sub-32B strided scatter must not compile.

    Before the fix this emitted the same broken multi-burst DMA as the copy
    (silently corrupting GM); there is no correct scalar fallback on A2/A3
    (the scalar atomic API only exists on 5102/3510-class parts), so the
    lowering now rejects the shape with a clear error.
    """
    with pytest.raises(Exception, match="atomic_add"):

        @T.prim_func
        def main(
            src: T.Tensor([1, 64, 16], DTYPE),  # type: ignore
            dst: T.Tensor([1, 64, 16], DTYPE),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                src_ub = T.alloc_ub([32], DTYPE)
                with T.Scope("V"):
                    T.copy(src[0, 0, 0:32], src_ub)
                    # 32 rows x 1 element (2 bytes) per row, stride 16.
                    T.tile.atomic_add(dst[0, 0:32, 8], src_ub)

        tilelang.compile(main, out_idx=[-1], target="ascendc")


def _broadcast_axis1_kernel(M: int, N: int, block_M: int, block_N: int):
    """A [block_M, 1] strided load (sub-32B bursts) feeding an axis-1
    broadcast and a tail-clamped store -- the tail_mask_codegen axis-1
    broadcast kernel. The load's DMA corrupted data before the fix (only
    burst 0 landed); the scalar fallback makes the kernel correct.
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, (N + block_N - 1) // block_N), DTYPE),  # type: ignore
        C: T.Tensor((M, N), DTYPE),  # type: ignore
    ):
        with T.Kernel(
            (M + block_M - 1) // block_M * ((N + block_N - 1) // block_N),
            is_npu=True,
        ) as (cid, _):
            n_num = (N + block_N - 1) // block_N
            bx = cid // n_num
            by = cid % n_num
            src_ub = T.alloc_ub((block_M, 1), DTYPE)
            dst_ub = T.alloc_ub((block_M, block_N), DTYPE)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by : by + 1], src_ub)
            T.tile.broadcast(dst_ub, src_ub, axis=1)
            T.copy(
                dst_ub,
                C[
                    bx * block_M : (bx + 1) * block_M,
                    by * block_N : (by + 1) * block_N,
                ],
            )

    return main


def test_broadcast_axis1_strided_load_runtime():
    kernel = tilelang.compile(_broadcast_axis1_kernel(5, 69, 4, 64), out_idx=[-1], pass_configs=PASS_CONFIGS, target="ascendc")
    n_num = (69 + 63) // 64
    a = torch.randn(5, n_num, dtype=torch.float16).npu()
    c = kernel(a)
    torch.npu.synchronize()
    ref = torch.zeros(5, 69, dtype=torch.float16)
    for by in range(n_num):
        lo, hi = by * 64, min((by + 1) * 64, 69)
        ref[:, lo:hi] = a[:, by : by + 1].expand(5, hi - lo)
    torch.testing.assert_close(c.cpu(), ref, rtol=1e-2, atol=1e-2)
