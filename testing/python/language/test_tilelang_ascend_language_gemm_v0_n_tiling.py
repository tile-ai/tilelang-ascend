import pytest
import tilelang
import tilelang.language as T
import torch

"""
Regression test for N-tiling in the AscendC ``gemm_v0`` (src/tl_templates/ascend/common.h).

Feature under test
------------------
``gemm_v0`` originally tiled only the K dimension: every ``mma`` loaded the whole
N of the B operand into a single L0B slot (size ``N * kL0Size``).  L0B is 64KB and
the kL0 ping-pong halves the per-slot budget to 32KB, so once N is large the B
sub-tile overflows L0B and the cube crashes at run time.  Concretely an attention
PV matmul is ``<M = 64, N = headDim = 512, K = window <= 128>``: the B tile is
``K * N * 2 = 128 * 512 * 2 = 128KB`` -- twice the whole L0B.

The fix adds an inner N-tiling loop (``nTile = 128``, matching the Ascend C
reference's ``N_SPLIT_SIZE``): each N-tile loads its own ``(kL0Size x nTile)`` B
sub-block into L0B and accumulates into its own column band of the L0C output.
The (N-tile, K-tile) sequence is flattened so the L0A/L0B ping-pong is primed and
drained once and runs continuously across all tiles.

Only the ``transpose_B == false`` path with ``N > nTile`` is tiled; every existing
caller (``transpose_B``, or ``N <= nTile``) keeps ``nL0split == 1`` and byte-for-byte
identical codegen.

Coverage (all ``target = ascendc``)
-----------------------------------
  * ``N = 512`` (the PV shape) -- the large-N path that overflowed L0B before the
    fix and now tiles N into 4 sub-blocks.  ``K = 128`` exercises a single K-tile
    with N-tiling; ``K = 256`` exercises the flattened (N-tile, K-tile) pipeline
    (kL0split == 2, nL0split == 4 -> 8 tiles).
  * ``N = 128`` (``<= nTile``) -- the compatibility case: ``nL0split == 1``, the
    original single-pass behaviour, proving the tiled loop still matches when N
    is not split.

This targets the ascendc backend only.  The PTO backend has its own separate
``gemm_v0`` (src/tl_templates/pto/common.h, a different ``gemm_v0_inner`` /
``TileMatL0B`` implementation) and is tracked independently.
"""

TARGET = "ascendc"

CUBE_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear the tilelang cache before the session (a stale kernel could mask a
    codegen regression -- a rebuilt kernel could otherwise return a cached one)."""
    tilelang.cache.clear_cache()
    yield


def _torch_dtype(dtype):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def gemm_nt(M, N, K, dtype, accum_dtype):
    """Plain ``C = A @ B`` (transpose_B=False) staged through one L1 tile.

    The whole (M, K) / (K, N) operands are copied into L1 in one shot; ``gemm_v0``
    then does the K-tiling (kL0Size=128) and -- for large N -- the N-tiling
    internally.  Mirrors ``full_copy_plain`` in the gm_to_l1 suite (default zN
    layout, no annotation)."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((M, K), dtype)
            B_L1 = T.alloc_L1((K, N), dtype)
            C_L0 = T.alloc_L0C((M, N), accum_dtype)
            with T.Scope("C"):
                T.copy(A[0, 0], A_L1)
                T.copy(B[0, 0], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                T.copy(C_L0, C[0, 0])

    return main


def run_test_gemm_nt(M, N, K, dtype, accum_dtype):
    torch.manual_seed(0)
    func = gemm_nt(M, N, K, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=CUBE_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    a = torch.randn(M, K, dtype=td).npu()
    b = torch.randn(K, N, dtype=td).npu()
    torch.npu.synchronize()
    c = func(a, b)
    ref = a @ b
    # bf16 accumulates more rounding over the larger N/K here than fp16.
    rtol, atol = (2e-2, 2e-2) if dtype == "bfloat16" else (1e-2, 1e-2)
    torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)


# (M, N, K) -- N=512 is the large-N path that overflowed L0B before N-tiling;
# N=128 (<= nTile) is the single-pass compatibility case.  M=64 keeps the N=512
# L0C tile (M*N*4) within the 128KB L0C budget, matching the real PV matmul.
gemm_configs = [
    (64, 512, 128),  # PV shape: single K-tile, N tiled into 4
    (64, 512, 256),  # flattened (N-tile, K-tile) pipeline: 2 K-tiles x 4 N-tiles
    (128, 128, 128),  # N <= nTile: nL0split == 1, original single-pass path
]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("M,N,K", gemm_configs)
def test_gemm_v0_n_tiling(M, N, K, dtype):
    run_test_gemm_nt(M, N, K, dtype, "float")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
