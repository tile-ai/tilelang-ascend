import pytest
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
import torch

"""
Regression test for the ``copy_pa`` primitive (paged-attention KV load) on the
AscendC backend.

Feature under test
------------------
``T.copy_pa`` (src/tl_templates/ascend/common.h ``copy_pa<T>``) loads ``copy_row_num``
rows of a *paged* KV cache straight from GM into an L1 tile, resolving each page
through a ``block_table``.  It is a faithful port of the reference op's
``DataCopyPA`` (PA_ND): the window is walked one page at a time, and each page's
contiguous row run is Nd2Nz-copied into L1, so a window that spans several pages
becomes several ``DataCopy`` calls (a runtime ``while`` loop over pages).  Running
on the cube (L1-direct) it needs no UB staging and no GM workspace round-trip.

Why it needs a primitive: the page walk is data-dependent -- the loop trip count,
the per-page row count, and the sub-window offset are all runtime values (they
depend on ``s2_idx`` and the ``block_table`` contents), which ``T.serial`` /
``T.copy`` (compile-time extents) and ``gemm_v0`` (no runtime-offset operands)
cannot express.

How this test exercises it
--------------------------
The kernel mirrors the reference dataflow: ``copy_pa`` loads the KV window into an
L1 tile, which is then consumed by a QK matmul (``q @ kv^T``, ``transpose_B=True``,
reusing the verified gm_to_l1 QK pattern).  The QK output is checked against a
torch reference that gathers the same window by hand through the ``block_table``.
If ``copy_pa`` reads the wrong physical rows (bad page indirection, wrong per-page
count, or a missed page boundary) the QK result diverges.

The KV cache is laid out as ``[num_phys_blocks * block_size, head_dim]`` (physical
page ``p`` occupies rows ``[p*block_size, (p+1)*block_size)``); the ``block_table``
maps a logical page to a physical page and is deliberately shuffled to exercise the
indirection.  ``N = win <= 128`` so the QK matmul stays on the single-pass
(``nL0split == 1``) gemm_v0 path and does not depend on any N-tiling.

Coverage (all ``target = ascendc``)
-----------------------------------
  * single page   -- window fits one page (``while`` runs once)
  * cross 2 pages -- window straddles a page boundary (``while`` runs twice)
  * cross 3 pages -- window starts mid-page and spans three pages (``while`` runs
    three times), stressing the runtime per-page row count and offset

This targets the ascendc backend only.
"""

TARGET = "ascendc"

DEV_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
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


def copy_pa_qk(block_M, win, dim, block_size, num_phys, s2_idx, dtype, accum_dtype):
    """``copy_pa`` loads a paged KV window into L1, then QK (``q @ kv^T``) consumes
    it.  ``win`` (== the QK N dimension) is the window row count and fills the whole
    ``k_l1`` tile; ``head_num``/``n2_idx``/``d_idx`` are the PA_ND parity values SWA
    uses (1 / 0 / 0)."""
    kv_rows = num_phys * block_size
    max_block_num = num_phys  # logical pages == physical pages here

    @T.prim_func
    def main(
        Q: T.Tensor([1, block_M, dim], dtype),  # type: ignore
        KV: T.Tensor([kv_rows, dim], dtype),  # type: ignore  paged cache
        BT: T.Tensor([max_block_num], "int32"),  # type: ignore  block table
        S: T.Tensor([1, block_M, win], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([win, dim], dtype)
            l0c = T.alloc_L0C([block_M, win], accum_dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                }
            )
            T.copy(Q[0, :, :], q_l1[:, :])
            # Load the paged window [s2_idx, s2_idx + win) into k_l1 via the block
            # table.  head_num=1, n2_idx=0, d_idx=0 (PA_ND parity); kv_stride is one
            # physical page = block_size * head_num * head_dim; copy_row_num_align =
            # win (k_l1 row capacity).
            T.copy_pa(
                k_l1[:, :],
                KV,
                BT,
                block_size,
                1,
                dim,
                block_size * dim,
                max_block_num,
                dim,
                win,
                win,
                0,
                0,
                s2_idx,
                0,
            )
            T.gemm_v0(q_l1, k_l1, l0c, transpose_B=True, init=True)
            T.copy(l0c, S[0, :, :])

    return main


def _ref_qk(q, kv_2d, bt, block_size, s2_idx, win):
    """Golden: gather the window row-by-row through the block table, then q @ kv^T."""
    gathered = torch.empty((win, kv_2d.shape[1]), dtype=torch.float32)
    for i in range(win):
        r = s2_idx + i
        logical_page = r // block_size
        row_in_page = r % block_size
        phys_page = int(bt[logical_page])
        gathered[i] = kv_2d[phys_page * block_size + row_in_page].float()
    return q.float() @ gathered.t()  # (block_M, win)


def run_test_copy_pa(block_M, win, dim, block_size, num_phys, s2_idx, bt_list, dtype):
    torch.manual_seed(0)
    accum_dtype = "float"
    func = copy_pa_qk(block_M, win, dim, block_size, num_phys, s2_idx, dtype, accum_dtype)
    func = tilelang.compile(func, out_idx=[-1], pass_configs=DEV_CONFIGS, target=TARGET)
    td = _torch_dtype(dtype)
    q = torch.randn(1, block_M, dim, dtype=td).npu()
    kv = torch.randn(num_phys * block_size, dim, dtype=td).npu()
    bt = torch.tensor(bt_list, dtype=torch.int32).npu()
    torch.npu.synchronize()
    s = func(q, kv, bt)
    ref = _ref_qk(q[0].cpu(), kv.cpu(), bt.cpu(), block_size, s2_idx, win)
    rtol, atol = (2e-2, 2e-2) if dtype == "bfloat16" else (1e-2, 1e-2)
    torch.testing.assert_close(s[0].cpu(), ref, rtol=rtol, atol=atol)


# (block_size, num_phys, s2_idx, block_table) -- win=64, so:
#   single page  : block_size=64, window [0,64) fits logical page 0        -> while 1x
#   cross 2 pages: block_size=32, window [0,64) = pages 0,1                 -> while 2x
#   cross 3 pages: block_size=32, window [16,80) = pages 0(16..),1,2(..16)  -> while 3x
# block tables are shuffled so the physical page != logical page (tests indirection).
copy_pa_configs = [
    (64, 2, 0, [1, 0]),  # single page
    (32, 4, 0, [2, 0, 3, 1]),  # cross 2 pages, aligned
    (32, 4, 16, [2, 0, 3, 1]),  # cross 3 pages, mid-page start
]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_size,num_phys,s2_idx,bt_list", copy_pa_configs)
def test_copy_pa(block_size, num_phys, s2_idx, bt_list, dtype):
    run_test_copy_pa(32, 64, 128, block_size, num_phys, s2_idx, bt_list, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
