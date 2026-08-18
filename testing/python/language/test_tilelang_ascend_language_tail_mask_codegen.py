"""Codegen-level checks for the Ascend vector tail-block scheme.

These tests only inspect the generated kernel source (host-side codegen), so
they do not require NPU hardware to *run* the kernel -- only a built tilelang
with the Ascend codegen. They verify that:

  * a kernel with a real tail (M and/or N not divisible by the block) emits the
    internal ``tl::ascend::tail_*`` helpers, and
  * guarded reduction contracts either emit the axis-0 tail helper or retain
    the native reduce path. The hybrid ``pad_value`` fallback remains available
    for unsupported full-tile readers.
"""

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.transform.pass_config import process_default_pass_config
from tilelang.utils.target import determine_platform

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    # Enable the opt-in tail-block scheme (default off) so the tail_* helpers
    # are actually emitted for these tests.
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}

TAIL_TARGETS = ("ascendc", "pto")
TAIL_REDUCE_KINDS = (
    "sum",
    pytest.param("max", marks=pytest.mark.low_priority),
    pytest.param("min", marks=pytest.mark.low_priority),
)
TAIL_REDUCE_AXIS0_DIMS = (0, -2)


def _tail_add(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            c_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.copy(c_ub, C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _tail_reduce(M, N, block_M, block_N, dtype="float", kind="sum", clear=True):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    reduce_fn = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }[kind]

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, block_N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((block_M, 1), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            reduce_fn(a_ub, r_ub, dim=-1, clear=clear)
            T.copy(r_ub, B[bx * block_M : (bx + 1) * block_M, by : by + 1])

    return main


def _tail_reduce_then_unary(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, n_num), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((block_M, 1), dtype)
            o_ub = T.alloc_ub((block_M, 1), dtype)
            T.copy(
                A[
                    bx * block_M : (bx + 1) * block_M,
                    by * block_N : (by + 1) * block_N,
                ],
                a_ub,
            )
            T.reduce_sum(a_ub, r_ub, dim=-1)
            T.tile.exp(o_ub, r_ub)
            T.copy(o_ub, B[bx * block_M : (bx + 1) * block_M, by : by + 1])

    return main


def _tail_reduce_axis0(
    M,
    N,
    block_M,
    block_N,
    dtype="float",
    kind="sum",
    dim=0,
    clear=True,
    real_shape=None,
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    reduce_fn = {
        "sum": T.reduce_sum,
        "max": T.reduce_max,
        "min": T.reduce_min,
    }[kind]

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((m_num, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((1, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            if real_shape is None:
                reduce_fn(a_ub, r_ub, dim=dim, clear=clear)
            else:
                reduce_fn(a_ub, r_ub, dim=dim, clear=clear, real_shape=real_shape)
            T.copy(r_ub, B[bx : bx + 1, by * block_N : (by + 1) * block_N])

    return main


def _tail_reduce_axis0_then_unary(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((m_num, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((1, block_N), dtype)
            o_ub = T.alloc_ub((1, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.reduce_sum(a_ub, r_ub, dim=0, clear=True)
            T.tile.exp(o_ub, r_ub)
            T.copy(o_ub, B[bx : bx + 1, by * block_N : (by + 1) * block_N])

    return main


def _tail_reduce_axis0_to_output_slice(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    def output_slice(buffer):
        return tvm.tir.BufferRegion(
            buffer,
            [
                tvm.ir.Range.from_min_extent(1, 1),
                tvm.ir.Range.from_min_extent(0, block_N),
            ],
        )

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((m_num, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            r_ub = T.alloc_ub((2, block_N), dtype)
            T.copy(
                A[
                    bx * block_M : (bx + 1) * block_M,
                    by * block_N : (by + 1) * block_N,
                ],
                a_ub,
            )
            T.reduce_sum(a_ub, output_slice(r_ub), dim=0, clear=True)
            T.copy(
                output_slice(r_ub),
                B[bx : bx + 1, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_unary(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.tile.exp(b_ub, a_ub)
            T.copy(b_ub, B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _tail_scalar(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.tile.add(b_ub, a_ub, 2.0)  # scalar immediate -> adds -> tail_scalar
            T.copy(b_ub, B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def _tail_compare_select(
    M,
    N,
    block_M,
    block_N,
    dtype="float",
    scalar_compare=False,
    scalar_select=False,
    mode="LT",
):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    mask_cols = T.ceildiv(N, 8)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        MaskOut: T.Tensor((M, mask_cols), "uint8"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            b_ub = T.alloc_ub((block_M, block_N), dtype)
            mask_ub = T.alloc_ub((block_M, block_N // 8), "uint8")
            out_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            if scalar_compare:
                T.tile.compare(mask_ub, a_ub, 0.0, mode)
            else:
                T.tile.compare(mask_ub, a_ub, b_ub, mode)
            if scalar_select:
                T.tile.select(out_ub, mask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
            else:
                T.tile.select(out_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
            T.copy(
                out_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )
            T.copy(
                mask_ub,
                MaskOut[
                    bx * block_M : (bx + 1) * block_M,
                    by * (block_N // 8) : (by + 1) * (block_N // 8),
                ],
            )

    return main


def _tail_compare_bufferload_overwrite(M, N, block_M, block_N):
    """Overwrite a tracked predicate through the unsupported BufferLoad ABI."""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float"),
        B: T.Tensor((M, N), "float"),
        C: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), "float")
            b_ub = T.alloc_ub((block_M, block_N), "float")
            mask_ub = T.alloc_ub((block_M, block_N // 8), "uint8")
            out_ub = T.alloc_ub((block_M, block_N), "float")
            scalar_ub = T.alloc_ub((1,), "float")
            scalar_ub[0] = T.cast(0.0, "float")
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(B[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], b_ub)
            T.tile.compare(mask_ub, a_ub, b_ub, "LT")
            T.tile.compare(mask_ub, a_ub, scalar_ub[0], "LT")
            T.tile.select(out_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
            T.copy(
                out_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_broadcast_axis1(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, n_num), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            src_ub = T.alloc_ub((block_M, 1), dtype)
            dst_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx * block_M : (bx + 1) * block_M, by : by + 1], src_ub)
            T.tile.broadcast(dst_ub, src_ub, axis=1)
            T.copy(
                dst_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_broadcast_axis0(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((m_num, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            src_ub = T.alloc_ub((1, block_N), dtype)
            dst_ub = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A[bx : bx + 1, by * block_N : (by + 1) * block_N], src_ub)
            T.tile.broadcast(dst_ub, src_ub, axis=0)
            T.copy(
                dst_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_broadcast_ambiguous_axis0(M, block_M, dtype="float"):
    """Legal axis-0 [1,1] -> [block_M,1], ambiguous to shape inference."""
    m_num = T.ceildiv(M, block_M)

    @T.prim_func
    def main(A: T.Tensor((m_num, 1), dtype), C: T.Tensor((M, 1), dtype)):
        with T.Kernel(m_num, is_npu=True) as (bx, _):
            src_ub = T.alloc_ub((1, 1), dtype)
            dst_ub = T.alloc_ub((block_M, 1), dtype)
            T.copy(A[bx : bx + 1, 0:1], src_ub)
            T.tile.broadcast(dst_ub, src_ub, axis=0)
            T.copy(dst_ub, C[bx * block_M : (bx + 1) * block_M, 0:1])

    return main


def _tail_broadcast_noop_then_unary(M, N, block_M, block_N):
    """A same-shape native broadcast must preserve tail provenance."""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), "float"), C: T.Tensor((M, N), "float")):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            src_ub = T.alloc_ub((block_M, block_N), "float")
            mid_ub = T.alloc_ub((block_M, block_N), "float")
            dst_ub = T.alloc_ub((block_M, block_N), "float")
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], src_ub)
            T.tile.broadcast(mid_ub, src_ub, axis=0)
            T.tile.exp(dst_ub, mid_ub)
            T.copy(
                dst_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_mixed_broadcast_compare_select(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Row: T.Tensor((M, n_num), dtype),
        Col: T.Tensor((m_num, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            row_ub = T.alloc_ub((block_M, 1), dtype)
            col_ub = T.alloc_ub((1, block_N), dtype)
            row_full_ub = T.alloc_ub((block_M, block_N), dtype)
            col_full_ub = T.alloc_ub((block_M, block_N), dtype)
            sum_ub = T.alloc_ub((block_M, block_N), dtype)
            abs_ub = T.alloc_ub((block_M, block_N), dtype)
            mask_ub = T.alloc_ub((block_M, block_N // 8), "uint8")
            out_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(Row[bx * block_M : (bx + 1) * block_M, by : by + 1], row_ub)
            T.copy(Col[bx : bx + 1, by * block_N : (by + 1) * block_N], col_ub)
            T.tile.broadcast(row_full_ub, row_ub, axis=1)
            T.tile.broadcast(col_full_ub, col_ub, axis=0)
            T.tile.add(sum_ub, row_full_ub, col_full_ub)
            T.tile.abs(abs_ub, a_ub)
            T.tile.compare(mask_ub, abs_ub, sum_ub, "LT")
            T.tile.select(out_ub, mask_ub, abs_ub, sum_ub, "VSEL_TENSOR_TENSOR_MODE")
            T.copy(
                out_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_mixed_unary_scalar_select(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), dtype)
            abs_ub = T.alloc_ub((block_M, block_N), dtype)
            shifted_ub = T.alloc_ub((block_M, block_N), dtype)
            mask_ub = T.alloc_ub((block_M, block_N // 8), "uint8")
            out_ub = T.alloc_ub((block_M, block_N), dtype)

            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.tile.abs(abs_ub, a_ub)
            T.tile.add(shifted_ub, abs_ub, 0.5)
            T.tile.compare(mask_ub, shifted_ub, 1.0, "LT")
            T.tile.select(out_ub, mask_ub, shifted_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
            T.copy(
                out_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _tail_select_external_mask(M, N, block_M, block_N):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float"),
        Mask: T.Tensor((M, N // 8), "uint8"),
        C: T.Tensor((M, N), "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_M, block_N), "float")
            mask_ub = T.alloc_ub((block_M, block_N // 8), "uint8")
            out_ub = T.alloc_ub((block_M, block_N), "float")
            T.copy(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], a_ub)
            T.copy(
                Mask[
                    bx * block_M : (bx + 1) * block_M,
                    by * (block_N // 8) : (by + 1) * (block_N // 8),
                ],
                mask_ub,
            )
            T.tile.select(out_ub, mask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
            T.copy(
                out_ub,
                C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N],
            )

    return main


def _source(func, target="ascendc", tail_mask=True):
    """Lower a PrimFunc and return generated device source.

    These source-inspection tests intentionally avoid creating a JIT adapter
    or compiling the generated AscendC/PTO source into a runnable kernel.
    """
    cfg = process_default_pass_config(
        target,
        {
            **pass_configs,
            tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: tail_mask,
        },
    )

    with tvm.transform.PassContext(opt_level=3, config=cfg):
        artifact = tilelang.lower(func, target=target, platform="auto")

    return artifact.kernel_source


def _lowered_tir(func, target="ascendc"):
    """Return post-tail-pass TIR without invoking backend source codegen."""
    platform = determine_platform("auto")
    target_obj = tvm.target.Target({"kind": "llvm", "model": target})
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    for gvar, prim_func in mod.functions_items():
        mod[gvar] = prim_func.with_attr("npu_platform", platform)
    cfg = process_default_pass_config(target, pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=cfg):
        mod = LowerAndLegalize(mod, target_obj)
    return str(mod)


# Per-backend "a tail-aware op was emitted" marker. The two backends express the
# valid-region compute differently in the generated source:
#   * ascendc -> a call to the internal ``tl::ascend::tail_<kind>`` device helper
#     (the mask/repeat/count ladder written in ascend/common.h).
#   * pto     -> a ``TileUbDataND<..., pto::DYNAMIC, pto::DYNAMIC>`` dynamic tile.
#     PTO reuses its native dynamic-tile op macros (TADD/TEXP/TADDS/...), so the
#     tell-tale is the DYNAMIC valid-shape tile, which is emitted by the tail
#     unary/binary/scalar/reduce codegen (CreateUbVariableDynamic) and nowhere
#     on the ordinary full-tile path.
def _emit_marker(target, kind):
    return f"tl::ascend::tail_{kind}" if target == "ascendc" else "pto::DYNAMIC"


def _no_tail_marker(target):
    # A substring that must be ABSENT when no op was rewritten to a tail variant.
    return "tl::ascend::tail_" if target == "ascendc" else "pto::DYNAMIC"


def _native_reduce_marker(kind, *, target="ascendc", dtype="float", clear=True, dim=-1):
    """Return a backend-specific marker for a native reduce path."""
    if target == "pto":
        direction = {
            -2: "col",
            -1: "row",
            0: "col",
            1: "row",
        }[dim]
        return {
            ("sum", "row"): "TROWSUM(",
            ("sum", "col"): "TCOLSUM(",
            ("max", "row"): "TROWMAX(",
            ("max", "col"): "TCOLMAX(",
            ("min", "row"): "TROWMIN(",
            ("min", "col"): "TCOLMIN(",
        }[(kind, direction)]

    # AscendC uses a dedicated helper for clear=true float16 sum reductions.
    if kind == "sum" and dtype == "float16" and clear:
        return "tl::ascend::reduce_sum_half<"

    return f"tl::ascend::reduce_{kind}<"


def _assert_tail_reduce_rewritten(src, target, kind):
    """Check the backend-specific lowering of the shared tail-reduce contract."""
    if target == "ascendc":
        assert f"tl::ascend::tail_reduce_{kind}" in src, src
        return

    assert "pto::DYNAMIC" in src, src
    assert "TileUbDataND<float, 32, 32, pto::DYNAMIC, pto::DYNAMIC>" in src, src
    assert "TileUbDataND<float, 1, 32, pto::DYNAMIC, pto::DYNAMIC>" in src, src
    assert _native_reduce_marker(kind, target="pto", dim=0) in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_add_emits_tail_helper(target):
    # 34x130 with 32x32 blocks => tail in both M and N.
    src = _source(_tail_add(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "binary") in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_unary_emits_tail_helper(target):
    src = _source(_tail_unary(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "unary") in src, src


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_tail_scalar_emits_tail_helper(target):
    src = _source(_tail_scalar(34, 130, 32, 32, "float"), target=target)
    assert _emit_marker(target, "scalar") in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("dtype", ["float16", "float"])
@pytest.mark.parametrize("scalar_compare", [False, True], ids=["tensor", "scalar"])
@pytest.mark.parametrize("scalar_select", [False, True], ids=["select_tensor", "select_scalar"])
def test_tail_compare_select_emits_backend_path(target, dtype, scalar_compare, scalar_select):
    # valid_col is 5 in the final block, which also exercises packed-byte
    # cleanup for comparison masks.
    func = _tail_compare_select(
        5,
        69,
        4,
        64,
        dtype=dtype,
        scalar_compare=scalar_compare,
        scalar_select=scalar_select,
    )
    src = _source(func, target=target)
    if target == "ascendc":
        compare = "tail_compare_scalar" if scalar_compare else "tail_compare"
        select = "tail_select_scalar" if scalar_select else "tail_select"
        assert f"tl::ascend::{compare}" in src, src
        assert f"tl::ascend::{select}" in src, src
        assert src.count(", 4, 64, 8);") >= 2, src
        if not scalar_select:
            mask_offset = re.search(r"mask_ub = .*?, (\d+)\);", src)
            out_offset = re.search(r"out_ub = .*?, (\d+)\);", src)
            assert mask_offset and out_offset, src
            # Four predicate rows require four 32-byte UB data blocks even
            # though their public packed Buffer shape is only [4, 8].
            assert int(out_offset.group(1)) - int(mask_offset.group(1)) >= 4 * 32, src
        if dtype == "float16" and scalar_compare and scalar_select:
            # Storage rewrite can reuse a half/float LocalTensor for the
            # uint8 predicate, but arena planning may instead retain a native
            # uint8 allocation.  Validate either semantically correct layout;
            # the exact reuse choice is deliberately planner-dependent.
            explicit_uint8_mask = re.search(r"auto mask_ub = .*GetWithOffset<uint8_t>", src)
            assert explicit_uint8_mask or src.count(".ReinterpretCast<uint8_t>()") >= 3, src
    else:
        compare = "compare_scalar(" if scalar_compare else "compare("
        select = "TSELS(" if scalar_select else "TSEL("
        assert "pto::DYNAMIC" in src, src
        assert f"tl::ascend_pto::{compare}" in src, src
        assert select in src, src
        assert "clear_compare_tail_bits" in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("mode", ["EQ", "NE", "GT", "GE", "LT", "LE"])
def test_tail_compare_supports_all_modes(target, mode):
    src = _source(_tail_compare_select(5, 69, 4, 64, mode=mode), target=target)
    expected = f"AscendC::CMPMODE::{mode}" if target == "ascendc" else f"CmpMode::{mode}"
    assert expected in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_bufferload_compare_clears_packed_mask_state(target):
    # The second compare takes a BufferLoad scalar and therefore remains on
    # the native path. It must invalidate the first compare's packed-mask
    # provenance before the following select is considered for tail rewrite.
    func = _tail_compare_bufferload_overwrite(5, 69, 4, 64)
    if target == "ascendc":
        src = _source(func, target=target)
        assert "tl::ascend::tail_compare" in src, src
        assert "tl::ascend::tail_select" not in src, src
    else:
        # PTO's established native scalar-compare source codegen does not
        # accept BufferLoad, so inspect the shared transformed TIR directly.
        tir = _lowered_tir(func, target=target)
        assert "T.ascend_tail_compare" in tir, tir
        assert "T.ascend_tail_select" not in tir, tir


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("dtype", ["float16", "float"])
@pytest.mark.parametrize("axis", [0, 1])
def test_tail_broadcast_emits_backend_path(target, dtype, axis):
    func = _tail_broadcast_axis0(5, 69, 4, 64, dtype) if axis == 0 else _tail_broadcast_axis1(5, 69, 4, 64, dtype)
    src = _source(func, target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_broadcast" in src, src
        if axis == 1:
            src_offset = re.search(r"src_ub = .*?, (\d+)\);", src)
            dst_offset = re.search(r"dst_ub = .*?, (\d+)\);", src)
            assert src_offset and dst_offset, src
            assert int(dst_offset.group(1)) - int(src_offset.group(1)) >= 4 * 32, src
    else:
        expected_op = "TCOLEXPAND(" if axis == 0 else "TROWEXPAND("
        assert expected_op in src, src
        if axis == 1:
            assert "TileUbDataND" in src and "pto::DYNAMIC, 1>" in src, src
            dst_addr = re.search(r"TASSIGN\(dst_ub, (\d+)\);", src)
            assert dst_addr and int(dst_addr.group(1)) >= 4 * 32, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_ambiguous_scalar_broadcast_keeps_native_path(target):
    src = _source(_tail_broadcast_ambiguous_axis0(5, 4), target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_broadcast" not in src, src
    else:
        assert "dst_ub_temp_" not in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_same_shape_broadcast_preserves_tail_state(target):
    src = _source(_tail_broadcast_noop_then_unary(5, 69, 4, 64), target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_broadcast" not in src, src
        assert "tl::ascend::tail_unary" in src, src
    else:
        assert "TEXP(" in src, src
        assert "pto::DYNAMIC" in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("dtype", ["float16", "float"])
def test_mixed_broadcast_compare_select_emits_all_tail_paths(target, dtype):
    src = _source(_tail_mixed_broadcast_compare_select(5, 69, 4, 64, dtype), target=target)
    if target == "ascendc":
        assert src.count("tl::ascend::tail_broadcast") == 2, src
        assert "tl::ascend::tail_binary" in src, src
        assert "tl::ascend::tail_unary" in src, src
        assert "tl::ascend::tail_compare" in src, src
        assert "tl::ascend::tail_select" in src, src
    else:
        assert "TROWEXPAND(" in src, src
        assert "TCOLEXPAND(" in src, src
        assert "TADD(" in src, src
        assert "TABS(" in src, src
        assert "tl::ascend_pto::compare(" in src, src
        assert "TSEL(" in src, src
        assert "pto::DYNAMIC" in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("dtype", ["float16", "float"])
def test_mixed_unary_scalar_select_emits_all_tail_paths(target, dtype):
    src = _source(_tail_mixed_unary_scalar_select(5, 69, 4, 64, dtype), target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_unary" in src, src
        assert "tl::ascend::tail_scalar" in src, src
        assert "tl::ascend::tail_compare_scalar" in src, src
        assert "tl::ascend::tail_select_scalar" in src, src
    else:
        assert "TABS(" in src, src
        assert "TADDS(" in src, src
        assert "tl::ascend_pto::compare_scalar(" in src, src
        assert "TSELS(" in src, src
        assert "pto::DYNAMIC" in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_full_tile_compare_select_keeps_native_path(target):
    src = _source(_tail_compare_select(4, 64, 4, 64), target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_compare" not in src, src
        assert "tl::ascend::tail_select" not in src, src
        assert "AscendC::Compare(" in src, src
        assert "AscendC::Select(" in src, src
    else:
        assert "pto::DYNAMIC" not in src, src
        assert "tl::ascend_pto::compare(" in src, src
        assert "TSEL(" in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_compare_select_flag_off_keeps_native_path(target):
    src = _source(
        _tail_compare_select(5, 69, 4, 64),
        target=target,
        tail_mask=False,
    )
    assert _no_tail_marker(target) not in src, src
    assert "tl::ascend::tail_compare" not in src, src
    assert "tl::ascend::tail_select" not in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_wide_tail_compare_select_falls_back_and_clears_state(target):
    # fp32 block_N=128 exceeds the single-vector AscendC contract. Both
    # backends deliberately keep the established native compare/select path.
    src = _source(_tail_compare_select(5, 129, 4, 128), target=target)
    assert "tl::ascend::tail_compare" not in src, src
    assert "tl::ascend::tail_select" not in src, src
    assert _no_tail_marker(target) not in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_select_requires_tracked_compare_mask(target):
    # A uint8 mask copied from GM is a regular byte tile, not a compare-packed
    # predicate carrying logical element extents.
    src = _source(_tail_select_external_mask(5, 72, 4, 64), target=target)
    assert "tl::ascend::tail_select" not in src, src
    assert _no_tail_marker(target) not in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("kind", TAIL_REDUCE_KINDS)
@pytest.mark.parametrize("dim", TAIL_REDUCE_AXIS0_DIMS)
def test_tail_reduce_float32_axis0_emits_backend_path(target, kind, dim):
    func = _tail_reduce_axis0(34, 130, 32, 32, "float", kind=kind, dim=dim)
    src = _source(func, target=target)
    _assert_tail_reduce_rewritten(src, target, kind)


@pytest.mark.parametrize(
    ("M", "N", "rewritten"),
    [
        pytest.param(34, 128, True, marks=pytest.mark.low_priority, id="row_tail"),
        pytest.param(32, 130, True, marks=pytest.mark.low_priority, id="column_tail"),
        (34, 130, True),
        (32, 128, False),
    ],
)
@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_reduce_sum_rewrites_only_partial_tiles(target, M, N, rewritten):
    src = _source(_tail_reduce_axis0(M, N, 32, 32), target=target)
    assert (_emit_marker(target, "reduce_sum") in src) is rewritten, src
    if rewritten:
        _assert_tail_reduce_rewritten(src, target, "sum")
    else:
        assert _native_reduce_marker("sum", target=target, dim=0) in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_reduce_accepts_explicit_matching_real_shape(target):
    func = _tail_reduce_axis0(34, 130, 32, 32, real_shape=[32, 32])
    src = _source(func, target=target)
    _assert_tail_reduce_rewritten(src, target, "sum")


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_reduce_supports_output_slice(target):
    src = _source(_tail_reduce_axis0_to_output_slice(34, 130, 32, 32), target=target)
    _assert_tail_reduce_rewritten(src, target, "sum")


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize("kind", TAIL_REDUCE_KINDS)
def test_tail_reduce_float32_last_axis_uses_native_path(target, kind):
    src = _source(_tail_reduce(34, 130, 32, 32, "float", kind=kind), target=target)
    assert _no_tail_marker(target) not in src, src
    assert _native_reduce_marker(kind, target=target, dtype="float", dim=-1) in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_reduce_axis0_propagates_column_tail_to_unary(target):
    src = _source(_tail_reduce_axis0_then_unary(34, 130, 32, 32), target=target)
    if target == "ascendc":
        assert "tl::ascend::tail_reduce_sum" in src, src
        assert "tl::ascend::tail_unary" in src, src
    else:
        assert _native_reduce_marker("sum", target="pto", dim=0) in src, src
        assert "TEXP(" in src, src
        # Dynamic views are emitted for both the tail reduction and unary op.
        assert src.count("pto::DYNAMIC") >= 8, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
@pytest.mark.parametrize(
    ("M", "N", "dtype"),
    [
        (34, 130, "float"),
        pytest.param(32, 130, "float", marks=pytest.mark.low_priority),
        pytest.param(34, 130, "float16", marks=pytest.mark.low_priority),
    ],
)
def test_native_reduce_clears_downstream_tail_state(target, M, N, dtype):
    src = _source(_tail_reduce_then_unary(M, N, 32, 32, dtype=dtype), target=target)
    assert _no_tail_marker(target) not in src, src
    assert _native_reduce_marker("sum", target=target, dtype=dtype, dim=-1) in src, src


@pytest.mark.parametrize(
    ("dtype", "clear", "real_shape"),
    [
        ("float", False, None),
        pytest.param("float16", True, None, marks=pytest.mark.low_priority),
        pytest.param("float", True, [31, 32], marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("kind", TAIL_REDUCE_KINDS)
@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_unsupported_tail_reduce_contracts_fall_back(target, kind, dtype, clear, real_shape):
    func = _tail_reduce_axis0(
        34,
        130,
        32,
        32,
        dtype=dtype,
        kind=kind,
        clear=clear,
        real_shape=real_shape,
    )
    src = _source(func, target=target)
    assert _no_tail_marker(target) not in src, src
    assert _native_reduce_marker(kind, target=target, dtype=dtype, clear=clear, dim=0) in src, src


@pytest.mark.parametrize(
    ("kind", "merge_marker"),
    [
        ("sum", "TADD("),
        pytest.param("max", "TMAX(", marks=pytest.mark.low_priority),
        pytest.param("min", "TMIN(", marks=pytest.mark.low_priority),
    ],
)
def test_pto_accumulating_tail_reduce_uses_native_reduce_and_merge(kind, merge_marker):
    func = _tail_reduce_axis0(34, 130, 32, 32, kind=kind, clear=False)
    src = _source(func, target="pto")
    assert _no_tail_marker("pto") not in src, src
    assert _native_reduce_marker(kind, target="pto", dim=0) in src, src
    assert merge_marker in src, src


@pytest.mark.parametrize("target", TAIL_TARGETS)
def test_tail_reduce_flag_off_emits_no_tail_helper(target):
    func = _tail_reduce_axis0(34, 130, 32, 32)
    src_on = _source(func, target=target, tail_mask=True)
    src_off = _source(func, target=target, tail_mask=False)
    assert _emit_marker(target, "reduce_sum") in src_on, src_on
    assert _no_tail_marker(target) not in src_off, src_off
    assert _native_reduce_marker("sum", target=target, dtype="float", dim=0) in src_off, src_off


@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_flag_off_emits_no_tail_helper(target):
    # Opt-in default: with the switch off the pass is a no-op, so a tail kernel
    # generates the same full-tile ops as upstream (no tail variant at all).
    src = _source(_tail_add(34, 130, 32, 32, "float"), target=target, tail_mask=False)
    assert _no_tail_marker(target) not in src, src


if __name__ == "__main__":
    print(_source(_tail_add(34, 130, 32, 32, "float")))
