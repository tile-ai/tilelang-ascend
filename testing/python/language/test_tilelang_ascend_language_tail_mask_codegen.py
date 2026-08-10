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

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.transform.pass_config import process_default_pass_config

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
