# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Ordinary GEMM invocation and padded refresh contracts on real lowered IR."""

from math import prod
from pathlib import Path
import shutil
import subprocess

import pytest

from tilelang import language as T
from tilelang import tvm

from test_ascend_pipeline_ring_provenance import _nodes, _pre_sync
from test_ascend_upstream_sync_composite import _events, _gemms, _RESERVED, _run
from test_ascend_upstream_sync_composite_loop import _execute_k_protocol, _operation
from test_ascend_upstream_sync_pipeline import _assert_balanced_events

tir = tvm.tir


def _independent_gemms(batches=2, *, shared_a=False):
    # Extend the established gemm_nt staging pattern with independent L0C
    # outputs. Hidden L0A/B and helper events are shared even when user buffers
    # have no common Var, so their lifetimes need a separate dependency.
    @T.prim_func
    def main(
        A: T.Tensor((16, 64), "float16"),
        B: T.Tensor((64, 16), "float16"),
        D: T.Tensor((16, 64), "float16"),
        E: T.Tensor((64, 16), "float16"),
        O: T.Tensor((batches * 16, 16), "float16"),
        P: T.Tensor((batches * 16, 16), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            d = T.alloc_L1((16, 64), "float16")
            e = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            f = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                for batch in T.serial(batches):
                    T.copy(A, a)
                    T.copy(B, b)
                    T.copy(D, d)
                    T.copy(E, e)
                    T.gemm_v0(a, b, c, init=True, kL0Size=64)
                    if shared_a:
                        T.gemm_v0(a, e, f, init=True, kL0Size=64)
                    else:
                        T.gemm_v0(d, e, f, init=True, kL0Size=64)
                    T.copy(c, O[batch * 16, 0])
                    T.copy(f, P[batch * 16, 0])

    return main


def _nested_gemm(batches=3, trips=2):
    @T.prim_func
    def main(
        A: T.Tensor((16, trips * 64), "float16"),
        B: T.Tensor((trips * 64, 16), "float16"),
        O: T.Tensor((batches * 16, 16), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                for batch in T.serial(batches):
                    for k in T.serial(trips):
                        T.copy(A[0, k * 64], a)
                        T.copy(B[k * 64, 0], b)
                        T.gemm_v0(a, b, c, init=(k == 0), kL0Size=64)
                    T.copy(c, O[batch * 16, 0])

    return main


def _dynamic_nested_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float16"),
        B: T.Tensor((128, 16), "float16"),
        O: T.Tensor((48, 16), "float16"),
        count: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                for batch in T.serial(T.min(T.max(count, 0), 3)):
                    for k in T.serial(2):
                        T.copy(A[0, k * 64], a)
                        T.copy(B[k * 64, 0], b)
                        T.gemm_v0(a, b, c, init=(k == 0), kL0Size=64)
                    T.copy(c, O[batch * 16, 0])

    return main


def _padded_refresh_gemm(valid_m=1, valid_n=13, valid_k=47, dtype="float16", tile_k=64):
    # Poison all physical L1 rows/columns before the partial refresh. Reading
    # the full GEMM output makes stale padding observable, including the 15
    # nonlogical GEMV rows that its ordinary one-row output would hide.
    @T.prim_func
    def main(
        PoisonA: T.Tensor((16, tile_k), dtype),
        PoisonB: T.Tensor((tile_k, 16), dtype),
        A: T.Tensor((valid_m, valid_k), dtype),
        B: T.Tensor((valid_k, valid_n), dtype),
        O: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, tile_k), dtype)
            b = T.alloc_L1((tile_k, 16), dtype)
            c = T.alloc_L0C((16, 16), "float32")
            with T.Scope("C"):
                T.copy(PoisonA, a)
                T.copy(PoisonB, b)
                for _repeat in T.serial(2):
                    T.copy(A[0, 0], a)
                    T.copy(B[0, 0], b)
                    T.gemm_v0(a, b, c, init=True, kL0Size=64)
                T.copy(c, O)

    return main


def _repeated_tiled_gemm(
    m,
    n,
    k,
    k_l0,
    dtype="float16",
    repeats=2,
    transpose_b=False,
    transpose_a=False,
):
    # Preserve the complete review reproducer: its logical GM/L1 shapes are
    # unchanged, and every invocation stores before the next hidden-pool reuse.
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    padded_m = ((m + 15) // 16) * 16
    accum_dtype = "int32" if dtype == "int8" else "float32"

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        O: T.Tensor((repeats, m, n), accum_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1(a_shape, dtype)
            b = T.alloc_L1(b_shape, dtype)
            # Real physical backing and a logical view, following the existing
            # decl_buffer pattern in flash_attention/fa_opt/expert_v3. This
            # preserves helper M for a tail without inventing size_map space.
            c_storage = T.alloc_L0C((padded_m, n), accum_dtype)
            c = T.decl_buffer((m, n), accum_dtype, data=c_storage.data, scope="wmma.accumulator")
            with T.Scope("C"):
                for repeat in T.serial(repeats):
                    T.copy(A, a)
                    T.copy(B, b)
                    T.gemm_v0(
                        a,
                        b,
                        c,
                        init=True,
                        transpose_A=transpose_a,
                        transpose_B=transpose_b,
                        kL0Size=k_l0,
                    )
                    T.copy(c, O[repeat, 0, 0])

    return main


def _calculation_calls(function):
    operations = {"copy_gm_to_l1", "tl.ascend_gemm_v0", "copy_l0c_to_gm"}
    return [call for call in _nodes(function.body, tir.Call) if _operation(call) in operations]


def _assert_calculation_preserved(before, after):
    old, new = _calculation_calls(before), _calculation_calls(after)
    assert len(old) == len(new)
    for original, synchronized in zip(old, new):
        tvm.ir.assert_structural_equal(original, synchronized)
    old_allocations = [allocation.buffer_var for allocation in _nodes(before.body, tir.Allocate)]
    new_allocations = [allocation.buffer_var for allocation in _nodes(after.body, tir.Allocate)]
    assert old_allocations == new_allocations, "hidden resources must not allocate memory"
    assert all(event_id not in _RESERVED.get(event, set()) for _, event, event_id in _events(after))


def _rewrite_outer_loop(function, rewrite):
    outer = _nodes(function.body, tir.For)[-1]
    return function.with_body(
        tir.stmt_functor.ir_transform(
            function.body,
            lambda node: rewrite(node) if node.same_as(outer) else None,
            None,
            ["tir.For"],
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def invocation_kernels():
    return {
        "independent": _pre_sync(_independent_gemms())[0],
        "shared_a": _pre_sync(_independent_gemms(shared_a=True))[0],
        "nested": _pre_sync(_nested_gemm())[0],
    }


@pytest.fixture(scope="module")
def gemm_layout_probe(tmp_path_factory):
    # Compile the same dependency-free header consumed by the device helper
    # and native admission check. Check physical interval properties below
    # rather than reproducing its N-tile selection algorithm in Python.
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler for the shared GEMM layout header")
    directory = tmp_path_factory.mktemp("ascend_gemm_layout")
    source = directory / "probe.cc"
    binary = directory / "probe"
    source.write_text(
        """#include <iostream>
#include <string>
#include "tl_templates/ascend/gemm_layout.h"

int main(int argc, char **argv) {
  if (argc != 8) return 1;
  const tl::ascend::GemmV0Layout layout(
      std::stoull(argv[1]), std::stoull(argv[2]), std::stoull(argv[3]),
      std::stoull(argv[4]), std::stoull(argv[5]), std::stoull(argv[6]) != 0,
      std::stoull(argv[7]) != 0);
  std::cout << layout.n_tile << ' ' << layout.n_tiles << ' '
            << layout.k_tiles << ' ' << layout.steps << ' '
            << layout.slot_count << ' ' << layout.slot_budget << ' '
            << layout.k_stride << ' ' << layout.l0a_elements << ' '
            << layout.l0b_elements << ' ' << layout.l0a_bytes << ' '
            << layout.l0b_bytes << ' ' << layout.valid() << '\\n';
}
""",
    )
    repository = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(repository / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def query(geometry):
        if len(geometry) == 6:
            geometry = (*geometry, False)
        result = subprocess.run(
            [str(binary), *(str(int(value)) for value in geometry)],
            check=True,
            capture_output=True,
            text=True,
        )
        names = (
            "n_tile",
            "n_tiles",
            "k_tiles",
            "steps",
            "slot_count",
            "slot_budget",
            "k_stride",
            "l0a_elements",
            "l0b_elements",
            "l0a_bytes",
            "l0b_bytes",
            "valid",
        )
        values = [int(field) for field in result.stdout.split()]
        assert len(values) == len(names)
        return dict(zip(names, values))

    return query


@pytest.mark.parametrize(
    "geometry,valid",
    [
        ((16, 1360, 96, 96, 2, False), True),
        ((16, 1360, 48, 48, 4, False), True),
        ((16, 688, 192, 96, 2, False), True),
        ((16, 688, 96, 48, 4, False), True),
        ((13, 16, 128, 64, 2, False), True),
        ((512, 16, 64, 64, 2, False), True),
        ((256, 16, 128, 64, 2, False), True),
        ((16, 512, 64, 64, 2, True), True),
        ((16, 256, 128, 64, 2, True), True),
        ((32, 16, 256, 128, 1, False), True),
        ((32, 48, 256, 128, 1, False), True),
        ((16, 32, 256, 128, 1, False, True), True),
        ((48, 32, 256, 128, 1, False, True), True),
        ((16, 16, 256, 128, 1, True, True), True),
        ((528, 16, 64, 64, 2, False), False),
        ((272, 16, 128, 64, 2, False), False),
        ((16, 528, 64, 64, 2, True), False),
        ((16, 512, 128, 64, 2, True), False),
    ],
)
def test_shared_gemm_layout_covers_fractals_with_aligned_disjoint_physical_slots(
    gemm_layout_probe,
    geometry,
    valid,
):
    m, n, k, k_l0, element_bytes, transpose_b, *transpose_a = geometry
    transpose_a = bool(transpose_a and transpose_a[0])
    layout = gemm_layout_probe(geometry)
    assert bool(layout["valid"]) is valid
    c0 = 32 // element_bytes
    assert layout["k_stride"] == ((k_l0 + c0 - 1) // c0) * c0
    a_quantum = 32 if element_bytes == 1 and transpose_a else 16
    b_quantum = 32 if element_bytes == 1 and not transpose_b else 16
    a_rows = ((m + a_quantum - 1) // a_quantum) * a_quantum
    b_columns = ((layout["n_tile"] + b_quantum - 1) // b_quantum) * b_quantum
    a_bytes = a_rows * layout["k_stride"] * element_bytes
    b_bytes = b_columns * layout["k_stride"] * element_bytes
    assert layout["l0a_bytes"] == a_bytes == layout["l0a_elements"] * element_bytes
    assert layout["l0b_bytes"] == b_bytes == layout["l0b_elements"] * element_bytes
    assert layout["n_tiles"] * layout["n_tile"] == n
    assert layout["k_tiles"] == (k + k_l0 - 1) // k_l0
    assert layout["steps"] == layout["n_tiles"] * layout["k_tiles"]
    assert layout["slot_count"] == (2 if layout["steps"] > 1 else 1)
    assert layout["slot_budget"] == 65536 // layout["slot_count"]
    if layout["n_tiles"] > 1:
        assert layout["n_tile"] % max(16, c0) == 0
    if valid:
        for physical_bytes in (a_bytes, b_bytes):
            assert physical_bytes <= layout["slot_budget"]
            starts = range(0, layout["slot_count"] * physical_bytes, physical_bytes)
            intervals = [(start, start + physical_bytes) for start in starts]
            assert all(start % 512 == 0 for start, _end in intervals)
            assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
            assert intervals[-1][1] <= 65536
    else:
        assert max(a_bytes, b_bytes) > layout["slot_budget"]


@pytest.mark.parametrize("kind", ["independent", "shared_a", "nested"])
def test_invocations_keep_real_calculations_and_complete_hidden_pool(invocation_kernels, kind):
    original = invocation_kernels[kind]
    result = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, result)
    for _ in range(3):
        events, computations, clears = _execute_k_protocol(result, hidden_pool=True)
        _assert_balanced_events(events, set())
        if kind == "nested":
            assert computations["tl.ascend_gemm_v0"] == 6
            assert computations["copy_l0c_to_gm"] == 3
            assert clears == [1, 0] * 3
        else:
            assert computations["tl.ascend_gemm_v0"] == 4
            assert computations["copy_l0c_to_gm"] == 4
            assert clears == [1] * 4


@pytest.mark.parametrize(
    "dtype,m,n,k,k_l0,transpose_b",
    [
        ("float16", 16, 1360, 96, 96, False),
        ("float32", 16, 1360, 48, 48, False),
        ("float16", 16, 688, 192, 96, False),
        ("float32", 16, 688, 96, 48, False),
        ("float16", 13, 16, 128, 64, False),
        ("float16", 512, 16, 64, 64, False),
        ("float16", 256, 16, 128, 64, False),
        ("float16", 16, 512, 64, 64, True),
        ("float16", 16, 256, 128, 64, True),
    ],
)
def test_fractal_l0_slots_preserve_complete_invocations_and_internal_event_lifetimes(
    dtype,
    m,
    n,
    k,
    k_l0,
    transpose_b,
):
    program = _repeated_tiled_gemm(m, n, k, k_l0, dtype, transpose_b=transpose_b)
    original = _pre_sync(program)[0]
    after = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, after)
    gemm = _gemms(original)[0]
    template_fields = gemm.args[0].value.partition("<")[2].removesuffix(">").split(",")
    assert int(template_fields[2]) == m
    accumulator = gemm.args[3].args[1]
    old_allocations = _nodes(original.body, tir.Allocate)
    new_allocations = _nodes(after.body, tir.Allocate)
    old_allocation = next(node for node in old_allocations if node.buffer_var.same_as(accumulator))
    new_allocation = next(node for node in new_allocations if node.buffer_var.same_as(accumulator))
    tvm.ir.assert_structural_equal(old_allocation.extents, new_allocation.extents)
    required_bytes = ((m + 15) // 16) * 16 * n * 4
    assert prod(int(extent) for extent in old_allocation.extents) * 4 >= required_bytes
    for function in (original, after):
        assert int(function.attrs["size_map"][accumulator]) >= required_bytes
    for name in ("size_map", "address_map"):
        for root, value in original.attrs[name].items():
            tvm.ir.assert_structural_equal(value, after.attrs[name][root])
    events, computations, clears = _execute_k_protocol(after, hidden_pool=True)
    _assert_balanced_events(events, set())
    assert computations == {
        "copy_gm_to_l1": 4,
        "tl.ascend_gemm_v0": 2,
        "copy_l0c_to_gm": 2,
    }
    assert clears == [1, 1]


@pytest.mark.parametrize("m,n,transpose_a", [(32, 16, False), (32, 48, False), (16, 32, True)])
def test_flat_int8_internal_k_slots_include_both_fractals_of_transpose_loads(m, n, transpose_a):
    program = _repeated_tiled_gemm(
        m,
        n,
        256,
        128,
        "int8",
        repeats=1,
        transpose_a=transpose_a,
    )
    original = _pre_sync(program)[0]
    assert not _nodes(original.body, tir.For)
    after = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, after)
    assert f", {m}, {n}, 256," in _gemms(after)[0].args[0].value


@pytest.mark.parametrize(
    "m,n,k,k_l0,transpose_b",
    [
        (528, 16, 64, 64, False),
        (272, 16, 128, 64, False),
        (16, 528, 64, 64, True),
        (16, 512, 128, 64, True),
    ],
)
def test_fractal_l0_footprints_exceeding_single_or_ping_pong_slots_are_rejected(
    m,
    n,
    k,
    k_l0,
    transpose_b,
):
    program = _repeated_tiled_gemm(m, n, k, k_l0, transpose_b=transpose_b)
    original = _pre_sync(program)[0]
    with pytest.raises(tvm.error.TVMError, match="(?i)(physical footprint|L0A|L0B|capacity)"):
        _run(original, target_kind="llvm")


@pytest.mark.parametrize("kind", [tir.ForKind.SERIAL, tir.ForKind.UNROLLED])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_dynamic_outer_loop_preserves_zero_trip_and_backedges(invocation_kernels, kind, count):
    original = invocation_kernels["nested"]
    runtime_count = tir.Var("runtime_count", "int32")
    # Every signed input is safe: the actual extent is clamped to the fixed
    # output capacity. No unchecked input assumption is added to the function.
    extent = tir.min(tir.max(runtime_count, 0), 3)
    original = _rewrite_outer_loop(
        original,
        lambda loop: tir.For(loop.loop_var, loop.min, extent, kind, loop.body),
    )
    original = tir.PrimFunc(
        [*original.params, runtime_count],
        original.body,
        original.ret_type,
        original.buffer_map,
        original.attrs,
    )
    result = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, result)
    events, computations, clears = _execute_k_protocol(
        result,
        {runtime_count: count},
        hidden_pool=True,
    )
    _assert_balanced_events(events, set())
    assert computations["tl.ascend_gemm_v0"] == 2 * count
    assert computations["copy_l0c_to_gm"] == count
    assert clears == [1, 0] * count


def test_independent_user_buffers_still_require_hidden_pool_completion(invocation_kernels):
    result = _run(invocation_kernels["independent"], target_kind="llvm")
    first, second = _gemms(result)
    first_roots = {pointer.args[1] for pointer in first.args[1:4]}
    second_roots = {pointer.args[1] for pointer in second.args[1:4]}
    assert first_roots.isdisjoint(second_roots)
    assert any(event == "M_MTE1" for _, event, _ in _events(result))

    def remove_hidden_order(statement):
        if isinstance(statement, tir.Evaluate) and isinstance(statement.value, tir.Call):
            call = statement.value
            is_generated_flag = _operation(call) in (
                "tl.ascend_auto_set_flag",
                "tl.ascend_auto_wait_flag",
            )
            if is_generated_flag and call.args[0].value == "M_MTE1":
                return tir.Evaluate(0)
        return None

    corrupted = result.with_body(
        tir.stmt_functor.ir_transform(
            result.body,
            remove_hidden_order,
            None,
            ["tir.Evaluate"],
        ),
    )
    with pytest.raises(AssertionError, match="completion edge missing"):
        _execute_k_protocol(corrupted, hidden_pool=True)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "shape,tile_k",
    [((1, 16, 64), 64), ((13, 11, 47), 64), ((13, 11, 47), 47)],
)
def test_partial_refresh_uses_real_source_view_and_full_padded_destination(dtype, shape, tile_k):
    original = _pre_sync(_padded_refresh_gemm(*shape, dtype=dtype, tile_k=tile_k))[0]
    result = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, result)
    copies = [call for call in _calculation_calls(result) if _operation(call) == "copy_gm_to_l1"]
    assert len(copies) == 4
    rows, columns, reduction = shape
    for call, expected in zip(copies[2:], [(rows, reduction), (reduction, columns)]):
        source = call.args[1].args[1]
        buffer = next(buf for buf in result.buffer_map.values() if buf.data.same_as(source))
        # Scalar-origin access_ptr extent describes the requested tile. The
        # real GM capacity comes from its parameter buffer, independently of
        # that annotation; the helper masks describe the actual DMA rows/cols.
        assert prod(int(dimension) for dimension in buffer.shape) == prod(expected)
        assert (int(call.args[4]), int(call.args[5])) == expected


def _rewrite_partial_a_copy(function, rewrite):
    copies = [call for call in _calculation_calls(function) if _operation(call) == "copy_gm_to_l1"]
    partial = copies[2]

    def visit(statement):
        if not isinstance(statement, tir.Evaluate):
            return None
        if isinstance(statement.value, tir.Call) and statement.value.same_as(partial):
            return tir.Evaluate(rewrite(partial))
        return None

    return function.with_body(
        tir.stmt_functor.ir_transform(function.body, visit, None, ["tir.Evaluate"]),
    )


@pytest.mark.parametrize(
    "change",
    [
        "zero_means_full",
        "too_many_rows",
        "short_stride",
        "wide_stride",
        "source_end",
        "destination_view",
    ],
)
def test_partial_refresh_rejects_actual_dma_or_destination_contract_violation(change):
    original = _pre_sync(_padded_refresh_gemm())[0]

    def rewrite(call):
        arguments = list(call.args)
        if change in ("zero_means_full", "too_many_rows"):
            arguments[4] = tir.IntImm("int32", 0 if change == "zero_means_full" else 17)
        elif change == "short_stride":
            arguments[3] = tir.IntImm("int32", 46)
        elif change == "wide_stride":
            arguments[3] = tir.IntImm("int64", 1 << 32)
        else:
            index = 1 if change == "source_end" else 2
            pointer = arguments[index]
            fields = list(pointer.args)
            fields[2] = tir.IntImm("int32", 1)
            arguments[index] = tir.Call(pointer.dtype, pointer.op, fields)
        return tir.Call(call.dtype, call.op, arguments)

    with pytest.raises(tvm.error.TVMError, match="(?i)(Composite|GM-to-L1|refresh)"):
        _run(_rewrite_partial_a_copy(original, rewrite), target_kind="llvm")


def test_partial_refresh_still_requires_the_complete_physical_l1_allocation():
    original = _pre_sync(_padded_refresh_gemm())[0]
    sizes = dict(original.attrs["size_map"].items())
    sizes[_gemms(original)[0].args[1].args[1]] = 128
    with pytest.raises(tvm.error.TVMError, match="(?i)(padded physical footprint|allocation)"):
        _run(original.with_attr("size_map", sizes), target_kind="llvm")


@pytest.mark.parametrize("operand", [1, 3])
def test_generic_invocation_checks_real_allocate_even_when_size_map_claims_more(
    invocation_kernels,
    operand,
):
    original = invocation_kernels["nested"]
    root = _gemms(original)[0].args[operand].args[1]

    def shorten_allocation(node):
        if isinstance(node, tir.Allocate) and node.buffer_var.same_as(root):
            return tir.Allocate(
                node.buffer_var,
                node.dtype,
                [tir.IntImm("int32", 1)],
                node.condition,
                node.body,
                node.annotations,
            )
        return None

    changed = original.with_body(
        tir.stmt_functor.ir_transform(
            original.body,
            shorten_allocation,
            None,
            ["tir.Allocate"],
        ),
    )
    actual_sizes, original_sizes = changed.attrs["size_map"], original.attrs["size_map"]
    assert len(actual_sizes) == len(original_sizes)
    for storage, size in original_sizes.items():
        assert storage in actual_sizes
        tvm.ir.assert_structural_equal(actual_sizes[storage], size)
    with pytest.raises(tvm.error.TVMError, match="(?i)(allocation|footprint|pool)"):
        _run(changed, target_kind="llvm")


@pytest.fixture(scope="module")
def nonaligned_l1_kernel():
    return _pre_sync(_padded_refresh_gemm(13, 11, 47, tile_k=47))[0]


def _with_local_allocation_elements(function, storage, elements):
    replaced = []

    def replace(node):
        if isinstance(node, tir.Allocate) and node.buffer_var.same_as(storage):
            replaced.append(node)
            return tir.Allocate(
                node.buffer_var,
                node.dtype,
                [tir.IntImm("int32", elements)],
                node.condition,
                node.body,
                node.annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(function.body, replace, None, ["tir.Allocate"])
    assert len(replaced) == 1
    return function.with_body(body)


@pytest.mark.parametrize("operand", [1, 2])
def test_l1_logical_allocate_uses_its_matching_physical_padding_descriptor(
    nonaligned_l1_kernel,
    operand,
):
    original = nonaligned_l1_kernel
    storage = _gemms(original)[0].args[operand].args[1]
    logical_shape = original.attrs["initial_buffer_shapes"][storage]
    logical_elements = prod(int(dimension) for dimension in logical_shape)
    allocations = _nodes(original.body, tir.Allocate)
    actual = next(node for node in allocations if node.buffer_var.same_as(storage))
    assert prod(int(extent) for extent in actual.extents) > logical_elements
    changed = _with_local_allocation_elements(original, storage, logical_elements)
    result = _run(changed, target_kind="llvm")
    _assert_calculation_preserved(changed, result)
    for name in ("logic_buffer_shapes", "address_map", "size_map"):
        expected, actual_map = original.attrs[name], result.attrs[name]
        assert len(expected) == len(actual_map)
        for root, value in expected.items():
            assert root in actual_map
            tvm.ir.assert_structural_equal(actual_map[root], value)


@pytest.mark.parametrize("operand", [1, 2])
@pytest.mark.parametrize("change", ["unrelated_descriptor", "short_logical_allocate"])
def test_l1_padding_descriptor_cannot_justify_unrelated_geometry_or_short_allocation(
    nonaligned_l1_kernel,
    operand,
    change,
):
    original = nonaligned_l1_kernel
    call = _gemms(original)[0]
    storage = call.args[operand].args[1]
    logical_shape = original.attrs["initial_buffer_shapes"][storage]
    logical_elements = prod(int(dimension) for dimension in logical_shape)
    elements = logical_elements - int(change == "short_logical_allocate")
    changed = _with_local_allocation_elements(original, storage, elements)
    if change == "unrelated_descriptor":
        other = call.args[3 - operand].args[1]
        descriptors = dict(original.attrs["logic_buffer_shapes"].items())
        descriptors[storage] = descriptors[other]
        changed = changed.with_attr("logic_buffer_shapes", descriptors)
    with pytest.raises(tvm.error.TVMError, match="(?i)(allocation|footprint|descriptor)"):
        _run(changed, target_kind="llvm")


@pytest.mark.parametrize("operand,pool_bytes", [(1, 512 * 1024), (3, 128 * 1024)])
def test_generic_invocation_checks_physical_address_plus_span(
    invocation_kernels,
    operand,
    pool_bytes,
):
    original = invocation_kernels["nested"]
    root = _gemms(original)[0].args[operand].args[1]
    addresses = dict(original.attrs["address_map"].items())
    addresses[root] = tir.IntImm("int64", pool_bytes - int(original.attrs["size_map"][root]) // 2)
    with pytest.raises(tvm.error.TVMError, match="(?i)(allocation|footprint|pool)"):
        _run(original.with_attr("address_map", addresses), target_kind="llvm")


def test_general_invocation_cannot_borrow_the_existing_mixed_publication_profile():
    @T.prim_func
    def mixed(
        A: T.Tensor((16, 64), "float16"),
        B: T.Tensor((64, 16), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
            a = T.alloc_L1((16, 64), "float16")
            b = T.alloc_L1((64, 16), "float16")
            c = T.alloc_L0C((16, 16), "float32")
            vector = T.alloc_ub((64,), "float32")
            with T.Scope("C"):
                for _repeat in T.serial(2):
                    T.copy(A, a)
                    T.copy(B, b)
                    T.gemm_v0(a, b, c, init=True, kL0Size=64)
            with T.Scope("V"):
                T.tile.fill(vector, 1)

    original = _pre_sync(mixed)[0]
    assert len(_gemms(original)) == 1
    with pytest.raises(tvm.error.TVMError, match="General GEMM invocation.*one Cube"):
        _run(original, target_kind="llvm")


def test_no_branch_normalization_does_not_audit_an_unused_narrow_integer_let():
    from test_tilelang_ascend_language_gemm_v0_n_tiling import gemm_nt

    original = _pre_sync(gemm_nt(16, 16, 64, "float16", "float"))[0]
    source = tir.Var("unrelated_input", "int32")
    value = tir.Var("unrelated_value", "int32")
    original = tir.PrimFunc(
        [*original.params, source],
        tir.LetStmt(value, source + 1, original.body),
        original.ret_type,
        original.buffer_map,
        original.attrs,
    )
    result = _run(original, target_kind="llvm")
    _assert_calculation_preserved(original, result)
    lets = [node for node in _nodes(result.body, tir.LetStmt) if node.var.same_as(value)]
    assert len(lets) == 1
    tvm.ir.assert_structural_equal(lets[0].value, source + 1)
