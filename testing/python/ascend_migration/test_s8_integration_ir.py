"""Actual finite auto analysis/materialization and legacy isolation; no Bisheng/NPU."""

from pathlib import Path
import sys

import pytest

import tilelang
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from test_s8_kernels import (
    make_gemm_kernel,
    make_mixed_kernel,
    make_reduce_kernel,
    make_vector_kernel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ascend_migration"))
from _s8_npu import inspect_function, pass_configs  # noqa: E402

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")


def before_auto(function):
    with tvm.transform.PassContext(config=pass_configs("auto")):
        module = LowerAndLegalize(tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3")), TARGET)
        return tir.transform.PlanAndUpdateBufferAllocationLocation()(module)


def source_analysis(function):
    return tvm.get_global_func("tl.transform.AnalyzeAscendAutoSchedule")(function)


def lowered(function, strategy="auto"):
    with tvm.transform.PassContext(config=pass_configs(strategy)):
        return tilelang.lower(function, target="ascendc", platform="A3")


def final_verifier_input(function):
    """Capture the actual final IR after Simplify, before instruction selection."""
    captured = []
    original = tilelang.transform.AscendVerifyAutoSchedule

    def observe(*args, **kwargs):
        checker = original(*args, **kwargs)

        def apply(module):
            captured.append(module["main"])
            return checker(module)

        return apply

    tilelang.transform.AscendVerifyAutoSchedule = observe
    try:
        with tvm.transform.PassContext(config=pass_configs("auto")):
            module = LowerAndLegalize(tvm.IRModule.from_expr(function.with_attr("npu_platform", "A3")), TARGET)
            OptimizeForTarget(module, TARGET, "A3")
    finally:
        tilelang.transform.AscendVerifyAutoSchedule = original
    assert len(captured) == 1
    return captured[0]


def emitted_instances(function):
    instances = []

    def visit(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "tl.ascend_auto_schedule_task_id":
            assert int(node.value) == 1
            instances.append(int(node.node))

    tir.stmt_functor.post_order_visit(function.body, visit)
    return instances


def small_program(kind, trips=1, cores=1):
    if kind == "vector":
        return make_vector_kernel(trips, cores)
    if kind == "gemm":
        return make_gemm_kernel(trips, cores, m=64, n=64, k=32, k_l0=32)
    if kind == "reduce":
        return make_reduce_kernel(trips, cores)
    return make_mixed_kernel(trips, cores, operation=kind.removeprefix("mixed_"))


@pytest.mark.parametrize("kind", ["vector", "gemm", "reduce", "mixed_add", "mixed_reduce"])
def test_readonly_analysis_of_real_lowered_operations(kind):
    function = before_auto(small_program(kind))["main"]
    original = function
    before = tvm.ir.save_json(function)
    analysis = source_analysis(function)
    assert function.same_as(original) and tvm.ir.save_json(function) == before
    assert analysis["tasks"] and analysis["storages"]
    assert sorted(int(task["id"]) for task in analysis["tasks"]) == list(range(len(analysis["tasks"])))
    assert all(task["reads"] or task["writes"] for task in analysis["tasks"])


@pytest.mark.parametrize("trips", [0, 1, 2, 3])
def test_finite_source_loop_has_exact_operation_coverage(trips):
    analysis = source_analysis(before_auto(make_vector_kernel(trips))["main"])
    # Two independent Vector actors, four real operations per finite epoch.
    assert len(analysis["tasks"]) == trips * 2 * 4
    assert {tuple(int(value) for value in task["epoch"]) for task in analysis["tasks"]} == {(index,) for index in range(trips)}


@pytest.mark.parametrize("kind", ["gemm", "reduce", "mixed_add", "mixed_reduce"])
@pytest.mark.parametrize("trips", [0, 1])
def test_compound_zero_one_domain_has_actual_exact_once_emission(kind, trips):
    function = final_verifier_input(small_program(kind, trips=trips))
    record = inspect_function(function)
    # Count real emitted marker nodes, independently of the stored solver permutation.
    assert sorted(emitted_instances(function)) == list(range(record["task_count"]))
    if trips == 0:
        assert record["task_count"] == 0 and record["buffer_remap"] == []
        assert record["integer_verification"]["numeric_verifier_called"] is False
    else:
        assert record["task_count"] > 0


def test_static_guard_keeps_tail_output_unwritten_in_source_and_actual_instances():
    source = make_vector_kernel(3)
    guarded = []

    def replace(node):
        if isinstance(node, tir.For):
            guarded.append(node.loop_var)
            return tir.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                tir.IfThenElse(node.loop_var < 2, node.body, None),
                node.thread_binding,
                node.annotations,
            )
        return None

    source = source.with_body(tir.stmt_functor.ir_transform(source.body, None, replace))
    assert len(guarded) == 1
    function = final_verifier_input(source)
    record = inspect_function(function)
    assert record["task_count"] == 16
    assert sorted(emitted_instances(function)) == list(range(16))
    plan = function.attrs["tl.ascend_auto_schedule"]
    original = plan["source_function"]
    output = original.buffer_map[original.params[-1]].data
    written = {
        (int(effect["begin"]), int(effect["end"]))
        for task in plan["source_analysis"]["tasks"]
        for effect in task["writes"]
        if effect["data"].same_as(output)
    }
    # Two epochs, two 64-float halves: neither the skipped third tile nor the canary is written.
    assert written == {(index * 256, (index + 1) * 256) for index in range(4)}


@pytest.mark.parametrize("m,n,k", [(64, 64, 32), (128, 64, 32)])
def test_rectangular_gm_to_l1_footprint_matches_actual_tensor_geometry(m, n, k):
    function = before_auto(make_gemm_kernel(1, m=m, n=n, k=k, k_l0=32))["main"]
    analysis = source_analysis(function)
    a, b = [function.buffer_map[parameter].data for parameter in function.params[:2]]
    for variable, expected in ((a, m * k * 2), (b, k * n * 2)):
        effects = [effect for task in analysis["tasks"] for effect in task["reads"] if effect["data"].same_as(variable)]
        assert effects
        # DMA analysis emits exact per-row byte intervals, rather than a padded hull.
        cursor = 0
        for begin, end in sorted((int(effect["begin"]), int(effect["end"])) for effect in effects):
            assert begin == cursor and end > begin
            cursor = end
        assert cursor == expected


@pytest.mark.parametrize("kind", ["vector", "gemm", "reduce", "mixed_add", "mixed_reduce"])
def test_complete_new_backend_has_real_materialization_and_final_codegen(kind):
    artifact = lowered(small_program(kind, trips=3))
    records = []
    for function in artifact.device_mod.functions.values():
        if not isinstance(function, tir.PrimFunc):
            continue
        records.append(inspect_function(function))
        old_annotations = []

        def visit(node, old_annotations=old_annotations):
            if isinstance(node, tir.For):
                old_annotations.extend(str(key) for key in node.annotations if str(key).startswith("software_pipeline_"))

        tir.stmt_functor.post_order_visit(function.body, visit)
        assert not old_annotations
    assert records and all(record["model"] == "abstract_issue_step_v1" for record in records)
    if kind in ("vector", "gemm", "reduce"):
        assert any(record["within_actor_reordered"] for record in records)
        assert any(row["versions"] == 2 for record in records for row in record["buffer_remap"])
    source = artifact.kernel_source
    assert source and "PIPE_ALL" in source
    assert not any(token in source for token in ("DataCopyUB2L1", "DataCopyL0C2UB", "VFCall", "flag_spill"))


def test_materializer_does_not_silently_apply_versions_twice():
    with tvm.transform.PassContext(config=pass_configs("auto")):
        module = before_auto(make_vector_kernel(3))
        planned = tilelang.transform.AscendAutoSchedule(TARGET, "A3")(module)
        materialized = tilelang.transform.AscendMaterializeSchedule(TARGET, "A3")(planned)
        before = tvm.ir.save_json(materialized)
        try:
            repeated = tilelang.transform.AscendMaterializeSchedule(TARGET, "A3")(materialized)
        except tvm.error.TVMError as error:
            assert any(word in str(error).lower() for word in ("materializ", "already", "owner"))
        else:
            assert tvm.ir.save_json(repeated) == before
        assert tvm.ir.save_json(materialized) == before


def test_four_core_mixed_resource_masks_preserve_all_physical_domains_above_bit31():
    function = final_verifier_input(make_mixed_kernel(3, 4, operation="reduce"))
    record = inspect_function(function)
    tasks = list(function.attrs["tl.ascend_auto_schedule"]["source_analysis"]["tasks"])
    masks = record["problem"]["resource_flags"]
    domains = [{(int(task["group"]), int(task["actor"]), str(pipe)) for pipe in task["resources"]} for task in tasks]
    assert len(tasks) == len(masks) == 120
    assert max(masks) >= 1 << 32
    union = 0
    for mask, physical in zip(masks, domains):
        assert 0 <= mask <= (1 << 63) - 1 and mask.bit_count() == len(physical)
        union |= mask
    assert union.bit_count() == len(set().union(*domains)) == 40
    for i, first in enumerate(masks):
        for j, second in enumerate(masks):
            assert bool(first & second) == bool(domains[i] & domains[j])


@pytest.mark.parametrize("mutation", ["wrong_actor", "nested_conflict"])
def test_active_cube_call_in_conflicting_scope_is_rejected(mutation):
    function = make_gemm_kernel(1)

    def replace(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "resource_scope" and int(node.value) == 0:
            if mutation == "wrong_actor":
                return tir.AttrStmt(node.node, node.attr_key, 1, node.body)
            return tir.AttrStmt(
                node.node,
                node.attr_key,
                node.value,
                tir.AttrStmt(tir.const(0), "resource_scope", 1, node.body),
            )
        return None

    changed = function.with_body(tir.stmt_functor.ir_transform(function.body, None, replace))
    with pytest.raises(tvm.error.TVMError, match="(?i)scope|actor"):
        lowered(changed)


@pytest.mark.parametrize("platform", ["A5", "unknown"])
def test_native_auto_entry_rejects_other_platforms(platform):
    module = before_auto(make_vector_kernel(1))
    with (
        tvm.transform.PassContext(config=pass_configs("auto")),
        pytest.raises(tvm.error.TVMError),
    ):
        tilelang.transform.AscendAutoSchedule(TARGET, platform)(module)


@pytest.mark.parametrize("kind", ["gemm", "reduce"])
def test_legacy_remains_outside_the_complete_auto_contract(kind):
    artifact = lowered(small_program(kind), strategy="legacy")
    for function in artifact.device_mod.functions.values():
        assert not any(str(key).startswith("tl.ascend_auto_schedule") for key in function.attrs.keys())
