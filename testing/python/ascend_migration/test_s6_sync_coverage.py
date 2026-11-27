"""Fault injection separates source byte readiness from physical completion."""

import pytest

import tilelang
from tilelang import tvm

from test_s2_task_ir import _add, _buffer, _dma, _fill, _function
from test_s4_sync_coverage import _barrier, _flag
from test_s5_sync_coverage import _fixture
from test_s6_lifecycle import _analyze

tir = tvm.tir
TARGET = tvm.target.Target("c -keys=ascend -model=ascendc")
MANIFEST = "tl.ascend_constraint_plan"
REMAP = "tl.ascend_pipeline_buffer_remap"


def _strip_markers(function):
    def remove(attribute):
        if attribute.attr_key in ("tl.ascend_constraint_task_id", "tl.ascend_constraint_iteration"):
            return attribute.body
        return attribute

    return function.with_body(tir.stmt_functor.ir_transform(function.body, None, remove, ["tir.AttrStmt"]))


def _upgrade(function):
    manifest = dict(function.attrs[MANIFEST].items())
    contract = _analyze(_strip_markers(manifest["source_function"]))
    manifest.update(
        schema_version=3,
        source_function=contract["source_function"],
        source_graph=contract["source_graph"],
        lifetime_contract=contract,
    )
    return function.with_attr(MANIFEST, manifest)


def _verify(function):
    before = tvm.ir.save_json(function)
    with tvm.transform.PassContext(config={"tl.ascend_pipeline_planner": "constraint"}):
        result = tilelang.transform.AscendVerifyConstraintSync(TARGET, "A3")(tvm.IRModule.from_expr(function))["main"]
    assert result.same_as(function)
    assert tvm.ir.save_json(function) == before


@pytest.mark.parametrize("versions", [1, 2, 3])
@pytest.mark.parametrize("trips", [0, 1, 4])
def test_schema_three_recomputes_real_version_readiness(versions, trips):
    _verify(_upgrade(_fixture(versions, trips)))


def test_completed_overwrite_cannot_replace_the_expected_source_value():
    with pytest.raises(tvm.TVMError, match="reaching writer|overwritten|producer"):
        _verify(_upgrade(_fixture(2, 4, overwrite=True)))


def _unit(statement, unit):
    return tir.AttrStmt(tir.IntImm("int64", unit), "tl.ascend_constraint_task_id", 1, statement)


def _union_fixture(*, physical_alias=False, missing_completion=False):
    value = _buffer("logical_value", shape=(128,))
    unrelated = _buffer("unrelated_value", shape=(128,))
    output = _buffer("output", "global", shape=(128,))
    source_calls = [
        _fill(value),
        _fill(value, offset=64),
        _dma(value, output, count=128),
        _fill(unrelated, count=128),
    ]
    source = _function([value, unrelated, output], source_calls, noalias=True)
    contract = _analyze(source)
    order = [0, 1, 3, 2] if physical_alias else [0, 1, 2, 3]
    statements = []
    for unit in order:
        if not missing_completion:
            statements.append(_barrier("PIPE_ALL"))
        statements.append(_unit(source_calls[unit], unit))
    final = _function([value, unrelated, output], statements, noalias=True)
    final = final.with_attr("address_map", {value.data: 0, unrelated.data: 0 if physical_alias else 512})
    final = final.with_attr("size_map", {value.data: 512, unrelated.data: 512})
    return final.with_attr(REMAP, []).with_attr(
        MANIFEST,
        {
            "schema_version": 3,
            "source_function": contract["source_function"],
            "source_graph": contract["source_graph"],
            "candidates": [],
            "lifetime_contract": contract,
        },
    )


def test_two_partial_producers_can_define_one_complete_read():
    _verify(_union_fixture())


def test_all_segments_require_real_producer_completion():
    with pytest.raises(tvm.TVMError, match="Uncovered.*dependency"):
        _verify(_union_fixture(missing_completion=True))


def test_different_logical_storage_cannot_silently_replace_one_read_segment():
    with pytest.raises(tvm.TVMError, match="reaching writer|overwritten|producer"):
        _verify(_union_fixture(physical_alias=True))


def test_stored_readiness_records_do_not_replace_rebuilding_the_source():
    function = _union_fixture(physical_alias=True)
    manifest = dict(function.attrs[MANIFEST].items())
    contract = dict(manifest["lifetime_contract"].items())
    cases = []
    for original in contract["cases"]:
        case = dict(original.items())
        case["reads"] = []
        cases.append(case)
    contract["cases"] = cases
    manifest["lifetime_contract"] = contract
    with pytest.raises(tvm.TVMError, match="reaching writer|overwritten|producer"):
        _verify(function.with_attr(MANIFEST, manifest))


def test_stale_stored_owner_capacity_is_rejected_against_the_source():
    function = _union_fixture()
    manifest = dict(function.attrs[MANIFEST].items())
    contract = dict(manifest["lifetime_contract"].items())
    catalog = dict(contract["storage_catalog"].items())
    storages = [dict(storage.items()) for storage in catalog["storages"]]
    changed = False
    for storage in storages:
        if str(storage["scope"]) == "shared.ub" and not changed:
            storage["capacity_bytes"] = int(storage["capacity_bytes"]) + 32
            changed = True
    assert changed
    catalog["storages"] = storages
    contract["storage_catalog"] = catalog
    manifest["lifetime_contract"] = contract
    with pytest.raises(tvm.TVMError, match="catalog.*metadata differs"):
        _verify(function.with_attr(MANIFEST, manifest))


def _boolean_fixture(*, unmatched_wait=False):
    value = _buffer("boolean_value", shape=(64,))
    output = _buffer("output", "global", shape=(64,))
    predicate = tir.Var("predicate", "bool")
    producer, consumer = _fill(value), _dma(value, output)
    source = _function(
        [value, output],
        tir.IfThenElse(predicate, tir.SeqStmt([producer, consumer]), None),
        scalars=[predicate],
    )
    contract = _analyze(source)
    statements = [_barrier("PIPE_ALL"), _unit(producer, 0), _barrier("PIPE_ALL")]
    if unmatched_wait:
        statements.append(_flag("wait", "MTE2_V", 0))
    statements.append(_unit(consumer, 1))
    final = _function(
        [value, output],
        tir.IfThenElse(predicate, tir.SeqStmt(statements), None),
        scalars=[predicate],
    )
    return (
        final.with_attr("address_map", {value.data: 0})
        .with_attr("size_map", {value.data: 256})
        .with_attr(REMAP, [])
        .with_attr(
            MANIFEST,
            {
                "schema_version": 3,
                "source_function": contract["source_function"],
                "source_graph": contract["source_graph"],
                "candidates": [],
                "lifetime_contract": contract,
            },
        )
    )


def test_final_boolean_paths_preserve_conditional_source_readiness():
    _verify(_boolean_fixture())


def test_an_omitted_stored_boolean_case_cannot_hide_a_wait_without_source():
    function = _boolean_fixture(unmatched_wait=True)
    manifest = dict(function.attrs[MANIFEST].items())
    contract = dict(manifest["lifetime_contract"].items())
    contract["cases"] = list(contract["cases"])[:1]
    manifest["lifetime_contract"] = contract
    with pytest.raises(tvm.TVMError, match="wait without matching"):
        _verify(function.with_attr(MANIFEST, manifest))


def test_same_task_partial_physical_overlap_is_rejected_even_after_full_initialization():
    a, b = _buffer("input_storage", shape=(128,)), _buffer("output_storage", shape=(128,))
    producer, update = _fill(a, count=128), _add(b, a, a, count=128)
    contract = _analyze(_function([a, b], [producer, update]))
    function = _function([a, b], [_unit(producer, 0), _barrier("PIPE_ALL"), _unit(update, 1)])
    function = (
        function.with_attr("address_map", {a.data: 0, b.data: 128})
        .with_attr("size_map", {a.data: 512, b.data: 512})
        .with_attr(REMAP, [])
        .with_attr(
            MANIFEST,
            {
                "schema_version": 3,
                "source_function": contract["source_function"],
                "source_graph": contract["source_graph"],
                "candidates": [],
                "lifetime_contract": contract,
            },
        )
    )
    with pytest.raises(tvm.TVMError, match="same-task.*physical.*overlap"):
        _verify(function)
