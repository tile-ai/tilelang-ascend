# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU contracts for fresh Cython workspaces and aliased user parameters."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_FRESH_CONTRACT_VERSION,
)
from test_ascend_upstream_sync_cross_core import _commands, _intrinsic, _name, _run

tir = tvm.tir


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and the migrated sync engine")


def _contract(function: tir.PrimFunc, indices: object = (1,)) -> tir.PrimFunc:
    return function.with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tir.StringImm(AUTO_GM_FRESH_CONTRACT_VERSION)).with_attr(
        "auto_gm_indices", list(indices)
    )


def _copy(
    name: str,
    source: tir.Buffer,
    destination: tir.Buffer,
    *,
    source_offset: object = 0,
    destination_offset: object = 0,
    arguments: tuple[object, ...] = (),
) -> tir.Stmt:
    return tir.Evaluate(
        tir.call_extern(
            "handle",
            name,
            source.access_ptr("r", offset=source_offset),
            destination.access_ptr("w", offset=destination_offset),
            *arguments,
        )
    )


def _single_resource() -> tir.PrimFunc:
    source = tir.decl_buffer((64,), "float32", name="source")
    workspace = tir.decl_buffer((64,), "float32", name="workspace")
    first = tir.decl_buffer((64,), "float32", name="first", scope="shared.ub")
    second = tir.decl_buffer((64,), "float32", name="second", scope="shared.ub")
    body = tir.SeqStmt(
        [
            _copy("copy_ub_to_gm", first, workspace),
            _copy("tl::ascend::copy_gm_to_ub<float,64>", source, second),
        ]
    )
    body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, body)
    for buffer in (first, second):
        body = tir.Allocate(buffer.data, buffer.dtype, [64], tir.const(True, "bool"), body)
    # The handle parameter is deliberately distinct from buffer.data. The
    # contract uses lowered parameter positions, never a Var/name guess.
    handles = [tir.Var("source_handle", "handle"), tir.Var("workspace_handle", "handle")]
    return (
        tir.PrimFunc(handles, body, buffer_map=dict(zip(handles, (source, workspace))))
        .with_attr("address_map", {first.data: 0, second.data: 256})
        .with_attr("size_map", {first.data: 256, second.data: 256})
    )


def _local_events(function: tir.PrimFunc) -> list[str]:
    return [call.args[0].value for _, call in _commands(function.body) if _name(call) == "tl.ascend_auto_set_flag"]


def test_only_versioned_allocator_contract_proves_workspace_disjoint() -> None:
    original = _single_resource()
    assert "MTE3_MTE2" in _local_events(_run(original))
    assert "MTE3_MTE2" in _local_events(_run(original.with_attr("auto_gm_indices", [1])))
    result = _run(_contract(original))
    assert "MTE3_MTE2" not in _local_events(result)
    assert "tir.noalias" not in result.attrs


@pytest.mark.parametrize("indices", [[-1], [2], [1, 1], [True], [0, 1, 0]])
def test_invalid_fresh_workspace_parameter_positions_reject(indices: list[object]) -> None:
    with pytest.raises(tvm.error.TVMError, match="fresh-allocation contract"):
        _run(_contract(_single_resource(), indices))


@pytest.mark.parametrize("version", ["", "cython-auto-gm-v2", 1])
def test_unknown_fresh_workspace_schema_rejects(version: object) -> None:
    function = _single_resource().with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, version)
    with pytest.raises(tvm.error.TVMError, match="fresh-allocation contract"):
        _run(function)


def test_missing_fresh_workspace_indices_are_not_an_empty_list() -> None:
    function = _single_resource().with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tir.StringImm(AUTO_GM_FRESH_CONTRACT_VERSION))
    with pytest.raises(tvm.error.TVMError, match="requires auto_gm_indices"):
        _run(function)


@pytest.mark.parametrize("change", ["remove_contract", "change_version", "change_indices"])
def test_sync_proof_cannot_survive_allocator_contract_changes(change: str) -> None:
    synchronized = _run(_contract(_single_resource()))
    if change == "remove_contract":
        modified = synchronized.without_attr(AUTO_GM_FRESH_CONTRACT_ATTR)
    elif change == "change_version":
        modified = synchronized.with_attr(AUTO_GM_FRESH_CONTRACT_ATTR, tir.StringImm("cython-auto-gm-v2"))
    else:
        modified = synchronized.with_attr("auto_gm_indices", [])
    with pytest.raises(tvm.error.TVMError, match="inputs changed after synchronization"):
        _run(modified)


def _mixed(
    *,
    grid: int = 1,
    output_pitch: int = 64,
    output_stride: int = 64,
    output_rows: int = 1,
    output_columns: int = 64,
    vector_external_read: bool = False,
    vector_second_output: bool = False,
    late_cube_read: bool = False,
) -> tir.PrimFunc:
    vid = tir.Var("vid", "int32")
    source = tir.decl_buffer((128,), "float32", name="source")
    workspace = tir.decl_buffer((128,), "float32", name="workspace")
    output = tir.decl_buffer((128,), "float32", name="output")
    another = tir.decl_buffer((128,), "float32", name="another")
    l1 = tir.decl_buffer((128,), "float32", name="l1", scope="shared.l1")
    accumulator = tir.decl_buffer((128,), "float32", name="acc", scope="wmma.accumulator")
    local = tir.decl_buffer((64,), "float32", name="local", scope="shared.ub")
    cube_read = _copy("copy_gm_to_l1", source, l1)
    cube = [
        cube_read,
        _copy("copy_l0c_to_gm", accumulator, workspace),
        _intrinsic("tl.ascend_auto_set_cross_flag", 2, "FIX", 3),
    ]
    if late_cube_read:
        cube.append(cube_read)
    vector = [
        _intrinsic("tl.ascend_auto_wait_cross_flag", 3, "MTE2"),
        _copy(
            "tl::ascend::copy_gm_to_ub<float,64>",
            source if vector_external_read else workspace,
            local,
            source_offset=vid * 64,
        ),
        _intrinsic("tl.ascend_abs", local.access_ptr("w"), local.access_ptr("r"), 64),
        _copy(
            "tl::ascend::copy_ub_to_gm<float,64>",
            local,
            output,
            destination_offset=vid * output_pitch,
            arguments=(output_stride, output_rows, output_columns),
        ),
    ]
    if vector_second_output:
        vector.append(_copy("copy_ub_to_gm", local, another))
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, tir.SeqStmt(cube)),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.SeqStmt(vector)),
        ]
    )
    for buffer in (l1, accumulator, local):
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    body = tir.AttrStmt(tir.IterVar(None, vid, tir.IterVar.ThreadIndex, "threadIdx.x"), "thread_extent", 2, body)
    body = tir.AttrStmt(
        tir.IterVar(None, tir.Var("bx", "int32"), tir.IterVar.ThreadIndex, "blockIdx.x"),
        "thread_extent",
        grid,
        body,
    )
    parameters = [source, workspace, output, another]
    handles = [tir.Var(f"arg{index}", "handle") for index in range(len(parameters))]
    function = (
        tir.PrimFunc(handles, body, buffer_map=dict(zip(handles, parameters)))
        .with_attr("address_map", {l1.data: 0, accumulator.data: 0, local.data: 0})
        .with_attr("size_map", {l1.data: 512, accumulator.data: 512, local.data: 256})
        .with_attr("npu_cv_ratio", tir.StringImm("cv_1_2"))
    )
    return _contract(function)


def test_fresh_workspace_publication_orders_potential_input_output_war() -> None:
    result = _run(_mixed())
    assert "tir.noalias" not in result.attrs
    # C's input MTE2 read has no real link to the FIX workspace write. This
    # completion edge must come from the separate publication WAR shadow.
    assert "MTE2_FIX" in _local_events(result)
    assert "S_MTE2" in _local_events(result)
    cross = [call for _, call in _commands(result.body) if "cross_flag" in _name(call)]
    assert len(cross) == 2


def test_default_auto_target_keeps_the_existing_ascendc_dispatch() -> None:
    target = tvm.target.Target("llvm -model=auto")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        result = tilelang.transform.AscendInsertSync(target, "A3")(tvm.IRModule({"main": _mixed()}))["main"]
    assert "MTE2_FIX" in _local_events(result)
    assert "S_MTE2" in _local_events(result)


@pytest.mark.parametrize(
    "options,diagnostic",
    [
        ({"grid": 2}, "one static blockIdx.x"),
        ({"output_pitch": 0}, "footprints may overlap"),
        ({"output_pitch": 63}, "footprints may overlap"),
        ({"output_pitch": 65}, "exceeds its parameter"),
        ({"output_stride": 65}, "unsupported stride or mask"),
        ({"output_rows": 2}, "unsupported stride or mask"),
        ({"output_columns": 63}, "unsupported stride or mask"),
        ({"vector_external_read": True}, "reads only from trusted workspaces"),
        ({"vector_second_output": True}, "one external V GM write"),
        ({"late_cube_read": True}, "every C input read before Set"),
    ],
)
def test_fresh_workspace_admission_does_not_hide_other_alias_hazards(options: dict[str, object], diagnostic: str) -> None:
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_mixed(**options))
