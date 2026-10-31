# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU admission and placement tests for one existing flat C->V publication.

The fixture states a FIX write to a GM workspace and an MTE2 read from it;
it needs no composite GEMM support. No kernel is compiled or run on a device.
"""

import pytest

import tilelang
from tilelang import tvm

tir = tvm.tir


@pytest.fixture(autouse=True)
def _requires_proof_service() -> None:
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON and the migrated sync engine")


def _intrinsic(name: str, *arguments: object) -> tir.Evaluate:
    return tir.Evaluate(tir.call_intrin("handle", tvm.ir.Op.get(name), *arguments))


def _copy(name: str, source: tir.Buffer, destination: tir.Buffer) -> tir.Evaluate:
    arguments = []
    if "copy_gm_to_ub" in name:
        arguments = [64, 1, 64, tir.const(0, "float32"), 64]
    return tir.Evaluate(tir.call_extern("handle", name, source.access_ptr("r"), destination.access_ptr("w"), *arguments))


def _function(
    *,
    manual: bool = False,
    mode: int = 2,
    set_pipe: str = "FIX",
    wait_pipe: str = "MTE2",
    set_id: int = 3,
    wait_id: int = 3,
    noalias: bool = True,
    vector_extent: int = 2,
    writer_after_set: bool = False,
    reader_before_wait: bool = False,
    vector_write: bool = False,
    duplicate_set: bool = False,
    omit_wait: bool = False,
    cross_l1_alias: bool = False,
    control: str | None = None,
    version: int = 1,
    cube_reserve_all_s_mte2: bool = False,
    private_alias: str | None = None,
    topology_tag: str = "blockIdx.y",
    cv_ratio: str | tir.PrimExpr | None = None,
    double_topology: bool = False,
) -> tir.PrimFunc:
    vid = tir.Var("vid", "int32")
    workspace = tir.decl_buffer((64,), "float32", name="workspace")
    output = tir.decl_buffer((64,), "float32", name="output")
    accumulator = tir.decl_buffer((64,), "float32", name="accumulator", scope="wmma.accumulator")
    local = tir.decl_buffer((64,), "float32", name="local", scope="shared.ub")
    l1 = tir.decl_buffer((64,), "float32", name="l1", scope="shared.l1")
    l0a = tir.decl_buffer((64,), "float32", name="l0a", scope="wmma.matrix_a")
    if manual:
        publish = _intrinsic("tl.ascend_set_cross_flag", set_pipe, set_id, mode)
        acquire = _intrinsic("tl.ascend_wait_cross_flag", wait_id, wait_pipe)
    else:
        publish = _intrinsic("tl.ascend_auto_set_cross_flag", mode, set_pipe, set_id)
        acquire = _intrinsic("tl.ascend_auto_wait_cross_flag", wait_id, wait_pipe)
    writer = _copy("copy_l0c_to_gm", accumulator, workspace)
    reader = _copy("tl::ascend::copy_gm_to_ub<float, 64>", workspace, local)
    cube = [publish, writer] if writer_after_set else [writer, publish]
    if duplicate_set:
        cube.append(publish)
    if cube_reserve_all_s_mte2:
        reservations = []
        for flag_id in range(8):
            reservations.extend(
                [
                    _intrinsic("tl.ascend_set_flag", "S", "MTE2", flag_id),
                    _intrinsic("tl.ascend_wait_flag", "S", "MTE2", flag_id),
                ]
            )
        cube = reservations + cube
    vector = [reader, acquire] if reader_before_wait else [acquire, reader]
    if omit_wait:
        vector = [reader]
    vector.extend(
        [
            _intrinsic("tl.ascend_abs", local.access_ptr("w"), local.access_ptr("r"), 64),
            _copy("copy_ub_to_gm", local, output),
        ]
    )
    if vector_write:
        vector.append(_copy("copy_ub_to_gm", local, workspace))
    if cross_l1_alias:
        cube.insert(0, _copy("copy_l1_to_l0a", l1, l0a))
        vector.append(_copy("copy_ub_to_l1", local, l1))
    parameters = [workspace, output]
    saved = tir.Var("saved", "bool")
    control_c = tir.decl_buffer((1,), "bool", name="control_c")
    control_v = tir.decl_buffer((1,), "bool", name="control_v")
    if private_alias is not None:
        auxiliary = tir.decl_buffer((64,), "float32", name="auxiliary")
        parameters.append(auxiliary)
        # Both resource streams can use the same original Var object, but
        # their subcore ids and runtime snapshots are independent values.
        index = vid if private_alias == "subcore" else tir.Cast("int32", saved)
        cube.append(tir.BufferStore(auxiliary, tir.FloatImm("float32", 1), [index + 1]))
        vector.append(
            tir.LetStmt(
                tir.Var("unused_read", "float32"),
                tir.BufferLoad(auxiliary, [index]),
                tir.Evaluate(0),
            )
        )
        if private_alias == "snapshot":
            parameters += [control_c, control_v]
    vector_body = tir.SeqStmt(vector)
    if control == "for":
        vector_body = tir.For(tir.Var("i", "int32"), 0, 1, tir.ForKind.SERIAL, vector_body)
    elif control == "if":
        vector_body = tir.IfThenElse(tir.const(True, "bool"), vector_body, None)
    cube_body = tir.SeqStmt(cube)
    if private_alias == "snapshot":
        cube_body = tir.LetStmt(saved, tir.BufferLoad(control_c, [0]), cube_body)
        vector_body = tir.LetStmt(saved, tir.BufferLoad(control_v, [0]), vector_body)
    cube_body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, cube_body)
    vector_body = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, vector_body)
    body = tir.SeqStmt([cube_body, vector_body])
    allocated = [accumulator, local]
    if cross_l1_alias:
        allocated += [l1, l0a]
    for buffer in allocated:
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    subgroup = tir.IterVar(None, vid, tir.IterVar.ThreadIndex, topology_tag)
    body = tir.AttrStmt(subgroup, "thread_extent", vector_extent, body)
    if double_topology:
        other_tag = "threadIdx.x" if topology_tag == "blockIdx.y" else "blockIdx.y"
        other = tir.IterVar(None, tir.Var("other_vid", "int32"), tir.IterVar.ThreadIndex, other_tag)
        body = tir.AttrStmt(other, "thread_extent", 2, body)
    function = (
        tir.PrimFunc(
            [buffer.data for buffer in parameters],
            body,
            buffer_map={buffer.data: buffer for buffer in parameters},
        )
        .with_attr("tir.noalias", noalias)
        .with_attr("address_map", {buffer.data: 0 for buffer in allocated})
        .with_attr("size_map", {buffer.data: 256 for buffer in allocated})
        .with_attr("buffer_versions", {buffer.data: version for buffer in allocated})
    )
    if cv_ratio is not None:
        ratio = tir.StringImm(cv_ratio) if isinstance(cv_ratio, str) else cv_ratio
        function = function.with_attr("npu_cv_ratio", ratio)
    return function


def _run(function: tir.PrimFunc, *, platform: str = "A3") -> tir.PrimFunc:
    target = tvm.target.Target("c -keys=ascend -model=ascendc")
    with tvm.transform.PassContext(config={"tl.ascend_auto_sync": True, "tl.ascend_auto_sync_vs": True}):
        result = tilelang.transform.AscendInsertSync(target, platform)(tvm.IRModule({"main": function}))
    return result["main"]


def _commands(statement: tir.Stmt) -> list[tuple[int | None, tir.Call]]:
    result = []

    def visit(node: tir.Stmt, owner: int | None = None) -> None:
        if isinstance(node, tir.SeqStmt):
            for child in node.seq:
                visit(child, owner)
        elif isinstance(node, tir.AttrStmt):
            if node.attr_key == "resource_scope":
                owner = int(node.value)
            visit(node.body, owner)
        elif isinstance(node, (tir.Allocate, tir.DeclBuffer, tir.LetStmt)):
            visit(node.body, owner)
        elif isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            result.append((owner, node.value))

    visit(statement)
    return result


def _name(call: tir.Call) -> str:
    assert isinstance(call.op, tvm.ir.Op)
    return call.op.name


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("topology_tag,cv_ratio", [("blockIdx.y", None), ("threadIdx.x", "cv_1_2")])
def test_existing_publication_gets_local_completion_fences_without_new_cross_calls(
    manual: bool, platform: str, topology_tag: str, cv_ratio: str | None
) -> None:
    original = _function(manual=manual, topology_tag=topology_tag, cv_ratio=cv_ratio)
    result = _run(original, platform=platform)
    before = [call for _, call in _commands(original.body)]
    commands = _commands(result.body)
    retained = [
        call for _, call in commands if _name(call) not in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag", "tl.ascend_auto_barrier")
    ]
    assert len(retained) == len(before)
    assert all(new.same_as(old) for new, old in zip(retained, before))
    cross_calls = [(owner, call) for owner, call in commands if "cross_flag" in _name(call)]
    assert len(cross_calls) == 2
    assert [owner for owner, _ in cross_calls] == [0, 1]
    set_index = next(i for i, (_, call) in enumerate(commands) if call.same_as(cross_calls[0][1]))
    wait_index = next(i for i, (_, call) in enumerate(commands) if call.same_as(cross_calls[1][1]))
    assert any(
        owner == 0 and _name(call) == "tl.ascend_auto_barrier" and call.args[0].value == "PIPE_FIX" for owner, call in commands[:set_index]
    )
    scalar_sets = [
        i
        for i, (owner, call) in enumerate(commands)
        if owner == 1 and _name(call) == "tl.ascend_auto_set_flag" and call.args[0].value == "S_MTE2"
    ]
    scalar_waits = [
        i
        for i, (owner, call) in enumerate(commands)
        if owner == 1 and _name(call) == "tl.ascend_auto_wait_flag" and call.args[0].value == "S_MTE2"
    ]
    reader_index = next(
        i for i, (_, call) in enumerate(commands) if _name(call) == "tir.call_extern" and "copy_gm_to_ub" in call.args[0].value
    )
    assert scalar_sets and scalar_waits
    assert wait_index < scalar_sets[0] < scalar_waits[0] < reader_index
    assert tvm.ir.structural_equal(result, _run(result, platform=platform))


def test_local_event_reservations_are_independent_between_cube_and_vector() -> None:
    # C occupies all eight S_MTE2 slots. V must still allocate its own S_MTE2
    # handoff after CrossCoreWaitFlag; a global reservation table would fail.
    result = _run(_function(cube_reserve_all_s_mte2=True))
    assert any(
        owner == 1 and _name(call) == "tl.ascend_auto_set_flag" and call.args[0].value == "S_MTE2" for owner, call in _commands(result.body)
    )


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"noalias": False}, "tir.noalias"),
        ({"mode": 4}, "mode-2"),
        ({"set_pipe": "MTE2"}, "mode-2 FIX"),
        ({"wait_pipe": ""}, "MTE2 consumer intent"),
        ({"set_id": 16}, "cross event ID"),
        ({"wait_id": 4}, "matching literal C->V channel"),
        ({"duplicate_set": True}, "exactly one mode-2"),
        ({"omit_wait": True}, "matching literal C->V channel"),
        ({"writer_after_set": True}, "every C writer before Set"),
        ({"reader_before_wait": True}, "every V reader after Wait"),
        ({"vector_write": True}, "uncovered cross-resource"),
        ({"cross_l1_alias": True}, "uncovered cross-resource"),
        ({"vector_extent": 1}, "C1/V2|subgroup extent"),
        ({"control": "for"}, "flat input"),
        ({"control": "if"}, "flat input"),
        ({"version": 2}, "B1|version|provenance"),
    ],
)
def test_unsupported_or_uncovered_cross_protocol_is_rejected(kwargs: dict[str, object], diagnostic: str) -> None:
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_function(**kwargs))


@pytest.mark.parametrize("private_alias", ["subcore", "snapshot"])
def test_private_resource_variables_cannot_discharge_a_real_alias(private_alias: str) -> None:
    # With one shared proof identity, [x+1,x+2) and [x,x+1) appear disjoint.
    # C can observe x=0 while V observes x=1, making these real accesses alias.
    # The C write follows publication, so the existing channel cannot cover it.
    with pytest.raises(tvm.error.TVMError, match="every C writer before Set"):
        _run(_function(private_alias=private_alias))


def test_developer_subcore_coordinate_is_private_to_each_resource() -> None:
    with pytest.raises(tvm.error.TVMError, match="every C writer before Set"):
        _run(_function(topology_tag="threadIdx.x", cv_ratio="cv_1_2", private_alias="subcore"))


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"topology_tag": "threadIdx.x"}, "npu_cv_ratio.*cv_1_2"),
        ({"topology_tag": "threadIdx.x", "cv_ratio": "cv_1_1"}, "StringImm cv_1_2"),
        ({"cv_ratio": "cv_1_1"}, "StringImm cv_1_2"),
        (
            {"topology_tag": "threadIdx.x", "cv_ratio": tir.IntImm("int32", 2)},
            "StringImm cv_1_2",
        ),
        (
            {"topology_tag": "threadIdx.x", "cv_ratio": "cv_1_2", "vector_extent": 1},
            "C1/V2",
        ),
        ({"cv_ratio": "cv_1_2", "double_topology": True}, "unambiguous C1/V2"),
        (
            {"topology_tag": "threadIdx.x", "cv_ratio": "cv_1_2", "double_topology": True},
            "unambiguous C1/V2",
        ),
        ({"topology_tag": "threadIdx.y", "cv_ratio": "cv_1_2"}, "unsupported thread topology"),
    ],
)
def test_ambiguous_or_unproven_developer_topology_is_rejected(kwargs: dict[str, object], diagnostic: str) -> None:
    with pytest.raises(tvm.error.TVMError, match=diagnostic):
        _run(_function(**kwargs))
