# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Fresh Cython roles preserve complete legacy Flat publication certificates.

Compare the real Developer lowering with its v1 baseline, including the WAR
completion events needed for potentially aliased external outputs. Richer
programs must still obtain a Closed certificate before using all fresh facts.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import tilelang
from tilelang import language as T
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_FRESH_CONTRACT_VERSION,
    CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION,
    CYTHON_WORKSPACE_INDICES_ATTR,
    prepare_auto_gm_contract,
)
from test_ascend_upstream_sync_cross_core import _commands, _intrinsic, _name, _run
from test_ascend_upstream_sync_cross_core_owner_tree import (
    _byte_completion_oracle,
    _closed_tree,
)
from test_ascend_upstream_sync_fresh_fact_scope import (
    _assert_same_local_program,
    _lower_and_capture,
)
from test_ascend_upstream_sync_runtime_contract import _copy, _local_events, _mixed

tir = tvm.tir
_PASS_CONFIGS = {
    "tl.ascend_sync_backend": "upstream",
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_memory_planning": False,
}


@pytest.fixture(scope="module", autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def flat_developer_program():
    # Load the original factory without importing its device-only skip markers
    # or probing a device. Preserve its AST/source path for T.prim_func parsing.
    source = Path(__file__).with_name("test_ascend_upstream_sync_cross_core_device.py")
    tree = ast.parse(source.read_text(), filename=str(source))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_flat_developer_gemm_exp":
            definitions.append(node)
    assert len(definitions) == 1
    namespace = {
        "__name__": "__flat_compat_developer__",
        "__file__": str(source),
        "tilelang": tilelang,
        "tvm": tvm,
        "T": T,
        "_M": 32,
        "_N": 16,
        "_K": 64,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    program = namespace["_flat_developer_gemm_exp"](explicit_noalias=False)
    assert len(program.params) == 3 and "tir.noalias" not in program.attrs
    return program


@pytest.fixture(scope="module")
def flat_developer_baseline(flat_developer_program):
    prepared = prepare_auto_gm_contract(flat_developer_program, "cython", "ascendc")
    return _lower_and_capture(prepared, SimpleNamespace(target="ascendc"), _PASS_CONFIGS)


@pytest.mark.parametrize("workspace", [None, [-1], [3], [-1, 3, -1]])
def test_real_developer_external_output_keeps_the_v1_flat_program(
    flat_developer_program,
    flat_developer_baseline,
    workspace,
):
    original_recipe = None if workspace is None else workspace[:]
    prepared = prepare_auto_gm_contract(
        flat_developer_program,
        "cython",
        "ascendc",
        out_idx=[],
        workspace_idx=workspace,
    )
    captured = _lower_and_capture(prepared, SimpleNamespace(target="ascendc"), _PASS_CONFIGS)
    assert workspace == original_recipe
    before, after = captured["before"], captured["after"]
    assert len(before.params) == 4
    assert [int(index) for index in before.attrs["auto_gm_indices"]] == [3]
    expected_version = AUTO_GM_FRESH_CONTRACT_VERSION
    if workspace is not None:
        expected_version = CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION
    assert before.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == expected_version
    if workspace is not None:
        assert [int(index) for index in before.attrs[CYTHON_WORKSPACE_INDICES_ATTR]] == workspace
    _assert_same_local_program(
        flat_developer_baseline["after"],
        after,
        compare_physical_attrs=False,
    )
    # GEMM already connects the input reads to FIX through its Cube dataflow;
    # the v1 baseline therefore needs no additional direct MTE2_FIX event.
    events = _local_events(after)
    assert "MTE2_MTE1" in events and "M_FIX" in events
    assert "S_MTE2" in events


@pytest.mark.parametrize(
    "results,workspaces",
    [
        ([], [3]),  # Unused explicit fresh parameter, not an automatic workspace.
        ([], [0]),  # C reads this explicit fresh parameter before publication.
        ([2], []),  # A real fresh output may retain the conservative WAR proof.
        ([3], [0, 3]),  # Nonautomatic roles also overlap one another.
    ],
)
def test_nonautomatic_fresh_roles_do_not_change_a_complete_flat_certificate(results, workspaces):
    program = _mixed()
    baseline = _run(prepare_auto_gm_contract(program, "cython", "ascendc"))
    prepared = prepare_auto_gm_contract(
        program,
        "cython",
        "ascendc",
        out_idx=results,
        workspace_idx=workspaces,
    )
    actual = _run(prepared)
    _assert_same_local_program(baseline, actual)
    # Body equivalence plus this WAR edge distinguishes conservative Flat from
    # Closed planning, which would use output/source freshness to remove it.
    assert "MTE2_FIX" in _local_events(actual)
    assert "S_MTE2" in _local_events(actual)
    assert "tir.noalias" not in actual.attrs
    tvm.ir.assert_structural_equal(actual, _run(actual))


def _canonical_flat(*, extra_vector_read=False, output_pitch=64, late_cube_read=False):
    # Keep _mixed's publication/WAR topology, with real aligned typed DMA so
    # the Closed byte audit can independently certify or reject this profile.
    vid = tir.Var("vid", "int32")
    source = tir.decl_buffer((16, 16), "float32", name="source")
    workspace = tir.decl_buffer((16, 16), "float32", name="workspace")
    output = tir.decl_buffer((8, 16), "float32", name="output")
    l1 = tir.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    acc = tir.decl_buffer((16, 16), "float32", name="acc", scope="wmma.accumulator")
    local = tir.decl_buffer((4, 16), "float32", name="local", scope="shared.ub")
    cube_read = _copy(
        "tl::ascend::copy_gm_to_l1<float,16,16>",
        source,
        l1,
        arguments=(16, 16, 16, 16, 16),
    )
    cube = [
        cube_read,
        _copy(
            "tl::ascend::copy_l0c_to_gm<float,float,layout::RowMajor,16,16,0>",
            acc,
            workspace,
            arguments=(16, 16, 16, 16, 16, 0, 0),
        ),
        _intrinsic("tl.ascend_auto_set_cross_flag", 2, "FIX", 3),
    ]
    if late_cube_read:
        cube.append(cube_read)
    vector = [_intrinsic("tl.ascend_auto_wait_cross_flag", 3, "MTE2")]
    for buffer in [workspace, source] if extra_vector_read else [workspace]:
        vector.append(
            _copy(
                "tl::ascend::copy_gm_to_ub<float,16,4>",
                buffer,
                local,
                source_offset=vid * 64,
                arguments=(16, 4, 16, tir.const(0, "float32"), 4, 16),
            ),
        )
    vector.extend(
        [
            _intrinsic("tl.ascend_abs", local.access_ptr("w"), local.access_ptr("r"), 64),
            _copy(
                "tl::ascend::copy_ub_to_gm<float,16,4>",
                local,
                output,
                destination_offset=vid * output_pitch,
                arguments=(16, 4, 16),
            ),
        ],
    )
    body = tir.SeqStmt(
        [
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, tir.SeqStmt(cube)),
            tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 1, tir.SeqStmt(vector)),
        ],
    )
    for buffer in (l1, acc, local):
        body = tir.Allocate(buffer.data, buffer.dtype, buffer.shape, tir.const(True, "bool"), body)
    for tag, variable, extent in (
        ("threadIdx.x", vid, 2),
        ("blockIdx.x", tir.Var("bx", "int32"), 1),
    ):
        axis = tir.IterVar(None, variable, tir.IterVar.ThreadIndex, tag)
        body = tir.AttrStmt(axis, "thread_extent", extent, body)
    buffers = [source, workspace, output]
    handles = [tir.Var(f"arg{index}", "handle") for index in range(len(buffers))]
    return (
        tir.PrimFunc(handles, body, buffer_map=dict(zip(handles, buffers)))
        .with_attr("auto_gm_indices", [1])
        .with_attr("address_map", {l1.data: 0, acc.data: 0, local.data: 0})
        .with_attr("size_map", {l1.data: 1024, acc.data: 1024, local.data: 256})
        .with_attr("npu_cv_ratio", tir.StringImm("cv_1_2"))
    )


def test_vector_read_from_nonautomatic_input_still_requires_closed_admission():
    program = _canonical_flat(extra_vector_read=True)
    prepared = prepare_auto_gm_contract(program, "cython", "ascendc", out_idx=[2])
    actual = _run(prepared)
    # Flat admits V GM reads only from auto workspaces, whereas Closed proves
    # this read-only input disjoint from its fresh writes. Success requires the
    # complete Closed certificate, even though the control tree remains flat.
    assert "S_MTE2" in _local_events(actual)
    assert "MTE2_FIX" not in _local_events(actual)
    cross = [call for _, call in _commands(actual.body) if "cross_flag" in _name(call)]
    assert len(cross) == 2
    tvm.ir.assert_structural_equal(actual, _run(actual))


def test_multiple_workgroups_keep_the_closed_byte_ownership_certificate():
    result = _run(_closed_tree(groups=2, wait_pipe="MTE2"))
    counts = _byte_completion_oracle(result, 2)
    assert counts["copy_l0c_to_gm"] == 4
    assert counts["copy_ub_to_gm"] == 32


@pytest.mark.parametrize(
    "options,closed_reason,flat_reason",
    [
        (
            {"output_pitch": 0},
            "GM rectangles may overlap across AIV executions",
            "AIV output footprints may overlap",
        ),
        (
            {"late_cube_read": True},
            "requires complete Cube GM effects before Set",
            "requires every C input read before Set and V output write after Wait",
        ),
    ],
)
def test_failed_closed_certificate_does_not_fall_back_to_flat(options, closed_reason, flat_reason):
    program = _canonical_flat(**options)
    prepared = prepare_auto_gm_contract(program, "cython", "ascendc", out_idx=[2])
    with pytest.raises(tvm.error.TVMError, match="Legacy closed C/V publication") as rejected:
        _run(prepared)
    message = str(rejected.value)
    assert closed_reason in message and flat_reason in message
    # Keep the actual Closed failure primary while retaining the independently
    # rejected Flat profile's reason. Neither certificate failure permits fallback.
    assert message.index(closed_reason) < message.index(flat_reason)


def test_malformed_role_recipe_is_rejected_before_selecting_a_planner():
    prepared = prepare_auto_gm_contract(_mixed(), "cython", "ascendc", workspace_idx=[3])
    malformed = prepared.with_attr(CYTHON_WORKSPACE_INDICES_ATTR, [tir.IntImm("int64", -5)])
    with pytest.raises(tvm.error.TVMError, match="fresh allocation index.*final signature"):
        _run(malformed)
