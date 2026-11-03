# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Keep local event planning stable when explicit Cython roles acquire freshness.

The real RMS factory is loaded from its original AST without executing sample
tests or cloning/editing a kernel. Host lowering runs the full original pass
pipeline and codegen, with device compilation and execution disabled.
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

import tilelang
from tilelang import tvm
from tilelang.jit.ascend_runtime_contract import (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    AUTO_GM_FRESH_CONTRACT_VERSION,
    CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION,
    CYTHON_FINAL_SIGNATURE_ATTR,
    CYTHON_INPUT_SIGNATURE_ATTR,
    CYTHON_RESULT_INDICES_ATTR,
    CYTHON_WORKSPACE_INDICES_ATTR,
    prepare_auto_gm_contract,
)
from tilelang.transform.pass_config import process_default_pass_config
from test_ascend_upstream_sync_cross_core import _run
from test_ascend_upstream_sync_runtime_contract import _single_resource

tir = tvm.tir
_REPO = Path(__file__).resolve().parents[3]
_PRIVATE_ATTRS = (
    AUTO_GM_FRESH_CONTRACT_ATTR,
    CYTHON_RESULT_INDICES_ATTR,
    CYTHON_WORKSPACE_INDICES_ATTR,
    CYTHON_INPUT_SIGNATURE_ATTR,
    CYTHON_FINAL_SIGNATURE_ATTR,
    "tl.ascend_sync_proof",
)


@pytest.fixture(autouse=True)
def _requires_proof_service():
    if not tvm.get_global_func("tl.ascend_sync.Z3ProofServiceAvailable")():
        pytest.skip("requires USE_ASCEND_SYNC_Z3=ON")


@pytest.fixture(scope="module")
def rms_factory():
    source = _REPO / "examples/add_rms_norm_dynamic_quant/add_rms_norm_dynamic_quant.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    definitions = []
    for node in tree.body:
        definition = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        constant = isinstance(node, ast.Assign) and isinstance(node.value, (ast.Dict, ast.Constant))
        if definition or constant:
            definitions.append(node)
    # Preserve the original FunctionDef objects and their source locations;
    # T.prim_func still reads the unchanged example body from its real file.
    namespace = {"__name__": "__fresh_scope_rms__", "__file__": str(source)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_kernel_impl"], namespace["_get_tiling"]


def _one_function(module):
    functions = module.functions.values()
    functions = [function for function in functions if isinstance(function, tir.PrimFunc)]
    assert len(functions) == 1
    return functions[0]


def _lower_and_capture(program, implementation, configs):
    captured = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                assert "before" not in captured
                captured["before"] = _one_function(module)

        def run_after_pass(self, module, info):
            if info.name == "tl.AscendInsertSync":
                assert "after" not in captured
                captured["after"] = _one_function(module)

    # Choose the actual Ascend storage-rewrite path without querying a device.
    # This is the same path used by the original example on the NPU runner.
    with (
        tvm.transform.PassContext(config=configs, instruments=[Capture()]),
        patch("tilelang.utils.target.check_npu_availability", return_value=True),
    ):
        artifact = tilelang.lower(
            program,
            target=implementation.target,
            platform="A3",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    assert set(captured) == {"before", "after"}
    assert artifact.kernel_source
    return captured


def _without_private_attrs(function):
    for name in _PRIVATE_ATTRS:
        function = function.without_attr(name)
    return function


def _sync_events(function):
    events = []

    def visit(node):
        if not isinstance(node, tir.Call):
            return
        if node.op.name in ("tl.ascend_auto_set_flag", "tl.ascend_auto_wait_flag"):
            event, event_id = node.args
            assert isinstance(event, tir.StringImm) and isinstance(event_id, tir.IntImm)
            assert 0 <= int(event_id) < 8
            events.append((node.op.name, event.value, int(event_id)))
        elif node.op.name == "tl.ascend_auto_barrier":
            events.append((node.op.name, node.args[0].value))

    tir.stmt_functor.post_order_visit(function.body, visit)
    return events


def _assert_same_local_program(first, second, *, compare_physical_attrs=True):
    # The full function comparison binds external parameter order as well as
    # local storage identities. Only private provenance/proof attrs differ.
    tvm.ir.assert_structural_equal(first.body, second.body, map_free_vars=True)
    if compare_physical_attrs:
        left, right = _without_private_attrs(first), _without_private_attrs(second)
    else:
        # Independent storage-planning runs can reorder disjoint UB regions.
        # Bind the complete ABI/body here; compare physical attributes below
        # only when both proofs consume the exact same pre-sync physical IR.
        left = tir.PrimFunc(first.params, first.body, first.ret_type, first.buffer_map)
        right = tir.PrimFunc(second.params, second.body, second.ret_type, second.buffer_map)
    tvm.ir.assert_structural_equal(left, right, map_free_vars=True)
    assert _sync_events(first) == _sync_events(second)


@pytest.mark.parametrize(
    "rows,columns,dtype",
    [
        (256, 256, "float16"),
        (1024, 2048, "bfloat16"),
        (4096, 4096, "bfloat16"),
        (8192, 2048, "bfloat16"),
    ],
)
def test_original_rms_local_planning_ignores_explicit_fresh_roles(
    rms_factory,
    rows,
    columns,
    dtype,
):
    factory, get_tiling = rms_factory
    implementation = factory.__jit_impl__
    block_m, block_n, padded_m, padded_n = get_tiling(rows, columns, dtype)
    program = factory.__wrapped__(
        rows,
        columns,
        padded_m,
        padded_n,
        block_m,
        block_n,
        1e-6,
        dtype,
    )
    assert "tir.noalias" not in program.attrs
    assert implementation.out_idx == [3, 4, 5]
    assert implementation.workspace_idx is None
    configs = dict(process_default_pass_config(implementation.target, implementation.pass_configs))
    configs.update(
        {
            "tl.ascend_sync_backend": "upstream",
            "tl.ascend_auto_sync": True,
            "tl.ascend_auto_sync_vs": True,
        }
    )
    v1 = prepare_auto_gm_contract(program, implementation.execution_backend, implementation.target)
    v2 = prepare_auto_gm_contract(
        program,
        implementation.execution_backend,
        implementation.target,
        out_idx=implementation.out_idx,
        workspace_idx=implementation.workspace_idx,
    )
    assert v1.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == AUTO_GM_FRESH_CONTRACT_VERSION
    assert v2.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value == CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION
    captures = [_lower_and_capture(function, implementation, configs) for function in (v1, v2)]
    for stage in ("before", "after"):
        first, second = [capture[stage] for capture in captures]
        _assert_same_local_program(first, second, compare_physical_attrs=False)
        contract = second.attrs[AUTO_GM_FRESH_CONTRACT_ATTR].value
        assert contract == CYTHON_ALLOCATOR_FRESH_CONTRACT_VERSION
        assert [int(index) for index in second.attrs[CYTHON_RESULT_INDICES_ATTR]] == [3, 4, 5]
        assert len(second.attrs[CYTHON_WORKSPACE_INDICES_ATTR]) == 0
        signature = second.attrs[CYTHON_FINAL_SIGNATURE_ATTR]
        assert len(signature) == len(second.params)
        assert all(parameter.same_as(signature[i]) for i, parameter in enumerate(second.params))
    assert _sync_events(captures[0]["after"])

    # Isolate the new fact from all earlier storage-planning choices. Both
    # native runs must use identical physical maps, bytes, controls and ABI;
    # only their allocator provenance differs. No kernel operation is edited.
    shared = captures[0]["before"]
    scoped_inputs = [
        prepare_auto_gm_contract(shared, implementation.execution_backend, implementation.target),
        prepare_auto_gm_contract(
            shared,
            implementation.execution_backend,
            implementation.target,
            out_idx=implementation.out_idx,
            workspace_idx=implementation.workspace_idx,
        ),
    ]
    synchronized = []
    for function in scoped_inputs:
        with tvm.transform.PassContext(config=configs):
            module = tilelang.transform.AscendInsertSync(shared.attrs["target"], "A3")(
                tvm.IRModule({"same_physical_input": function}),
            )
        synchronized.append(module["same_physical_input"])
    _assert_same_local_program(*scoped_inputs)
    _assert_same_local_program(*synchronized)


@pytest.mark.parametrize("compiler_workspaces", [[], [1]])
def test_local_vector_scope_preserves_old_auto_gm_facts(compiler_workspaces):
    original = _single_resource().with_attr("auto_gm_indices", compiler_workspaces)
    v1 = prepare_auto_gm_contract(original, "cython", "ascendc")
    v2 = prepare_auto_gm_contract(original, "cython", "ascendc", out_idx=[1])
    first, second = _run(v1), _run(v2)
    _assert_same_local_program(first, second)
    events = _sync_events(second)
    needs_conservative_gm_edge = not compiler_workspaces
    assert any(event[1] == "MTE3_MTE2" for event in events) == needs_conservative_gm_edge
    assert "tir.noalias" not in second.attrs


def test_local_scope_still_validates_the_complete_explicit_recipe():
    original = _single_resource().with_attr("auto_gm_indices", [])
    v2 = prepare_auto_gm_contract(original, "cython", "ascendc", out_idx=[1])
    malformed = v2.with_attr(CYTHON_RESULT_INDICES_ATTR, [tir.IntImm("int64", 2)])
    with pytest.raises(tvm.error.TVMError, match="fresh allocation index|final signature"):
        _run(malformed)
