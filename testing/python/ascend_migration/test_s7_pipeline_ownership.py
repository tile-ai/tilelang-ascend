"""Real phase invocation counts establish a single owner for each synchronization domain."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import tilelang
from tilelang import tvm
from test_s5_versions_ir import fixture_profile
from test_s7_kernels import make_cv_handoff_kernel, make_vector_sync_kernel


def configs(planner, sync):
    return {
        "tl.ascend_pipeline_planner": planner,
        "tl.ascend_sync_strategy": sync,
        "tl.ascend_pipeline_cost_model": fixture_profile(),
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_sync_vs": planner == "legacy",
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": False,
    }


@pytest.mark.parametrize(
    "planner,sync,old_cross",
    [
        ("constraint", "legacy", False),
        ("constraint", "a3", False),
        ("legacy", "a3", False),
        ("legacy", "a3", True),
    ],
)
def test_actual_phase_calls_assign_each_domain_once(monkeypatch, planner, sync, old_cross):
    names = (
        "AscendSyncInsert",
        "AscendSyncInsertVS",
        "AscendInsertSyncA3",
        "AscendPrepareSyncHandoff",
        "AscendInsertSyncHandoff",
        "AscendVerifySyncHandoff",
        "AscendVerifyConstraintSync",
    )
    calls = []
    for name in names:
        original = getattr(tilelang.transform, name)

        def wrap(*args, name=name, original=original, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(tilelang.transform, name, wrap)
    function = (
        make_vector_sync_kernel(trips=3, versions=1, cores=1)
        if planner == "constraint"
        else make_cv_handoff_kernel(trips=3, cores=1, lowercase_workspace=old_cross)
    )
    configuration = configs(planner, sync)
    configuration["tl.ascend_auto_cross_core_sync"] = old_cross
    with tvm.transform.PassContext(config=configuration):
        artifact = tilelang.lower(function, target="ascendc", platform="A3")
    assert calls.count("AscendSyncInsertVS") == 1
    assert calls.count("AscendInsertSyncA3") == int(planner == "constraint" and sync == "a3")
    assert calls.count("AscendSyncInsert") == int(not (planner == "constraint" and sync == "a3"))
    for name in ("AscendPrepareSyncHandoff", "AscendInsertSyncHandoff", "AscendVerifySyncHandoff"):
        assert calls.count(name) == int(planner == "legacy" and sync == "a3")
    assert calls.count("AscendVerifyConstraintSync") == int(planner == "constraint")
    if planner == "legacy":
        assert calls.index("AscendPrepareSyncHandoff") < calls.index("AscendSyncInsert")
        assert calls.index("AscendSyncInsertVS") < calls.index("AscendInsertSyncHandoff")
        assert calls.index("AscendInsertSyncHandoff") < calls.index("AscendVerifySyncHandoff")
        sets, waits = [], []

        def visit(node):
            if isinstance(node, tvm.tir.Call):
                name = str(getattr(node.op, "name", ""))
                if name == "tl.ascend_auto_set_cross_flag":
                    sets.append(node)
                if name == "tl.ascend_auto_wait_cross_flag":
                    waits.append(node)

        for device_function in artifact.device_mod.functions.values():
            assert int(device_function.attrs["tl.ascend_handoff_plan"]["channel_count"]) == 1
            tvm.tir.stmt_functor.post_order_visit(device_function.body, visit)
        # One READY and one RELEASE in the static round body. Enabling the old
        # cross flag cannot add a second owner for either notification.
        assert len(sets) == len(waits) == 2


def test_prepared_handoff_replaces_an_actually_active_old_cross_generator():
    function = make_cv_handoff_kernel(trips=3, cores=1, lowercase_workspace=True)

    def cross_calls(artifact):
        found = {"tl.ascend_auto_set_cross_flag": [], "tl.ascend_auto_wait_cross_flag": []}

        def visit(node):
            if isinstance(node, tvm.tir.Call) and str(getattr(node.op, "name", "")) in found:
                found[node.op.name].append(node)

        for device_function in artifact.device_mod.functions.values():
            tvm.tir.stmt_functor.post_order_visit(device_function.body, visit)
        return found

    legacy = configs("legacy", "legacy")
    legacy["tl.ascend_auto_cross_core_sync"] = True
    with tvm.transform.PassContext(config=legacy):
        previous = tilelang.lower(function, target="ascendc", platform="A3")
    old = cross_calls(previous)
    assert all(old.values()), "The lower-case workspace must actually activate the old generator"
    current = dict(legacy, **{"tl.ascend_sync_strategy": "a3"})
    with tvm.transform.PassContext(config=current):
        artifact = tilelang.lower(function, target="ascendc", platform="A3")
    generated = cross_calls(artifact)
    assert all(len(calls) == 2 for calls in generated.values())
    assert {int(call.args[2]) for call in generated["tl.ascend_auto_set_cross_flag"]} == {0, 1}


def test_cold_default_legacy_lowering_does_not_request_z3_or_new_sync():
    source = """import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0]=='z3' or 'ascend_z3_scheduler' in fullname:
            raise AssertionError('legacy requested solver: '+fullname)
sys.meta_path.insert(0,Guard())
import tilelang
from tilelang import tvm
from test_s5_kernels import make_pipeline_kernel
with tvm.transform.PassContext(config={'tl.ascend_auto_sync':True,'tl.ascend_memory_planning':True}):
    artifact=tilelang.lower(make_pipeline_kernel(trips=3,num_stages=0),target='ascendc',platform='A3')
for function in artifact.device_mod.functions.values():
    assert not any(str(key).startswith(('tl.ascend_handoff_', 'tl.ascend_local_sync_')) for key in function.attrs.keys())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", source], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("strategy", ["unknown", "A3"])
def test_unknown_sync_strategy_is_not_treated_as_legacy(strategy):
    with (
        tvm.transform.PassContext(config=configs("constraint", strategy)),
        pytest.raises(ValueError),
    ):
        tilelang.lower(make_vector_sync_kernel(trips=3, versions=1, cores=1), target="ascendc", platform="A3")


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_legacy_manual_reference_keeps_balanced_mode2_protocol(direction):
    with tvm.transform.PassContext(config=configs("legacy", "legacy")):
        artifact = tilelang.lower(
            make_cv_handoff_kernel(direction, trips=3, cores=1, manual_legacy=True),
            target="ascendc",
            platform="A3",
        )
    assert "KERNEL_TYPE_MIX_AIC_1_2" in artifact.kernel_source
    assert "CrossCoreSetFlag" in artifact.kernel_source and "CrossCoreWaitFlag" in artifact.kernel_source
    sets, waits = [], []

    def visit(node):
        if isinstance(node, tvm.tir.Call):
            name = str(getattr(node.op, "name", ""))
            if name == "tl.ascend_set_cross_flag":
                sets.append(node)
            if name == "tl.ascend_wait_cross_flag":
                waits.append(node)

    for function in artifact.device_mod.functions.values():
        assert not any(str(key).startswith("tl.ascend_handoff_") for key in function.attrs.keys())
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    assert len(sets) == len(waits) == 2
    assert {int(call.args[1]) for call in sets} == {0, 1}
    assert {int(call.args[0]) for call in waits} == {0, 1}
    assert all(int(call.args[2]) == 2 for call in sets)
