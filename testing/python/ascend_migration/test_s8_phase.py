"""Exercise real phase routing without importing TVM or an NPU runtime."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
AUTO = {
    "tl.ascend_pipeline_planner": "auto",
    "tl.ascend_sync_strategy": "a3",
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_sync_vs": False,
}
OLD = {
    "CrossCorePipeline",
    "CombineCV",
    "PipelinePlanning",
    "InjectSoftwarePipeline",
    "AscendSyncInsert",
    "AscendSyncInsertVS",
    "AscendPrepareSyncHandoff",
    "AscendInsertSyncHandoff",
}
NEW = (
    "AscendAutoSchedule",
    "AscendMaterializeSchedule",
    "AscendInsertScheduledSync",
    "AscendVerifyAutoSchedule",
)


class PassFactory:
    def __init__(self, calls, context):
        self.calls = calls
        self.context = context

    def get_pass_context(self):
        return self.context

    def __getattr__(self, name):
        def construct(*args, **kwargs):
            def apply(module):
                self.calls.append(name)
                return module

            return apply

        return construct


def load_phase(monkeypatch, config):
    calls = []
    context = SimpleNamespace(config=config)
    tilelang = ModuleType("tilelang")
    tilelang.transform = PassFactory(calls, context)
    target_module = ModuleType("tilelang.utils.target")
    target_module.check_npu_availability = lambda: True
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(sys.modules, "tilelang.utils", ModuleType("tilelang.utils"))
    monkeypatch.setitem(sys.modules, "tilelang.utils.target", target_module)
    names = {
        "allow_vectorize",
        "_ascend_sync_strategy",
        "_ascend_pipeline_planner",
        "LowerAndLegalize",
        "OptimizeForTarget",
    }
    path = ROOT / "tilelang/engine/phase.py"
    source = ast.parse(path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    namespace = {
        "tilelang": tilelang,
        "tir": SimpleNamespace(transform=PassFactory(calls, context)),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace), calls


def test_auto_owns_the_complete_route(monkeypatch):
    phase, calls = load_phase(monkeypatch, dict(AUTO))
    original = object()
    assert phase.OptimizeForTarget(original, SimpleNamespace(model="ascendc"), "A3") is original
    assert not OLD.intersection(calls)
    assert [name for name in calls if name in NEW] == list(NEW)
    assert all(calls.count(name) == 1 for name in NEW)
    assert "AscendLowerScheduleHints" not in calls
    assert calls.index("AscendMaterializeSchedule") < calls.index("AscendMemoryPlanning")
    assert calls.index("AscendMemoryPlanning") < calls.index("AscendInsertScheduledSync")
    assert calls.index("AscendInsertScheduledSync") < calls.index("AscendVerifyAutoSchedule")
    assert calls[-3:] == [
        "AscendResourceScopeVerify",
        "AscendVectorInstructionSelection",
        "AscendVectorMaskLegalize",
    ]


def test_auto_preserves_control_before_lowering(monkeypatch):
    phase, calls = load_phase(monkeypatch, dict(AUTO))
    original = object()
    assert phase.LowerAndLegalize(original, SimpleNamespace(model="ascendc")) is original
    assert calls[0] == "AscendPrepareAutoSchedule"
    assert calls.index("AscendPrepareAutoSchedule") < calls.index("AscendValidateScheduleHints")
    assert calls.index("AscendPrepareAutoSchedule") < calls.index("Simplify")
    assert "AscendPrepareConstraintPipeline" not in calls


@pytest.mark.parametrize("planner", ["legacy", "constraint"])
def test_existing_routes_keep_their_backend(monkeypatch, planner):
    config = {**AUTO, "tl.ascend_pipeline_planner": planner}
    if planner == "legacy":
        config["tl.ascend_sync_strategy"] = "legacy"
    phase, calls = load_phase(monkeypatch, config)
    phase.OptimizeForTarget(object(), SimpleNamespace(model="ascendc"), "A3")
    assert not set(NEW).intersection(calls)
    for name in ("CrossCorePipeline", "CombineCV", "PipelinePlanning", "InjectSoftwarePipeline"):
        assert calls.count(name) == 1
    assert calls.count("AscendSyncInsertVS") == 1
    assert calls.count("AscendVerifyConstraintSync") == (planner == "constraint")


@pytest.mark.parametrize(
    "key,value",
    [
        ("tl.ascend_auto_sync", False),
        ("tl.ascend_memory_planning", False),
        ("tl.ascend_auto_cv_combine", False),
        ("tl.ascend_auto_cross_core_sync", True),
        ("tl.ascend_auto_sync_vs", True),
        ("tl.ascend_sync_strategy", "legacy"),
        ("tl.ascend_pipeline_cost_model", {}),
    ],
)
def test_conflicting_ownership_fails_before_any_pass(monkeypatch, key, value):
    phase, calls = load_phase(monkeypatch, {**AUTO, key: value})
    with pytest.raises(ValueError):
        phase.OptimizeForTarget(object(), SimpleNamespace(model="ascendc"), "A3")
    assert calls == []


@pytest.mark.parametrize("model,platform", [("pto", "A3"), ("ascendc", "A5")])
def test_unsupported_backend_fails_before_any_pass(monkeypatch, model, platform):
    phase, calls = load_phase(monkeypatch, dict(AUTO))
    with pytest.raises(ValueError, match="A3 AscendC"):
        phase.OptimizeForTarget(object(), SimpleNamespace(model=model), platform)
    assert calls == []


def test_auto_uses_default_cv_ownership(monkeypatch):
    config = dict(AUTO)
    del config["tl.ascend_auto_cv_combine"]
    phase, calls = load_phase(monkeypatch, config)
    phase.OptimizeForTarget(object(), SimpleNamespace(model="auto"), "A3")
    assert [name for name in calls if name in NEW] == list(NEW)
