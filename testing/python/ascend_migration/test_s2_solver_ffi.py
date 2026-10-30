"""Old TVM FFI boundary for the explicitly registered optional solver."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _module():
    from tilelang.transform import ascend_schedule_solver_ffi

    return ascend_schedule_solver_ffi


def test_cold_legacy_import_and_vector_lowering_never_request_z3():
    language = Path(__file__).resolve().parents[1] / "language"
    code = """
import sys
requests = []
class RejectSolverImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'z3':
            requests.append(fullname)
            raise AssertionError('cold legacy requested optional solver: ' + fullname)
sys.meta_path.insert(0, RejectSolverImport())
import tilelang
from tilelang import tvm
assert tvm.get_global_func('tl.ascend.solve_schedule', allow_missing=True) is None
sys.path.insert(0, sys.argv[1])
from test_tilelang_ascend_resource_scope import (
    test_default_combine_lowers_vector_kernel_with_optional_scope,
)
test_default_combine_lowers_vector_kernel_with_optional_scope(True, 'ascendc')
assert requests == [], requests
assert 'z3' not in sys.modules
assert tvm.get_global_func('tl.ascend.solve_schedule', allow_missing=True) is None
print('COLD_LEGACY_WITHOUT_Z3=PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(language)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COLD_LEGACY_WITHOUT_Z3=PASS" in result.stdout


def test_registration_does_not_import_solver_dependency(monkeypatch):
    from tilelang import tvm

    called = []
    original = importlib.import_module

    def guarded(name, *args, **kwargs):
        if name == "z3":
            called.append(name)
            raise ModuleNotFoundError("controlled absent optional dependency", name="z3")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded)
    function = _module().register_solver()
    assert not called
    result = function(
        tvm.runtime.convert(
            {
                "schema_version": 1,
                "mode": "straight",
                "latencies": [1],
                "iis": [1],
                "resource_flags": [1],
            }
        )
    )
    assert str(result["status"]) == "MISSING_DEPENDENCY"
    assert called == ["z3"]


def test_numeric_old_ffi_round_trip():
    from tilelang import tvm

    function = _module().register_solver()
    result = function(
        tvm.runtime.convert(
            {
                "schema_version": 1,
                "mode": "straight",
                "latencies": [3, 4],
                "iis": [1, 1],
                "resource_flags": [1, 2],
                "data_deps": [[0, 1, 0, 3]],
            }
        )
    )
    assert str(result["status"]) == "SAT"
    assert int(result["start_times"][1]) >= int(result["start_times"][0]) + 3
    assert [int(item) for item in result["order"]] == [0, 1]


def test_old_ffi_preserves_int64_gaps():
    from tilelang import tvm

    function = _module().register_solver()
    gap = (1 << 40) + 17
    result = function(
        tvm.runtime.convert(
            {
                "mode": "straight",
                "latencies": [1, 1],
                "iis": [1, 1],
                "resource_flags": [1, 2],
                "data_deps": [[0, 1, 0, gap]],
            }
        )
    )
    assert str(result["status"]) == "SAT"
    assert int(result["start_times"][1]) - int(result["start_times"][0]) >= gap


def test_old_ffi_preserves_boolean_policy_and_raw_manual_stages():
    from tilelang import tvm

    function = _module().register_solver()
    result = function(
        tvm.runtime.convert(
            {
                "mode": "modulo",
                "latencies": [1, 1],
                "iis": [1, 1],
                "resource_flags": [1, 2],
                "ii_bounds": [1, 8],
                "enable_offset": True,
                "manual_stages": [4, 5],
                "register_phase_deps": [[0, 1, 0, 0]],
            }
        )
    )
    assert str(result["status"]) == "SAT"
    ii = int(result["ii"])
    assert [int(start) // ii for start in result["start_times"]] == [4, 5]


def test_ffi_does_not_evaluate_symbolic_tir():
    from tilelang import tvm

    function = _module().register_solver()
    request = tvm.runtime.convert(
        {
            "schema_version": 1,
            "mode": "straight",
            "latencies": [tvm.tir.Var("n", "int32")],
            "iis": [1],
            "resource_flags": [1],
        }
    )
    with pytest.raises((ValueError, tvm.error.TVMError), match="numeric IntImm"):
        function(request)


def test_symbolic_mapping_keys_cannot_impersonate_protocol_fields():
    from tilelang import tvm

    function = _module().register_solver()
    request = tvm.runtime.convert({tvm.tir.Var("mode", "int32"): "straight"})
    with pytest.raises((ValueError, tvm.error.TVMError), match="field names must be strings"):
        function(request)


def test_registration_preserves_existing_user_callback():
    from tilelang import tvm

    name = "tl.ascend.solve_schedule"
    previous = tvm.get_global_func(name, allow_missing=True)
    tvm.register_func(name, lambda *_: "user-callback", override=True)
    try:
        function = _module().register_solver()
        assert function() == "user-callback"
        assert tvm.get_global_func(name)() == "user-callback"
    finally:
        if previous is None:
            tvm._ffi.registry.remove_global_func(name)
        else:
            tvm.register_func(name, previous, override=True)


def test_glue_import_does_not_register_or_analyze(monkeypatch):
    from tilelang import tvm

    calls = []
    monkeypatch.setattr(tvm, "register_func", lambda *args, **kwargs: calls.append(args))
    importlib.reload(_module())
    assert calls == []
