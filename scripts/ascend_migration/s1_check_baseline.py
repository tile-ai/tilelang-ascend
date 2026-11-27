"""Audit the fixed S1 source baseline without importing TileLang, TVM or NPU modules."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess

from _common import BASELINE, ROOT, base_report, finish, git

DEFAULTS = (
    ("src/transform/ascend_combinecv.cc", "ascendAutoCombine", True),
    ("src/transform/ascend_combinecv.cc", "ascendAutoCrossCoreSync", False),
    ("src/transform/ascend_sync_insert.cc", "kAscendAutoSync", False),
    ("src/transform/ascend_sync_insert_vs.cc", "kAscendAutoSyncVs", False),
    ("src/transform/ascend_memory_planning.cc", "kAscendMemoryPlanning", False),
)

# This is an audit index, not a substitute for executing the native/device suites.
TEST_MAP = {
    "resource ownership and default CV": (
        "test_tilelang_ascend_resource_scope.py",
        [
            "test_combine_default_and_explicit_opt_out",
            "test_pipeline_with_nested_serial_loop_lowers_after_cv_split",
        ],
    ),
    "backend terminal boundaries and fail-closed resource validation": (
        "test_tilelang_ascend_vector_mask.py",
        [
            "test_selection_runs_after_phase_two_and_legalizer_runs_last",
            "test_resource_scope_is_explicit_nested_and_fail_closed",
        ],
    ),
    "optional workspace ABI": (
        "test_tilelang_ascend_language_explicit_tmp.py",
        ["test_optional_workspace_metadata_uses_the_lowered_call_layout"],
    ),
    "Persistent guards": (
        "test_tilelang_ascend_language_persistent.py",
        ["test_persistent_guard_for_static_domains", "test_persistent_guard_for_dynamic_domain"],
    ),
    "physical alias and loop lifetime": (
        "test_ascend_sync_insert.py",
        [
            "test_outstanding_reader_survives_another_pipe_read",
            "test_optional_loops_compose_without_losing_the_entry_reader",
        ],
    ),
    "one-trip lifetime": (
        "test_ascend_sync_insert_one_trip.py",
        ["test_loop_exit_does_not_borrow_a_second_iteration_handoff"],
    ),
    "L1 physical multiversion tails (device tests)": (
        "test_tilelang_ascend_language_gemm_v0_k_tail.py",
        ["test_gemm_v0_ktail_pipeline", "test_gemm_v0_ktail_pipeline_parallel"],
    ),
    "compile flags and cache recovery distinction": (
        "test_ascend_compile_flags.py",
        [],
    ),
}


def read_source(path: str, baseline: bool) -> str:
    return git("show", f"{BASELINE}:{path}") if baseline else (ROOT / path).read_text()


def default_value(source: str, key: str) -> bool:
    pattern = rf"GetConfig\s*<\s*Bool\s*>\s*\(\s*{key}\s*,\s*Bool\((true|false)\)"
    match = re.search(pattern, source)
    assert match, f"Cannot locate explicit C++ default for {key}"
    return match[1] == "true"


def phase_calls(source: str, function: str) -> list[str]:
    module = ast.parse(source)
    definition = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function)
    calls = []
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Call)
            and isinstance(node.value.func.func, ast.Attribute)
        ):
            calls.append((node.lineno, node.value.func.func.attr))
    return [name for _, name in sorted(calls)]


def audit_phase(source: str) -> dict:
    first = phase_calls(source, "LowerAndLegalize")
    second = phase_calls(source, "OptimizeForTarget")
    assert first.index("AscendInferBufferScope") < first.index("InjectTmpBuffer")
    assert first.index("InjectTmpBuffer") < first.index("AscendVidReduction")
    bridge = [
        "CrossCorePipeline",
        "CombineCV",
        "PlanAndUpdateBufferAllocationLocation",
        "PipelinePlanning",
        "InjectSoftwarePipeline",
    ]
    start = second.index("CrossCorePipeline")
    assert second[start : start + len(bridge)] == bridge, "Legacy bridge order changed"
    assert second.count("PipelinePlanning") == second.count("InjectSoftwarePipeline") == 1
    terminal = [
        "AscendMemoryPlanning",
        "AscendSyncInsert",
        "AscendSyncInsertVS",
        "AscendResourceScopeVerify",
        "AscendVectorInstructionSelection",
        "AscendVectorMaskLegalize",
    ]
    positions = [second.index(name) for name in terminal]
    assert positions == sorted(positions), "Backend terminal boundary changed"
    assert second[-1] == "AscendVectorMaskLegalize", "Transform follows mask legalization"
    return {"phase_one": first, "phase_two": second}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON report")
    args = parser.parse_args()
    report = base_report("s1_check_baseline", args)
    if not __debug__:
        return finish(args, report, "BLOCKED", "Run source assertions without Python -O/PYTHONOPTIMIZE")
    report["validation_layer"] = "source audit only; no native or device execution"
    report["checks"] = []

    def check(name, action):
        try:
            detail = action()
            report["checks"].append({"name": name, "status": "PASS", "detail": detail})
        except (AssertionError, ValueError, StopIteration) as exc:
            report["checks"].append({"name": name, "status": "FAIL", "error": str(exc)})

    def identity():
        assert git("rev-parse", f"{BASELINE}^{{commit}}") == BASELINE
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASELINE, "HEAD"],
            check=True,
            capture_output=True,
        )
        return {
            "source_delta": git("diff", "--name-only", BASELINE),
            "submodules": git("submodule", "status"),
        }

    try:
        check("fixed baseline identity and ancestry", identity)
        for path, key, expected in DEFAULTS:

            def verify_default(path=path, key=key, expected=expected):
                values = {
                    "baseline": default_value(read_source(path, True), key),
                    "current": default_value(read_source(path, False), key),
                }
                assert values["baseline"] == values["current"] == expected, values
                return {"path": path, "key": key, **values}

            check(f"default {key}", verify_default)
        for baseline in (True, False):
            check(
                f"Pass order ({'baseline' if baseline else 'current'})",
                lambda baseline=baseline: audit_phase(read_source("tilelang/engine/phase.py", baseline)),
            )

        def ffi():
            source = read_source("tilelang/transform/_ffi_api.py", False)
            assert '_init_api("tl.transform"' in source or "_init_api('tl.transform'" in source
            cmake = read_source("CMakeLists.txt", False)
            assert re.search(r"CMAKE_CXX_STANDARD\s+17", cmake)
            for path, global_name in (
                ("src/transform/pipeline_planning.cc", "PipelinePlanning"),
                ("src/transform/inject_pipeline.cc", "InjectSoftwarePipeline"),
            ):
                assert f'"tl.transform.{global_name}"' in read_source(path, False)
            return {"ffi": "tvm._ffi / tvm.tir", "cxx_standard": 17}

        check("legacy FFI and C++ ABI", ffi)

        def defaults():
            source = read_source("tilelang/transform/pass_config.py", False)
            tree = ast.parse(source)
            target_defaults = next(
                n.value
                for n in tree.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "_TARGET_PASS_DEFAULTS"
            )
            pto = next(
                value
                for key, value in zip(target_defaults.keys, target_defaults.values)
                if isinstance(key, ast.Constant) and key.value == "pto"
            )
            assert any(
                isinstance(key, ast.Attribute)
                and key.attr == "TL_ASCEND_AUTO_SYNC_VS"
                and isinstance(value, ast.Constant)
                and value.value is True
                for key, value in zip(pto.keys, pto.values)
            )
            for target, values in zip(target_defaults.keys, target_defaults.values):
                if isinstance(target, ast.Constant) and target.value in {"ascendc", "auto"}:
                    assert not any(isinstance(key, ast.Attribute) and key.attr == "TL_ASCEND_AUTO_SYNC_VS" for key in values.keys), (
                        "AscendC/auto must retain the existing C++ VS default"
                    )
            return {
                "pto_vs_default": True,
                "ascendc_vs_cpp_default": False,
                "ascendc_auto_python_vs_override": False,
            }

        check("PTO-specific VS default is separate from AscendC", defaults)

        for contract, (filename, names) in TEST_MAP.items():

            def test_mapping(filename=filename, names=names):
                path = f"testing/python/language/{filename}"
                tree = ast.parse((ROOT / path).read_text())
                functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
                assert set(names) <= functions, f"Missing test nodes: {set(names) - functions}"
                return {"path": path, "nodes": names, "executed": False}

            check(f"test mapping: {contract}", test_mapping)
    except (OSError, subprocess.CalledProcessError, SyntaxError) as exc:
        return finish(args, report, "BLOCKED", str(exc))
    report["counts"] = {
        "passed": sum(c["status"] == "PASS" for c in report["checks"]),
        "failed": sum(c["status"] == "FAIL" for c in report["checks"]),
        "skipped": 0,
        "total": len(report["checks"]),
    }
    return finish(args, report, "FAIL" if report["counts"]["failed"] else "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
