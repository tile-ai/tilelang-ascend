#!/usr/bin/env python3
"""Run the S1 legacy IR/codegen contracts without compiling or launching NPU code."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from _common import ROOT, base_report, finish, run_pytest
from _native_build import verify_manifest


LANGUAGE = "testing/python/language/"
CONTRACT_TESTS = {
    "resource_ownership": (
        "test_tilelang_ascend_resource_scope.py",
        (
            "test_combine_default_and_explicit_opt_out",
            "test_default_combine_lowers_vector_kernel_with_optional_scope",
            "test_default_combine_preserves_explicit_owners_and_rejects_conflicts",
            "test_unscoped_gm_scalar_store_is_not_inferred_from_context",
            "test_default_combine_keeps_dependencies_across_explicit_vector_blocks",
            "test_default_combine_does_not_use_local_events_for_cross_core_dependencies",
            "test_pipeline_with_nested_serial_loop_lowers_after_cv_split",
        ),
    ),
    "final_backend_boundary": (
        "test_tilelang_ascend_vector_mask.py",
        (
            "test_selection_runs_after_phase_two_and_legalizer_runs_last",
            "test_resource_scope_is_explicit_nested_and_fail_closed",
            "test_resource_scope_normalizes_pipe_case_but_rejects_ambiguous_sync",
            "test_resource_scope_checks_automatic_barriers",
            "test_opaque_symbol_changes_invalidate_mask_facts",
            "test_target_scope_keeps_a5_and_pto_unchanged",
        ),
    ),
    "optional_workspace": (
        "test_tilelang_ascend_language_explicit_tmp.py",
        (
            "test_every_workspace_api_has_the_target_specific_lowered_layout",
            "test_optional_workspace_metadata_uses_the_lowered_call_layout",
            "test_pto_select_metadata_tracks_shifted_src_and_destination",
        ),
    ),
    "persistent_domain": (
        "test_tilelang_ascend_language_persistent.py",
        (
            "test_persistent_guard_for_static_domains",
            "test_persistent_guard_for_dynamic_domain",
            "test_persistent_rejects_invalid_static_arguments",
            "test_partial_single_wave_guard_reaches_codegen",
            "test_multi_wave_break_guard_reaches_codegen",
        ),
    ),
    "physical_alias_and_optional_loop": (
        "test_ascend_sync_insert.py",
        (
            "test_outstanding_reader_survives_another_pipe_read",
            "test_new_reader_is_not_protected_by_old_alias_handoff",
            "test_loop_exit_preserves_the_zero_trip_reader",
            "test_optional_loops_compose_without_losing_the_entry_reader",
        ),
    ),
    "one_trip": (
        "test_ascend_sync_insert_one_trip.py",
        (
            "test_loop_exit_does_not_borrow_a_second_iteration_handoff",
            "test_nested_loop_exit_keeps_a_writer_created_in_the_outer_iteration",
        ),
    ),
    "s1_offsets_memory_and_l1": (
        "test_ascend_migration_baseline.py",
        (
            "test_copy_elem_offset",
            "test_atomic_elem_offset",
            "test_contiguous_view_offsets",
            "test_tile_region_elem_offset",
            "test_tile_fill_view_elem_offset",
            "test_l1_pipeline_physical_versions",
            "test_legacy_memory_maps",
        ),
    ),
}

_PREFLIGHT = """
import json
from pathlib import Path
import tilelang
import tvm
from tvm._ffi import base
result = {
    'tilelang_version': tilelang.__version__,
    'tilelang_python': tilelang.__file__,
    'tilelang_library': str(Path(tilelang._LIB_PATH).resolve()),
    'tvm_version': tvm.__version__,
    'tvm_python': tvm.__file__,
    'tvm_library': str(Path(base._LIB._name).resolve()),
}
print('S1_NATIVE_ENV=' + json.dumps(result))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2", "a3"), required=True)
    parser.add_argument("--timeout", type=float, default=120, help="Pytest process timeout in seconds")
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON result")
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                str(ROOT / "build/ascend_migration_native_build.json"),
            )
        ),
        help="Source/library identity recorded after the explicit native rebuild",
    )
    args = parser.parse_args()
    report = base_report("s1_check_legacy_lowering", args)
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Run native assertions without -O/PYTHONOPTIMIZE")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    report["verification_scope"] = (
        "Host-only legacy TIR and generated-source contracts; existing fixtures keep their "
        "own A2/A3/PTO target parameters. This is not device acceptance for --platform."
    )
    report["fixture_configuration"] = {
        "resource_ownership": {
            "targets": ["ascendc", "pto"],
            "platform": "A3 for existing full-lowering fixtures; direct IR otherwise",
            "pass_configs": {
                "tl.ascend_auto_cv_combine": ["default true", True, False],
                "tl.ascend_auto_sync": ["default false", True],
                "tl.ascend_auto_sync_vs": ["C++ default false", True],
            },
        },
        "final_backend_boundary": {
            "target": "ascendc/A2; explicit PTO/A2 and AscendC/A5 exclusion checks",
            "pass_configs": "Fixture-owned mask and resource contracts; no new planner config",
        },
        "optional_workspace": {
            "targets": ["ascendc", "pto"],
            "platform": "A2 for sync metadata; target-specific standalone IR otherwise",
            "pass_configs": {"tl.ascend_auto_sync": True},
        },
        "persistent_domain": {
            "targets": ["ascendc", "pto"],
            "platform": "Pure frontend IR or existing lower() default; no device launch",
            "pass_configs": "Existing fixture defaults",
        },
        "physical_alias_and_optional_loop": {
            "target": "ascendc/A3",
            "pass_configs": {"tl.ascend_auto_sync": True},
            "memory": "Explicit fixture address_map/size_map; no NPU allocation",
        },
        "one_trip": {
            "target": "ascendc/A3",
            "pass_configs": {"tl.ascend_auto_sync": True},
        },
        "s1_offsets_memory_and_l1": {
            "platform": args.platform.upper(),
            "selection": "TL_ASCEND_MIGRATION_PLATFORM for the new S1 native fixtures",
        },
    }
    report["contracts"] = {name: [f"{LANGUAGE}{file}::{node}" for node in nodes] for name, (file, nodes) in CONTRACT_TESTS.items()}
    nodes = [node for group in report["contracts"].values() for node in group]
    artifacts = args.output.resolve().parent / f"{args.output.stem}-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        report["native_build"] = verify_manifest(args.build_manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(args, report, "BLOCKED", f"Native build identity: {error}")

    env = os.environ.copy()
    tvm_python = Path(report["native_build"]["tvm_source_root"]) / "python"
    python_paths = [str(ROOT), str(tvm_python)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["TVM_IMPORT_PYTHON_PATH"] = str(tvm_python)
    env["TVM_LIBRARY_PATH"] = str(ROOT / "build/tvm")
    env["TILELANG_LIBRARY_PATH"] = str(ROOT / "build")
    env["TL_ASCEND_MIGRATION_PLATFORM"] = args.platform.upper()
    command = [sys.executable, "-c", _PREFLIGHT]
    log = artifacts / "native-environment.log"
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=min(args.timeout, 60),
            check=False,
        )
        log.write_text(result.stdout + result.stderr)
        report["native_preflight"] = {"command": command, "log": str(log)}
        if result.returncode:
            return finish(args, report, "BLOCKED", "Native imports failed; see environment log")
        payload = next(line.removeprefix("S1_NATIVE_ENV=") for line in result.stdout.splitlines() if line.startswith("S1_NATIVE_ENV="))
        loaded = json.loads(payload)
        expected = {item["path"] for item in report["native_build"]["libraries"]}
        if loaded["tilelang_library"] not in expected or loaded["tvm_library"] not in expected:
            return finish(args, report, "BLOCKED", "Native imports loaded libraries outside manifest")
        if not Path(loaded["tilelang_python"]).resolve().is_relative_to(ROOT / "tilelang"):
            return finish(args, report, "BLOCKED", "Native imports loaded a different TileLang package")
        if not Path(loaded["tvm_python"]).resolve().is_relative_to(tvm_python):
            return finish(args, report, "BLOCKED", "Native imports loaded unverified TVM Python")
        report["native_environment"] = loaded
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error))
        return finish(args, report, "TIMEOUT", "Native import preflight timed out")
    except (OSError, ValueError, KeyError, StopIteration) as error:
        return finish(args, report, "BLOCKED", f"Native import preflight: {error}")

    try:
        result = run_pytest(nodes, artifacts, env=env, timeout=args.timeout)
    except OSError as error:
        return finish(args, report, "BLOCKED", f"Cannot launch native pytest process: {error}")
    report["pytest"] = result
    report["counts"] = result["counts"]
    return finish(args, report, result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
