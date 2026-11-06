#!/usr/bin/env python3
"""Verify S2 native task analysis, old FFI and the S1 legacy IR regressions."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

from _common import ROOT, base_report, finish, run_pytest
from _native_build import verify_manifest
from s1_check_legacy_lowering import CONTRACT_TESTS, LANGUAGE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2", "a3"), required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                str(ROOT / "build/ascend_migration_s2_native_build.json"),
            )
        ),
    )
    args = parser.parse_args()
    report = base_report("s2_check_task_ir", args, step="S2")
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Native assertions require no -O/PYTHONOPTIMIZE")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    try:
        manifest = verify_manifest(args.build_manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(args, report, "BLOCKED", f"Native build identity: {error}")
    report["native_build"] = manifest
    report["validation_layer"] = "read-only native analysis and generated source; no NPU launch"
    report["fixture_scope"] = (
        "New S1 fixtures use requested platform; existing legacy fixtures retain their "
        "audited A2/A3/PTO parameters. Pure capability profiles are not device measurements."
    )
    nodes = [f"{LANGUAGE}{filename}::{node}" for filename, names in CONTRACT_TESTS.values() for node in names]
    nodes += [
        "testing/python/ascend_migration/test_s2_task_ir.py",
        "testing/python/ascend_migration/test_s2_capability_ir.py",
        "testing/python/ascend_migration/test_s2_solver_ffi.py",
    ]
    report["required_nodes"] = nodes
    env = dict(os.environ)
    tvm_python = str(Path(manifest["tvm_source_root"]) / "python")
    env["TVM_IMPORT_PYTHON_PATH"] = tvm_python
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), tvm_python, env.get("PYTHONPATH", "")])
    env["TVM_LIBRARY_PATH"] = str(ROOT / "build/tvm")
    env["TILELANG_LIBRARY_PATH"] = str(ROOT / "build")
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(args.build_manifest.resolve())
    env["TL_ASCEND_MIGRATION_PLATFORM"] = args.platform.upper()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"{args.output.stem}-artifacts-", dir=args.output.resolve().parent))
    report["artifacts"] = str(artifacts)
    try:
        result = run_pytest(nodes, artifacts, env=env, timeout=args.timeout)
    except OSError as error:
        return finish(args, report, "BLOCKED", f"Cannot launch native pytest: {error}")
    report["pytest"] = result
    report["counts"] = result["counts"]
    return finish(args, report, result["status"], result.get("error"))


if __name__ == "__main__":
    raise SystemExit(main())
