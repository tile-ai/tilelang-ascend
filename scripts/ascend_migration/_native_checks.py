"""Shared identity, preflight and reporting for native-only migration checks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from _common import ROOT, base_report, finish, run_process, run_pytest
from _native_build import verify_manifest


def parser(description: str, *, step: str = "S3") -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--platform", choices=("a3",), required=True)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                str(ROOT / f"build/ascend_migration_{step.lower()}_native_build.json"),
            )
        ),
    )
    return result


def run(args, entry: str, nodes, *, step: str = "S3", require_solver: bool = False) -> int:
    report = base_report(entry, args, step=step)
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Native assertions require no -O/PYTHONOPTIMIZE")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    try:
        manifest = verify_manifest(args.build_manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return finish(args, report, "BLOCKED", f"Native build identity: {error}")
    report["native_build"] = manifest
    report["solver_required"] = require_solver
    report["validation_layer"] = "native IR, interfaces and generated source; no NPU launch"
    report["required_nodes"] = list(nodes)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    tvm_python = Path(manifest["tvm_source_root"]) / "python"
    env["TVM_IMPORT_PYTHON_PATH"] = str(tvm_python)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(tvm_python), env.get("PYTHONPATH", "")])
    env["TVM_LIBRARY_PATH"] = str(ROOT / "build/tvm")
    env["TILELANG_LIBRARY_PATH"] = str(ROOT / "build")
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(args.build_manifest.resolve())
    env["TL_ASCEND_MIGRATION_PLATFORM"] = args.platform.upper()
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"{args.output.stem}-artifacts-", dir=args.output.resolve().parent))
    report["artifacts"] = str(artifacts)
    solver_preflight = "import z3\nsolver_version = z3.get_version_string()\n" if require_solver else "solver_version = None\n"
    preflight = (
        solver_preflight
        + """import json
from pathlib import Path
import tilelang
from tilelang import tvm
print('NATIVE_IDENTITY=' + json.dumps({
    'tilelang_python': tilelang.__file__,
    'tvm_python': tvm.__file__,
    'tilelang_library': str(Path(tilelang._LIB_PATH).resolve()),
    'tvm_library': str(Path(tvm._ffi.base._LIB._name).resolve()),
    'tvm_version': tvm.__version__,
    'solver_version': solver_version,
}))
"""
    )
    try:
        checked = run_process([sys.executable, "-c", preflight], artifacts, "preflight", env=env, timeout=args.timeout)
        report["native_preflight"] = checked
        if checked["exit_code"]:
            status = "TIMEOUT" if checked["exit_code"] == 124 else "BLOCKED"
            return finish(args, report, status, "Native import preflight did not pass; see log")
        payload = next(
            line.removeprefix("NATIVE_IDENTITY=")
            for line in Path(checked["log"]).read_text().splitlines()
            if line.startswith("NATIVE_IDENTITY=")
        )
        loaded = json.loads(payload)
        expected = {item["path"] for item in manifest["libraries"]}
        if loaded["tilelang_library"] not in expected or loaded["tvm_library"] not in expected:
            return finish(args, report, "BLOCKED", "Native preflight loaded libraries outside manifest")
        if Path(loaded["tilelang_python"]).resolve() != (ROOT / "tilelang/__init__.py").resolve():
            return finish(args, report, "BLOCKED", "Native preflight loaded a different TileLang package")
        if Path(loaded["tvm_python"]).resolve() != (tvm_python / "tvm/__init__.py").resolve():
            return finish(args, report, "BLOCKED", "Native preflight loaded unverified TVM Python")
        report["native_environment"] = loaded
        result = run_pytest(nodes, artifacts, env=env, timeout=args.timeout)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        return finish(args, report, "BLOCKED", f"Cannot execute native validation: {error}")
    report["pytest"] = result
    report["counts"] = result["counts"]
    return finish(args, report, result["status"], result.get("error"))
