#!/usr/bin/env python3
"""Validate S2's numeric solver and independent verifier without TileLang/TVM."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys

from _common import ROOT, base_report, finish, run_pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = base_report("s2_check_solver", args, step="S2")
    try:
        report["versions"]["z3-solver"] = importlib.metadata.version("z3-solver")
    except importlib.metadata.PackageNotFoundError:
        report["versions"]["z3-solver"] = None
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Python optimization disables mandatory assertions")
    if not 0 < args.timeout < float("inf"):
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    module_path = ROOT / "tilelang/transform/ascend_z3_scheduler.py"
    spec = importlib.util.spec_from_file_location("ascend_scheduler_host_check", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    probe = module.solve_schedule(
        {
            "mode": "straight",
            "latencies": [1],
            "iis": [1],
            "resource_flags": [1],
        }
    )
    report["solver_probe"] = probe
    if "z3" in sys.modules:
        report["solver_module"] = sys.modules["z3"].__file__
    if probe["status"] == "MISSING_DEPENDENCY":
        return finish(args, report, "BLOCKED", probe["reason"])
    if probe["status"] != "SAT":
        status = "TIMEOUT" if probe["status"] == "TIMEOUT" else "FAIL"
        return finish(args, report, status, probe["reason"])
    artifacts = args.output.resolve().parent / (args.output.stem + "-artifacts")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = run_pytest(
        [
            str(ROOT / "testing/python/ascend_migration/test_s2_solver.py"),
            str(ROOT / "testing/python/ascend_migration/test_s2_solver_oracle.py"),
        ],
        artifacts,
        env=env,
        timeout=args.timeout,
    )
    report["verification"] = result
    report["counts"] = result["counts"]
    report["timing_source"] = "synthetic integer hints; no hardware latency table"
    return finish(args, report, result["status"], result.get("error"))


if __name__ == "__main__":
    raise SystemExit(main())
