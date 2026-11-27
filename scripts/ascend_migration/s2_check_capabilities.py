"""Validate the host-only A3 capability contract without importing a runtime."""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys

from _common import ROOT, base_report, finish, run_pytest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["a3"], default="a3")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = base_report("s2_check_capabilities", args, step="S2")
    report["validation_layer"] = "host capability contracts only; no native or device execution"
    if sys.flags.optimize:
        return finish(args, report, "BLOCKED", "Run capability assertions without Python -O/PYTHONOPTIMIZE")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        return finish(args, report, "BLOCKED", "--timeout must be finite and positive")
    try:
        path = ROOT / "tilelang/transform/ascend_schedule_capabilities.py"
        spec = importlib.util.spec_from_file_location("s2_capabilities_entry", path)
        model = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = model
        spec.loader.exec_module(model)
        report["capabilities"] = model.get_capabilities(args.platform).to_dict()
    except (OSError, ImportError, ValueError) as error:
        return finish(args, report, "FAIL", str(error))
    try:
        result = run_pytest(
            ["testing/python/ascend_migration/test_s2_capabilities.py"],
            args.output.resolve().parent / f"{args.output.stem}-artifacts",
            env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
            timeout=args.timeout,
        )
    except OSError as error:
        return finish(args, report, "BLOCKED", f"Cannot start capability checks: {error}")
    report["checks"] = [result]
    report["counts"] = result["counts"]
    return finish(args, report, result["status"], result.get("error"))


if __name__ == "__main__":
    raise SystemExit(main())
