"""Run all active Gated MLA core suites on a real NPU with hosted pytest flags.

Records the full pytest process wall time: collection, empty kernel-cache
compilation, loading, test bodies and audits. Exit zero requires both correctness
and the 120-second gate. Native/Cython toolchain caches are retained.
The runtime must already provide pytest-forked and its complete py dependency.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("external output must be new")
    args.output.mkdir(parents=True)
    resolver = args.repo_root / "scripts/ci/resolve_operator_tests.py"
    listed = subprocess.run(
        [sys.executable, str(resolver), "list-tests", "--experiment-dirs", "gated_mla"],
        cwd=args.repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    tests = listed.stdout.splitlines()
    if not tests or not any(t.endswith("test_gated_mla_decode.py") for t in tests):
        raise AssertionError("directory route must retain the original L0")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--strict-markers",
        "-m",
        "not (low_priority or ci_skip)",
        "-o",
        f"pythonpath={args.runtime_root} {args.repo_root / 'testing/python'} {args.repo_root}",
        "--durations=30",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    cmd += ["--forked", "-n", "8"]
    cmd += tests
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        GMLA_CI_OUTPUT=str(args.output / "automatic_ci"),
        GMLA_CI_RUNTIME_ROOT=str(args.runtime_root),
        GMLA_CI_COMPILE_WORKERS=str(args.workers),
    )
    env.pop("GMLA_CI_COMPILE_ON_COLLECT", None)
    started = time.perf_counter()
    with (args.output / "pytest.log").open("w") as log:
        result = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    elapsed = time.perf_counter() - started
    ci = args.output / "automatic_ci"
    artifacts_ok, artifact_error = False, None
    try:
        pool = json.loads((ci / "compile/pool_report.json").read_text())
        ready = [json.loads(p.read_text()) for p in ci.glob("ready_*.json")]
        audits = [json.loads(p.read_text()) for p in ci.glob("session_audit_*.json")]
        assert pool["status"] == "CPU_compile_and_publish_passed_parent_load_deferred"
        assert ready and audits
        assert all(a["published_cache_unchanged"] and not a["native_compile_violations"] and a["pytest_exitstatus"] == 0 for a in audits)
        assert all(len(r["loaded"]) == pool["unique_factories"] for r in ready)
        artifacts_ok = True
    except Exception as error:
        artifact_error = repr(error)
    report = {
        "returncode": result.returncode,
        "artifact_checks_passed": artifacts_ok,
        "artifact_error": artifact_error,
        "command": cmd,
        "test_files": tests,
        "automatic_conftest_integration": True,
        "process_wall_seconds_including_collection_compile_load_tests_audit": elapsed,
        "executed_test_bodies": True,
        "hosted_parallel_flags": True,
        "120s_gate_passed": result.returncode == 0 and artifacts_ok and elapsed <= 120,
        "scope": "whole gated_mla directory core, including legacy L0; actual NPU execution",
    }
    (args.output / "external_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    raise SystemExit(0 if report["120s_gate_passed"] else 1)


if __name__ == "__main__":
    main()
