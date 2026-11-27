"""Small, standard-library-only helpers for independently runnable migration checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib.metadata
from contextlib import suppress
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "43b0b69720f479894a13c02746599eccee5c5769"
EXIT_CODES = {"PASS": 0, "FAIL": 1, "BLOCKED": 2, "SKIP": 3, "NOT_IMPLEMENTED": 3, "TIMEOUT": 124}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.PIPE).strip()


def base_report(entry: str, args: argparse.Namespace, *, step: str = "S1") -> dict:
    """Capture the tree actually under test, without importing a device runtime."""
    versions = {"python": platform.python_version()}
    for name in ("pytest", "torch", "torch-npu", "tilelang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1,
        "step": step,
        "entry": entry,
        "command": [sys.executable, *sys.argv],
        "commit": git("rev-parse", "HEAD"),
        "baseline": BASELINE,
        "dirty": bool(git("status", "--porcelain")),
        "source_status": git("status", "--porcelain", "--untracked-files=all"),
        "platform": getattr(args, "platform", None),
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "repeat": getattr(args, "repeat", None),
        "versions": versions,
        "python_optimization": sys.flags.optimize,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "total": 0},
    }


def finish(args: argparse.Namespace, report: dict, status: str, error: str | None = None) -> int:
    report["status"] = status
    report["exit_code"] = EXIT_CODES[status]
    if error:
        report["error"] = error
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
    temporary.replace(output)
    print(f"{status}: {output}")
    if error:
        print(error, file=sys.stderr)
    return EXIT_CODES[status]


def run_process(
    command: Sequence[str],
    artifact_dir: str | Path,
    name: str,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> dict:
    """Keep possibly hanging device work in its own killable process group."""
    directory = Path(artifact_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{name}.log"
    started = time.monotonic()
    result = {"command": list(map(str, command)), "log": str(log)}
    with log.open("w") as stream:
        process = subprocess.Popen(
            result["command"],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            # Kill remaining descendants even if the process leader already exited.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            code = 124
    result.update(exit_code=code, elapsed_seconds=time.monotonic() - started)
    result["status"] = "PASS" if code == 0 else "TIMEOUT" if code == 124 else "FAIL"
    return result


def run_pytest(
    nodes: Sequence[str],
    artifact_dir: str | Path,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> dict:
    """Use pytest's own assertions and JUnit report, including skipped cases."""
    directory = Path(artifact_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    junit = directory / "pytest.xml"
    junit.unlink(missing_ok=True)
    child_env = dict(os.environ if env is None else env)
    child_env.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-m",
        "",
        f"--junitxml={junit}",
        *nodes,
    ]
    result = run_process(command, directory, "pytest", child_env, timeout)
    result.update(junit=str(junit), counts={"passed": 0, "failed": 0, "skipped": 0, "total": 0}, cases=[])
    if result["exit_code"] == 124:
        return result
    if not junit.is_file():
        result.update(status="BLOCKED", error="pytest did not produce a JUnit report")
        return result
    try:
        cases = ET.parse(junit).getroot().iter("testcase")
        for case in cases:
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            status = "failed" if failure is not None or error is not None else "skipped" if skipped is not None else "passed"
            result["counts"][status] += 1
            result["counts"]["total"] += 1
            record = {
                "name": case.get("name"),
                "classname": case.get("classname"),
                "status": status,
            }
            detail = failure if failure is not None else error if error is not None else skipped
            if detail is not None:
                record["message"] = detail.get("message", "")
            result["cases"].append(record)
    except (ET.ParseError, OSError) as exc:
        result.update(status="BLOCKED", error=f"Invalid JUnit report: {exc}")
        return result
    counts = result["counts"]
    if result["exit_code"] in (2, 3, 4):
        result["status"] = "BLOCKED"
    elif result["exit_code"] or counts["failed"] or not counts["total"]:
        result["status"] = "FAIL"
    elif counts["skipped"]:
        result["status"] = "SKIP" if counts["skipped"] == counts["total"] else "BLOCKED"
    else:
        result["status"] = "PASS"
    missing = []
    for node in nodes:
        filename, *qualifiers = node.split("::")
        if not qualifiers:
            if not any(Path(filename).stem in (case["classname"] or "").split(".") for case in result["cases"]):
                missing.append(node)
            continue
        function = qualifiers[-1]
        module = Path(filename).stem
        found = False
        for case in result["cases"]:
            classname = (case["classname"] or "").split(".")
            name = case["name"] or ""
            function_matches = name == function or ("[" not in function and name.startswith(function + "["))
            if module in classname and function_matches:
                class_position = classname.index(module) + 1
                if classname[class_position:] == qualifiers[:-1]:
                    found = True
                    break
        if not found:
            missing.append(node)
    result["missing_required_nodes"] = missing
    if missing and result["status"] == "PASS":
        result.update(status="FAIL", error="Mandatory pytest nodes were not executed")
    return result
