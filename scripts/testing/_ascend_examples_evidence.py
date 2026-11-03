# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Classify complete example entries from recorded execution evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


FAILURE_PATTERNS = (
    r"^Case \d+: FAILED -.*$",
    r"^测试完成: \d+ passed, [1-9]\d* failed$",
    r"^Some tests failed\.$",
    r"^\s+FAILED \(max_diff=.*$",
    r"^Test failed! The precision is not correct!$",
    r"^\s+\[FAILED\] .*$",
    r"^\s*\[(?:PRECISION_FAIL|LEVEL_FAIL)\].*$",
    r"^\s*Match: False\s*$",
    r"^\s*Is output sorted \(descending\): False\s*$",
    r"^\s*\d+-way merge: FAIL\s*$",
    r"^FAILED\s+\S+\.py(?:::\S+)*(?:\s+-\s+.*)?$",
    r"^\s*(?:=+\s*)?[1-9]\d* failed(?:, [^\n]*| in [^\n]*|\s*=+)?\s*$",
)


def failure_evidence(log: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Recognize failures without consulting the example's filename."""
    matches = [
        {"kind": "sample_failure_message", "message": match.group(0).strip()}
        for pattern in FAILURE_PATTERNS
        for match in re.finditer(pattern, log, re.MULTILINE)
    ]
    for field, kind in (
        ("assertion_failures", "assertion_failure"),
        ("launch_failures", "launch_failure"),
    ):
        failures = metadata.get(field, [])
        if isinstance(failures, list):
            matches.extend({"kind": kind, "detail": failure} for failure in failures)
        else:
            matches.append({"kind": "invalid_run_evidence", "message": f"invalid {field}"})
    kernels = metadata.get("kernels", [])
    if isinstance(kernels, list):
        matches.extend(
            {"kind": "compile_failure", "message": str(kernel["error"])}
            for kernel in kernels
            if isinstance(kernel, dict) and kernel.get("error") is not None
        )
    sessions = metadata.get("pytest_sessions", [])
    if not isinstance(sessions, list):
        matches.append({"kind": "invalid_run_evidence", "message": "invalid pytest sessions"})
        sessions = []
    for session in sessions:
        if not isinstance(session, dict):
            matches.append({"kind": "invalid_pytest_session"})
            continue
        if type(session.get("exitstatus")) is not int or session["exitstatus"] != 0:
            matches.append(
                {"kind": "pytest_exitstatus", "value": session.get("exitstatus")},
            )
        reports = session.get("failed_reports")
        if isinstance(reports, list):
            matches.extend({"kind": "pytest_failed_report", "detail": report} for report in reports)
        else:
            matches.append({"kind": "invalid_pytest_session", "message": "missing reports"})
    unique = []
    seen = set()
    for match in matches:
        key = json.dumps(match, sort_keys=True, default=str)
        if key not in seen:
            unique.append(match)
            seen.add(key)
    return unique


def _diagnostic(log: str, metadata: dict[str, Any]) -> str:
    body = log.split("PR1965_RESULT ", 1)[0]
    for line in body.splitlines():
        match = re.search(
            r"Check failed:.*|\b(?:TVMError|InternalError|CompileError): (?!Traceback).*",
            line,
        )
        if match:
            return match.group(0).strip()
    error = metadata.get("error")
    if error:
        return str(error).strip().splitlines()[-1]
    kernels = metadata.get("kernels", [])
    if isinstance(kernels, list):
        for kernel in kernels:
            if isinstance(kernel, dict) and kernel.get("error"):
                return str(kernel["error"]).strip().splitlines()[-1]
    return ""


def _normalized_configs(configs: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    normalized = {}
    for key, value in configs.items():
        canonical = aliases.get(key, key)
        if canonical in normalized and normalized[canonical] != value:
            raise ValueError(f"conflicting configuration aliases for {canonical}")
        normalized[canonical] = value
    return normalized


def classify_result(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    log: str,
    returncode: int,
    timed_out: bool,
    expected_abi: str,
    device: int,
    aliases: dict[str, str],
    run_dir: Path,
    key: str,
) -> dict[str, Any]:
    """Require actual compilation, launch and matching provenance for PASS.

    Zero captured assertions is an instrumentation gap, not proof of a failed
    example. A program without a recorded TileLang launch is NO_EXECUTION.
    """
    if not isinstance(metadata, dict):
        metadata = {}
    evidence = failure_evidence(log, metadata)
    diagnostic = _diagnostic(log, metadata)
    source_hashes = {}

    def finish(status: str, reason: str = "") -> dict[str, Any]:
        assertions = metadata.get("assertions", 0)
        verified = status == "PASS" and type(assertions) is int and assertions > 0
        return {
            "status": status,
            "failure_evidence": evidence,
            "diagnostic": diagnostic or reason,
            "precision_verified": verified,
            "precision_validation": (
                "instrumented_assertions_passed"
                if verified
                else "entry_completed_without_instrumented_assertions"
                if status == "PASS"
                else "not_an_entry_pass"
            ),
            "generated_sources_sha256": source_hashes,
        }

    if timed_out:
        return finish("TIMEOUT", "Entry exceeded its timeout")
    if returncode != 0:
        return finish("FAIL", f"Example process exited with status {returncode}")
    if not metadata:
        evidence.append({"kind": "invalid_run_evidence", "message": "missing child metadata"})
        return finish("FAIL", "Child exited without execution metadata")
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(type(metadata.get("returncode")) is int, "invalid child exit status")
    require(metadata.get("returncode") == 0, "nonzero child exit status")
    require(metadata.get("error") is None, "child recorded an error")
    require(metadata.get("path") == entry["path"], "wrong child path")
    require(metadata.get("mode") == entry["mode"], "wrong child mode")
    require(metadata.get("variant") == "upstream", "wrong execution variant")
    require(
        type(metadata.get("npu_device")) is int and metadata["npu_device"] == device,
        "wrong execution device",
    )
    require(metadata.get("native_abi") == expected_abi, "wrong native synchronization ABI")
    require(metadata.get("cache_disabled") is True, "TileLang cache was not disabled")
    require(metadata.get("original_args") == entry.get("args", []), "changed script arguments")
    abis = re.findall(r"^PR1965_ENGINE_ABI (.+)$", log, re.MULTILINE)
    require(abis == [expected_abi], "missing or mismatched ABI log record")
    records = re.findall(r"^PR1965_RESULT (.+)$", log, re.MULTILINE)
    try:
        require(
            len(records) == 1 and json.loads(records[0]) == metadata,
            "missing or mismatched child result log record",
        )
    except (ValueError, TypeError):
        errors.append("malformed child result log record")
    counters = ("compile_attempts", "compiled", "launches", "assertions")
    for counter in counters:
        value = metadata.get(counter)
        require(type(value) is int and value >= 0, f"invalid {counter} counter")
    kernels = metadata.get("kernels")
    sessions = metadata.get("pytest_sessions")
    require(isinstance(kernels, list), "missing kernel records")
    require(isinstance(sessions, list), "missing pytest session records")
    require(isinstance(metadata.get("assertion_failures"), list), "missing assertion records")
    require(isinstance(metadata.get("launch_failures"), list), "missing launch failure records")
    if entry["mode"] == "pytest":
        require(bool(sessions), "missing required pytest session")
        require(
            metadata.get("pytest_markers") == entry.get("pytest_markers", "not bench"),
            "changed pytest selection",
        )
    if isinstance(kernels, list):
        require(metadata.get("compiled") == len(kernels), "incomplete kernel records")
    require(
        metadata.get("compile_attempts") == metadata.get("compiled"),
        "an attempted compilation did not complete",
    )
    if errors:
        evidence.extend({"kind": "invalid_run_evidence", "message": error} for error in errors)
        return finish("FAIL", errors[0])
    if evidence:
        return finish("FAIL", str(evidence[0].get("message", evidence[0])))

    directory = (run_dir / "kernels").resolve()
    for index, kernel in enumerate(kernels, 1):
        if not isinstance(kernel, dict):
            errors.append("invalid kernel record")
            continue
        require(kernel.get("index") == index, "nonsequential kernel index")
        require(kernel.get("error") is None, "kernel recorded a compilation error")
        configs = kernel.get("pass_configs")
        if not isinstance(configs, dict):
            errors.append("missing kernel configurations")
            continue
        try:
            configs = _normalized_configs(configs, aliases)
        except ValueError as error:
            errors.append(str(error))
            continue
        require(configs.get("tl.ascend_sync_backend") == "upstream", "wrong sync backend")
        for option in ("tl.ascend_auto_sync", "tl.ascend_auto_sync_vs"):
            require(configs.get(option) is True, f"disabled {option}")
        flags = kernel.get("compile_flags", [])
        if isinstance(flags, list) and all(isinstance(flag, str) for flag in flags):
            sync = [flag for flag in flags if flag.startswith("--cce-auto-sync=")]
            require(bool(sync) and sync[-1] == "--cce-auto-sync=off", "Bisheng sync enabled")
        else:
            errors.append("invalid compiler flags")
        source = kernel.get("source")
        if not isinstance(source, str):
            errors.append("missing generated source path")
            continue
        try:
            path = Path(source)
            if not path.is_absolute():
                path = run_dir / path
            path = path.resolve()
            expected_name = f"{key}-{index:03d}.cc"
            if path.parent != directory or path.name != expected_name:
                errors.append("generated source belongs to another entry or directory")
                continue
            if not path.is_file() or path.stat().st_size == 0:
                errors.append("missing generated kernel source")
                continue
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            require(kernel.get("source_sha256") == source_hash, "generated kernel source changed")
            source_hashes[str(path)] = source_hash
        except (OSError, ValueError, RuntimeError):
            errors.append("unreadable generated kernel source")
    if errors:
        evidence.extend({"kind": "invalid_run_evidence", "message": error} for error in errors)
        return finish("FAIL", errors[0])
    if metadata["launches"] == 0:
        return finish("NO_EXECUTION", "No recorded TileLang kernel launch")
    if metadata["compiled"] == 0:
        evidence.append({"kind": "invalid_run_evidence", "message": "launch without compilation"})
        return finish("FAIL", "Recorded launch has no corresponding compilation")
    return finish("PASS")
