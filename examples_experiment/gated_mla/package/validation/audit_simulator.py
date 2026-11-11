"""Audit real simulator log launch count, successful completion and raw artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    entry = json.loads((args.entry_dir / "entry.json").read_text())
    log = args.log.read_text(errors="replace")
    starts = re.findall(r"Start profiling on kernel:\s*(\S+)", log)
    expected = entry["expected_physical_launches"]
    if entry["precision"] != "pass" or entry["application_calls"] != 1 or entry["warmup_calls"] != 0:
        raise AssertionError("Entry was not one successful forward without warmup")
    if len(starts) != expected or any(not name.startswith("main_kernel") for name in starts):
        raise AssertionError(f"Expected {expected} filtered physical launches; got {starts}")
    required = (
        "GMLA_RELEASE_SINGLE_SHOT_PASS",
        "Model stopped successfully.",
        "Profiling running finished. All task success.",
        "Profiling data parse finished.",
    )
    if any(marker not in log for marker in required):
        raise AssertionError("Simulator/entry completion markers missing")
    artifacts = sorted(str(p.relative_to(args.profile_dir)) for p in args.profile_dir.rglob("*") if p.is_file())
    if not artifacts:
        raise AssertionError("No raw simulator artifacts")
    warnings = sorted(set(line.strip() for line in log.splitlines() if "[WARN]" in line))
    report = dict(
        status="pass",
        source="actual msprof op simulator log and raw output",
        geometry=entry["geometry"],
        filtered_physical_launches=len(starts),
        launch_order=[
            dict(ordinal=i, stage=stage, reported_name=name)
            for i, (stage, name) in enumerate(zip(entry["launch_order"], starts, strict=True))
        ],
        source_sha256=entry["source_sha256"],
        kernel_source_sha256=entry["kernel_source_sha256"],
        entry_sha256=hashlib.sha256((args.entry_dir / "entry.json").read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        artifacts=artifacts,
        warnings=warnings,
        scope="Filtered main_kernel launches only; not a whole-process trace. Simulator timings are not hardware timings.",
        source_line_mapping="unavailable"
        if any("code info" in w or "call stack" in w or "debug_line" in w for w in warnings)
        else "inspect raw artifacts",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GMLA_RELEASE_SIMULATOR_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
