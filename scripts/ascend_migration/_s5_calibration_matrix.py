"""Reproducible legacy calibration controller; no runtime imports in this process."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

from _common import ROOT, finish, run_process
from _npu_runner import digest
from _s5_calibration import assemble_profile, fit_effective_cost, read_profiler_samples, save


class CollectionFailure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def merge_attempts(batches):
    """Keep every failed/passing batch and give its rounds disjoint identities."""
    if not 1 <= len(batches) <= 2:
        raise ValueError("Calibration permits at most two complete attempts")
    result = []
    for attempt, batch in enumerate(batches):
        if len(batch) != 375 or {item["round"] for item in batch} != {0, 1, 2}:
            raise ValueError("Every attempt must retain all 375 launches and all three rounds")
        result.extend(
            {
                **sample,
                "attempt": attempt,
                "round_within_attempt": sample["round"],
                "round": sample["round"] + attempt * 3,
            }
            for sample in batch
        )
    return result


def profiler_command(worker, directory, *, launches=375):
    return [
        "msprof",
        "op",
        "--kernel-name=main_kernel",
        "--aic-metrics=BasicInfo",
        "--warm-up=0",
        f"--launch-count={launches}",
        "--replay-mode=kernel",
        "--kill=off",
        "--output=" + str(directory),
        "--application=" + shlex.join(worker),
    ]


def run_calibration(args, report):
    """Run accuracy then BasicInfo-only collection for all 32 fixed geometries.

    An optional independent, verified source snapshot can retain a legacy
    baseline while the invoking repository is rebuilt. Its own manifest helper
    checks the snapshot; no source path or original CMake identity is rewritten.
    """
    source = Path(args.calibration_source_root or ROOT).resolve()
    try:
        helper = source / "scripts/ascend_migration/_native_build.py"
        worker = source / "scripts/ascend_migration/_s5_calibration.py"
        if not worker.is_file():
            raise ValueError("Calibration source does not contain the measurement worker")
        native = _module("s5_calibration_native_identity", helper)
        report["native_build"] = native.verify_manifest(args.build_manifest)
        if shutil.which("msprof") is None:
            raise ValueError("msprof op is required for real calibration")
    except (OSError, ValueError, KeyError, ImportError) as error:
        return finish(args, report, "BLOCKED", f"Legacy calibration preflight: {error}")
    directory = Path(tempfile.mkdtemp(prefix=args.output.stem + "-calibration-", dir=args.output.parent))
    report.update(
        calibration_source_root=str(source),
        artifacts=str(directory),
        calibration_mode="legacy effective operation costs; no existing profile required",
        required_cases=32,
        cases=[],
        status="RUNNING",
    )
    report["counts"]["total"] = 32
    save(report, args.output)
    cases = []
    try:
        for operation in ("gmub", "ubgm", "fill", "add"):
            for dtype in ("float16", "float32"):
                for tile in (32, 64):
                    for pattern in ("dependent", "independent"):
                        name = f"{operation}-{dtype}-t{tile}-{pattern}"
                        root = directory / name
                        root.mkdir()
                        common = [
                            sys.executable,
                            str(worker),
                            "--platform",
                            args.platform,
                            "--device",
                            str(args.device),
                            "--seed",
                            str(args.seed),
                            "--operation",
                            operation,
                            "--pattern",
                            pattern,
                            "--dtype",
                            dtype,
                            "--tile",
                            str(tile),
                            "--build-manifest",
                            str(Path(args.build_manifest).resolve()),
                        ]
                        env = dict(os.environ)
                        env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
                        accuracy = root / "accuracy.json"
                        report["active_case"] = name
                        report["active_phase"] = "accuracy"
                        save(report, args.output)
                        checked = run_process(
                            [*common, "--mode", "accuracy", "--output", str(accuracy)],
                            root,
                            "accuracy",
                            env=env,
                            timeout=args.timeout,
                        )
                        if checked["exit_code"]:
                            raise CollectionFailure(
                                "TIMEOUT" if checked["exit_code"] == 124 else "BLOCKED" if checked["exit_code"] == 2 else "FAIL",
                                f"Accuracy process failed: {checked}",
                            )
                        accuracy_report = json.loads(accuracy.read_text())
                        if (
                            accuracy_report.get("status") != "PASS"
                            or accuracy_report.get("golden_checks") != 15
                            or len(accuracy_report.get("compilations", [])) != 1
                        ):
                            raise CollectionFailure("FAIL", f"Incomplete accuracy checks: {accuracy}")
                        batches, attempts, commands, worker_reports = [], [], [], []
                        first_worker = None
                        for attempt in range(2):
                            target = root / f"attempt-{attempt}"
                            target.mkdir()
                            output, collector = target / "profile.json", target / "msprof"
                            command = profiler_command([*common, "--mode", "profile", "--output", str(output)], collector)
                            report["active_phase"] = f"BasicInfo-attempt-{attempt}"
                            save(report, args.output)
                            result = run_process(
                                command,
                                target,
                                "msprof",
                                env=env,
                                timeout=args.calibration_timeout,
                            )
                            save(result, target / "collector.json")
                            if result["exit_code"]:
                                raise CollectionFailure(
                                    "TIMEOUT" if result["exit_code"] == 124 else "BLOCKED",
                                    f"Profiler process failed: {result}",
                                )
                            observed = json.loads(output.read_text())
                            if first_worker is None:
                                first_worker = observed
                            elif (
                                first_worker["kernel"]["source"]["sha256"] != observed["kernel"]["source"]["sha256"]
                                or first_worker["toolchain"] != observed["toolchain"]
                            ):
                                raise ValueError("A bounded repeat changed implementation/toolchain")
                            batch = read_profiler_samples(collector, observed)
                            batch_fit = fit_effective_cost(batch)
                            save({"fit": batch_fit, "raw_samples": batch}, target / "fit.json")
                            reference = {"path": str(output), "sha256": digest(output)}
                            worker_reports.append(reference)
                            commands.append(command)
                            attempts.append({"worker_report": reference, "fit": batch_fit})
                            batches.append(batch)
                            samples = merge_attempts(batches)
                            fit = fit_effective_cost(samples)
                            save(
                                {
                                    "fit": fit,
                                    "raw_samples": samples,
                                    "includes_initial_failed_attempt": attempt == 1,
                                },
                                root / "combined-fit.json",
                            )
                            if fit["status"] == "PASS":
                                break
                        if fit["status"] != "PASS":
                            raise ValueError(f"{name}: two-attempt combined fit rejected: {fit}")
                        cases.append(
                            {
                                "worker": first_worker,
                                "samples": samples,
                                "fit": fit,
                                "command": commands[0],
                                "collector_commands": commands,
                                "worker_report": worker_reports[0],
                                "worker_reports": worker_reports,
                                "attempts": attempts,
                                "accuracy_report": {
                                    "path": str(accuracy),
                                    "sha256": digest(accuracy),
                                },
                            }
                        )
                        report["cases"].append(
                            {
                                "name": name,
                                "status": "PASS",
                                "attempt_count": len(attempts),
                                "fit": fit,
                                "application_launches": len(samples),
                                "accuracy_report": cases[-1]["accuracy_report"],
                                "accuracy_golden_checks": accuracy_report["golden_checks"],
                                "accepted_pilot_reused": False,
                                "profile_attempts": attempts,
                                "profile_worker_reports": worker_reports,
                            }
                        )
                        report["counts"]["passed"] += 1
                        save(report, args.output)
        profile = assemble_profile(cases, quantum_ns=1)
        validator = _module("s5_calibration_cost_validator", ROOT / "tilelang/transform/ascend_pipeline_cost.py")
        validator.validate_cost_model(profile)
        save(profile, args.calibrate_profile)
        report.update(
            profile=str(args.calibrate_profile.resolve()),
            profile_sha256=profile["profile_sha256"],
            application_launches=profile["evidence"]["application_launches"],
            measured_samples=profile["evidence"]["measured_samples"],
            accuracy_launches_and_golden_checks=sum(case["accuracy_golden_checks"] for case in report["cases"]),
            profile_post_group_golden_checks=sum(15 * len(case["profile_attempts"]) for case in report["cases"]),
        )
    except CollectionFailure as error:
        return finish(args, report, error.status, str(error))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return finish(args, report, "BLOCKED", str(error))
    finally:
        report.pop("active_case", None)
        report.pop("active_phase", None)
    return finish(args, report, "PASS")
