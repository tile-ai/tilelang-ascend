"""Same-input msprof op timings, kept separate from correctness launch counts."""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
import traceback

from _common import base_report, finish, run_process
from _npu_runner import digest, prepare_worker
from _s5_calibration import compile_kernel, save
from _s5_calibration_matrix import profiler_command
from _s5_device_cases import build_program, data_and_golden, describe_kernel
from _s5_npu import check_identity, pass_configs, tensor_digest

IMPORTED_SOURCE = {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)}


def run_measured_worker(args, spec):
    report = base_report("s5_msprof_worker", args, step="S5")
    report["performance_helper"] = {**IMPORTED_SOURCE, "process_id": os.getpid()}
    report.update(status="BLOCKED", parameters=spec, launches=[], golden_checks=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="timing-worker-", dir=args.output.parent))
    os.environ["TILELANG_CACHE_DIR"] = str(artifacts / "cache")
    save(report, args.output)
    try:
        tilelang, torch = prepare_worker(args, report)
        check_identity(args, report, tilelang)
        kernel = compile_kernel(
            tilelang,
            build_program(spec),
            args,
            report,
            artifacts,
            configs=pass_configs(args, spec["strategy"]),
        )
        arguments, outputs, check, metadata, indices = data_and_golden(spec, args, torch)
        metadata.update(describe_kernel(kernel, spec, indices))
        for round_index in range(3):
            outputs[0].fill_(float("nan"))
            torch.npu.synchronize()
            for phase, count in (("warmup", 5), ("measured", 20)):
                for _ in range(count):
                    kernel(*arguments)
                    torch.npu.synchronize()
                    report["launches"].append(
                        {
                            "index": len(report["launches"]),
                            "round": round_index,
                            "phase": phase,
                        }
                    )
            check()
            report["golden_checks"] += 1
            save(report, args.output)
        metadata["output_sha256"] = tensor_digest(outputs[0], torch)
        report.update(
            case_parameters=metadata,
            sampling_policy={"rounds": 3, "warmup": 5, "samples": 20},
            measurement="msprof op BasicInfo-only Task Duration; warm data; no timed cache flush",
        )
    except Exception as error:
        report["traceback"] = traceback.format_exc()
        return finish(args, report, "FAIL" if "kernel" in report else "BLOCKED", str(error))
    return finish(args, report, "PASS")


def read_execution_samples(directory, worker):
    roots = list(Path(directory).glob("OPPROF*/main_kernel"))
    if worker.get("status") != "PASS" or len(roots) != 1:
        raise ValueError("Missing successful performance worker or ambiguous profiler output")
    journal = [
        {"index": index, "round": round_index, "phase": phase}
        for index, (round_index, phase) in enumerate(
            (round_index, phase) for round_index in range(3) for phase, repeat in (("warmup", 5), ("measured", 20)) for _ in range(repeat)
        )
    ]
    if worker["launches"] != journal or worker["golden_checks"] != 3:
        raise ValueError("Performance journal/golden count differs from the common policy")
    root = roots[0]
    observed = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdecimal())
    if observed != list(range(75)):
        raise ValueError("Profiler invocation IDs do not cover exactly 75 application launches")
    samples = []
    for launch in journal:
        folder = root / str(launch["index"])
        basics = list(folder.glob("OpBasicInfo*.csv"))
        if len(basics) != 1 or list(folder.glob("PipeUtilization*.csv")):
            raise ValueError("Performance requires exactly one BasicInfo-only row per invocation")
        with basics[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError("Ambiguous per-invocation BasicInfo rows")
        row = rows[0]
        if row["Op Name"] != "main_kernel" or int(row["Device Id"]) != worker["actual_device"]["logical_device"]:
            raise ValueError("Performance kernel/device mismatch")
        numeric = Decimal(row["Task Duration(us)"])
        current, rated = int(row["Current Freq"]), int(row["Rated Freq"])
        if not numeric.is_finite() or numeric <= 0 or current != rated:
            raise ValueError("Invalid duration or unstable device frequency")
        samples.append(
            {
                **launch,
                "duration_us_raw": row["Task Duration(us)"],
                "duration_ns": int((numeric * 1000).to_integral_value(rounding=ROUND_CEILING)),
                "current_freq_mhz": current,
                "rated_freq_mhz": rated,
                "basic_info": {"path": str(basics[0]), "sha256": digest(basics[0])},
            }
        )
    if len({(item["current_freq_mhz"], item["rated_freq_mhz"]) for item in samples}) != 1:
        raise ValueError("Performance calls use inconsistent frequencies")
    return samples


def _summary(samples):
    rounds = {}
    for sample in samples:
        if sample["phase"] == "measured":
            rounds.setdefault(sample["round"], []).append(sample["duration_ns"])
    generator = random.Random(0)
    distribution = []
    for _ in range(1000):
        pooled = []
        for index in generator.choices(sorted(rounds), k=len(rounds)):
            pooled.extend(generator.choices(rounds[index], k=len(rounds[index])))
        distribution.append(statistics.median(pooled))
    distribution.sort()
    return {
        "measured_samples": sum(map(len, rounds.values())),
        "median_ns": format(statistics.median(value for group in rounds.values() for value in group), ".12g"),
        "median_ci95_low_ns": format(distribution[24], ".12g"),
        "median_ci95_high_ns": format(distribution[974], ".12g"),
        "round_median_ns": [format(statistics.median(values), ".12g") for values in rounds.values()],
    }


def collect_performance(report, args, entry, bases):
    try:
        return _collect_performance(report, args, entry, bases)
    except OSError as error:
        # run_cases owns the final atomic report and already handles ValueError.
        # Keep collector/file failures on that path instead of losing the JSON.
        raise ValueError(f"Cannot collect performance evidence: {error}") from error


def _collect_performance(report, args, entry, bases):
    """No pass/fail speedup threshold: retain observed medians and confidence ranges."""
    if shutil.which("msprof") is None:
        raise ValueError("msprof op is required for S5 performance evidence")
    directory = Path(report["artifacts"]) / "performance"
    directory.mkdir()
    correctness = {item["name"]: item["evidence"]["case_parameters"] for item in report["cases"]}
    report["performance"] = {
        "helper": {**IMPORTED_SOURCE, "process_id": os.getpid()},
        "method": "msprof op BasicInfo-only, kernel replay, 3 rounds x (5 warmup + 20 measured)",
        "cache_policy": "independent cold compilation per worker; warm inputs; no timed flush",
        "cases": [],
        "comparisons": [],
    }
    by_name = {}
    for base in bases:
        for strategy in ("legacy", "constraint"):
            name = base + "_" + strategy
            target = directory / name
            target.mkdir()
            output = target / "worker.json"
            worker_command = [
                sys.executable,
                str(entry),
                "--platform",
                args.platform,
                "--device",
                str(args.device),
                "--seed",
                str(args.seed),
                "--repeat",
                str(args.repeat),
                "--timeout",
                str(args.timeout),
                "--output",
                str(output),
                "--build-manifest",
                str(args.build_manifest),
                "--cost-profile",
                str(args.cost_profile),
                "--_case",
                name,
                "--_measure",
            ]
            command = profiler_command(worker_command, target / "msprof", launches=75)
            collected = run_process(command, target, "collector", timeout=args.timeout)
            save(collected, target / "collector.json")
            if collected["exit_code"]:
                raise ValueError(f"Performance process did not pass: {collected}")
            worker = json.loads(output.read_text())
            if worker["performance_helper"]["sha256"] != IMPORTED_SOURCE["sha256"]:
                raise ValueError("Performance worker/controller imported different helper source")
            samples = read_execution_samples(target / "msprof", worker)
            identity = args.cost_model["identity"]
            for field in ("current_freq_mhz", "rated_freq_mhz"):
                if any(sample[field] != identity[field] for sample in samples):
                    raise ValueError("Performance frequency differs from the calibrated identity")
            actual = worker["case_parameters"]
            for key in ("input_sha256", "output_sha256", "golden_sha256"):
                if actual[key] != correctness[name][key]:
                    raise ValueError(f"Performance/correctness inputs or outputs differ: {name}/{key}")
            record = {
                "name": name,
                "status": "PASS",
                "collector": collected,
                "worker_report": {"path": str(output), "sha256": digest(output)},
                "summary": _summary(samples),
                "raw_samples": samples,
                "input_sha256": actual["input_sha256"],
                "output_sha256": actual["output_sha256"],
                "resolved_flags": worker["kernel"]["resolved_flags"],
                "compiler_sha256": worker["toolchain"]["sha256"],
                "kernel_source_sha256": worker["kernel"]["source"]["sha256"],
            }
            report["performance"]["cases"].append(record)
            by_name[name] = record
        left, right = [by_name[base + "_" + strategy] for strategy in ("legacy", "constraint")]
        if (
            left["input_sha256"] != right["input_sha256"]
            or left["output_sha256"] != right["output_sha256"]
            or left["resolved_flags"] != right["resolved_flags"]
            or left["compiler_sha256"] != right["compiler_sha256"]
        ):
            raise ValueError("Paired performance strategies use different inputs/outputs")
        report["performance"]["comparisons"].append(
            {
                "fixture": base,
                "legacy": left["summary"],
                "constraint": right["summary"],
                "legacy_over_constraint_median": format(
                    float(left["summary"]["median_ns"]) / float(right["summary"]["median_ns"]),
                    ".12g",
                ),
                "interpretation": "observed kernel-duration ratio; no required speedup threshold",
            }
        )
