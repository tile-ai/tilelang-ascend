#!/usr/bin/env python3
"""Collect legacy A3 effective-cost microbenchmarks; raw data are not a timing model."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from _common import ROOT, base_report, finish
from _npu_runner import digest, prepare_worker

COUNTS = (0, 32, 128, 256, 512)


def read_profiler_samples(directory, worker, *, metrics="BasicInfo"):
    """Attribute the observed CANN 9.1 per-invocation reports to the launch journal."""
    if worker.get("status") != "PASS":
        raise ValueError("The profiling worker did not pass its output checks")
    roots = list(Path(directory).glob("OPPROF*/main_kernel"))
    if len(roots) != 1:
        raise ValueError("Expected exactly one profiler run and matching kernel")
    root = roots[0]
    observed = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdecimal())
    launches = worker["launches"]
    counts = worker.get("count_values", sorted({launch["count"] for launch in launches}))
    if len(counts) != 5 or counts[0] != 0:
        raise ValueError("Expected five operation counts including the fixed-cost baseline")
    expected = []
    parameters = worker["parameters"]
    if parameters["mode"] != "profile":
        raise ValueError("The worker did not run in profiling mode")
    for round_index in range(parameters["rounds"]):
        for count in counts:
            for phase, repeat in (
                ("warmup", parameters["warmup"]),
                ("measured", parameters["samples"]),
            ):
                for _ in range(repeat):
                    expected.append(
                        {
                            "index": len(expected),
                            "round": round_index,
                            "count": count,
                            "phase": phase,
                        }
                    )
    if launches != expected:
        raise ValueError("Launch journal does not exactly match the declared sampling policy")
    if observed != list(range(len(launches))):
        raise ValueError("Profiler invocation IDs do not exactly cover the launch journal")
    result = []
    for index, launch in enumerate(launches):
        if launch["index"] != index:
            raise ValueError("Nonsequential profiling launch journal")
        basics = list((root / str(index)).glob("OpBasicInfo*.csv"))
        pipes = list((root / str(index)).glob("PipeUtilization*.csv"))
        pipe_metrics = "PipeUtilization" in metrics.split(",")
        if len(basics) != 1 or len(pipes) != int(pipe_metrics):
            raise ValueError("Missing or ambiguous per-invocation BasicInfo/PipeUtilization")
        with basics[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError("Expected one BasicInfo row per application kernel invocation")
        row = rows[0]
        if row["Op Name"] != "main_kernel" or int(row["Device Id"]) != worker["actual_device"]["logical_device"]:
            raise ValueError("Profiler kernel/device differs from the verified worker")
        vector = []
        if pipe_metrics:
            with pipes[0].open(newline="") as stream:
                vector = [row for row in csv.DictReader(stream) if row["sub_block_id"] == "vector0"]
            if len(vector) != 1:
                raise ValueError("Expected one active Vector subcore in the calibration fixture")

        def ns(value):
            numeric = Decimal(value)
            if not numeric.is_finite() or numeric < 0:
                raise ValueError("Profiler duration must be finite and nonnegative")
            return int((numeric * 1000).to_integral_value(rounding=ROUND_CEILING))

        sample = {
            **launch,
            "duration_ns": ns(row["Task Duration(us)"]),
            "duration_us_raw": row["Task Duration(us)"],
            "current_freq_mhz": int(row["Current Freq"]),
            "rated_freq_mhz": int(row["Rated Freq"]),
            "basic_info": {"path": str(basics[0]), "sha256": digest(basics[0])},
        }
        if pipe_metrics:
            sample.update(
                aiv_time_ns=ns(vector[0]["aiv_time(us)"]),
                aiv_vec_time_ns=ns(vector[0]["aiv_vec_time(us)"]),
                aiv_mte2_time_ns=ns(vector[0]["aiv_mte2_time(us)"]),
                aiv_mte3_time_ns=ns(vector[0]["aiv_mte3_time(us)"]),
                pipe_utilization={"path": str(pipes[0]), "sha256": digest(pipes[0])},
            )
        result.append(sample)
    clocks = {(sample["current_freq_mhz"], sample["rated_freq_mhz"]) for sample in result}
    if len(clocks) != 1 or any(current != rated for current, rated in clocks):
        raise ValueError("Profiler records disagree on current/rated frequency")
    return result


def fit_effective_cost(samples, *, bootstrap=1000):
    """A robust count slope; retain confidence failures instead of inventing unit costs."""
    counts = tuple(sorted({sample["count"] for sample in samples}))
    if len(counts) != 5 or counts[0] != 0:
        raise ValueError("A fit requires five operation counts including zero")
    groups = {count: [] for count in counts}
    rounds = {}
    for sample in samples:
        if sample["phase"] == "measured":
            groups[sample["count"]].append(sample["duration_ns"])
            rounds.setdefault(sample["round"], {count: [] for count in counts})[sample["count"]].append(sample["duration_ns"])
    if any(not values for values in groups.values()):
        raise ValueError("Missing measured samples for a required operation count")

    def slope(medians):
        return statistics.median(
            (medians[right] - medians[left]) / (right - left) for position, left in enumerate(counts) for right in counts[position + 1 :]
        )

    medians = {count: statistics.median(values) for count, values in groups.items()}
    estimate = slope(medians)
    intercept = statistics.median(medians[count] - estimate * count for count in counts)
    residual = max(abs(medians[count] - (intercept + estimate * count)) for count in counts)
    generator = random.Random(0)
    distribution = []
    for _ in range(bootstrap):
        pooled = {count: [] for count in counts}
        for round_index in generator.choices(sorted(rounds), k=len(rounds)):
            for count, values in rounds[round_index].items():
                pooled[count].extend(generator.choices(values, k=len(values)))
        resampled = {count: statistics.median(values) for count, values in pooled.items()}
        distribution.append(slope(resampled))
    distribution.sort()
    low = distribution[int(0.025 * (bootstrap - 1))]
    high = distribution[int(0.975 * (bootstrap - 1))]
    relative_half_width = (high - low) / (2 * abs(estimate)) if estimate else math.inf
    relative_residual = residual / abs(estimate * (counts[-1] - counts[0])) if estimate else math.inf
    reasons = []
    if low <= 0:
        reasons.append("slope confidence interval includes zero")
    if relative_half_width > 0.20:
        reasons.append("relative confidence half-width exceeds 20%")
    if relative_residual > 0.10:
        reasons.append("relative residual exceeds 10% of count-dependent span")
    return {
        "status": "BLOCKED" if reasons else "PASS",
        "reasons": reasons,
        "unit": "ns_per_operation",
        "method": "Theil-Sen count-median slope; seeded hierarchical round/sample bootstrap",
        "bootstrap_resamples": bootstrap,
        "slope": format(estimate, ".12g"),
        "intercept_ns": format(intercept, ".12g"),
        "ci95_low": format(low, ".12g"),
        "ci95_high": format(high, ".12g"),
        "conservative_effective_ns": math.ceil(high),
        "relative_confidence_half_width": format(relative_half_width, ".12g"),
        "relative_residual": format(relative_residual, ".12g"),
        "count_median_ns": {str(count): format(value, ".12g") for count, value in medians.items()},
        "round_slopes": [
            format(slope({count: statistics.median(values) for count, values in group.items()}), ".12g")
            for _, group in sorted(rounds.items())
        ],
        "measured_sample_count": sum(len(values) for values in groups.values()),
    }


def assemble_profile(cases, *, quantum_ns=1):
    """Build the full canonical map only after every measured geometry/pattern passed."""
    if type(quantum_ns) is not int or quantum_ns < 1:
        raise ValueError("The scheduling quantum must be a positive integer nanosecond count")
    if len(cases) != 32:
        raise ValueError("A complete initial model requires all 32 operation/pattern cases")
    indexed = {}
    identity = None
    source_identity = None
    library_identity = None
    raw = []
    fits = []
    for case in cases:
        worker, samples, fit = case["worker"], case["samples"], case["fit"]
        parameters = worker["parameters"]
        if worker["status"] != "PASS" or fit["status"] != "PASS":
            raise ValueError("Calibration accuracy and statistical gates must both pass")
        key = (
            parameters["operation"],
            parameters["dtype"],
            parameters["tile"],
            parameters["pattern"],
        )
        if key in indexed:
            raise ValueError("Duplicate calibration geometry/pattern")
        indexed[key] = case
        toolchain = worker["toolchain"]
        version = next(line.removeprefix("Version=") for line in toolchain["cann_version_info"].splitlines() if line.startswith("Version="))
        actual = {
            "platform": "A3",
            "soc": worker["actual_device"]["soc"],
            "cann_version": version,
            "bisheng_sha256": toolchain["sha256"],
            "flags": worker["kernel"]["resolved_flags"],
            "current_freq_mhz": samples[0]["current_freq_mhz"],
            "rated_freq_mhz": samples[0]["rated_freq_mhz"],
        }
        current_source = worker["native_build"]["project_source"]
        current_libraries = sorted((Path(entry["path"]).name, entry["sha256"]) for entry in worker["native_build"]["libraries"])
        if identity is not None and (identity != actual or source_identity != current_source or library_identity != current_libraries):
            raise ValueError("Calibration cases use different actual hardware/compiler/native identities")
        identity, source_identity, library_identity = actual, current_source, current_libraries
        if worker.get("count_values") != list(COUNTS) or worker.get("independent_batch_tiles") != 128:
            raise ValueError("Calibration cases do not use the common bounded-batch count policy")
        expected_policy = {"warmup": 5, "samples": 20, "rounds": 3}
        if any(parameters.get(key) != value for key, value in expected_policy.items()):
            raise ValueError("Calibration cases use different sampling policies")
        attempts = case.get("attempts", [{"worker_report": case["worker_report"]}])
        if not 1 <= len(attempts) <= 2 or len(samples) != 375 * len(attempts):
            raise ValueError("Calibration requires every sample from at most two complete attempts")
        name = "/".join(map(str, key))
        raw.append(
            {
                "case": name,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "samples": [
                    {
                        "attempt": sample.get("attempt", 0),
                        "round_within_attempt": sample.get("round_within_attempt", sample["round"]),
                        **{
                            key: sample[key]
                            for key in (
                                "index",
                                "round",
                                "count",
                                "phase",
                                "duration_ns",
                                "duration_us_raw",
                            )
                        },
                    }
                    for sample in samples
                ],
                "worker_report": case["worker_report"],
                "worker_reports": case.get("worker_reports", [case["worker_report"]]),
                "accuracy_report": case["accuracy_report"],
                "collector_command": case["command"],
                "collector_commands": case.get("collector_commands", [case["command"]]),
                "source_sha256": worker["kernel"]["source"]["sha256"],
                "compiler_source_sha256": worker["compilations"][0]["source"]["sha256"],
                "harness_source_sha256": worker["harness_source"]["sha256"],
                "fixture_source_sha256": worker["fixture_source"]["sha256"],
            }
        )
        fits.append({"case": name, "attempt_count": len(attempts), **fit})
    records = []
    kind_names = {"gmub": "gm_to_ub", "ubgm": "ub_to_gm", "fill": "fill", "add": "add"}
    for operation, kind in kind_names.items():
        for dtype in ("float16", "float32"):
            for tile in (32, 64):
                dependent = indexed[(operation, dtype, tile, "dependent")]["fit"]
                independent = indexed[(operation, dtype, tile, "independent")]["fit"]
                latency, ii = (
                    dependent["conservative_effective_ns"],
                    independent["conservative_effective_ns"],
                )
                if min(latency, ii) < 1:
                    raise ValueError("A nonpositive measured cost cannot be replaced by unit cost")
                records.append(
                    {
                        "kind": kind,
                        "dtype": dtype,
                        "elements": tile,
                        "bytes": tile * (2 if dtype == "float16" else 4),
                        "rows": 1,
                        "latency_effective_ns": latency,
                        "ii_effective_ns": ii,
                        "latency": (latency + quantum_ns - 1) // quantum_ns,
                        "ii": (ii + quantum_ns - 1) // quantum_ns,
                    }
                )
    profile = {
        "schema_version": 1,
        "model": "a3-effective-op-v1",
        "unit": "ns",
        "quantum_ns": quantum_ns,
        "identity": identity,
        "records": records,
        "evidence": {
            "status": "PASS",
            "method": "msprof op BasicInfo-only; Theil-Sen count-median slope and hierarchical round/sample bootstrap",
            "metrics": "BasicInfo",
            "replay_mode": "kernel",
            "cache_policy": "warm input data; cold real compilation; no timed cache flush",
            "count_values": list(COUNTS),
            "independent_batch_tiles": 128,
            "warmup_per_count": 5,
            "samples_per_count": 20,
            "rounds_per_attempt": 3,
            "maximum_attempts": 2,
            "attempt_merge_policy": "Include every round/sample from all attempts; accept only the combined fit. Never discard a failed batch.",
            "application_launches": sum(len(case["samples"]) for case in cases),
            "measured_samples": sum(sum(sample["phase"] == "measured" for sample in case["samples"]) for case in cases),
            "native_source": source_identity,
            "native_libraries": [{"name": name, "sha256": sha} for name, sha in library_identity],
            "scope": "Effective issue/loop/synchronization cost, including bounded-batch drains; no pure instruction latency or safety guarantee.",
            "raw_samples": raw,
            "fits": fits,
        },
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    profile["profile_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return profile


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--platform", choices=("a3",), default="a3")
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--operation", choices=("gmub", "ubgm", "fill", "add"), default="add")
    result.add_argument("--pattern", choices=("dependent", "independent"), required=True)
    result.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    result.add_argument("--tile", type=int, choices=(32, 64), default=64)
    result.add_argument("--mode", choices=("accuracy", "profile"), default="accuracy")
    result.add_argument("--accuracy-repeat", type=int, default=3)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--samples", type=int, default=20)
    result.add_argument("--rounds", type=int, default=3)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--build-manifest", type=Path, required=True)
    return result


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        json.dump(report, stream, indent=2, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)


def compile_kernel(tilelang, program, args, report, artifacts, *, configs=None):
    from tilelang.jit.adapter import libgen
    from test_s5_kernels import calibration_pass_configs

    real_run = subprocess.run
    compilations = report["compilations"] = []

    def observed_run(command, *positional, **keywords):
        observed = bool(command) and Path(command[0]).name == "bisheng"
        record = None
        if observed:
            record = {"argv": list(map(str, command)), "status": "RUNNING"}
            compilations.append(record)
            for part in command:
                path = Path(str(part))
                if path.suffix == ".cpp" and path.is_file():
                    copied = artifacts / "compiler.cpp"
                    shutil.copyfile(path, copied)
                    record["source"] = {"path": str(copied), "sha256": digest(copied)}
            save(report, args.output)
        started = time.monotonic()
        try:
            result = real_run(command, *positional, **keywords)
            if record is not None:
                record.update(returncode=result.returncode, status="PASS" if not result.returncode else "FAIL")
                if not result.returncode and "-o" in command:
                    copied = artifacts / "npu_kernel.so"
                    shutil.copyfile(command[command.index("-o") + 1], copied)
                    record["artifact"] = {"path": str(copied), "sha256": digest(copied)}
            return result
        finally:
            if record is not None:
                record["elapsed_seconds"] = time.monotonic() - started
                save(report, args.output)

    libgen.subprocess.run = observed_run
    try:
        kernel = tilelang.compile(
            program,
            out_idx=[],
            target="ascendc",
            platform="A3",
            pass_configs=calibration_pass_configs() if configs is None else configs,
        )
    finally:
        libgen.subprocess.run = real_run
    if len(compilations) != 1 or compilations[0]["status"] != "PASS":
        raise AssertionError("Calibration must invoke real Bisheng exactly once")
    source = artifacts / "kernel.cpp"
    source.write_text(kernel.get_kernel_source())
    report["kernel"] = {
        "resolved_pass_configs": dict(kernel.pass_configs),
        "resolved_flags": kernel.compile_flags,
        "source": {"path": str(source), "sha256": digest(source)},
        "kernel_id": str(id(kernel)),
    }
    return kernel


def expected_output(args, a, b, count, torch):
    if args.operation == "add":
        if args.pattern == "dependent":
            expected = a.clone()
            for _ in range(count):
                expected[0] += b[0]
        else:
            expected = torch.zeros_like(a)
            extent = min(count, 128)
            expected[:extent] = a[:extent] + b[:extent]
    elif args.operation == "fill":
        expected = a.clone() if args.pattern == "dependent" else torch.zeros_like(a)
        if count:
            expected[: 1 if args.pattern == "dependent" else min(count, 128)].fill_(1)
    elif args.operation == "gmub":
        expected = torch.zeros_like(a)
        if count:
            extent = 1 if args.pattern == "dependent" else min(count, 128)
            expected[:extent] = a[:extent]
    else:
        expected = torch.full_like(a, float("nan"))
        if count:
            extent = 1 if args.pattern == "dependent" else min(count, 128)
            expected[:extent] = a[:extent]
    return expected


def run_worker(args, report, artifacts):
    tilelang, torch = prepare_worker(args, report)
    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s5_kernels import make_calibration_kernel

    kernel = compile_kernel(
        tilelang,
        make_calibration_kernel(args.operation, args.pattern, args.dtype, args.tile),
        args,
        report,
        artifacts,
    )
    dtype = getattr(torch, args.dtype)
    indices = torch.arange(128 * args.tile).reshape(128, args.tile)
    a_host = ((indices % 16).float() / 128 + 0.125).to(dtype)
    b_host = ((indices % 8 + 1).float() / 1024).to(dtype)
    a, b = a_host.npu(), b_host.npu()
    output = torch.empty_like(a)
    counts = {count: torch.empty((count,), dtype=torch.int32, device="npu") for count in COUNTS}
    report["launches"] = []
    report["golden_checks"] = 0
    report["cache_policy"] = "warm input data; cold real compilation; no timed cache flush"
    report["measurement_scope"] = (
        "Target kernel duration contains fixed setup/drain plus operation count-dependent "
        "issue/loop/synchronization overhead. No pure instruction latency claim."
    )
    rounds = args.rounds if args.mode == "profile" else 1
    for round_index in range(rounds):
        for count in COUNTS:
            expected = expected_output(args, a_host, b_host, count, torch)
            output.fill_(float("nan"))
            torch.npu.synchronize()
            phases = (
                (("warmup", args.warmup), ("measured", args.samples)) if args.mode == "profile" else (("accuracy", args.accuracy_repeat),)
            )
            for phase, repeat in phases:
                for _ in range(repeat):
                    if args.mode == "accuracy":
                        output.fill_(float("nan"))
                        torch.npu.synchronize()
                    kernel(a, b, output, counts[count])
                    torch.npu.synchronize()
                    report["launches"].append(
                        {
                            "index": len(report["launches"]),
                            "round": round_index,
                            "count": count,
                            "phase": phase,
                        }
                    )
                    if args.mode == "accuracy":
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
                        report["golden_checks"] += 1
            if args.mode == "profile":
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
                report["golden_checks"] += 1
            save(report, args.output)
    report["counts"] = {
        "passed": report["golden_checks"],
        "failed": 0,
        "skipped": 0,
        "total": report["golden_checks"],
    }
    report["profile_status"] = "NOT_ESTIMATED"


def main():
    args = parser().parse_args()
    report = base_report("s5_calibration_worker", args, step="S5")
    report["harness_source"] = {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)}
    fixture = ROOT / "testing/python/ascend_migration/test_s5_kernels.py"
    report["fixture_source"] = {"path": str(fixture), "sha256": digest(fixture)}
    report["parameters"] = vars(args)
    report["count_values"] = list(COUNTS)
    report["independent_batch_tiles"] = 128
    if sys.flags.optimize or min(args.accuracy_repeat, args.samples, args.rounds) < 1 or args.warmup < 0:
        return finish(args, report, "BLOCKED", "Assertions and positive sampling counts are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=args.output.stem + "-artifacts-", dir=args.output.parent))
    report["artifacts"] = str(artifacts)
    for name, path in (("harness_source", Path(__file__)), ("fixture_source", fixture)):
        copied = artifacts / path.name
        shutil.copyfile(path, copied)
        report[name]["snapshot"] = str(copied)
    os.environ["TILELANG_CACHE_DIR"] = str(artifacts / "cache")
    report["status"] = "BLOCKED"
    save(report, args.output)
    try:
        run_worker(args, report, artifacts)
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        status = "FAIL" if "kernel" in report else "BLOCKED"
        return finish(args, report, status, str(error))
    return finish(args, report, "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
