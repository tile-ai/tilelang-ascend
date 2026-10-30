"""Fixed-policy Event timing and cold compilation; runtime imports are worker-only."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path
import random
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback

from _common import base_report, finish
from _npu_runner import digest, prepare_worker
from _s8_npu import (
    BENCHMARK_BASES,
    BENCHMARK_WARMUPS,
    COMPILE_FLAGS,
    PARAMETERS,
    CaseData,
    build_kernel,
    compiler_options,
    describe_kernel,
    preflight,
    reference_role,
)


def event_span_ns(milliseconds):
    if isinstance(milliseconds, bool):
        raise ValueError("A real NPU Event span cannot be boolean")
    try:
        value = Decimal(str(milliseconds))
    except InvalidOperation as error:
        raise ValueError("A real NPU Event span must be numeric") from error
    if not value.is_finite() or value <= 0:
        raise ValueError("A real NPU Event span must be finite and positive")
    return int((value * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def sample_summary(values):
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Timing samples must be positive integer nanoseconds")
    generator = random.Random(0)
    distribution = sorted(statistics.median(generator.choices(values, k=len(values))) for _ in range(1000))
    return {
        "samples": len(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "median_ns": format(statistics.median(values), ".12g"),
        "median_ci95_low_ns": format(distribution[24], ".12g"),
        "median_ci95_high_ns": format(distribution[974], ".12g"),
        "median_ci95_method": "1000 bootstrap median resamples, seed 0, percentile indices 24/974",
    }


def round_phases():
    return (*("warmup" for _ in range(BENCHMARK_WARMUPS)), "measured")


def _save(report, output):
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        path = Path(stream.name)
        json.dump(report, stream, indent=2, default=str)
        stream.write("\n")
    path.replace(output)


def _compile(tilelang, spec, args, report, artifacts):
    """Only in-memory observations occur inside the compile timing interval."""
    from tilelang.jit.adapter import libgen

    real_run = subprocess.run
    records = report["compilations"]

    def observed(command, *positional, **keywords):
        is_bisheng = bool(command) and Path(command[0]).name == "bisheng"
        record = None
        if is_bisheng:
            record = {"argv": list(map(str, command)), "status": "RUNNING"}
            records.append(record)
        started = time.perf_counter_ns()
        try:
            result = real_run(command, *positional, **keywords)
            if record is not None:
                record.update(
                    returncode=result.returncode,
                    status="PASS" if result.returncode == 0 else "FAIL",
                )
            return result
        finally:
            if record is not None:
                record["elapsed_ns"] = time.perf_counter_ns() - started

    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
    libgen.subprocess.run = observed
    try:
        kernel = build_kernel(tilelang, spec)
    finally:
        cpu_finished, finished = time.process_time_ns(), time.perf_counter_ns()
        libgen.subprocess.run = real_run
    after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    if len(records) != 1 or records[0]["status"] != "PASS":
        raise AssertionError("Each benchmark worker must invoke real Bisheng once")
    report["compile_timing"] = {
        "wall_ns": finished - started,
        "process_cpu_ns": cpu_finished - cpu_started,
        "reaped_child_cpu_ns": round(
            1e9 * (after_children.ru_utime + after_children.ru_stime - before_children.ru_utime - before_children.ru_stime)
        ),
        "bisheng_wall_ns": records[0]["elapsed_ns"],
        "note": "Frontend/TVM/Bisheng/adapter compile wall; no evidence serialization or data setup",
    }
    # Copy/hash artifacts only after the compile clock has stopped.
    record = records[0]
    record["compiler_realpath"] = str(Path(shutil.which(record["argv"][0])).resolve())
    inputs = [Path(item) for item in record["argv"] if item.endswith(".cpp")]
    if len(inputs) != 1 or not inputs[0].is_file():
        raise AssertionError("The exact real compiler input must remain available for evidence")
    source, library = artifacts / "bisheng-input.cpp", artifacts / "npu-kernel.so"
    shutil.copyfile(inputs[0], source)
    shutil.copyfile(record["argv"][record["argv"].index("-o") + 1], library)
    record["source"] = {"path": str(source), "sha256": digest(source)}
    record["artifact"] = {"path": str(library), "sha256": digest(library)}
    report["kernels"].append(
        {
            "kernel_id": str(id(kernel)),
            "resolved_flags": list(kernel.compile_flags),
            "resolved_pass_configs": dict(kernel.pass_configs),
            "requested_compile_flags": list(COMPILE_FLAGS),
        }
    )
    return kernel


def run_measured_worker(args):
    report = base_report("s8_benchmark_worker", args, step="S8")
    report.update(
        status="BLOCKED",
        case=args._case,
        compilations=[],
        kernels=[],
        launches={},
        golden_checks=0,
        journal=[],
        samples=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=args._case + "-timing-", dir=args.output.parent))
    report["artifacts"] = str(artifacts)
    _save(report, args.output)
    stage = "preflight"
    try:
        tilelang, torch = prepare_worker(args, report)
        preflight(args, report, tilelang)
        spec = PARAMETERS[args._case]
        stage = "compile"
        kernel = _compile(tilelang, spec, args, report, artifacts)
        metadata = report["case_parameters"] = {**spec, **describe_kernel(kernel, args, spec)}
        metadata["reference_role"] = reference_role(spec)
        state = CaseData(spec, args, torch)
        state.prepare(0)
        metadata["input_sha256"] = dict(state.input_hashes)
        metadata["golden_sha256"] = state.golden_hash
        metadata["buffer_pointers"] = state.buffer_pointers
        stream = torch.npu.Stream(device=args.device)
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record(stream)
        end.record(stream)
        stream.synchronize()
        report["timing_identity"] = {
            "stream_handle": int(stream.npu_stream),
            "event_handles": [int(start.npu_event), int(end.npu_event)],
            "event_unit": "elapsed_time milliseconds, converted to integer nanoseconds",
            "interpretation": "NPU stream Event span, including dispatch gaps; not pure instruction latency",
        }
        counts = report["launches"][str(id(kernel))] = {"attempted": 0, "completed": 0}
        report["sampling_policy"] = {
            "rounds": args.repeat,
            "warmups_per_round": BENCHMARK_WARMUPS,
            "measured_per_round": 1,
            "cache": "No explicit cache flush; identical input/workspace copies before every call",
            "reset": "Restore inputs/workspace and NaN output, then synchronize outside timings",
            "cold_compilation": "separate fresh process and disabled kernel caches",
            "golden": "Every warmup and measured call, outside timing; warmups precede each sample",
        }
        stage = "execute"
        with torch.npu.stream(stream):
            for round_index in range(args.repeat):
                for phase in round_phases():
                    state.restore()
                    if phase == "measured":
                        start.record(stream)
                    counts["attempted"] += 1
                    host_started = time.perf_counter_ns()
                    kernel(*state.arguments)
                    host_finished = time.perf_counter_ns()
                    counts["completed"] += 1
                    if phase == "measured":
                        end.record(stream)
                        end.synchronize()
                        raw_ms = start.elapsed_time(end)
                        span = event_span_ns(raw_ms)
                    else:
                        stream.synchronize()
                    checked = state.check()
                    report["golden_checks"] += 1
                    entry = {
                        "index": len(report["journal"]),
                        "round": round_index,
                        "phase": phase,
                        **checked,
                    }
                    report["journal"].append(entry)
                    if phase == "measured":
                        host_ns = host_finished - host_started
                        if host_ns <= 0:
                            raise ValueError("Host dispatch clock did not produce a positive interval")
                        report["samples"].append(
                            {
                                "index": len(report["samples"]),
                                "round": round_index,
                                "host_dispatch_ns": host_ns,
                                "event_span_ms_raw": repr(raw_ms),
                                "event_span_ns": span,
                                **checked,
                            }
                        )
                _save(report, args.output)
        expected = args.repeat * (BENCHMARK_WARMUPS + 1)
        if counts != {"attempted": expected, "completed": expected} or report["golden_checks"] != expected:
            raise AssertionError("Benchmark launch/golden accounting is incomplete")
        if len(report["journal"]) != expected or len(report["samples"]) != args.repeat:
            raise AssertionError("Benchmark sample coverage differs from the fixed policy")
        report["summary"] = {
            metric: sample_summary([row[metric] for row in report["samples"]]) for metric in ("host_dispatch_ns", "event_span_ns")
        }
    except Exception as error:
        report["traceback"] = traceback.format_exc()
        return finish(args, report, "BLOCKED" if stage == "preflight" else "FAIL", str(error))
    return finish(args, report, "PASS")


def compare_performance(report):
    results = {row["name"]: row["evidence"] for row in report["cases"]}
    comparisons = report["performance_comparisons"] = []
    for base in BENCHMARK_BASES:
        left, right = [results[f"{base}_{strategy}"] for strategy in ("legacy", "auto")]
        expected = report["repeat"] * (BENCHMARK_WARMUPS + 1)
        for worker in (left, right):
            if (
                worker["status"] != "PASS"
                or len(worker["compilations"]) != 1
                or worker["compilations"][0]["status"] != "PASS"
                or len(worker["kernels"]) != 1
                or len(worker["launches"]) != 1
                or next(iter(worker["launches"].values())) != {"attempted": expected, "completed": expected}
                or worker["golden_checks"] != expected
                or len(worker["journal"]) != expected
                or len(worker["samples"]) != report["repeat"]
                or worker["sampling_policy"]["rounds"] != report["repeat"]
                or worker["sampling_policy"]["warmups_per_round"] != BENCHMARK_WARMUPS
                or worker["sampling_policy"]["measured_per_round"] != 1
            ):
                raise AssertionError("Incomplete benchmark compile/launch/golden/sample accounting")
            for index, row in enumerate(worker["journal"]):
                if (row["index"], row["round"], row["phase"], row["golden_status"]) != (
                    index,
                    index // (BENCHMARK_WARMUPS + 1),
                    round_phases()[index % (BENCHMARK_WARMUPS + 1)],
                    "PASS",
                ):
                    raise AssertionError("Benchmark journal does not establish checked warmups before samples")
            for index, sample in enumerate(worker["samples"]):
                if (sample["index"], sample["round"], sample["golden_status"]) != (
                    index,
                    index,
                    "PASS",
                ):
                    raise AssertionError("Missing ordered measured sample/golden")
                if sample["host_dispatch_ns"] <= 0 or event_span_ns(sample["event_span_ms_raw"]) != sample["event_span_ns"]:
                    raise AssertionError("Invalid actual host/Event timing samples")
        if left["sampling_policy"] != right["sampling_policy"]:
            raise AssertionError("Paired timing policies differ")
        if left["kernels"][0]["resolved_flags"] != right["kernels"][0]["resolved_flags"]:
            raise AssertionError("Paired benchmark compiler flags differ")
        if compiler_options(left["compilations"][0]) != compiler_options(right["compilations"][0]):
            raise AssertionError("Paired benchmark actual Bisheng options differ")
        if left["toolchain"] != right["toolchain"] or left["actual_device"] != right["actual_device"]:
            raise AssertionError("Paired benchmark toolchain/device identity differs")
        if len(left["journal"]) != len(right["journal"]):
            raise AssertionError("Paired launch counts differ")
        fields = (
            "index",
            "round",
            "phase",
            "input_sha256",
            "golden_sha256",
            "output_sha256",
            "workspace_output_sha256",
            "workspace_golden_sha256",
        )
        if any(
            {key: a.get(key) for key in fields} != {key: b.get(key) for key in fields} for a, b in zip(left["journal"], right["journal"])
        ):
            raise AssertionError("Paired benchmark inputs/output/golden/workspace differ")
        change = {}
        for metric in ("host_dispatch_ns", "event_span_ns"):
            a = Decimal(left["summary"][metric]["median_ns"])
            b = Decimal(right["summary"][metric]["median_ns"])
            change[metric] = {
                "legacy": left["summary"][metric],
                "auto": right["summary"][metric],
                "auto_change_percent": str((b / a - 1) * 100),
            }
        comparisons.append(
            {
                "fixture": base,
                "status": "PASS",
                "measurements": change,
                "compile": {"legacy": left["compile_timing"], "auto": right["compile_timing"]},
                "interpretation": "Observed matched-input times; positive change is regression, no speedup threshold",
            }
        )
