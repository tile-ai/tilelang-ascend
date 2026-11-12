"""Run one alignment scope in independent processes; preserve every trial."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def summarize(reports):
    """Keep independent trial medians separate; never pool their raw samples."""
    groups = {}
    for report in reports:
        key = (report["case"], report["dtype"], report["target"])
        groups.setdefault(key, []).append(report)
    rows = []
    for (case, dtype, target), trials in sorted(groups.items()):
        fields = (
            "alignment_scope",
            "aligned_methods",
            "aligned_execution_sha256",
            "candidate_kernels",
            "core_prepared_inputs",
            "shape",
            "visible",
            "qualification_seeds",
            "timed_seed",
            "timed_inputs",
            "timed_metadata",
            "budget",
            "environment",
            "package_sources",
            "gate",
            "adapter_sha256",
            "baseline_sources",
            "protocol_manifest_sha256",
            "paired_timing_sha256",
            "timing_engine_identity",
            "harness_sha256",
            "graph_batch",
            "rounds",
            "repeats_per_round",
            "graph_repeats_per_round",
        )
        identities = []
        for r in trials:
            identity = {k: r.get(k) for k in fields}
            identity["engine_config"] = {k: v for k, v in r.get("measurement", {}).get("config", {}).items() if k != "order_seed"}
            identities.append(json.dumps(identity, sort_keys=True))
        if len(set(identities)) != 1:
            raise ValueError("independent trials have different source/runtime/input/timing identities")
        trial_ids = [r.get("trial_id") for r in trials]
        if any(type(value) is not int for value in trial_ids) or len(set(trial_ids)) != len(trial_ids):
            raise ValueError("independent trials require distinct integer trial_id values")
        for metric in ("eager_event_us", "eager_wall_us", "graph_per_call_us"):
            methods = sorted(set().union(*(r.get("measurement", {}).get("methods", {}) for r in trials)))
            for method in methods:
                medians, p95s, ratios = [], [], []
                boundary_statuses, partial_medians = [], []
                qualified = True
                for r in trials:
                    measurement = r.get("measurement", {})
                    method_data = measurement.get("methods", {}).get(method, {})
                    stats = method_data.get("metrics", {}).get(metric, {})
                    boundary = "graph" if metric == "graph_per_call_us" else "eager"
                    boundary_status = method_data.get(boundary + "_status")
                    boundary_statuses.append(boundary_status)
                    expected_samples = r["rounds"] * r["graph_repeats_per_round" if boundary == "graph" else "repeats_per_round"]
                    complete = boundary_status == "measured" and stats.get("count") == expected_samples
                    if boundary_status == "measured" and not complete:
                        raise ValueError("measured boundary has incomplete sample count")
                    if complete and stats.get("median") is not None:
                        medians.append(stats["median"])
                        p95s.append(stats["p95"])
                    elif stats.get("median") is not None:
                        partial_medians.append(stats["median"])
                    if method != "candidate":
                        pair = measurement.get("paired_ratios", {}).get(metric, {}).get(method, {})
                        ok = r.get("status") == "balanced_benchmark_completed" and pair.get("qualified") is True
                        qualified &= ok
                        if ok:
                            ratios.append(pair["ratio_of_medians"])
                valid_ratio = method != "candidate" and qualified and len(ratios) == len(trials)
                rows.append(
                    {
                        "alignment_scope": trials[0].get("alignment_scope"),
                        "case": case,
                        "dtype": dtype,
                        "target": target,
                        "metric": metric,
                        "method": method,
                        "trials": len(trials),
                        "measured_trials": len(medians),
                        "boundary_status_by_trial": boundary_statuses,
                        "diagnostic_partial_trial_medians_us": partial_medians,
                        "trial_medians_us": medians,
                        "median_of_trial_medians_us": statistics.median(medians) if medians else None,
                        "range_of_trial_medians_us": [min(medians), max(medians)] if medians else None,
                        "trial_p95_us": p95s,
                        "qualified_ratio": valid_ratio,
                        "reference_over_candidate_median": statistics.median(ratios) if valid_ratio else None,
                        "trial_ratios": ratios if valid_ratio else [],
                        "scope": "same case/runtime boundary; independent processes; no raw-sample pooling",
                    }
                )
    return rows


def write_summary(output, reports, state):
    rows = summarize(reports)
    atomic_json(output / "summary.json", {"status": state, "completed_trials": len(reports), "rows": rows})
    text = [
        "# Aligned Gated MLA performance comparison",
        "",
        "Each row summarizes independent process trials. Units: microseconds. Ratios require candidate/reference precision qualification before and after that measurement boundary. Failed comparators have no ratio.",
        "",
        "|Case / dtype / target|Boundary|Method|Measured trials|Median of trial medians|Range|Qualified ref/candidate|",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for r in rows:
        median = "—" if r["median_of_trial_medians_us"] is None else f"{r['median_of_trial_medians_us']:.3f}"
        span = "—" if r["range_of_trial_medians_us"] is None else "–".join(f"{x:.3f}" for x in r["range_of_trial_medians_us"])
        ratio = "—" if not r["qualified_ratio"] else f"{r['reference_over_candidate_median']:.2%}"
        text.append(
            f"|{r['case']} / {r['dtype']} / {r['target']}|{r['metric']}|{r['method']}|{r['measured_trials']}/{r['trials']}|{median}|{span}|{ratio}|"
        )
    (output / "SUMMARY.md").write_text("\n".join(text) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("prepared-full-chain", "shared-lp-core"), required=True)
    p.add_argument("--installed-root", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=("decode4096", "decode17", "prefill128"), default=["decode4096", "decode17", "prefill128"])
    p.add_argument("--dtypes", nargs="+", choices=("float16", "bfloat16"), default=["float16", "bfloat16"])
    p.add_argument("--targets", nargs="+", choices=("pto", "ascendc"), default=["pto", "ascendc"])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--graph-repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trial-timeout", type=int, default=900)
    p.add_argument("--time-unqualified", action="store_true")
    p.add_argument(
        "--torch-rounding-gate", action="store_true", help="Explicitly use the unfused diagnostic gate rather than the original Triton gate"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if min(args.trials, args.trial_timeout) < 1:
        p.error("counts/timeouts must be positive")
    if any(len(set(x)) != len(x) for x in (args.cases, args.dtypes, args.targets)):
        p.error("duplicate cases/dtypes/targets are not independent repetitions")
    if args.output.exists():
        p.error("output must be new")
    args.output.mkdir(parents=True)
    plan = []
    for trial, case, dtype, target in itertools.product(range(args.trials), args.cases, args.dtypes, args.targets):
        destination = args.output / f"{case}_{dtype}_{target}_trial{trial}"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("aligned_benchmark.py")),
            "--scope",
            args.scope,
            "--case",
            case,
            "--dtype",
            dtype,
            "--target",
            target,
            "--trial-id",
            str(trial),
            "--order-seed",
            str(1700 + trial),
            "--rounds",
            str(args.rounds),
            "--repeats",
            str(args.repeats),
            "--graph-repeats",
            str(args.graph_repeats),
            "--warmup",
            str(args.warmup),
            "--benchmark-root",
            str(args.benchmark_root),
            "--installed-root",
            str(args.installed_root),
            "--inventory",
            str(args.inventory),
            "--output",
            str(destination),
        ]
        if not args.torch_rounding_gate:
            cmd += ["--gate-module", "source_gate.isolated_gate", "--gate-label", "vllm_e4d533d_original_gate_direct_launcher"]
        if args.time_unqualified:
            cmd += ["--time-unqualified"]
        plan.append(
            {
                "case": case,
                "dtype": dtype,
                "target": target,
                "trial": trial,
                "output": str(destination),
                "command": cmd,
                "shell_command": shlex.join(cmd),
            }
        )
    state = {
        "alignment_scope": args.scope,
        "status": "plan_only" if args.dry_run else "running",
        "planned_trials": len(plan),
        "independent_processes": True,
        "concurrent_NPU_jobs": 1,
        "time_boundary": "child measurements only; compilation/qualification before timers; process timeout is not latency",
        "plan": plan,
        "completed": [],
    }
    atomic_json(args.output / "suite.json", state)
    if args.dry_run:
        print(json.dumps({"status": "plan_only", "trials": len(plan)}))
        return
    reports = []
    for item in plan:
        started = time.perf_counter()
        with (args.output / (Path(item["output"]).name + ".log")).open("w") as log:
            kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            process = subprocess.Popen(item["command"], stdout=log, stderr=subprocess.STDOUT, **kwargs)
            try:
                code = process.wait(timeout=args.trial_timeout)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=10)
                code = 124
        result = Path(item["output"]) / "aligned_benchmark.json"
        report = json.loads(result.read_text()) if result.exists() else {}
        state["completed"].append(
            {
                "output": item["output"],
                "returncode": code,
                "report_status": report.get("status"),
                "process_wall_seconds_not_operator_latency": time.perf_counter() - started,
            }
        )
        if code != 0 or report.get("status") != "balanced_benchmark_completed":
            state["status"] = "failed"
            atomic_json(args.output / "suite.json", state)
            write_summary(args.output, reports, "partial_failed")
            raise SystemExit("trial failed; inspect preserved log/report; no later NPU trial was started")
        reports.append(report)
        atomic_json(args.output / "suite.json", state)
        try:
            write_summary(args.output, reports, "partial_running")
        except Exception as error:
            state.update(status="failed_summary_integrity", error=repr(error))
            atomic_json(args.output / "suite.json", state)
            raise
    state["status"] = "completed"
    atomic_json(args.output / "suite.json", state)
    write_summary(args.output, reports, "completed")
    print(json.dumps({"status": "completed", "trials": len(reports)}))


if __name__ == "__main__":
    main()
