"""Balanced timing protocol, separate from the historical frozen protocol.

``backend`` has the torch.npu Event/synchronize/NPUGraph/graph interface. Torch
is imported only when a real run requests the default backend. ``check`` runs
outside timing and must raise for readonly/shape violations; it returns a bool
or a dict with a bool ``qualified``/``passed`` for numerical qualification.
``on_progress`` may persist the report but must not mutate it.
"""

import itertools
import math
import random
import statistics
import time


class MeasurementFailure(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.partial_report = report


def summarize(samples):
    values = list(samples)
    if not values:
        return dict(count=0, raw_samples=[], minimum=None, median=None, mean=None, p90=None, p95=None, stdev=None)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("samples must be finite and nonnegative")
    ordered = sorted(values)
    return dict(
        count=len(values),
        raw_samples=values,
        minimum=ordered[0],
        median=statistics.median(values),
        mean=statistics.mean(values),
        p90=ordered[math.ceil(0.90 * len(values)) - 1],
        p95=ordered[math.ceil(0.95 * len(values)) - 1],
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def balanced_orders(names, rounds, order_seed):
    names = tuple(names)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("this protocol requires exactly three distinct method names")
    if type(rounds) is not int or rounds < 6 or rounds % 6:
        raise ValueError("rounds must be a positive multiple of six")
    if type(order_seed) is not int:
        raise ValueError("order_seed must be an explicit integer")
    rng = random.Random(order_seed)
    result = []
    for _ in range(rounds // 6):
        cycle = list(itertools.permutations(names))
        rng.shuffle(cycle)
        result.extend(cycle)
    return result


def _qualification(result):
    if type(result) is bool:
        return result
    if isinstance(result, dict):
        found = [result[key] for key in ("qualified", "passed") if key in result]
        if found and all(type(value) is bool for value in found) and len(set(found)) == 1:
            if result.get("read_only") is False or result.get("readonly") is False:
                raise AssertionError("readonly check failed")
            return found[0]
    raise TypeError("check must return bool or an unambiguous bool qualified/passed record")


def run_measurements(
    methods,
    *,
    qualified,
    check,
    warmup=10,
    rounds=6,
    repeats=10,
    graph_batch=50,
    graph_repeats=5,
    order_seed=0,
    time_unqualified=False,
    backend=None,
    clock=None,
    graph_error_policy=None,
    on_progress=None,
):
    """Run one measurement trial; caller owns independent-process repetition.

    Each round has ``repeats`` eager samples or ``graph_repeats`` replay samples
    per selected method, in that round's balanced order. Graphs are captured once
    per method, then warmed five times. Graph samples are Event microseconds per
    captured call. Eager wall includes end-event synchronization. Neither metric
    is hardware kernel Task time. Inputs/weights are reused; no L2 flush is added.

    Qualified methods failing a post-round check abort. Initially unqualified
    methods may be timed only with the explicit diagnostic flag and never gain a
    ratio. Unknown graph errors abort; the caller may supply the frozen graph
    error policy to classify known unsupported graph capture/replay operations.
    A failure carries ``partial_report``; completed eager evidence is retained.
    """
    methods = dict(methods)
    names = tuple(methods)
    orders = balanced_orders(names, rounds, order_seed)
    if "candidate" not in methods or any(not callable(fn) for fn in methods.values()):
        raise ValueError("three callables including candidate are required")
    if set(qualified) != set(names) or any(type(value) is not bool for value in qualified.values()):
        raise ValueError("qualified must contain one bool for every method")
    qualified = dict(qualified)  # Initial diagnostic status cannot be promoted by a callback.
    if not callable(check):
        raise TypeError("check is required")
    for name, value in (("warmup", warmup), ("repeats", repeats), ("graph_batch", graph_batch), ("graph_repeats", graph_repeats)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    selected = [name for name in names if qualified[name] or time_unqualified]
    metrics = ("eager_event_us", "eager_wall_us", "graph_per_call_us")
    samples = {name: {metric: [] for metric in metrics} for name in names}
    report = {
        "schema_version": 1,
        "protocol": "balanced_three_method_rounds_v1",
        "status": "running",
        "unit": "microseconds",
        "percentile": "nearest_rank_ceil_p_times_n; sample stdev ddof=1, single sample=0",
        "config": dict(
            warmup=warmup,
            rounds=rounds,
            repeats=repeats,
            graph_batch=graph_batch,
            graph_repeats=graph_repeats,
            graph_warmup_replays=5,
            order_seed=order_seed,
            time_unqualified=bool(time_unqualified),
        ),
        "order_plan": [list(order) for order in orders],
        "scope": "one trial; same bound callables, reused inputs/weights, no L2 flush; not kernel Task time",
        "methods": {
            name: {
                "initial_qualified": qualified[name],
                "diagnostic_only": not qualified[name],
                "eager_status": "pending" if name in selected else "not_timed_accuracy_failed",
                "graph_status": "pending" if name in selected else "not_timed_accuracy_failed",
                "checks": {"eager": [], "graph": []},
                "metrics": {metric: summarize([]) for metric in metrics},
            }
            for name in names
        },
        "rounds": {"eager": [], "graph": []},
        "paired_ratios": {metric: {} for metric in metrics},
    }

    def publish():
        for name in names:
            for metric in metrics:
                report["methods"][name]["metrics"][metric] = summarize(samples[name][metric])
        if on_progress is not None:
            on_progress(report)

    def postcheck(name, output, boundary, index):
        result = check(name, output, boundary)
        passed = _qualification(result)
        report["methods"][name]["checks"][boundary].append(dict(round=index, qualified=passed, result=result))
        if qualified[name] and not passed:
            raise AssertionError(f"previously qualified {name} failed post-{boundary} check at round {index}")

    def ratios(boundary, metric):
        valid = {}
        for name in names:
            row = report["methods"][name]
            valid[name] = (
                qualified[name]
                and row[boundary + "_status"] == "measured"
                and len(row["checks"][boundary]) == rounds
                and all(item["qualified"] for item in row["checks"][boundary])
            )
        for name in names:
            if name == "candidate":
                continue
            entry = dict(
                qualified=False,
                unit="dimensionless",
                definition="reference/candidate in the same round and repeat",
                ratio_of_medians=None,
                paired_samples=[],
                summary=summarize([]),
            )
            if valid["candidate"] and valid[name]:
                values = []
                for row in report["rounds"][boundary]:
                    for sample in row["samples"]:
                        pair = sample["methods"]
                        denominator = pair["candidate"][metric]
                        if denominator <= 0:
                            raise ValueError("nonpositive candidate timing cannot produce a ratio")
                        value = pair[name][metric] / denominator
                        values.append(value)
                        entry["paired_samples"].append(dict(round=row["round"], repeat=sample["repeat"], ratio=value))
                entry.update(
                    qualified=True,
                    summary=summarize(values),
                    ratio_of_medians=statistics.median(samples[name][metric]) / statistics.median(samples["candidate"][metric]),
                )
            else:
                entry["reason"] = "requires both initial and post-round qualification plus complete measurement for this metric"
            report["paired_ratios"][metric][name] = entry

    def graph_failure(name, error):
        policy = graph_error_policy(error) if graph_error_policy is not None else {"continue": False, "category": "unclassified_abort"}
        report["methods"][name].update(graph_status="graph_unavailable", graph_error=repr(error), graph_error_policy=policy)
        publish()
        if not isinstance(policy, dict) or policy.get("continue") is not True:
            raise error

    try:
        publish()
        if not selected:
            report["status"] = "completed_no_eligible_methods"
            for metric in metrics:
                ratios("graph" if metric == "graph_per_call_us" else "eager", metric)
            publish()
            return report
        if backend is None:
            import torch
            import torch_npu  # noqa: F401 -- lazy backend registration

            backend = torch.npu
        clock = time.perf_counter if clock is None else clock
        start, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
        for iteration in range(warmup):
            for name in orders[iteration % rounds]:
                if name in selected:
                    methods[name]()
        backend.synchronize()

        for index, order in enumerate(orders):
            actual_order = [name for name in order if name in selected]
            row = dict(round=index, planned_order=list(order), order=actual_order, samples=[])
            report["rounds"]["eager"].append(row)
            outputs = {}
            for repeat in range(repeats):
                sample = dict(repeat=repeat, methods={})
                row["samples"].append(sample)
                for name in actual_order:
                    start.record()
                    before = clock()
                    outputs[name] = methods[name]()
                    end.record()
                    end.synchronize()
                    wall = (clock() - before) * 1e6
                    event = start.elapsed_time(end) * 1000
                    sample["methods"][name] = dict(eager_event_us=event, eager_wall_us=wall)
                    samples[name]["eager_event_us"].append(event)
                    samples[name]["eager_wall_us"].append(wall)
            for name in actual_order:
                postcheck(name, outputs[name], "eager", index)
            publish()
        for name in selected:
            report["methods"][name]["eager_status"] = "measured"
        ratios("eager", "eager_event_us")
        ratios("eager", "eager_wall_us")
        publish()  # Completed eager data and qualification survive graph failure.

        graphs, captured_outputs = {}, {}
        for name in [name for name in orders[0] if name in selected]:
            try:
                graph = backend.NPUGraph()
                with backend.graph(graph):
                    for _ in range(graph_batch):
                        captured_outputs[name] = methods[name]()
                graphs[name] = graph
                report["methods"][name]["graph_status"] = "captured"
            except Exception as error:
                graph_failure(name, error)
        for iteration in range(5):
            for name in orders[iteration % rounds]:
                if name in graphs:
                    try:
                        graphs[name].replay()
                    except Exception as error:
                        graph_failure(name, error)
                        del graphs[name]
        backend.synchronize()
        publish()

        for index, order in enumerate(orders):
            actual_order = [name for name in order if name in graphs]
            row = dict(round=index, planned_order=list(order), order=actual_order, samples=[])
            report["rounds"]["graph"].append(row)
            for repeat in range(graph_repeats):
                sample = dict(repeat=repeat, methods={})
                row["samples"].append(sample)
                for name in actual_order:
                    if name not in graphs:
                        continue
                    try:
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        value = start.elapsed_time(end) * 1000 / graph_batch
                    except Exception as error:
                        graph_failure(name, error)
                        del graphs[name]
                        continue
                    sample["methods"][name] = dict(graph_per_call_us=value)
                    samples[name]["graph_per_call_us"].append(value)
            for name in actual_order:
                if name in graphs:
                    postcheck(name, captured_outputs[name], "graph", index)
            publish()
        for name in graphs:
            report["methods"][name]["graph_status"] = "measured"
        ratios("graph", "graph_per_call_us")
        report["status"] = "completed_with_explicit_qualification"
        publish()
        return report
    except Exception as error:
        report.update(status="failed", error=repr(error), error_type=type(error).__name__)
        try:
            publish()
        except Exception as progress_error:
            report["progress_error"] = repr(progress_error)
        raise MeasurementFailure(str(error), report) from error
