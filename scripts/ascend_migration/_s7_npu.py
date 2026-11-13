"""S7 device fixtures, two-stream submission and evidence; stdlib-only on import."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

from _common import ROOT
from _npu_runner import digest
from _s5_npu import check_identity, physical_evidence
from _s6_npu import validate as validate


FIXTURES = {
    "zero": {"kind": "vector", "trips": 0, "versions": 1, "cores": 1},
    "one": {"kind": "vector", "trips": 1, "versions": 1, "cores": 1},
    "three": {"kind": "vector", "trips": 3, "versions": 2, "cores": 1},
    "long_n1": {"kind": "vector", "trips": 257, "versions": 1, "cores": 1},
    "long_n2": {"kind": "vector", "trips": 257, "versions": 2, "cores": 8},
    "long_n3": {"kind": "vector", "trips": 257, "versions": 3, "cores": 8},
    "gm_v2c": {"kind": "v2c", "trips": 17, "cores": 4},
    "gm_c2v": {"kind": "c2v", "trips": 17, "cores": 4},
    "streams_vector": {
        "kind": "vector",
        "trips": 513,
        "versions": 3,
        "cores": 8,
        "streams": 2,
        "batches": 2,
    },
    "streams_workspace": {
        "kind": "c2v",
        "trips": 17,
        "cores": 1,
        "streams": 2,
        "batches": 2,
    },
}
PARAMETERS = {
    f"{name}_{sync}": {"streams": 1, "batches": 1, **spec, "sync": sync} for name, spec in FIXTURES.items() for sync in ("legacy", "a3")
}


def parser():
    result = argparse.ArgumentParser(description="S7 A3 synchronization and two-stream stress")
    result.set_defaults(require_solver=True)
    result.add_argument("--platform", choices=("a3",), required=True)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--repeat", type=int, default=100)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--build-manifest",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_MIGRATION_BUILD_MANIFEST",
                ROOT / "build/ascend_migration_s7_native_build.json",
            )
        ),
    )
    result.add_argument(
        "--cost-profile",
        type=Path,
        default=Path(
            os.environ.get(
                "TL_ASCEND_PIPELINE_COST_PROFILE",
                ROOT / "build/ascend_migration_s5_cost_profile.json",
            )
        ),
    )
    result.add_argument("--_case", choices=("preflight", *PARAMETERS), help=argparse.SUPPRESS)
    return result


def preflight(args, evidence, tilelang):
    check_identity(args, evidence, tilelang)


def pass_configs(args, spec):
    vector = spec["kind"] == "vector"
    return {
        "tl.ascend_pipeline_planner": "constraint" if vector else "legacy",
        "tl.ascend_sync_strategy": spec["sync"],
        "tl.ascend_pipeline_cost_model": args.cost_model,
        "tl.ascend_auto_sync": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_auto_cross_core_sync": False,
        "tl.ascend_auto_sync_vs": not vector,
    }


def program(spec):
    sys.path.insert(0, str(ROOT / "testing/python/ascend_migration"))
    from test_s7_kernels import make_cv_handoff_kernel, make_vector_sync_kernel

    if spec["kind"] == "vector":
        return make_vector_sync_kernel(trips=spec["trips"], versions=spec["versions"], cores=spec["cores"])
    return make_cv_handoff_kernel(
        direction=spec["kind"],
        trips=spec["trips"],
        cores=spec["cores"],
        manual_legacy=spec["sync"] == "legacy",
    )


def tensor_digest(tensor, torch):
    """Preserve the S5 dtype/shape/byte hash, without a Python list per GM byte."""
    host = tensor.detach().cpu().contiguous()
    result = hashlib.sha256(f"{host.dtype}:{tuple(host.shape)}:".encode())
    result.update(host.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def concurrency_claim(streams):
    return (
        "start-gated simultaneously pending submissions; no hardware timing-overlap claim"
        if streams == 2
        else "single stream; no concurrency claim"
    )


def configure_submission_environment(spec):
    """Disable per-stream queue mode and launch blocking before runtime import."""
    names = ("PER_STREAM_QUEUE", "ASCEND_LAUNCH_BLOCKING")
    inherited = {name: os.environ.get(name) for name in names}
    if spec.get("streams", 1) == 2:
        for name in names:
            os.environ[name] = "0"
    return {
        "inherited": inherited,
        "effective": {name: os.environ.get(name) for name in names},
    }


def create_submission_control(torch, device):
    """External gates are eagerly created; materialize only lazy done events."""
    release_stream = torch.npu.Stream(device=device)
    gates = [torch.npu.ExternalEvent() for _ in range(2)]
    done = [torch.npu.Event(enable_timing=False) for _ in range(2)]
    handles = [int(gate.npu_event) for gate in gates]
    if not all(handles) or len(set(handles)) != 2:
        raise RuntimeError("Independent eagerly created ExternalEvent value gates are required")
    for event in done:
        event.record(release_stream)
    release_stream.synchronize()
    return gates, done, release_stream


def submit_round(
    kernel,
    calls,
    streams,
    torch,
    window,
    *,
    gates=None,
    done=None,
    release_stream=None,
):
    """Hold independent target queues until both real completion markers are pending."""
    armed = []
    window.update(
        enqueues=[],
        queries=[],
        pending=[],
        both_pending=False,
        gate_waits=[],
        releases=[],
        synchronizations=[],
    )

    def observe(action, records, **identity):
        record = {**identity, "host_started_ns": time.perf_counter_ns(), "returned": False}
        records.append(record)
        try:
            action()
            record["returned"] = True
        finally:
            record["host_completed_ns"] = time.perf_counter_ns()
            record["host_elapsed_ns"] = record["host_completed_ns"] - record["host_started_ns"]

    if gates is not None:
        window.update(done_queries=[], armed_streams=[], released_streams=[])
    try:
        for stream_index, stream in enumerate(streams):
            if gates is not None:
                armed.append(stream_index)
                window["armed_streams"].append(stream_index)
                observe(
                    lambda gate=gates[stream_index], target=stream: gate.wait(target),
                    window["gate_waits"],
                    stream=stream_index,
                    event_handle=int(gates[stream_index].npu_event),
                )
            with torch.npu.stream(stream):
                for batch, call_args in enumerate(calls[stream_index]):
                    started = time.perf_counter_ns()
                    kernel(*call_args)
                    completed = time.perf_counter_ns()
                    window["enqueues"].append(
                        {
                            "stream": stream_index,
                            "batch": batch,
                            "host_started_ns": started,
                            "host_completed_ns": completed,
                            "host_elapsed_ns": completed - started,
                        }
                    )
                if gates is not None:
                    done[stream_index].record(stream)
        if gates is not None:
            window["gate_handles"] = [int(gate.npu_event) for gate in gates]
            window["done_handles"] = [int(event.npu_event) for event in done]
            for stream_index, event in enumerate(done):
                started = time.perf_counter_ns()
                complete = bool(event.query())
                completed = time.perf_counter_ns()
                window["done_queries"].append(
                    {
                        "stream": stream_index,
                        "complete": complete,
                        "host_started_ns": started,
                        "host_completed_ns": completed,
                        "host_elapsed_ns": completed - started,
                    }
                )
        for stream_index, stream in enumerate(streams):
            started = time.perf_counter_ns()
            complete = bool(stream.query())
            completed = time.perf_counter_ns()
            window["queries"].append(
                {
                    "stream": stream_index,
                    "complete": complete,
                    "host_started_ns": started,
                    "host_completed_ns": completed,
                    "host_elapsed_ns": completed - started,
                }
            )
            window["pending"].append(not complete)
        window["both_pending"] = len(window["pending"]) == 2 and all(window["pending"])
        if gates is not None and (not window["both_pending"] or any(item["complete"] for item in window["done_queries"])):
            raise AssertionError("Both queued target batches and completion markers must be pending")
    finally:
        if gates is not None:
            for stream_index in armed:
                observe(
                    lambda gate=gates[stream_index]: gate.record(release_stream),
                    window["releases"],
                    stream=stream_index,
                    event_handle=int(gates[stream_index].npu_event),
                )
                window["released_streams"].append(stream_index)
            observe(release_stream.synchronize, window["synchronizations"], role="release")
        for stream_index, stream in enumerate(streams):
            observe(stream.synchronize, window["synchronizations"], role="target", stream=stream_index)


def sync_evidence(kernel, args, spec):
    from tilelang import tvm

    compiled_input = Path(kernel.adapter.srcpath)
    source = compiled_input.read_text()
    if source != kernel.get_kernel_source():
        raise AssertionError("The preserved source must be the actual Bisheng input")
    preserved_cpp = args.output.parent / f"{args.output.stem}-compiled.cpp"
    preserved_cpp.write_text(source)
    cpp_fences = [line.strip() for line in source.splitlines() if "PipeBarrier" in line and "PIPE_ALL" in line]
    records = []
    for symbol, function in kernel.artifact.device_mod.functions.items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue
        attrs = function.attrs or {}
        captured = {str(key): attrs[key] for key in attrs.keys() if str(key).startswith(("tl.ascend_local_sync_", "tl.ascend_handoff_"))}
        record = {"function": symbol.name_hint, "attribute_names": sorted(captured)}
        if spec["sync"] == "a3":
            expected = "tl.ascend_local_sync_plan" if spec["kind"] == "vector" else None
            if expected is not None:
                if expected not in captured or str(captured[expected]["generator"]) != "a3":
                    raise AssertionError("New local generator must produce its actual IR contract")
                if str(captured[expected]["source_phase"]) != "post_memory_planning":
                    raise AssertionError("New local evidence requires the final physical memory plan")
                model = str(captured[expected]["lifetime_model"])
                if model != "wait_all_fence_before_reuse":
                    raise AssertionError("New binary event leases need a fence-based lifetime model")
                record["lifetime_model"] = model
            elif not any(key.startswith("tl.ascend_handoff_") for key in captured):
                raise AssertionError("New GM handoff requires actual preserved joint protocol facts")
            else:
                plan = captured["tl.ascend_handoff_plan"]
                if (
                    int(plan["schema_version"]) != 1
                    or int(plan["mode"]) != 2
                    or int(plan["channel_count"]) != 1
                    or bool(plan["round_overlap"])
                    or int(plan["user_event_count"]) != 11
                    or int(plan["counter_depth"]) != 15
                ):
                    raise AssertionError("Unexpected actual fixed single-slot A3 mode2 contract")
                record["handoff_plan"] = {
                    key: int(plan[key])
                    for key in (
                        "schema_version",
                        "mode",
                        "ready_id",
                        "release_id",
                        "user_event_count",
                        "counter_depth",
                        "channel_count",
                    )
                }
                record["handoff_plan"]["round_overlap"] = bool(plan["round_overlap"])
            destination = args.output.parent / f"{args.output.stem}-{symbol.name_hint}-s7.json"
            destination.write_text(tvm.ir.save_json(tvm.runtime.convert(captured)) + "\n")
            record["contract"] = {"path": str(destination), "sha256": digest(destination)}
        events, barriers = [], []

        def visit(node, events=events, barriers=barriers):
            if isinstance(node, tvm.tir.Call):
                name = str(getattr(node.op, "name", ""))
                item = {"op": name, "args": [str(value) for value in node.args]}
                if "flag" in name:
                    events.append(item)
                elif "barrier" in name:
                    barriers.append(item)

        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
        record["actual_flag_calls"] = events
        record["actual_barrier_calls"] = barriers
        record["bisheng_source"] = {
            "path": str(preserved_cpp),
            "sha256": digest(preserved_cpp),
            "original_input": str(compiled_input),
            "all_fence_lines": cpp_fences,
        }
        host_call = source[source.index('extern "C" void call(') :]
        record["bisheng_source"]["host_call_lines"] = host_call.splitlines()
        record["bisheng_source"]["host_call_has_explicit_synchronization"] = any(
            token in host_call for token in ("Synchronize", "synchronize")
        )
        if (
            spec["kind"] == "vector"
            and spec["sync"] == "a3"
            and (not cpp_fences or not any("PIPE_ALL" in " ".join(row["args"]) for row in barriers))
        ):
            raise AssertionError("New local IR and actual Bisheng source must retain ALL fences")
        records.append(record)
    if not records:
        raise AssertionError("Missing actual compiled device synchronization evidence")
    if spec["kind"] != "vector" and not all(
        token in source for token in ("KERNEL_TYPE_MIX_AIC_1_2", "CrossCoreSetFlag", "CrossCoreWaitFlag")
    ):
        raise AssertionError("A3 workspace execution needs the real 1C:2V cross protocol")
    return records


def run_case(name, args, torch, *, metadata=None):
    import tilelang
    from _s5_npu import exact_output_trace

    spec = PARAMETERS[name]
    vector = spec["kind"] == "vector"
    kernel = tilelang.compile(
        program(spec),
        out_idx=[],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs(args, spec),
    )
    physical = physical_evidence(kernel)
    actual = [int(row["versions"]) for fn in physical for row in fn["buffer_remap"]]
    if vector and spec["trips"] >= 3 and (not actual or max(actual) != spec["versions"]):
        raise AssertionError(f"Actual private versions {actual} differ from the tested bound")
    streams = [torch.npu.Stream(device=args.device) for _ in range(2)] if spec["streams"] == 2 else [torch.npu.current_stream()]
    gates = done = release_stream = None
    if spec["streams"] == 2:
        if any(os.environ.get(name) != "0" for name in ("PER_STREAM_QUEUE", "ASCEND_LAUNCH_BLOCKING")):
            raise RuntimeError("Wait-before-record submission requires both host queue options set to 0")
        gates, done, release_stream = create_submission_control(torch, args.device)
    if metadata is None:
        metadata = {}
    metadata.update(
        {
            **spec,
            "physical_ir": physical,
            "actual_materialized_versions": actual,
            "sync_ir": sync_evidence(kernel, args, spec),
            "rounds": [],
            "submission_windows": [],
            "stream_handles": [int(stream.npu_stream) for stream in streams],
            "directive_difference": (
                "Legacy reference has explicit balanced READY0/FREE1; a3 source has none"
                if not vector
                else "Same constraint planner, directives and logical data; sync differs"
            ),
            "workspace_source": "none" if vector else "caller supplied explicit disjoint GM parameter",
            "host_observation_policy": ("perf_counter_ns host enqueue/query observations; not device performance timings"),
            "submission_environment": getattr(args, "submission_environment", None),
        }
    )
    if release_stream is not None:
        metadata["submission_control"] = {
            "kind": "independent ExternalEvent wait-before-record gates",
            "gate_count": len(gates),
            "gate_handles": [int(gate.npu_event) for gate in gates],
            "done_handles": [int(event.npu_event) for event in done],
            "release_stream_handle": int(release_stream.npu_stream),
            "claim": "target batches simultaneously queued before gate release; no hardware overlap claim",
            "resource_initialization": (
                "ExternalEvent constructors create both zero-value gates; pre-record only done "
                "events on control stream and synchronize before the first target wait"
            ),
            "reuse_policy": (
                "Each wait queues automatic ValueWrite0; each release record permits one next wait; "
                "reuse only after both target streams completed, without explicit reset"
            ),
        }
        if metadata["submission_control"]["release_stream_handle"] in metadata["stream_handles"]:
            raise AssertionError("Gate release requires a third independent real stream")
    metadata["reference_role"] = (
        "existing local generator"
        if vector and spec["sync"] == "legacy"
        else "new local generator"
        if vector
        else "explicit balanced manual mode2 legacy reference"
        if spec["sync"] == "legacy"
        else "new generated joint mode2 workspace protocol"
    )
    if len(set(metadata["stream_handles"])) != spec["streams"]:
        raise AssertionError("Two-stream acceptance requires distinct real stream handles")
    cores, trips = spec["cores"], spec["trips"]
    if vector:
        input_shape = output_shape = (cores, trips + 2, 64)
        valid = [((core * (trips + 2) + row) * 64 + lane) for core in range(cores) for row in range(trips + 1) for lane in range(64)]
        metadata["final_output_trace"] = exact_output_trace(kernel, valid)
    else:
        input_shape, output_shape = (cores, trips, 128, 128), (cores, trips + 1, 128, 128)
    inputs, outputs, workspaces, pointer_sets = {}, {}, {}, []
    for stream in range(spec["streams"]):
        for batch in range(spec["batches"]):
            key = (stream, batch)
            dtype = torch.float32 if vector else torch.float16
            a = torch.empty(input_shape, dtype=dtype, device="npu")
            b = torch.empty_like(a)
            output = torch.empty(output_shape, dtype=torch.float32, device="npu")
            workspace = (
                None
                if vector
                else torch.empty(
                    (cores, 144, 128),
                    dtype=torch.float16 if spec["kind"] == "v2c" else torch.float32,
                    device="npu",
                )
            )
            inputs[key], outputs[key], workspaces[key] = (a, b), output, workspace
            pointer_sets.append(
                {
                    "stream": stream,
                    "batch": batch,
                    "A": a.data_ptr(),
                    "B": b.data_ptr(),
                    "O": output.data_ptr(),
                    "Workspace": None if vector else workspace.data_ptr(),
                }
            )
    for field in ("A", "B", "O", *(() if vector else ("Workspace",))):
        if len({item[field] for item in pointer_sets}) != spec["streams"] * spec["batches"]:
            raise AssertionError("Queued calls require independently allocated inputs/output/workspace")
    metadata["independent_buffer_pointers"] = pointer_sets
    generator = torch.Generator().manual_seed(args.seed)
    for round_index in range(args.repeat):
        host = {}
        for key, (a, b) in inputs.items():
            stream_index, batch = key
            tag = (round_index % 11) + 3 * stream_index + batch
            core_labels = torch.arange(cores, dtype=a.dtype).reshape(cores, *([1] * (len(input_shape) - 1)))
            a_host = (torch.randint(-4, 5, input_shape, generator=generator).to(a.dtype) + tag + core_labels) / 32
            b_host = (torch.randint(-4, 5, input_shape, generator=generator).to(b.dtype) - tag - core_labels) / 32
            expected = torch.full(output_shape, float("nan"), dtype=torch.float32)
            if vector:
                expected[:, 0, :] = 7
                expected[:, 1 : trips + 1, :] = a_host[:, 1 : trips + 1, :] + b_host[:, 1 : trips + 1, :]
            else:
                expected[:, :trips, :, :] = a_host.float() @ b_host.float()
            hashes = {"A": tensor_digest(a_host, torch), "B": tensor_digest(b_host, torch)}
            workspace_expected = None
            with torch.npu.stream(streams[stream_index]):
                a.copy_(a_host)
                b.copy_(b_host)
                outputs[key].fill_(float("nan"))
                workspace = workspaces[key]
                if workspace is not None:
                    w_host = torch.full(workspace.shape, float("nan"), dtype=workspace.dtype)
                    w_host[:, :128, :] = (
                        torch.randint(-8, 9, (cores, 128, 128), generator=generator).to(workspace.dtype)
                        + tag
                        + torch.arange(cores, dtype=workspace.dtype).reshape(cores, 1, 1)
                    ) / 16
                    workspace.copy_(w_host)
                    hashes["Workspace"] = tensor_digest(w_host, torch)
                    workspace_expected = torch.full_like(w_host, float("nan"))
                    if trips:
                        workspace_expected[:, :128, :] = (
                            a_host[:, trips - 1, :, :] if spec["kind"] == "v2c" else expected[:, trips - 1, :, :]
                        )
            host[key] = (expected, workspace_expected, hashes)
        window = {"round": round_index}
        metadata["submission_windows"].append(window)
        calls = []
        for stream_index in range(spec["streams"]):
            batch_calls = []
            for batch in range(spec["batches"]):
                key = (stream_index, batch)
                a, b = inputs[key]
                batch_calls.append((a, b, outputs[key]) if vector else (a, b, workspaces[key], outputs[key]))
            calls.append(batch_calls)
        submit_round(
            kernel,
            calls,
            streams,
            torch,
            window,
            gates=gates,
            done=done,
            release_stream=release_stream,
        )
        for stream_index in range(spec["streams"]):
            for batch in range(spec["batches"]):
                key = (stream_index, batch)
                expected, workspace_expected, hashes = host[key]
                torch.testing.assert_close(outputs[key].cpu(), expected, rtol=1e-4, atol=1e-4, equal_nan=True)
                record = {
                    "index": len(metadata["rounds"]),
                    "round": round_index,
                    "stream": stream_index,
                    "batch": batch,
                    "golden_status": "PASS",
                    "input_sha256": hashes,
                    "output_sha256": tensor_digest(outputs[key], torch),
                    "golden_sha256": tensor_digest(expected, torch),
                }
                if workspace_expected is not None:
                    torch.testing.assert_close(
                        workspaces[key].cpu(),
                        workspace_expected,
                        rtol=1e-4,
                        atol=1e-4,
                        equal_nan=True,
                    )
                    record["workspace_output_sha256"] = tensor_digest(workspaces[key], torch)
                    record["workspace_golden_sha256"] = tensor_digest(workspace_expected, torch)
                metadata["rounds"].append(record)
    if spec["streams"] == 2 and not any(item["both_pending"] for item in metadata["submission_windows"]):
        raise AssertionError("No simultaneous pending submissions were observed on the two real streams")
    metadata["output_sentinel"] = "NaN output/canary each call; GM workspace finite random prefix plus NaN tail"
    metadata["concurrency_claim"] = concurrency_claim(spec["streams"])
    return metadata


def compare_pairs(report):
    results = {row["name"]: row["evidence"] for row in report["cases"]}
    comparisons = report["strategy_comparisons"] = []
    for fixture in FIXTURES:
        names = [fixture + "_" + strategy for strategy in ("legacy", "a3")]
        left, right = [results[name]["case_parameters"] for name in names]
        spec = PARAMETERS[names[0]]
        count = report["repeat"] * spec["streams"] * spec["batches"]
        if len(left["rounds"]) != count or left["rounds"] != right["rounds"]:
            raise AssertionError("Paired per-launch input/output/golden/workspace evidence differs")
        for name in names:
            evidence = results[name]
            data = evidence["case_parameters"]
            if any(data[key] != value for key, value in PARAMETERS[name].items()):
                raise AssertionError("Wrong actual S7 fixture parameters")
            if evidence["kernels"][0]["resolved_pass_configs"].get("tl.ascend_sync_strategy") != PARAMETERS[name]["sync"]:
                raise AssertionError("Wrong actual synchronization strategy")
            if len(set(data["stream_handles"])) != spec["streams"]:
                raise AssertionError("Stream handles do not establish independent real streams")
            if len(data["submission_windows"]) != report["repeat"]:
                raise AssertionError("Every round requires its actual submission window")
            for index, window in enumerate(data["submission_windows"]):
                if (
                    window["round"] != index
                    or len(window["pending"]) != spec["streams"]
                    or window["both_pending"] != (spec["streams"] == 2 and all(window["pending"]))
                ):
                    raise AssertionError("Invalid simultaneously-pending submission evidence")
            if spec["streams"] == 2 and not any(x["both_pending"] for x in data["submission_windows"]):
                raise AssertionError("Missing two-stream outstanding submission evidence")
            if spec["streams"] == 2:
                control = data["submission_control"]
                if (
                    control["kind"] != "independent ExternalEvent wait-before-record gates"
                    or control["gate_count"] != 2
                    or len(set(control["gate_handles"])) != 2
                    or len(set(control["done_handles"])) != 2
                    or control["release_stream_handle"] in data["stream_handles"]
                    or any(
                        data["submission_environment"]["effective"].get(key) != "0"
                        for key in ("PER_STREAM_QUEUE", "ASCEND_LAUNCH_BLOCKING")
                    )
                ):
                    raise AssertionError("Invalid independent start-gate submission resources")
                for window in data["submission_windows"]:
                    if (
                        not window["both_pending"]
                        or window["armed_streams"] != [0, 1]
                        or window["released_streams"] != [0, 1]
                        or window["gate_handles"] != control["gate_handles"]
                        or window["done_handles"] != control["done_handles"]
                        or [item["stream"] for item in window["done_queries"]] != [0, 1]
                        or any(item["complete"] for item in window["done_queries"])
                        or [item["complete"] for item in window["queries"]] != [False, False]
                        or [(item["stream"], item["batch"]) for item in window["enqueues"]]
                        != [(s, b) for s in range(2) for b in range(spec["batches"])]
                    ):
                        raise AssertionError("Missing queued targets or unreleased completion markers")
            buffers = data["independent_buffer_pointers"]
            if len(buffers) != spec["streams"] * spec["batches"]:
                raise AssertionError("Missing independent per-call device buffers")
            pointers = [value for row in buffers for key, value in row.items() if key in ("A", "B", "O", "Workspace") and value is not None]
            if len(set(pointers)) != len(pointers) or any(value <= 0 for value in pointers):
                raise AssertionError("Concurrent logical calls alias their device buffers")
        for index, record in enumerate(left["rounds"]):
            if (
                record["index"] != index
                or record["golden_status"] != "PASS"
                or (record["round"], record["stream"], record["batch"])
                != (
                    index // (spec["streams"] * spec["batches"]),
                    (index % (spec["streams"] * spec["batches"])) // spec["batches"],
                    index % spec["batches"],
                )
            ):
                raise AssertionError("Missing per-launch golden or ordinal")
        for key in left["rounds"][0]["input_sha256"]:
            if len({row["input_sha256"][key] for row in left["rounds"]}) != count:
                raise AssertionError("Every call must use fresh labeled input/workspace contents")
        if results[names[0]]["kernels"][0]["resolved_flags"] != results[names[1]]["kernels"][0]["resolved_flags"]:
            raise AssertionError("Paired strategies use different compiler flags")
        comparisons.append({"cases": names, "status": "PASS", "compared_launches": count})
