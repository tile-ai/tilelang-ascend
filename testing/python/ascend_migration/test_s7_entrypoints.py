"""Stdlib-only CLI gates and independent per-stream evidence validation."""

import copy
from contextlib import nullcontext
import importlib.util
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_s2_entrypoints import RUNNERS, isolated_environment
from test_s5_cost_model import synthetic_profile

ENTRIES = ("s7_check_event_budget", "s7_check_handoff_ir", "s7_run_sync_stress_npu")


def command(entry, output):
    return [
        sys.executable,
        str(RUNNERS / (entry + ".py")),
        "--platform",
        "a3",
        "--output",
        str(output),
    ]


def blocked(tmp_path, entry, extra=(), optimize=False):
    output = tmp_path / "report.json"
    args = command(entry, output)
    if optimize:
        args.insert(1, "-O")
    env = isolated_environment(tmp_path, block_z3=True)
    env["TL_ASCEND_MIGRATION_BUILD_MANIFEST"] = str(tmp_path / "missing-manifest.json")
    env["TL_ASCEND_PIPELINE_COST_PROFILE"] = str(tmp_path / "missing-profile.json")
    result = subprocess.run([*args, *extra], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["step"] == "S7" and report["status"] == "BLOCKED"
    assert report["counts"]["passed"] == 0
    assert "runtime import forbidden" not in result.stdout + result.stderr
    return report


@pytest.mark.parametrize("entry", ENTRIES)
def test_help_does_not_import_runtime_or_solver(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(RUNNERS / (entry + ".py")), "--help"],
        env=isolated_environment(tmp_path, block_z3=True),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


@pytest.mark.parametrize("entry", ENTRIES)
def test_missing_build_evidence_precedes_runtime_import(tmp_path, entry):
    profile = tmp_path / "synthetic.json"
    profile.write_text(json.dumps(synthetic_profile()))
    extra = ["--cost-profile", str(profile)] if entry == ENTRIES[-1] else []
    assert "manifest" in blocked(tmp_path, entry, extra)["error"].lower()


@pytest.mark.parametrize("entry", ENTRIES)
def test_optimization_cannot_disable_protocol_assertions(tmp_path, entry):
    error = blocked(tmp_path, entry, optimize=True)["error"].lower()
    assert "optimization" in error or "pythonoptimize" in error


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_requires_finite_positive_seconds(tmp_path, entry, value):
    blocked(tmp_path, entry, ["--timeout", value])


@pytest.mark.parametrize("flag,value", [("--repeat", "0"), ("--seed", "-1"), ("--device", "-1")])
def test_stress_input_gates_precede_runtime(tmp_path, flag, value):
    blocked(tmp_path, ENTRIES[-1], [flag, value])


@pytest.mark.parametrize("content", [None, "{}", "not JSON"])
def test_profile_requires_real_existing_valid_evidence(tmp_path, content):
    path = tmp_path / "bad-profile.json"
    if content is not None:
        path.write_text(content)
    blocked(tmp_path, ENTRIES[-1], ["--cost-profile", str(path)])


@pytest.fixture
def stress(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNERS))
    spec = importlib.util.spec_from_file_location("s7_host_stress", RUNNERS / "_s7_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_report(module):
    cases = []
    for name, spec in module.PARAMETERS.items():
        width = spec["streams"] * spec["batches"]
        pointers = [
            {
                "stream": i // spec["batches"],
                "batch": i % spec["batches"],
                "A": 100 + 10 * i,
                "B": 101 + 10 * i,
                "O": 102 + 10 * i,
                "Workspace": None if spec["kind"] == "vector" else 103 + 10 * i,
            }
            for i in range(width)
        ]
        data = {
            **spec,
            "rounds": [
                {
                    "index": i,
                    "round": i // width,
                    "stream": (i % width) // spec["batches"],
                    "batch": i % spec["batches"],
                    "golden_status": "PASS",
                    "input_sha256": {"A": f"a{i}", "B": f"b{i}"},
                    "output_sha256": f"o{i}",
                    "golden_sha256": f"g{i}",
                }
                for i in range(3 * width)
            ],
            "stream_handles": list(range(1, 1 + spec["streams"])),
            "independent_buffer_pointers": pointers,
            "submission_windows": [
                {
                    "round": i,
                    "pending": [True] * spec["streams"],
                    "both_pending": spec["streams"] == 2,
                    "enqueues": [{"stream": s, "batch": b} for s in range(spec["streams"]) for b in range(spec["batches"])],
                    "queries": [{"stream": s, "complete": False} for s in range(spec["streams"])],
                    "done_queries": [{"stream": s, "complete": False} for s in range(2)],
                    "armed_streams": [0, 1],
                    "released_streams": [0, 1],
                    "gate_handles": [10, 11],
                    "done_handles": [20, 21],
                }
                for i in range(3)
            ],
            "submission_control": {
                "kind": "independent ExternalEvent wait-before-record gates",
                "gate_count": 2,
                "gate_handles": [10, 11],
                "done_handles": [20, 21],
                "release_stream_handle": 99,
            },
            "submission_environment": {"effective": {"PER_STREAM_QUEUE": "0", "ASCEND_LAUNCH_BLOCKING": "0"}},
        }
        cases.append(
            {
                "name": name,
                "evidence": {
                    "case_parameters": data,
                    "kernels": [
                        {
                            "resolved_flags": ["-O2"],
                            "resolved_pass_configs": {"tl.ascend_sync_strategy": spec["sync"]},
                        }
                    ],
                },
            }
        )
    return {"repeat": 3, "cases": cases}


def test_all_logical_calls_match_between_old_and_new_generators(stress):
    report = paired_report(stress)
    stress.compare_pairs(report)
    assert len(report["strategy_comparisons"]) == 10
    assert sum(row["compared_launches"] for row in report["strategy_comparisons"]) == 48


@pytest.mark.parametrize("streams", [1, 2])
def test_concurrency_description_matches_the_actual_scope(stress, streams):
    description = stress.concurrency_claim(streams)
    if streams == 1:
        assert description == "single stream; no concurrency claim"
    else:
        assert "simultaneously pending submissions" in description
        assert "no hardware timing-overlap claim" in description


@pytest.mark.parametrize(
    "damage",
    [
        "input",
        "output",
        "golden",
        "missing",
        "index",
        "repeat_input",
        "stream_identity",
        "fake_pending",
        "no_pending",
        "missing_window",
        "aliased_workspace",
        "strategy",
        "flags",
        "wrong_round",
        "wrong_stream",
        "wrong_batch",
        "gate_alias",
        "release_alias",
        "done_complete",
        "missing_release",
        "forged_query",
        "missing_enqueue",
    ],
)
def test_incomplete_or_forged_concurrent_evidence_is_rejected(stress, damage):
    report = copy.deepcopy(paired_report(stress))
    pair = [row["evidence"] for row in report["cases"] if row["name"].startswith("streams_workspace_")]
    left, right = [item["case_parameters"] for item in pair]
    if damage in ("input", "output", "golden"):
        if damage == "input":
            right["rounds"][0]["input_sha256"]["A"] = "changed"
        else:
            right["rounds"][0][damage + "_sha256"] = "changed"
    elif damage == "missing":
        right["rounds"].pop()
    elif damage == "strategy":
        pair[1]["kernels"][0]["resolved_pass_configs"]["tl.ascend_sync_strategy"] = "legacy"
    elif damage == "flags":
        pair[1]["kernels"][0]["resolved_flags"] = ["-O1"]
    else:
        for data in (left, right):
            if damage == "index":
                data["rounds"][0]["index"] = 5
            elif damage == "repeat_input":
                for row in data["rounds"]:
                    row["input_sha256"]["B"] = "stale"
            elif damage == "stream_identity":
                data["stream_handles"] = [1, 1]
            elif damage == "fake_pending":
                data["submission_windows"][0]["pending"] = [True, False]
            elif damage == "no_pending":
                for window in data["submission_windows"]:
                    window.update(pending=[False, False], both_pending=False)
            elif damage == "missing_window":
                data["submission_windows"].pop()
            elif damage == "aliased_workspace":
                data["independent_buffer_pointers"][1]["Workspace"] = data["independent_buffer_pointers"][0]["Workspace"]
            elif damage == "gate_alias":
                data["submission_control"]["gate_handles"] = [10, 10]
            elif damage == "release_alias":
                data["submission_control"]["release_stream_handle"] = data["stream_handles"][0]
            elif damage == "done_complete":
                data["submission_windows"][0]["done_queries"][0]["complete"] = True
            elif damage == "missing_release":
                data["submission_windows"][0]["released_streams"] = [0]
            elif damage == "forged_query":
                data["submission_windows"][0]["queries"][0]["complete"] = True
            elif damage == "missing_enqueue":
                data["submission_windows"][0]["enqueues"].pop()
            else:
                data["rounds"][0][damage.removeprefix("wrong_")] = 5
    with pytest.raises(AssertionError):
        stress.compare_pairs(report)


@pytest.mark.parametrize("failure", [AssertionError, OSError])
def test_worker_catch_preserves_live_window_evidence(stress, tmp_path, monkeypatch, failure):
    """Exercise the real worker catch through the public entry without a runtime."""
    import _npu_runner

    fake_tilelang = ModuleType("tilelang")
    fake_tilelang.compile = lambda *args, **kwargs: None
    adapter = ModuleType("tilelang.jit.adapter")
    adapter.libgen = SimpleNamespace(subprocess=SimpleNamespace(run=subprocess.run))
    kernel_module = ModuleType("tilelang.jit.kernel")

    class FakeKernel:
        def __call__(self, *args, **kwargs):
            return None

    kernel_module.JITKernel = FakeKernel
    for name, module in (
        ("tilelang", fake_tilelang),
        ("tilelang.jit.adapter", adapter),
        ("tilelang.jit.kernel", kernel_module),
        ("_s7_npu", stress),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_npu_runner, "prepare_worker", lambda *args: (fake_tilelang, None))
    spec = importlib.util.spec_from_file_location("s7_fault_entry", RUNNERS / "s7_run_sync_stress_npu.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    output = tmp_path / "failure.json"
    case = "streams_vector_legacy"
    original = dict(stress.PARAMETERS[case])
    args = SimpleNamespace(_case=case, output=output, repeat=3)
    monkeypatch.setenv("PER_STREAM_QUEUE", "1")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    monkeypatch.setattr(entry, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(entry, "base_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "validate", lambda *args: None)
    monkeypatch.setattr(entry, "preflight", lambda *args: None)
    observed = {
        "stream_handles": [123, 456],
        "rounds": [{"index": 0, "golden_status": "PASS", "output_sha256": "host fixture"}],
        "submission_windows": [
            {
                "round": 0,
                "pending": [False, True],
                "both_pending": False,
                "queries": [{"stream": 0, "complete": True}],
            }
        ],
    }

    def reject(name, worker_args, torch, *, metadata):
        assert name == case and worker_args is args and torch is None
        metadata.update(copy.deepcopy(observed))
        raise failure("injected late pending/query failure")

    monkeypatch.setattr(entry, "run_case", reject)
    assert entry.main() == 1
    recorded = json.loads(output.read_text())
    assert recorded["status"] == "FAIL" and "injected late" in recorded["error"]
    assert recorded["case_parameters"] == {**original, **observed}
    assert stress.PARAMETERS[case] == original


@pytest.mark.parametrize("streams", [1, 2])
def test_submission_environment_is_worker_local_and_precedes_import(stress, monkeypatch, streams):
    monkeypatch.setenv("PER_STREAM_QUEUE", "1")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    environment = stress.configure_submission_environment({"streams": streams})
    assert environment["inherited"] == {
        "PER_STREAM_QUEUE": "1",
        "ASCEND_LAUNCH_BLOCKING": "1",
    }
    assert set(environment["effective"].values()) == ({"0"} if streams == 2 else {"1"})


@pytest.mark.parametrize("fault", [None, "enqueue", "query", "done_complete"])
def test_start_gates_materialize_first_and_release_on_success_or_failure(stress, fault):
    log = []

    class Stream:
        def __init__(self, name, handle):
            self.name, self.npu_stream = name, handle

        def synchronize(self):
            log.append(("sync", self.name))

        def query(self):
            log.append(("query_stream", self.name))
            return False

    class Event:
        def __init__(self, kind, index):
            self.kind, self.index = kind, index
            self.npu_event = 100 + index + (10 if kind == "done" else 0)

        def record(self, stream):
            log.append(("record", self.kind, self.index, stream.name))

        def reset(self, stream):
            raise AssertionError("External value-event reset is a no-op and must not be used")

        def wait(self, stream):
            log.append(("wait", self.index, stream.name))

        def query(self):
            log.append(("query_done", self.index))
            if fault == "query":
                raise OSError("injected query error")
            return fault == "done_complete"

    target = [Stream("target0", 1), Stream("target1", 2)]
    control = Stream("control", 3)
    created = {"gate": [], "done": []}

    def event(kind):
        result = Event(kind, len(created[kind]))
        created[kind].append(result)
        return result

    torch = SimpleNamespace(
        npu=SimpleNamespace(
            Stream=lambda **kwargs: control,
            ExternalEvent=lambda: event("gate"),
            Event=lambda **kwargs: event("done"),
            stream=lambda stream: nullcontext(),
        )
    )
    gates, done, release = stress.create_submission_control(torch, 0)
    assert log == [
        ("record", "done", 0, "control"),
        ("record", "done", 1, "control"),
        ("sync", "control"),
    ]
    log.clear()

    def kernel(stream, batch):
        log.append(("kernel", stream, batch))
        if fault == "enqueue" and stream == 1:
            raise ValueError("injected enqueue error")

    window = {}

    def submit():
        stress.submit_round(
            kernel,
            [[(s, b) for b in range(2)] for s in range(2)],
            target,
            torch,
            window,
            gates=gates,
            done=done,
            release_stream=release,
        )

    if fault is None:
        submit()
        assert window["both_pending"] and len(window["enqueues"]) == 4
        assert [row["complete"] for row in window["done_queries"]] == [False, False]
    else:
        with pytest.raises((ValueError, OSError, AssertionError)):
            submit()
    assert window["armed_streams"] == window["released_streams"] == [0, 1]
    assert len(window["gate_waits"]) == len(window["releases"]) == 2
    assert len(window["synchronizations"]) == 3
    assert all(
        item["returned"] and item["host_completed_ns"] >= item["host_started_ns"]
        for field in ("gate_waits", "releases", "synchronizations")
        for item in window[field]
    )
    assert log[0] == ("wait", 0, "target0")
    assert log[-5:] == [
        ("record", "gate", 0, "control"),
        ("record", "gate", 1, "control"),
        ("sync", "control"),
        ("sync", "target0"),
        ("sync", "target1"),
    ]
    assert not any(row[0] == "sync" and row[1].startswith("target") for row in log[:-5])
