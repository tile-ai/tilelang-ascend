"""Host-only timing policy tests; fake runtime objects never execute a device kernel."""

from contextlib import nullcontext
import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


RUNNERS = Path(__file__).resolve().parents[3] / "scripts/ascend_migration"
if str(RUNNERS) not in sys.path:
    sys.path.insert(0, str(RUNNERS))
benchmark = importlib.import_module("_s8_benchmark")


@pytest.mark.parametrize("milliseconds,expected", [("0.001", 1000), ("0.0000001", 1)])
def test_event_units_preserve_resolution(milliseconds, expected):
    assert benchmark.event_span_ns(milliseconds) == expected


@pytest.mark.parametrize("value", [0, -1, "nan", "inf", True, "not a number"])
def test_invalid_event_times_cannot_become_samples(value):
    with pytest.raises(ValueError):
        benchmark.event_span_ns(value)


def test_median_and_bootstrap_use_actual_positive_samples():
    result = benchmark.sample_summary([10, 20, 30, 40, 50])
    assert result == benchmark.sample_summary([10, 20, 30, 40, 50])
    assert result["samples"] == 5 and result["median_ns"] == "30"
    assert result["min_ns"] == 10 and result["max_ns"] == 50
    assert float(result["median_ci95_low_ns"]) <= 30 <= float(result["median_ci95_high_ns"])


@pytest.mark.parametrize("samples", [[], [0], [-1], [True], [1.5]])
def test_invalid_sample_sets_are_rejected(samples):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.sample_summary(samples)


def install_fake_worker(monkeypatch, tmp_path, *, raw_ms=0.005, fail_golden=False):
    """Exercise the real controller while observing reset/check versus Event boundaries."""
    observed = SimpleNamespace(active=False, calls=0, resets=0, goldens=0, saves=0)

    class Stream:
        npu_stream = 12

        def synchronize(self):
            observed.active = False

    class Event:
        next_id = 20

        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
            self.npu_event = Event.next_id
            Event.next_id += 1

        def record(self, stream):
            assert stream.npu_stream == 12
            observed.active = True

        def synchronize(self):
            observed.active = False

        def elapsed_time(self, end):
            assert not observed.active and self is not end
            return raw_ms

    torch = SimpleNamespace(npu=SimpleNamespace(Stream=lambda **kwargs: Stream(), Event=Event, stream=lambda stream: nullcontext()))

    class State:
        input_hashes = {"A": "input"}
        golden_hash = "expected"
        buffer_pointers = {"A": 1, "O": 2}
        arguments = ("input", "output")

        def __init__(self, spec, args, runtime):
            assert runtime is torch

        def prepare(self, index):
            assert index == 0 and not observed.active

        def restore(self):
            assert not observed.active, "Reset entered the timed Event span"
            observed.resets += 1

        def check(self):
            assert not observed.active, "Golden entered the timed Event span"
            observed.goldens += 1
            if fail_golden:
                raise AssertionError("injected incorrect output")
            return {
                "input_sha256": dict(self.input_hashes),
                "golden_sha256": self.golden_hash,
                "output_sha256": "actual",
                "golden_status": "PASS",
            }

    def kernel(*arguments):
        assert arguments == State.arguments
        observed.calls += 1

    def compile_kernel(tilelang, spec, args, report, artifacts):
        report["compilations"].append({"status": "PASS"})
        report["kernels"].append({"resolved_flags": ["-O3"]})
        report["compile_timing"] = {"wall_ns": 100}
        return kernel

    original_save = benchmark._save

    def save(report, path):
        assert not observed.active, "Evidence serialization entered the Event span"
        observed.saves += 1
        original_save(report, path)

    monkeypatch.setattr(benchmark, "base_report", lambda *a, **k: {"repeat": 2})
    monkeypatch.setattr(benchmark, "prepare_worker", lambda *a: (object(), torch))
    monkeypatch.setattr(benchmark, "preflight", lambda *a: None)
    monkeypatch.setattr(benchmark, "_compile", compile_kernel)
    monkeypatch.setattr(benchmark, "describe_kernel", lambda *a: {})
    monkeypatch.setattr(benchmark, "CaseData", State)
    monkeypatch.setattr(benchmark, "_save", save)
    args = SimpleNamespace(_case="gemm_batch3_legacy", device=0, repeat=2, output=tmp_path / "worker.json")
    return args, observed


def test_real_controller_checks_three_warmups_before_each_measured_call(monkeypatch, tmp_path):
    args, observed = install_fake_worker(monkeypatch, tmp_path)
    assert benchmark.run_measured_worker(args) == 0
    report = json.loads(args.output.read_text())
    assert observed.calls == observed.resets == observed.goldens == 8
    assert report["golden_checks"] == 8 and len(report["samples"]) == 2
    assert [row["phase"] for row in report["journal"]] == list(benchmark.round_phases()) * 2
    assert [row["event_span_ns"] for row in report["samples"]] == [5000, 5000]
    assert report["sampling_policy"]["warmups_per_round"] == 3
    assert report["timing_identity"]["event_handles"] == [20, 21]


def test_failed_warmup_stops_before_a_timed_sample(monkeypatch, tmp_path):
    args, observed = install_fake_worker(monkeypatch, tmp_path, fail_golden=True)
    assert benchmark.run_measured_worker(args) == 1
    report = json.loads(args.output.read_text())
    assert observed.calls == observed.goldens == 1
    assert report["status"] == "FAIL" and report["samples"] == []
    assert report["golden_checks"] == 0


def test_zero_resolution_event_fails_instead_of_exporting_a_time(monkeypatch, tmp_path):
    args, observed = install_fake_worker(monkeypatch, tmp_path, raw_ms=0)
    assert benchmark.run_measured_worker(args) == 1
    report = json.loads(args.output.read_text())
    assert observed.calls == 4 and observed.goldens == 3
    assert "positive" in report["error"] and report["samples"] == []


def test_compile_clock_excludes_evidence_copy_and_hash(monkeypatch, tmp_path):
    """The actual observer is exercised around a fake compiler subprocess, without TVM."""
    libgen = ModuleType("tilelang.jit.adapter.libgen")
    libgen.subprocess = benchmark.subprocess
    for name in ("tilelang", "tilelang.jit", "tilelang.jit.adapter"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["tilelang.jit.adapter"].libgen = libgen
    monkeypatch.setitem(sys.modules, "tilelang.jit.adapter.libgen", libgen)
    source, library = tmp_path / "input.cpp", tmp_path / "kernel.so"
    source.write_text("// fake compiler input for host policy test\n")
    compiler = tmp_path / "bisheng"
    compiler.write_text("host test placeholder\n")
    compiler.chmod(0o700)
    clock = []

    def now():
        clock.append(len(clock) + 1)
        return len(clock) * 100

    def subprocess_run(command, *args, **kwargs):
        assert len(clock) == 2
        library.write_bytes(b"fake shared object")
        return SimpleNamespace(returncode=0)

    def build(tilelang, spec):
        libgen.subprocess.run([str(compiler), str(source), "-o", str(library)])
        return SimpleNamespace(compile_flags=["-O3"], pass_configs={})

    original_copy = benchmark.shutil.copyfile

    def copyfile(*args):
        assert len(clock) == 4, "Compiler artifact copy was inside the timed interval"
        return original_copy(*args)

    monkeypatch.setattr(benchmark.time, "perf_counter_ns", now)
    monkeypatch.setattr(benchmark.subprocess, "run", subprocess_run)
    monkeypatch.setattr(benchmark, "build_kernel", build)
    monkeypatch.setattr(benchmark.shutil, "copyfile", copyfile)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = {"compilations": [], "kernels": []}
    benchmark._compile(object(), {}, None, report, artifacts)
    assert report["compile_timing"]["wall_ns"] == 300
    assert report["compile_timing"]["bisheng_wall_ns"] == 100
    assert report["compilations"][0]["status"] == "PASS"
    assert (artifacts / "bisheng-input.cpp").read_bytes() == source.read_bytes()
    assert (artifacts / "npu-kernel.so").read_bytes() == library.read_bytes()


def paired_report():
    checked = {
        "input_sha256": {"A": "input"},
        "golden_sha256": "expected",
        "output_sha256": "actual",
        "golden_status": "PASS",
    }
    journal = [{"index": index, "round": 0, "phase": phase, **checked} for index, phase in enumerate(benchmark.round_phases())]
    worker = {
        "status": "PASS",
        "compilations": [{"status": "PASS", "argv": ["bisheng", "-O3", "worker.cpp", "-o", "worker.so"]}],
        "kernels": [{"resolved_flags": ["-O2", "--cce-auto-sync=off", "-O3"]}],
        "launches": {"kernel": {"attempted": 4, "completed": 4}},
        "golden_checks": 4,
        "journal": journal,
        "samples": [
            {
                "index": 0,
                "round": 0,
                "host_dispatch_ns": 100,
                "event_span_ms_raw": "0.001",
                "event_span_ns": 1000,
                **checked,
            }
        ],
        "sampling_policy": {"rounds": 1, "warmups_per_round": 3, "measured_per_round": 1},
        "toolchain": {"bisheng_sha256": "compiler"},
        "actual_device": {"soc": "host-test-placeholder"},
        "summary": {
            "host_dispatch_ns": benchmark.sample_summary([100]),
            "event_span_ns": benchmark.sample_summary([1000]),
        },
        "compile_timing": {"wall_ns": 10000},
    }
    return {
        "repeat": 1,
        "cases": [
            {"name": f"{base}_{strategy}", "evidence": copy.deepcopy(worker)}
            for base in benchmark.BENCHMARK_BASES
            for strategy in ("legacy", "auto")
        ],
    }


def test_pair_comparison_reports_regression_without_a_speedup_threshold():
    report = paired_report()
    for row in report["cases"][1::2]:
        worker = row["evidence"]
        worker["samples"][0]["event_span_ms_raw"] = "0.002"
        worker["samples"][0]["event_span_ns"] = 2000
        worker["summary"]["event_span_ns"] = benchmark.sample_summary([2000])
    benchmark.compare_performance(report)
    assert len(report["performance_comparisons"]) == 3
    assert all(row["measurements"]["event_span_ns"]["auto_change_percent"] == "100" for row in report["performance_comparisons"])


@pytest.mark.parametrize("corruption", ["count", "warmup", "flags", "argv", "input", "span"])
def test_pair_comparison_rejects_incomplete_or_unmatched_evidence(corruption):
    report = paired_report()
    worker = report["cases"][1]["evidence"]
    if corruption == "count":
        worker["golden_checks"] -= 1
    elif corruption == "warmup":
        worker["journal"][0]["phase"] = "measured"
    elif corruption == "flags":
        worker["kernels"][0]["resolved_flags"].append("-O1")
    elif corruption == "argv":
        worker["compilations"][0]["argv"].insert(1, "--cce-auto-sync=on")
    elif corruption == "input":
        worker["journal"][0]["input_sha256"] = {"A": "different"}
    else:
        worker["samples"][0]["event_span_ns"] += 1
    with pytest.raises(AssertionError):
        benchmark.compare_performance(report)
