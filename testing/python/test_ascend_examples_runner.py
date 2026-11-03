# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""CPU checks for reproducible example execution and result accounting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/testing"


def _load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return _load_module("_tested_ascend_examples_runner", SCRIPTS / "run_ascend_examples.py")


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return _load_module(
        "_tested_ascend_examples_evidence",
        SCRIPTS / "_ascend_examples_evidence.py",
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _entry(path="examples/unit/example.py", **fields):
    return {"path": path, "mode": "script", "device": 0, "reason": "CPU fixture", **fields}


def test_default_manifest_preserves_the_original_comparison_scope(runner):
    current = runner.load_manifest(REPO / "ci/ascend_sync_examples.json")
    canonical = json.dumps(current, sort_keys=True, separators=(",", ":")).encode()
    original_sha256 = "049c3cc1ea8f075d7b76092ed87058389c4b647645458237c70ed948d8f781e4"
    assert hashlib.sha256(canonical).hexdigest() == original_sha256
    original_path = REPO / "build/pr1965-next-failures-v26-20261103/manifest-original-215.json"
    if original_path.is_file():
        assert current == json.loads(original_path.read_text())
    assert len(current) == len({entry["path"] for entry in current}) == 215
    assert sum(entry["mode"] == "skip" for entry in current) == 54
    assert sum(entry["mode"] == "pytest" for entry in current) == 17


def test_manifest_keeps_arguments_marker_and_explicit_budget(runner, tmp_path):
    source = tmp_path / "examples/unit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    entry = _entry(
        args=["--name", "literal `$(not-a-shell)` value"],
        timeout_seconds=300,
        pytest_markers="not bench and not low_priority",
    )
    manifest = _write_json(tmp_path / "manifest.json", [entry])
    assert runner.load_manifest(manifest, repo=tmp_path) == [entry]


@pytest.mark.parametrize(
    "mutation",
    [
        {"path": "examples/../../outside.py"},
        {"path": "src/not_an_example.py"},
        {"path": "examples/missing.py"},
        {"mode": "unknown"},
        {"device": -1},
        {"device": True},
        {"args": "--unexpected-string"},
        {"args": [1]},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"mode": "skip", "reason": ""},
    ],
)
def test_malformed_manifest_does_not_start_a_different_workload(runner, tmp_path, mutation):
    source = tmp_path / "examples/unit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    entry = _entry()
    entry.update(mutation)
    manifest = _write_json(tmp_path / "manifest.json", [entry])
    with pytest.raises((ValueError, TypeError)):
        runner.load_manifest(manifest, repo=tmp_path)


def test_duplicate_manifest_entries_are_rejected(runner, tmp_path):
    source = tmp_path / "examples/unit/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    manifest = _write_json(tmp_path / "manifest.json", [_entry(), _entry()])
    with pytest.raises(ValueError):
        runner.load_manifest(manifest, repo=tmp_path)


def test_device_selection_rejects_ambiguous_or_invalid_assignments(runner):
    assert runner.parse_devices("0,1") == [0, 1]
    assert runner.parse_devices("3") == [3]
    for invalid in ("", "0,0", "-1", "one", "0,,1"):
        with pytest.raises((ValueError, TypeError, argparse.ArgumentTypeError)):
            runner.parse_devices(invalid)


def test_failed_selection_is_a_new_subset_without_historical_passes(runner, tmp_path):
    entries = [_entry(f"examples/op/{name}.py") for name in ("pass", "fail", "timeout")]
    previous = _write_json(
        tmp_path / "previous.json",
        {
            "entries": [
                dict(entries[0], status="PASS"),
                dict(entries[1], status="FAIL"),
                dict(entries[2], status="TIMEOUT"),
            ]
        },
    )
    assert runner.select_entries(entries, failed_from=previous) == entries[1:]
    assert runner.select_entries(entries, only=[entries[2]["path"], entries[0]["path"]]) == [
        entries[0],
        entries[2],
    ]
    with pytest.raises(ValueError):
        runner.select_entries(entries, only=["examples/not_in_manifest.py"])


def test_existing_run_directory_is_never_reused(runner, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "result.json"
    sentinel.write_text("preserve this evidence")
    with pytest.raises((ValueError, FileExistsError)):
        runner.create_run_directory(output)
    assert sentinel.read_text() == "preserve this evidence"


def test_dry_run_does_not_import_device_packages_build_or_create_output(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "sitecustomize.py").write_text(
        "import sys\n"
        "class BlockDevicePackages:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('tilelang', 'torch', 'torch_npu'):\n"
        "            raise RuntimeError('dry-run imported a device package')\n"
        "sys.meta_path.insert(0, BlockDevicePackages())\n"
    )
    output = tmp_path / "must-not-be-created"
    env = dict(os.environ, PYTHONPATH=str(blocked))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_ascend_examples.py"),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout
    assert "215" in result.stdout
    assert not output.exists()


def _snapshot_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("src/operator.cc", "tilelang/entry.py", "examples/op/example.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src", "tilelang", "examples"], check=True)
    return tmp_path


def test_source_snapshot_detects_edits_additions_and_deletions(runner, tmp_path):
    repo = _snapshot_repo(tmp_path)
    snapshot = runner.SourceSnapshot.capture(repo)
    assert snapshot.check()["valid"]
    (repo / "src/operator.cc").write_text("changed\n")
    (repo / "tilelang/new.py").write_text("new\n")
    (repo / "examples/op/example.py").unlink()
    result = snapshot.check()
    assert not result["valid"]
    assert any("operator.cc" in str(path) for path in result["changed"])
    assert any("new.py" in str(path) for path in result["added"])
    assert any("example.py" in str(path) for path in result["removed"])


def test_generated_build_and_bytecode_files_do_not_invalidate_source(runner, tmp_path):
    repo = _snapshot_repo(tmp_path)
    snapshot = runner.SourceSnapshot.capture(repo)
    for name in ("build/result.cc", "tilelang/__pycache__/entry.pyc"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    assert snapshot.check()["valid"]


@pytest.fixture
def accepted_case(tmp_path):
    key = "fixture"
    source = tmp_path / "kernels" / f"{key}-001.cc"
    source.parent.mkdir()
    source.write_text("// A generated device kernel\n")
    entry = _entry()
    aliases = {"PassConfigKey.TL_ASCEND_AUTO_SYNC": "tl.ascend_auto_sync"}
    metadata = dict(
        path=entry["path"],
        mode=entry["mode"],
        variant="upstream",
        npu_device=0,
        native_abi="test-native-abi",
        cache_disabled=True,
        instrumentation_version=6,
        original_args=[],
        compile_attempts=1,
        compiled=1,
        launches=1,
        assertions=0,
        assertion_failures=[],
        launch_failures=[],
        kernels=[
            dict(
                index=1,
                pass_configs={
                    "tl.ascend_sync_backend": "upstream",
                    "PassConfigKey.TL_ASCEND_AUTO_SYNC": True,
                    "tl.ascend_auto_sync_vs": True,
                },
                compile_flags=["--cce-auto-sync=off"],
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
        ],
        error=None,
        returncode=0,
        pytest_sessions=[],
    )
    return dict(
        entry=entry,
        metadata=metadata,
        log="",
        returncode=0,
        timed_out=False,
        expected_abi="test-native-abi",
        device=0,
        aliases=aliases,
        run_dir=tmp_path,
        key=key,
    )


def _classify(evidence, case):
    """Refresh the authentic child log so each case changes only its intended evidence."""
    log = "\n".join(
        [
            f"PR1965_ENGINE_ABI {case['expected_abi']}",
            case["log"],
            f"PR1965_RESULT {json.dumps(case['metadata'])}",
        ],
    )
    return evidence.classify_result(**dict(case, log=log))


def test_success_without_an_instrumented_assertion_is_not_fabricated_precision(
    evidence,
    accepted_case,
):
    result = _classify(evidence, accepted_case)
    assert result["status"] == "PASS"
    assert not result["precision_verified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_abi", "another-build"),
        ("path", "examples/another/example.py"),
        ("variant", "legacy"),
        ("npu_device", 1),
        ("cache_disabled", False),
        ("original_args", ["--different-workload"]),
        ("compiled", 0),
        ("error", "caught top-level error"),
    ],
)
def test_metadata_mismatch_cannot_be_a_pass(evidence, accepted_case, field, value):
    accepted_case["metadata"][field] = value
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_a_caught_compilation_failure_is_not_hidden_by_a_later_success(evidence, accepted_case):
    metadata = accepted_case["metadata"]
    metadata["compile_attempts"] = 2
    metadata["kernels"].append(dict(index=2, error="compiler refused the second kernel"))
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


@pytest.mark.parametrize(
    "setting,value",
    [
        ("tl.ascend_sync_backend", "legacy"),
        ("tl.ascend_auto_sync_vs", False),
    ],
)
def test_wrong_compiler_configuration_is_not_accepted(evidence, accepted_case, setting, value):
    accepted_case["metadata"]["kernels"][0]["pass_configs"][setting] = value
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_a_later_bisheng_auto_sync_flag_overrides_the_earlier_off_flag(evidence, accepted_case):
    accepted_case["metadata"]["kernels"][0]["compile_flags"].append("--cce-auto-sync=on")
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_generated_source_must_belong_to_this_run(evidence, accepted_case, tmp_path):
    foreign = tmp_path / "outside-kernels.cc"
    foreign.write_text("old binary's source")
    accepted_case["metadata"]["kernels"][0]["source"] = str(foreign)
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_generated_source_changed_after_compilation_is_not_valid_evidence(evidence, accepted_case):
    source = Path(accepted_case["metadata"]["kernels"][0]["source"])
    source.write_text("// A different kernel replaced the compiled source\n")
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_script_cannot_swallow_a_failed_embedded_pytest_session(evidence, accepted_case):
    accepted_case["metadata"]["pytest_sessions"] = [
        dict(
            exitstatus=1,
            collected=2,
            passed=1,
            failed=1,
            skipped=0,
            failed_reports=[dict(nodeid="example::failed", when="call")],
        ),
    ]
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


@pytest.mark.parametrize(
    "message",
    [
        "Case 2: FAILED - incorrect result",
        "Some tests failed.",
        "[PRECISION_FAIL]",
        "Match: False",
        "Is output sorted (descending): False",
        "4-way merge: FAIL",
    ],
)
def test_explicit_sample_failure_messages_are_not_success(evidence, accepted_case, message):
    accepted_case["log"] = message
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_a_negative_test_name_is_not_itself_failure_evidence(evidence, accepted_case):
    accepted_case["log"] = "example::test_expected_failure PASSED"
    assert _classify(evidence, accepted_case)["status"] == "PASS"


def test_timeout_and_no_execution_are_distinct_from_pass_and_failure(evidence, accepted_case):
    timed_out = copy.deepcopy(accepted_case)
    timed_out.update(timed_out=True, returncode=-15)
    assert _classify(evidence, timed_out)["status"] == "TIMEOUT"
    accepted_case["metadata"].update(compile_attempts=0, compiled=0, launches=0, kernels=[])
    assert _classify(evidence, accepted_case)["status"] == "NO_EXECUTION"


def test_missing_child_record_cannot_reuse_an_apparent_zero_exit(evidence, accepted_case):
    accepted_case["metadata"] = {}
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


@pytest.mark.parametrize("field", ["assertion_failures", "launch_failures"])
def test_caught_runtime_failure_is_still_recorded(evidence, accepted_case, field):
    accepted_case["metadata"][field] = [{"message": "caught failure"}]
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_conflicting_configuration_aliases_cannot_hide_disabled_sync(evidence, accepted_case):
    accepted_case["metadata"]["kernels"][0]["pass_configs"]["tl.ascend_auto_sync"] = False
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_missing_or_duplicate_log_records_cannot_reuse_old_metadata(evidence, accepted_case):
    assert evidence.classify_result(**accepted_case)["status"] == "FAIL"
    accepted_case["log"] = "PR1965_ENGINE_ABI test-native-abi"
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


def test_pytest_entry_preserves_its_selection_and_complete_session(evidence, accepted_case):
    accepted_case["entry"].update(mode="pytest", pytest_markers="not slow")
    accepted_case["metadata"].update(
        mode="pytest",
        pytest_markers="not slow",
        assertions=1,
        pytest_sessions=[
            dict(
                exitstatus=0,
                collected=1,
                tests_collected=1,
                passed=1,
                failed=0,
                skipped=0,
                failed_reports=[],
                collection_reports=[],
                args=["-m", "not slow"],
            ),
        ],
    )
    result = _classify(evidence, accepted_case)
    assert result["status"] == "PASS" and result["precision_verified"]
    accepted_case["metadata"]["pytest_markers"] = ""
    assert _classify(evidence, accepted_case)["status"] == "FAIL"


@pytest.mark.parametrize("worker_fails", [False, True], ids=["all-pass", "one-failure"])
def test_controller_retries_only_the_new_subset_with_one_task_per_device(
    runner,
    monkeypatch,
    tmp_path,
    accepted_case,
    worker_fails,
):
    repo = _snapshot_repo(tmp_path / "repo")
    config = repo / "tilelang/transform/pass_config.py"
    config.parent.mkdir()
    config.write_text('class PassConfigKey:\n    TL_ASCEND_AUTO_SYNC = "tl.ascend_auto_sync"\n')
    entries = [
        _entry("examples/op/old_pass.py"),
        _entry(
            "examples/op/retry_first.py",
            device=0,
            args=["--shape", "17", "literal $(not-a-shell)"],
            timeout_seconds=300,
        ),
        _entry(
            "examples/op/retry_second.py",
            mode="pytest",
            device=1,
            args=["--second"],
            pytest_markers="not bench and not slow",
        ),
    ]
    for entry in entries:
        (repo / entry["path"]).write_text("pass\n")
    manifest = _write_json(tmp_path / "manifest.json", entries)
    statuses = ["PASS", "FAIL", "TIMEOUT"]
    historical_rows = [dict(entry, status=status) for entry, status in zip(entries, statuses)]
    previous = _write_json(
        tmp_path / "old-results.json",
        {"entries": historical_rows},
    )
    old_results = previous.read_bytes()
    native = tmp_path / "test-native.so"
    native.write_bytes(b"CPU fixture for native fingerprinting")
    output = tmp_path / "new-run"
    original_loader = runner.load_manifest
    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner, "load_manifest", lambda path: original_loader(path, repo=repo))
    lock = threading.Lock()
    active, peak_active = 0, 0
    executed = []

    def fake_process(command, *, log_path, env, timeout, manager):
        nonlocal active, peak_active
        if "--preflight" in command:
            log_path.write_text(
                json.dumps(
                    dict(
                        abi="test-native-abi",
                        z3_available=True,
                        device_count=1,
                        native_libraries=[str(native)],
                    ),
                ),
            )
            return 0, False
        selected = json.loads(Path(command[command.index("--entry-file") + 1]).read_text())
        entry = selected[int(command[command.index("--index") + 1])]
        device = int(command[command.index("--device") + 1])
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            executed.append((entry, device, timeout))
        try:
            time.sleep(0.05)
            key = runner.entry_key(entry["path"])
            source = output / "kernels" / f"{key}-001.cc"
            source.write_text("// CPU fake worker generated source\n")
            metadata = copy.deepcopy(accepted_case["metadata"])
            metadata.update(
                path=entry["path"],
                mode=entry["mode"],
                npu_device=device,
                native_libraries=[str(native)],
                original_args=entry.get("args", []),
                pytest_markers=entry.get("pytest_markers", "not bench"),
            )
            kernel = metadata["kernels"][0]
            kernel["source"] = str(source)
            kernel["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            if entry["mode"] == "pytest":
                metadata["pytest_sessions"] = [
                    dict(
                        exitstatus=0,
                        collected=1,
                        tests_collected=1,
                        passed=1,
                        failed=0,
                        skipped=0,
                        failed_reports=[],
                        collection_reports=[],
                        args=["-m", entry["pytest_markers"]],
                    ),
                ]
            if worker_fails and entry["path"] == entries[1]["path"]:
                metadata["error"] = "Synthetic child failure"
            _write_json(log_path.with_suffix(".json"), metadata)
            log_path.write_text(
                f"PR1965_ENGINE_ABI test-native-abi\nPR1965_RESULT {json.dumps(metadata)}\n",
            )
            return 0, False
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(runner, "run_process", fake_process)
    command = [
        "--manifest",
        str(manifest),
        "--failed-from",
        str(previous),
        "--output",
        str(output),
        "--devices",
        "0",
        "--timeout",
        "17",
        "--skip-build",
    ]
    assert runner.main(command) == int(worker_fails)
    assert peak_active == 1
    assert executed == [(entries[1], 0, 300), (entries[2], 0, 17)]
    assert previous.read_bytes() == old_results
    assert json.loads((output / "entries.json").read_text()) == entries[1:]
    result_path = output / "results.json"
    results = json.loads(result_path.read_text())
    assert results["state"] == "complete" and results["valid"]
    assert results["scope"] == sum(results["counts"].values()) == 2
    assert results["counts"]["PASS"] == 2 - int(worker_fails)
    assert results["counts"]["FAIL"] == int(worker_fails)
    assert [entry["path"] for entry in results["entries"]] == [
        entries[1]["path"],
        entries[2]["path"],
    ]
    assert [entry["timeout_limit"] for entry in results["entries"]] == [300, 17]
    assert all(entry["executed_device"] == 0 for entry in results["entries"])
    saved_results = result_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.main(command)
    assert result_path.read_bytes() == saved_results
    assert len(executed) == 2


def _process_running(pid):
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _descendant_command(pid_file, *, parent_exits=False):
    code = (
        "import json,os,pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid])); "
        + ("sys.exit(0)" if parent_exits else "time.sleep(600)")
    )
    return [sys.executable, "-c", code]


def _wait_stopped(pids):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if not any(_process_running(pid) for pid in pids):
            return
        time.sleep(0.02)
    assert not [pid for pid in pids if _process_running(pid)]


@pytest.mark.skipif(sys.platform != "linux", reason="process-group checks use /proc")
def test_timeout_terminates_parent_and_its_descendants(runner, tmp_path):
    pid_file = tmp_path / "processes.json"
    manager = runner.ProcessManager()
    returncode, timed_out = runner.run_process(
        _descendant_command(pid_file),
        log_path=tmp_path / "child.log",
        env=dict(os.environ),
        timeout=0.5,
        manager=manager,
    )
    assert timed_out and returncode != 0
    _wait_stopped(json.loads(pid_file.read_text()))


@pytest.mark.skipif(sys.platform != "linux", reason="process-group checks use /proc")
def test_normal_parent_exit_does_not_leak_a_background_descendant(runner, tmp_path):
    pid_file = tmp_path / "processes.json"
    returncode, timed_out = runner.run_process(
        _descendant_command(pid_file, parent_exits=True),
        log_path=tmp_path / "child.log",
        env=dict(os.environ),
        timeout=5,
        manager=runner.ProcessManager(),
    )
    assert returncode == 0 and not timed_out
    _wait_stopped(json.loads(pid_file.read_text()))


@pytest.mark.skipif(sys.platform != "linux", reason="process-group checks use /proc")
def test_cancellation_terminates_an_active_process_group(runner, tmp_path):
    pid_file = tmp_path / "processes.json"
    manager = runner.ProcessManager()
    results, errors = [], []

    def execute():
        try:
            results.append(
                runner.run_process(
                    _descendant_command(pid_file),
                    log_path=tmp_path / "child.log",
                    env=dict(os.environ),
                    timeout=30,
                    manager=manager,
                ),
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        until = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < until:
            time.sleep(0.02)
        assert pid_file.exists()
        manager.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
        assert results and results[0][0] != 0 and not results[0][1]
        _wait_stopped(json.loads(pid_file.read_text()))
    finally:
        manager.stop()
        thread.join(timeout=5)
