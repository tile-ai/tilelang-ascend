#!/usr/bin/env python3
# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Re-run the recorded Ascend example entries and report only this run's results."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from _ascend_examples_evidence import classify_result
from _ascend_examples_worker import atomic_json, entry_key

REPO = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("_ascend_examples_worker.py")
DEFAULT_MANIFEST = REPO / "ci/ascend_sync_examples.json"
STATUSES = ("PASS", "FAIL", "SKIP", "TIMEOUT", "NO_EXECUTION", "INTERRUPTED", "PENDING")
RETRY_STATUSES = {"FAIL", "TIMEOUT", "NO_EXECUTION", "INTERRUPTED", "PENDING"}
SOURCE_PATHS = (
    "src",
    "tilelang",
    "examples",
    "ci",
    "cmake",
    "include",
    "scripts/testing",
    "scripts/run_ascend_examples.sh",
    "set_env.sh",
    "install_ascend.sh",
    "CMakeLists.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "requirements-build.txt",
    "requirements-test.txt",
    "requirements-dev.txt",
    "requirements-lint.txt",
    ".gitmodules",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repo: Path, *args: str, optional: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        if optional:
            return ""
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode(errors="surrogateescape")


def _source_files(repo: Path) -> set[Path]:
    listing = git_output(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *SOURCE_PATHS,
    )
    return {
        repo / name
        for name in listing.split("\0")
        if name
        and (repo / name).is_file()
        and not {"__pycache__", ".cycache"}.intersection(Path(name).parts)
        and Path(name).suffix not in {".pyc", ".pyo"}
    }


def _submodules(repo: Path) -> dict:
    result = {}
    for record in git_output(repo, "ls-files", "--stage", "-z").split("\0"):
        if not record or not record.startswith("160000 "):
            continue
        _, name = record.split("\t", 1)
        path = repo / name
        head = git_output(path, "rev-parse", "HEAD", optional=True) if path.exists() else ""
        patch = git_output(path, "diff", "--binary", "HEAD", optional=True) if head else ""
        untracked = (
            git_output(
                path,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                optional=True,
            )
            if head
            else ""
        )
        untracked_files = {}
        for item in untracked.split("\0"):
            if item and (path / item).is_file():
                untracked_files[item] = digest(path / item)
        result[name] = {
            "head": head.strip(),
            "patch_sha256": hashlib.sha256(patch.encode(errors="surrogateescape")).hexdigest(),
            "untracked": untracked_files,
        }
    return result


@dataclass
class SourceSnapshot:
    repo: Path
    extra_files: tuple[Path, ...]
    files: dict[str, str]
    head: str
    submodules: dict

    @classmethod
    def capture(cls, repo: Path, extra_files: Iterable[Path] = ()) -> SourceSnapshot:
        repo = repo.resolve()
        extra = tuple(Path(path).resolve() for path in extra_files)
        paths = _source_files(repo) | set(extra)
        missing = [str(path) for path in extra if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing snapshot inputs: {missing}")
        files = {str(path.resolve()): digest(path) for path in sorted(paths)}
        return cls(
            repo,
            extra,
            files,
            git_output(repo, "rev-parse", "HEAD", optional=True).strip(),
            _submodules(repo),
        )

    def as_dict(self) -> dict:
        return {"files": self.files, "head": self.head, "submodules": self.submodules}

    def check(self) -> dict:
        current = SourceSnapshot.capture(self.repo, ())
        for path in self.extra_files:
            if path.is_file():
                current.files[str(path)] = digest(path)
        old_paths, new_paths = set(self.files), set(current.files)
        changed = sorted(
            (path for path in old_paths & new_paths if self.files[path] != current.files[path]),
        )
        added, removed = sorted(new_paths - old_paths), sorted(old_paths - new_paths)
        same_head = current.head == self.head
        same_submodules = current.submodules == self.submodules
        return {
            "valid": not changed and not added and not removed and same_head and same_submodules,
            "files_checked": len(self.files),
            "changed": changed,
            "added": added,
            "removed": removed,
            "head_preserved": same_head,
            "submodules_preserved": same_submodules,
        }


def load_manifest(path: Path, repo: Path = REPO) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("entries")
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest must contain a nonempty entry list")
    seen = set()
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Every manifest entry requires a path")
        relative = Path(entry["path"])
        target = (repo / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to((repo / "examples").resolve())
            or target.suffix != ".py"
            or not target.is_file()
        ):
            raise ValueError(f"Invalid or missing example path: {entry['path']}")
        if entry["path"] in seen:
            raise ValueError(f"Duplicate manifest path: {entry['path']}")
        seen.add(entry["path"])
        if entry.get("mode") not in {"script", "pytest", "skip"}:
            raise ValueError(f"Invalid execution mode: {entry['path']}")
        if type(entry.get("device")) is not int or entry["device"] < 0:
            raise ValueError(f"Invalid manifest device: {entry['path']}")
        arguments = entry.get("args", [])
        if not isinstance(arguments, list):
            raise ValueError(f"Example arguments must be strings: {entry['path']}")
        if any(not isinstance(value, str) for value in arguments):
            raise ValueError(f"Example arguments must be strings: {entry['path']}")
        if not isinstance(entry.get("pytest_markers", "not bench"), str):
            raise ValueError(f"Invalid pytest selection: {entry['path']}")
        if "timeout_seconds" in entry:
            value = entry["timeout_seconds"]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid timeout: {entry['path']}")
        if entry["mode"] == "skip":
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Skipped entry requires a reason: {entry['path']}")
    return data


def parse_devices(text: str) -> list[int]:
    try:
        devices = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated NPU IDs, such as 0,1") from error
    if not devices or any(value < 0 for value in devices) or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("NPU IDs must be nonnegative and distinct")
    return devices


def select_entries(
    entries: list[dict],
    only: list[str] | None = None,
    failed_from: Path | None = None,
) -> list[dict]:
    if only is not None and failed_from is not None:
        raise ValueError("Use either --only or --failed-from")
    if only is not None:
        wanted = set(only)
    elif failed_from is not None:
        previous = json.loads(failed_from.read_text())
        if isinstance(previous, dict):
            previous = previous.get("entries", previous.get("files"))
        if not isinstance(previous, list):
            raise ValueError("Previous results require entries/files or a result list")
        wanted = {row["path"] for row in previous if row.get("status") in RETRY_STATUSES}
    else:
        return list(entries)
    missing = wanted - {entry["path"] for entry in entries}
    if missing:
        raise ValueError(f"Selection is not in the manifest: {sorted(missing)}")
    return [entry for entry in entries if entry["path"] in wanted]


def create_run_directory(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = REPO / "build/ascend-examples" / f"{stamp}-{os.getpid()}"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "kernels"):
        (path / name).mkdir()
    return path


class ProcessManager:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.processes: dict[int, subprocess.Popen] = {}

    @staticmethod
    def terminate(process: subprocess.Popen) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        # A finished parent can still have live grandchildren in its group.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def stop(self) -> None:
        self.cancelled.set()
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            self.terminate(process)


def run_process(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    timeout: float,
    manager: ProcessManager,
) -> tuple[int, bool]:
    if manager.cancelled.is_set():
        return -signal.SIGTERM, False
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with manager.lock:
            manager.processes[process.pid] = process
        timed_out = False
        try:
            if manager.cancelled.is_set():
                manager.terminate(process)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                manager.terminate(process)
                code = process.returncode
            return code, timed_out
        finally:
            manager.terminate(process)
            with manager.lock:
                manager.processes.pop(process.pid, None)


def config_aliases() -> dict[str, str]:
    tree = ast.parse((REPO / "tilelang/transform/pass_config.py").read_text())
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PassConfigKey":
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and isinstance(item.value, ast.Constant)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value.value, str)
                ):
                    aliases[f"PassConfigKey.{item.targets[0].id}"] = item.value.value
    if not aliases:
        raise RuntimeError("Cannot read the current PassConfigKey definitions")
    return aliases


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1"))
    parser.add_argument("--timeout", type=positive, default=120, help="Default entry timeout")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--skip-build", action="store_true", help="Use the existing native build")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no imports/build/NPU")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--only", nargs="+", metavar="EXAMPLE_PATH")
    select.add_argument("--failed-from", type=Path, metavar="RESULTS_JSON")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


class RunReport:
    def __init__(self, directory: Path, entries: list[dict], context: dict) -> None:
        self.directory, self.entries, self.context = directory, entries, context
        self.rows = {}
        for entry in entries:
            if entry["mode"] == "skip":
                self.rows[entry["path"]] = dict(entry, status="SKIP", diagnostic=entry["reason"])
        self.lock = threading.Lock()

    def write(self, state: str, integrity: dict | None = None, error: str | None = None) -> dict:
        rows = []
        for entry in self.entries:
            rows.append(self.rows.get(entry["path"], dict(entry, status="PENDING")))
        counts = Counter(row["status"] for row in rows)
        result = {
            "state": state,
            "valid": state == "complete" and bool(integrity and integrity["valid"]),
            "scope": len(self.entries),
            "counts": {name: counts[name] for name in STATUSES},
            "context": self.context,
            "source_integrity": integrity,
            "error": error,
            "entries": rows,
        }
        atomic_json(self.directory / "results.json", result)
        lines = [
            "# Ascend examples 复测",
            "",
            f"状态：**{state}**；有效完整结果：**{result['valid']}**。",
            "",
            f"本轮选择 {len(rows)} 项；不累加历史通过或失败。",
            "",
            "| 状态 | 数量 |",
            "|---|---:|",
            *[f"| {name} | {counts[name]} |" for name in STATUSES],
            "",
            "PASS 要求本轮完整入口成功、编译和调用证据及配置/ABI核验；",
            "断言计数为 0 的 PASS 仅表示原入口完成，不额外宣称数值断言覆盖。",
            "SKIP 沿用清单中的明确原因；NO_EXECUTION 表示没有记录到 TileLang 调用。",
            "",
        ]
        if error:
            lines.extend(["", f"运行错误：{_markdown(error)}", ""])
        lines.extend(
            [
                "| 入口 | 状态 | 设备 | 编译/调用/断言 | 日志 | 诊断 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in rows:
            key = entry_key(row["path"])
            has_log = (self.directory / "logs" / f"{key}.log").exists()
            log = f"[log](logs/{key}.log)" if has_log else ""
            counters = "/".join(
                (str(row.get(name, 0)) for name in ("compiled", "launches", "assertions")),
            )
            fields = (
                row["path"],
                row["status"],
                row.get("executed_device", ""),
                counters,
                log,
                row.get("diagnostic", "")[:350],
            )
            lines.append("| " + " | ".join(map(_markdown, fields)) + " |")
        text = "\n".join(lines) + "\n"
        temporary = self.directory / "summary.md.tmp"
        temporary.write_text(text)
        temporary.replace(self.directory / "summary.md")
        return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    entries = select_entries(manifest, args.only, args.failed_from)
    devices = {}
    for entry in entries:
        devices[entry["device"]] = args.devices[entry["device"] % len(args.devices)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest.resolve()),
                    "manifest_entries": len(manifest),
                    "selected": len(entries),
                    "modes": dict(Counter(e["mode"] for e in entries)),
                    "device_mapping": devices,
                    "build": not args.skip_build,
                    "default_timeout_seconds": args.timeout,
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    directory = create_run_directory(args.output)
    print(f"运行目录：{directory}", flush=True)
    entry_file = directory / "entries.json"
    atomic_json(entry_file, entries)
    context = {
        "manifest": str(args.manifest.resolve()),
        "manifest_entries": len(manifest),
        "manifest_sha256": digest(args.manifest),
        "device_mapping": devices,
        "default_timeout_seconds": args.timeout,
        "python": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    report = RunReport(directory, entries, context)
    manager = ProcessManager()
    snapshot = None
    state, error = "running", None
    report.write(state)
    environment = dict(os.environ, ACL_OP_INIT_MODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2")
    executor = None
    try:
        executable = [entry for entry in entries if entry["mode"] != "skip"]
        if executable:
            if not args.skip_build:
                if not (REPO / "build/CMakeCache.txt").is_file():
                    message = "Missing configured build/. Run bash install_ascend.sh first."
                    raise RuntimeError(message)
                print("增量构建；输出保存到 build.log", flush=True)
                code, timed_out = run_process(
                    ["cmake", "--build", str(REPO / "build"), "--parallel", str(args.jobs)],
                    log_path=directory / "build.log",
                    env=environment,
                    timeout=3600,
                    manager=manager,
                )
                if code or timed_out:
                    raise RuntimeError("Native build failed; see build.log")
            code, timed_out = run_process(
                [sys.executable, str(WORKER), "--preflight"],
                log_path=directory / "preflight.log",
                env=environment,
                timeout=180,
                manager=manager,
            )
            if code or timed_out:
                message = "Environment preflight failed; see preflight.log (CANN/torch_npu)."
                raise RuntimeError(message)
            lines = (directory / "preflight.log").read_text(errors="replace").splitlines()
            preflight = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
            context["runtime"] = preflight
            if not preflight["z3_available"] or not preflight["abi"]:
                raise RuntimeError("The upstream Z3 proof service is not available")
            requested = {devices[entry["device"]] for entry in executable}
            if max(requested) >= preflight["device_count"]:
                raise RuntimeError("Requested NPU is unavailable; use --devices 0 for one NPU")
            native = [Path(path) for path in preflight["native_libraries"]]
            if not native:
                raise RuntimeError("Cannot identify the loaded native library")
        else:
            preflight, native = {"abi": None}, []
        build_config_names = ("config.cmake", "build/config.cmake", "build/CMakeCache.txt")
        build_inputs = []
        for name in build_config_names:
            if (REPO / name).is_file():
                build_inputs.append(REPO / name)
        selected_sources = [REPO / entry["path"] for entry in entries]
        extra = [
            args.manifest,
            entry_file,
            WORKER,
            Path(__file__),
            *native,
            *build_inputs,
            *selected_sources,
        ]
        snapshot = SourceSnapshot.capture(REPO, extra)
        atomic_json(directory / "source_snapshot.json", snapshot.as_dict())
        atomic_json(directory / "run_context.json", context)
        aliases = config_aliases()

        def run_group(group: list[tuple[int, dict]], device: int) -> None:
            for index, entry in group:
                if manager.cancelled.is_set():
                    return
                key = entry_key(entry["path"])
                log_path = directory / "logs" / f"{key}.log"
                started = time.monotonic()
                timeout = entry.get("timeout_seconds", args.timeout)
                command = [
                    sys.executable,
                    str(WORKER),
                    "--entry-file",
                    str(entry_file),
                    "--index",
                    str(index),
                    "--run-dir",
                    str(directory),
                    "--device",
                    str(device),
                    "--expected-abi",
                    preflight["abi"],
                ]
                print(f"RUN device={device} {entry['path']}", flush=True)
                code, timed_out = run_process(
                    command,
                    log_path=log_path,
                    env=environment,
                    timeout=timeout,
                    manager=manager,
                )
                metadata_path = log_path.with_suffix(".json")
                try:
                    metadata = json.loads(metadata_path.read_text())
                    if not isinstance(metadata, dict):
                        metadata = {}
                except (OSError, ValueError):
                    metadata = {}
                if manager.cancelled.is_set():
                    outcome = {"status": "INTERRUPTED", "diagnostic": "Run interrupted"}
                else:
                    outcome = classify_result(
                        entry,
                        metadata,
                        log_path.read_text(errors="replace"),
                        code,
                        timed_out,
                        preflight["abi"],
                        device,
                        aliases,
                        directory,
                        key,
                    )
                    if outcome["status"] == "PASS":
                        libraries = metadata.get("native_libraries", [])
                        try:
                            loaded = {str(Path(path).resolve()) for path in libraries}
                            expected = {str(path.resolve()) for path in native}
                        except (TypeError, ValueError, OSError):
                            loaded, expected = set(), {"invalid native library record"}
                        if loaded != expected:
                            outcome.update(
                                status="FAIL",
                                diagnostic="Child loaded a different native library from preflight",
                                precision_verified=False,
                                precision_validation="not_an_entry_pass",
                            )
                row = dict(entry)
                row.update(outcome)
                counter_names = ("compile_attempts", "compiled", "launches", "assertions")
                counters = {name: metadata.get(name, 0) for name in counter_names}
                row.update(
                    executed_device=device,
                    returncode=code,
                    seconds=round(time.monotonic() - started, 2),
                    timeout_limit=timeout,
                    log=str(log_path),
                    metadata=str(metadata_path),
                    **counters,
                )
                with report.lock:
                    report.rows[entry["path"]] = row
                    report.write("running")
                    progress = f"[{len(report.rows)}/{len(entries)}] {row['status']}"
                    print(f"{progress} {entry['path']} ({row['seconds']}s)", flush=True)

        groups = {}
        for index, entry in enumerate(entries):
            if entry["mode"] != "skip":
                groups.setdefault(devices[entry["device"]], []).append((index, entry))
        executor = ThreadPoolExecutor(max_workers=max(len(groups), 1))
        futures = [executor.submit(run_group, group, device) for device, group in groups.items()]
        for future in as_completed(futures):
            future.result()
        state = "complete"
    except KeyboardInterrupt:
        state, error = "interrupted", "Interrupted by user; active process groups terminated"
        manager.stop()
    except Exception as exception:
        state, error = "error", f"{type(exception).__name__}: {exception}"
        manager.stop()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        try:
            integrity = snapshot.check() if snapshot is not None else {"valid": False}
        except Exception as exception:
            integrity = {"valid": False, "error": f"{type(exception).__name__}: {exception}"}
        if state == "complete" and not integrity["valid"]:
            state, error = "invalid", "Source/native inputs changed during the run"
        context["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result = report.write(state, integrity, error)
        print(json.dumps(result["counts"], ensure_ascii=False), flush=True)
        print(f"结果：{directory / 'results.json'}\n报告：{directory / 'summary.md'}", flush=True)
        if error:
            print(error, file=sys.stderr)
    if state == "interrupted":
        return 130
    if not result["valid"]:
        return 2
    return int(any(result["counts"][name] for name in ("FAIL", "TIMEOUT", "NO_EXECUTION")))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError) as exception:
        print(f"Error: {exception}", file=sys.stderr)
        raise SystemExit(2) from exception
