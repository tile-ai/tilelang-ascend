"""Automatic GMLA-only pytest integration; loaded by package/tests/conftest.py.

No global workflow flag is required. Other operators retain their environment,
cache directory and native compiler. CPU compilation is inside the pytest session.
"""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import pytest

from .cpu_compile_pool import verify_parent_load
from .registry_plugin import registry_from_items


def file_inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def file_stamps(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
    return result


def is_under(path, root):
    return Path(path).resolve().is_relative_to(root.resolve())


class ScopedGmlaCI:
    def __init__(self, config, package_root):
        self.config, self.package = config, Path(package_root).resolve()
        self.operator_root = self.package.parent
        self.active = False
        self.started = time.perf_counter()
        self.uid = getattr(config, "workerinput", {}).get("testrunuid") or os.environ.get("PYTEST_XDIST_TESTRUNUID") or uuid.uuid4().hex
        configured = os.environ.get("GMLA_CI_OUTPUT")
        base = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
        self.output = Path(configured).resolve() if configured else base / ("gmla-ci-" + self.uid)
        self.published = self.output / "compile/published_cache"
        self.workers = int(os.environ.get("GMLA_CI_COMPILE_WORKERS", "8"))
        if self.workers not in (4, 8):
            raise pytest.UsageError("GMLA_CI_COMPILE_WORKERS must be4 or8")

    def owns(self, item):
        return is_under(item.path, self.operator_root)

    @contextmanager
    def cache_scope(self):
        # Import in the original environment first, so other TileLang cache
        # classes also retain their original defaults. Only this scope redirects.
        env_names = ("TILELANG_CACHE_DIR", "TL_PLATFORM", "TL_RUN_MODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        original_env = {name: os.environ.get(name) for name in env_names}
        original_path = list(sys.path)
        import tilelang
        from tilelang.cache import get_cache_dir, set_cache_dir

        if Path(tilelang.__file__).resolve().parent.parent != self.runtime_root:
            raise pytest.UsageError(
                "pytest resolved a different TileLang runtime; use explicit -o pythonpath in an unbuilt staging checkout"
            )
        previous_cache = get_cache_dir()
        try:
            set_cache_dir(str(self.published))
            os.environ.update(TILELANG_CACHE_DIR=str(self.published), TL_PLATFORM="A2", TL_RUN_MODE="npu")
            yield
        finally:
            set_cache_dir(str(previous_cache))
            sys.path[:] = original_path
            for name, value in original_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def pytest_collection_finish(self, session):
        own_items = [item for item in session.items if self.owns(item)]
        registry = registry_from_items(own_items)
        if not registry["cases"]:
            return
        # Normal hosted execution prepares automatically. Plain collection stays
        # lightweight; CPU-only rehearsal explicitly opts into compilation.
        if self.config.option.collectonly and os.environ.get("GMLA_CI_COMPILE_ON_COLLECT") != "1":
            return
        spec = importlib.util.find_spec("tilelang")
        if spec is None or spec.origin is None:
            raise pytest.UsageError("selected built TileLang runtime is unavailable")
        self.runtime_root = Path(spec.origin).resolve().parent.parent
        requested_runtime = os.environ.get("GMLA_CI_RUNTIME_ROOT")
        if requested_runtime and Path(requested_runtime).resolve() != self.runtime_root:
            raise pytest.UsageError(
                "explicit runtime root does not match import resolution; fix -o pythonpath, not a hidden sys.path override"
            )
        package_hash = hashlib.sha256((self.package / "source_manifest.json").read_bytes()).hexdigest()
        signature_data = {
            "registry": registry,
            "package_manifest": package_hash,
            "runtime_root": str(self.runtime_root),
            "compiler": hashlib.sha256(Path(__file__).with_name("cpu_compile_pool.py").read_bytes()).hexdigest(),
        }
        signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        import fcntl

        self.output.mkdir(parents=True, exist_ok=True)
        wait_started = time.perf_counter()
        with (self.output / "compile.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            identity_file = self.output / "session_identity.json"
            if identity_file.exists():
                prior = json.loads(identity_file.read_text())
                if prior != {"uid": self.uid, "signature": signature}:
                    raise pytest.UsageError("existing GMLA output belongs to another session/source; cold output must be new")
            else:
                if (self.output / "compile").exists():
                    raise pytest.UsageError("existing kernel cache cannot establish a cold session")
                identity_file.write_text(json.dumps({"uid": self.uid, "signature": signature}))
            done = self.output / "published.json"
            if not done.exists():
                registry_path = self.output / "selected_registry.json"
                registry_path.write_text(json.dumps(registry, indent=2))
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("cpu_compile_pool.py")),
                    "--registry",
                    str(registry_path),
                    "--package-root",
                    str(self.package),
                    "--runtime-root",
                    str(self.runtime_root),
                    "--workers",
                    str(self.workers),
                    "--output",
                    str(self.output / "compile"),
                    "--skip-parent-load",
                ]
                with (self.output / "precompile.log").open("w") as log:
                    result = subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=900
                    )
                if result.returncode:
                    raise pytest.UsageError("GMLA CPU pool failed; see " + str(self.output / "precompile.log"))
                pool_report = json.loads((self.output / "compile/pool_report.json").read_text())
                if pool_report["status"] != "CPU_compile_and_publish_passed_parent_load_deferred":
                    raise pytest.UsageError("incomplete GMLA publication")
                done.write_text(json.dumps({"inventory": file_inventory(self.published), "stamps": file_stamps(self.published)}, indent=2))
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        baseline = json.loads(done.read_text())
        self.baseline_inventory, self.baseline_stamps = baseline["inventory"], baseline["stamps"]
        pool_report = json.loads((self.output / "compile/pool_report.json").read_text())
        load_started = time.perf_counter()
        with self.cache_scope():
            loaded = verify_parent_load(
                self.package,
                self.published,
                pool_report["results"],
                registry.get("legacy_roots", []),
                self.runtime_root,
                require_uninitialized=False,
            )
        self.active = True
        record = {
            "pid": os.getpid(),
            "automatic_hosted_integration": True,
            "package_manifest_sha256": package_hash,
            "selected_GMLA_tests": len(own_items),
            "selected_device_tests": registry["selected_npu_tests"],
            "other_operator_tests_excluded": len(session.items) - len(own_items),
            "compile_or_wait_seconds": load_started - wait_started,
            "parent_load_seconds": time.perf_counter() - load_started,
            "loaded": loaded,
            "environment_cache_and_path_restored": True,
            "output": str(self.output),
        }
        (self.output / f"ready_{os.getpid()}.json").write_text(json.dumps(record, indent=2))
        reporter = self.config.pluginmanager.getplugin("terminalreporter")
        if reporter:
            reporter.write_line("GMLA automatic cold CPU preparation ready: " + str(self.output))

    def delta(self):
        now = file_stamps(self.published)
        return sorted(name for name in self.baseline_stamps.keys() | now.keys() if self.baseline_stamps.get(name) != now.get(name))

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        if not self.active or not self.owns(item):
            yield
            return
        with self.cache_scope():
            from tilelang.jit.adapter.libgen import LibraryGenerator

            original = LibraryGenerator.compile_lib

            def reject_compile(generator, *args, **kwargs):
                event = {
                    "nodeid": item.nodeid,
                    "pid": os.getpid(),
                    "target": str(getattr(generator, "target", None)),
                    "source_sha256": hashlib.sha256(str(getattr(generator, "lib_code", "")).encode()).hexdigest(),
                }
                (self.output / f"unexpected_compile_{os.getpid()}_{time.time_ns()}.json").write_text(json.dumps(event))
                raise AssertionError("GMLA_UNPLANNED_NATIVE_COMPILE: " + str(event))

            LibraryGenerator.compile_lib = reject_compile
            try:
                yield
            finally:
                LibraryGenerator.compile_lib = original

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        if not self.active or not self.owns(item) or call.when not in ("call", "teardown"):
            return
        changed = self.delta()
        if changed:
            report = outcome.get_result()
            message = "GMLA published cache changed (including subprocess writes): " + str(changed)
            report.sections.append(("GMLA cache audit", message))
            if report.outcome != "failed":
                report.outcome, report.longrepr = "failed", message

    def pytest_sessionfinish(self, session, exitstatus):
        if not self.active:
            return
        after = file_inventory(self.published)
        changed = sorted(
            name for name in self.baseline_inventory.keys() | after.keys() if self.baseline_inventory.get(name) != after.get(name)
        )
        violations = sorted(p.name for p in self.output.glob("unexpected_compile_*.json"))
        if changed or violations:
            session.exitstatus = 1
        report = {
            "published_cache_unchanged": not changed,
            "changed_files": changed,
            "native_compile_violations": violations,
            "pytest_exitstatus": int(session.exitstatus),
            "collect_only": bool(self.config.option.collectonly),
            "plugin_elapsed_seconds": time.perf_counter() - self.started,
            "note": "normal default pytest activation; compiler cost is inside session; external wall still required for120s gate",
        }
        (self.output / f"session_audit_{os.getpid()}.json").write_text(json.dumps(report, indent=2))


def configure(config, package_root):
    existing = {line.split(":", 1)[0].strip() for line in config.getini("markers")}
    if "npu" not in existing:
        config.addinivalue_line("markers", "npu: requires a real NPU; absence is a failure when selected")
    current = config.pluginmanager.getplugin("gmla-local-cold-ci")
    if current is not None:
        if current.package != Path(package_root).resolve():
            raise pytest.UsageError("multiple GMLA package versions in one test session")
        return
    config.pluginmanager.register(ScopedGmlaCI(config, package_root), "gmla-local-cold-ci")
