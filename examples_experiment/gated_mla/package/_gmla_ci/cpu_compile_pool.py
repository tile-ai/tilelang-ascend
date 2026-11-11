"""CPU-only spawn ProcessPool proof; no tensor allocation or kernel invocation."""

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import warnings


WORKER = {}
REQUIRED_CACHE_FILES = ("kernel.cu", "wrapped_kernel.cu", "kernel_lib.so", "params.pkl", "auto_gm_idx.pkl")


def requests_for(case, block_n_selector=None):
    if case.get("legacy"):
        target, mode, visible = case["target"], case["legacy_mode"], case["visible"]
        specs = []
        if mode == "full":
            specs = [
                ("q_absorb_kernel", []),
                ("score_kernel", [visible, case.get("scale") or 1 / math.sqrt(192)]),
                ("softmax_kernel", []),
                ("latent_readout_kernel", [visible]),
                ("value_gate_kernel", []),
            ]
        elif mode == "zero":
            specs = [("zero_output_kernel", [])]
        elif mode == "latent":
            specs = [("latent_readout_kernel", [visible])]
        elif mode != "reject":
            raise ValueError("unknown legacy stage")
        return [dict(provider="legacy", factory=name, args=args, kwargs={"target": target}, cases=[case["id"]]) for name, args in specs]
    B, H, Q, K = (case[name] for name in ("B", "H", "Tq", "Tk"))
    dtype, target = case["dtype"], case.get("target", "pto")
    if any(type(x) is not int or x < 0 for x in (B, H, Q, K)) or min(B, H) < 1:
        raise ValueError("invalid registry shape")
    if dtype not in ("float16", "bfloat16") or target not in ("pto", "ascendc"):
        raise ValueError("unsupported registry dtype/target")
    stages = set(case["stages"])
    if not stages <= {"query", "value", "attention", "zero", "cache", "cache_init", "cache_append"}:
        raise ValueError("unknown stage")
    jobs = []

    def add(provider, factory, args=(), **kwargs):
        jobs.append({"provider": provider, "factory": factory, "args": list(args), "kwargs": kwargs, "cases": [case["id"]]})

    if Q == 0:
        return jobs
    if K == 0:
        add("empty_cache_module", "build_empty_cache_zero", batch=B, query_capacity=Q, heads=H, dtype=dtype, target=target)
        return jobs
    residual, packed = dtype == "bfloat16", case.get("layout", "separate") == "packed"
    if "query" in stages:
        name = "q_absorb" + ("_packed" if packed else "") + ("_residual" if residual else "") + "_kernel"
        add("query_module" if residual else "projections", name, (B, Q, H, dtype), target=target)
    if "value" in stages:
        add("projections", "value_gate_packed_kernel" if packed else "value_gate_kernel", (B, Q, H, dtype), target=target)
    scale = case.get("scale", 1 / math.sqrt(192))
    block_n = 64 if packed or block_n_selector is None else block_n_selector(B, Q, H, K, scale)
    if "attention" in stages:
        decode = Q == 1 and scale == 1 / math.sqrt(192)
        provider = "decode_module" if decode else "prefill_module"
        common = dict(batch=B, heads=H, cache_capacity=K, block_m=32, block_n=block_n, dtype=dtype, target=target, has_ql_residual=residual)
        if decode:
            add(provider, "build_cube_attention", **common, split_count=6, max_blocks=20)
            add(
                provider,
                "build_split_merge",
                batch=B,
                heads=H,
                cache_capacity=K,
                block_m=32,
                block_n=64,
                split_count=6,
                max_blocks=20,
                target=target,
            )
        else:
            add(provider, "build_cube_attention", **common, query_capacity=Q, max_blocks=20, scale=scale)
        extent = H if decode else Q * H
        if extent % 32:
            add(
                provider,
                "build_tail_triple" if residual else "build_tail_pair",
                batch=B,
                extent=extent,
                tile=32,
                dtype=dtype,
                target=target,
            )
        if K % block_n:
            add(provider, "build_tail_pair", batch=B, extent=K, tile=block_n, dtype=dtype, target=target)
    if stages & {"cache", "cache_init"}:
        add("gmla.kernels.cache", "initialize_cache_kernel", (B, K, dtype), target=target)
    if stages & {"cache", "cache_append"}:
        add("gmla.kernels.cache", "append_cache_kernel", (B, Q, K, dtype), target=target)
    return jobs


def selector_from_package(package):
    path = Path(package) / "gmla/_contracts.py"
    spec = importlib.util.spec_from_file_location("_gmla_ci_source_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "public_prefill_block_n", None)):
        raise ValueError("r17 public profile selector is missing; do not assume N64")
    return module.public_prefill_block_n


def plan(registry, block_n_selector=None):
    unique, count = {}, 0
    for case in registry["cases"]:
        for job in requests_for(case, block_n_selector):
            count += 1
            key = hashlib.sha256(json.dumps({k: v for k, v in job.items() if k != "cases"}, sort_keys=True).encode()).hexdigest()
            if key in unique:
                unique[key]["cases"].extend(job["cases"])
            else:
                unique[key] = dict(job, key=key)
    return list(unique.values()), count


def abi_snapshot():
    import platform
    import torch
    import torch_npu
    import tilelang

    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ("libtvm" in line or "libtilelang" in line) and "/" in line
        }
    )
    libraries = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths if Path(path).is_file()}
    if not libraries:
        raise AssertionError("native library provenance unavailable")
    return {
        "python": sys.version,
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "tilelang_file": tilelang.__file__,
        "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
        "libraries": libraries,
    }


def initialize(package, staging, legacy_roots=(), runtime_root=None):
    started = time.perf_counter()
    cache = Path(staging) / str(os.getpid())
    cache.mkdir(parents=True, exist_ok=False)
    os.environ.update(TILELANG_CACHE_DIR=str(cache), TL_PLATFORM="A2", TL_RUN_MODE="npu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for legacy_root in legacy_roots:
        sys.path.insert(0, legacy_root)
    sys.path.insert(0, package)
    if runtime_root:
        sys.path.insert(0, runtime_root)
    import torch
    from gmla import build_runtime

    assert not torch.npu.is_initialized()
    bundle = build_runtime().kernel_bundle
    assert not torch.npu.is_initialized()
    WORKER.update(bundle=bundle, torch=torch, cache=str(cache), initializer_seconds=time.perf_counter() - started, abi=abi_snapshot())


def factory(bundle, job):
    if job["provider"] == "legacy":
        return getattr(importlib.import_module("example_gated_mla_decode"), job["factory"])
    provider = importlib.import_module(job["provider"]) if job["provider"].startswith("gmla.") else getattr(bundle, job["provider"])
    return getattr(provider, job["factory"])


@contextmanager
def factory_warning_scope():
    """Only two known Python3.12/TVM AST notices during factory preparation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Attribute [ns] is deprecated and will be removed in Python 3\.14; use value instead$",
            category=DeprecationWarning,
            module=r"^tvm\.script\.parser\.core\.doc$",
        )
        yield


def compile_job(job):
    start = time.perf_counter()
    result = dict(job=job, pid=os.getpid(), cache=WORKER["cache"], initializer_seconds=WORKER["initializer_seconds"], abi=WORKER["abi"])
    try:
        assert not WORKER["torch"].npu.is_initialized()
        with factory_warning_scope():
            kernel = factory(WORKER["bundle"], job)(*job["args"], **job["kwargs"])
        assert str(kernel.target) == job["kwargs"]["target"]
        assert not WORKER["torch"].npu.is_initialized()
        result.update(
            status="compiled_CPU_only",
            actual_target=str(kernel.target),
            generated_source_sha256=hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest(),
        )
    except Exception:
        result.update(status="failed", traceback=traceback.format_exc())
    result["seconds"] = time.perf_counter() - start
    return result


def grouped_jobs(jobs):
    groups = {}
    for job in jobs:
        if job["factory"] in ("build_tail_pair", "build_tail_triple"):
            # Same-signature tail helpers from C3/C4 run in one worker, without
            # assuming their IR is identical. The real KernelCache decides reuse.
            key = json.dumps({k: job[k] for k in ("factory", "args", "kwargs")}, sort_keys=True)
        else:
            key = job["key"]
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def compile_group(jobs):
    return [compile_job(job) for job in jobs]


def publish(staging, published):
    records = {}
    for worker in sorted(staging.iterdir()):
        for source in sorted(worker.iterdir()):
            if not source.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", source.name):
                continue
            if not all((source / name).is_file() for name in REQUIRED_CACHE_FILES):
                raise AssertionError(f"incomplete kernel cache: {source}")
            hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in REQUIRED_CACHE_FILES}
            if source.name in records:
                # Fail rather than assuming different binary artifacts with the
                # same key are interchangeable; registry canonicalization is next.
                if hashes != records[source.name]["files"]:
                    raise AssertionError(f"cross-worker cache-key collision needs review: {source.name}")
                records[source.name]["duplicate_identical"] = True
                continue
            destination = published / source.name
            assert destination.resolve().is_relative_to(published.resolve())
            shutil.copytree(source, destination)
            records[source.name] = {"worker": worker.name, "files": hashes, "duplicate_identical": False}
    return records


def verify_parent_load(package, published, compiled, legacy_roots=(), runtime_root=None, require_uninitialized=True):
    if "tilelang" in sys.modules:
        from tilelang.cache import get_cache_dir

        assert Path(get_cache_dir()).resolve() == published.resolve(), "already loaded TileLang uses a different cache"
    os.environ.update(TILELANG_CACHE_DIR=str(published), TL_PLATFORM="A2", TL_RUN_MODE="npu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for legacy_root in legacy_roots:
        sys.path.insert(0, legacy_root)
    sys.path.insert(0, str(package))
    if runtime_root:
        sys.path.insert(0, str(runtime_root))
    import torch
    from gmla import build_runtime
    from tilelang.jit.adapter.libgen import LibraryGenerator

    before_initialized = bool(torch.npu.is_initialized())
    if require_uninitialized:
        assert not before_initialized
    bundle = build_runtime().kernel_bundle
    parent_abi = abi_snapshot()
    assert all(record["abi"] == parent_abi for record in compiled), "worker/parent ABI mismatch"
    original = LibraryGenerator.compile_lib

    def reject_recompile(*args, **kwargs):
        raise AssertionError("published-cache load attempted native recompilation")

    records = []
    LibraryGenerator.compile_lib = reject_recompile
    try:
        for record in compiled:
            job, start = record["job"], time.perf_counter()
            with factory_warning_scope():
                kernel = factory(bundle, job)(*job["args"], **job["kwargs"])
            digest = hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
            assert digest == record["generated_source_sha256"] and str(kernel.target) == record["actual_target"]
            assert bool(torch.npu.is_initialized()) == before_initialized
            records.append({"job_key": job["key"], "load_seconds": time.perf_counter() - start, "source_hash_identical": True})
    finally:
        LibraryGenerator.compile_lib = original
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--accept-empty-published", action="store_true")
    parser.add_argument("--skip-parent-load", action="store_true", help="Pytest parent will perform the audited load itself")
    args = parser.parse_args()
    started = time.perf_counter()
    registry = json.loads(args.registry.read_text())
    jobs, requested = plan(registry, selector_from_package(args.package_root))
    staging, published = args.output / "workers", args.output / "published_cache"
    if args.output.exists():
        accepted = (
            args.accept_empty_published
            and published.is_dir()
            and not any(published.iterdir())
            and list(args.output.iterdir()) == [published]
        )
        if not accepted:
            raise ValueError("output must be absent or explicitly accepted with only an empty published cache")
    staging.mkdir(parents=True)
    published.mkdir(exist_ok=True)
    report = {
        "status": "running",
        "workers": args.workers,
        "requested_factories": requested,
        "unique_factories": len(jobs),
        "scheduled_groups": len(grouped_jobs(jobs)),
        "jobs": jobs,
        "explicit_runtime_root": str(args.runtime_root) if args.runtime_root else None,
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "registry_coverage": registry.get("coverage", []),
        "legacy_source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for legacy_root in registry.get("legacy_roots", [])
            for p in Path(legacy_root).glob("*.py")
        },
        "package_manifest_sha256": hashlib.sha256((args.package_root / "source_manifest.json").read_bytes()).hexdigest(),
        "package_python_sha256": {
            str(p.relative_to(args.package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((args.package_root / "gmla").rglob("*.py"))
        },
        "kernel_cache_initially_empty": not any(published.iterdir()),
        "parallelism": "spawn ProcessPool; per-worker cache, no shared writes",
        "Cython_toolchain_cache": "existing runtime cache retained; not a clean native-library build",
        "NPU_execution": False,
        "results": [],
    }
    if args.plan_only:
        report["status"] = "plan_only"
    else:
        try:
            pool_start = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(
                    str(args.package_root),
                    str(staging),
                    tuple(registry.get("legacy_roots", [])),
                    str(args.runtime_root) if args.runtime_root else None,
                ),
            ) as pool:
                for future in as_completed([pool.submit(compile_group, group) for group in grouped_jobs(jobs)]):
                    report["results"].extend(future.result())
            report["pool_seconds_including_worker_startup"] = time.perf_counter() - pool_start
            if any(result["status"] != "compiled_CPU_only" for result in report["results"]):
                raise AssertionError("a CPU compile failed; do not publish partial cache")
            report["published_entries"] = publish(staging, published)
            verify_start = time.perf_counter()
            report["parent_cache_load"] = (
                []
                if args.skip_parent_load
                else verify_parent_load(
                    args.package_root, published, report["results"], registry.get("legacy_roots", []), args.runtime_root
                )
            )
            report["parent_cache_load_deferred_to_pytest"] = args.skip_parent_load
            report["parent_import_and_load_seconds"] = time.perf_counter() - verify_start
            report["status"] = (
                "CPU_compile_and_publish_passed_parent_load_deferred"
                if args.skip_parent_load
                else "CPU_compile_publish_and_no_recompile_load_passed"
            )
        except Exception:
            report.update(status="failed", traceback=traceback.format_exc())
    report["total_seconds"] = time.perf_counter() - started
    (args.output / "pool_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("status", "workers", "requested_factories", "unique_factories", "total_seconds")}))
    if report["status"] == "failed":
        print(report["traceback"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
