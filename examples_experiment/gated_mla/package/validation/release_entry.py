"""Single-shot final-package smoke/simulator entry; prepare never launches NPU work."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    return (
        {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(path.rglob("*")) if p.is_file()}
        if path.exists()
        else {}
    )


def fresh(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {path}")


def make_inputs(torch, tq, dtype, seed):
    """Same finite, head-specific fixture family as package tests; quantize on CPU."""
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def normal(shape, scale=1.0):
        return (torch.randn(shape, generator=rng, dtype=torch.float64) * scale).to(dtype)

    inputs = {
        "q_content": normal((1, tq, 96, 128)),
        "q_aux": normal((1, tq, 96, 64)),
        "latent_kv": normal((1, 128, 512)),
        "key_aux": normal((1, 128, 64)),
        "w_k": normal((96, 128, 512), 1 / math.sqrt(512)),
        "w_v": normal((96, 128, 512), 1 / math.sqrt(512)),
    }
    head = torch.arange(96, dtype=torch.float64).view(1, 1, 96, 1)
    token = torch.arange(tq, dtype=torch.float64).view(1, tq, 1, 1)
    channel = torch.linspace(-1, 1, 128, dtype=torch.float64).view(1, 1, 1, 128)
    gate = (((head % 9) - 4) * 0.5 + channel * 0.125 + token / 32).to(dtype)
    gate[:, :, 0, :16] = -20
    gate[:, :, -1, 96:] = 20
    inputs["gate_logits"] = gate
    return inputs


def selected_factories(runtime, case, dtype, target):
    """Mirror r17's pure public route selector; real forward must reuse each object."""
    from gmla._contracts import public_prefill_block_n
    from gmla._selection import selected_prefill128_bundle

    tq = 1 if case == "decode" else 128
    selected_n = public_prefill_block_n(1, tq, 96, 128, 1 / math.sqrt(192))
    bundle = selected_prefill128_bundle(target) if selected_n == 128 else runtime.kernel_bundle
    if bundle.block_n != selected_n:
        raise AssertionError("Diagnostic factory bundle disagrees with public route selector")
    residual = dtype == "bfloat16"
    qname = "q_absorb_residual" if residual else "q_absorb"
    qfactory = getattr(bundle.query_module if residual else bundle.projections, qname + "_kernel")
    kernels = {qname: qfactory(1, tq, 96, dtype, target=target)}
    # lru_cache distinguishes keyword insertion order. Keep the exact order
    # used by cube_decode/cube_prefill so forward reuses these very objects.
    if case == "decode":
        kernels["attention"] = bundle.decode_module.build_cube_attention(
            batch=1,
            heads=96,
            cache_capacity=128,
            block_m=bundle.block_m,
            block_n=bundle.block_n,
            dtype=dtype,
            target=target,
            has_ql_residual=residual,
            split_count=6,
            max_blocks=20,
        )
        kernels["merge"] = bundle.decode_module.build_split_merge(
            batch=1,
            heads=96,
            cache_capacity=128,
            block_m=bundle.block_m,
            block_n=bundle.block_n,
            split_count=6,
            max_blocks=20,
            target=target,
        )
    else:
        kernels["attention"] = bundle.prefill_module.build_cube_attention(
            batch=1,
            heads=96,
            query_capacity=tq,
            cache_capacity=128,
            block_m=bundle.block_m,
            block_n=bundle.block_n,
            dtype=dtype,
            target=target,
            max_blocks=20,
            scale=1 / math.sqrt(192),
            has_ql_residual=residual,
        )
    kernels["value_gate"] = bundle.projections.value_gate_kernel(1, tq, 96, dtype, target=target)
    return kernels, {
        "block_m": bundle.block_m,
        "block_n": bundle.block_n,
        "max_blocks": bundle.max_blocks,
        "selector": "gmla._contracts.public_prefill_block_n",
        "bundle": "selected_prefill128_bundle" if selected_n == 128 else "runtime.kernel_bundle",
    }


def native_inventory(tilelang):
    root = Path(tilelang.__file__).resolve().parent
    adapter = root / "jit/adapter/cython/.cycache"
    files = {str(p): sha(p) for p in sorted(adapter.rglob("*")) if p.is_file() and p.suffix in (".so", ".pyd")}
    maps = Path("/proc/self/maps")
    lines = maps.read_text().splitlines() if maps.exists() else []
    paths = sorted(
        {
            line.split()[-1]
            for line in lines
            if any(s in line for s in ("libtilelang", "libtvm", "libpython", "cython_wrapper")) and line.split()[-1].startswith("/")
        }
    )
    return {"cython_binaries": files, "loaded_libraries": {p: sha(Path(p)) for p in paths if Path(p).is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--package-root", type=Path, required=True, help="Directory containing gmla/")
    parser.add_argument("--case", choices=("decode", "prefill"), required=True)
    parser.add_argument("--target", choices=("pto", "ascendc"), default="pto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "run" and args.prepared_dir is None:
        parser.error("run requires --prepared-dir from a separate prepare invocation")
    if os.environ.get("TILELANG_CLEAR_CACHE", "0") != "0":
        parser.error("Prepared-cache validation requires TILELANG_CLEAR_CACHE=0")
    # Avoid the platform='auto' path probing a device during CPU compilation.
    # The simulator intercepts an ordinary A2 NPU binary; do not compile in a
    # different run mode between prepare and run.
    for name, value in (("TL_PLATFORM", "A2"), ("TL_RUN_MODE", "npu")):
        os.environ.setdefault(name, value)
        if os.environ[name] != value:
            parser.error(f"This A2 release entry requires {name}={value}")
    package = args.package_root.resolve()
    cache = args.cache_dir.resolve()
    os.environ["TILELANG_CACHE_DIR"] = str(cache)
    sys.path.insert(0, str(package))
    fresh(args.output)
    import torch
    import torch_npu  # Backend registration is allowed; device initialization is not.

    prepare_initialization = None
    if args.mode == "prepare":
        if not hasattr(torch.npu, "_initialized"):
            raise RuntimeError("Cannot audit the pinned torch_npu initialization flag")
        before_initialized = torch.npu._initialized
        if before_initialized is not False:
            raise AssertionError("NPU was initialized before prepare; use a fresh process")
        prepare_initialization = {"before": before_initialized}
    import gmla
    from gmla import reference
    from gmla.metadata import metadata_for_reference
    import tilelang
    from tilelang.env import is_cache_enabled

    if Path(gmla.__file__).resolve() != package / "gmla/__init__.py":
        raise RuntimeError("Wrong gmla import origin")
    if not is_cache_enabled() or Path(tilelang.cache.get_cache_dir()).resolve() != cache:
        raise RuntimeError("Requested isolated TileLang cache was not selected")
    torch.set_num_threads(4)
    sources = {str(p.relative_to(package)).replace("\\", "/"): sha(p) for p in sorted((package / "gmla").rglob("*.py"))}
    sources["validation_entry"] = sha(Path(__file__))
    tq = 1 if args.case == "decode" else 128
    geometry = dict(B=1, H=96, Tq=tq, physical_Tk=128, dtype=args.dtype, target=args.target)
    before = snapshot(cache)
    prepared = None
    if args.mode == "run":
        prepared = json.loads((args.prepared_dir / "entry.json").read_text())
        if prepared["source_sha256"] != sources or prepared["geometry"] != geometry:
            raise RuntimeError("Source/geometry changed since prepare")
        if prepared["cache_dir"] != str(cache) or not before or prepared["cache_files"] != before:
            raise RuntimeError("Prepared disk cache missing or changed")
    runtime = gmla.build_runtime(target=args.target)
    kernels, public_route = selected_factories(runtime, args.case, args.dtype, args.target)
    kernel_hashes = {}
    for name, kernel in kernels.items():
        source = kernel.get_kernel_source()
        if not re.search(r"\bmain_kernel\s*\(", source):
            raise RuntimeError(f"{name} no longer has the audited main_kernel launch name")
        (args.output / f"{name}.cpp").write_text(source, encoding="utf-8")
        kernel_hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    after = snapshot(cache)
    native = native_inventory(tilelang)
    if prepared is not None:
        if after != before or kernel_hashes != prepared["kernel_source_sha256"]:
            raise RuntimeError("Compilation/cache mutation in run process; discard this capture")
        if native != prepared["native"]:
            raise RuntimeError("Native runtime/adapter origin or bytes changed since prepare")
        if public_route != prepared["public_route"]:
            raise RuntimeError("Public route/configuration changed since prepare")
    manifest = dict(
        mode=args.mode,
        geometry=geometry,
        source_sha256=sources,
        package_root=str(package),
        cache_dir=str(cache),
        cache_files=after,
        kernel_source_sha256=kernel_hashes,
        kernel_targets={n: str(k.target) for n, k in kernels.items()},
        native=native,
        python=sys.version,
        executable=sys.executable,
        public_route=public_route,
        release="r1.7",
        compile_environment={"TL_PLATFORM": os.environ["TL_PLATFORM"], "TL_RUN_MODE": os.environ["TL_RUN_MODE"]},
        torch=torch.__version__,
        tilelang=tilelang.__version__,
        tilelang_file=tilelang.__file__,
        launch_order=list(kernels),
        expected_physical_launches=len(kernels),
        physical_launches_verified=False,
        launch_evidence="requires external simulator/profiler output",
        warmup_calls=0,
        application_calls=0,
        precision="not_run",
        scope="seven finite contiguous inputs; read-only preallocated cache; no cache append or H2D/D2H timing",
    )
    if args.mode == "prepare":
        after_initialized = torch.npu._initialized
        if after_initialized is not False:
            raise AssertionError("NPU initialized during prepare; discard its compilation report")
        prepare_initialization["after"] = after_initialized
        manifest["prepare_npu_initialization"] = prepare_initialization
        (args.output / "entry.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("GMLA_RELEASE_PREPARED_NO_KERNEL_LAUNCH", flush=True)
        return

    if not torch.npu.is_available():
        raise RuntimeError("NPU runtime required by this entry; no silent skips")
    torch.npu.set_device(args.device)
    cpu = make_inputs(torch, tq, getattr(torch, args.dtype), args.seed)
    rows = [[1, 128, 127]] if args.case == "decode" else [[128, 128, 0]]
    cpu_meta = gmla.prepare_metadata(rows, batch=1, query_capacity=tq, cache_capacity=128)
    kwargs = metadata_for_reference(cpu_meta)
    expected = reference.expanded_reference(**cpu, **kwargs)
    absorbed = reference.absorbed_reference(**cpu, **kwargs)
    torch.testing.assert_close(expected, absorbed, rtol=2e-12, atol=2e-12)
    device = {n: v.to(f"npu:{args.device}") for n, v in cpu.items()}
    metadata = cpu_meta.to(f"npu:{args.device}")
    layout = {
        n: {
            "shape": list(v.shape),
            "stride": list(v.stride()),
            "dtype": str(v.dtype),
            "contiguous": v.is_contiguous(),
            "format": int(torch_npu.get_npu_format(v)),
        }
        for n, v in {**device, "metadata": metadata}.items()
    }
    torch.npu.synchronize()
    print("GMLA_RELEASE_SINGLE_SHOT_READY " + json.dumps(list(kernels)), flush=True)
    with torch.inference_mode():
        output, intermediates, audit = runtime.forward(**device, metadata=metadata, return_intermediates=True)
    torch.npu.synchronize()
    if list(audit["kernels"]) != list(kernels):
        raise AssertionError("Unexpected launch registry/order (possible added tail or route)")
    for name, kernel in kernels.items():
        if audit["kernels"][name] is not kernel:
            raise AssertionError(f"{name} used a different compiled object than prepare selected")
    if snapshot(cache) != after:
        raise AssertionError("Disk compilation/cache write inside selected forward")
    expected_route = "c3_decode" if args.case == "decode" else "c4_prefill"
    if audit["selected_attention"] != expected_route:
        raise AssertionError("Wrong attention route")
    metrics = reference.assert_fp16_close(output.cpu(), expected)
    for name, value in device.items():
        torch.testing.assert_close(value.cpu(), cpu[name], rtol=0, atol=0)
    torch.testing.assert_close(metadata.cpu(), cpu_meta, rtol=0, atol=0)
    manifest.update(
        precision="pass",
        metrics=metrics,
        read_only="pass",
        metadata=cpu_meta.tolist(),
        seed=args.seed,
        input_layouts=layout,
        application_calls=1,
        selected_attention=audit["selected_attention"],
        compiled_object_identity="pass",
        device=torch.npu.get_device_name(args.device),
    )
    (args.output / "entry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GMLA_RELEASE_SINGLE_SHOT_PASS", flush=True)


if __name__ == "__main__":
    main()
