"""CPU source/import audit or one real Triton gate call; never pretends CPU is NPU."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-check", "npu-probe"), default="cpu-check")
    parser.add_argument("--isolated-site", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=12288)
    parser.add_argument("--gate-prefix", type=int, default=0)
    parser.add_argument(
        "--preload-gmla", action="store_true", help="Load installed candidate runtime before NPU operations to check coexistence"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.hidden) < 1 or args.gate_prefix < 0:
        parser.error("rows/hidden must be positive and gate-prefix nonnegative")
    if args.output.exists():
        parser.error("output must be a new directory")
    if os.environ.get("TRITON_INTERPRET", "0") != "0":
        raise RuntimeError("A real compiled Triton kernel is required; interpreter mode refused")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    original = (root / "original/flash_attention_output.py.txt").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()
    if blob != manifest["expected_git_blob"] or sha(root / "isolated_gate.py") != manifest["isolated_sha256"]:
        raise RuntimeError("Pinned original or isolated gate source bytes changed")
    tree = ast.parse((root / "isolated_gate.py").read_text())
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, expected in manifest["unchanged_function_ast_sha256"].items():
        if hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() != expected:
            raise RuntimeError(f"Official function AST changed: {name}")

    import torch
    import torch_npu

    before = torch.npu._initialized
    if before is not False:
        raise RuntimeError("Use a fresh process; NPU was initialized before this probe")
    import triton
    import isolated_gate as gate

    isolated_site = args.isolated_site.resolve()
    for module in (torch_npu, triton):
        if not Path(module.__file__).resolve().is_relative_to(isolated_site):
            raise RuntimeError(f"Wrong isolated package origin: {module.__file__}")
    if importlib.metadata.version("torch-npu") != "2.7.1.post11" or importlib.metadata.version("triton-ascend") != "3.2.2.post2":
        raise RuntimeError("This probe pins the documented A2/CANN9.2/Torch2.7.1 package pairing")
    report = {
        "mode": args.mode,
        "status": "running",
        "scope": "source-derived isolated official gate; vLLM warmup/host dispatcher not executed",
        "official_git_blob": blob,
        "official_source_sha256": sha(root / "original/flash_attention_output.py.txt"),
        "isolated_sha256": sha(root / "isolated_gate.py"),
        "source_manifest_sha256": sha(root / "source_manifest.json"),
        "function_ast_checks": "pass",
        "python": sys.version,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_file": torch_npu.__file__,
        "triton_ascend": importlib.metadata.version("triton-ascend"),
        "triton_file": triton.__file__,
        "npu_initialized_before": before,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "launcher_difference": manifest["launcher_difference"],
        "actual_device_kernel_execution": False,
    }
    if args.mode == "cpu-check":
        for n, block in ((1, 512), (8192, 512), (8193, 2048), (65535, 2048), (65536, 4096), (1048575, 4096), (1048576, 8192)):
            assert gate._flash_attention_gate_block_size(n) == block
        empty = torch.empty((0, 128), dtype=torch.float16)
        assert gate.flash_attention_gate(empty, empty) is empty
        rejected = 0
        for projected, gates in (
            (torch.empty(2), torch.empty(2)),
            (torch.empty((2, 3)), torch.empty((2, 4))),
            (torch.empty((2, 3), dtype=torch.float16), torch.empty((2, 3), dtype=torch.bfloat16)),
            (torch.empty((2, 6))[:, ::2], torch.empty((2, 3))),
        ):
            try:
                gate.flash_attention_gate(projected, gates)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Original metadata guard did not reject invalid input")
        assert rejected == 4 and torch.npu._initialized is False and gate._NUM_VECTORCORE == -1
        report.update(
            status="cpu_source_import_parameter_checks_passed",
            rejected_parameter_cases=rejected,
            npu_initialized_after=False,
            kernel_launches=0,
        )
    else:
        if args.preload_gmla:
            import gmla

            _ = gmla.build_runtime().kernel_bundle
            report["preloaded_candidate"] = {"file": gmla.__file__, "sources": gmla.package_sources()}
            assert torch.npu._initialized is False
        torch.npu.set_device(0)
        dtype = getattr(torch, args.dtype)
        rng = torch.Generator(device="cpu").manual_seed(0)
        cpu_projected = torch.randn((args.rows, args.hidden), generator=rng, dtype=torch.float32).to(dtype)
        cpu_gate_storage = torch.randn((args.rows, args.hidden + args.gate_prefix), generator=rng, dtype=torch.float32).to(dtype)
        cpu_gate = cpu_gate_storage[:, args.gate_prefix :]
        # LP sigmoid output and LP product are the actual source-level rounding contract.
        expected = cpu_projected * torch.sigmoid(cpu_gate)
        with torch.no_grad():
            projected, storage = cpu_projected.npu(), cpu_gate_storage.npu()
            gates = storage[:, args.gate_prefix :]
            pointer = projected.data_ptr()
            result = gate.flash_attention_gate(projected, gates)
            torch.npu.synchronize()
            assert result is projected and result.data_ptr() == pointer
            actual = result.cpu()
            torch.testing.assert_close(actual.double(), expected.double(), rtol=0.01, atol=0.002)
            error = (actual.double() - expected.double()).abs()
            normalized = float(error.max()) / max(float(expected.double().abs().max()), 1e-9)
            assert normalized <= 0.005
            torch.testing.assert_close(storage.cpu(), cpu_gate_storage, rtol=0, atol=0)
        block = gate._flash_attention_gate_block_size(result.numel())
        tasks = (
            triton.cdiv(result.numel(), block)
            if projected.stride(0) == gates.stride(0) == args.hidden
            else args.rows * triton.cdiv(args.hidden, block)
        )
        report.update(
            status="real_triton_gate_precision_passed",
            actual_device_kernel_execution=True,
            dtype=args.dtype,
            shape=[args.rows, args.hidden],
            gate_prefix=args.gate_prefix,
            projected_stride=list(projected.stride()),
            gate_stride=list(gates.stride()),
            block=block,
            tasks=tasks,
            grid=[min(tasks, gate.get_vectorcore_num())],
            vector_core_num=gate.get_vectorcore_num(),
            max_abs=float(error.max()),
            max_normalized=normalized,
            bitwise_equal=bool(torch.equal(actual, expected)),
            in_place="pass",
            gate_storage_read_only="pass",
            timing="not measured",
        )
    (args.output / "gate_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VLLM_SOURCE_GATE_" + report["status"].upper(), flush=True)


if __name__ == "__main__":
    main()
