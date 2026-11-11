import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_selected_sources_and_all_retained_definition_asts():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["selected_sources"]:
        path = ROOT / entry["destination"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"], path
        nodes = {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, digest in entry["definition_ast_sha256"].items():
            assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == digest, (path, name)


def test_import_and_construction_do_not_load_kernel_runtime():
    # Torch may automatically register torch_npu on the installed ARM stack.
    # Compare the package with that Torch-only baseline; registration is not
    # device initialization. is_initialized() reads backend state only.
    code = """
import json
import sys
import torch


def state():
    npu = getattr(torch, "npu", None)
    if npu is not None:
        assert callable(getattr(npu, "is_initialized", None)), "Cannot audit NPU initialization state"
    return {
        "tilelang_modules": sorted(name for name in sys.modules if name == "tilelang" or name.startswith("tilelang.")),
        "torch_npu_modules": sorted(name for name in sys.modules if name == "torch_npu" or name.startswith("torch_npu.")),
        "npu_initialized": False if npu is None else bool(npu.is_initialized()),
    }


baseline = state()
assert not baseline["tilelang_modules"], "Torch-only baseline unexpectedly loaded TileLang"
assert not baseline["npu_initialized"], "Torch-only baseline already initialized an NPU"
import gmla
runtime = gmla.build_runtime()
after = state()
assert runtime.target == "pto"
assert not after["tilelang_modules"], "Package construction loaded TileLang"
assert not after["npu_initialized"], "Package construction initialized an NPU"
assert after == baseline, {"baseline": baseline, "after": after}
print(json.dumps({"baseline": baseline, "after": after, "target": runtime.target}))
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)


def test_original_failure_fixture_ast_preserved():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    assert any(
        e["destination"] == "tests/fixture_decode.py" and "make_inputs" in e["definition_ast_sha256"] for e in manifest["selected_sources"]
    )


def test_127_128_129_query_and_value_plans_accept_tail_capacity():
    for filename in ("query.py", "value.py"):
        path = ROOT / "gmla/kernels" / filename
        tree = ast.parse(path.read_text())
        plan = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan")
        scope = {}
        exec(compile(ast.Module(body=[plan], type_ignores=[]), str(path), "exec"), scope)
        for tq in (127, 128, 129):
            result = scope["plan"](1, tq)
            assert result["Tq"] == tq and result["tile_m"] == 32 and result["blocks"] == 20
