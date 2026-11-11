"""Load only this checkout's performance tools without changing import paths."""

import hashlib
import importlib.util
from pathlib import Path
import sys


TOOLS = frozenset(("paired_timing", "paired_benchmark", "host_decode_diagnostic", "run_paired_suite"))


def load_perf_tool(name):
    if name not in TOOLS:
        raise ValueError("Unknown GMLA performance test module: " + str(name))
    package = Path(__file__).resolve().parents[1]
    path = package / "benchmark" / (name + ".py")
    identity = hashlib.sha256(str(package).encode()).hexdigest()[:16]
    module_name = "_gmla_perf_cpu_" + identity + "_" + name
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("GMLA performance module identity collision")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
