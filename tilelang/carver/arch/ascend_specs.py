"""Exact A3 SDK resources and the budgets used by the Ascend backend."""

from __future__ import annotations

import configparser
import hashlib
import os
from pathlib import Path
import platform

A3_SKUS = ("Ascend910_9392", "Ascend910_9382")

# Match AscendMemoryPlanning and CodeGenTileLangAscend's A2/A3 arenas.
# L1 and UB each retain the backend's 256-byte reservation.
BACKEND_CAPACITIES = {
    "L0A": 65536,
    "L0B": 65536,
    "L0C": 131072,
    "L1": 524032,
    "UB": 196352,
}


def a3_sku(name: str | None) -> str | None:
    """Recognize exact supported SKUs without assigning a family default."""
    if name is not None:
        for sku in A3_SKUS:
            if name.upper() == sku.upper():
                return sku
    return None


def _platform_file(sku: str, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve(strict=True)
    root = None
    for variable in ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(variable)
        if value:
            root = Path(value)
            break
    if root is not None:
        for directory in (root / "data/platform_config", root / f"{platform.machine()}-linux" / "data/platform_config"):
            path = directory / f"{sku}.ini"
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"{sku} needs its own selected SDK platform file; activate the SDK or pass platform_config explicitly")


def a3_resources(
    sku: str,
    platform_config: str | Path | None = None,
) -> dict:
    """Read one SKU's raw SDK capacities and clip to backend usable budgets.

    An INI describes SDK resources, not a physical-path or latency measurement.
    Never substitute another SKU's configuration when this one is missing.
    """
    sku = a3_sku(sku)
    if sku is None:
        raise ValueError("Need exact Ascend910_9392 or Ascend910_9382 SKU")
    path = _platform_file(sku, platform_config)
    content = path.read_bytes()
    config = configparser.ConfigParser()
    config.read_string(content.decode())
    if config["version"]["SoC_version"].upper() != sku.upper() or config["version"]["NpuArch"] != "2201":
        raise ValueError(f"Platform file does not describe {sku}/dav-2201")
    memory_keys = {
        "UB": "ub_size",
        "L1": "l1_size",
        "L0A": "l0_a_size",
        "L0B": "l0_b_size",
        "L0C": "l0_c_size",
    }
    raw = {name: config.getint("AICoreSpec", key) for name, key in memory_keys.items()}
    cores = config.getint("SoCInfo", "cube_core_cnt")
    vector_cores = config.getint("SoCInfo", "vector_core_cnt")
    l2 = config.getint("SoCInfo", "l2_size")
    if any(value <= 0 for value in (*raw.values(), cores, vector_cores, l2)):
        raise ValueError("SDK capacities and core counts must be positive")
    cube = [config.getint("AICoreSpec", f"cube_{axis}_size") for axis in ("m", "n", "k")]
    if any(value <= 0 for value in cube):
        raise ValueError("SDK Cube dimensions must be positive")
    usable = {name: min(value, BACKEND_CAPACITIES[name]) for name, value in raw.items()}
    return {
        "sku": sku,
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "raw_memory_capacities": raw,
        "backend_memory_capacities": dict(BACKEND_CAPACITIES),
        "usable_memory_capacities": usable,
        "cores": cores,
        "vector_cores": vector_cores,
        "L2": l2,
        "cube": cube,
        "supports_bf16": config.getboolean("SoCInfo", "support_bf16"),
    }
