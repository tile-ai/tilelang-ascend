from __future__ import annotations

import configparser
import hashlib
from pathlib import Path

A3_SKUS = ("Ascend910_9392", "Ascend910_9382")
BACKEND_CAPACITIES = {
    "L0A": 65536,
    "L0B": 65536,
    "L0C": 131072,
    "L1": 524032,
    "UB": 196352,
}


def a3_sku(name: str | None) -> str | None:
    name = (name or "").upper()
    return next((sku for sku in A3_SKUS if sku.upper() == name), None)


def a3_resources(sku: str, platform_config: str | Path) -> dict:
    path = Path(platform_config).resolve()
    content = path.read_bytes()
    config = configparser.ConfigParser()
    config.read_string(content.decode())
    memory_keys = {
        "UB": "ub_size",
        "L1": "l1_size",
        "L0A": "l0_a_size",
        "L0B": "l0_b_size",
        "L0C": "l0_c_size",
    }
    raw = {name: config.getint("AICoreSpec", key) for name, key in memory_keys.items()}
    return {
        "sku": sku,
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "raw_memory_capacities": raw,
        "usable_memory_capacities": {name: min(value, BACKEND_CAPACITIES[name]) for name, value in raw.items()},
        "cores": config.getint("SoCInfo", "cube_core_cnt"),
        "vector_cores": config.getint("SoCInfo", "vector_core_cnt"),
        "L2": config.getint("SoCInfo", "l2_size"),
        "cube": [config.getint("AICoreSpec", f"cube_{axis}_size") for axis in ("m", "n", "k")],
        "supports_bf16": config.getboolean("SoCInfo", "support_bf16"),
    }
