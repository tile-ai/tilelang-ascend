"""Canonical, explicit effective-cost profiles for A3 pipeline planning.

This module uses only the standard library. A profile is measured optimization
data, not a hardware timing guarantee or a synchronization correctness proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL = "a3-effective-op-v1"
COST_CONFIG = "tl.ascend_pipeline_cost_model"
_KINDS = {"gm_to_ub", "ub_to_gm", "fill", "add"}
_DTYPE_BYTES = {"float16": 2, "float32": 4}
_INT64_MAX = (1 << 63) - 1


class CostModelError(ValueError):
    """An effective-cost artifact is invalid or does not match its requested use."""


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= _INT64_MAX:
        raise CostModelError(f"{field} must be an integer in [{minimum}, {_INT64_MAX}]")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CostModelError(f"{field} must be a nonempty string")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CostModelError(f"{field} must be a lowercase SHA256 digest")
    return value


def _json_value(value: Any, path: str = "profile") -> None:
    # Keep the payload lossless across JSON, PassContext and old-TVM containers.
    # Statistical fractions belong in named strings; scheduling quantities use
    # explicitly rounded integer nanoseconds and ticks.
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CostModelError(f"{path} mapping keys must be strings")
            _json_value(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
    elif type(value) is int:
        if not -_INT64_MAX - 1 <= value <= _INT64_MAX:
            raise CostModelError(f"{path} is outside the int64 container domain")
    elif type(value) not in (str, bool):
        raise CostModelError(f"{path} must use strings, bools or int64 values, not floats/null")


def cost_model_digest(profile: Mapping[str, Any]) -> str:
    """Hash the complete artifact, excluding only its own digest field."""
    if not isinstance(profile, Mapping):
        raise CostModelError("cost model must be a mapping")
    payload = {key: value for key, value in profile.items() if key != "profile_sha256"}
    _json_value(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_cost_model(profile: Mapping[str, Any], *, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and copy a full profile; never load a filename hidden in config."""
    if not isinstance(profile, Mapping):
        raise CostModelError("cost model must be a mapping, not a profile filename")
    result = deepcopy(dict(profile))
    _json_value(result)
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise CostModelError("unsupported cost model schema_version")
    if result.get("model") != MODEL or result.get("unit") != "ns":
        raise CostModelError(f"cost model requires model={MODEL!r} and unit='ns'")
    quantum = _integer(result.get("quantum_ns"), "quantum_ns")
    identity = result.get("identity")
    if not isinstance(identity, dict) or identity.get("platform") != "A3":
        raise CostModelError("cost model requires an explicit A3 identity")
    for field in ("soc", "cann_version"):
        _string(identity.get(field), f"identity.{field}")
    _digest(identity.get("bisheng_sha256"), "identity.bisheng_sha256")
    flags = identity.get("flags")
    if not isinstance(flags, list) or not flags or any(not isinstance(x, str) for x in flags):
        raise CostModelError("identity.flags must contain the ordered compiler flags")
    if expected_identity is not None:
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise CostModelError(f"calibration identity mismatch for {key}")
    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise CostModelError("cost model requires nonempty calibrated records")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CostModelError(f"records[{index}] must be a mapping")
        kind, dtype = record.get("kind"), record.get("dtype")
        if not isinstance(kind, str) or not isinstance(dtype, str) or kind not in _KINDS or dtype not in _DTYPE_BYTES:
            raise CostModelError(f"records[{index}] has an unsupported operation/dtype")
        elements = _integer(record.get("elements"), f"records[{index}].elements")
        size = _integer(record.get("bytes"), f"records[{index}].bytes")
        if size != elements * _DTYPE_BYTES[dtype] or size % 32:
            raise CostModelError(f"records[{index}] requires a complete 32-byte aligned tile")
        if type(record.get("rows")) is not int or record["rows"] != 1:
            raise CostModelError(f"records[{index}] requires the audited single-row geometry")
        key = kind, dtype, elements, size, 1
        if key in seen:
            raise CostModelError(f"duplicate cost model geometry: {key}")
        seen.add(key)
        for ticks_key, ns_key in (
            ("latency", "latency_effective_ns"),
            ("ii", "ii_effective_ns"),
        ):
            nanoseconds = _integer(record.get(ns_key), f"records[{index}].{ns_key}")
            ticks = _integer(record.get(ticks_key), f"records[{index}].{ticks_key}")
            if ticks != (nanoseconds // quantum + bool(nanoseconds % quantum)):
                raise CostModelError(f"records[{index}].{ticks_key} is not conservative ns rounding")
    evidence = result.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "PASS":
        raise CostModelError("calibration evidence must have passed its measurement checks")
    _string(evidence.get("method"), "evidence.method")
    samples = evidence.get("raw_samples")
    if not isinstance(samples, (dict, list)) or not samples:
        raise CostModelError("calibration evidence requires the actual raw samples")
    digest = _digest(result.get("profile_sha256"), "profile_sha256")
    if digest != cost_model_digest(result):
        raise CostModelError("cost model content does not match profile_sha256")
    return result


def load_cost_model(path: str | Path, *, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load artifact contents for use as the actual pass_configs value."""
    with Path(path).open(encoding="utf-8") as stream:
        profile = json.load(stream)
    return validate_cost_model(profile, expected_identity=expected_identity)
