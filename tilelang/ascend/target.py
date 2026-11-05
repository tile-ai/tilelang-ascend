from __future__ import annotations

import os

from tvm.target import Target

from tilelang.backend.target import (
    TargetLike,
    register_target_detector,
    register_target_execution_normalizer,
    register_target_normalizer,
)


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def _make_ascend_target(target_dict: dict | None = None) -> Target:
    target_dict = dict(target_dict or {})
    target_dict["kind"] = "ascend"
    return Target(target_dict)


def _with_pto_key(target: Target) -> Target:
    target_dict = dict(target.export())
    # The native kind supplies Ascend semantics; "pto" selects PTO codegen,
    # while "ascend" preserves the target kind's default feature key.
    target_dict["keys"] = list(dict.fromkeys(["pto", *target_dict.get("keys", ()), "ascend"]))
    return Target(target_dict)


def target_is_ascend(target: Target) -> bool:
    """Return whether *target* uses the Ascend architecture."""
    return _target_ffi_api().TargetIsAscend(target)


def target_is_pto(target: Target) -> bool:
    """Return whether *target* selects the official A5 PTO backend."""
    return target_is_ascend(target) and not is_a2a3_target(target) and "pto" in target.keys


def target_is_plain_ascend(target: Target) -> bool:
    """Return whether *target* selects the official A5 AscendC backend."""
    return target_is_ascend(target) and not is_a2a3_target(target) and "pto" not in target.keys


def is_a2a3_target(target: Target) -> bool:
    """Return whether an Ascend target selects the A2/A3 compiler variant."""
    return target_is_ascend(target) and str(target.attrs.get("arch", "")) == "dav-2201"


def _get_soc_name() -> str | None:
    try:
        import acl

        name = acl.get_soc_name()
        if name:
            return str(name)
    except Exception:
        pass

    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return str(torch.npu.get_device_name())
    except Exception:
        pass
    return None


def determine_platform(platform: str = "auto") -> str | None:
    """Resolve A2/A3/A5 from an explicit platform, environment, or known SoC."""
    automatic = platform == "auto"
    requested = os.environ.get("TL_PLATFORM") if automatic else platform
    if requested or not automatic:
        value = requested.strip().upper()
        if value in ("A2", "A3", "A5"):
            return value
        source = "TL_PLATFORM" if automatic else "platform"
        raise ValueError(f"Invalid {source}={requested!r}; expected A2, A3, or A5")

    soc_name = _get_soc_name()
    if soc_name is None:
        return None
    soc_name = soc_name.upper()
    if "950" in soc_name or "910_95" in soc_name:
        return "A5"
    if "910B" in soc_name:
        return "A2"
    if "910C" in soc_name or "910_93" in soc_name:
        return "A3"
    return None


def detect_a2a3_arch() -> str | None:
    """Select the A2/A3 architecture for a confirmed A2/A3 platform."""
    return "dav-2201" if determine_platform() in ("A2", "A3") else None


def _detected_ascend_target(*, pto: bool = False) -> Target:
    arch = detect_a2a3_arch()
    target = _make_ascend_target({"arch": arch} if arch else None)
    return _with_pto_key(target) if pto else target


def check_ascend_availability() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _detect_ascend_target() -> Target | None:
    target = _detected_ascend_target()
    if os.environ.get("TL_PLATFORM") or is_a2a3_target(target) or check_ascend_availability():
        return target
    return None


def normalize_ascend_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "ascend":
        return None
    return _detected_ascend_target()


def normalize_pto_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "pto":
        return None
    return _detected_ascend_target(pto=True)


def normalize_pto_execution_target(target: Target, execution_backend: str | None) -> Target | None:
    """Select the PTO target variant for an explicit PTO execution backend."""

    if execution_backend is None or str(execution_backend).lower() != "pto":
        return None
    if not target_is_ascend(target):
        return None
    if "pto" in target.keys:
        return target
    return _with_pto_key(target)


def normalize_asc_target(target: TargetLike) -> Target | None:
    """Accept ``asc`` as the concise name for the AscendC backend."""
    if isinstance(target, str) and target.strip() == "asc":
        return normalize_ascend_target("ascend")
    return None


register_target_detector("ascend", _detect_ascend_target, override=True)
register_target_normalizer("ascend", normalize_ascend_target, override=True)
register_target_normalizer("asc", normalize_asc_target, override=True)
register_target_normalizer("pto", normalize_pto_target, override=True)
register_target_execution_normalizer("pto", normalize_pto_execution_target, override=True)
