"""Target predicates and explicit aliases for the A2/A3 variant."""

from __future__ import annotations

from tvm.target import Target

from tilelang.ascend.target import is_a2a3_target
from tilelang.backend.target import TargetLike, register_target_execution_normalizer, register_target_normalizer

from .platform import normalize_a2a3_platform_target


def target_is_ascend_a2a3(target: Target) -> bool:
    return is_a2a3_target(target) and "pto" not in target.keys


def target_is_pto_a2a3(target: Target) -> bool:
    return is_a2a3_target(target) and "pto" in target.keys


def normalize_ascendc_target(target: TargetLike) -> Target | None:
    """The explicit AscendC alias always selects A2/A3, even without a device."""
    if isinstance(target, str) and target.strip() == "ascendc":
        return Target({"kind": "ascend", "arch": "dav-2201"})
    return None


def normalize_a2a3_execution_target(target: Target, execution_backend: str | None) -> Target | None:
    """Freeze either dialect's platform before creating a compilation cache key."""
    if not is_a2a3_target(target) or str(execution_backend).lower() == "pto":
        return None
    return normalize_a2a3_platform_target(target, execution_backend)


register_target_execution_normalizer("ascend_a2a3", normalize_a2a3_execution_target)
register_target_normalizer("ascendc", normalize_ascendc_target)
