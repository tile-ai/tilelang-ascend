"""Resolve the concrete A2/A3 platform without changing target ownership."""

from __future__ import annotations

import os

from tvm.target import Target

from tilelang.ascend import target as ascend_target


def _known_platform(model: str | None) -> str | None:
    if model is None:
        return None
    name = str(model).strip().upper()
    if "950" in name or "910_95" in name:
        return None
    if name in ("A2", "A3"):
        return name
    if name.startswith(("ASCEND910B", "910B")):
        return "A2"
    if name.startswith(("ASCEND910C", "910C", "ASCEND910_93", "910_93")):
        return "A3"
    return None


def resolve_a2a3_platform(target: Target, *, required: bool = True) -> str | None:
    """Resolve model, then TL_PLATFORM, then the detected SoC.

    Unknown devices never imply A3. Optional resolution is suitable for a
    future target normalizer: it leaves unresolved explicit arch targets
    alone. Required resolution is for consumers that need the concrete
    hardware distinction, such as platform-dependent Reduce2D codegen.
    """
    if not ascend_target.is_a2a3_target(target):
        if required:
            raise ValueError("A2/A3 platform resolution requires an ascend target with arch=dav-2201.")
        return None

    platform = _known_platform(target.attrs.get("model"))
    if platform is not None:
        return platform

    override = os.environ.get("TL_PLATFORM")
    if override:
        value = override.strip().upper()
        if value in ("A2", "A3"):
            return value
        if required:
            raise ValueError(
                f"TL_PLATFORM={override!r} does not select A2/A3. Set target.model=A2/A3 or TL_PLATFORM=A2/A3 for arch=dav-2201."
            )
        return None

    try:
        soc_name = ascend_target._get_soc_name()
    except Exception:
        soc_name = None
    platform = _known_platform(soc_name)
    if platform is not None:
        return platform
    if required:
        raise ValueError("Cannot confirm the A2/A3 platform. Set target.model=A2/A3 or TL_PLATFORM=A2/A3 for arch=dav-2201.")
    return None


def normalize_a2a3_platform_target(target: Target, execution_backend: str | None = None) -> Target | None:
    """Record a confirmed platform in model for target/cache serialization.

    This callback is deliberately not registered here. Unresolved targets
    return None, and an already canonical model returns the original target,
    following the registered target-normalizer convention.
    """
    platform = resolve_a2a3_platform(target, required=False)
    if platform is None:
        return None
    if str(target.attrs.get("model", "")) == platform:
        return target
    config = dict(target.export())
    config["model"] = platform
    return Target(config, host=target.host)
