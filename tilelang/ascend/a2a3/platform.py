"""Resolve the concrete A2/A3 platform without changing target ownership."""

from __future__ import annotations

import os

from tvm.target import Target

from tilelang.ascend import target as ascend_target


def resolve_a2a3_platform(target: Target, *, required: bool = True) -> str | None:
    """Use an explicit A2/A3 model, or query TL_PLATFORM and the detected SoC.

    Unknown devices never imply A3. Optional resolution is suitable for a
    future target normalizer: it leaves unresolved explicit arch targets
    alone. Required resolution supplies the resolved platform for existing
    backend policies, such as Reduce2D scheduling.
    """
    if not ascend_target.is_a2a3_target(target):
        if required:
            raise ValueError("A2/A3 platform resolution requires an ascend target with arch=dav-2201.")
        return None

    model = target.attrs.get("model")
    if model is not None:
        platform = str(model)
    else:
        try:
            platform = ascend_target.determine_platform()
        except Exception:
            platform = None
    if platform in ("A2", "A3"):
        return platform
    if required:
        if model is not None:
            raise ValueError(f"target.model={model!r} does not select A2/A3. Set target.model=A2/A3 for arch=dav-2201.")
        # Read the override only to preserve the established required diagnostic.
        override = os.environ.get("TL_PLATFORM")
        if override:
            raise ValueError(
                f"TL_PLATFORM={override!r} does not select A2/A3. Set target.model=A2/A3 or TL_PLATFORM=A2/A3 for arch=dav-2201."
            )
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
