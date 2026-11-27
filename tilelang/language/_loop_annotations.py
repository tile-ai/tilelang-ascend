# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Validate user loop metadata without changing scheduling controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_RESERVED_KEYS = frozenset(
    {
        "num_stages",
        "tl_pipeline_order",
        "tl_pipeline_stage",
        "tl_pipeline_sync",
        "tl_pipeline_group",
        "tl_cross_interval",
        "tl.ascend_task",
        "tl.ascend_stage",
        "tl.ascend_pipeline_loop_id",
        "tl.ascend_pipeline_versions",
        "tl.ascend_persistent_guard",
        "enable_offset",
        "parallel_loop_layout",
    }
)


def loop_annotations(annotations: Mapping[str, Any] | None, *, allow_offset: bool = False) -> dict[str, Any]:
    """Copy metadata; scheduling keys require their dedicated supported API."""
    if annotations is None:
        return {}
    if not isinstance(annotations, Mapping):
        raise TypeError("loop annotations must be a mapping with string keys")
    result = dict(annotations)
    for key in result:
        if not isinstance(key, str):
            raise TypeError("loop annotations must have string keys")
        if key == "enable_offset" and allow_offset:
            if type(result[key]) is not bool:
                raise TypeError("Pipelined enable_offset must be bool")
            continue
        if key in _RESERVED_KEYS or key.startswith(
            (
                "software_pipeline_",
                "tl.ascend_storage_",
                "tl.ascend_handoff_",
                "tl.ascend_local_sync_",
                "tl.ascend_auto_schedule",
            )
        ):
            raise ValueError(f"loop annotation {key!r} is reserved; use its dedicated supported API")
    return result
