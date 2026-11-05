"""A2/A3 pass-key compatibility and original per-dialect defaults."""

from __future__ import annotations

from typing import Any

from tvm.target import Target

from tilelang.ascend.target import is_a2a3_target
from tilelang.transform.pass_config import PassConfigKey, normalize_pass_configs

# Standard TVM options renamed by TIRX. Keep compatibility local to the TA
# backend rather than registering legacy keys globally or changing A5 inputs.
_LEGACY_TIR_KEYS = {
    "tir.disable_cse_tir": PassConfigKey.TIR_DISABLE_CSE.value,
    "tir.Simplify": PassConfigKey.TIR_SIMPLIFY.value,
    "tir.disable_storage_rewrite": PassConfigKey.TIR_DISABLE_STORAGE_REWRITE.value,
    "tir.disable_vectorize": PassConfigKey.TIR_DISABLE_VECTORIZE.value,
    "tir.use_async_copy": PassConfigKey.TIR_USE_ASYNC_COPY.value,
    "tir.enable_debug": PassConfigKey.TIR_ENABLE_DEBUG.value,
    "tir.merge_static_smem": PassConfigKey.TIR_MERGE_STATIC_SMEM.value,
    "tir.add_lower_pass": PassConfigKey.TIR_ADD_LOWER_PASS.value,
    "tir.noalias": PassConfigKey.TIR_NOALIAS.value,
}


def normalize_a2a3_pass_configs(target: Target, pass_configs: dict | None = None) -> dict[str, Any]:
    """Return effective pass configs for an already resolved A2/A3 target.

    AscendC retains the source's empty Python defaults. PTO alone defaults VS
    synchronization on; explicit values always win. Target keys identify the
    dialect while model continues to identify the concrete A2/A3 hardware.
    Native Bool defaults remain in their unique owning components. Compiler
    options and environment-derived Bisheng flags belong to compiler.py.
    """
    if not isinstance(target, Target) or not is_a2a3_target(target):
        raise ValueError("A2/A3 pass defaults require a resolved ascend Target with arch=dav-2201.")
    mapped = {}
    for key, value in (pass_configs or {}).items():
        raw_key = key.value if hasattr(key, "value") else key
        mapped[_LEGACY_TIR_KEYS.get(raw_key, raw_key)] = value
    configs = normalize_pass_configs(mapped)
    if "pto" in target.keys:
        configs.setdefault(PassConfigKey.TL_ASCEND_AUTO_SYNC_VS.value, True)
    return configs
