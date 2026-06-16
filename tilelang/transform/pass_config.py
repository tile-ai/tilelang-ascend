# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# TODO: Add more documentation for each pass config

from __future__ import annotations

from enum import Enum


class PassConfigKey(str, Enum):
    """Pass configuration keys for TileLang compiler."""

    # TileLang specific configs
    TL_SIMPLIFY = "tl.Simplify"
    """Enable/disable TileLang simplification passes. Default: True"""

    TL_DYNAMIC_ALIGNMENT = "tl.dynamic_alignment"
    """Memory alignment requirement for dynamic shapes. Default: 16"""

    TL_DISABLE_DYNAMIC_TAIL_SPLIT = "tl.disable_dynamic_tail_split"
    """Disable dynamic tail splitting optimization. Default: False"""

    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    """Disable warp specialization optimization. Default: False"""

    TL_CONFIG_INDEX_BITWIDTH = "tl.config_index_bitwidth"
    """Bitwidth for configuration indices. Default: 32"""

    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"
    """Disable TMA (Tensor Memory Access) lowering. Default: False"""

    TL_DISABLE_SAFE_MEMORY_ACCESS = "tl.disable_safe_memory_legalize"
    """Disable safe memory access optimization. Default: False"""

    TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS = "tl.debug_merge_shared_memory_allocations"
    """Enable debug information for merge shared memory allocations. Default: False"""

    TL_ASCEND_AUTO_SYNC = "tl.ascend_auto_sync"
    """Enable/disable TileLang AscendSyncInsert pass. Default: False"""

    TL_ASCEND_AUTO_SYNC_VS = "tl.ascend_auto_sync_vs"
    """Enable/disable TileLang AscendSyncInsertVS pass. Default value setted dynamically according target"""

    TL_ASCEND_MEMORY_PLANNING = "tl.ascend_memory_planning"
    """Enable/disable TileLang AscendMemoryPlanning pass. Default: False"""

    TL_ASCEND_AUTO_CV_COMBINE = "tl.ascend_auto_cv_combine"
    """Enable/disable TileLang CombineCV pass. Default: False"""

    TL_ASCEND_AUTO_CV_SYNC = "tl.ascend_auto_cross_core_sync"
    """Enable/disable TileLang Auto CV Synchronization. Default: False"""

    TL_ASCEND_TAIL_MASK = "tl.ascend_tail_mask"
    """Enable/disable the AscendC tail-block valid-region scheme
    (AscendTailMaskPropagation): rewrites unary/binary/scalar ops on a UB tail
    tile to compute only the valid region. Opt-in so non-tail kernels are
    unaffected. Default: False"""

    TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY = "tl.ascend_pto_use_pipe_in_cv_copy"
    """Enable PTO copy-through-pipe path in cross CV copy. When False, PTO falls back to copy-through-GM path. Default: True"""

    # TIR related configs
    TIR_ENABLE_EQUIV_TERMS_IN_CSE = "tir.enable_equiv_terms_in_cse_tir"
    """Enable equivalent terms in TIR Common Subexpression Elimination. Default: True"""

    TIR_DISABLE_CSE = "tir.disable_cse_tir"
    """Disable TIR Common Subexpression Elimination. Default: False"""

    TIR_SIMPLIFY = "tir.Simplify"
    """Enable/disable TIR simplification passes. Default: True"""

    TIR_DISABLE_STORAGE_REWRITE = "tir.disable_storage_rewrite"
    """Disable storage rewrite optimization. Default: False"""

    TIR_DISABLE_VECTORIZE = "tir.disable_vectorize"
    """Disable vectorization optimization. Default: False"""

    TIR_USE_ASYNC_COPY = "tir.use_async_copy"
    """Enable asynchronous memory copy operations. Default: True"""

    TIR_ENABLE_DEBUG = "tir.enable_debug"
    """Enable debug information in generated code. Default: False"""

    TIR_MERGE_STATIC_SMEM = "tir.merge_static_smem"
    """Merge static shared memory allocations. Default: True"""

    TIR_ADD_LOWER_PASS = "tir.add_lower_pass"
    """Additional lowering passes to be applied. Default: None"""

    TIR_NOALIAS = "tir.noalias"
    """Enable pointer non-aliasing assumptions. Default: True"""

    CUDA_KERNELS_OUTPUT_DIR = "cuda.kernels_output_dir"
    """Output directory for generated CUDA kernels. Default: empty string"""


from tvm.target import Target
from os import environ
from logging import getLogger

logger = getLogger(__name__)

_TARGET_PASS_DEFAULTS: dict[str, dict[str, bool]] = {
    "ascendc": {},
    "pto": {
        PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    },
}


def _resolve_target_model(target: str | Target) -> str:
    """Extract the effective target model string from a target argument.

    Returns one of: 'ascendc', 'pto', 'auto', 'cuda', 'hip', 'llvm', etc.
    For :class:`tvm.target.Target` inputs, returns ``target.model``;
    for strings, returns the string as-is.

    Parameters
    ----------
    target : Union[str, Target]
        User-specified target. May be ``"auto"``, ``"ascendc"``, ``"pto"``,
        a TVM ``Target`` object, or other backend strings.

    Returns
    -------
    str
        The effective target model string. Empty string if target is falsy.
    """
    if isinstance(target, Target):
        return getattr(target, "model", "") or ""
    return target or ""


def _apply_target_pass_defaults(
    target: str | Target,
    pass_configs: dict | None,
) -> dict:
    """Fill in per-target default values for pass_configs keys not set by the user.

    Returns a new dict; the input is not modified. User-explicit settings
    always take priority over per-target defaults.

    Parameters
    ----------
    target : Union[str, Target]
        User-specified target. ``"auto"`` is treated as ``"ascendc"``
    pass_configs : Optional[dict]
        User-provided pass_configs dict. May be ``None``.

    Returns
    -------
    dict
        A new dict with per-target defaults applied for any keys the user
        did not explicitly set.
    """
    configs = {}
    if pass_configs:
        for k, v in pass_configs.items():
            configs[k.value if hasattr(k, "value") else k] = v

    model = _resolve_target_model(target)
    effective = "ascendc" if model in ("auto", "") else model
    defaults = _TARGET_PASS_DEFAULTS.get(effective, {})

    for key_str, val in defaults.items():
        raw_key = key_str.value if hasattr(key_str, "value") else key_str
        if raw_key not in configs:
            configs[raw_key] = val

    return configs


def _process_for_auto_sync_insert_vs(pass_configs):
    vs_enabled = bool(pass_configs and pass_configs.get(PassConfigKey.TL_ASCEND_AUTO_SYNC_VS, False))
    if vs_enabled:
        if "TL_CCE_AUTO_SYNC" not in environ:
            environ["TL_CCE_AUTO_SYNC"] = "off"
        if "TL_CCE_OPT_LEVEL" not in environ:
            environ["TL_CCE_OPT_LEVEL"] = "3"


def process_default_pass_config(target, pass_configs):
    pass_configs = _apply_target_pass_defaults(target, pass_configs)
    _process_for_auto_sync_insert_vs(pass_configs)
    return pass_configs
