"""A2/A3 source generation through the shared device-codegen interface."""

from __future__ import annotations

from tilelang import tvm
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.instrumentation import run_codegen_with_instrumentation

from .platform import resolve_a2a3_platform


_ASCENDC_NATIVE = "target.build.tilelang_ascend_a2a3"
_PTO_NATIVE = "target.build.tilelang_pto_a2a3"


def _build_ascendc(mod, target):
    platform = resolve_a2a3_platform(target)
    return run_codegen_with_instrumentation(
        _ASCENDC_NATIVE,
        mod,
        target,
        lambda: tvm.ffi.get_global_func(_ASCENDC_NATIVE)(mod, target, platform),
        role="device",
    )


def _preserve_physical_terminals(mod, _target):
    # A2/A3 mask legalization is the final TIR transformation. Its physical
    # terminals must reach the source builder without the shared preparation.
    return mod


def _build_pto(mod, target):
    platform = resolve_a2a3_platform(target)
    builder = tvm.ffi.get_global_func(_PTO_NATIVE, allow_missing=True)
    if builder is None:
        raise RuntimeError(
            f"A2/A3 PTO source generation requires USE_ASCEND=ON with the complete PTO native components; missing {_PTO_NATIVE}."
        )
    return run_codegen_with_instrumentation(_PTO_NATIVE, mod, target, lambda: builder(mod, target, platform), role="device")


def _prepare_pto_codegen(mod, _target):
    # TA engine/lower.py device_codegen performs this extra Simplify for PTO,
    # without the shared LowerIntrin or HoistBroadcastValues preparation.
    from .transform import AscendSimplify

    return AscendSimplify()(mod)


# The TA native builder emits source. The shared Cython execution backend compiles
# that source separately, so both device-codegen modes use the same builder.
ASCENDC_CODEGEN = DeviceCodegen(
    "ascend_a2a3", build=_build_ascendc, build_without_compile=_build_ascendc, preprocess=_preserve_physical_terminals
)
PTO_CODEGEN = DeviceCodegen("pto_a2a3", build=_build_pto, build_without_compile=_build_pto, preprocess=_prepare_pto_codegen)
