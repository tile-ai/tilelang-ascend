# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TA resource scopes and its retained warp specialization helper."""

from tvm.tirx.script.builder.frame import TIRFrame

from tilelang.cuda.language.warpgroup import WarpSpecialize, WarpSpecializeFrame, ws
from .._native import register_optional_object, require_native

__all__ = ["Scope", "ScopeFrame", "WarpSpecialize", "WarpSpecializeFrame", "ws"]


@register_optional_object("tl.A2A3ScopeFrame", "tl.A2A3Scope")
class ScopeFrame(TIRFrame):
    """A frame carrying TA's ``resource_scope`` annotation."""


def Scope(name: str) -> ScopeFrame:
    """Select a TA resource scope: ``V`` maps to 1 and other names to 0.

    In particular ``T.Scope("C")`` marks Cube work and ``T.Scope("V")``
    marks Vector work. This preserves the TA attribute representation;
    it does not introduce the official A5 dialect's CUBE/VECTOR blocks.
    """
    return require_native("tl.A2A3Scope", "Scope")(name)
