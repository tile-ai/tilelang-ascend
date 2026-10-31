from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tvm import IRModule
from tvm.target import Target

if TYPE_CHECKING:
    from tilelang.engine.param import KernelParam

LowerFunc = Callable[[IRModule, Target], IRModule]
ParameterExtractor = Callable[[IRModule], list["KernelParam"]]


class PassPipeline:
    """Lowering pass pipeline for a specific backend.

    Each :class:`BackendModule` declares the pipelines it owns so the compiler
    can select the correct pass sequence from the resolved backend context.
    """

    def __init__(self, name: str, lower: LowerFunc, parameter_extractor: ParameterExtractor | None = None):
        self.name = name
        self._lower = lower
        self.parameter_extractor = parameter_extractor

    def extract_parameters(self, mod: IRModule) -> list[KernelParam] | None:
        """Optionally describe the final lowered kernel's argument contract."""
        return None if self.parameter_extractor is None else self.parameter_extractor(mod)

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        """Run the pipeline and render source snippets for located errors."""
        try:
            # Full compiler entry points own instrumentation sessions.  This
            # boundary only identifies the active backend pipeline within the
            # caller's session.
            from tilelang.instrumentation import pass_pipeline

            with pass_pipeline(self.name):
                return self._lower(mod, target)
        except Exception as exc:
            # Compiler passes append a machine-readable `--> file:line:col`
            # marker when the relevant IR node carries a span. Keep enrichment
            # at this shared boundary so every pipeline caller gets the same
            # user-facing diagnostic without changing exception semantics.
            from tilelang.errors import enrich_error

            enrich_error(exc)
            raise
