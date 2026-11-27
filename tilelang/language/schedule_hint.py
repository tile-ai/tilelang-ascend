"""The currently consumed subset of Ascend task/stage markers."""

from __future__ import annotations

from tvm import tir
from tvm.script.ir_builder.tir import attr

__all__ = ["Task", "Stage"]


def Task(latency: int | None = None, ii: int | None = None):
    """Mark one audited Vector call as one task, without launching a core.

    Supports a single audited float16/float32 GM/UB copy, fill or add. Nested tasks,
    multiple calls, control flow, manual synchronization and compound helpers
    are diagnosed. Pipeline costs come from an explicit calibrated model.
    """
    if latency is not None or ii is not None:
        raise ValueError("Task latency/ii overrides are unsupported; use a calibrated pipeline cost model")
    return attr({}, "tl.ascend_task", tir.IntImm("int32", 1))


def Stage(stage: int):
    """Constrain complete tasks to a nonnegative stage in constraint mode.

    Nonzero stages require the supported constraint pipeline and calibrated
    costs. Legacy compilation retains its stage-zero-only contract. Structural
    and configuration conflicts are diagnosed by the consuming compiler.
    """
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < (1 << 31):
        raise ValueError("Stage requires a nonnegative constant signed int32 value")
    return attr(tir.IntImm("int64", stage), "tl.ascend_stage", tir.IntImm("int32", 1))
