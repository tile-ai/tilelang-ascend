"""TensorPulse DSL — V1.0 skeleton.

Wraps ``tl.tensorpulse_*`` TIR operators registered in
``src/op/tensorpulse.cc``. Real intrinsics (gemm_v0, set/wait_flag,
pipe_barrier, sync_all, ...) are added incrementally as codegen lands.
"""

from __future__ import annotations

from tvm import tir
from tvm.tir import Buffer


def add(dst: Buffer, src0: Buffer, src1: Buffer):
    """Skeleton elementwise add: ``dst = src0 + src1``.

    Emits a TIR intrinsic call to ``tl.tensorpulse_add`` that the TensorPulse
    codegen will lower to an SMC microinstruction sequence (INTADD / FPADD /
    FP16MUL depending on dtype).
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tensorpulse_add"),
        dst.access_ptr("w"),
        src0.access_ptr("r"),
        src1.access_ptr("r"),
    )
