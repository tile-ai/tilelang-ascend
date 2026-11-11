"""Device-only high/low Q-latent projection storage; isolated precision candidate.

FP32 reduction is unchanged. Hi is the original LP cast; Lo stores its residual.
Consumers must use both components without host-side device tensor arithmetic.
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T

from .._contracts import validate_projection_spans

EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


def projection_plan(B, Tq, H, tiles_per_row):
    if any(type(value) is not int or value <= 0 for value in (B, Tq, H)):
        raise ValueError("B, Tq and H must be positive Python integers")
    validate_projection_spans(B, Tq, H)
    tasks = B * Tq * H * tiles_per_row
    if tasks > 2**31 - 1:
        raise ValueError("projection task count exceeds signed int32 indexing")
    blocks = min(20, (tasks + 1) // 2)
    workers = 2 * blocks
    return dict(tasks=tasks, blocks=blocks, aiv_workers=workers, rounds=(tasks + workers - 1) // workers)


def _target(target):
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be 'pto' or 'ascendc'")


def _tensor(name, value, shape=None, dtype=None, device=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous; strided deployment inputs are not supported")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")
    if value.device.type not in ("npu", "privateuseone"):
        raise ValueError(f"{name} must be on an NPU")


@lru_cache(maxsize=128)
def q_absorb_residual_kernel(B, Tq, H, dtype="float16", *, target="pto"):
    _target(target)
    if dtype not in DTYPES.values():
        raise TypeError("projection dtype must be float16 or bfloat16")
    plan = projection_plan(B, Tq, H, 8)
    tasks, blocks, workers, rounds = (plan[name] for name in ("tasks", "blocks", "aiv_workers", "rounds"))

    @T.prim_func
    def main(
        Q: T.Tensor((B, Tq, H, 128), dtype),
        W: T.Tensor((H, 128, 512), dtype),
        QL: T.Tensor((B, Tq, H, 512), dtype),
        QLo: T.Tensor((B, Tq, H, 512), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            qh = T.alloc_ub((128,), dtype)
            qf = T.alloc_ub((128,), "float32")
            wh = T.alloc_ub((128, 64), dtype)
            wf = T.alloc_ub((128, 64), "float32")
            out = T.alloc_ub((64,), "float32")
            outh = T.alloc_ub((64,), dtype)
            roundtrip = T.alloc_ub((64,), "float32")
            outlo = T.alloc_ub((64,), dtype)
            T.annotate_address({qh: 0, qf: 256, wh: 768, wf: 17152, out: 49920, outh: 50176, roundtrip: 50304, outlo: 50560})
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid * 2 + vid + turn * workers
                    if task < tasks:
                        row = task // 8
                        r0 = (task % 8) * 64
                        h = row % H
                        t = (row // H) % Tq
                        b = row // (H * Tq)
                        T.copy(Q[b, t, h, 0], qh)
                        T.copy(W[h, 0, r0], wh)
                        T.barrier_all()
                        T.copy(qh, qf)
                        T.copy(wh, wf)
                        T.barrier_all()
                        for i, j in T.Parallel(128, 64):
                            wf[i, j] = wf[i, j] * qf[i]
                        T.barrier_all()
                        T.reduce_sum(wf, out, dim=0)
                        T.barrier_all()
                        T.copy(out, outh)
                        T.barrier_all()
                        T.copy(outh, roundtrip)
                        T.barrier_all()
                        T.tile.sub(roundtrip, out, roundtrip)
                        T.barrier_all()
                        T.copy(roundtrip, outlo)
                        T.barrier_all()
                        T.copy(outh, QL[b, t, h, r0])
                        T.copy(outlo, QLo[b, t, h, r0])
                        T.barrier_all()

    return tilelang.compile(main, out_idx=[-2, -1], target=target, pass_configs=EXPERT_PASSES)


@lru_cache(maxsize=128)
def q_absorb_packed_residual_kernel(B, Tq, H, dtype="float16", *, target="pto"):
    _target(target)
    if dtype not in DTYPES.values():
        raise TypeError("projection dtype must be float16 or bfloat16")
    validate_projection_spans(B, Tq, H, packed=True)
    plan = projection_plan(B, Tq, H, 8)
    tasks, blocks, workers, rounds = (plan[name] for name in ("tasks", "blocks", "aiv_workers", "rounds"))

    @T.prim_func
    def main(
        Q: T.Tensor((B, Tq, H, 192), dtype),
        W: T.Tensor((H, 256, 512), dtype),
        QL: T.Tensor((B, Tq, H, 512), dtype),
        QLo: T.Tensor((B, Tq, H, 512), dtype),
        QA: T.Tensor((B, Tq, H, 64), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            qh = T.alloc_ub((128,), dtype)
            qf = T.alloc_ub((128,), "float32")
            wh = T.alloc_ub((128, 64), dtype)
            wf = T.alloc_ub((128, 64), "float32")
            out = T.alloc_ub((64,), "float32")
            outh = T.alloc_ub((64,), dtype)
            qauxh = T.alloc_ub((64,), dtype)
            roundtrip = T.alloc_ub((64,), "float32")
            outlo = T.alloc_ub((64,), dtype)
            T.annotate_address({qh: 0, qf: 256, wh: 768, wf: 17152, out: 49920, outh: 50176, qauxh: 50304, roundtrip: 50432, outlo: 50688})
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid * 2 + vid + turn * workers
                    if task < tasks:
                        row = task // 8
                        r0 = (task % 8) * 64
                        h = row % H
                        t = (row // H) % Tq
                        b = row // (H * Tq)
                        T.copy(Q[b, t, h, 0], qh)
                        T.copy(W[h, 0, r0], wh)
                        T.barrier_all()
                        T.copy(qh, qf)
                        T.copy(wh, wf)
                        T.barrier_all()
                        for i, j in T.Parallel(128, 64):
                            wf[i, j] = wf[i, j] * qf[i]
                        T.barrier_all()
                        T.reduce_sum(wf, out, dim=0)
                        T.barrier_all()
                        T.copy(out, outh)
                        T.barrier_all()
                        T.copy(outh, roundtrip)
                        T.barrier_all()
                        T.tile.sub(roundtrip, out, roundtrip)
                        T.barrier_all()
                        T.copy(roundtrip, outlo)
                        T.barrier_all()
                        T.copy(outh, QL[b, t, h, r0])
                        T.copy(outlo, QLo[b, t, h, r0])
                        T.barrier_all()
                        # r_tile=0 is the unique QA writer for this logical row.
                        if task % 8 == 0:
                            T.copy(Q[b, t, h, 128], qauxh)
                            T.barrier_all()
                            T.copy(qauxh, QA[b, t, h, 0])
                            T.barrier_all()

    return tilelang.compile(main, out_idx=[-3, -2, -1], target=target, pass_configs=EXPERT_PASSES)


def q_absorb_residual(q_content, w_k, *, target="pto"):
    """Return separate low-precision (Hi, Lo); FP32 reduction and residual stay on device."""
    _target(target)
    _tensor("q_content", q_content)
    if q_content.ndim != 4 or q_content.shape[-1] != 128:
        raise ValueError("q_content must have shape [B,Tq,H,128]")
    if q_content.dtype not in DTYPES:
        raise TypeError("q_content must be float16 or bfloat16")
    B, Tq, H, _ = map(int, q_content.shape)
    projection_plan(B, Tq, H, 8)
    _tensor("w_k", w_k, (H, 128, 512), q_content.dtype, q_content.device)
    hi, lo = q_absorb_residual_kernel(B, Tq, H, DTYPES[q_content.dtype], target=target)(q_content, w_k)
    return hi, lo


def q_absorb_packed_residual(packed_q, packed_w, *, target="pto"):
    """Return (Hi, Lo, QA) from actual packed bases; no host casts, sums or packing."""
    _target(target)
    _tensor("packed_q", packed_q)
    if packed_q.ndim != 4 or packed_q.shape[-1] != 192:
        raise ValueError("packed_q must have shape [B,Tq,H,192]")
    if packed_q.dtype not in DTYPES:
        raise TypeError("packed_q must be float16 or bfloat16")
    B, Tq, H, _ = map(int, packed_q.shape)
    projection_plan(B, Tq, H, 8)
    _tensor("packed_w", packed_w, (H, 256, 512), packed_q.dtype, packed_q.device)
    hi, lo, qa = q_absorb_packed_residual_kernel(B, Tq, H, DTYPES[packed_q.dtype], target=target)(packed_q, packed_w)
    return hi, lo, qa
