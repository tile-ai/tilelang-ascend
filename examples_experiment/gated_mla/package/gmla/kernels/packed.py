"""Packed Q/W Gated MLA projections; isolated candidate, not L3 acceptance.

The af32a45 Expert GEMV/reduction and stable sigmoid are retained. Batch, token
and head work is distributed by device-side task loops, never Python launches
per head. Q absorption deliberately rounds its FP32 result to the input dtype.
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
    validate_projection_spans(B, Tq, H, packed=True)
    tasks = B * Tq * H * tiles_per_row
    if tasks > 2**31 - 1:
        raise ValueError("projection task count exceeds signed int32 indexing")
    blocks = min(24, (tasks + 1) // 2)
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
def q_absorb_packed_kernel(B, Tq, H, dtype="float16", *, target="pto"):
    _target(target)
    if dtype not in DTYPES.values():
        raise TypeError("projection dtype must be float16 or bfloat16")
    plan = projection_plan(B, Tq, H, 8)
    tasks, blocks, workers, rounds = (plan[name] for name in ("tasks", "blocks", "aiv_workers", "rounds"))

    @T.prim_func
    def main(
        Q: T.Tensor((B, Tq, H, 192), dtype),
        W: T.Tensor((H, 256, 512), dtype),
        QL: T.Tensor((B, Tq, H, 512), dtype),
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
            T.annotate_address({qh: 0, qf: 256, wh: 768, wf: 17152, out: 49920, outh: 50176, qauxh: 50304})
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
                        T.copy(outh, QL[b, t, h, r0])
                        T.barrier_all()
                        # r_tile=0 is the unique QA writer for this logical row.
                        if task % 8 == 0:
                            T.copy(Q[b, t, h, 128], qauxh)
                            T.barrier_all()
                            T.copy(qauxh, QA[b, t, h, 0])
                            T.barrier_all()

    return tilelang.compile(main, out_idx=[-2, -1], target=target, pass_configs=EXPERT_PASSES)


@lru_cache(maxsize=128)
def value_gate_packed_kernel(B, Tq, H, dtype="float16", *, target="pto"):
    _target(target)
    if dtype not in DTYPES.values():
        raise TypeError("projection dtype must be float16 or bfloat16")
    plan = projection_plan(B, Tq, H, 4)
    tasks, blocks, workers, rounds = (plan[name] for name in ("tasks", "blocks", "aiv_workers", "rounds"))

    @T.prim_func
    def main(
        Z: T.Tensor((B, Tq, H, 512), "float32"),
        W: T.Tensor((H, 256, 512), dtype),
        G: T.Tensor((B, Tq, H, 128), dtype),
        Y: T.Tensor((B, Tq, H, 128), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            z = T.alloc_ub((512,), "float32")
            wh = T.alloc_ub((32, 512), dtype)
            wf = T.alloc_ub((32, 512), "float32")
            out = T.alloc_ub((32,), "float32")
            gh = T.alloc_ub((32,), dtype)
            gf = T.alloc_ub((32,), "float32")
            sigmoid = T.alloc_ub((32,), "float32")
            yh = T.alloc_ub((32,), dtype)
            T.annotate_address({z: 0, wh: 2048, wf: 34816, out: 100352, gh: 100480, gf: 100544, sigmoid: 100672, yh: 100800})
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid * 2 + vid + turn * workers
                    if task < tasks:
                        row = task // 4
                        v0 = (task % 4) * 32
                        h = row % H
                        t = (row // H) % Tq
                        b = row // (H * Tq)
                        T.copy(Z[b, t, h, 0], z)
                        T.copy(W[h, 128 + v0, 0], wh)
                        T.copy(G[b, t, h, v0], gh)
                        T.barrier_all()
                        T.copy(wh, wf)
                        T.copy(gh, gf)
                        T.barrier_all()
                        for i, j in T.Parallel(32, 512):
                            wf[i, j] = wf[i, j] * z[j]
                        T.barrier_all()
                        T.reduce_sum(wf, out, dim=1)
                        T.barrier_all()
                        T.tile.abs(sigmoid, gf)
                        T.barrier_all()
                        T.tile.mul(sigmoid, sigmoid, -1.0)
                        T.barrier_all()
                        T.tile.exp(sigmoid, sigmoid)
                        T.barrier_all()
                        for i in T.Parallel(32):
                            sigmoid[i] = T.if_then_else(gf[i] >= 0, 1.0 / (1.0 + sigmoid[i]), sigmoid[i] / (1.0 + sigmoid[i]))
                        T.barrier_all()
                        T.tile.mul(out, out, sigmoid)
                        T.barrier_all()
                        T.copy(out, yh)
                        T.barrier_all()
                        T.copy(yh, Y[b, t, h, v0])
                        T.barrier_all()

    return tilelang.compile(main, out_idx=[-1], target=target, pass_configs=EXPERT_PASSES)


def q_absorb_packed(packed_q, packed_w, *, target="pto"):
    """Return (QL, QA) from real packed Q/W bases, without host slicing/copies."""
    _target(target)
    _tensor("packed_q", packed_q)
    if packed_q.ndim != 4 or packed_q.shape[-1] != 192:
        raise ValueError("packed_q must have shape [B,Tq,H,192]")
    if packed_q.dtype not in DTYPES:
        raise TypeError("packed_q must be float16 or bfloat16")
    B, Tq, H, _ = map(int, packed_q.shape)
    projection_plan(B, Tq, H, 8)
    _tensor("packed_w", packed_w, (H, 256, 512), packed_q.dtype, packed_q.device)
    ql, qa = q_absorb_packed_kernel(B, Tq, H, DTYPES[packed_q.dtype], target=target)(packed_q, packed_w)
    return ql, qa


def value_gate_packed(latent_out, packed_w, gate_logits, *, target="pto"):
    """Read W_V at row 128 of each 256-row packed head; retain FP32 Z and gate math."""
    _target(target)
    _tensor("latent_out", latent_out, dtype=torch.float32)
    if latent_out.ndim != 4 or latent_out.shape[-1] != 512:
        raise ValueError("latent_out must have shape [B,Tq,H,512]")
    B, Tq, H, _ = map(int, latent_out.shape)
    projection_plan(B, Tq, H, 4)
    if not isinstance(packed_w, torch.Tensor) or packed_w.dtype not in DTYPES:
        raise TypeError("packed_w must be float16 or bfloat16")
    _tensor("packed_w", packed_w, (H, 256, 512), packed_w.dtype, latent_out.device)
    _tensor("gate_logits", gate_logits, (B, Tq, H, 128), packed_w.dtype, latent_out.device)
    return value_gate_packed_kernel(B, Tq, H, DTYPES[packed_w.dtype], target=target)(latent_out, packed_w, gate_logits)
