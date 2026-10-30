"""First Gated MLA decode baseline: five fully manual Vector Expert stages.

Calculation structure follows kda/kda_recurrent.py GEMV/reduction patterns.
All four automation passes are disabled. This is a correctness baseline, not
a claim of optimized decode performance. No NPU arithmetic runs in the wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import tilelang
import tilelang.language as T

from gated_mla_golden import validate_inputs

EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


@tilelang.jit(target="ascendc", out_idx=[-1], pass_configs=EXPERT_PASSES)
def q_absorb_kernel():

    @T.prim_func
    def main(
        Q: T.Tensor((1, 1, 1, 128), "float16"),
        W: T.Tensor((1, 128, 512), "float16"),
        QL: T.Tensor((512,), "float32"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, vid):
            r0 = (cid * 2 + vid) * 64
            qh = T.alloc_ub((128,), "float16")
            qf = T.alloc_ub((128,), "float32")
            wh = T.alloc_ub((128, 64), "float16")
            wf = T.alloc_ub((128, 64), "float32")
            out = T.alloc_ub((64,), "float32")
            T.annotate_address({qh: 0, qf: 256, wh: 768, wf: 17152, out: 49920})
            with T.Scope("V"):
                T.copy(Q[0, 0, 0, 0], qh)
                T.copy(W[0, 0, r0], wh)
                T.barrier_all()
                T.copy(qh, qf)
                T.copy(wh, wf)
                T.barrier_all()
                # Outer-index broadcast must scale the destination in place.
                for i, j in T.Parallel(128, 64):
                    wf[i, j] = wf[i, j] * qf[i]
                T.barrier_all()
                T.reduce_sum(wf, out, dim=0)
                T.barrier_all()
                T.copy(out, QL[r0])
                T.barrier_all()

    return main


@tilelang.jit(target="ascendc", out_idx=[-1], pass_configs=EXPERT_PASSES)
def score_kernel(visible_keys: int, scale: float):

    @T.prim_func
    def main(
        QL: T.Tensor((512,), "float32"),
        QA: T.Tensor((1, 1, 1, 64), "float16"),
        C: T.Tensor((1, 64, 512), "float16"),
        E: T.Tensor((1, 64, 64), "float16"),
        S: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            t0 = vid * 32
            qf = T.alloc_ub((64,), "float32")
            ch = T.alloc_ub((32, 64), "float16")
            cf = T.alloc_ub((32, 64), "float32")
            part = T.alloc_ub((32,), "float32")
            acc = T.alloc_ub((32,), "float32")
            qh = T.alloc_ub((64,), "float16")
            T.annotate_address({qf: 0, ch: 256, cf: 4352, part: 12544, acc: 12672, qh: 12800})
            with T.Scope("V"):
                T.tile.fill(acc, 0.0)
                T.barrier_all()
                for k in T.serial(8):
                    T.copy(QL[k * 64], qf)
                    T.copy(C[0, t0, k * 64], ch)
                    T.barrier_all()
                    T.copy(ch, cf)
                    T.barrier_all()
                    for i, j in T.Parallel(32, 64):
                        cf[i, j] = cf[i, j] * qf[j]
                    T.barrier_all()
                    T.reduce_sum(cf, part, dim=1)
                    T.barrier_all()
                    T.tile.add(acc, acc, part)
                    T.barrier_all()
                T.copy(QA[0, 0, 0, 0], qh)
                T.copy(E[0, t0, 0], ch)
                T.barrier_all()
                T.copy(qh, qf)
                T.copy(ch, cf)
                T.barrier_all()
                for i, j in T.Parallel(32, 64):
                    cf[i, j] = cf[i, j] * qf[j]
                T.barrier_all()
                T.reduce_sum(cf, part, dim=1)
                T.barrier_all()
                T.tile.add(acc, acc, part)
                T.barrier_all()
                for i in T.Parallel(32):
                    acc[i] = T.if_then_else(t0 + i < visible_keys, acc[i] * scale, -T.infinity("float32"))
                T.barrier_all()
                T.copy(acc, S[t0])
                T.barrier_all()

    return main


@tilelang.jit(target="ascendc", out_idx=[-1], pass_configs=EXPERT_PASSES)
def softmax_kernel():

    @T.prim_func
    def main(S: T.Tensor((64,), "float32"), P: T.Tensor((64,), "float32")):
        with T.Kernel(1, is_npu=True) as (_, vid):
            x = T.alloc_ub((64,), "float32")
            maximum = T.alloc_ub((1,), "float32")
            total = T.alloc_ub((1,), "float32")
            # Reserve separate 32-byte slots for scalar reduction outputs.
            T.annotate_address({x: 0, maximum: 256, total: 288})
            with T.Scope("V"):
                if vid == 0:
                    T.copy(S, x)
                    T.barrier_all()
                    T.reduce_max(x, maximum, dim=0)
                    T.barrier_all()
                    for j in T.Parallel(64):
                        x[j] = x[j] - maximum[0]
                    T.barrier_all()
                    T.tile.exp(x, x)
                    T.barrier_all()
                    T.reduce_sum(x, total, dim=0)
                    T.barrier_all()
                    for j in T.Parallel(64):
                        x[j] = x[j] / total[0]
                    T.barrier_all()
                    T.copy(x, P)
                    T.barrier_all()

    return main


@tilelang.jit(target="ascendc", out_idx=[-1], pass_configs=EXPERT_PASSES)
def latent_readout_kernel(visible_keys: int):

    @T.prim_func
    def main(
        P: T.Tensor((64,), "float32"),
        C: T.Tensor((1, 64, 512), "float16"),
        Z: T.Tensor((512,), "float32"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, vid):
            r0 = (cid * 2 + vid) * 64
            p = T.alloc_ub((64,), "float32")
            ch = T.alloc_ub((64, 64), "float16")
            cf = T.alloc_ub((64, 64), "float32")
            out = T.alloc_ub((64,), "float32")
            T.annotate_address({p: 0, ch: 256, cf: 8448, out: 24832})
            with T.Scope("V"):
                T.copy(P, p)
                T.copy(C[0, 0, r0], ch)
                T.barrier_all()
                T.copy(ch, cf)
                T.barrier_all()
                # Whole rows are contiguous and 256-byte aligned. A region fill
                # lowers to vector Duplicate rather than scalar Get/SetValue.
                if visible_keys < 64:
                    T.tile.fill(cf[visible_keys:64, 0:64], 0.0)
                T.barrier_all()
                for i, j in T.Parallel(64, 64):
                    cf[i, j] = cf[i, j] * p[i]
                T.barrier_all()
                T.reduce_sum(cf, out, dim=0)
                T.barrier_all()
                T.copy(out, Z[r0])
                T.barrier_all()

    return main


@tilelang.jit(target="ascendc", out_idx=[-2, -1], pass_configs=EXPERT_PASSES)
def value_gate_kernel():

    @T.prim_func
    def main(
        Z: T.Tensor((512,), "float32"),
        W: T.Tensor((1, 128, 512), "float16"),
        G: T.Tensor((1, 1, 1, 128), "float16"),
        U: T.Tensor((128,), "float32"),
        Y: T.Tensor((1, 1, 1, 128), "float16"),
    ):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            v0 = (cid * 2 + vid) * 32
            z = T.alloc_ub((512,), "float32")
            wh = T.alloc_ub((32, 512), "float16")
            wf = T.alloc_ub((32, 512), "float32")
            out = T.alloc_ub((32,), "float32")
            gh = T.alloc_ub((32,), "float16")
            gf = T.alloc_ub((32,), "float32")
            sigmoid = T.alloc_ub((32,), "float32")
            yh = T.alloc_ub((32,), "float16")
            T.annotate_address({z: 0, wh: 2048, wf: 34816, out: 100352, gh: 100480, gf: 100544, sigmoid: 100672, yh: 100800})
            with T.Scope("V"):
                T.copy(Z, z)
                T.copy(W[0, v0, 0], wh)
                T.copy(G[0, 0, 0, v0], gh)
                T.barrier_all()
                T.copy(wh, wf)
                T.copy(gh, gf)
                T.barrier_all()
                for i, j in T.Parallel(32, 512):
                    wf[i, j] = wf[i, j] * z[j]
                T.barrier_all()
                T.reduce_sum(wf, out, dim=1)
                T.barrier_all()
                T.copy(out, U[v0])
                T.barrier_all()
                # Keep gf's sign; explicit tile operations avoid nested scalar exp lowering.
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
                T.copy(yh, Y[0, 0, 0, v0])
                T.barrier_all()

    return main


@tilelang.jit(target="ascendc", out_idx=[-1], pass_configs=EXPERT_PASSES)
def zero_output_kernel():

    @T.prim_func
    def main(
        Q: T.Tensor((1, 1, 1, 128), "float16"),
        Y: T.Tensor((1, 1, 1, 128), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            out = T.alloc_ub((64,), "float16")
            T.annotate_address({out: 0})
            with T.Scope("V"):
                T.tile.fill(out, 0.0)
                T.barrier_all()
                T.copy(out, Y[0, 0, 0, vid * 64])
                T.barrier_all()

    return main


def gated_mla_core(
    q_content: torch.Tensor,
    q_aux: torch.Tensor,
    latent_kv: torch.Tensor,
    key_aux: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    query_start: Sequence[int],
    scale: float | None = None,
    return_intermediates: bool = False,
):
    """Execute the declared L0 domain; reject all other shapes explicitly.

    Intermediate tensors are retained only as diagnostics when requested. JIT
    output allocation uses torch.empty internally; all arithmetic is in kernels.
    """
    spec = validate_inputs(
        q_content,
        q_aux,
        latent_kv,
        key_aux,
        w_k,
        w_v,
        gate_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        query_start=query_start,
        scale=scale,
        require_npu=True,
        allowed_dtypes=(torch.float16,),
    )
    shape = (spec.B, spec.Tq, spec.Tk, spec.H, spec.D, spec.E, spec.R, spec.V)
    if shape != (1, 1, 64, 1, 128, 64, 512, 128):
        raise NotImplementedError("L0 decode supports only B=H=Tq=1, Tk=64, D=128, E=64, R=512, V=128")
    if spec.q_lens[0] == 0:
        fn = zero_output_kernel()
        result = fn(q_content)
        if return_intermediates:
            return result, {}, {"zero_output": fn}
        return result
    visible = min(spec.kv_lens[0], spec.query_start[0] + 1)
    kernels = {
        "q_absorb": q_absorb_kernel(),
        "scores": score_kernel(visible, spec.scale),
        "softmax": softmax_kernel(),
        "latent_readout": latent_readout_kernel(visible),
        "value_gate": value_gate_kernel(),
    }
    q_latent = kernels["q_absorb"](q_content, w_k)
    scores = kernels["scores"](q_latent, q_aux, latent_kv, key_aux)
    probs = kernels["softmax"](scores)
    latent_out = kernels["latent_readout"](probs, latent_kv)
    ungated, result = kernels["value_gate"](latent_out, w_v, gate_logits)
    if return_intermediates:
        intermediates = {
            "q_latent": q_latent,
            "scores": scores,
            "probs": probs,
            "latent_out": latent_out,
            "ungated": ungated,
        }
        return result, intermediates, kernels
    return result


if __name__ == "__main__":
    import torch_npu  # noqa: F401
    from gated_mla_golden import check_case, make_decode_inputs

    if not torch.npu.is_available():
        raise RuntimeError("An Ascend NPU is required")
    torch.npu.set_device(0)
    check_case(make_decode_inputs(0), dict(q_lens=(1,), kv_lens=(64,), query_start=(63,)), repeat=3)
    print("TEST_PASSED")
