"""MHC BWD operator (Sinkhorn backward via implicit CG) for Ascend NPU.

Implements the backward pass of Sinkhorn normalization using implicit
differentiation with the Conjugate Gradient (CG) method.

Reference: tilelang main repo CUDA version examples/deepseek_mhc/example_mhc_bwd.py
           (community-written, reference only — PR #1758)

Math:
  Forward: R = sinkhorn(M)  (doubly stochastic matrix)
  Backward: dL/dM = (dR - x1 - x2^T) * R
    where x1, x2 are solved from the linear system via CG iterations.

Note:
  T.copy on a 3D tensor (e.g. [TS, NS, NS]) only transfers the trailing
  2D tile on Ascend NPU, silently dropping the leading dim. The 3D
  GM<->UB transfers must therefore be split into per-tile 2D copies:
      for i_tile in T.serial(TS):
          T.copy(gm[i_seq * TS + i_tile, :, :], ub[i_tile, :, :])
  Otherwise only the first tile (row 0) is loaded and the CG diverges.

Architecture (pure Vector, no Cube):
  - All buffers in UB (no fragment/L0C)
  - T.macro for matvec_A and dot sub-functions
  - T.tile.fill for initialization (T.fill not on Ascend)
  - T.Parallel for element-wise ops on UB
  - T.reduce_sum with real_shape for reductions

Migration from CUDA:
  1. T.alloc_fragment -> T.alloc_ub (UB scope for Vector ops)
  2. T.fill -> T.tile.fill
  3. T.const -> compile-time Python constant
  4. threads=N -> is_npu=True with (cid, vid) binding
  5. 3D T.Parallel -> T.serial + 2D T.Parallel (Ascend only supports 1D/2D)
  6. device="cuda" -> device="npu"
  7. torch.cuda.synchronize -> torch.npu.synchronize
  8. Removed tilelang.autotune (fixed config for simplicity)
"""

import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

EPS = 1e-10


@T.macro
def matvec_A(R, x1, x2, buf, y1, y2, tilesize, n_stream):
    for i_tile in T.serial(tilesize):
        for i, j in T.Parallel(n_stream, n_stream):
            buf[i, j] = R[i_tile, i, j] * x2[i_tile, j]
        T.reduce_sum(buf, y1[i_tile, :], dim=-1, real_shape=[n_stream, n_stream])

        for i, j in T.Parallel(n_stream, n_stream):
            buf[i, j] = R[i_tile, i, j] * x1[i_tile, i]
        T.reduce_sum(buf, y2[i_tile, :], dim=-2, real_shape=[n_stream, n_stream])

        for i in T.Parallel(n_stream):
            y1[i_tile, i] += x1[i_tile, i]
            y2[i_tile, i] += x2[i_tile, i]


@T.macro
def dot(x1, x2, y1, y2, buf, out, tilesize, n_stream):
    for i_tile, i in T.Parallel(tilesize, n_stream):
        buf[i_tile, i] = x1[i_tile, i] * y1[i_tile, i] + x2[i_tile, i] * y2[i_tile, i]
    T.reduce_sum(buf, out, dim=-1, real_shape=[tilesize, n_stream])


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def sinkhorn_bwd_implicit_cg(n_stream, tilesize=8):
    """Sinkhorn backward via implicit CG on Ascend NPU.

    Args:
        n_stream: matrix dimension (hc * hc after reshape)
        tilesize: number of rows per kernel block (compile-time constant)
    """
    seqlen = T.symbolic("seqlen")
    dtype = "float"
    TS = tilesize
    NS = n_stream
    tensor_shape = [seqlen, NS, NS]

    @T.prim_func
    def main(
        out: T.Tensor(tensor_shape, dtype),
        dout: T.Tensor(tensor_shape, dtype),
        res: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seqlen, TS), is_npu=True) as (i_seq, vid):
            if vid == 0:
                R = T.alloc_ub([TS, NS, NS], dtype=dtype)
                dR = T.alloc_ub([TS, NS, NS], dtype=dtype)
                RdR = T.alloc_ub([TS, NS, NS], dtype=dtype)
                res_tile = T.alloc_ub([TS, NS, NS], dtype=dtype)
                b1 = T.alloc_ub([TS, NS], dtype=dtype)
                b2 = T.alloc_ub([TS, NS], dtype=dtype)
                x1 = T.alloc_ub([TS, NS], dtype=dtype)
                x2 = T.alloc_ub([TS, NS], dtype=dtype)
                r1 = T.alloc_ub([TS, NS], dtype=dtype)
                r2 = T.alloc_ub([TS, NS], dtype=dtype)
                p1 = T.alloc_ub([TS, NS], dtype=dtype)
                p2 = T.alloc_ub([TS, NS], dtype=dtype)
                alpha = T.alloc_ub([TS, NS], dtype=dtype)
                beta = T.alloc_ub([TS, NS], dtype=dtype)
                r_normsq = T.alloc_ub([TS], dtype=dtype)
                r_new_normsq = T.alloc_ub([TS], dtype=dtype)
                Ap1 = T.alloc_ub([TS, NS], dtype=dtype)
                Ap2 = T.alloc_ub([TS, NS], dtype=dtype)
                pAp = T.alloc_ub([TS], dtype=dtype)

                buf1 = T.alloc_ub([NS, NS], dtype=dtype)
                buf2 = T.alloc_ub([TS, NS], dtype=dtype)

                for i_tile in T.serial(TS):
                    T.copy(out[i_seq * TS + i_tile, :, :], R[i_tile, :, :])
                for i_tile in T.serial(TS):
                    T.copy(dout[i_seq * TS + i_tile, :, :], dR[i_tile, :, :])

                for i_tile in T.serial(TS):
                    for i_nx, i_ny in T.Parallel(NS, NS):
                        RdR[i_tile, i_nx, i_ny] = R[i_tile, i_nx, i_ny] * dR[i_tile, i_nx, i_ny]

                    tmp2d = T.alloc_ub([NS, NS], dtype=dtype)
                    T.copy(RdR[i_tile, 0, 0], tmp2d)
                    T.reduce_sum(tmp2d, b1[i_tile, :], dim=-1, real_shape=[NS, NS])
                    T.reduce_sum(tmp2d, b2[i_tile, :], dim=-2, real_shape=[NS, NS])

                T.tile.fill(x1, 0.0)
                T.tile.fill(x2, 0.0)

                matvec_A(R, x1, x2, buf1, r1, r2, TS, NS)

                for i_tile, i_n in T.Parallel(TS, NS):
                    r1[i_tile, i_n] = b1[i_tile, i_n] - r1[i_tile, i_n]

                for i_tile, i_n in T.Parallel(TS, NS):
                    r2[i_tile, i_n] = b2[i_tile, i_n] - r2[i_tile, i_n]

                T.copy(r1, p1)
                T.copy(r2, p2)

                dot(r1, r2, r1, r2, buf2, r_normsq, TS, NS)

                for _ in T.serial(2 * NS):
                    matvec_A(R, p1, p2, buf1, Ap1, Ap2, TS, NS)

                    dot(p1, p2, Ap1, Ap2, buf2, pAp, TS, NS)

                    for i_tile, i_n in T.Parallel(TS, NS):
                        alpha[i_tile, i_n] = r_normsq[i_tile] / (pAp[i_tile] + EPS)
                    for i_tile, i_n in T.Parallel(TS, NS):
                        x1[i_tile, i_n] += alpha[i_tile, i_n] * p1[i_tile, i_n]
                    for i_tile, i_n in T.Parallel(TS, NS):
                        x2[i_tile, i_n] += alpha[i_tile, i_n] * p2[i_tile, i_n]
                    for i_tile, i_n in T.Parallel(TS, NS):
                        r1[i_tile, i_n] -= alpha[i_tile, i_n] * Ap1[i_tile, i_n]
                    for i_tile, i_n in T.Parallel(TS, NS):
                        r2[i_tile, i_n] -= alpha[i_tile, i_n] * Ap2[i_tile, i_n]

                    dot(r1, r2, r1, r2, buf2, r_new_normsq, TS, NS)

                    for i_tile, i_n in T.Parallel(TS, NS):
                        beta[i_tile, i_n] = r_new_normsq[i_tile] / (r_normsq[i_tile] + EPS)
                    for i_tile, i_n in T.Parallel(TS, NS):
                        p1[i_tile, i_n] = r1[i_tile, i_n] + beta[i_tile, i_n] * p1[i_tile, i_n]
                    for i_tile, i_n in T.Parallel(TS, NS):
                        p2[i_tile, i_n] = r2[i_tile, i_n] + beta[i_tile, i_n] * p2[i_tile, i_n]

                    T.copy(r_new_normsq, r_normsq)

                for i_tile in T.serial(TS):
                    for i_nx, i_ny in T.Parallel(NS, NS):
                        res_tile[i_tile, i_nx, i_ny] = (dR[i_tile, i_nx, i_ny] - x1[i_tile, i_nx] - x2[i_tile, i_ny]) * R[
                            i_tile, i_nx, i_ny
                        ]

                for i_tile in T.serial(TS):
                    T.copy(res_tile[i_tile, :, :], res[i_seq * TS + i_tile, :, :])

    return main


# ============================================================
# Golden reference
# ============================================================


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


def sinkhorn_bwd_ref(out, dout, n_stream, tilesize=8):
    """PyTorch reference for sinkhorn backward via CG."""
    seqlen = out.shape[0]
    m_num = (seqlen + tilesize - 1) // tilesize
    res = torch.empty_like(out)

    for i_seq in range(m_num):
        start = i_seq * tilesize
        end = min(start + tilesize, seqlen)
        ts = end - start

        R = out[start:end].cpu()
        dR = dout[start:end].cpu()

        RdR = R * dR
        b1 = RdR.sum(-1)
        b2 = RdR.sum(-2)

        x1 = torch.zeros(ts, n_stream)
        x2 = torch.zeros(ts, n_stream)

        def matvec(x1, x2, R=R):
            y1 = (R * x2.unsqueeze(1)).sum(-1) + x1
            y2 = (R * x1.unsqueeze(2)).sum(-2) + x2
            return y1, y2

        r1, r2 = matvec(x1, x2)
        r1 = b1 - r1
        r2 = b2 - r2

        p1 = r1.clone()
        p2 = r2.clone()

        r_normsq = (r1 * r2).sum(-1)

        for _ in range(2 * n_stream):
            Ap1, Ap2 = matvec(p1, p2)
            pAp = (p1 * Ap1 + p2 * Ap2).sum(-1)
            alpha = r_normsq / (pAp + EPS)
            x1 += alpha.unsqueeze(-1) * p1
            x2 += alpha.unsqueeze(-1) * p2
            r1 -= alpha.unsqueeze(-1) * Ap1
            r2 -= alpha.unsqueeze(-1) * Ap2

            r_new_normsq = (r1 * r2).sum(-1)
            beta = r_new_normsq / (r_normsq + EPS)
            p1 = r1 + beta.unsqueeze(-1) * p1
            p2 = r2 + beta.unsqueeze(-1) * p2
            r_normsq = r_new_normsq

        res[start:end] = (dR - x1.unsqueeze(2) - x2.unsqueeze(1)) * R

    return res


# ============================================================
# Tests
# ============================================================


def generate_test_data(seqlen, n_stream, device="npu"):
    torch.random.manual_seed(42)
    dist = torch.distributions.uniform.Uniform(0.0, 4.0)
    M = dist.sample((seqlen, n_stream, n_stream)).to(device)
    M.requires_grad_()
    return M


def test():
    print("=" * 60)
    print("MHC BWD (Sinkhorn implicit CG) test (Ascend NPU)")
    print("=" * 60)

    seqlen = 256
    n_stream = 16
    tilesize = 8
    iters = 20

    M = generate_test_data(seqlen, n_stream)
    R, P = sinkhorn_forward(M, iters)
    loss_weight = torch.randn_like(R)

    loss_a = (R * loss_weight).sum()
    loss_a.backward()
    grad_M_autograd = M.grad.detach().clone()

    grad_R = loss_weight

    kernel = sinkhorn_bwd_implicit_cg(n_stream, tilesize)
    grad_M_implicit = kernel(R.detach(), grad_R)

    abs_diff = (grad_M_autograd.cpu() - grad_M_implicit.cpu()).abs()
    rel_diff = abs_diff / (torch.maximum(grad_M_autograd.cpu().abs(), grad_M_implicit.cpu().abs()) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"  seqlen={seqlen}, n_stream={n_stream}, tilesize={tilesize}")
    print(f"  max_abs_diff = {max_abs_diff:.6e}")
    print(f"  mean_abs_diff = {mean_abs_diff:.6e}")
    print(f"  max_rel_diff = {max_rel_diff:.6e}")

    print(f"\n  Grad (autograd) sample:\n{grad_M_autograd[0, :3, :3]}")
    print(f"\n  Grad (implicit) sample:\n{grad_M_implicit[0, :3, :3]}")

    if max_abs_diff < 1e-3:
        print("\n  PASSED (implicit CG matches autograd within fp32 tolerance)")
        print("Kernel Output Match!")
    else:
        print(f"\n  FAILED (max_abs_diff={max_abs_diff:.6e} > 1e-3)")

    print("=" * 60)


if __name__ == "__main__":
    tilelang.disable_cache()
    test()
