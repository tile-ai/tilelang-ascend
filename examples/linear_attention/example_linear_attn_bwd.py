import tilelang
import tilelang.language as T
import torch
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F


tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(
    out_idx=[4, 5, 6],
    workspace_idx=[7, 8, 9, 10, 11, 12],
    pass_configs=pass_configs,
)
def fla_fused_chunk_bwd_kernel(
    B: int,
    S: int,
    H: int,
    DK: int,
    DV: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
    scale: Optional[float] = None,
):
    if scale is None:
        scale = DK**-0.5
    chunk_size = 64
    BK = BV = 64
    NK = tilelang.cdiv(DK, BK)
    NV = tilelang.cdiv(DV, BV)
    NT = tilelang.cdiv(S, chunk_size)
    total_blocks = B * H
    VEC_NUM = 2
    sub_C = chunk_size // VEC_NUM

    @T.prim_func
    def main(
        Q: T.Tensor([B, S, H, DK], dtype),
        K: T.Tensor([B, S, H, DK], dtype),
        V: T.Tensor([B, S, H, DV], dtype),
        dO: T.Tensor([B, S, H, DV], dtype),
        dQ: T.Tensor([B, S, H, DK], accum_dtype),
        dK: T.Tensor([B, S, H, DK], accum_dtype),
        dV: T.Tensor([B, S, H, DV], accum_dtype),
        ws_s: T.Tensor([total_blocks * NK * NV, chunk_size, chunk_size], accum_dtype),
        ws_h: T.Tensor([total_blocks * NK * NV, BV, BK], accum_dtype),
        ws_dh: T.Tensor([total_blocks * NK * NV, BK, BV], accum_dtype),
        ws_s2: T.Tensor([total_blocks * NK * NV, chunk_size, chunk_size], dtype),
        ws_h2: T.Tensor([total_blocks * NK * NV, BV, BK], dtype),
        ws_dh2: T.Tensor([total_blocks * NK * NV, BK, BV], dtype),
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            i_bh = cid
            i_b = i_bh // H
            i_h = i_bh % H
            blk_off = cid * NK * NV

            with T.Scope("C"):
                k_l1 = T.alloc_L1([chunk_size, BK], dtype)
                v_l1 = T.alloc_L1([chunk_size, BV], dtype)
                do_l1 = T.alloc_L1([chunk_size, BV], dtype)
                q_l1 = T.alloc_L1([chunk_size, BK], dtype)
                ds_l1 = T.alloc_L1([chunk_size, chunk_size], dtype)
                h_l1 = T.alloc_L1([BV, BK], dtype)
                dh_l1 = T.alloc_L1([BK, BV], dtype)
                ds = T.alloc_L0C([chunk_size, chunk_size], accum_dtype)
                dq = T.alloc_L0C([chunk_size, BK], accum_dtype)
                dk = T.alloc_L0C([chunk_size, BK], accum_dtype)
                dv = T.alloc_L0C([chunk_size, BV], accum_dtype)
                h = T.alloc_L0C([BV, BK], accum_dtype)
                dh = T.alloc_L0C([BK, BV], accum_dtype)

                T.set_cross_flag("FIX", 0)

                for i_k in T.serial(NK):
                    kv_base = blk_off + i_k * NV
                    for i_v in T.serial(NV):
                        bv_idx = kv_base + i_v

                        # === Pass 1: dQ ===
                        for i in T.serial(NT):
                            T.wait_cross_flag(1)
                            T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k_l1)
                            T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v_l1)
                            T.copy(dO[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do_l1)
                            T.gemm_v0(do_l1, v_l1, ds, transpose_B=True, init=True)
                            T.copy(ds, ws_s[bv_idx, :, :])
                            T.set_cross_flag("FIX", 2)
                            T.wait_cross_flag(3)
                            T.copy(ws_s2[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, k_l1, dq, init=True)
                            T.copy(ws_h2[bv_idx, :, :], h_l1)
                            T.gemm_v0(do_l1, h_l1, dq, init=False)
                            T.gemm_v0(v_l1, k_l1, h, transpose_A=True, init=(i == 0))
                            T.copy(h, ws_h[bv_idx, :, :])
                            T.copy(dq, ws_s[bv_idx, :, :])
                            T.set_cross_flag("FIX", 4)

                        T.wait_cross_flag(1)

                        # === Pass 2: dK, dV ===
                        for i in T.serial(NT):
                            T.wait_cross_flag(5)
                            start = NT - 1 - i
                            T.set_cross_flag("FIX", 6)
                            T.wait_cross_flag(7)
                            T.copy(ws_s2[bv_idx, :, :], q_l1)
                            T.copy(K[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k_l1)
                            T.copy(V[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v_l1)
                            T.copy(dO[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do_l1)
                            T.gemm_v0(v_l1, do_l1, ds, transpose_B=True, init=True)
                            T.copy(ds, ws_s[bv_idx, :, :])
                            T.set_cross_flag("FIX", 8)
                            T.wait_cross_flag(9)
                            T.copy(ws_s2[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, q_l1, dk, init=True)
                            T.copy(ws_dh2[bv_idx, :, :], dh_l1)
                            T.gemm_v0(v_l1, dh_l1, dk, transpose_B=True, init=False)
                            T.gemm_v0(k_l1, q_l1, ds, transpose_B=True, init=True)
                            T.copy(ds, ws_s[bv_idx, :, :])
                            T.set_cross_flag("FIX", 10)
                            T.wait_cross_flag(11)
                            T.copy(ws_s2[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, do_l1, dv, init=True)
                            T.copy(ws_dh2[bv_idx, :, :], dh_l1)
                            T.gemm_v0(k_l1, dh_l1, dv, init=False)
                            T.gemm_v0(q_l1, do_l1, dh, transpose_A=True, init=(i == 0))
                            T.copy(dh, ws_dh[bv_idx, :, :])
                            T.copy(dk, ws_s[bv_idx, :, :])
                            T.copy(dv, ws_h[bv_idx, :, :])
                            T.set_cross_flag("FIX", 12)

                        T.wait_cross_flag(5)

            with T.Scope("V"):
                ds_ub = T.alloc_ub([sub_C, chunk_size], accum_dtype)
                ds_half = T.alloc_ub([sub_C, chunk_size], dtype)
                dq_ub = T.alloc_ub([sub_C, BK], accum_dtype)
                dk_ub = T.alloc_ub([sub_C, BK], accum_dtype)
                dv_ub = T.alloc_ub([sub_C, BV], accum_dtype)
                q_ub = T.alloc_ub([sub_C, BK], dtype)
                q_cal = T.alloc_ub([sub_C, BK], accum_dtype)
                dq_tmp = T.alloc_ub([sub_C, BK], accum_dtype)
                dk_tmp = T.alloc_ub([sub_C, BK], accum_dtype)
                dv_tmp = T.alloc_ub([sub_C, BV], accum_dtype)
                h_half = T.alloc_ub([sub_C, chunk_size], dtype)
                dh_half = T.alloc_ub([sub_C, chunk_size], dtype)

                T.wait_cross_flag(0)

                # Init h, dh workspaces
                T.tile.fill(ds_ub, 0)
                T.tile.fill(h_half, 0)
                for init_i in T.serial(NK * NV):
                    T.copy(ds_ub, ws_h[blk_off + init_i, vid * sub_C : (vid + 1) * sub_C, :])
                    T.copy(h_half, ws_h2[blk_off + init_i, vid * sub_C : (vid + 1) * sub_C, :])
                    T.copy(ds_ub, ws_dh[blk_off + init_i, vid * sub_C : (vid + 1) * sub_C, :])
                    T.copy(h_half, ws_dh2[blk_off + init_i, vid * sub_C : (vid + 1) * sub_C, :])

                T.pipe_barrier("mte3")
                T.set_cross_flag("MTE3", 1)

                for i_k in T.serial(NK):
                    kv_base = blk_off + i_k * NV
                    for i_v in T.serial(NV):
                        bv_idx = kv_base + i_v

                        # === Pass 1: mask ds, scale dq, RMW to dQ ===
                        for chk in T.serial(NT):
                            T.wait_cross_flag(2)
                            T.copy(ws_s[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C < c:
                                        ds_ub[r, c] = 0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, ws_s2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 3)

                            # Cast h for GEMM read
                            T.copy(ws_h[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            T.copy(ds_ub, h_half)
                            T.copy(h_half, ws_h2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])

                            T.wait_cross_flag(4)
                            T.copy(ws_s[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dq_ub)
                            for r, c in T.Parallel(sub_C, BK):
                                dq_ub[r, c] = dq_ub[r, c] * scale
                            T.copy(
                                dQ[
                                    i_b,
                                    chk * chunk_size + vid * sub_C : chk * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                dq_tmp,
                            )
                            for r, c in T.Parallel(sub_C, BK):
                                dq_tmp[r, c] = dq_tmp[r, c] + dq_ub[r, c]
                            T.copy(
                                dq_tmp,
                                dQ[
                                    i_b,
                                    chk * chunk_size + vid * sub_C : chk * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                            )
                            T.set_cross_flag("MTE3", 1)

                        # === Pass 2: scale q, mask s, RMW dk/dv ===
                        T.set_cross_flag("MTE3", 5)

                        for chk in T.serial(NT):
                            rev = NT - 1 - chk
                            T.wait_cross_flag(6)
                            T.copy(
                                Q[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                q_ub,
                            )
                            T.tile.cast(q_cal, q_ub, "CAST_NONE", sub_C * BK)
                            for r, c in T.Parallel(sub_C, BK):
                                q_cal[r, c] = q_cal[r, c] * scale
                            T.copy(q_cal, ds_half)
                            T.copy(ds_half, ws_s2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 7)

                            T.wait_cross_flag(8)
                            T.copy(ws_s[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C > c:
                                        ds_ub[r, c] = 0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, ws_s2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 9)

                            # Cast dh for GEMM read
                            T.copy(ws_dh[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            T.copy(ds_ub, dh_half)
                            T.copy(dh_half, ws_dh2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])

                            T.wait_cross_flag(10)
                            T.copy(ws_s[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C > c:
                                        ds_ub[r, c] = 0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, ws_s2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 11)

                            T.wait_cross_flag(12)
                            T.copy(ws_s[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dk_ub)
                            T.copy(
                                dK[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                dk_tmp,
                            )
                            for r, c in T.Parallel(sub_C, BK):
                                dk_tmp[r, c] = dk_tmp[r, c] + dk_ub[r, c]
                            T.copy(
                                dk_tmp,
                                dK[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                            )
                            T.copy(ws_h[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dv_ub)
                            T.copy(
                                dV[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_v * BV : (i_v + 1) * BV,
                                ],
                                dv_tmp,
                            )
                            for r, c in T.Parallel(sub_C, BV):
                                dv_tmp[r, c] = dv_tmp[r, c] + dv_ub[r, c]
                            T.copy(
                                dv_tmp,
                                dV[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + vid * sub_C + sub_C,
                                    i_h,
                                    i_v * BV : (i_v + 1) * BV,
                                ],
                            )
                            T.set_cross_flag("MTE3", 5)

                        if i_v < NV - 1 or i_k < NK - 1:
                            T.set_cross_flag("MTE3", 1)

    return main


def fla_fused_chunk_bwd(q: Tensor, k: Tensor, v: Tensor, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    B, S, H, DK = q.shape
    DV = v.shape[-1]
    kernel = fla_fused_chunk_bwd_kernel(B, S, H, DK, DV)
    dQ, dK, dV = kernel(q, k, v, dO)
    return dQ, dK, dV


def ref_bwd_program(q: Tensor, k: Tensor, v: Tensor, dO: Tensor, scale: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
    q_cpu = q.cpu().clone().float().requires_grad_(True)
    k_cpu = k.cpu().clone().float().requires_grad_(True)
    v_cpu = v.cpu().clone().float().requires_grad_(True)
    dO_cpu = dO.cpu().clone().float()
    if scale is None:
        scale = float(q_cpu.shape[-1] ** -0.5)
    chunk_size = 64
    B, S, H, D = q_cpu.shape
    DV = v_cpu.shape[-1]
    NT = S // chunk_size
    q_s = q_cpu * scale
    # q: [B, S, H, D]. Permute H to before S, then reshape S into NT*C
    q_chunks = q_s.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, D)  # [B,H,NT,C,DK]
    k_chunks = k_cpu.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, D)  # [B,H,NT,C,DK]
    v_chunks = v_cpu.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, DV)  # [B,H,NT,C,DV]
    kv = k_chunks.transpose(-1, -2) @ v_chunks  # [B, H, NT, DK, DV]
    kv = kv.cumsum(2)
    kv_shifted = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q_chunks @ kv_shifted  # [B, H, NT, C, D]
    intra_attn = q_chunks @ k_chunks.transpose(-1, -2)  # [B, H, NT, C, C]
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.float32), diagonal=1)
    intra_attn.masked_fill_(mask.bool(), 0)
    intra = intra_attn @ v_chunks  # [B, H, NT, C, D]
    o_chunks = inter + intra
    o = o_chunks.reshape(B, H, S, D).permute(0, 2, 1, 3)  # [B,S,H,D]
    o.backward(dO_cpu, retain_graph=True)
    return q_cpu.grad, k_cpu.grad, v_cpu.grad


def main(B=1, S=128, H=1, D=64):
    torch.manual_seed(0)
    DV = D
    q = torch.randn(B, S, H, D, dtype=torch.float16).npu()
    k = torch.randn(B, S, H, D, dtype=torch.float16).npu()
    v = torch.randn(B, S, H, DV, dtype=torch.float16).npu()
    dO = torch.randn(B, S, H, DV, dtype=torch.float16).npu()

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    dq, dk, dv = fla_fused_chunk_bwd(q, k, v, dO)
    torch.npu.synchronize()
    ref_dq, ref_dk, ref_dv = ref_bwd_program(q, k, v, dO)

    def check(name, a, b):
        ok = torch.allclose(a.cpu().float(), b, atol=5e-2, rtol=5e-2)
        if not ok:
            err = (a.cpu().float() - b).abs()
            print(f"{name} mismatch: max_err={err.max():.6f}, mean_err={err.mean():.6f}")
        return ok

    assert check("dQ", dq, ref_dq), "dQ mismatch"
    assert check("dK", dk, ref_dk), "dK mismatch"
    assert check("dV", dv, ref_dv), "dV mismatch"
    print("All Test Passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8, help="Batch size")
    parser.add_argument("--S", type=int, default=1024, help="Seq len")
    parser.add_argument("--H", type=int, default=32, help="Num heads")
    parser.add_argument("--D", type=int, default=128, help="Head dim")
    args = parser.parse_args()
    main(args.B, args.S, args.H, args.D)
