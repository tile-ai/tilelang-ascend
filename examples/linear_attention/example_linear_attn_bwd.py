import tilelang
import tilelang.language as T
import torch
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(
    workspace_idx=[7, 10, 11, 12],
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
        Q: T.Tensor((B, S, H, DK), dtype),
        K: T.Tensor((B, S, H, DK), dtype),
        V: T.Tensor((B, S, H, DV), dtype),
        dO: T.Tensor((B, S, H, DV), dtype),
        dQ: T.Tensor((B, S, H, DK), accum_dtype),
        dK: T.Tensor((B, S, H, DK), accum_dtype),
        dV: T.Tensor((B, S, H, DV), accum_dtype),
        workspace1: T.Tensor((total_blocks * NK * NV, chunk_size, chunk_size), accum_dtype),
        workspace2: T.Tensor((total_blocks * NK * NV, BV, BK), accum_dtype),
        workspace3: T.Tensor((total_blocks * NK * NV, BK, BV), accum_dtype),
        workspace4: T.Tensor((total_blocks * NK * NV, chunk_size, chunk_size), dtype),
        workspace5: T.Tensor((total_blocks * NK * NV, BV, BK), dtype),
        workspace6: T.Tensor((total_blocks * NK * NV, BK, BV), dtype),
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            i_bh = cid
            i_b = i_bh // H
            i_h = i_bh % H
            blk_off = cid * NK * NV

            with T.Scope("C"):
                k_l1 = T.alloc_shared([chunk_size, BK], dtype)
                v_l1 = T.alloc_shared([chunk_size, BV], dtype)
                do_l1 = T.alloc_shared([chunk_size, BV], dtype)
                q_l1 = T.alloc_shared([chunk_size, BK], dtype)
                ds_l1 = T.alloc_shared([chunk_size, chunk_size], dtype)
                h_l1 = T.alloc_shared([BV, BK], dtype)
                dh_l1 = T.alloc_shared([BK, BV], dtype)
                ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
                dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
                dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
                h = T.alloc_fragment([BV, BK], accum_dtype)
                dh = T.alloc_fragment([BK, BV], accum_dtype)

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
                            T.copy(ds, workspace1[bv_idx, :, :])
                            T.set_cross_flag("FIX", 2)
                            T.wait_cross_flag(3)
                            T.copy(workspace4[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, k_l1, dq, init=True)
                            T.copy(workspace5[bv_idx, :, :], h_l1)
                            T.gemm_v0(do_l1, h_l1, dq, init=False)
                            T.gemm_v0(v_l1, k_l1, h, transpose_A=True, init=(i == 0))
                            T.copy(h, workspace2[bv_idx, :, :])
                            T.copy(dq, workspace1[bv_idx, :, :])
                            T.set_cross_flag("FIX", 4)

                        T.wait_cross_flag(1)

                        # === Pass 2: dK, dV ===
                        for i in T.serial(NT):
                            T.wait_cross_flag(5)
                            start = NT - 1 - i
                            T.set_cross_flag("FIX", 6)
                            T.wait_cross_flag(7)
                            T.copy(workspace4[bv_idx, :, :], q_l1)
                            T.copy(K[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k_l1)
                            T.copy(V[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v_l1)
                            T.copy(dO[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do_l1)
                            T.gemm_v0(v_l1, do_l1, ds, transpose_B=True, init=True)
                            T.copy(ds, workspace1[bv_idx, :, :])
                            T.set_cross_flag("FIX", 8)
                            T.wait_cross_flag(9)
                            T.copy(workspace4[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, q_l1, dk, init=True)
                            T.copy(workspace6[bv_idx, :, :], dh_l1)
                            T.gemm_v0(v_l1, dh_l1, dk, transpose_B=True, init=False)
                            T.gemm_v0(k_l1, q_l1, ds, transpose_B=True, init=True)
                            T.copy(ds, workspace1[bv_idx, :, :])
                            T.set_cross_flag("FIX", 10)
                            T.wait_cross_flag(11)
                            T.copy(workspace4[bv_idx, :, :], ds_l1)
                            T.gemm_v0(ds_l1, do_l1, dv, init=True)
                            T.gemm_v0(k_l1, dh_l1, dv, init=False)
                            T.gemm_v0(q_l1, do_l1, dh, transpose_A=True, init=(i == 0))
                            T.copy(dh, workspace3[bv_idx, :, :])
                            T.copy(dk, workspace1[bv_idx, :, :])
                            T.copy(dv, workspace2[bv_idx, :, :])
                            T.set_cross_flag("FIX", 12)

                        T.wait_cross_flag(5)

            with T.Scope("V"):
                ds_ub = T.alloc_shared([sub_C, chunk_size], accum_dtype)
                ds_half = T.alloc_shared([sub_C, chunk_size], dtype)
                dq_ub = T.alloc_shared([sub_C, BK], accum_dtype)
                dk_ub = T.alloc_shared([sub_C, BK], accum_dtype)
                dv_ub = T.alloc_shared([sub_C, BV], accum_dtype)
                q_ub = T.alloc_shared([sub_C, BK], dtype)
                q_cal = T.alloc_shared([sub_C, BK], accum_dtype)
                dq_tmp = T.alloc_shared([sub_C, BK], accum_dtype)
                dk_tmp = T.alloc_shared([sub_C, BK], accum_dtype)
                dv_tmp = T.alloc_shared([sub_C, BV], accum_dtype)

                T.wait_cross_flag(0)

                T.tile.fill(ds_ub, 0.0)
                T.tile.fill(dv_ub, 0.0)
                T.tile.fill(ds_half, 0.0)

                T.set_cross_flag("MTE3", 1)

                for i_k in T.serial(NK):
                    kv_base = blk_off + i_k * NV
                    for i_v in T.serial(NV):
                        bv_idx = kv_base + i_v

                        # === Pass 1: mask ds, scale dq, RMW to dQ ===
                        for chk in T.serial(NT):
                            T.wait_cross_flag(2)
                            T.copy(workspace1[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C < c:
                                        ds_ub[r, c] = 0.0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, workspace4[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.copy(workspace2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, workspace5[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])

                            T.set_cross_flag("MTE3", 3)

                            T.wait_cross_flag(4)

                            # dQ
                            T.copy(workspace1[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dq_ub)
                            T.tile.mul(dq_ub, dq_ub, scale)
                            T.tile.atomic_add(
                                dQ[
                                    i_b,
                                    chk * chunk_size + vid * sub_C : chk * chunk_size + (vid + 1) * sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                dq_ub,
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
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + (vid + 1) * sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                q_ub,
                            )
                            T.tile.cast(q_cal, q_ub, "CAST_NONE", sub_C * BK)

                            T.tile.mul(q_cal, q_cal, scale)

                            T.copy(q_cal, ds_half)
                            T.copy(ds_half, workspace4[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 7)

                            T.wait_cross_flag(8)
                            T.copy(workspace3[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, workspace6[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.copy(workspace1[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C > c:
                                        ds_ub[r, c] = 0.0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, workspace4[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 9)

                            T.wait_cross_flag(10)
                            T.copy(workspace1[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], ds_ub)
                            for r in range(sub_C):
                                for c in range(chunk_size):
                                    if r + vid * sub_C > c:
                                        ds_ub[r, c] = 0.0
                            T.copy(ds_ub, ds_half)
                            T.copy(ds_half, workspace4[bv_idx, vid * sub_C : (vid + 1) * sub_C, :])
                            T.set_cross_flag("MTE3", 11)

                            T.wait_cross_flag(12)

                            # dK
                            T.copy(workspace1[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dk_ub)
                            T.tile.atomic_add(
                                dK[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + (vid + 1) * sub_C,
                                    i_h,
                                    i_k * BK : (i_k + 1) * BK,
                                ],
                                dk_ub,
                            )

                            # dV
                            T.copy(workspace2[bv_idx, vid * sub_C : (vid + 1) * sub_C, :], dv_ub)
                            T.tile.atomic_add(
                                dV[
                                    i_b,
                                    rev * chunk_size + vid * sub_C : rev * chunk_size + (vid + 1) * sub_C,
                                    i_h,
                                    i_v * BV : (i_v + 1) * BV,
                                ],
                                dv_ub,
                            )
                            T.set_cross_flag("MTE3", 5)

                        if i_v < NV - 1 or i_k < NK - 1:
                            T.set_cross_flag("MTE3", 1)

    return main


def fla_fused_chunk_bwd(q: Tensor, k: Tensor, v: Tensor, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    B, S, H, DK = q.shape
    DV = v.shape[-1]

    kernel = fla_fused_chunk_bwd_kernel(B, S, H, DK, DV)

    dQ = torch.zeros(B, S, H, DK, dtype=torch.float32).npu()
    dK = torch.zeros(B, S, H, DK, dtype=torch.float32).npu()
    dV = torch.zeros(B, S, H, DV, dtype=torch.float32).npu()
    workspace2 = torch.zeros(B * H * tilelang.cdiv(DK, 64) * tilelang.cdiv(DV, 64), 64, 64, dtype=torch.float32).npu()
    workspace3 = torch.zeros(B * H * tilelang.cdiv(DK, 64) * tilelang.cdiv(DV, 64), 64, 64, dtype=torch.float32).npu()
    torch.npu.synchronize()

    kernel(q, k, v, dO, dQ, dK, dV, workspace2, workspace3)

    return dQ.to(q.dtype), dK.to(q.dtype), dV.to(q.dtype)


def ref_bwd_program(q: Tensor, k: Tensor, v: Tensor, dO: Tensor, scale: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
    q_cpu = q.cpu().clone().float().requires_grad_(True)
    k_cpu = k.cpu().clone().float().requires_grad_(True)
    v_cpu = v.cpu().clone().float().requires_grad_(True)
    dO_cpu = dO.cpu().clone().float()
    if scale is None:
        scale = float(q_cpu.shape[-1] ** -0.5)
    chunk_size = 64
    B, S, H, D = q_cpu.shape
    NT = S // chunk_size
    q_s = q_cpu * scale
    q_chunks = q_s.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, D)
    k_chunks = k_cpu.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, D)
    v_chunks = v_cpu.permute(0, 2, 1, 3).reshape(B, H, NT, chunk_size, D)
    kv = k_chunks.transpose(-1, -2) @ v_chunks
    kv = kv.cumsum(2)
    kv_shifted = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q_chunks @ kv_shifted
    intra_attn = q_chunks @ k_chunks.transpose(-1, -2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.float32), diagonal=1)
    intra_attn.masked_fill_(mask.bool(), 0)
    intra = intra_attn @ v_chunks
    o_chunks = inter + intra
    o = o_chunks.reshape(B, H, S, D).permute(0, 2, 1, 3)
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
        ok = torch.allclose(a.cpu().float(), b, atol=1e-2, rtol=1e-2)
        if not ok:
            err = (a.cpu().float() - b).abs()
            print(f"{name} mismatch: max_err={err.max():.6f}, mean_err={err.mean():.6f}")
        return ok

    assert check("dQ", dq, ref_dq), "dQ mismatch"
    assert check("dK", dk, ref_dk), "dK mismatch"
    assert check("dV", dv, ref_dv), "dV mismatch"
    print("Linear Attention Backward Test Passed!")


if __name__ == "__main__":
    for B, S, H, D in [(1, 128, 8, 64), (2, 1024, 16, 128), (8, 1024, 32, 128)]:
        print(f"Testing B={B}, S={S}, H={H}, D={D}...")
        main(B, S, H, D)
