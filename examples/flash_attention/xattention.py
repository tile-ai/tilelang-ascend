import time, statistics
import tilelang
from tilelang import language as T
import torch

torch.set_default_device("npu")
torch.manual_seed(42)
tilelang.disable_cache()
V8 = 8
NEG_INF = -(2.0**30)
PRE = 1
NS = 3
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False, tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True}


@tilelang.jit(out_idx=[5], workspace_idx=[6, 7, 8], target="pto", pass_configs=pass_configs)
def x_attention(kv_heads, dim, beam_size, shared_block_tokens, max_decode_step, q_heads_per_kv=2, fuse=16, block_M=16, block_K=128):
    BM, BK, D, G, BS, MDS = block_M, block_K, dim, kv_heads, beam_size, max_decode_step
    ST = shared_block_tokens
    sm_scale = 1.0 / (float(D) ** 0.5)
    dt, adt = "float16", "float"
    ubn = T.symbolic("UBN")
    H, NT, NQ = q_heads_per_kv, BS * G // fuse, fuse * q_heads_per_kv
    NQT, BPT = NQ // BM, BM // H

    @T.prim_func
    def main(
        Q: T.Tensor([G, BS * H, D], dt),
        SK: T.Tensor([G, ST, D], dt),
        SV: T.Tensor([G, ST, D], dt),
        UK: T.Tensor([ubn, BS, G, MDS, D], dt),
        UV: T.Tensor([ubn, BS, G, MDS, D], dt),
        Out: T.Tensor([G, BS * H, D], dt),
        w2: T.Tensor([NT, NS, BM, BK], adt),
        w3: T.Tensor([NT, NS, BM, BK], dt),
        w4: T.Tensor([NT, 8, BM, D], adt),
    ):
        with T.Kernel(NT, is_npu=True) as (cid, vid):
            gi = cid % G
            bt0 = (cid // G) * fuse
            idx = cid
            ql = T.alloc_L1([BM, D], dt)
            kl = T.alloc_L1([BK, D], dt)
            vl = T.alloc_L1([BK, D], dt)
            pl = T.alloc_L1([BM, BK], dt)
            sl = T.alloc_L0C([BM, BK], adt)
            ol = T.alloc_L0C([BM, D], adt)
            ao = T.alloc_ub([V8, D], adt)
            se = T.alloc_ub([V8], adt)
            mx = T.alloc_ub([V8], adt)
            mp = T.alloc_ub([V8], adt)
            sb = T.alloc_ub([V8, BK], adt)
            sn = T.alloc_ub([V8, BK], adt)
            ln = T.alloc_ub([V8], adt)
            sh = T.alloc_ub([V8, BK], dt)
            ob = T.alloc_ub([V8, D], adt)
            oh = T.alloc_ub([V8, D], dt)
            kb = T.alloc_ub([D], dt)
            se2 = T.alloc_ub([V8], adt)
            om = T.alloc_ub([V8], adt)
            nm = T.alloc_ub([V8], adt)
            ar = T.alloc_ub([D], adt)
            vf = T.alloc_ub([D], adt)
            vh = T.alloc_ub([D], dt)
            T.annotate_address({sl: 0, ol: 16384})
            for qi in range(NQT):
                qr0 = qi * BM
                bb = bt0 + qi * BPT
                with T.Scope("C"):
                    T.copy(Q[gi, bt0 * H + qr0 : bt0 * H + qr0 + BM, :D], ql)
                    for si in range(PRE):
                        T.copy(SK[gi, si * BK : si * BK + BK, :D], kl)
                        T.set_flag("mte2", "m", 3)
                        T.wait_flag("mte2", "m", 3)
                        T.gemm_v0(ql, kl, sl, transpose_B=True, init=True)
                        T.set_flag("m", "fix", 4)
                        T.wait_flag("m", "fix", 4)
                        T.copy(sl, w2[idx, si % NS, 0:BM, 0:BK])
                        T.set_cross_flag("FIX", (si % NS) * 3 + 0)
                    for si in range(PRE, 8 + PRE):
                        pvs = si - PRE
                        sp = pvs % NS
                        T.barrier_all()
                        if si < 8:
                            T.copy(SK[gi, si * BK : si * BK + BK, :D], kl)
                            T.set_flag("mte2", "m", 3)
                            T.wait_flag("mte2", "m", 3)
                            T.gemm_v0(ql, kl, sl, transpose_B=True, init=True)
                            T.set_flag("m", "fix", 4)
                            T.wait_flag("m", "fix", 4)
                            T.copy(sl, w2[idx, si % NS, 0:BM, 0:BK])
                            T.barrier_all()
                            T.set_cross_flag("FIX", (si % NS) * 3 + 0)
                        T.barrier_all()
                        T.wait_cross_flag(sp * 3 + 1, "MTE2")
                        T.copy(w3[idx, sp, 0:BM, 0:BK], pl)
                        T.copy(SV[gi, pvs * BK : pvs * BK + BK, :D], vl)
                        T.set_flag("mte2", "m", 5)
                        T.wait_flag("mte2", "m", 5)
                        T.gemm_v0(pl, vl, ol, init=True)
                        T.set_flag("m", "fix", 6)
                        T.wait_flag("m", "fix", 6)
                        T.copy(ol, w4[idx, pvs, 0:BM, 0:D])
                        T.barrier_all()
                        T.set_cross_flag("FIX", sp * 3 + 2)
                with T.Scope("V"):
                    T.tile.fill(ao, 0.0)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    T.tile.fill(se, 0.0)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    T.tile.fill(mx, NEG_INF)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for si in range(8):
                        ss = si % NS
                        T.wait_cross_flag(ss * 3 + 0, "MTE2")
                        T.tile.fill(sb, 0.0)
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        T.copy(mx, mp)
                        T.pipe_barrier("v")
                        T.copy(w2[idx, ss, vid * V8 : vid * V8 + V8, :], sn)
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)
                        T.pipe_barrier("v")
                        T.tile.add(sb, sb, sn)
                        T.tile.mul(sb, sb, sm_scale)
                        T.pipe_barrier("v")
                        T.reduce_max(sb, mx, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.max(mx, mx, mp)
                        T.pipe_barrier("v")
                        T.tile.sub(mp, mp, mx)
                        T.pipe_barrier("v")
                        T.tile.exp(mp, mp)
                        for hi in range(V8):
                            T.pipe_barrier("v")
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            T.tile.sub(sb[hi, :], sb[hi, :], mx[hi])
                        T.pipe_barrier("v")
                        T.tile.exp(sb, sb)
                        T.pipe_barrier("v")
                        T.reduce_sum(sb, ln, dim=-1)
                        T.tile.mul(se, se, mp)
                        T.pipe_barrier("v")
                        T.tile.add(se, se, ln)
                        for hi in range(V8):
                            T.pipe_barrier("v")
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            T.tile.mul(ao[hi, :], ao[hi, :], mp[hi])
                        T.set_flag("v", "mte3", 4)
                        T.wait_flag("v", "mte3", 4)
                        T.copy(sb, sh)
                        T.copy(sh, w3[idx, ss, vid * V8 : vid * V8 + V8, :])
                        T.barrier_all()
                        T.set_cross_flag("MTE3", ss * 3 + 1)
                        T.barrier_all()
                        if si >= PRE:
                            asp = (si - PRE) % NS
                            T.wait_cross_flag(asp * 3 + 2, "MTE2")
                            T.copy(w4[idx, si - PRE, vid * V8 : vid * V8 + V8, :], ob)
                            T.set_flag("mte2", "v", 5)
                            T.wait_flag("mte2", "v", 5)
                            T.tile.add(ao, ao, ob)
                            T.barrier_all()
                    for si in range(8 - PRE, 8):
                        asp = si % NS
                        T.wait_cross_flag(asp * 3 + 2, "MTE2")
                        T.copy(w4[idx, si, vid * V8 : vid * V8 + V8, :], ob)
                        T.set_flag("mte2", "v", 1)
                        T.wait_flag("mte2", "v", 1)
                        T.tile.add(ao, ao, ob)
                    for bt_i in range(BPT):
                        bt = bb + bt_i
                        br = bt_i * H
                        T.set_flag("v", "mte2", 3)
                        T.wait_flag("v", "mte2", 3)
                        for di in range(MDS):
                            T.copy(UK[0, bt, gi, di, :D], kb)
                            T.set_flag("mte2", "v", 2)
                            T.wait_flag("mte2", "v", 2)
                            T.pipe_barrier("v")
                            T.copy(kb, sn[2 * di, :D])
                            T.copy(kb, sn[2 * di + 1, :D])
                        T.set_flag("v", "mte2", 5)
                        T.wait_flag("v", "mte2", 5)
                        for h in range(H):
                            T.copy(Q[gi, bt * H + h : bt * H + h + 1, :D], kb)
                            T.set_flag("mte2", "v", 6)
                            T.wait_flag("mte2", "v", 6)
                            T.pipe_barrier("v")
                            T.copy(kb, sb[h, :D])
                            T.copy(kb, sb[h + H, :D])
                            T.copy(kb, sb[h + H * 2, :D])
                            T.copy(kb, sb[h + H * 3, :D])
                        T.pipe_barrier("v")
                        T.tile.mul(sb, sb, sn)
                        T.reduce_sum(sb, ln, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.mul(ln, ln, sm_scale)
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        T.tile.fill(se2, NEG_INF)
                        T.pipe_barrier("v")
                        T.copy(ln, se2)
                        T.copy(mx, om)
                        T.pipe_barrier("v")
                        T.tile.max(nm, om, se2)
                        T.pipe_barrier("v")
                        T.copy(nm, mx)
                        T.tile.sub(om, om, nm)
                        T.pipe_barrier("v")
                        T.tile.exp(om, om)
                        T.pipe_barrier("v")
                        T.tile.mul(se, se, om)
                        T.pipe_barrier("v")
                        T.tile.exp(se2, se2)
                        T.pipe_barrier("v")
                        T.tile.add(se, se, se2)
                        for hi in range(V8):
                            T.pipe_barrier("v")
                            T.tile.mul(ao[hi, :], ao[hi, :], om[hi])
                        for di in range(MDS):
                            T.copy(UV[0, bt, gi, di, :D], vh)
                            T.set_flag("mte2", "v", 1)
                            T.wait_flag("mte2", "v", 1)
                            T.copy(vh, vf)
                            for h in range(H):
                                iv = h + di * H
                                r = br + h - vid * V8
                                if r >= 0 and r < V8:
                                    T.set_flag("v", "s", 0)
                                    T.wait_flag("v", "s", 0)
                                    T.tile.mul(vf, vf, se2[iv])
                                    T.copy(ao[r, :D], ar)
                                    T.pipe_barrier("v")
                                    T.tile.add(ar, ar, vf)
                                    T.pipe_barrier("v")
                                    T.copy(ar, ao[r, :D])
                    T.barrier_all()
                    for hi in range(V8):
                        T.pipe_barrier("v")
                        T.tile.div(ao[hi, :], ao[hi, :], se[hi])
                        T.pipe_barrier("v")
                    T.copy(ao, oh)
                    T.set_flag("v", "mte3", 4)
                    T.wait_flag("v", "mte3", 4)
                    T.copy(oh, Out[gi, bt0 * H + qr0 + vid * V8 : bt0 * H + qr0 + vid * V8 + V8, :D])

    return main


def pq(Qr, kv, bm, dim):
    H = Qr.shape[1] // kv
    Qp = torch.zeros((kv, bm * H, dim), dtype=Qr.dtype, device=Qr.device)
    [Qp.__setitem__((g, bt * H + h, slice(None)), Qr[bt, g * H + h, :]) for g in range(kv) for bt in range(bm) for h in range(H)]
    return Qp


def ro(O, kv, bm, dim, H=2):
    Out = torch.zeros((bm, H * kv, dim), dtype=O.dtype, device=O.device)
    [Out.__setitem__((bt, g * H + h, slice(None)), O[g, bt * H + h, :]) for bt in range(bm) for g in range(kv) for h in range(H)]
    return Out


def golden(Q, SK, SV, UK, UV, kl, ul, kv, dim):
    s = 1.0 / (dim**0.5)
    H = Q.shape[1] // kv
    BS = Q.shape[0]
    Qf = Q.float().cpu()
    SKf = SK.float().cpu()
    SVf = SV.float().cpu()
    UKf = UK.float().cpu()
    UVf = UV.float().cpu()
    O = torch.zeros((BS, H * kv, dim), dtype=torch.float32)
    for b in range(BS):
        for g in range(kv):
            ks = SKf[g, :kl]
            vs = SVf[g, :kl]
            ku = UKf[0, b, g, :ul]
            vu = UVf[0, b, g, :ul]
            kt = torch.cat([ks, ku], 0)
            vt = torch.cat([vs, vu], 0)
            qb = Qf[b, g * H : (g + 1) * H]
            p = torch.softmax(qb @ kt.T * s, -1)
            O[b, g * H : (g + 1) * H] = p.to(vt.dtype) @ vt
    return O.to(Q.dtype).npu()


if __name__ == "__main__":
    kv, dim, BM, BK, pr, ds = 8, 128, 16, 128, 1024, 4
    ST, UBN, md = pr + 8192, 1 + 209, 4
    beam, fu = 128, 16
    NT = beam * kv // fu
    print("t21 compiling...")
    func = x_attention(kv_heads=kv, dim=dim, beam_size=beam, shared_block_tokens=ST, max_decode_step=ds, fuse=fu)
    torch.npu.synchronize()
    src = func.get_kernel_source()
    open("/tmp/opt2b_bk128_pto.cc", "w").write(src)
    print(f"  {len(src)} chars saved to /tmp/opt2b_bk128_pto.cc")
    torch.manual_seed(42)
    Qr = torch.randn((beam, 16, dim), dtype=torch.float16) * 0.3
    Qp = pq(Qr, kv, beam, dim)
    SK = torch.randn((kv, ST, dim), dtype=torch.float16) * 0.5
    SV = torch.randn((kv, ST, dim), dtype=torch.float16) * 0.5
    UK = torch.randn((UBN, beam, kv, md, dim), dtype=torch.float16) * 0.5
    UV = torch.randn((UBN, beam, kv, md, dim), dtype=torch.float16) * 0.5
    Op = torch.zeros((kv, beam * 2, dim), dtype=torch.float16)
    w2 = torch.zeros((NT, NS, BM, BK), dtype=torch.float32)
    w3 = torch.zeros((NT, NS, BM, BK), dtype=torch.float16)
    w4 = torch.zeros((NT, 8, BM, dim), dtype=torch.float32)
    for _ in range(5):
        func(Qp, SK, SV, UK, UV, Op, w2, w3, w4)
        torch.npu.synchronize()
    N = 20
    times = []
    for i in range(N):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        func(Qp, SK, SV, UK, UV, Op, w2, w3, w4)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    avg = sum(times) / N
    mn = min(times)
    mx = max(times)
    print(f"Latency: min={mn:.0f}us avg={avg:.0f}us max={mx:.0f}us")
    out = func(Qp, SK, SV, UK, UV, Op, w2, w3, w4)
    torch.npu.synchronize()
    o = ro(out, kv, beam, dim)
    g = golden(Qr, SK, SV, UK, UV, pr, ds, kv, dim)
    d = (o.cpu().float() - g.cpu().float()).abs()
    print(f"Precision: max={d.max():.6f} mean={d.mean():.6f}")
    torch.testing.assert_close(o.cpu(), g.cpu(), rtol=1e-2, atol=0.01)
    print("  PASS")
