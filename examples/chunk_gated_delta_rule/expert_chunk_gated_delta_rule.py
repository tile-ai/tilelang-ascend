import argparse
import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


# ==========================================
# 1. Helper Functions
# ==========================================
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute starting offset of each sequence's chunks in output h tensor"""
    chunk_offsets = []
    offset = 0
    cu_seqlens_np = cu_seqlens.cpu().numpy()
    for i in range(len(cu_seqlens_np) - 1):
        T_len = int(cu_seqlens_np[i + 1] - cu_seqlens_np[i])
        NT = (T_len + chunk_size - 1) // chunk_size
        chunk_offsets.append(offset)
        offset += NT
    return torch.tensor(chunk_offsets, dtype=torch.int32, device=cu_seqlens.device)


# ==========================================
# 2. TileLang Unified Kernel (Fully 1D Packed)
# ==========================================
@tilelang.jit(workspace_idx=[9, 10, 11, 12], pass_configs=pass_configs)
def chunk_gated_delta_rule_fwd_kernel(
    N,
    H,
    T_total,
    Hg,
    K,
    V,
    NT_max,
    BT=64,
    USE_G=True,
    STORE_FINAL_STATE=True,
    SAVE_NEW_VALUE=True,
    dtype="float16",
    accum_dtype="float32",
):
    V_half = V // 2

    SEM_WH_C2V = 0
    SEM_VNEW_V2C = 2
    SEM_HUPD_C2V = 4
    SEM_H_V2C = 6

    @T.prim_func
    def main(
        h: T.Tensor([N, NT_max, H, K, V], dtype),  # type: ignore
        k: T.Tensor([T_total, Hg, K], dtype),  # type: ignore
        v: T.Tensor([T_total, H, V], dtype),  # type: ignore
        w: T.Tensor([T_total, H, K], dtype),  # type: ignore
        g: T.Tensor([H, T_total], accum_dtype),  # type: ignore
        v_new: T.Tensor([T_total, H, V], dtype),  # type: ignore
        h0: T.Tensor([N, H, K, V], dtype),  # type: ignore
        ht: T.Tensor([N, H, K, V], dtype),  # type: ignore
        cu_seqlens: T.Tensor([N + 1], "int32"),  # type: ignore
        ws_wh: T.Tensor([N, H, 2, BT, V_half], accum_dtype),  # type: ignore
        ws_vnew: T.Tensor([N, H, 2, BT, V_half], dtype),  # type: ignore
        ws_hupd: T.Tensor([N, H, 2, K, V_half], accum_dtype),  # type: ignore
        ws_h: T.Tensor([N, H, 2, K, V_half], dtype),  # type: ignore
    ):
        with T.Kernel(N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H

            hg_ratio = H // Hg
            k_head = i_h // hg_ratio

            h_state_ub = T.alloc_ub([2, K // 2, V_half], dtype)
            h_state_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            hupd_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            wh_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)

            v_chunk_ub = T.alloc_ub([2, 2, BT // 2, V_half], dtype)
            v_chunk_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)

            g_chunk_ub = T.alloc_ub([2, BT // 2], accum_dtype)
            g_last_scalar = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([BT // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([BT // 2, V_half], accum_dtype)

            k_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            w_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            h_state_l1 = T.alloc_L1([2, K, V_half], dtype)
            wh_frag = T.alloc_L0C([2, BT, V_half], accum_dtype)
            v_new_l1 = T.alloc_L1([2, BT, V_half], dtype)
            hupd_frag = T.alloc_L0C([2, K, V_half], accum_dtype)

            with T.Scope("C"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, BT)

                actual_len = T.if_then_else(T_len < BT, T_len, BT)
                T.copy(w[bos : bos + actual_len, i_h, :], w_chunk_l1[0, :, :])
                T.copy(k[bos : bos + actual_len, k_head, :], k_chunk_l1[0, :, :])
                T.pipe_barrier("mte2")
                T.set_flag("mte2", "m", 0)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    chunk_start_next = bos + (i + 1) * BT

                    chunk_len = T.if_then_else(i * BT + BT > T_len, T_len - i * BT, BT)

                    if i + 1 < NT_i:
                        next_len = T.if_then_else((i + 1) * BT + BT > T_len, T_len - (i + 1) * BT, BT)
                        T.copy(w[chunk_start_next : chunk_start_next + next_len, i_h, :], w_chunk_l1[next_pid, :, :])
                        T.copy(k[chunk_start_next : chunk_start_next + next_len, k_head, :], k_chunk_l1[next_pid, :, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "m", next_pid)

                    # w @ h
                    T.wait_flag("mte2", "m", pid)
                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_H_V2C + j)
                        T.copy(ws_h[i_n, i_h, j, :, :], h_state_l1[j, :, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "m", 2)
                        T.wait_flag("mte2", "m", 2)
                        T.gemm_v0(w_chunk_l1[pid, :, :], h_state_l1[j, :, :], wh_frag[j, :, :], init=True)
                        T.pipe_barrier("m")
                        T.set_flag("m", "fix", 3)
                        T.wait_flag("m", "fix", 3)
                        T.copy(wh_frag[j, :, :], ws_wh[i_n, i_h, j, :, :])
                        T.pipe_barrier("fix")
                        T.set_cross_flag("FIX", SEM_WH_C2V + j)

                    # k @ v_new
                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_VNEW_V2C + j)
                        T.copy(ws_vnew[i_n, i_h, j, :chunk_len, :], v_new_l1[j, :, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "m", 4)
                        T.wait_flag("mte2", "m", 4)
                        T.gemm_v0(k_chunk_l1[pid, :, :], v_new_l1[j, :, :], hupd_frag[j, :, :], transpose_A=True, init=True)
                        T.pipe_barrier("m")
                        T.set_flag("m", "fix", 5)
                        T.wait_flag("m", "fix", 5)
                        T.copy(hupd_frag[j, :, :], ws_hupd[i_n, i_h, j, :, :])
                        T.pipe_barrier("fix")
                        T.set_cross_flag("FIX", SEM_HUPD_C2V + j)

            with T.Scope("V"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, BT)

                for j in T.serial(2):
                    T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half], h_state_ub[j, :, :])

                chunk_len = T.if_then_else(T_len < BT, T_len, BT)
                vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, chunk_len), T.max(chunk_len - BT // 2, 0))
                vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)
                vec_global_start = bos + vec_start_in_chunk

                for j in T.serial(2):
                    T.copy(
                        v[vec_global_start : vec_global_start + vec_chunk_len, i_h, j * V_half : (j + 1) * V_half],
                        v_chunk_ub[0, j, :, :],
                    )
                if USE_G:
                    T.copy(g[i_h, vec_global_start : vec_global_start + vec_chunk_len], g_chunk_ub[0, :])
                T.pipe_barrier("mte2")
                T.set_flag("mte2", "v", 0)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    v_flag_pid = pid
                    v_flag_next = next_pid
                    g_start = bos + i * BT
                    g_start_next = bos + (i + 1) * BT

                    chunk_len = T.if_then_else(i * BT + BT > T_len, T_len - i * BT, BT)
                    vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, chunk_len), T.max(chunk_len - BT // 2, 0))
                    vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)

                    # v[t+1], g[t+1]
                    if i + 1 < NT_i:
                        next_chunk_len = T.if_then_else((i + 1) * BT + BT > T_len, T_len - (i + 1) * BT, BT)
                        next_vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)
                        next_vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, next_chunk_len), T.max(next_chunk_len - BT // 2, 0))
                        next_vec_global_start = g_start_next + next_vec_start_in_chunk

                        for j in T.serial(2):
                            T.copy(
                                v[next_vec_global_start : next_vec_global_start + next_vec_chunk_len, i_h, j * V_half : (j + 1) * V_half],
                                v_chunk_ub[next_pid, j, :, :],
                            )
                        if USE_G:
                            T.copy(g[i_h, next_vec_global_start : next_vec_global_start + next_vec_chunk_len], g_chunk_ub[next_pid, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "v", v_flag_next)

                    # h to cube
                    T.barrier_all()
                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], ws_h[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :])
                        T.pipe_barrier("mte3")
                        T.set_cross_flag("MTE3", SEM_H_V2C + j)

                    # save h[t]
                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], h[i_n, i, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half])

                    T.wait_flag("mte2", "v", v_flag_pid)
                    # prepare gating
                    if USE_G:
                        g_last = T.if_then_else(i * BT + BT <= T_len, g[i_h, g_start + BT - 1], g[i_h, g_start + T_len - i * BT - 1])

                        T.tile.fill(g_exp_ub, g_last)
                        T.set_flag("mte2", "v", 2)
                        T.wait_flag("mte2", "v", 2)
                        T.barrier_all()
                        T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid, :])
                        T.pipe_barrier("v")
                        T.tile.exp(g_exp_ub, g_exp_ub)
                        T.pipe_barrier("v")
                        T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)

                        T.tile.fill(g_last_scalar, g_last)
                        T.barrier_all()
                        T.tile.exp(g_last_scalar, g_last_scalar)

                    for j in T.serial(2):
                        T.copy(v_chunk_ub[pid, j, :, :], v_chunk_ub_float[j, :, :])

                        # v_new = v - w @ h
                        T.wait_cross_flag(SEM_WH_C2V + j)
                        T.copy(ws_wh[i_n, i_h, j, vec_start_in_chunk : vec_start_in_chunk + BT // 2, :], wh_ub_float[j, :, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "v", 3)
                        T.wait_flag("mte2", "v", 3)
                        T.pipe_barrier("v")
                        T.tile.sub(v_chunk_ub_float[j, :, :], v_chunk_ub_float[j, :, :], wh_ub_float[j, :, :])

                        if SAVE_NEW_VALUE:
                            T.pipe_barrier("v")
                            T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                            T.pipe_barrier("v")
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(
                                v_chunk_ub[pid, j, :vec_chunk_len, :],
                                v_new[
                                    g_start + vec_start_in_chunk : g_start + vec_start_in_chunk + vec_chunk_len,
                                    i_h,
                                    j * V_half : j * V_half + V_half,
                                ],
                            )

                        if USE_G:
                            # v_new *= exp(g_last - g)
                            T.pipe_barrier("v")
                            T.tile.mul(v_chunk_ub_float[j, :, :], v_chunk_ub_float[j, :, :], g_exp_ub_broc)
                            # h *= exp(g_last)
                            T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])
                            T.pipe_barrier("v")
                            T.tile.mul(h_state_ub_float[j, :, :], h_state_ub_float[j, :, :], g_last_scalar[0])
                        else:
                            T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])

                        T.set_flag("mte3", "v", 0)
                        T.wait_flag("mte3", "v", 0)
                        T.pipe_barrier("v")
                        T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(v_chunk_ub[pid, j, :, :], ws_vnew[i_n, i_h, j, vec_start_in_chunk : vec_start_in_chunk + BT // 2, :])
                        T.pipe_barrier("mte3")
                        T.set_cross_flag("MTE3", SEM_VNEW_V2C + j)

                    for j in T.serial(2):
                        # h += k @ v_new
                        T.wait_cross_flag(SEM_HUPD_C2V + j)
                        T.copy(ws_hupd[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :], hupd_ub_float[j, :, :])
                        T.pipe_barrier("mte2")
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.pipe_barrier("v")
                        T.tile.add(h_state_ub_float[j, :, :], h_state_ub_float[j, :, :], hupd_ub_float[j, :, :])
                        T.pipe_barrier("v")
                        T.copy(h_state_ub_float[j, :, :], h_state_ub[j, :, :])

                    T.set_flag("v", "mte3", 2)
                    T.wait_flag("v", "mte3", 2)

                if STORE_FINAL_STATE:
                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], ht[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half])

    return main


# ==========================================
# 3. Python Wrapper Layer
# ==========================================
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    BT = chunk_size
    USE_G = g is not None

    k_flat = k.squeeze(0)  # [T_total, Hg, K]
    w_flat = w.squeeze(0)  # [T_total, H, K]
    u_flat = u.squeeze(0)  # [T_total, H, V]
    g_flat = g.squeeze(0) if g is not None else None  # [T_total, H]

    T_total, Hg, K = k_flat.shape
    _, H, V = u_flat.shape
    N = len(cu_seqlens) - 1

    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    cu_seqlens_np = cu_seqlens.cpu().numpy()
    NT_max = 0
    NT_total = 0
    for i in range(N):
        T_len = int(cu_seqlens_np[i + 1] - cu_seqlens_np[i])
        NT = (T_len + BT - 1) // BT
        NT_max = max(NT_max, NT)
        NT_total += NT

    if USE_G:
        g_c_t = g_flat.float().transpose(0, 1).contiguous()  # [H, T_total] — transpose required
    else:
        g_c_t = torch.empty((H, T_total), dtype=torch.float32, device=k.device)

    v_new_flat = torch.empty((T_total, H, V), dtype=torch.float16, device=k.device)

    # Allocate state outputs
    h_out = torch.zeros((N, NT_max, H, K, V), dtype=torch.float16, device=k.device)
    h0 = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)
    if initial_state is not None:
        h0.copy_(initial_state.squeeze(0))

    ht = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)

    ker = chunk_gated_delta_rule_fwd_kernel(
        N,
        H,
        T_total,
        Hg,
        K,
        V,
        NT_max,
        BT=BT,
        USE_G=USE_G,
        STORE_FINAL_STATE=output_final_state,
        SAVE_NEW_VALUE=save_new_value,
    )
    ker(h_out, k_flat, u_flat, w_flat, g_c_t, v_new_flat, h0, ht, cu_seqlens.to(torch.int32))

    v_new_ret = v_new_flat.unsqueeze(0)  # [1, T_total, H, V]

    h_ret = torch.zeros((NT_total, H, K, V), dtype=torch.float16, device=k.device)
    for i in range(N):
        NT_i = (int(cu_seqlens_np[i + 1]) - int(cu_seqlens_np[i]) + BT - 1) // BT
        offset = int(chunk_offsets[i].item())
        h_ret[offset : offset + NT_i] = h_out[i, :NT_i]
    h_ret = h_ret.unsqueeze(0)  # [1, NT_total, H, K, V]

    ht_ret = ht if output_final_state else None  # [N, H, K, V]

    return h_ret, v_new_ret, ht_ret


# ==========================================
# 4. Golden Reference
# ==========================================
def ref_chunk_gated_delta_rule(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    BT = chunk_size

    k = k.float().squeeze(0)  # [T_total, Hg, K]
    w = w.float().squeeze(0)  # [T_total, H, K]
    u = u.float().squeeze(0)  # [T_total, H, V]
    g = g.float().squeeze(0) if g is not None else None  # [T_total, H]
    initial_state = initial_state.float().squeeze(0) if initial_state is not None else None  # [N, H, K, V]

    T_total, Hg, K = k.shape
    _, H, V = u.shape
    N = len(cu_seqlens) - 1

    NT_total = sum([(int(cu_seqlens[i + 1]) - int(cu_seqlens[i]) + BT - 1) // BT for i in range(N)])

    h = torch.zeros(NT_total, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.zeros(T_total, H, V, dtype=torch.float32, device=k.device)
    final_state = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device) if output_final_state else None

    chunk_offset = 0
    for i_n in range(N):
        bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
        T_len = eos - bos
        NT = (T_len + BT - 1) // BT

        for i_h in range(H):
            h_state = (
                initial_state[i_n, i_h].clone() if initial_state is not None else torch.zeros(K, V, dtype=torch.float32, device=k.device)
            )
            k_head = i_h // (H // Hg)

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T_len)

                h[chunk_offset + i_t, i_h] = h_state
                k_chunk, w_chunk, v_chunk = (
                    k[bos + t_start : bos + t_end, k_head, :],
                    w[bos + t_start : bos + t_end, i_h, :],
                    u[bos + t_start : bos + t_end, i_h, :],
                )

                v_n = v_chunk - torch.matmul(w_chunk, h_state)
                v_new[bos + t_start : bos + t_end, i_h, :] = v_n

                if g is not None:
                    g_chunk = g[bos + t_start : bos + t_end, i_h]
                    g_last = g_chunk[-1].item()
                    v_n = v_n * torch.exp(g_last - g_chunk)[:, None]
                    h_state = h_state * torch.exp(torch.tensor(g_last, device=k.device))

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)

            if output_final_state:
                final_state[i_n, i_h] = h_state
        chunk_offset += NT

    return h.half().unsqueeze(0), v_new.half().unsqueeze(0), final_state.half() if final_state is not None else None


# ==========================================
# 5. Test Functions
# ==========================================
def test_chunk_gated_delta_rule(seqlens, H, Hg, K, V, use_g=True, use_initial_state=True):
    print(f"Testing Varlen seqlens={seqlens}, H={H}, Hg={Hg}, K={K}, V={V}, use_g={use_g}, use_initial_state={use_initial_state}")
    torch.manual_seed(41)

    T_total = sum(seqlens)
    N = len(seqlens)
    cu_seqlens = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))], dtype=torch.int32).npu()

    torch.manual_seed(41)
    k = torch.rand(1, T_total, Hg, K, dtype=torch.float16).npu() * 0.01
    w = torch.rand(1, T_total, H, K, dtype=torch.float16).npu() * 0.01
    u = torch.rand(1, T_total, H, V, dtype=torch.float16).npu() * 0.01
    g = torch.rand(1, T_total, H, dtype=torch.float32).npu() * -1.0 if use_g else None
    initial_state = torch.rand(1, N, H, K, V, dtype=torch.float16).npu() * 0.01 if use_initial_state else None

    torch.npu.synchronize()

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state=initial_state, output_final_state=True, cu_seqlens=cu_seqlens)
    torch.npu.synchronize()
    ref_h, ref_v_new, ref_ht = ref_chunk_gated_delta_rule(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu() if g is not None else None,
        initial_state=initial_state.cpu() if initial_state is not None else None,
        output_final_state=True,
        cu_seqlens=cu_seqlens.cpu(),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(h.cpu(), ref_h.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_new.cpu(), ref_v_new.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ht.cpu(), ref_ht.cpu(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test chunk gated delta rule (varlen mode only: [1, T_total])")
    parser.add_argument("--use_g", type=lambda x: x.lower() == "true", default=True, help="Whether to use gating (True/False)")
    parser.add_argument(
        "--use_initial_state", type=lambda x: x.lower() == "true", default=True, help="Whether to use initial state (True/False)"
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        default="2048",
        help="Sequence lengths for varlen mode (comma-separated)",
    )
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--Hg", type=int, default=4, help="Number of grouped heads (must be <= H)")
    parser.add_argument("--K", type=int, default=128, help="Key dimension")
    parser.add_argument("--V", type=int, default=128, help="Value dimension")
    args = parser.parse_args()

    print("=" * 60)
    seqlens = [int(x) for x in args.seqlens.split(",")]
    test_chunk_gated_delta_rule(
        seqlens=seqlens, H=args.H, Hg=args.Hg, K=args.K, V=args.V, use_g=args.use_g, use_initial_state=args.use_initial_state
    )
    print("Batch Kernel Output Match!")
