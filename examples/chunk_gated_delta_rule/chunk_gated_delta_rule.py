import tilelang
from tilelang import language as T
import torch
import argparse

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
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
def chunk_gated_delta_rule(
    N,
    H,
    T_total_pad,
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
    @T.prim_func
    def main(
        h: T.Tensor([N, NT_max, H, K, V], dtype),  # type: ignore
        k: T.Tensor([T_total_pad, Hg, K], dtype),  # type: ignore
        v: T.Tensor([T_total_pad, H, V], dtype),  # type: ignore
        w: T.Tensor([T_total_pad, H, K], dtype),  # type: ignore
        g: T.Tensor([H, T_total_pad], accum_dtype),  # type: ignore
        v_new: T.Tensor([T_total_pad, H, V], dtype),  # type: ignore
        h0: T.Tensor([N, H, K, V], dtype),  # type: ignore
        ht: T.Tensor([N, H, K, V], dtype),  # type: ignore
        cu_seqlens: T.Tensor([N + 1], "int32"),  # type: ignore
        workspace_wh: T.Tensor([N, H, BT, V], accum_dtype),  # type: ignore
        workspace_vnew: T.Tensor([N, H, BT, V], dtype),  # type: ignore
        workspace_hupd: T.Tensor([N, H, K, V], accum_dtype),  # type: ignore
        workspace_h: T.Tensor([N, H, K, V], dtype),  # type: ignore
    ):
        with T.Kernel(N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H

            hg_ratio = H // Hg
            k_head = i_h // hg_ratio

            h_state_ub = T.alloc_ub([K // 2, V], dtype)
            h_state_ub_float = T.alloc_ub([K // 2, V], accum_dtype)
            hupd_ub_float = T.alloc_ub([K // 2, V], accum_dtype)
            wh_ub_float = T.alloc_ub([BT // 2, V], accum_dtype)
            v_chunk_ub = T.alloc_ub([BT // 2, V], dtype)
            v_chunk_ub_float = T.alloc_ub([BT // 2, V], accum_dtype)
            v_new_ub = T.alloc_ub([BT // 2, V], dtype)
            v_new_ub_float = T.alloc_ub([BT // 2, V], accum_dtype)

            g_chunk_ub = T.alloc_ub([BT // 2], accum_dtype)
            g_last_ub = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([BT // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([BT // 2, V], accum_dtype)

            k_chunk_l1 = T.alloc_L1([BT, K], dtype)
            w_chunk_l1 = T.alloc_L1([BT, K], dtype)
            h_state_l1 = T.alloc_L1([K, V], dtype)
            wh_frag = T.alloc_L0C([BT, V], accum_dtype)
            v_new_l1 = T.alloc_L1([BT, V], dtype)
            hupd_frag = T.alloc_L0C([K, V], accum_dtype)

            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            T_len = eos - bos
            NT_i = T.ceildiv(T_len, BT)

            T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, :], h_state_ub)

            for i in T.serial(NT_i):
                g_start = bos + i * BT
                # 1: save h[t]
                T.copy(h_state_ub, h[i_n, i, i_h, K // 2 * vid : K // 2 * vid + K // 2, :])

                # 2: W @ h
                T.copy(h_state_ub, workspace_h[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, :])

                T.copy(workspace_h[i_n, i_h, :, :], h_state_l1)
                T.copy(w[g_start : g_start + BT, i_h, :], w_chunk_l1)
                T.gemm_v0(w_chunk_l1, h_state_l1, wh_frag, init=True)
                T.copy(wh_frag, workspace_wh[i_n, i_h, :, :])

                T.copy(workspace_wh[i_n, i_h, BT // 2 * vid : BT // 2 * vid + BT // 2, :], wh_ub_float)

                # 3: v_new = v - wh
                T.copy(v[g_start + BT // 2 * vid : g_start + BT // 2 * vid + BT // 2, i_h, :], v_chunk_ub)
                T.copy(v_chunk_ub, v_chunk_ub_float)
                T.tile.sub(v_new_ub_float, v_chunk_ub_float, wh_ub_float)
                if SAVE_NEW_VALUE:
                    T.copy(v_new_ub_float, v_new_ub)
                    T.copy(v_new_ub, v_new[g_start + BT // 2 * vid : g_start + BT // 2 * vid + BT // 2, i_h, :])

                # 4: gated
                if USE_G:
                    g_last = T.if_then_else(i * BT + BT <= T_len, g[i_h, g_start + BT - 1], g[i_h, g_start + T_len - i * BT - 1])
                    # exp(g_last - g)
                    T.copy(g[i_h, g_start + BT // 2 * vid : g_start + BT // 2 * vid + BT // 2], g_chunk_ub)
                    T.tile.fill(g_exp_ub, g_last)
                    T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub)
                    T.tile.exp(g_exp_ub, g_exp_ub)
                    T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)
                    # v_new = v_new * exp(g_last - g)
                    T.tile.mul(v_new_ub_float, v_new_ub_float, g_exp_ub_broc)
                    # h = h * exp(g_last)
                    T.tile.fill(g_last_ub, g_last)
                    T.tile.exp(g_last_ub, g_last_ub)
                    T.copy(h_state_ub, h_state_ub_float)
                    T.tile.mul(h_state_ub_float, h_state_ub_float, g_last_ub[0])
                else:
                    T.copy(h_state_ub, h_state_ub_float)

                # 5: k @ v_new
                T.copy(v_new_ub_float, v_new_ub)
                T.copy(v_new_ub, workspace_vnew[i_n, i_h, BT // 2 * vid : BT // 2 * vid + BT // 2, :])

                T.copy(workspace_vnew[i_n, i_h, :, :], v_new_l1)
                T.copy(k[g_start : g_start + BT, k_head, :], k_chunk_l1)
                T.gemm_v0(k_chunk_l1, v_new_l1, hupd_frag, transpose_A=True, init=True)
                T.copy(hupd_frag, workspace_hupd[i_n, i_h, :, :])

                T.copy(workspace_hupd[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, :], hupd_ub_float)

                # 6: h = h + k @ v_new
                T.tile.add(h_state_ub_float, h_state_ub_float, hupd_ub_float)
                T.copy(h_state_ub_float, h_state_ub)

            if STORE_FINAL_STATE:
                T.copy(h_state_ub, ht[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, :])

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

    T_total_pad = T_total + BT
    if USE_G:
        g_c_t = g_flat.float().transpose(0, 1).contiguous()  # [H, T_total] — transpose required
        g_pad = torch.cat([g_c_t, torch.zeros((g_c_t.shape[0], BT), dtype=torch.float32, device=k.device)], dim=1)
    else:
        g_pad = torch.empty((H, T_total_pad), dtype=torch.float32, device=k.device)

    v_new_flat = torch.empty((T_total, H, V), dtype=torch.float16, device=k.device)

    k_pad = torch.cat([k_flat, torch.zeros((BT, Hg, K), dtype=torch.float16, device=k.device)], dim=0)
    w_pad = torch.cat([w_flat, torch.zeros((BT, H, K), dtype=torch.float16, device=k.device)], dim=0)
    u_pad = torch.cat([u_flat, torch.zeros((BT, H, V), dtype=torch.float16, device=k.device)], dim=0)
    v_new_pad = torch.cat([v_new_flat, torch.zeros((BT, H, V), dtype=torch.float16, device=k.device)], dim=0)

    # Allocate state outputs
    h_out = torch.zeros((N, NT_max, H, K, V), dtype=torch.float16, device=k.device)
    h0 = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)
    if initial_state is not None:
        h0.copy_(initial_state.squeeze(0))

    ht = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)

    ker = chunk_gated_delta_rule(
        N,
        H,
        T_total_pad,
        Hg,
        K,
        V,
        NT_max,
        BT=BT,
        USE_G=USE_G,
        STORE_FINAL_STATE=output_final_state,
        SAVE_NEW_VALUE=save_new_value,
    )
    ker(h_out, k_pad, u_pad, w_pad, g_pad, v_new_pad, h0, ht, cu_seqlens.to(torch.int32))

    v_new_flat = v_new_pad[:T_total, :, :]  # [T_total, H, V]
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
# 4. Test Functions
# ==========================================
def test_chunk_gated_delta_rule_varlen(seqlens, H, Hg, K, V, use_g=True, use_initial_state=True):
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
    test_chunk_gated_delta_rule_varlen(
        seqlens=seqlens, H=args.H, Hg=args.Hg, K=args.K, V=args.V, use_g=args.use_g, use_initial_state=args.use_initial_state
    )
    print("Batch Kernel Output Match!")
