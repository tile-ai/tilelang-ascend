import tilelang
from tilelang import language as T
import torch

'''
Functionality:
O = ((Q * K^T) \odot M) * V, where M is the causal mask

Persistent-kernel version:
- T.Kernel uses a fixed core_num (not data-dependent B*H)
- B and L are symbolic (compiled once, runs for any batch/seq-len)
- Each core processes ceil(B*H / core_num) work items sequentially
- V scope zeros workspace_2 per work item and signals C (init handshake),
  so there is no stale accumulated-H from previous work items
'''

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

try:
    core_num = int(torch.npu.get_device_properties("npu").cube_core_num)
except Exception:
    core_num = 24


@tilelang.jit(out_idx=[-1], workspace_idx=[3, 4], pass_configs=pass_configs)
def linear_attention_ker(H, D, C, dtype="float16", accum_dtype="float"):
    B = T.symbolic("B")
    L = T.symbolic("L")

    shape = [B, H, L, D]
    chunk_num = T.ceildiv(L, C)
    VEC_NUM = 2

    @T.prim_func
    def main(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            workspace_1: T.Tensor([core_num, C, C], dtype),
            workspace_2: T.Tensor([core_num, D, D], dtype),
            O: T.Tensor(shape, dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([C, D], dtype)
            k_l1 = T.alloc_L1([C, D], dtype)
            v_l1 = T.alloc_L1([C, D], dtype)
            h_l1 = T.alloc_L1([D, D], dtype)
            acc_l1 = T.alloc_L1([C, C], dtype)
            h_l0 = T.alloc_L0C([D, D], accum_dtype)
            acc_l0 = T.alloc_L0C([C, C], accum_dtype)
            o_l0 = T.alloc_L0C([C, D], accum_dtype)

            hsum_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
            h_ub = T.alloc_ub([D // VEC_NUM, D], dtype)
            acc_ub = T.alloc_ub([C // VEC_NUM, C], dtype)
            zero_ub = T.alloc_ub([C // VEC_NUM, C], dtype)

            with T.Scope("C"):
                for work_idx in T.serial(T.ceildiv(B * H, core_num)):
                    pid = work_idx * core_num + cid
                    if pid < B * H:
                        by = pid % H
                        bz = pid // H

                        # Wait for V scope to zero workspace_2 for this work item.
                        # This prevents C scope from reading stale accumulated-H
                        # left behind by a previous work item.
                        T.wait_cross_flag(1)

                        for i in T.serial(chunk_num):
                            T.copy(Q[bz, by, i * C, 0], q_l1)
                            T.copy(K[bz, by, i * C, 0], k_l1)
                            T.copy(V[bz, by, i * C, 0], v_l1)
                            T.copy(workspace_2[cid, 0, 0], h_l1)
                            T.gemm_v0(q_l1, k_l1, acc_l0, transpose_B=True, init=True)
                            T.copy(acc_l0, workspace_1[cid, 0, 0])
                            T.gemm_v0(k_l1, v_l1, h_l0, transpose_A=True, init=True)
                            T.copy(h_l0, workspace_2[cid, 0, 0])
                            T.set_cross_flag("FIX", 0)

                            T.wait_cross_flag(1)
                            T.copy(workspace_1[cid, 0, 0], acc_l1)
                            T.gemm_v0(acc_l1, v_l1, o_l0, init=True)
                            T.gemm_v0(q_l1, h_l1, o_l0, init=False)
                            T.copy(o_l0, O[bz, by, i * C, 0])

            with T.Scope("V"):
                T.tile.fill(zero_ub, 0.0)

                for work_idx in T.serial(T.ceildiv(B * H, core_num)):
                    pid = work_idx * core_num + cid
                    if pid < B * H:
                        # Per-work-item init: zero hsum and workspace_2, then
                        # signal C scope that workspace_2 is ready (all-zero).
                        T.tile.fill(hsum_ub, 0.0)
                        T.copy(hsum_ub, workspace_2[cid, vid * D // VEC_NUM, 0])
                        T.set_cross_flag("MTE3", 1)

                        for i in T.serial(chunk_num):
                            T.wait_cross_flag(0)
                            T.copy(workspace_1[cid, vid * C // VEC_NUM, 0], acc_ub)
                            T.copy(workspace_2[cid, vid * D // VEC_NUM, 0], h_ub)
                            for j in range(C // VEC_NUM):
                                for k in range(C):
                                    if (j + vid * C // VEC_NUM) < k:
                                        acc_ub[j, k] = zero_ub[j, k]
                            T.tile.add(hsum_ub, hsum_ub, h_ub)
                            T.copy(acc_ub, workspace_1[cid, vid * C // VEC_NUM, 0])
                            T.copy(hsum_ub, workspace_2[cid, vid * D // VEC_NUM, 0])
                            T.set_cross_flag("MTE3", 1)

    return main


def linear_attention(q, k, v, C):
    B, H, L, D = q.shape
    # Compile once for (H, D, C); reuse across different (B, L)
    ker = linear_attention_ker(H, D, C)
    o = ker(q, k, v)
    return o


def ref_linear_attention(q, k, v):
    B, H, L, D = q.shape
    q = q.float()
    k = k.float()
    v = v.float()
    h = torch.zeros([B, H, D, D]).npu().to(torch.float)
    o = torch.zeros([B, H, L, D]).npu().to(torch.float)
    for i in range(L):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]
        dh = torch.einsum("bhi,bhj->bhij", k_i, v_i)
        h = h + dh
        o_i = torch.einsum("bhi,bhij->bhj", q_i, h)
        o[:, :, i, :] = o_i
    return o.to(torch.float16)


torch.manual_seed(0)
torch.set_printoptions(threshold=float('inf'), sci_mode=False)

# All configs share H=2, D=128, C=64 → compiled ONCE; only B and L vary at runtime.
#
# Coverage rationale (core_num=20, so B*H work items vs 20 cores):
#   B*H <  core_num : idle cores present, single work-item per active core
#   B*H == core_num : perfect fit
#   B*H >  core_num : multiple sequential work items per core (tests state reset)
#
# L coverage: single chunk (L==C), small, medium, large (up to 4096)
test_configs = [
    # --- single chunk edge case ---
    (1, 2,   64, 128, 64),   # L==C, 1 chunk;  B*H=2  < 20

    # --- B*H < core_num ---
    (1, 2,  256, 128, 64),   # B*H=2
    (4, 2,  128, 128, 64),   # B*H=8
    (8, 2,  512, 128, 64),   # B*H=16

    # --- B*H == core_num ---
    (12, 2, 512, 128, 64),   # B*H=20

    # --- B*H > core_num (multiple work items per core) ---
    (16, 2,  256, 128, 64),  # B*H=32  → 2 work items/core
    (32, 2,  128, 128, 64),  # B*H=64  → 4 work items/core
    (50, 20, 128, 128, 64),  # B*H=1000  → (large B + large H)

    # --- large L ---
    (1,  2, 1024, 128, 64),  # L=1k,  B*H=2
    (8,  2, 2048, 128, 64),  # L=2k,  B*H=16
    (2,  2, 4096, 128, 64),  # L=4k,  B*H=4
    (16, 2, 1024, 128, 64),  # L=1k,  B*H=32 (large B + large L)
]

for B, H, L, D, C in test_configs:
    print(f"Testing B={B}, H={H}, L={L}, D={D}, C={C}  (B*H={B*H})")
    q = torch.randn([B, H, L, D]).npu().to(torch.float16)
    k = torch.randn([B, H, L, D]).npu().to(torch.float16)
    v = torch.randn([B, H, L, D]).npu().to(torch.float16)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    o = linear_attention(q, k, v, C)
    ref_o = ref_linear_attention(q, k, v)

    # numerical error accumulates for longer sequence
    if L >= 4096:
        atol = 4e-2
    elif L >= 2048:
        atol = 2e-2
    else:
        atol = 1e-2
    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-2, atol=atol)
    print("  passed!")

print("All tests passed!")
